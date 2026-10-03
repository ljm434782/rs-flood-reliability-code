"""Auditable adapter around a hash-pinned upstream SMAGNet source file.

The upstream implementation is deliberately loaded from an explicit source path.
This keeps the third-party source separate from this project and makes a future
formal run fail closed if it is not the reviewed revision.
"""

from __future__ import annotations

import hashlib
import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import torch
from torch.nn import functional as F


AUTHOR_COMMIT = "4371df08e6ca3b9d71c0385ad57b589830469a0c"
AUTHOR_SOURCE_SHA256 = "0f2bc65ba0b163c475772bbb3171a818e3947c29fbc5ec54f7bc5c479e4a4a58"
IGNORE_INDEX = 255


def _source_digest(source_path: Path) -> str:
    return hashlib.sha256(source_path.read_bytes()).hexdigest()


def load_author_module(source_path: str | Path) -> ModuleType:
    """Load the reviewed MIT source after verifying its exact content hash."""
    source = Path(source_path).resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    digest = _source_digest(source)
    if digest != AUTHOR_SOURCE_SHA256:
        raise ValueError(
            f"Unexpected SMAGNet source SHA256 {digest}; expected "
            f"{AUTHOR_SOURCE_SHA256}")
    module_name = "flood_reliability._author_smagnet_4371df08"
    if module_name in sys.modules:
        return sys.modules[module_name]
    spec = importlib.util.spec_from_file_location(module_name, source)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load SMAGNet source from {source}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def split_s1_s2(fused_input: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Split a [B,6,H,W] tensor into VV/VH and B04/B03/B02/B08 respectively."""
    if fused_input.ndim != 4 or fused_input.shape[1] != 6:
        raise ValueError(f"Expected [B,6,H,W] input, got {tuple(fused_input.shape)}")
    return fused_input[:, :2], fused_input[:, 2:]


def make_smagnet_resnet50(source_path: str | Path) -> torch.nn.Module:
    """Instantiate the author architecture as a three-class, two-head model.

    No pretrained weights are requested.  The returned upstream forward accepts
    ``(s1, s2, invalid_mask)`` and returns ``(fused_logits, sar_logits, gates)``.
    """
    module = load_author_module(source_path)
    return module.SMAGNet(
        encoder_name="resnet50",
        encoder_weights_sar=None,
        encoder_weights_msi=None,
        classes=3,
        decoder_use_batchnorm=False,
        sarmsiff_method="sar_msi_gated",
        enable_spatial_mask=True,
    )


def validate_batch(
    fused_input: torch.Tensor, invalid_mask: torch.Tensor, label: torch.Tensor
) -> None:
    """Validate the fixed six-band input and 0/1/2/255 target contract."""
    split_s1_s2(fused_input)
    batch, _, height, width = fused_input.shape
    if height % 32 or width % 32:
        raise ValueError("SMAGNet ResNet encoders require H and W divisible by 32")
    if invalid_mask.shape != (batch, 1, height, width):
        raise ValueError("invalid_mask must have exact shape [B,1,H,W]")
    if invalid_mask.dtype == torch.bool:
        binary = True
    else:
        if invalid_mask.is_floating_point() and not bool(torch.isfinite(invalid_mask).all()):
            raise ValueError("invalid_mask must contain only finite values")
        binary = bool(((invalid_mask == 0) | (invalid_mask == 1)).all())
    if not binary:
        raise ValueError("invalid_mask must be binary: 0 for valid or 1 for invalid")
    if label.shape != (batch, height, width):
        raise ValueError("label must have shape [B,H,W]")
    allowed = (label == 0) | (label == 1) | (label == 2) | (label == IGNORE_INDEX)
    if label.dtype != torch.long or not bool(allowed.all()):
        raise ValueError("label must be torch.long and contain only 0, 1, 2, or 255")


def flood_soft_dice(logits: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
    """Existing study's class-2 Dice loss, excluding 255 pixels."""
    active = label != IGNORE_INDEX
    target = (label == 2).to(logits.dtype)
    probability = torch.softmax(logits, dim=1)[:, 2] * active
    intersection = (probability * target).sum()
    denominator = probability.sum() + target.sum()
    return 1 - (2 * intersection + 1) / (denominator + 1)


def two_head_loss(
    fused_logits: torch.Tensor,
    sar_logits: torch.Tensor,
    label: torch.Tensor,
    class_weights: torch.Tensor,
    dice_weight: float = 0.3,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Apply weighted CE plus 0.3 Dice independently to each output head."""
    if fused_logits.shape != sar_logits.shape or fused_logits.shape[1] != 3:
        raise ValueError("Both heads must have matching [B,3,H,W] logits")
    if class_weights.shape != (3,):
        raise ValueError("class_weights must have exactly three entries")
    if not bool((label != IGNORE_INDEX).any()):
        raise ValueError("label contains no non-Ignore pixels; loss is undefined")
    parts: dict[str, torch.Tensor] = {}
    for name, logits in (("fused", fused_logits), ("sar", sar_logits)):
        ce = F.cross_entropy(logits, label, weight=class_weights, ignore_index=IGNORE_INDEX)
        dice = flood_soft_dice(logits, label)
        parts[f"{name}_ce"] = ce
        parts[f"{name}_dice"] = dice
        parts[f"{name}_loss"] = ce + dice_weight * dice
    return 0.5 * (parts["fused_loss"] + parts["sar_loss"]), parts
