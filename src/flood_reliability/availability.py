"""Whole-optical-modality dropout shared by B5 and P training.

The dropout draw and rate belong to the frozen training configuration. This
function only applies an explicit per-sample draw; it does not choose one.
"""

import torch


def apply_modality_dropout(
    s2: torch.Tensor,
    optical_valid: torch.Tensor,
    s2_present: torch.Tensor,
    drop: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return new optical tensors with selected samples fully unavailable."""
    if s2.ndim != 4:
        raise ValueError("S2 must be BCHW")
    batch, _, height, width = s2.shape
    if optical_valid.shape != (batch, 1, height, width) or optical_valid.dtype != torch.bool:
        raise ValueError("Optical-valid mask must be boolean N1HW")
    if s2_present.shape != (batch,) or s2_present.dtype != torch.bool:
        raise ValueError("S2 presence must be a boolean vector")
    if drop.shape != (batch,) or drop.dtype != torch.bool:
        raise ValueError("Drop draw must be a boolean vector")
    if torch.any(optical_valid & ~s2_present[:, None, None, None]):
        raise ValueError("Absent S2 cannot have valid optical pixels")
    retained = ~drop
    new_present = s2_present & retained
    new_valid = optical_valid & retained[:, None, None, None]
    new_s2 = torch.where(retained[:, None, None, None], s2, torch.zeros_like(s2))
    return new_s2, new_valid, new_present
