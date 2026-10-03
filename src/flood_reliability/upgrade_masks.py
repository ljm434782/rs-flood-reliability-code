"""Pure-array candidate mask tools for the 2026-09-29 upgrade draft.

These helpers have no archive, label, model, or torch dependency.  They only
operate on synthetic or already supplied NumPy arrays.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Literal

import numpy as np
from scipy.ndimage import gaussian_filter


IMAGE_SHAPE = (256, 256)
BLOCK_SIZE = 32
GRID_SIZE = 8
RETENTION_LEVELS = (100, 75, 50, 25, 0)
MaskNamespace = Literal["cal-state-v1", "eval-block-v1"]


def _ascii(value: str, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    try:
        value.encode("ascii")
    except UnicodeEncodeError as exc:
        raise ValueError(f"{name} must be ASCII") from exc
    return value


def _valid_mask(mask: np.ndarray) -> np.ndarray:
    if not isinstance(mask, np.ndarray) or mask.shape != IMAGE_SHAPE or mask.dtype != np.bool_:
        raise ValueError("valid mask must be a boolean 256x256 array")
    return mask


def sha256_pcg64(*parts: object) -> np.random.Generator:
    """Make a named PCG64 stream from SHA-256's first 16 hex as uint64."""
    text = "|".join(str(part) for part in parts)
    _ascii(text, "seed material")
    entropy = int(hashlib.sha256(text.encode("ascii")).hexdigest()[:16], 16)
    return np.random.Generator(np.random.PCG64(entropy))


def whole_dropout_decision(seed: int, epoch: int, sample_id: str) -> bool:
    """Return the shared whole-dropout-v1 Bernoulli(p=0.25) decision.

    Candidate identity is intentionally absent.  Thus B5 A/C, P, and the
    SMAGNet-style baseline receive the same decision for a common
    seed/epoch/sample tuple, as required by the upgrade draft.
    """
    if not isinstance(seed, (int, np.integer)) or not isinstance(epoch, (int, np.integer)):
        raise ValueError("seed and epoch must be integers")
    sample_id = _ascii(sample_id, "sample_id")
    return bool(sha256_pcg64(int(seed), int(epoch), sample_id, "whole-dropout-v1").random() < 0.25)


def _block_order(namespace: MaskNamespace, seed: int, sample_id: str) -> list[int]:
    if namespace not in ("cal-state-v1", "eval-block-v1"):
        raise ValueError("unsupported block-mask namespace")
    if not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    sample_id = _ascii(sample_id, "sample_id")
    return sorted(
        range(GRID_SIZE * GRID_SIZE),
        key=lambda block: (
            hashlib.sha256(f"{namespace}|{int(seed)}|{sample_id}|{block}".encode("ascii")).hexdigest(),
            block,
        ),
    )


def block_retention_mask(
    valid: np.ndarray,
    sample_id: str,
    level: int,
    seed: int,
    *,
    namespace: MaskNamespace,
) -> np.ndarray:
    """Return the nested draft cal/evaluation retained-valid mask.

    Nominal retention is a count of 32x32 blocks.  Existing invalid pixels
    remain invalid, so pixel-level retention can differ from that percentage.
    """
    valid = _valid_mask(valid)
    if level not in RETENTION_LEVELS:
        raise ValueError("level must be one of 100, 75, 50, 25, 0")
    if level == 100:
        return valid.copy()
    blocks = np.zeros((GRID_SIZE, GRID_SIZE), dtype=bool)
    count = level * GRID_SIZE * GRID_SIZE // 100
    blocks.ravel()[_block_order(namespace, seed, sample_id)[:count]] = True
    expanded = np.repeat(np.repeat(blocks, BLOCK_SIZE, axis=0), BLOCK_SIZE, axis=1)
    return valid & expanded


@dataclass(frozen=True)
class LocalMaskPlan:
    """One train-local-v1 draw, represented at image and block resolution."""

    apply: bool
    target_blocks: int | None
    grid_shape: tuple[int, int] | None
    top: int | None
    left: int | None
    block_mask: np.ndarray


def train_local_plan(seed: int, epoch: int, sample_id: str, *, whole_dropped: bool) -> LocalMaskPlan:
    """Draw the draft local rectangle without touching imagery.

    A whole-modality dropout pre-empts the local draw.  Otherwise one named
    stream draws p=0.5 application, target area 16/32/48 equally, shape and
    orientation, then a uniformly valid top-left block origin.
    """
    if not isinstance(seed, (int, np.integer)) or not isinstance(epoch, (int, np.integer)):
        raise ValueError("seed and epoch must be integers")
    sample_id = _ascii(sample_id, "sample_id")
    empty = np.zeros(IMAGE_SHAPE, dtype=bool)
    if whole_dropped:
        return LocalMaskPlan(False, None, None, None, None, empty)

    rng = sha256_pcg64(int(seed), int(epoch), sample_id, "train-local-v1")
    if rng.random() >= 0.5:
        return LocalMaskPlan(False, None, None, None, None, empty)

    target = (16, 32, 48)[int(rng.integers(0, 3))]
    if target == 16:
        shape = ((4, 4), (2, 8))[int(rng.integers(0, 2))]
    elif target == 32:
        shape = (4, 8)
    else:
        shape = (6, 8)
    if rng.integers(0, 2):
        shape = (shape[1], shape[0])
    rows, cols = shape
    top = int(rng.integers(0, GRID_SIZE - rows + 1))
    left = int(rng.integers(0, GRID_SIZE - cols + 1))
    grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=bool)
    grid[top:top + rows, left:left + cols] = True
    block_mask = np.repeat(np.repeat(grid, BLOCK_SIZE, axis=0), BLOCK_SIZE, axis=1)
    return LocalMaskPlan(True, target, shape, top, left, block_mask)


@dataclass(frozen=True)
class OpticalExposure:
    """Pure-array optical exposure result for training or evaluation."""

    optical: np.ndarray
    valid: np.ndarray
    present: bool
    time_offset: float
    local_plan: LocalMaskPlan | None


def _optical_and_valid(optical: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    valid = _valid_mask(valid)
    if not isinstance(optical, np.ndarray) or optical.shape != (12, *IMAGE_SHAPE):
        raise ValueError("optical input must be a 12x256x256 array")
    if not np.issubdtype(optical.dtype, np.floating):
        raise ValueError("optical input must have a floating dtype")
    return optical, valid


def apply_train_local_exposure(
    optical: np.ndarray,
    valid: np.ndarray,
    *,
    present: bool,
    time_offset: float,
    seed: int,
    epoch: int,
    sample_id: str,
    whole_dropped: bool,
) -> OpticalExposure:
    """Apply whole dropout or the candidate local rectangle to a copy.

    Whole dropout is mutually exclusive with local masking.  A local mask
    retains modality presence and time offset, while its entire 12-channel
    rectangle is zeroed and removed from the valid mask.
    """
    optical, valid = _optical_and_valid(optical, valid)
    if not isinstance(present, (bool, np.bool_)):
        raise ValueError("present must be boolean")
    if not np.isfinite(time_offset):
        raise ValueError("time_offset must be finite")
    if not present and valid.any():
        raise ValueError("absent optical input cannot have valid pixels")

    if whole_dropped:
        return OpticalExposure(
            np.zeros_like(optical), np.zeros_like(valid), False, 0.0,
            train_local_plan(seed, epoch, sample_id, whole_dropped=True),
        )
    plan = train_local_plan(seed, epoch, sample_id, whole_dropped=False)
    new_optical = optical.copy()
    new_valid = valid.copy()
    if plan.apply:
        new_optical[:, plan.block_mask] = 0
        new_valid &= ~plan.block_mask
    return OpticalExposure(new_optical, new_valid, bool(present), float(time_offset), plan)


def apply_evaluation_mask(
    optical: np.ndarray,
    retained_valid: np.ndarray,
    *,
    time_offset: float,
) -> OpticalExposure:
    """Apply a cal/evaluation mask, including empty-retained-mask behavior."""
    optical, retained_valid = _optical_and_valid(optical, retained_valid)
    if not np.isfinite(time_offset):
        raise ValueError("time_offset must be finite")
    output = optical.copy()
    output[:, ~retained_valid] = 0
    if not retained_valid.any():
        output.fill(0)
        return OpticalExposure(output, retained_valid.copy(), False, 0.0, None)
    return OpticalExposure(output, retained_valid.copy(), True, float(time_offset), None)


def irregular_retention_mask(valid: np.ndarray, sample_id: str, retained_fraction: float) -> np.ndarray:
    """Return eval-irregular-v1 retained-valid pixels without accessing labels.

    One field/rank order is generated per sample so 25%, 50%, and 75%
    retained masks are nested.  The removal count is floor((1-L) * N),
    allowing at most a 1/N deviation from nominal retention.
    """
    valid = _valid_mask(valid)
    sample_id = _ascii(sample_id, "sample_id")
    if retained_fraction not in (1.0, 0.75, 0.5, 0.25):
        raise ValueError("retained_fraction must be one of 1.0, 0.75, 0.5, 0.25")
    if retained_fraction == 1.0:
        return valid.copy()
    valid_count = int(valid.sum())
    remove_count = int(np.floor(valid_count * (1.0 - retained_fraction)))
    seed_text = f"{sample_id}|eval-irregular-v1"
    rng = sha256_pcg64(seed_text)
    field = gaussian_filter(rng.standard_normal(IMAGE_SHAPE), sigma=12, mode="reflect", truncate=4)
    coordinates = np.flatnonzero(valid)
    # Primary: descending smooth field. Secondary: row-major index ascending.
    order = np.lexsort((coordinates, -field.ravel()[coordinates]))
    retained = valid.copy()
    retained.ravel()[coordinates[order[:remove_count]]] = False
    return retained
