"""Lossless categorical counts; metric reduction belongs to the frozen protocol."""

from collections.abc import Iterable

import numpy as np

from flood_reliability.splits import canonical_process_id


def confusion_counts(
    target: np.ndarray,
    prediction: np.ndarray,
    *,
    num_classes: int,
    ignore_values: tuple[int, ...] = (-1, 255),
    evaluation_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Count true/predicted class pairs after Ignore and optional support masking.

    Rows are target classes; columns are predicted classes. An ownership mask for
    overlapping tiles must be supplied by the caller after its rule is frozen.
    """
    truth = np.asarray(target)
    guess = np.asarray(prediction)
    if num_classes < 2 or truth.ndim != 2 or truth.shape != guess.shape:
        raise ValueError("Expected matching 2D rasters and at least two classes")
    if not np.issubdtype(truth.dtype, np.integer) or not np.issubdtype(guess.dtype, np.integer):
        raise ValueError("Targets and hard predictions must contain integer class codes")
    if any(0 <= value < num_classes for value in ignore_values):
        raise ValueError("Ignore codes must not overlap a class code")
    valid = ~np.isin(truth, ignore_values)
    if evaluation_mask is not None:
        support = np.asarray(evaluation_mask)
        if support.shape != truth.shape or support.dtype != np.bool_:
            raise ValueError("Evaluation mask must be boolean and match the raster grid")
        valid &= support
    if np.any((truth[valid] < 0) | (truth[valid] >= num_classes)):
        raise ValueError("Target contains an unknown non-Ignore class code")
    if np.any((guess[valid] < 0) | (guess[valid] >= num_classes)):
        raise ValueError("Prediction contains an out-of-range class code")
    codes = truth[valid].astype(np.int64) * num_classes + guess[valid].astype(np.int64)
    return np.bincount(codes, minlength=num_classes * num_classes).reshape(num_classes, num_classes)


def pool_process_confusions(records: Iterable[tuple[str, np.ndarray]]) -> dict[str, np.ndarray]:
    """Sum per-tile counts within process; do not average tile-level scores.

    Input matrices must already reflect any required overlap ownership masks.
    This function deliberately computes no IoU, F1, or empty-class policy.
    """
    pooled: dict[str, np.ndarray] = {}
    seen = False
    for process_id, matrix in records:
        seen = True
        counts = np.asarray(matrix)
        process_key = canonical_process_id(process_id)
        if counts.ndim != 2 or counts.shape[0] != counts.shape[1] or counts.shape[0] < 2:
            raise ValueError("Each process needs a square confusion matrix with at least two classes")
        if not np.issubdtype(counts.dtype, np.integer) or np.any(counts < 0):
            raise ValueError("Confusion matrices must contain nonnegative integer counts")
        if process_key not in pooled:
            pooled[process_key] = np.zeros_like(counts, dtype=np.int64)
        if pooled[process_key].shape != counts.shape:
            raise ValueError("Class counts differ within a process")
        pooled[process_key] += counts
    if not seen:
        raise ValueError("At least one tile confusion matrix is required")
    return pooled
