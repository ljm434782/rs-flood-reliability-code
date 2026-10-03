"""Additive probability scores; process-level reduction stays explicit."""

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np

from flood_reliability.splits import canonical_process_id


@dataclass(frozen=True)
class ProbabilitySums:
    pixels: int
    nll: float
    brier: float


def probability_sums(
    target: np.ndarray,
    probabilities: np.ndarray,
    *,
    ignore_values: tuple[int, ...] = (-1, 255),
    evaluation_mask: np.ndarray | None = None,
) -> ProbabilitySums:
    """Sum multiclass NLL and Brier over owned, non-Ignore pixels.

    Probabilities have shape (height, width, classes), with Brier defined as
    the sum of squared class errors per pixel. Exact zero assigned to the
    correct class produces infinite NLL; no unregistered clipping is applied.
    """
    truth = np.asarray(target)
    probs = np.asarray(probabilities)
    if truth.ndim != 2 or not np.issubdtype(truth.dtype, np.integer):
        raise ValueError("Target must be a 2D integer raster")
    if probs.ndim != 3 or probs.shape[:2] != truth.shape or probs.shape[2] < 2:
        raise ValueError("Probabilities must match target grid and have at least two classes")
    if not np.issubdtype(probs.dtype, np.floating):
        raise ValueError("Probabilities must be floating point")
    classes = probs.shape[2]
    if any(0 <= code < classes for code in ignore_values):
        raise ValueError("Ignore codes must not overlap a class code")
    valid = ~np.isin(truth, ignore_values)
    if evaluation_mask is not None:
        support = np.asarray(evaluation_mask)
        if support.shape != truth.shape or support.dtype != np.bool_:
            raise ValueError("Evaluation mask must be boolean and match the raster grid")
        valid &= support
    labels = truth[valid]
    selected = probs[valid].astype(np.float64)
    if np.any((labels < 0) | (labels >= classes)):
        raise ValueError("Target contains an unknown non-Ignore class code")
    if not np.all(np.isfinite(selected)) or np.any((selected < 0) | (selected > 1)):
        raise ValueError("Evaluated probabilities must be finite and lie in [0, 1]")
    if not np.allclose(selected.sum(axis=1), 1.0, rtol=0, atol=1e-5):
        raise ValueError("Evaluated class probabilities must sum to one")
    if not len(labels):
        return ProbabilitySums(0, 0.0, 0.0)
    correct = selected[np.arange(len(labels)), labels]
    with np.errstate(divide="ignore"):
        nll = float(np.sum(-np.log(correct)))
    squared = selected.copy()
    squared[np.arange(len(labels)), labels] -= 1.0
    brier = float(np.sum(squared * squared))
    return ProbabilitySums(len(labels), nll, brier)


def pool_process_probability_sums(
    records: Iterable[tuple[str, ProbabilitySums]],
) -> dict[str, ProbabilitySums]:
    """Add tile scores within each process, never across processes."""
    pooled: dict[str, ProbabilitySums] = {}
    for process_id, score in records:
        process_key = canonical_process_id(process_id)
        if not isinstance(score, ProbabilitySums):
            raise ValueError("Expected a process ID and ProbabilitySums")
        if score.pixels < 0 or score.nll < 0 or score.brier < 0 or np.isnan(score.nll) or not np.isfinite(score.brier):
            raise ValueError("Invalid probability sums")
        old = pooled.get(process_key, ProbabilitySums(0, 0.0, 0.0))
        pooled[process_key] = ProbabilitySums(old.pixels + score.pixels, old.nll + score.nll, old.brier + score.brier)
    if not pooled:
        raise ValueError("At least one tile score is required")
    return pooled
