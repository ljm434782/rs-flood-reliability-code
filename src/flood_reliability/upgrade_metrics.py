"""Pure-array metrics for the bounded 2026 experiment-upgrade protocol.

These functions deliberately know nothing about files, rasters, split manifests,
or pixel ownership. Callers must supply the frozen boolean owner mask and must
keep process/state grouping intact when reducing scores.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp


IGNORE = 255
N_CLASSES = 3
FLOOD_CLASS = 2
RELIABILITY_BINS = 15
BOOTSTRAP_SEED = 20261029


@dataclass(frozen=True)
class CalibrationCell:
    """One process × optical-state calibration cell, already owner-masked."""

    process_id: str
    state_id: str
    logits: np.ndarray
    labels: np.ndarray
    owner_mask: np.ndarray


@dataclass(frozen=True)
class TemperatureFit:
    temperature: float
    log_temperature: float
    objective_nll: float
    boundary_hit: bool
    optimizer_success: bool
    evaluations: int
    process_ids: tuple[str, ...]
    state_ids: tuple[str, ...]


def _validate_arrays(logits: np.ndarray, labels: np.ndarray,
                     owner_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    scores = np.asarray(logits)
    truth = np.asarray(labels)
    owner = np.asarray(owner_mask)
    if scores.ndim != 2 or scores.shape[1] != N_CLASSES or not np.issubdtype(scores.dtype, np.number):
        raise ValueError("logits must be a numeric N×3 array")
    if truth.ndim != 1 or truth.shape[0] != scores.shape[0] or not np.issubdtype(truth.dtype, np.integer):
        raise ValueError("labels must be an integer length-N array matching logits")
    if owner.ndim != 1 or owner.shape != truth.shape or owner.dtype != np.bool_:
        raise ValueError("owner_mask must be a boolean length-N array matching labels")
    if np.any(~np.isin(truth, (0, 1, 2, IGNORE))):
        raise ValueError("labels must contain only 0, 1, 2, or Ignore=255")
    selected = owner & (truth != IGNORE)
    if not np.all(np.isfinite(scores[selected])):
        raise ValueError("owned non-Ignore logits must be finite")
    return scores[selected].astype(np.float64, copy=False), truth[selected].astype(np.int64, copy=False), selected


def probabilities_from_logits(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Stable three-class softmax of logits / a finite positive temperature."""
    values = np.asarray(logits)
    if values.ndim != 2 or values.shape[1] != N_CLASSES or not np.issubdtype(values.dtype, np.number):
        raise ValueError("logits must be a numeric N×3 array")
    if not np.all(np.isfinite(values)):
        raise ValueError("logits must be finite")
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    scaled = values.astype(np.float64, copy=False) / float(temperature)
    log_probs = scaled - logsumexp(scaled, axis=1, keepdims=True)
    return np.exp(log_probs)


def fit_global_temperature(
    cells: Sequence[CalibrationCell],
    *,
    strategy: str,
) -> TemperatureFit:
    """Fit one global positive T by process/state-equal native-label NLL.

    `strategy` is `original` (one state) or `multistate` (five states). Every
    listed process × required-state cell must have at least one owned,
    non-Ignore pixel. Missing/empty cells fail closed; they are never dropped.
    The bounded SciPy fit follows the protocol exactly and does not expand a
    boundary solution.
    """
    if strategy not in {"original", "multistate"}:
        raise ValueError("strategy must be 'original' or 'multistate'")
    if not cells:
        raise ValueError("at least one calibration cell is required")

    prepared: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]] = {}
    processes: set[str] = set()
    states: set[str] = set()
    for cell in cells:
        if not isinstance(cell.process_id, str) or not cell.process_id:
            raise ValueError("process_id must be a nonempty string")
        if not isinstance(cell.state_id, str) or not cell.state_id:
            raise ValueError("state_id must be a nonempty string")
        key = (cell.process_id, cell.state_id)
        if key in prepared:
            raise ValueError(f"duplicate calibration cell: {key}")
        logits, labels, _ = _validate_arrays(cell.logits, cell.labels, cell.owner_mask)
        if labels.size == 0:
            raise ValueError(f"empty owned calibration cell: {key}")
        prepared[key] = (logits, labels)
        processes.add(cell.process_id)
        states.add(cell.state_id)

    expected_state_count = 1 if strategy == "original" else 5
    if len(states) != expected_state_count:
        raise ValueError(f"{strategy} calibration requires exactly {expected_state_count} state(s)")
    expected = {(process, state) for process in processes for state in states}
    if set(prepared) != expected:
        absent = sorted(expected - set(prepared))
        raise ValueError(f"calibration process×state grid is incomplete: {absent[:3]}")

    ordered_processes = tuple(sorted(processes))
    ordered_states = tuple(sorted(states))

    def objective(log_t: float) -> float:
        temperature = float(np.exp(log_t))
        cell_losses = []
        for process in ordered_processes:
            for state in ordered_states:
                logits, labels = prepared[(process, state)]
                scaled = logits / temperature
                # Mean over pixels inside each process×state, then equal cell weight.
                cell_losses.append(float(np.mean(logsumexp(scaled, axis=1) - scaled[np.arange(labels.size), labels])))
        value = float(np.mean(cell_losses))
        if not np.isfinite(value):
            raise ValueError("temperature objective became non-finite")
        return value

    result = minimize_scalar(
        objective, bounds=(-3.0, 3.0), method="bounded",
        options={"xatol": 0.001, "maxiter": 100},
    )
    if not result.success or not np.isfinite(result.fun) or not np.isfinite(result.x):
        raise RuntimeError(f"temperature optimizer failed: {result.message}")
    log_t = float(result.x)
    return TemperatureFit(
        temperature=float(np.exp(log_t)),
        log_temperature=log_t,
        objective_nll=float(result.fun),
        boundary_hit=bool(min(abs(log_t + 3.0), abs(log_t - 3.0)) <= 0.001),
        optimizer_success=bool(result.success),
        evaluations=int(result.nfev),
        process_ids=ordered_processes,
        state_ids=ordered_states,
    )


def _average_precision(labels: np.ndarray, scores: np.ndarray) -> float | None:
    positives = int(labels.sum())
    if positives == 0:
        return None
    # A tied score is one threshold: update cumulative TP at each score-group end.
    order = np.argsort(-scores, kind="stable")
    ordered_scores, ordered_labels = scores[order], labels[order]
    ends = np.r_[np.flatnonzero(np.diff(ordered_scores)), len(scores) - 1]
    tp_at_end = np.cumsum(ordered_labels, dtype=np.int64)[ends]
    precision = tp_at_end / (ends + 1)
    delta_recall = np.diff(np.r_[0, tp_at_end]) / positives
    return float(np.sum(precision * delta_recall))


def _ratio_or_none(numerator: int, denominator: int) -> float | None:
    return float(numerator / denominator) if denominator else None


def score_process_state(
    logits: np.ndarray,
    labels: np.ndarray,
    owner_mask: np.ndarray,
    *,
    temperature: float = 1.0,
    reliability_bins: int = RELIABILITY_BINS,
) -> dict:
    """Return one process/state's scores on externally owned, non-Ignore pixels.

    Confusion rows are truth classes and columns are predicted classes. Macro IoU
    averages only defined class IoUs. AP is exact non-interpolated average
    precision with tied probabilities treated as one threshold. The reliability
    bins are fixed-width [left,right), except the last bin includes probability 1.
    """
    if reliability_bins != RELIABILITY_BINS:
        raise ValueError("the frozen reliability diagnostic uses exactly 15 bins")
    clean_logits, truth, _ = _validate_arrays(logits, labels, owner_mask)
    if truth.size == 0:
        return {"pixels": 0, "confusion": np.zeros((3, 3), dtype=np.int64),
                "class_iou": [None, None, None], "macro_iou": None,
                "flood_iou": None, "flood_precision": None,
                "flood_recall": None, "flood_f1": None,
                "flood_auprc": None, "binary_flood_brier": None,
                "multiclass_nll": None, "multiclass_brier": None,
                "reliability": {"count": np.zeros(15, dtype=np.int64),
                                "confidence_sum": np.zeros(15),
                                "observed_rate_sum": np.zeros(15),
                                "mean_confidence": [None] * 15,
                                "observed_rate": [None] * 15,
                                "ece15": None}}

    probs = probabilities_from_logits(clean_logits, temperature)
    prediction = np.argmax(probs, axis=1)
    confusion = np.zeros((3, 3), dtype=np.int64)
    np.add.at(confusion, (truth, prediction), 1)
    class_ious: list[float | None] = []
    for cls in range(3):
        tp = int(confusion[cls, cls])
        union = int(confusion[cls, :].sum() + confusion[:, cls].sum() - tp)
        class_ious.append(_ratio_or_none(tp, union))
    defined_ious = [value for value in class_ious if value is not None]
    tp = int(confusion[2, 2])
    fp = int(confusion[:, 2].sum() - tp)
    fn = int(confusion[2, :].sum() - tp)
    flood_union = tp + fp + fn
    flood_precision = _ratio_or_none(tp, tp + fp)
    flood_recall = _ratio_or_none(tp, tp + fn)
    flood_f1 = _ratio_or_none(2 * tp, 2 * tp + fp + fn)

    scaled_logits = clean_logits.astype(np.float64, copy=False) / float(temperature)
    log_probs = scaled_logits - logsumexp(scaled_logits, axis=1, keepdims=True)
    multiclass_nll = float(np.mean(-log_probs[np.arange(truth.size), truth]))
    onehot_delta = probs.copy()
    onehot_delta[np.arange(truth.size), truth] -= 1.0
    multiclass_brier = float(np.mean(np.sum(onehot_delta ** 2, axis=1)))
    p_flood = probs[:, FLOOD_CLASS]
    binary_brier = float(np.mean((p_flood - (truth == FLOOD_CLASS)) ** 2))

    bin_index = np.minimum((p_flood * RELIABILITY_BINS).astype(np.int64), RELIABILITY_BINS - 1)
    counts = np.bincount(bin_index, minlength=RELIABILITY_BINS).astype(np.int64)
    confidence_sum = np.bincount(bin_index, weights=p_flood, minlength=RELIABILITY_BINS)
    observed_sum = np.bincount(bin_index, weights=(truth == FLOOD_CLASS), minlength=RELIABILITY_BINS)
    occupied = counts > 0
    mean_confidence: list[float | None] = [None] * RELIABILITY_BINS
    observed_rate: list[float | None] = [None] * RELIABILITY_BINS
    for index in np.flatnonzero(occupied):
        mean_confidence[int(index)] = float(confidence_sum[index] / counts[index])
        observed_rate[int(index)] = float(observed_sum[index] / counts[index])
    ece = float(np.sum((counts[occupied] / truth.size) * np.abs(
        confidence_sum[occupied] / counts[occupied]
        - observed_sum[occupied] / counts[occupied])))

    return {
        "pixels": int(truth.size),
        "confusion": confusion,
        "class_iou": class_ious,
        "macro_iou": float(np.mean(defined_ious)) if defined_ious else None,
        "flood_iou": class_ious[2],
        "flood_precision": flood_precision,
        "flood_recall": flood_recall,
        "flood_f1": flood_f1,
        "flood_auprc": _average_precision(truth == FLOOD_CLASS, p_flood),
        "binary_flood_brier": binary_brier,
        "multiclass_nll": multiclass_nll,
        "multiclass_brier": multiclass_brier,
        "reliability": {
            "count": counts,
            "confidence_sum": confidence_sum,
            "observed_rate_sum": observed_sum,
            "mean_confidence": mean_confidence,
            "observed_rate": observed_rate,
            "ece15": ece,
        },
    }


def five_state_process_brier(state_scores: dict[str, dict],
                             state_order: Sequence[str]) -> float | None:
    """Equal-weight defined process×state binary flood Brier; require five states.

    Protocol summaries separately use the common process support across all
    compared candidates. This helper returns None if any requested state is
    missing/undefined, allowing a caller to form that support mechanically.
    """
    if len(state_order) != 5 or len(set(state_order)) != 5:
        raise ValueError("exactly five unique state IDs are required")
    values = []
    for state in state_order:
        record = state_scores.get(state)
        if record is None or record.get("binary_flood_brier") is None or record.get("pixels", 0) <= 0:
            return None
        values.append(float(record["binary_flood_brier"]))
    if not np.all(np.isfinite(values)):
        raise ValueError("state Brier scores must be finite")
    return float(np.mean(values))


def paired_difference(left: float | None, right: float | None) -> float | None:
    """Return right-left only when both paired values are defined and finite."""
    if left is None or right is None:
        return None
    if not np.isfinite(left) or not np.isfinite(right):
        raise ValueError("paired values must be finite or None")
    return float(right - left)


def grouped_process_bootstrap(
    values: np.ndarray,
    process_ids: Sequence[str],
    *,
    repetitions: int = 10_000,
    seed: int = BOOTSTRAP_SEED,
) -> dict:
    """Percentile bootstrap of a complete process×cell tensor.

    Input axis 0 is the process axis; every trailing axis may represent any
    combination of candidate, state, and seed. One sampled process-index vector
    is applied to the entire tensor in every replicate, preserving all those
    cells together. Undefined/common support must be resolved before this call;
    NaN/Inf is rejected instead of silently changing denominators by cell.
    """
    array = np.asarray(values, dtype=np.float64)
    ids = tuple(process_ids)
    if array.ndim < 1 or array.shape[0] != len(ids) or not ids:
        raise ValueError("values axis 0 must match a nonempty process_ids sequence")
    if len(set(ids)) != len(ids) or any(not isinstance(pid, str) or not pid for pid in ids):
        raise ValueError("process IDs must be unique nonempty strings")
    if repetitions < 1:
        raise ValueError("repetitions must be positive")
    if not np.all(np.isfinite(array)):
        raise ValueError("resolve undefined values on frozen common support before bootstrap")
    point = np.mean(array, axis=0)
    rng = np.random.Generator(np.random.PCG64(seed))
    boot = np.empty((repetitions,) + array.shape[1:], dtype=np.float64)
    for replicate in range(repetitions):
        draw = rng.integers(0, len(ids), size=len(ids))
        boot[replicate] = np.mean(array[draw], axis=0)
    ci = np.quantile(boot, [0.025, 0.975], axis=0)
    return {
        "process_ids": ids,
        "n_processes": len(ids),
        "repetitions": repetitions,
        "seed": seed,
        "mean": point,
        "ci95": ci,
    }
