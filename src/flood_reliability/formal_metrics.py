"""Frozen process-first reductions for formal ImpactMesh evaluation."""

import numpy as np


def average_precision_binary(labels: np.ndarray, scores: np.ndarray) -> float | None:
    """Exact non-interpolated AP with tied scores treated as one threshold."""
    truth = np.asarray(labels, dtype=bool).ravel()
    values = np.asarray(scores, dtype=np.float64).ravel()
    if truth.shape != values.shape or not np.isfinite(values).all():
        raise ValueError("Matching finite labels and scores required")
    positives = int(truth.sum())
    if positives == 0:
        return None
    order = np.argsort(-values, kind="stable")
    sorted_scores = values[order]
    sorted_truth = truth[order]
    end = np.r_[np.flatnonzero(np.diff(sorted_scores)), len(values) - 1]
    tp = np.cumsum(sorted_truth, dtype=np.int64)[end]
    precision = tp / (end + 1)
    delta_recall = np.diff(np.r_[0, tp]) / positives
    return float(np.sum(precision * delta_recall))


def class_iou(confusion: np.ndarray, class_index: int) -> float | None:
    matrix = np.asarray(confusion, dtype=np.int64)
    tp = int(matrix[class_index, class_index])
    union = int(matrix[class_index].sum() + matrix[:, class_index].sum() - tp)
    return tp / union if union else None


def flood_f1(confusion: np.ndarray) -> float | None:
    matrix = np.asarray(confusion, dtype=np.int64)
    tp = int(matrix[2, 2])
    denominator = int(matrix[2].sum() + matrix[:, 2].sum())
    return 2 * tp / denominator if denominator else None


def process_metrics(confusion: np.ndarray, pixels: int, nll_sum: float,
                    brier_sum: float, bin_count: np.ndarray,
                    bin_conf_sum: np.ndarray, bin_correct_sum: np.ndarray,
                    flood_ap: float | None) -> dict:
    if pixels <= 0:
        raise ValueError("Process has no valid, owned pixels")
    if int(np.asarray(confusion).sum()) != pixels:
        raise ValueError("Confusion and probability pixel counts disagree")
    if int(np.asarray(bin_count).sum()) != pixels:
        raise ValueError("ECE and probability pixel counts disagree")
    ious = [class_iou(confusion, class_index) for class_index in range(3)]
    valid = np.asarray(bin_count) > 0
    ece = float(np.sum((bin_count[valid] / pixels) * np.abs(
        bin_correct_sum[valid] / bin_count[valid]
        - bin_conf_sum[valid] / bin_count[valid])))
    return {"pixels": pixels, "nonwater_iou": ious[0],
            "permanent_iou": ious[1], "flood_iou": ious[2],
            "macro_iou": float(np.mean([value for value in ious if value is not None])),
            "flood_f1": flood_f1(confusion), "flood_auprc": flood_ap,
            "nll": nll_sum / pixels, "brier": brier_sum / pixels,
            "ece15": ece, "confusion": np.asarray(confusion).tolist()}


def macro_with_bootstrap(rows: dict[str, dict], key: str, *, seed=20260927,
                         repetitions=10000) -> dict:
    process_ids = sorted(rows)
    values = np.array([np.nan if rows[process][key] is None else rows[process][key]
                       for process in process_ids], dtype=np.float64)
    defined = values[~np.isnan(values)]
    if len(defined) == 0:
        return {"mean": None, "median": None, "ci95": None,
                "defined_processes": 0, "undefined_processes": len(process_ids)}
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(values), size=(repetitions, len(values)))
    sampled = values[draws]
    numerator = np.nansum(sampled, axis=1)
    denominator = (~np.isnan(sampled)).sum(axis=1)
    boot = numerator[denominator > 0] / denominator[denominator > 0]
    return {"mean": float(defined.mean()), "median": float(np.median(defined)),
            "ci95": np.quantile(boot, [.025, .975]).tolist(),
            "defined_processes": len(defined),
            "undefined_processes": len(process_ids) - len(defined)}
