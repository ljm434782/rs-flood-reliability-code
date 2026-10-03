"""Retrospective optical-benefit diagnostics on fixed, paired flood processes."""

from __future__ import annotations

import math

import numpy as np


def classify_joint_gain(iou_gain: float | None, brier_gain: float) -> str:
    """Name both outcomes without treating ties or undefined IoU as a loss."""
    if iou_gain is None:
        return "undefined_iou"
    if iou_gain == 0 or brier_gain == 0:
        return "tie"
    if iou_gain > 0 and brier_gain > 0:
        return "both_better"
    if iou_gain > 0:
        return "segmentation_only"
    if brier_gain > 0:
        return "probability_only"
    return "both_worse"


def pair_process_cells(sar: dict, fusion: dict) -> dict:
    """Compare fixed models on exactly the same owned, non-Ignore pixel count."""
    if sar["pixels"] <= 0 or sar["pixels"] != fusion["pixels"]:
        raise ValueError("paired model cells must have identical nonempty pixel support")
    baseline_brier = float(sar["binary_flood_brier"])
    fusion_brier = float(fusion["binary_flood_brier"])
    if not (math.isfinite(baseline_brier) and math.isfinite(fusion_brier)):
        raise ValueError("paired flood Brier scores must be finite")
    sar_iou, fusion_iou = sar["flood_iou"], fusion["flood_iou"]
    iou_gain = None if sar_iou is None or fusion_iou is None else float(fusion_iou) - float(sar_iou)
    if iou_gain is not None and not math.isfinite(iou_gain):
        raise ValueError("paired flood IoU gain must be finite or undefined")
    brier_gain = baseline_brier - fusion_brier
    return {"pixels": int(sar["pixels"]), "flood_iou_gain": iou_gain,
            "binary_flood_brier_gain": brier_gain,
            "joint_outcome": classify_joint_gain(iou_gain, brier_gain)}


def class_probability_sums(logits: np.ndarray, labels: np.ndarray,
                           owner_mask: np.ndarray, *, temperature: float) -> dict[int, dict]:
    """Additive probability losses by true class on one owned tile.

    Each class sum contributes to the all-pixel score; its conditional mean is
    descriptive and is not itself a proper score for the original population.
    """
    z = np.asarray(logits)
    y = np.asarray(labels)
    owner = np.asarray(owner_mask)
    if (z.ndim != 3 or z.shape[0] != 3 or z.shape[1:] != y.shape or
            owner.shape != y.shape or owner.dtype != np.bool_):
        raise ValueError("expected [3,H,W] logits and matching label/boolean owner grids")
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    selected = owner & (y != 255)
    truth = y[selected]
    if not np.isin(truth, (0, 1, 2)).all():
        raise ValueError("labels must be 0, 1, 2, or Ignore=255")
    scores = np.moveaxis(z, 0, -1)[selected].astype(np.float64) / temperature
    if not np.isfinite(scores).all():
        raise ValueError("evaluated logits must be finite")
    if len(truth):
        scores -= scores.max(axis=1, keepdims=True)
        exponential = np.exp(scores)
        probabilities = exponential / exponential.sum(axis=1, keepdims=True)
        log_probabilities = scores - np.log(exponential.sum(axis=1, keepdims=True))
    else:
        probabilities = np.empty((0, 3), dtype=np.float64)
        log_probabilities = probabilities
    result: dict[int, dict] = {}
    for cls in range(3):
        p = probabilities[truth == cls]
        lp = log_probabilities[truth == cls]
        delta = p.copy()
        delta[:, cls] -= 1.0
        result[cls] = {
            "pixels": int(len(p)),
            "binary_flood_brier_sum": float(np.sum((p[:, 2] - (cls == 2)) ** 2)),
            "multiclass_brier_sum": float(np.sum(delta ** 2)),
            "multiclass_nll_sum": float(-np.sum(lp[:, cls])),
        }
    return result


def pair_class_rows(sar: dict, fusion: dict) -> dict:
    """Additive B2-minus-fusion class loss contributions on common labels."""
    if (sar["total_pixels"] <= 0 or sar["total_pixels"] != fusion["total_pixels"]
            or sar["class_pixels"] != fusion["class_pixels"]):
        raise ValueError("paired true-class support must be identical")
    result = {"class_pixels": int(sar["class_pixels"]),
              "total_pixels": int(sar["total_pixels"])}
    for loss in ("binary_flood_brier", "multiclass_brier", "multiclass_nll"):
        gain = float(sar[f"{loss}_sum"]) - float(fusion[f"{loss}_sum"])
        if not math.isfinite(gain):
            raise ValueError("class loss gain must be finite")
        result[f"{loss}_all_pixel_gain"] = gain / sar["total_pixels"]
        result[f"{loss}_conditional_gain"] = (
            gain / sar["class_pixels"] if sar["class_pixels"] else None)
    return result
