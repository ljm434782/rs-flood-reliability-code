import numpy as np
import pytest

from flood_reliability.optical_benefit import (
    class_probability_sums,
    classify_joint_gain,
    pair_process_cells,
    pair_class_rows,
)


def test_class_probability_sums_reconstructs_binary_brier_on_owned_pixels():
    logits = np.array([
        [[2.0, 0.0, 9.0]],
        [[0.0, 2.0, 9.0]],
        [[0.0, 0.0, 9.0]],
    ])
    labels = np.array([[0, 2, 255]])
    owner = np.array([[True, True, True]])
    result = class_probability_sums(logits, labels, owner, temperature=2.0)
    assert [result[c]["pixels"] for c in range(3)] == [1, 0, 1]
    probabilities = np.exp(logits[:, 0, :2] / 2.0)
    probabilities /= probabilities.sum(axis=0, keepdims=True)
    expected = np.mean((probabilities[2] - np.array([0.0, 1.0])) ** 2)
    total = sum(result[c]["binary_flood_brier_sum"] for c in range(3))
    assert total / 2 == pytest.approx(expected)
    assert sum(result[c]["multiclass_brier_sum"] for c in range(3)) > total


def test_pair_process_cells_requires_identical_support_and_keeps_undefined_iou():
    sar = {"pixels": 20, "binary_flood_brier": 0.2, "flood_iou": None}
    fusion = {"pixels": 20, "binary_flood_brier": 0.1, "flood_iou": None}
    paired = pair_process_cells(sar, fusion)
    assert paired["binary_flood_brier_gain"] == pytest.approx(0.1)
    assert paired["flood_iou_gain"] is None
    assert paired["joint_outcome"] == "undefined_iou"
    with pytest.raises(ValueError, match="support"):
        pair_process_cells(sar, {**fusion, "pixels": 19})


def test_joint_gain_reports_ties_separately():
    assert classify_joint_gain(0.1, 0.2) == "both_better"
    assert classify_joint_gain(-0.1, 0.2) == "probability_only"
    assert classify_joint_gain(0.1, -0.2) == "segmentation_only"
    assert classify_joint_gain(-0.1, -0.2) == "both_worse"
    assert classify_joint_gain(0.0, 0.2) == "tie"


def test_pair_class_rows_is_additive_and_rejects_changed_label_support():
    sar = {"class_pixels": 2, "total_pixels": 5,
           "binary_flood_brier_sum": 0.6, "multiclass_brier_sum": 1.0,
           "multiclass_nll_sum": 0.8}
    fusion = {**sar, "binary_flood_brier_sum": 0.3}
    result = pair_class_rows(sar, fusion)
    assert result["binary_flood_brier_all_pixel_gain"] == pytest.approx(0.06)
    assert result["binary_flood_brier_conditional_gain"] == pytest.approx(0.15)
    with pytest.raises(ValueError, match="class support"):
        pair_class_rows(sar, {**fusion, "class_pixels": 1})
