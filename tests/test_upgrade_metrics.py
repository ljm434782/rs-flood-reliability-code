"""Synthetic-only tests for the pure-array G3 metric/calibration helpers."""

import numpy as np
import pytest

from flood_reliability.upgrade_metrics import (
    CalibrationCell,
    fit_global_temperature,
    five_state_process_brier,
    grouped_process_bootstrap,
    paired_difference,
    probabilities_from_logits,
    score_process_state,
)


def _logits_for_probabilities(probabilities):
    return np.log(np.asarray(probabilities, dtype=np.float64))


def test_process_metrics_hand_computed_with_owner_and_ignore():
    probabilities = np.array([[0.2, 0.1, 0.7], [0.8, 0.1, 0.1],
                              [0.2, 0.1, 0.7], [0.2, 0.1, 0.7]])
    logits = _logits_for_probabilities(probabilities)
    labels = np.array([2, 0, 255, 1], dtype=np.uint8)
    owner = np.array([True, True, True, False])

    result = score_process_state(logits, labels, owner)

    assert result["pixels"] == 2
    np.testing.assert_array_equal(result["confusion"], [[1, 0, 0], [0, 0, 0], [0, 0, 1]])
    assert result["class_iou"] == [1.0, None, 1.0]
    assert result["macro_iou"] == 1.0
    assert result["flood_iou"] == result["flood_precision"] == 1.0
    assert result["flood_recall"] == result["flood_f1"] == result["flood_auprc"] == 1.0
    assert result["binary_flood_brier"] == pytest.approx(0.05)
    assert result["multiclass_brier"] == pytest.approx(0.1)
    assert result["multiclass_nll"] == pytest.approx((-np.log(0.7) - np.log(0.8)) / 2)
    reliability = result["reliability"]
    assert reliability["count"][1] == 1
    assert reliability["count"][10] == 1
    assert reliability["mean_confidence"][1] == pytest.approx(0.1)
    assert reliability["observed_rate"][10] == 1.0
    assert reliability["ece15"] == pytest.approx(0.2)


def test_empty_owned_support_returns_undefined_metrics_not_fake_zeros():
    result = score_process_state(np.zeros((2, 3)), np.array([255, 0]),
                                 np.array([True, False]))
    assert result["pixels"] == 0
    assert result["flood_iou"] is None
    assert result["binary_flood_brier"] is None
    assert result["flood_auprc"] is None
    assert result["reliability"]["ece15"] is None


def test_undefined_flood_ap_and_iou_rules():
    # Flood truth absent and flood prediction absent: both AP and flood IoU undefined.
    logits = _logits_for_probabilities([[0.9, 0.1, 0.0 + 1e-12],
                                        [0.1, 0.8, 0.1]])
    result = score_process_state(logits, np.array([0, 1]), np.ones(2, dtype=bool))
    assert result["flood_auprc"] is None
    assert result["flood_iou"] is None
    assert result["flood_precision"] is None
    assert result["flood_recall"] is None
    # No positives but a false positive gives flood IoU=0 and precision=0.
    logits_fp = _logits_for_probabilities([[0.1, 0.1, 0.8]])
    false_positive = score_process_state(logits_fp, np.array([0]), np.ones(1, dtype=bool))
    assert false_positive["flood_iou"] == 0.0
    assert false_positive["flood_precision"] == 0.0
    assert false_positive["flood_recall"] is None


def test_temperature_fit_is_process_and_state_equal_and_positive():
    cells = []
    # Distinct pixel counts do not alter the process/state-equal objective.
    for process, count in (("p1", 1), ("p2", 8)):
        for state in ("100", "75", "50", "25", "0"):
            labels = np.zeros(count, dtype=np.uint8)
            logits = np.tile(np.array([[2.0, 0.0, -1.0]]), (count, 1))
            cells.append(CalibrationCell(process, state, logits, labels,
                                         np.ones(count, dtype=bool)))
    fit = fit_global_temperature(cells, strategy="multistate")
    assert fit.optimizer_success
    assert fit.temperature > 0
    assert -3 <= fit.log_temperature <= 3
    assert fit.process_ids == ("p1", "p2")
    assert fit.state_ids == ("0", "100", "25", "50", "75")
    # All cells are duplicates, so the fitted objective equals the per-pixel NLL.
    scaled = np.array([[2.0, 0.0, -1.0]]) / fit.temperature
    expected = np.log(np.exp(scaled).sum()) - scaled[0, 0]
    assert fit.objective_nll == pytest.approx(expected)
    assert isinstance(fit.boundary_hit, bool)


def test_temperature_calibration_fails_closed_on_missing_empty_or_duplicate_cells():
    logits = np.zeros((1, 3))
    labels = np.array([0], dtype=np.uint8)
    owner = np.array([True])
    cell = CalibrationCell("p", "100", logits, labels, owner)
    with pytest.raises(ValueError, match="requires exactly"):
        fit_global_temperature([cell], strategy="multistate")
    five = [CalibrationCell("p", state, logits, labels, owner)
            for state in ("100", "75", "50", "25", "0")]
    five[2] = CalibrationCell("p", "50", logits, labels,
                              np.array([False]))
    with pytest.raises(ValueError, match="empty owned"):
        fit_global_temperature(five, strategy="multistate")
    with pytest.raises(ValueError, match="duplicate"):
        fit_global_temperature([cell, cell], strategy="original")


def test_temperature_inputs_and_owner_contract_are_strict():
    with pytest.raises(ValueError, match="positive"):
        probabilities_from_logits(np.zeros((1, 3)), 0)
    with pytest.raises(ValueError, match="boolean"):
        score_process_state(np.zeros((1, 3)), np.array([0]), np.array([1]))
    with pytest.raises(ValueError, match="finite"):
        score_process_state(np.array([[np.nan, 0, 0]]), np.array([0]), np.array([True]))
    with pytest.raises(ValueError, match="only 0, 1, 2"):
        score_process_state(np.zeros((1, 3)), np.array([3]), np.array([True]))


def test_five_state_process_average_requires_all_five_states():
    rows = {state: {"pixels": 1, "binary_flood_brier": value}
            for state, value in zip(("100", "75", "50", "25", "0"),
                                    (0.1, 0.2, 0.3, 0.4, 0.5))}
    assert five_state_process_brier(rows, ("100", "75", "50", "25", "0")) == pytest.approx(0.3)
    assert five_state_process_brier({k: v for k, v in rows.items() if k != "0"},
                                    ("100", "75", "50", "25", "0")) is None
    assert paired_difference(0.2, 0.7) == pytest.approx(0.5)
    assert paired_difference(None, 0.7) is None


def test_grouped_process_bootstrap_resamples_all_candidate_state_seed_cells_together():
    # The second cell is exactly the negative of the first. Shared process draws
    # preserve that relation in every bootstrap replicate; independent cell-wise
    # resampling would not.
    values = np.array([[1.0, -1.0], [4.0, -4.0], [10.0, -10.0]])
    result = grouped_process_bootstrap(values, ("p1", "p2", "p3"), repetitions=250)
    assert result["seed"] == 20261029
    np.testing.assert_allclose(result["mean"], [5.0, -5.0])
    assert result["ci95"][0, 0] == pytest.approx(-result["ci95"][1, 1])
    assert result["ci95"][1, 0] == pytest.approx(-result["ci95"][0, 1])
    with pytest.raises(ValueError, match="common support"):
        grouped_process_bootstrap(np.array([[1.0], [np.nan]]), ("p1", "p2"), repetitions=5)
