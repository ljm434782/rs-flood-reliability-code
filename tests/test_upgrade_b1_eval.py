import numpy as np
import pytest

from scripts import upgrade_b1_eval


def test_b1_hard_metrics_match_hand_computed_confusion():
    result = upgrade_b1_eval._hard_metrics(np.array([[8, 1, 1],
                                                       [2, 5, 1],
                                                       [1, 2, 6]], dtype=np.int64))
    assert result["pixels"] == 27
    assert result["flood_iou"] == pytest.approx(6 / 11)
    assert result["flood_precision"] == pytest.approx(6 / 8)
    assert result["flood_recall"] == pytest.approx(6 / 9)
    assert result["flood_f1"] == pytest.approx(2 * .75 * (2 / 3) / (.75 + (2 / 3)))


def test_b1_draft_gate_precedes_threshold_and_reader_access(tmp_path, monkeypatch):
    def forbidden(*_args, **_kwargs):
        raise AssertionError("real archive reader must not be constructed for draft")

    monkeypatch.setattr(upgrade_b1_eval, "TrainArchiveReader", forbidden)
    with pytest.raises(RuntimeError, match="config is not FROZEN"):
        upgrade_b1_eval.evaluate_b1(tmp_path, {"status": "DRAFT_NOT_FROZEN"}, "internal_retro")


def test_b1_f1_zero_for_missed_or_false_flood_and_undefined_only_for_empty_union():
    assert upgrade_b1_eval._hard_metrics(np.array([[1, 0, 1], [0, 1, 0], [1, 0, 0]]))["flood_f1"] == 0
    assert upgrade_b1_eval._hard_metrics(np.diag([2, 3, 0]))["flood_f1"] is None
