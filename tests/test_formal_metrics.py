import numpy as np

from flood_reliability.formal_metrics import (
    average_precision_binary, class_iou, macro_with_bootstrap, process_metrics,
)
from scripts.evaluate_impactmesh_formal import block_valid


def test_average_precision_groups_tied_scores_at_one_threshold():
    labels = np.array([1, 0, 1, 0])
    scores = np.array([.9, .5, .5, .1])
    assert np.isclose(average_precision_binary(labels, scores),
                      .5 * 1 + .5 * (2 / 3))
    assert average_precision_binary(np.zeros(4), scores) is None


def test_no_flood_process_iou_is_undefined_only_without_prediction():
    matrix = np.array([[6, 0, 0], [0, 4, 0], [0, 0, 0]])
    assert class_iou(matrix, 2) is None
    matrix[0, 2] = 1
    assert class_iou(matrix, 2) == 0.


def test_process_ece_and_bootstrap_ignore_undefined_process_metric():
    matrix = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 0]])
    count = np.zeros(15, np.int64)
    conf = np.zeros(15)
    correct = np.zeros(15)
    count[12] = 2
    conf[12] = 1.6
    correct[12] = 2
    row = process_metrics(matrix, 2, .4, .2, count, conf, correct, None)
    assert np.isclose(row["ece15"], .2)
    summary = macro_with_bootstrap({"a": row, "b": {**row, "flood_iou": .4}},
                                   "flood_iou", repetitions=100)
    assert summary["defined_processes"] == 1
    assert summary["undefined_processes"] == 1
    assert summary["mean"] == .4


def test_block_retention_is_nested_and_respects_original_clouds():
    valid = np.ones((256, 256), dtype=bool)
    valid[:20] = False
    masks = [block_valid(valid, "EMSR150_example", level, 20260927)
             for level in (0, 25, 50, 75, 100)]
    assert not masks[0].any()
    assert all(np.all(~left | right) for left, right in zip(masks, masks[1:]))
    assert all(not mask[:20].any() for mask in masks)
