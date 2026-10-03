import numpy as np
import pytest

from flood_reliability.evaluation_counts import confusion_counts, pool_process_confusions


def test_ignore_and_owned_pixels_do_not_enter_confusion():
    truth = np.array([[0, 1, 2], [-1, 255, 2]], dtype=np.int16)
    guess = np.array([[0, 2, 1], [99, 99, 2]], dtype=np.int16)
    owned = np.array([[True, True, False], [True, True, True]])
    counts = confusion_counts(truth, guess, num_classes=3, evaluation_mask=owned)
    np.testing.assert_array_equal(counts, [[1, 0, 0], [0, 0, 1], [0, 0, 1]])
    assert counts.sum() == 3


def test_process_pooling_sums_counts_without_defining_empty_class_metric():
    one = confusion_counts(np.array([[2, 2]]), np.array([[2, 0]]), num_classes=3)
    two = confusion_counts(np.array([[0, 2]]), np.array([[0, 2]]), num_classes=3)
    empty = confusion_counts(np.array([[0, 1]]), np.array([[0, 1]]), num_classes=3)
    pooled = pool_process_confusions([("flood_process", one), ("flood_process", two), ("no_flood_process", empty)])
    np.testing.assert_array_equal(pooled["flood_process"], one + two)
    assert pooled["flood_process"][2].sum() == 3
    assert pooled["no_flood_process"][2].sum() == 0


def test_unknown_labels_predictions_and_bad_support_fail_loudly():
    target = np.array([[0, 4]])
    pred = np.array([[0, 1]])
    with pytest.raises(ValueError, match="unknown"):
        confusion_counts(target, pred, num_classes=3)
    with pytest.raises(ValueError, match="out-of-range"):
        confusion_counts(np.array([[0, 1]]), np.array([[0, 3]]), num_classes=3)
    with pytest.raises(ValueError, match="boolean"):
        confusion_counts(np.array([[0, 1]]), pred, num_classes=3, evaluation_mask=np.array([[1, 1]]))


def test_case_variants_cannot_split_one_process_for_hard_metrics():
    matrix = confusion_counts(np.array([[2]]), np.array([[2]]), num_classes=3)
    pooled = pool_process_confusions([("EMSR388", matrix), ("emsr388", matrix)])
    assert list(pooled) == ["emsr388"]
    np.testing.assert_array_equal(pooled["emsr388"], 2 * matrix)
    with pytest.raises(ValueError, match="Process ID"):
        pool_process_confusions([("EMSR388\u200b", matrix)])
