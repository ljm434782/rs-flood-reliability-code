import numpy as np

from scripts.evaluate_sen1floods11_formal import BinaryAccumulator


def test_external_binary_scores_ignore_unowned_pixels_and_no_water_event():
    score = BinaryAccumulator()
    label = np.array([[1, 0], [-1, 0]])
    probability = np.array([[.9, .1], [.8, .6]])
    owned = np.array([[True, True], [False, False]])
    score.add(label, probability, owned, probability_available=True)
    result = score.finish(probability_available=True)
    assert result["pixels"] == 2
    assert result["confusion"] == [[1, 0], [0, 1]]
    assert result["water_iou"] == 1.
    assert result["water_auprc"] == 1.

    empty = BinaryAccumulator()
    empty.add(np.zeros((1, 2), dtype=np.int64),
              np.array([[.1, .2]]), np.ones((1, 2), bool), probability_available=True)
    assert empty.finish(probability_available=True)["water_auprc"] is None
