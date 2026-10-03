import numpy as np
import pytest

from flood_reliability.probability_counts import (
    ProbabilitySums,
    pool_process_probability_sums,
    probability_sums,
)


def test_ignore_and_ownership_with_hand_calculated_scores():
    truth = np.array([[0, 1], [-1, 1]])
    probs = np.array([
        [[0.8, 0.2], [0.25, 0.75]],
        [[np.nan, np.nan], [0.9, 0.1]],
    ])
    owned = np.array([[True, True], [True, False]])
    sums = probability_sums(truth, probs, evaluation_mask=owned)
    assert sums.pixels == 2
    assert sums.nll == pytest.approx(-np.log(0.8) - np.log(0.75))
    assert sums.brier == pytest.approx(0.08 + 0.125)


def test_process_sums_keep_large_and_small_processes_separate():
    pooled = pool_process_probability_sums([
        ("flood_a", ProbabilitySums(2, 1.0, 0.4)),
        ("flood_a", ProbabilitySums(3, 2.0, 0.6)),
        ("flood_b", ProbabilitySums(1, 0.1, 0.02)),
    ])
    assert pooled == {
        "flood_a": ProbabilitySums(5, 3.0, 1.0),
        "flood_b": ProbabilitySums(1, 0.1, 0.02),
    }
    assert np.mean([s.nll / s.pixels for s in pooled.values()]) == pytest.approx(0.35)


def test_invalid_evaluated_probabilities_fail_and_zero_correct_is_infinite():
    target = np.array([[0]])
    for invalid in (np.array([[[0.8, 0.3]]]), np.array([[[np.nan, np.nan]]])):
        with pytest.raises(ValueError, match="probabilities"):
            probability_sums(target, invalid)
    assert np.isinf(probability_sums(target, np.array([[[0.0, 1.0]]])).nll)
    with pytest.raises(ValueError, match="unknown"):
        probability_sums(np.array([[3]]), np.array([[[0.5, 0.5]]]))


def test_case_variants_cannot_split_one_process_for_probability_metrics():
    tile = ProbabilitySums(1, 0.2, 0.1)
    pooled = pool_process_probability_sums([("EMSR388", tile), ("emsr388", tile)])
    assert pooled == {"emsr388": ProbabilitySums(2, 0.4, 0.2)}
    with pytest.raises(ValueError, match="Process ID"):
        pool_process_probability_sums([("ＥＭＳＲ３８８", tile)])
