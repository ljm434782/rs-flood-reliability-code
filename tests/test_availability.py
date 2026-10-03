import torch
import pytest

from flood_reliability.availability import apply_modality_dropout


def test_explicit_whole_modality_dropout_preserves_other_samples_and_inputs():
    optical = torch.arange(16, dtype=torch.float32).reshape(2, 2, 2, 2)
    valid = torch.tensor([[[[True, False], [True, True]]], [[[False, True], [True, True]]]])
    present = torch.tensor([True, True])
    original = optical.clone(), valid.clone(), present.clone()
    out, out_valid, out_present = apply_modality_dropout(optical, valid, present, torch.tensor([True, False]))
    assert torch.count_nonzero(out[0]) == 0
    assert not torch.any(out_valid[0])
    assert not out_present[0]
    torch.testing.assert_close(out[1], optical[1])
    torch.testing.assert_close(out_valid[1], valid[1])
    assert out_present[1]
    for before, after in zip(original, (optical, valid, present)):
        torch.testing.assert_close(before, after)


def test_absent_s2_cannot_carry_valid_pixels():
    with pytest.raises(ValueError, match="Absent S2"):
        apply_modality_dropout(
            torch.zeros(1, 2, 2, 2),
            torch.ones(1, 1, 2, 2, dtype=torch.bool),
            torch.tensor([False]),
            torch.tensor([False]),
        )
