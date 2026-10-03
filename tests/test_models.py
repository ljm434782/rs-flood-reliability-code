import torch
import pytest

from flood_reliability.models import DualUNet, SmallUNet


def test_s1_model_has_full_resolution_logits_and_receives_gradients():
    torch.manual_seed(3)
    model = SmallUNet(in_channels=2, num_classes=3, base_width=8)
    sample = torch.randn(1, 2, 33, 35)
    logits = model(sample)
    assert logits.shape == (1, 3, 33, 35)
    labels = torch.randint(0, 3, (1, 33, 35))
    labels[:, 0, :] = 255
    loss = torch.nn.functional.cross_entropy(logits, labels, ignore_index=255)
    loss.backward()
    assert torch.isfinite(loss)
    assert model.enc0.layers[0].weight.grad is not None
    assert torch.any(model.enc0.layers[0].weight.grad != 0)


@pytest.mark.parametrize("gated", [False, True])
def test_dual_encoder_logits_and_gate_gradients(gated):
    torch.manual_seed(4)
    model = DualUNet(2, 4, 3, base_width=8, gated=gated)
    s1 = torch.randn(1, 2, 33, 35)
    s2 = torch.randn(1, 4, 33, 35)
    valid = torch.ones(1, 1, 33, 35, dtype=torch.bool)
    logits = model(s1, s2, valid, torch.tensor([True]), normalized_time_offset=torch.tensor([-0.2]) if gated else None)
    assert logits.shape == (1, 3, 33, 35)
    logits.square().mean().backward()
    assert torch.any(model.sar_encoder[0].layers[0].weight.grad != 0)
    assert torch.any(model.optical_encoder[0].layers[0].weight.grad != 0)
    if gated:
        assert torch.any(model.optical_gates[0].weight.grad != 0)


def test_absent_optical_is_invariant_to_its_values_and_mask_is_checked():
    model = DualUNet(2, 4, 3, base_width=8, gated=False).eval()
    s1 = torch.randn(1, 2, 16, 16)
    s2 = torch.randn(1, 4, 16, 16)
    unavailable = torch.zeros(1, 1, 16, 16, dtype=torch.bool)
    absent = torch.tensor([False])
    with torch.no_grad():
        expected = model(s1, s2, unavailable, absent)
        actual = model(s1, torch.full_like(s2, float("nan")), unavailable, absent)
    torch.testing.assert_close(actual, expected)
    with pytest.raises(ValueError, match="absent S2"):
        model(s1, s2, torch.ones_like(unavailable), absent)
