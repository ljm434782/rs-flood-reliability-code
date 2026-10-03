import torch

from flood_reliability.exploratory_models import OpticalOnlyGatedDualUNet
from flood_reliability.models_v3 import ResidualSARGatedDualUNet
from scripts.train_impactmesh_exploratory_v3 import flood_soft_dice


def test_residual_gate_starts_at_identity_and_learns():
    torch.manual_seed(7)
    residual = ResidualSARGatedDualUNet(2, 4, 3, 8)
    baseline = OpticalOnlyGatedDualUNet(2, 4, 3, 8)
    baseline.load_state_dict(residual.state_dict())
    sar, optical = torch.randn(1, 2, 16, 16), torch.randn(1, 4, 16, 16)
    valid = torch.ones(1, 1, 16, 16, dtype=torch.bool)
    presence, offset = torch.tensor([True]), torch.tensor([-.5])
    actual = residual(sar, optical, valid, presence, normalized_time_offset=offset)
    expected = baseline(sar, optical, valid, presence, normalized_time_offset=offset)
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)
    actual.square().mean().backward()
    assert any(torch.any(gate.weight.grad != 0) for gate in residual.sar_gates)


def test_flood_soft_dice_ignores_label_255():
    logits = torch.randn(1, 3, 4, 4)
    label = torch.zeros(1, 4, 4, dtype=torch.long)
    label[0, 0, 0] = 255
    original = flood_soft_dice(logits, label)
    logits[0, :, 0, 0] = 1000
    torch.testing.assert_close(flood_soft_dice(logits, label), original)
