import torch

from flood_reliability.exploratory_models import OpticalOnlyGatedDualUNet
from flood_reliability.models import DualUNet


def test_optical_only_gate_is_p_without_sar_feature_scaling():
    torch.manual_seed(20260928)
    original = DualUNet(2, 4, 3, 8, gated=True).eval()
    ablated = OpticalOnlyGatedDualUNet(2, 4, 3, 8).eval()
    ablated.load_state_dict(original.state_dict())
    for gate in original.sar_gates:
        torch.nn.init.zeros_(gate.weight)
        torch.nn.init.constant_(gate.bias, 20.)
    sar = torch.randn(1, 2, 16, 16)
    optical = torch.randn(1, 4, 16, 16)
    valid = torch.ones(1, 1, 16, 16, dtype=torch.bool)
    present = torch.tensor([True])
    offset = torch.tensor([-.4])
    torch.testing.assert_close(
        ablated(sar, optical, valid, present, normalized_time_offset=offset),
        original(sar, optical, valid, present, normalized_time_offset=offset),
        atol=1e-6, rtol=1e-6)
    ablated(sar, optical, valid, present, normalized_time_offset=offset).square().mean().backward()
    assert ablated.sar_gates[0].weight.grad is None
    assert torch.any(ablated.optical_gates[0].weight.grad != 0)
