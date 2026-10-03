"""Contract checks for v5 feature fusion and SAR fallback."""

import torch

from flood_reliability.models import DualUNet
from flood_reliability.models_v5 import RadarFeatureResidualUNet


def test_feature_model_exact_sar_fallback_at_invalid_pixels():
    torch.manual_seed(11)
    model = RadarFeatureResidualUNet(width=4)
    model.eval()
    sar = torch.randn(2, 2, 16, 16)
    optical = torch.randn(2, 12, 16, 16)
    valid = torch.zeros(2, 1, 16, 16, dtype=torch.bool)
    valid[1, :, :8, :8] = True
    present = torch.tensor([False, True])
    age = torch.tensor([0.0, -0.2])
    with torch.no_grad():
        fused, radar = model.forward_with_radar(
            sar, optical, valid, present, normalized_time_offset=age)
    assert torch.equal(fused[0], radar[0])
    assert torch.equal(fused[1, :, 8:, 8:], radar[1, :, 8:, 8:])
    # Zero-initialized adapters make the initial model SAR-only even on valid S2.
    assert torch.equal(fused, radar)
    with torch.no_grad():
        model.optical_adapters[0].weight[0, 0, 0, 0] = 2.0
        fused, radar = model.forward_with_radar(
            sar, optical, valid, present, normalized_time_offset=age)
    assert torch.equal(fused[0], radar[0])
    assert torch.equal(fused[1, :, 8:, 8:], radar[1, :, 8:, 8:])


def test_feature_model_capacity_and_optical_gradient():
    model = RadarFeatureResidualUNet(width=16)
    control = DualUNet(2, 12, 3, 16)
    ratio = sum(p.numel() for p in model.parameters()) / sum(p.numel() for p in control.parameters())
    assert 0.95 <= ratio <= 1.05
    sar = torch.randn(1, 2, 16, 16)
    optical = torch.randn(1, 12, 16, 16)
    valid = torch.ones(1, 1, 16, 16, dtype=torch.bool)
    logits, radar = model.forward_with_radar(
        sar, optical, valid, torch.ones(1, dtype=torch.bool),
        normalized_time_offset=torch.zeros(1))
    (logits.square().mean() + 0.3 * radar.square().mean()).backward()
    assert model.optical_adapters[0].weight.grad is not None
    assert torch.isfinite(model.optical_adapters[0].weight.grad).all()
    assert model.optical_adapters[0].weight.grad.abs().sum() > 0
