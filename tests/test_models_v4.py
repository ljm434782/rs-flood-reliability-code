import torch

from flood_reliability.models_v4 import RadarFallbackResidualUNet


def test_fallback_is_exact_for_missing_or_invalid_optical_pixels():
    torch.manual_seed(19)
    model = RadarFallbackResidualUNet(radar_width=8, optical_width=4)
    with torch.no_grad():
        model.optical_correction.head.bias.fill_(1.0)
    sar = torch.randn(2, 2, 16, 16)
    optical = torch.randn(2, 12, 16, 16)
    valid = torch.ones(2, 1, 16, 16, dtype=torch.bool)
    valid[0] = False
    valid[1, :, :8] = False
    present = torch.tensor([False, True])
    fused, radar = model.forward_with_radar(sar, optical, valid, present,
                                             normalized_time_offset=torch.tensor([0., -.3]))
    assert torch.equal(fused[0], radar[0])
    assert torch.equal(fused[1, :, :8], radar[1, :, :8])
    assert torch.any(fused[1, :, 8:] != radar[1, :, 8:])
    assert (fused - radar).abs().max() <= 2.0


def test_zero_initial_correction_starts_from_radar_and_can_learn():
    torch.manual_seed(23)
    model = RadarFallbackResidualUNet(radar_width=8, optical_width=4)
    sar, optical = torch.randn(1, 2, 16, 16), torch.randn(1, 12, 16, 16)
    valid = torch.ones(1, 1, 16, 16, dtype=torch.bool)
    fused, radar = model.forward_with_radar(
        sar, optical, valid, torch.tensor([True]), normalized_time_offset=torch.tensor([-.2]))
    assert torch.equal(fused, radar)
    fused.square().mean().backward()
    assert torch.any(model.radar.head.weight.grad != 0)
    assert torch.any(model.optical_correction.head.weight.grad != 0)
