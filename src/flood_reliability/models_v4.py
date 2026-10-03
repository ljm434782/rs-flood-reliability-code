"""Exploratory v4 SAR fallback with bounded, validity-local optical correction."""

import math

import torch
from torch import nn

from flood_reliability.models import SmallUNet


class RadarFallbackResidualUNet(nn.Module):
    """Predict SAR-only logits first; S2 can only add a bounded correction."""

    def __init__(self, radar_width=16, optical_width=8, residual_bound=2.0,
                 initial_gate_bias=-2.0):
        super().__init__()
        if residual_bound <= 0:
            raise ValueError("Positive residual logit bound required")
        self.residual_bound = float(residual_bound)
        self.radar = SmallUNet(2, 3, radar_width)
        # S2 bands, valid mask, age, and detached SAR class probabilities.
        self.optical_correction = SmallUNet(17, 3, optical_width)
        # Validity, presence, age, SAR entropy, and valid optical magnitude.
        self.reliability = nn.Sequential(nn.Conv2d(5, 8, 3, padding=1),
                                         nn.ReLU(inplace=True), nn.Conv2d(8, 1, 1))
        nn.init.zeros_(self.optical_correction.head.weight)
        nn.init.zeros_(self.optical_correction.head.bias)
        nn.init.zeros_(self.reliability[-1].weight)
        nn.init.constant_(self.reliability[-1].bias, initial_gate_bias)

    def forward_with_radar(self, s1, s2, optical_valid, s2_present, *, normalized_time_offset):
        if (s1.ndim != 4 or s2.ndim != 4 or s1.shape[1] != 2 or s2.shape[1] != 12
                or s1.shape[0] != s2.shape[0] or s1.shape[-2:] != s2.shape[-2:]):
            raise ValueError("Expected matching N2HW SAR and N12HW optical tensors")
        if optical_valid.shape != (s1.shape[0], 1, *s1.shape[-2:]) or optical_valid.dtype != torch.bool:
            raise ValueError("Optical validity must be boolean N1HW")
        if s2_present.shape != (s1.shape[0],) or s2_present.dtype != torch.bool:
            raise ValueError("Optical presence must be boolean N")
        if torch.any(optical_valid & ~s2_present[:, None, None, None]):
            raise ValueError("Absent S2 cannot have valid pixels")
        if (normalized_time_offset is None or normalized_time_offset.shape != (s1.shape[0],)
                or not torch.is_floating_point(normalized_time_offset)
                or not torch.all(torch.isfinite(normalized_time_offset))):
            raise ValueError("Finite normalized pre-S1 S2 age required")
        radar_logits = self.radar(s1)
        safe_optical = torch.where(optical_valid, s2, torch.zeros_like(s2))
        effective_present = s2_present & optical_valid.flatten(1).any(dim=1)
        presence = effective_present.to(s1.dtype)[:, None, None, None]
        valid = optical_valid.to(s1.dtype) * presence
        time = (normalized_time_offset * effective_present).to(s1.dtype)[:, None, None, None]
        time_map = time.expand_as(valid)
        sar_prob = torch.softmax(radar_logits.detach(), dim=1)
        sar_entropy = -(sar_prob * torch.log(sar_prob.clamp_min(1e-8))).sum(
            dim=1, keepdim=True) / math.log(3)
        optical_magnitude = safe_optical.abs().mean(dim=1, keepdim=True)
        correction_input = torch.cat((safe_optical, valid, time_map, sar_prob), dim=1)
        raw_correction = self.optical_correction(correction_input)
        correction = self.residual_bound * torch.tanh(raw_correction / self.residual_bound)
        gate_input = torch.cat((valid, presence.expand_as(valid), time_map,
                                sar_entropy, optical_magnitude), dim=1)
        gate = torch.sigmoid(self.reliability(gate_input))
        return radar_logits + valid * gate * correction, radar_logits

    def forward(self, s1, s2, optical_valid, s2_present, *, normalized_time_offset=None):
        fused, _ = self.forward_with_radar(
            s1, s2, optical_valid, s2_present,
            normalized_time_offset=normalized_time_offset)
        return fused
