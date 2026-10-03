"""Exploratory feature-level optical residual with exact SAR fallback."""

import math

import torch
from torch import nn
from torch.nn import functional as F

from flood_reliability.models import ConvBlock, SmallUNet


class RadarFeatureResidualUNet(nn.Module):
    """Share a decoder between an independent SAR path and an optical feature update."""

    def __init__(self, width=16, gate_bias=-2.0):
        super().__init__()
        if width < 4 or width % 4:
            raise ValueError("Width must be a positive multiple of four")
        self.radar = SmallUNet(2, 3, width)
        widths = (width, width * 2, width * 4)
        self.optical_encoder = nn.ModuleList((
            ConvBlock(14, widths[0]),
            ConvBlock(widths[0], widths[1]),
            ConvBlock(widths[1], widths[2]),
        ))
        self.optical_adapters = nn.ModuleList(nn.Conv2d(w, w, 1) for w in widths)
        self.reliability = nn.ModuleList(nn.Conv2d(w + 4, 1, 1) for w in widths)
        for adapter, gate in zip(self.optical_adapters, self.reliability):
            nn.init.zeros_(adapter.weight)
            nn.init.zeros_(adapter.bias)
            nn.init.zeros_(gate.weight)
            nn.init.constant_(gate.bias, gate_bias)

    @staticmethod
    def _encode(blocks, values):
        first = blocks[0](values)
        second = blocks[1](F.max_pool2d(first, 2))
        third = blocks[2](F.max_pool2d(second, 2))
        return first, second, third

    def _decode(self, features):
        first, second, third = features
        up1 = F.interpolate(third, size=second.shape[-2:], mode="bilinear", align_corners=False)
        decoded1 = self.radar.dec1(torch.cat((up1, second), dim=1))
        up0 = F.interpolate(decoded1, size=first.shape[-2:], mode="bilinear", align_corners=False)
        return self.radar.head(self.radar.dec0(torch.cat((up0, first), dim=1)))

    def forward_with_radar(self, s1, s2, optical_valid, s2_present, *, normalized_time_offset):
        if (s1.ndim != 4 or s2.ndim != 4 or s1.shape[1] != 2 or s2.shape[1] != 12
                or s1.shape[0] != s2.shape[0] or s1.shape[-2:] != s2.shape[-2:]
                or min(s1.shape[-2:]) < 4):
            raise ValueError("Expected matching N2HW SAR and N12HW optical tensors")
        if optical_valid.shape != (s1.shape[0], 1, *s1.shape[-2:]) or optical_valid.dtype != torch.bool:
            raise ValueError("Optical validity must be boolean N1HW")
        if s2_present.shape != (s1.shape[0],) or s2_present.dtype != torch.bool:
            raise ValueError("Optical presence must be boolean N")
        if torch.any(optical_valid & ~s2_present[:, None, None, None]):
            raise ValueError("Absent S2 cannot have valid optical pixels")
        if (normalized_time_offset is None or normalized_time_offset.shape != (s1.shape[0],)
                or not torch.is_floating_point(normalized_time_offset)
                or not torch.all(torch.isfinite(normalized_time_offset))):
            raise ValueError("Finite normalized S2 age required")

        sar_features = self._encode((self.radar.enc0, self.radar.enc1, self.radar.bridge), s1)
        radar_logits = self._decode(sar_features)
        if not torch.any(optical_valid):
            return radar_logits, radar_logits

        present = s2_present & optical_valid.flatten(1).any(dim=1)
        valid_float = optical_valid.to(s1.dtype)
        time = (normalized_time_offset * present).to(s1.dtype)[:, None, None, None]
        time_map = time.expand_as(valid_float)
        safe_optical = torch.where(optical_valid, s2, torch.zeros_like(s2))
        optical_input = torch.cat((safe_optical, valid_float, time_map), dim=1)
        optical_features = self._encode(self.optical_encoder, optical_input)
        probability = torch.softmax(radar_logits.detach(), dim=1)
        entropy = -(probability * torch.log(probability.clamp_min(1e-8))).sum(
            dim=1, keepdim=True) / math.log(3)

        fused_features = []
        for sar, opt, adapter, gate in zip(
                sar_features, optical_features, self.optical_adapters, self.reliability):
            valid = F.adaptive_max_pool2d(valid_float, sar.shape[-2:])
            presence = present.to(s1.dtype)[:, None, None, None].expand_as(valid)
            age = F.interpolate(time_map, size=sar.shape[-2:], mode="nearest")
            uncertainty = F.interpolate(entropy, size=sar.shape[-2:], mode="bilinear",
                                        align_corners=False)
            reliability = torch.sigmoid(gate(torch.cat((opt, valid, presence, age,
                                                        uncertainty), dim=1)))
            fused_features.append(sar + valid * reliability * adapter(opt))
        optical_logits = self._decode(fused_features)
        # Enforce a per-pixel exact fallback even at an invalid pixel next to valid S2.
        return radar_logits + valid_float * (optical_logits - radar_logits), radar_logits

    def forward(self, s1, s2, optical_valid, s2_present, *, normalized_time_offset=None):
        fused, _ = self.forward_with_radar(
            s1, s2, optical_valid, s2_present,
            normalized_time_offset=normalized_time_offset)
        return fused
