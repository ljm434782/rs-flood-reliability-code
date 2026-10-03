"""Isolated v2 ablation of the existing P gate, without changing v1 code."""

import torch
from torch.nn import functional as F

from flood_reliability.models import DualUNet


class OpticalOnlyGatedDualUNet(DualUNet):
    """Keep P's optical gate but bypass its sigmoid SAR feature scaling."""

    def __init__(self, s1_channels=2, s2_channels=12, num_classes=3, base_width=16):
        super().__init__(s1_channels, s2_channels, num_classes, base_width, gated=True)

    def forward(self, s1, s2, optical_valid, s2_present, *, normalized_time_offset=None):
        if s1.ndim != 4 or s2.ndim != 4 or s1.shape[0] != s2.shape[0] or s1.shape[-2:] != s2.shape[-2:]:
            raise ValueError("S1 and S2 must be matching BCHW tensors")
        if optical_valid.shape != (s1.shape[0], 1, *s1.shape[-2:]) or optical_valid.dtype != torch.bool:
            raise ValueError("Optical-valid mask must be boolean N1HW")
        if s2_present.shape != (s1.shape[0],) or s2_present.dtype != torch.bool:
            raise ValueError("S2 presence must be a boolean vector")
        if torch.any(optical_valid & ~s2_present[:, None, None, None]):
            raise ValueError("An absent S2 cannot contain valid optical pixels")
        if (normalized_time_offset is None or normalized_time_offset.shape != (s1.shape[0],)
                or not torch.all(torch.isfinite(normalized_time_offset))):
            raise ValueError("A finite normalized time offset is required")
        optical = torch.where(optical_valid, s2, torch.zeros_like(s2))
        sar_features = self._encode(self.sar_encoder, s1)
        optical_features = self._encode(self.optical_encoder, optical)
        presence = (s2_present & optical_valid.flatten(1).any(dim=1)).to(s1.dtype)[:, None, None, None]
        fused = []
        for level, (sar, opt) in enumerate(zip(sar_features, optical_features)):
            opt = opt * presence
            valid_level = F.adaptive_max_pool2d(optical_valid.to(opt.dtype), opt.shape[-2:])
            present_map = s2_present.to(opt.dtype)[:, None, None, None].expand_as(valid_level)
            time_map = normalized_time_offset.to(opt.dtype)[:, None, None, None].expand_as(valid_level)
            opt = opt * torch.sigmoid(self.optical_gates[level](
                torch.cat((opt, valid_level, present_map, time_map), dim=1)))
            fused.append(self.fuse[level](torch.cat((sar, opt), dim=1)))
        up1 = F.interpolate(fused[2], size=fused[1].shape[-2:], mode="bilinear", align_corners=False)
        decoded1 = self.dec1(torch.cat((up1, fused[1]), dim=1))
        up0 = F.interpolate(decoded1, size=fused[0].shape[-2:], mode="bilinear", align_corners=False)
        return self.head(self.dec0(torch.cat((up0, fused[0]), dim=1)))
