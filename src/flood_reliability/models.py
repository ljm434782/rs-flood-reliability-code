"""Small encoder/decoder candidates for B2, B4/B5, and P.

The width and final class count remain configurable until the data and model
specification are frozen. This module does not define a trained model.
"""

import torch
from torch import nn
from torch.nn import functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.GroupNorm(4, out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.GroupNorm(4, out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.layers(values)


class SmallUNet(nn.Module):
    """Two-downsample U-Net family member with GroupNorm and full-size logits."""

    def __init__(self, in_channels: int, num_classes: int, base_width: int = 16):
        super().__init__()
        if in_channels < 1 or num_classes < 2 or base_width < 4 or base_width % 4:
            raise ValueError("Invalid input channels, class count, or width")
        self.enc0 = ConvBlock(in_channels, base_width)
        self.enc1 = ConvBlock(base_width, base_width * 2)
        self.bridge = ConvBlock(base_width * 2, base_width * 4)
        self.dec1 = ConvBlock(base_width * 6, base_width * 2)
        self.dec0 = ConvBlock(base_width * 3, base_width)
        self.head = nn.Conv2d(base_width, num_classes, 1)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        if values.ndim != 4 or values.shape[-2] < 4 or values.shape[-1] < 4:
            raise ValueError("Expected BCHW input of at least 4 x 4")
        skip0 = self.enc0(values)
        skip1 = self.enc1(F.max_pool2d(skip0, 2))
        bridge = self.bridge(F.max_pool2d(skip1, 2))
        up1 = F.interpolate(bridge, size=skip1.shape[-2:], mode="bilinear", align_corners=False)
        decoded1 = self.dec1(torch.cat((up1, skip1), dim=1))
        up0 = F.interpolate(decoded1, size=skip0.shape[-2:], mode="bilinear", align_corners=False)
        return self.head(self.dec0(torch.cat((up0, skip0), dim=1)))


class DualUNet(nn.Module):
    """B4/B5 shared backbone; optional availability gates make the P variant.

    B4 and B5 differ only in the training-time S2 dropout policy. The forward
    pass requires an optical-valid mask for safe zero filling but gives that
    mask, presence flag, and time feature to learned gates only when gated=True.
    """

    def __init__(
        self,
        s1_channels: int,
        s2_channels: int,
        num_classes: int,
        base_width: int = 16,
        *,
        gated: bool = False,
    ):
        super().__init__()
        if min(s1_channels, s2_channels) < 1 or num_classes < 2 or base_width < 4 or base_width % 4:
            raise ValueError("Invalid channel count, class count, or width")
        self.gated = gated
        widths = (base_width, base_width * 2, base_width * 4)
        self.sar_encoder = nn.ModuleList([
            ConvBlock(s1_channels, widths[0]),
            ConvBlock(widths[0], widths[1]),
            ConvBlock(widths[1], widths[2]),
        ])
        self.optical_encoder = nn.ModuleList([
            ConvBlock(s2_channels, widths[0]),
            ConvBlock(widths[0], widths[1]),
            ConvBlock(widths[1], widths[2]),
        ])
        self.fuse = nn.ModuleList(nn.Conv2d(width * 2, width, 1) for width in widths)
        if gated:
            self.sar_gates = nn.ModuleList(nn.Conv2d(width, width, 1) for width in widths)
            self.optical_gates = nn.ModuleList(nn.Conv2d(width + 3, width, 1) for width in widths)
        self.dec1 = ConvBlock(widths[2] + widths[1], widths[1])
        self.dec0 = ConvBlock(widths[1] + widths[0], widths[0])
        self.head = nn.Conv2d(widths[0], num_classes, 1)

    @staticmethod
    def _encode(encoder: nn.ModuleList, values: torch.Tensor) -> tuple[torch.Tensor, ...]:
        level0 = encoder[0](values)
        level1 = encoder[1](F.max_pool2d(level0, 2))
        level2 = encoder[2](F.max_pool2d(level1, 2))
        return level0, level1, level2

    def forward(
        self,
        s1: torch.Tensor,
        s2: torch.Tensor,
        optical_valid: torch.Tensor,
        s2_present: torch.Tensor,
        *,
        normalized_time_offset: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if s1.ndim != 4 or s2.ndim != 4 or s1.shape[0] != s2.shape[0] or s1.shape[-2:] != s2.shape[-2:]:
            raise ValueError("S1 and S2 must be matching BCHW tensors")
        if s1.shape[-2] < 4 or s1.shape[-1] < 4:
            raise ValueError("Spatial input must be at least 4 x 4")
        if optical_valid.shape != (s1.shape[0], 1, *s1.shape[-2:]) or optical_valid.dtype != torch.bool:
            raise ValueError("Optical-valid mask must be boolean N1HW")
        if s2_present.shape != (s1.shape[0],) or s2_present.dtype != torch.bool:
            raise ValueError("S2 presence must be a boolean vector")
        if torch.any(optical_valid & ~s2_present[:, None, None, None]):
            raise ValueError("An absent S2 cannot contain valid optical pixels")
        if self.gated and (
            normalized_time_offset is None
            or normalized_time_offset.shape != (s1.shape[0],)
            or not torch.is_floating_point(normalized_time_offset)
            or not torch.all(torch.isfinite(normalized_time_offset))
        ):
            raise ValueError("P requires one finite normalized time offset per sample")
        optical = torch.where(optical_valid, s2, torch.zeros_like(s2))
        sar_features = self._encode(self.sar_encoder, s1)
        optical_features = self._encode(self.optical_encoder, optical)
        any_valid = optical_valid.flatten(1).any(dim=1)
        effective_presence = (s2_present & any_valid).to(s1.dtype)[:, None, None, None]
        fused = []
        for level, (sar, opt) in enumerate(zip(sar_features, optical_features)):
            opt = opt * effective_presence
            if self.gated:
                valid_level = F.adaptive_max_pool2d(optical_valid.to(opt.dtype), opt.shape[-2:])
                presence_map = s2_present.to(opt.dtype)[:, None, None, None].expand_as(valid_level)
                time_map = normalized_time_offset.to(opt.dtype)[:, None, None, None].expand_as(valid_level)
                sar = sar * torch.sigmoid(self.sar_gates[level](sar))
                opt = opt * torch.sigmoid(self.optical_gates[level](torch.cat((opt, valid_level, presence_map, time_map), dim=1)))
            fused.append(self.fuse[level](torch.cat((sar, opt), dim=1)))
        up1 = F.interpolate(fused[2], size=fused[1].shape[-2:], mode="bilinear", align_corners=False)
        decoded1 = self.dec1(torch.cat((up1, fused[1]), dim=1))
        up0 = F.interpolate(decoded1, size=fused[0].shape[-2:], mode="bilinear", align_corners=False)
        return self.head(self.dec0(torch.cat((up0, fused[0]), dim=1)))
