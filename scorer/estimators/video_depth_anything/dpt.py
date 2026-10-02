"""The DPT decoder of Depth Anything V2, trimmed to the inference path.

Adapted from DepthAnything/Depth-Anything-V2 (Apache-2.0) as vendored by the official
Video-Depth-Anything repository. Batch-norm and class-token readout branches are kept
because the module names must line up with the released checkpoint, but the released
`vitl` weights use neither.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def _make_scratch(in_shape: list[int], out_shape: int) -> nn.Module:
    scratch = nn.Module()
    for index, channels in enumerate(in_shape, start=1):
        layer = nn.Conv2d(channels, out_shape, 3, 1, 1, bias=False)
        setattr(scratch, f"layer{index}_rn", layer)
    return scratch


class ResidualConvUnit(nn.Module):
    def __init__(self, features: int, activation: nn.Module, bn: bool):
        super().__init__()
        self.bn = bn
        self.conv1 = nn.Conv2d(features, features, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(features, features, 3, 1, 1, bias=True)
        if bn:
            self.bn1 = nn.BatchNorm2d(features)
            self.bn2 = nn.BatchNorm2d(features)
        self.activation = activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(self.activation(x))
        if self.bn:
            out = self.bn1(out)
        out = self.conv2(self.activation(out))
        if self.bn:
            out = self.bn2(out)
        return out + x


class FeatureFusionBlock(nn.Module):
    def __init__(self, features: int, activation: nn.Module, bn: bool = False, align_corners: bool = True):
        super().__init__()
        self.align_corners = align_corners
        self.out_conv = nn.Conv2d(features, features, 1, 1, 0, bias=True)
        self.resConfUnit1 = ResidualConvUnit(features, activation, bn)
        self.resConfUnit2 = ResidualConvUnit(features, activation, bn)

    def forward(self, *xs: torch.Tensor, size: tuple[int, int] | None = None) -> torch.Tensor:
        output = xs[0]
        if len(xs) == 2:
            output = output + self.resConfUnit1(xs[1])
        output = self.resConfUnit2(output)
        modifier = {"scale_factor": 2} if size is None else {"size": size}
        output = F.interpolate(output, mode="bilinear", align_corners=self.align_corners, **modifier)
        return self.out_conv(output)


def _make_fusion_block(features: int, use_bn: bool) -> FeatureFusionBlock:
    return FeatureFusionBlock(features, nn.ReLU(False), bn=use_bn, align_corners=True)


class DPTHead(nn.Module):
    def __init__(
        self,
        in_channels: int,
        features: int = 256,
        use_bn: bool = False,
        out_channels: list[int] = (256, 512, 1024, 1024),
        use_clstoken: bool = False,
    ):
        super().__init__()
        self.use_clstoken = use_clstoken
        out_channels = list(out_channels)

        self.projects = nn.ModuleList(
            [nn.Conv2d(in_channels, channels, 1, 1, 0) for channels in out_channels]
        )
        self.resize_layers = nn.ModuleList(
            [
                nn.ConvTranspose2d(out_channels[0], out_channels[0], 4, 4, 0),
                nn.ConvTranspose2d(out_channels[1], out_channels[1], 2, 2, 0),
                nn.Identity(),
                nn.Conv2d(out_channels[3], out_channels[3], 3, 2, 1),
            ]
        )
        if use_clstoken:
            self.readout_projects = nn.ModuleList(
                [nn.Sequential(nn.Linear(2 * in_channels, in_channels), nn.GELU()) for _ in out_channels]
            )

        self.scratch = _make_scratch(out_channels, features)
        self.scratch.refinenet1 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet2 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet3 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet4 = _make_fusion_block(features, use_bn)
        self.scratch.output_conv1 = nn.Conv2d(features, features // 2, 3, 1, 1)
        self.scratch.output_conv2 = nn.Sequential(
            nn.Conv2d(features // 2, 32, 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(32, 1, 1, 1, 0),
            nn.ReLU(True),
        )

    def project(self, out_features: tuple, patch_h: int, patch_w: int) -> list[torch.Tensor]:
        out = []
        for index, feature in enumerate(out_features):
            if self.use_clstoken:
                patches, cls_token = feature
                readout = cls_token.unsqueeze(1).expand_as(patches)
                patches = self.readout_projects[index](torch.cat((patches, readout), -1))
            else:
                patches = feature[0]
            patches = patches.permute(0, 2, 1).reshape(patches.shape[0], -1, patch_h, patch_w)
            out.append(self.resize_layers[index](self.projects[index](patches)))
        return out

    def fuse(self, layers: list[torch.Tensor], patch_h: int, patch_w: int) -> torch.Tensor:
        layer_1, layer_2, layer_3, layer_4 = layers
        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)

        path_4 = self.scratch.refinenet4(layer_4_rn, size=layer_3_rn.shape[2:])
        path_3 = self.scratch.refinenet3(path_4, layer_3_rn, size=layer_2_rn.shape[2:])
        path_2 = self.scratch.refinenet2(path_3, layer_2_rn, size=layer_1_rn.shape[2:])
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn)

        out = self.scratch.output_conv1(path_1)
        out = F.interpolate(out, (patch_h * 14, patch_w * 14), mode="bilinear", align_corners=True)
        return self.scratch.output_conv2(out)

    def forward(self, out_features: tuple, patch_h: int, patch_w: int) -> torch.Tensor:
        return self.fuse(self.project(out_features, patch_h, patch_w), patch_h, patch_w)
