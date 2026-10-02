"""The DPT decoder with temporal attention, as released by Video Depth Anything.

Adapted from DepthAnything/Video-Depth-Anything (Apache-2.0), inference path only.
"""

from __future__ import annotations

import torch

from .dpt import DPTHead
from .motion_module import TemporalModule

NUM_FRAMES = 32
OUTPUT_CHUNK = 4


class DPTHeadTemporal(DPTHead):
    """`DPTHead` with four temporal modules: two on the deep features, two on the paths."""

    def __init__(
        self,
        in_channels: int,
        features: int = 256,
        use_bn: bool = False,
        out_channels: list[int] = (256, 512, 1024, 1024),
        use_clstoken: bool = False,
        num_frames: int = NUM_FRAMES,
    ):
        super().__init__(in_channels, features, use_bn, out_channels, use_clstoken)
        widths = [out_channels[2], out_channels[3], features, features]
        self.motion_modules = torch.nn.ModuleList(
            [TemporalModule(in_channels=width, temporal_max_len=num_frames) for width in widths]
        )

    def _temporal(self, index: int, x: torch.Tensor, batch: int, frames: int) -> torch.Tensor:
        stacked = x.unflatten(0, (batch, frames)).permute(0, 2, 1, 3, 4)
        return self.motion_modules[index](stacked).permute(0, 2, 1, 3, 4).flatten(0, 1)

    def forward(self, out_features: tuple, patch_h: int, patch_w: int, frame_length: int) -> torch.Tensor:
        layer_1, layer_2, layer_3, layer_4 = self.project(out_features, patch_h, patch_w)
        batch = layer_1.shape[0] // frame_length

        layer_3 = self._temporal(0, layer_3, batch, frame_length)
        layer_4 = self._temporal(1, layer_4, batch, frame_length)

        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)

        path_4 = self.scratch.refinenet4(layer_4_rn, size=layer_3_rn.shape[2:])
        path_4 = self._temporal(2, path_4, batch, frame_length)
        path_3 = self.scratch.refinenet3(path_4, layer_3_rn, size=layer_2_rn.shape[2:])
        path_3 = self._temporal(3, path_3, batch, frame_length)

        # Nothing after the last temporal module mixes frames, and the two shallow fusion
        # blocks are where a whole window at full resolution would peak, so the tail runs
        # in frame slices. The result is identical to running the window in one go.
        size = (patch_h * 14, patch_w * 14)
        slices = [
            self._tail(
                path_3[begin : begin + OUTPUT_CHUNK],
                layer_2_rn[begin : begin + OUTPUT_CHUNK],
                layer_1_rn[begin : begin + OUTPUT_CHUNK],
                size,
            )
            for begin in range(0, path_3.shape[0], OUTPUT_CHUNK)
        ]
        return torch.cat(slices)

    def _tail(
        self,
        path_3: torch.Tensor,
        layer_2_rn: torch.Tensor,
        layer_1_rn: torch.Tensor,
        size: tuple[int, int],
    ) -> torch.Tensor:
        path_2 = self.scratch.refinenet2(path_3, layer_2_rn, size=layer_1_rn.shape[2:])
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn)
        out = self.scratch.output_conv1(path_1)
        out = torch.nn.functional.interpolate(out, size, mode="bilinear", align_corners=True)
        return self.scratch.output_conv2(out)
