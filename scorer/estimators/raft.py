"""RAFT-large optical flow of a video: the reference side of the Flow metric.

Loads torchvision's `C_T_SKHT_V2` weights from the checkpoint store and applies their
preprocessing (uint8 -> [-1, 1]). Computes forward flow for each pair `t -> t + stride`
with no occlusion mask (`valid` is all true), as in VisPhyWorld (arXiv 2602.13294).
Decodes frames one at a time; peak memory depends on `BATCH` and resolution only.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torchvision.models.optical_flow import raft_large


from ..prefetch import prefetch
from .video_depth import frame_stream

WEIGHTS = "raft_large_C_T_SKHT_V2.pth"
ITERATIONS = 12          # flow updates per pair, the released model's default
MULTIPLE = 8             # RAFT's feature stride: both sides are padded to it
BATCH = 2                # frame pairs per forward pass


def default_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(directory: str | Path,
               device: str | torch.device | None = None):
    """Load RAFT-large from `directory` onto `device` in evaluation mode."""

    device = torch.device(device) if device is not None else default_device()
    path = Path(directory) / WEIGHTS
    model = raft_large(weights=None)
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    return model.to(device).eval()


def preprocess(frames, device: torch.device) -> torch.Tensor:
    """Convert `(N, H, W, 3)` RGB uint8 frames to `(N, 3, H, W)` float in [-1, 1]."""

    batch = torch.from_numpy(frames).to(device).permute(0, 3, 1, 2).float() / 255.0
    return 2.0 * batch - 1.0


def _pad(images: torch.Tensor) -> torch.Tensor:
    """Pad height and width to a multiple of `MULTIPLE` by edge replication."""

    height, width = images.shape[-2:]
    return F.pad(images, (0, (-width) % MULTIPLE, 0, (-height) % MULTIPLE), mode="replicate")


@torch.inference_mode()
def stream_flow(video: str | Path, model, device: str | torch.device | None = None,
                batch: int = BATCH, stride: int = 1) -> Iterator[tuple]:
    """Yield `flow (H, W, 2)` float32 and `valid (H, W)` bool per sampled pair, on the CPU.

    The pairs are consecutive frames of the timeline sampled at `stride`,
    `t -> t + stride`. `flow` is `(u, v)` in full-resolution pixels; `valid` is all
    true.
    """

    device = torch.device(device) if device is not None else default_device()
    with prefetch(_windows(video, batch, stride)) as prepared:
        for window in prepared:
            yield from _pairs(window, model, device)


def _windows(video, batch, stride):
    """Yield overlapping windows of `batch + 1` sampled frames (the last may be shorter)."""
    source = frame_stream(video, stride)
    try:
        window = []
        for frame in source:
            window.append(frame)
            if len(window) == batch + 1:
                yield window
                window = window[-1:]
        if len(window) > 1:
            yield window
    finally:
        source.close()


def _pairs(window: list, model, device: torch.device) -> Iterator[tuple]:
    """Yield the forward flow of every consecutive pair in a window, from one pass."""

    images = preprocess(np.stack(window), device)
    forward = model(_pad(images[:-1]), _pad(images[1:]), num_flow_updates=ITERATIONS)[-1]
    forward = forward[..., :images.shape[-2], :images.shape[-1]]
    for index in range(len(forward)):
        yield (forward[index].permute(1, 2, 0).float().cpu().numpy(),
               np.ones(forward.shape[-2:], dtype=bool))


__all__ = ["BATCH", "ITERATIONS", "load_model", "preprocess", "stream_flow"]
