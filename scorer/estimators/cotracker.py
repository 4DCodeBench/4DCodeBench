"""CoTracker3 point tracks of a video: the reference side of the Track2D metric.

Runs the offline model (`scaled_offline.pth`) over the whole sampled video at once,
so a track resumes after an occlusion. The video is resized to at most `LONG_SIDE`
pixels on its long edge, and the tracks are scaled back to full-resolution pixels.
Queries run in chunks of `CHUNK` against one resident clip; memory grows with
queries times frames. Tracking runs forward only (`backward_tracking=False`): a
track is NaN and invisible before its query frame.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch


from .video_depth import frame_stream

WEIGHTS = "scaled_offline.pth"
LONG_SIDE = 768     # the long edge the video is tracked at
CHUNK = 256         # queries per forward pass


def default_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(directory: str | Path,
               device: str | torch.device | None = None):
    """Load the offline CoTracker3 predictor from `directory` onto `device`."""

    from cotracker.predictor import CoTrackerPredictor

    device = torch.device(device) if device is not None else default_device()
    path = Path(directory) / WEIGHTS
    return CoTrackerPredictor(checkpoint=str(path), offline=True).to(device).eval()


def tracking_shape(height: int, width: int, long_side: int = LONG_SIDE) -> tuple[int, int]:
    """Return the tracking size `(h, w)`: the input size, scaled down to `long_side` if larger."""

    if max(height, width) <= long_side:
        return height, width
    scale = long_side / max(height, width)
    return max(1, round(height * scale)), max(1, round(width * scale))


def _video(path: str | Path, device: torch.device, stride: int = 1) -> tuple[torch.Tensor, tuple[int, int]]:
    """Return the sampled video as `(1, F, 3, h, w)` float in [0, 255], and its `(H, W)`."""

    frames = list(frame_stream(path, stride))
    height, width = frames[0].shape[:2]
    shape = tracking_shape(height, width)
    resized = np.stack([cv2.resize(frame, shape[::-1], interpolation=cv2.INTER_AREA)
                        for frame in frames])
    return torch.from_numpy(resized).to(device).permute(0, 3, 1, 2).float()[None], (height, width)


@torch.inference_mode()
def track(video: str | Path, queries: np.ndarray, model,
          device: str | torch.device | None = None,
          stride: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """Return `tracks (Q, F, 2)` float32 and `visible (Q, F)` bool for `(Q, 3)` queries.

    Queries are `(t, x, y)` and tracks `(x, y)`, both in full-resolution pixels. The
    video is sampled at `stride`, so `F` is the sampled frame count and `t` indexes
    the sampled timeline. A track is NaN and invisible before frame `t`.
    """

    device = torch.device(device) if device is not None else default_device()
    clip, (height, width) = _video(video, device, stride)
    ratio = np.asarray([clip.shape[-1] / width, clip.shape[-2] / height], dtype=np.float32)

    seeds = np.asarray(queries, dtype=np.float32).copy()
    seeds[:, 1:] *= ratio
    frames = clip.shape[1]
    tracks = np.zeros((len(seeds), frames, 2), np.float32)
    visible = np.zeros((len(seeds), frames), bool)
    for lo in range(0, len(seeds), CHUNK):
        chunk = torch.from_numpy(seeds[lo:lo + CHUNK]).to(device)[None]
        positions, visibility = model(clip, queries=chunk, backward_tracking=False)
        tracks[lo:lo + CHUNK] = positions[0].permute(1, 0, 2).float().cpu().numpy() / ratio
        visible[lo:lo + CHUNK] = visibility[0].permute(1, 0).cpu().numpy()
        del positions, visibility
        torch.cuda.empty_cache()   # each chunk re-encodes the clip; limits fragmentation
    del clip

    unborn = np.arange(tracks.shape[1])[None] < np.asarray(queries[:, 0], dtype=np.int64)[:, None]
    tracks[unborn] = np.nan
    visible[unborn] = False
    return tracks, visible


__all__ = ["CHUNK", "LONG_SIDE", "load_model", "track", "tracking_shape"]
