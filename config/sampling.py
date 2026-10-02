"""The sampled evaluation timeline of a case.

Flow, Track2D, Depth, Semantic and GeoPhys use frames `0, k, 2k, ...` of an
`F`-frame video, with the stride fixed by `F` alone:

    k = min { k >= 1 : ceil(F / k) <= MAX_FRAMES }

A video of at most `MAX_FRAMES` frames has `k = 1`. The video gate still requires
`render.mp4` to carry all `F` frames.
"""

from __future__ import annotations

import numpy as np

MAX_FRAMES = 300   # the most frames the sampled timeline holds


def stride_for(frames: int) -> int:
    """Return the smallest stride with `ceil(frames / stride) <= MAX_FRAMES`."""

    frames = int(frames)
    if frames <= MAX_FRAMES:
        return 1
    stride = 1
    while -(-frames // stride) > MAX_FRAMES:
        stride += 1
    return stride


def sampled_length(frames: int, stride: int | None = None) -> int:
    """Return the number of sampled frames of a `frames`-frame video, `ceil(frames / stride)`."""

    stride = stride_for(frames) if stride is None else int(stride)
    return -(-int(frames) // max(stride, 1))


def sampled_indices(frames: int, stride: int | None = None) -> np.ndarray:
    """Return the sampled frame indices `0, k, 2k, ...` as int64."""

    stride = stride_for(frames) if stride is None else int(stride)
    return np.arange(0, int(frames), max(stride, 1), dtype=np.int64)


__all__ = ["MAX_FRAMES", "sampled_indices", "sampled_length", "stride_for"]
