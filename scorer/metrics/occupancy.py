"""Occupancy DTW: time-warped sliced Wasserstein distance between the two worlds' matter.

At `FRAMES` evenly spaced frames, both worlds' material is sampled as a point set in
reference radii and culled to the reference camera's view. Every reference frame is
compared with every submission frame by `semd`, capped at `CAP` (and set to `CAP` where
the submission has no matter). With `W` the per-step DTW cost of that matrix (Sakoe &
Chiba 1978), `occupancy_dtw_score = max(0, 1 - W / CAP)`. Frames where the reference
has no visible matter are skipped.
"""

from __future__ import annotations

import numpy as np

from ..samples import Sampling, dynamic_frames, frame_samples, visible
from ..world import World
from .distances import seed, semd
from .registration import Registration

CAP = 0.5          # per-pair distance cap, in reference radii
FRAMES = 16        # evenly spaced frames compared


def clouds_at(gt: World, pred: World, alignment: Registration, sampling: Sampling,
              frame: int) -> tuple[np.ndarray, np.ndarray]:
    """Both sides' material samples at one frame, in reference radii, culled to the reference view.

    `sampling` positions are already registered; each side's sampler is seeded with
    `(SEED, frame)`.
    """

    out = []
    for world, positions in ((gt, sampling.gt), (pred, sampling.pred)):
        generator = np.random.default_rng([seed(), frame])
        points = frame_samples(world, positions, sampling.h, frame, generator).positions(frame)
        if not len(points):
            out.append(np.zeros((0, 3), np.float32))
            continue
        out.append(points[visible(points, alignment, gt)])
    return out[0], out[1]


def warp(cost: np.ndarray) -> float:
    """Accumulated cost per step of the cheapest monotone path through `cost`."""

    n, m = cost.shape
    total = np.full((n + 1, m + 1), np.inf)
    steps = np.zeros((n + 1, m + 1))
    total[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            best = min((total[i - 1, j], steps[i - 1, j]),
                       (total[i, j - 1], steps[i, j - 1]),
                       (total[i - 1, j - 1], steps[i - 1, j - 1]))
            total[i, j] = best[0] + cost[i - 1, j - 1]
            steps[i, j] = best[1] + 1
    return float(total[n, m] / max(steps[n, m], 1.0))


def occupancy_error(gt: World, pred: World, alignment: Registration, sampling: Sampling) -> dict:
    """Occupancy DTW readings; the scores are None without a voxel size or reference matter."""

    empty = {"arrays": {"occupancy": {}}, "occupancy_dtw_error": None, "occupancy_dtw_score": None}
    if sampling.h is None:
        return empty
    read, left, right = [], [], []
    for frame in dynamic_frames(len(gt), FRAMES):
        x, y = clouds_at(gt, pred, alignment, sampling, frame)
        if not len(x):
            continue                      # no visible reference matter
        read.append(frame)
        left.append(x)
        right.append(y)
    if not read:
        return empty

    def gap(x: np.ndarray, y: np.ndarray) -> float:
        # an empty submission frame costs `CAP`
        return CAP if not len(y) else min(float(semd(x, y)), CAP)

    # every reference frame against every submission frame
    cross = np.asarray([[gap(x, y) for y in right] for x in left], np.float32)
    warped = warp(cross)
    return {"occupancy_dtw_error": warped, "occupancy_dtw_score": float(max(0.0, 1.0 - warped / CAP)),
            "occupancy_frames": [int(frame) for frame in read],
            "arrays": {"occupancy": {"cross": cross, "frames": np.asarray(read, np.int64)}}}


__all__ = ["CAP", "FRAMES", "clouds_at", "occupancy_error", "warp"]
