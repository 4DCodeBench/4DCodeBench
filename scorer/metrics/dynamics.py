"""EMD step: sliced Wasserstein distance between the two sides' one-step 3D displacements.

Inputs are the Trajectory paths, `(S, F, 3)` per side in reference radii, NaN outside
a sample's life or the reference view. The sides have no sample correspondence, so at
each frame `t` the displacement sets `U_t`, `V_t` are compared as point sets by `semd`:

    emd_step = max(0, 1 - sum_t SW(U_t, V_t) / sum_t (m(U_t) + m(V_t)))

with `m` the per-component RMS. Frames with fewer than `MIN_POINTS` reference
displacements are skipped; on a frame where the submission has fewer, `SW` is set to
the frame's normaliser `m(U_t) + m(V_t)`.
"""

from __future__ import annotations

import time

import numpy as np

from .distances import semd

CAP = 1.0           # upper bound of a frame's value in `emd_step_curve`
MIN_POINTS = 8      # minimum displacements for a side to count on a frame
EPS = 1e-9


def _alive(cloud: np.ndarray) -> np.ndarray:
    return cloud[np.isfinite(cloud).all(axis=1)]


def emd_step(gt: np.ndarray, pred: np.ndarray) -> dict:
    """Return the per-frame normalised curve and `emd_step_score` (None if no frame counts)."""

    curve, distances, magnitudes = [], [], []
    for t in range(gt.shape[1] - 1):
        moves = [_alive(np.where(np.isfinite(side[:, t + 1]) & np.isfinite(side[:, t]),
                                 side[:, t + 1] - side[:, t], np.nan))
                 for side in (gt, pred)]
        # the reference side selects the frames
        if len(moves[0]) < MIN_POINTS:
            continue
        magnitude = sum(float(np.sqrt((move ** 2).mean())) for move in moves if len(move))
        answered = len(moves[1]) >= MIN_POINTS
        distance = semd(*moves) if answered else magnitude
        curve.append(min(CAP, distance / (magnitude + EPS)) if answered else CAP)
        distances.append(distance)
        magnitudes.append(magnitude)
    total = sum(magnitudes)
    return {"emd_step_curve": curve,
            "emd_step_score": float(max(0.0, 1.0 - sum(distances) / total)) if total else None}


def dynamics_readings(gt: np.ndarray | None, pred: np.ndarray | None) -> dict:
    """EMD step readings and the per-frame curve array for one pair of Trajectory path sets."""

    if gt is None or pred is None or not len(gt) or not len(pred):
        return {"emd_step_score": None}
    started = time.time()
    reading = emd_step(gt, pred)
    curve = np.asarray(reading.pop("emd_step_curve"), np.float32)
    return {"arrays": {"dynamics": {"emd_step": curve}},
            "dynamics_paths": [len(gt), len(pred)], "dynamics_frames": gt.shape[1],
            **reading, "dynamics_seconds": round(time.time() - started, 2)}


__all__ = ["CAP", "EPS", "MIN_POINTS", "dynamics_readings", "emd_step"]
