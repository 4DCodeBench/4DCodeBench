"""Trajectory DTW: DTW distance between Hungarian-matched 3D material paths.

Each side's material is sampled with the shared voxel edge `h` on the frame each
column of `pos` becomes finite, and each sample is followed while it stays finite, in
reference radii, NaN outside the reference camera's view. Both sides draw the
same number of paths, at most `ROWS`. Pairwise costs are per-step DTW distances over
each path's finite frames (Sakoe & Chiba 1978; Berndt & Clifford 1994), capped at
`CAP`, and the paths are matched one to one by the Hungarian method, as in
multi-object tracking (Bernardin & Stiefelhagen 2008):

    trajectory_dtw_score = max(0, 1 - mean_i min(DTW(tau_i, tau'_pi(i)), CAP) / CAP).

An unmatched reference path costs `CAP`.
"""

from __future__ import annotations

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from ..samples import Samples, Sampling, frame_samples, visible
from ..world import World
from .distances import device, seed
from .dtw import dtw_matrix
from .registration import Registration

ROWS = 8192          # maximum paths drawn per side
CAP = 1.0            # per-pair DTW cost cap, in reference radii


def _births(positions: list[np.ndarray]) -> list[np.ndarray]:
    """Per material, the first frame each column is finite on; -1 if it never is."""

    out = []
    for pos in positions:
        alive = np.isfinite(pos).all(axis=2)
        out.append(np.where(alive.any(axis=0), alive.argmax(axis=0), -1))
    return out


def _paths(world: World, positions: list[np.ndarray], h: float, alignment: Registration,
           gt: World, generator: np.random.Generator) -> np.ndarray:
    """One side's samples over the take, `(S, F, 3)`, NaN off a sample's life or out of view."""

    frames = len(world)
    births = _births(positions)
    tracks = []
    for frame in range(frames):
        newborn = [born == frame for born in births]
        if not any(mask.any() for mask in newborn):
            continue
        # sample only material born on this frame, as a one-frame take
        masked = []
        for pos, mask in zip(positions, newborn, strict=True):
            here = pos[frame : frame + 1].copy()
            here[:, ~mask] = np.nan
            masked.append(here)
        samples = frame_samples(world, masked, h, 0, generator)
        if not len(samples):
            continue
        tracks.append(Samples(positions, samples.blocks).track(0, frames))
    if not tracks:
        return np.zeros((0, frames, 3), np.float32)
    track = np.concatenate(tracks, axis=1)                       # (F, S, 3)
    seen = visible(track.reshape(-1, 3), alignment, gt).reshape(track.shape[:2])
    track[~seen] = np.nan
    paths = np.ascontiguousarray(np.moveaxis(track, 0, 1))        # (S, F, 3)
    return paths[np.isfinite(paths).all(axis=2).any(axis=1)]


def draw(paths: np.ndarray, rows: int, generator: np.random.Generator) -> np.ndarray:
    if len(paths) <= rows:
        return paths
    return paths[np.sort(generator.choice(len(paths), rows, replace=False))]


def dtw_costs(a: np.ndarray, b: np.ndarray, cap: float = CAP) -> np.ndarray:
    """Per-step DTW distance of every reference path to every submission path, capped."""

    return np.minimum(dtw_matrix(a, b, device()), cap)


def _condition_costs(cost: np.ndarray, dev: torch.device) -> np.ndarray:
    """Reduce a square cost matrix by Sinkhorn dual potentials, computed on the GPU.

    The reduced matrix has the same optimal assignment and is solved faster by
    `linear_sum_assignment`.
    """

    values = torch.as_tensor(cost, dtype=torch.float32, device=dev)
    right = torch.zeros(len(cost), dtype=values.dtype, device=dev)
    for temperature in (0.01, 0.001):
        for _ in range(256):
            left = -temperature * torch.logsumexp((right[None] - values) / temperature, dim=1)
            right = -temperature * torch.logsumexp((left[:, None] - values) / temperature, dim=0)
    # Every square matching uses each row and column once; these shifts preserve its optimum.
    reduced = cost.astype(np.float64) - right.cpu().numpy()[None]
    return reduced - reduced.min(axis=1, keepdims=True)


def match(cost: np.ndarray, cap: float = CAP) -> tuple[np.ndarray, np.ndarray]:
    """Each reference path's matched cost and partner index; `cap` and -1 when unmatched."""

    n, m = cost.shape
    matched = np.full(n, cap)
    partner = np.full(n, -1, np.int64)
    if n and m:
        dev = device()
        conditioned = (_condition_costs(cost, dev)
                       if n == m and n >= 1024 and dev.type == "cuda" else cost)
        rows, cols = linear_sum_assignment(conditioned)
        matched[rows] = cost[rows, cols]
        partner[rows] = cols
    return matched, partner


def trajectory_paths(gt: World, pred: World, alignment: Registration, sampling: Sampling) -> dict:
    """Sample both sides' paths and draw equal counts.

    Returns `gt`, `pred`, `h` and `counts`, or only `h` when there is nothing to match.
    """
    h = sampling.h
    if h is None:
        return {"h": np.float64(np.nan)}
    sides = [_paths(world, positions, h, alignment, gt, np.random.default_rng(seed()))
             for world, positions in ((gt, sampling.gt), (pred, sampling.pred))]
    counts = [len(paths) for paths in sides]
    if not counts[0]:
        return {"h": np.float64(h)}
    rows = min(ROWS, *(count for count in counts if count))
    drawn = [draw(paths, rows, np.random.default_rng([seed(), 1])) for paths in sides]
    return {"gt": drawn[0], "pred": drawn[1], "h": np.float64(h), "counts": np.asarray(counts)}


def trajectory_reading(paths: dict) -> dict:
    """Trajectory DTW readings from the output of `trajectory_paths`."""
    h = None if np.isnan(paths["h"]) else float(paths["h"])
    if "gt" not in paths:
        return {"arrays": {"trajectory": {}}, "trajectory_h": h, "trajectory_dtw_error": None}
    drawn, counts = [paths["gt"], paths["pred"]], paths["counts"].tolist()
    if counts[1]:
        matched, partner = match(dtw_costs(*drawn))
    else:
        matched, partner = np.full(len(drawn[0]), CAP), np.full(len(drawn[0]), -1, np.int64)
    error = float(matched.mean() / CAP)
    arrays = {"gt": drawn[0], "pred": drawn[1], "h": np.float32(h), "cost_dtw": matched.astype(np.float32),
              "match_dtw": partner, "dtw_error": np.float32(error)}
    return {"arrays": {"trajectory": arrays}, "trajectory_h": h, "trajectory_paths": counts,
            "trajectory_rows": [len(paths) for paths in drawn],
            "trajectory_dtw_error": error, "trajectory_dtw_score": float(max(0.0, 1.0 - error))}


__all__ = ["CAP", "ROWS", "draw", "dtw_costs", "match", "trajectory_paths", "trajectory_reading"]
