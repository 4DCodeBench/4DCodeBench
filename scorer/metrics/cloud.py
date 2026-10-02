"""Camera-visible surfaces for registration and reference-view geometry scoring."""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from ..raster import rasterize, rasterize_frame
from ..world import World

HELD = 8              # size of the `cloud` LRU cache
SCENE_POINTS = 65536  # points subsampled from a frame-0 cloud for Scene3D and registration
SCENE_CAP = 0.5       # per-point Chamfer distance cap, in reference RMS radii


def backproject(depth: np.ndarray, intrinsics: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Back-project every finite pixel of a depth map to a camera-frame point.

    Returns `(P, 3)` float32 points and the `(H, W)` coverage mask. Points are in
    row-major mask order, so `ids[mask]` aligns with them.
    """

    covered = np.isfinite(depth)
    rows, columns = np.nonzero(covered)
    z = depth[covered].astype(np.float64)
    rays = np.column_stack([columns + 0.5, rows + 0.5, np.ones(len(z))])
    directions = rays @ np.linalg.inv(np.asarray(intrinsics, dtype=np.float64)).T
    return (directions * z[:, None]).astype(np.float32), covered


def frame_cloud(world: World, frame: int) -> tuple[np.ndarray, np.ndarray]:
    """Return one frame's visible surface in the world's camera frame, and its coverage mask."""

    depth, _ = rasterize_frame(world, frame)
    points, covered = backproject(depth, world.intrinsics)
    if not len(points):
        raise ValueError(f"{world.root}: frame {frame} rasterises to no covered pixel")
    return points, covered


@lru_cache(maxsize=HELD)
def cloud(world: World, frame: int) -> np.ndarray:
    """Cached `frame_cloud` points, without the mask."""

    return frame_cloud(world, frame)[0]


def reference_cloud(gt: World, pred: World, alignment, frame: int = 0) -> np.ndarray:
    """Rasterise the aligned submission through the reference camera and back-project it."""

    vertices, faces = pred.mesh(frame)
    reference_pose, submitted_pose = gt.extrinsics, pred.extrinsics
    rotation = reference_pose[:3, :3] @ alignment.rotation @ submitted_pose[:3, :3].T
    offset = (reference_pose[:3, 3] + reference_pose[:3, :3] @ alignment.offset
              - alignment.scale * rotation @ submitted_pose[:3, 3])
    placed = alignment.scale * (np.asarray(vertices, np.float64) @ rotation.T) + offset
    width, height = gt.resolution
    depth, _ = rasterize(placed, faces, gt.intrinsics, reference_pose, height, width)
    return backproject(depth, gt.intrinsics)[0]


def extent(points: np.ndarray) -> tuple[np.ndarray, float]:
    """Return a cloud's centroid and RMS radius about it; raises on zero radius."""

    centre = np.asarray(points, dtype=np.float64).mean(axis=0)
    offset = np.asarray(points, dtype=np.float64) - centre
    radius = float(np.sqrt((offset**2).sum(axis=1).mean()))
    if radius <= 0:
        raise ValueError("a scene cloud of zero radius carries no unit of length")
    return centre, radius


def subsample(points: np.ndarray, budget: int, generator: np.random.Generator) -> np.ndarray:
    """Draw `budget` points without replacement, or return all points if there are fewer."""

    if len(points) <= budget:
        return points
    return points[generator.choice(len(points), budget, replace=False)]


__all__ = ["SCENE_POINTS", "backproject", "cloud", "extent", "frame_cloud", "subsample"]
