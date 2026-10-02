"""Interpenetration: one minus the mean fraction of object vertices buried in other objects.

A vertex is buried in a solid object `B` (nonzero volume) of another id if it lies
in `B`'s bounding box grown by the tolerance, is farther than the tolerance from
`B`'s surface, and has |winding number| above `INSIDE` with respect to `B`. The
tolerance is `PENETRATION` times the smaller of the two objects' bounding-box
diagonals. A frame's value is the mean buried fraction over its objects, and the score
is one minus the mean over frames that contain an object.
"""

from __future__ import annotations

import os

import igl
import numpy as np
import trimesh
from scipy.spatial import cKDTree

from ..world import World
from .frames import WORKERS, judge_frames

PENETRATION = 0.03   # penetration tolerance, as a fraction of a bounding-box diagonal
INSIDE = 0.5         # |winding number| above which a point is inside


class Container:
    """One object's mesh with its bounds, winding-number containment and surface distance."""

    __slots__ = ("faces", "high", "low", "span", "verts")

    def __init__(self, mesh: trimesh.Trimesh):
        self.low, self.high = mesh.bounds
        self.span = float(np.linalg.norm(self.high - self.low))
        self.verts = np.asarray(mesh.vertices, dtype=np.float64)
        self.faces = np.asarray(mesh.faces, dtype=np.int64)

    def inside(self, points: np.ndarray) -> np.ndarray:
        with np.errstate(divide="ignore", invalid="ignore"):
            winding = igl.fast_winding_number(self.verts, self.faces, np.asarray(points, dtype=np.float64))
        return np.isfinite(winding) & (np.abs(winding) > INSIDE)

    def depth(self, points: np.ndarray) -> np.ndarray:
        """Unsigned distance from each point to the mesh surface (libigl AABB tree)."""

        squared, _, _ = igl.point_mesh_squared_distance(np.asarray(points, dtype=np.float64), self.verts, self.faces)
        return np.sqrt(squared)


def is_solid(mesh: trimesh.Trimesh) -> bool:
    with np.errstate(divide="ignore", invalid="ignore"):
        volume = mesh.volume
    return bool(np.isfinite(volume) and abs(volume) > 0)


def deep_candidates(mesh: trimesh.Trimesh, points: np.ndarray, tolerance: np.ndarray) -> np.ndarray:
    """Mask of points inside some connected component's bounding box shrunk by their tolerance.

    A point buried deeper than its tolerance in a closed component satisfies this, so
    the mask prefilters candidates.
    """

    labels = trimesh.graph.connected_component_labels(mesh.face_adjacency, node_count=len(mesh.faces))
    low = np.full((labels.max() + 1, 3), np.inf)
    high = np.full_like(low, -np.inf)
    np.minimum.at(low, labels, mesh.triangles.min(axis=1))
    np.maximum.at(high, labels, mesh.triangles.max(axis=1))

    half = (high - low) * 0.5 - tolerance.min()
    active = np.all(half >= 0, axis=1)
    low, high, half = low[active], high[active], half[active]
    selected = np.zeros(len(points), dtype=bool)
    if not len(low):
        return selected

    tree = cKDTree(points)
    nearby = tree.query_ball_point((low + high) * 0.5, np.linalg.norm(half, axis=1))
    for lo, hi, found in zip(low, high, nearby):
        if not found:
            continue
        found = np.asarray(found, dtype=np.int64)
        margin = tolerance[found, None]
        selected[found] |= np.all((points[found] >= lo + margin)
                                 & (points[found] <= hi - margin), axis=1)
    return selected


def buried_fractions(parts: list[trimesh.Trimesh]) -> list[float]:
    """Return the fraction of each object's finite vertices buried in another object."""

    probes, owner, spans = [], [], []
    for index, part in enumerate(parts):
        points = np.asarray(part.vertices, dtype=np.float64).reshape(-1, 3)
        points = points[np.isfinite(points).all(axis=1)]
        probes.append(np.asarray(points, dtype=np.float64))
        owner.append(np.full(len(points), index, dtype=np.int64))
        spans.append(float(np.linalg.norm(part.bounds[1] - part.bounds[0])) if len(part.faces) else 0.0)
    counts = np.asarray([len(item) for item in probes], dtype=np.int64)
    if not counts.sum():
        return [0.0] * len(parts)

    points = np.concatenate(probes).reshape(-1, 3)
    owner = np.concatenate(owner)
    span = np.asarray(spans, dtype=np.float64)[owner]

    buried = np.zeros(len(points), dtype=bool)
    tree = cKDTree(points)
    for index, part in enumerate(parts):
        if not is_solid(part):
            continue
        container = Container(part)
        # ball about the box centre containing every point the box test below can pass
        reach = 0.5 * container.span + PENETRATION * container.span
        near = np.asarray(tree.query_ball_point(0.5 * (container.low + container.high), reach), dtype=np.int64)
        near = near[~buried[near] & (owner[near] != index)]
        if not len(near):
            continue
        tolerance = PENETRATION * np.minimum(span[near], container.span)
        keep = np.all((points[near] >= container.low - tolerance[:, None])
                      & (points[near] <= container.high + tolerance[:, None]), axis=1)
        candidate, tolerance = near[keep], tolerance[keep]
        if not len(candidate):
            continue
        if part.is_watertight and part.is_winding_consistent:
            keep = deep_candidates(part, points[candidate], tolerance)
            candidate, tolerance = candidate[keep], tolerance[keep]
            if not len(candidate):
                continue
        # Distance to a surface vertex bounds distance to the surface from above.
        vertex_tree = cKDTree(part.vertices[part.referenced_vertices])
        distance, _ = vertex_tree.query(points[candidate], distance_upper_bound=float(tolerance.max()))
        keep = distance > tolerance
        candidate, tolerance = candidate[keep], tolerance[keep]
        if not len(candidate):
            continue
        deep = candidate[container.depth(points[candidate]) > tolerance]
        if not len(deep):
            continue
        buried[deep] = container.inside(points[deep])

    totals = np.bincount(owner[buried], minlength=len(parts))
    return [float(totals[i] / counts[i]) if counts[i] else 0.0 for i in range(len(parts))]


def judge(world: World, frame: int) -> list[float]:
    parts = [trimesh.Trimesh(vertices=verts, faces=faces, process=False)
             for _, verts, faces in world.parts(frame)]
    return buried_fractions(parts)


def interpenetration(world: World) -> dict:
    """Interpenetration readings of one world, omitting the score when no frame has objects."""

    frames = list(range(len(world)))
    workers = int(os.environ.get("OMP_NUM_THREADS", WORKERS))
    series = judge_frames(judge, world, frames, workers=workers)

    per_frame = [float(np.mean(values)) for values in series if values]
    components = int(sum(len(values) for values in series))
    # per-object fractions concatenated over frames; `components_per_frame` splits them
    kept = {
        "arrays": {"interpenetration": {
            "buried": np.concatenate([np.asarray(values, dtype=np.float32) for values in series])
                      if components else np.zeros(0, dtype=np.float32),
            "components_per_frame": np.asarray([len(values) for values in series], dtype=np.int32),
        }},
    }
    if not per_frame:
        return {
            **kept,
            "interpenetration_components": 0,
            "interpenetration_frames": len(frames),
            "interpenetration_judged_frames": 0,
        }
    return {
        **kept,
        "interpenetration": float(1.0 - float(np.mean(per_frame))),
        "interpenetration_components": components,
        "interpenetration_frames": len(frames),
        "interpenetration_judged_frames": len(per_frame),
    }
