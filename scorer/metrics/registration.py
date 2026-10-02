"""Similarity registration of the submission onto the reference world.

Two candidates are fitted: trimmed ICP on the two cameras' frame-0 visible surfaces
(`camera_candidate`) and multi-start ICP on the dynamic objects' surfaces over 16
frames (`mesh_candidate`). `register` keeps the candidate whose transformed
submission, rasterised through the reference camera, has the lower capped Chamfer
distance to the reference's frame-0 visible surface.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np
import torch

from ..world import World
from .cloud import SCENE_CAP, SCENE_POINTS, backproject, cloud, extent, reference_cloud, subsample
from .distances import bounded_chamfer, device, nearest, seed
from ..raster import NEAR, rasterize_frame

TRIM = 0.2         # fraction of the costliest pairs dropped per direction in each ICP step
ITERATIONS = 40    # maximum ICP iterations
POINTS = 8192      # points per cloud in the ICP fits; nearest-neighbour cost is quadratic in it
TOLERANCE = 1e-7   # relative residual change at which ICP stops


def rotation_angle(rotation: np.ndarray) -> float:
    """Return the rotation angle of a 3x3 rotation matrix, in radians."""

    return float(np.arccos(np.clip((np.trace(np.asarray(rotation)) - 1.0) / 2.0, -1.0, 1.0)))


@dataclass(frozen=True)
class Registration:
    """A similarity from submission to reference camera coordinates, and the unit of length.

    `scale`, `rotation` and `offset` map submission camera-frame points to reference
    camera-frame metres. `centre` and `radius` are the centroid and RMS radius of the
    reference frame-0 cloud; `gt` and `pred` map points into reference radii about it.
    """

    centre: np.ndarray
    radius: float
    scale: float
    rotation: np.ndarray
    offset: np.ndarray
    residual: float
    residual_start: float     # reference-view error of the camera candidate
    selected: str = "camera"
    candidate_errors: dict[str, float] = field(default_factory=dict)

    def gt(self, points: np.ndarray) -> np.ndarray:
        return (np.asarray(points) - self.centre) / self.radius

    def pred(self, points: np.ndarray) -> np.ndarray:
        moved = self.scale * (np.asarray(points) @ self.rotation.T) + self.offset
        return (moved - self.centre) / self.radius



    def as_dict(self) -> dict:
        return {
            "scale": float(self.scale),
            "rotation": [[float(v) for v in row] for row in self.rotation],
            "rotation_deg": float(np.degrees(rotation_angle(self.rotation))),
            "offset": [float(v) for v in self.offset],
            "residual": float(self.residual),
            "residual_start": float(self.residual_start),
            "centre": [float(v) for v in self.centre],
            "radius": float(self.radius),
            "selected": self.selected,
            "candidate_errors": self.candidate_errors,
        }


def _tensor(points: np.ndarray) -> torch.Tensor:
    return torch.as_tensor(np.asarray(points, dtype=np.float32), device=device())


def _solve(source: torch.Tensor, target: torch.Tensor) -> tuple[float, torch.Tensor, torch.Tensor]:
    """Umeyama similarity `(scale, rotation, translation)` taking `source` onto `target`.

    The smallest singular direction is flipped when the rotation would be a reflection.
    """

    source, target = source.double(), target.double()
    source_mean, target_mean = source.mean(dim=0), target.mean(dim=0)
    p, q = source - source_mean, target - target_mean
    left, _, right = torch.linalg.svd(q.T @ p / len(p))
    correction = torch.eye(3, dtype=source.dtype, device=source.device)
    correction[2, 2] = torch.sign(torch.det(left @ right))
    rotation = left @ correction @ right
    scale = float((p @ rotation.T * q).sum() / (p * p).sum())
    translation = target_mean - scale * (rotation @ source_mean)
    return scale, rotation.float(), translation.float()


def _compose(
    step: tuple[float, torch.Tensor, torch.Tensor],
    current: tuple[float, torch.Tensor, torch.Tensor],
) -> tuple[float, torch.Tensor, torch.Tensor]:
    step_scale, step_rotation, step_translation = step
    scale, rotation, translation = current
    return (step_scale * scale, step_rotation @ rotation,
            step_scale * (step_rotation @ translation) + step_translation)


def _trim(sources: list[torch.Tensor], targets: list[torch.Tensor], costs: list[torch.Tensor]):
    """Drop the costliest `TRIM` fraction of pairs; return the kept pairs and their mean cost."""

    cost = torch.cat(costs)
    keep = cost <= torch.quantile(cost.float(), 1.0 - TRIM)
    return torch.cat(sources)[keep], torch.cat(targets)[keep], float(cost[keep].mean())


def _correspond(
    gt_points: torch.Tensor,
    pred_points: torch.Tensor,
    transform: tuple[float, torch.Tensor, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Nearest-neighbour pairs both ways, each direction trimmed on its own costs."""

    scale, rotation, translation = transform
    moved = scale * (pred_points @ rotation.T) + translation
    values, index = nearest(moved, gt_points)
    source, target, cost = _trim([moved], [gt_points[index]], [values])
    values, index = nearest(gt_points, moved)
    other_source, other_target, other_cost = _trim([moved[index]], [gt_points], [values])
    return (torch.cat([source, other_source]), torch.cat([target, other_target]),
            0.5 * (cost + other_cost))


def _fit(
    gt_points: torch.Tensor,
    pred_points: torch.Tensor,
    scale0: float,
) -> tuple[float, torch.Tensor, torch.Tensor, float, float]:
    """Trimmed ICP from a scaled identity.

    Returns scale, rotation, translation, the final residual and the initial residual.
    """

    device = gt_points.device
    transform = (float(scale0), torch.eye(3, device=device), torch.zeros(3, device=device))
    start = previous = float("inf")
    for iteration in range(ITERATIONS):
        source, target, residual = _correspond(gt_points, pred_points, transform)
        if iteration == 0:
            start = residual
        if abs(previous - residual) < TOLERANCE * max(residual, 1e-6):
            break
        previous = residual
        transform = _compose(_solve(source, target), transform)
    return (*transform, _correspond(gt_points, pred_points, transform)[2], start)


def _reach(points: torch.Tensor) -> float:
    """Return a cloud's RMS distance from the camera origin; sets the initial scale."""

    return float((points**2).sum(dim=1).mean().sqrt())


def camera_candidate(gt: World, pred: World, centre, radius) -> Registration | None:
    """Trimmed ICP between the two frame-0 visible surfaces in camera coordinates.

    Returns None when the submission shows fewer than 3 points.
    """

    depth, _ = rasterize_frame(pred, 0)
    visible, _ = backproject(depth, pred.intrinsics)
    if len(visible) < 3:
        return None
    gt_generator, pred_generator = (np.random.default_rng(seed()) for _ in range(2))
    gt_points = _tensor(subsample(cloud(gt, 0), POINTS, gt_generator) / radius)
    pred_points = _tensor(subsample(visible, POINTS, pred_generator) / radius)
    scale0 = _reach(gt_points) / _reach(pred_points)

    scale, rotation, translation, residual, start = _fit(gt_points, pred_points, scale0)
    return Registration(
        centre=np.asarray(centre, dtype=np.float64),
        radius=float(radius),
        scale=float(scale),
        rotation=rotation.cpu().numpy().astype(np.float64),
        offset=translation.cpu().numpy().astype(np.float64) * radius,
        residual=float(residual),
        residual_start=float(start),
    )


def surface(world: World, frame: int) -> np.ndarray | None:
    """Sample `POINTS` points area-uniformly on one frame's dynamic surfaces; None if no area."""

    vertices, faces, count = [], [], 0
    for oid in sorted(world.dynamic_ids, key=lambda value: str(int(value))):
        v, f = world.object_mesh(int(oid), frame)
        if not len(f):
            continue
        vertices.append(v)
        faces.append(f + count)
        count += len(v)
    if not faces:
        return None
    triangles = np.concatenate(vertices)[np.concatenate(faces)].astype(np.float64)
    area = np.linalg.norm(np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]), axis=1)
    if area.sum() == 0:
        return None
    generator = np.random.default_rng([seed(), frame])
    triangle = triangles[generator.choice(len(triangles), POINTS, p=area / area.sum())]
    u, v = generator.random((2, POINTS))
    u = np.sqrt(u)
    return (triangle[:, 0] * (1 - u[:, None]) + triangle[:, 1] * (u * (1-v))[:, None]
            + triangle[:, 2] * (u*v)[:, None]).astype(np.float32)


def mesh_candidate(gt: World, pred: World, centre, radius) -> Registration | None:
    """Multi-start ICP on dynamic surfaces at 16 frames, reference samples in view only.

    Returns None when no frame has samples on both sides.
    """

    from .multistart import fit

    left, right = [], []
    width, height = gt.resolution
    for frame in np.linspace(0, len(gt) - 1, 16).astype(int):
        x, y = surface(gt, int(frame)), surface(pred, int(frame))
        if x is None or y is None:
            continue
        camera = gt.camera(x)
        projected = camera @ gt.intrinsics.T
        with np.errstate(divide="ignore", invalid="ignore"):
            pixel = projected[:, :2] / projected[:, 2:3]
        visible = ((camera[:, 2] > NEAR) & (pixel[:, 0] >= 0) & (pixel[:, 0] < width)
                   & (pixel[:, 1] >= 0) & (pixel[:, 1] < height))
        held = x[visible]
        if not len(held):
            continue
        left.append(held[np.arange(POINTS) * len(held) // POINTS])
        right.append(y)
    if not left:
        return None
    transform = fit(np.stack(left), np.stack(right), radius)
    eg, ep = gt.extrinsics, pred.extrinsics
    scale = transform["scale"]
    rotation, offset = np.asarray(transform["rotation"]), np.asarray(transform["offset"])
    return Registration(centre, radius, scale, eg[:3, :3].T @ rotation @ ep[:3, :3],
                        eg[:3, :3].T @ (scale * rotation @ ep[:3, 3] + offset - eg[:3, 3]),
                        0., 0., selected="mesh")


@torch.inference_mode()
def register(gt: World, pred: World) -> Registration:
    """Fit both candidates and return the one with the lower reference-view Scene3D error.

    `residual` is the chosen candidate's error, `residual_start` the camera
    candidate's (`SCENE_CAP` when it is missing).
    """

    centre, radius = extent(cloud(gt, 0))
    x = (subsample(cloud(gt, 0), SCENE_POINTS, np.random.default_rng(seed())) - centre) / radius
    candidates = [camera_candidate(gt, pred, centre, radius), mesh_candidate(gt, pred, centre, radius)]
    candidates = [candidate for candidate in candidates if candidate is not None]
    if not candidates:
        raise ValueError("no visible-camera or dynamic-mesh registration candidate")
    errors = {}
    for candidate in candidates:
        points = reference_cloud(gt, pred, candidate)
        y = candidate.gt(subsample(points, SCENE_POINTS, np.random.default_rng(seed())))
        errors[candidate.selected] = bounded_chamfer(x, y, SCENE_CAP)
    chosen = min(candidates, key=lambda candidate: errors[candidate.selected])
    return replace(chosen, residual=errors[chosen.selected],
                   residual_start=errors.get("camera", SCENE_CAP), candidate_errors=errors)
