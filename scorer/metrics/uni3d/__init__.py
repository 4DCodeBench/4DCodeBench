"""Uni3D similarity: cosine between Uni3D embeddings of the submission and a reference cloud.

The submission cloud is its frame-0 visible surface (registered, as in Scene3D, on a
synthetic case). The reference is either

    point     the reference world's frame-0 cloud: `uni3d_point_scene`, synthetic cases
    moge      the MoGe-3 point map of `reference.mp4` frame 0, cached in `estimates/`:
              `uni3d_moge_scene`, real cases

Each cloud is centred, scaled by its `QUANTILE`th-percentile radius with points
beyond the unit sphere projected onto it, and given the constant colour
`UNI3D_COLOUR`. The raw cosine is returned; `scorer.score` maps it to `(1 + cos) / 2`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from config.estimates import (
    MOGE_FILENAME,
    MOGE_MASK_KEY,
    MOGE_POINTS_KEY,
)
from config.models import checkpoint as stored
from config.uni3d import UNI3D_ARGS, UNI3D_CHECKPOINT, UNI3D_COLOUR, UNI3D_POINTS

from ..distances import device, seed
from .uni3d import Uni3D, create_uni3d

QUANTILE = 99.0    # percentile of point radii used as the unit radius


def point_tower(checkpoints: str | Path | None = None) -> Uni3D:
    """Load the Uni3D point encoder from the model store."""

    return load(stored(UNI3D_CHECKPOINT, checkpoints))


def load(checkpoint: str | Path) -> Uni3D:
    """Build Uni3D, load a checkpoint's `module` state dict, and move it to the device."""

    model = create_uni3d(UNI3D_ARGS)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)["module"]
    if next(iter(state)).startswith("module."):
        state = {name[len("module."):]: tensor for name, tensor in state.items()}
    model.load_state_dict(state, strict=True)
    return model.to(device()).eval()


def normalise(points: np.ndarray) -> np.ndarray:
    """Centre a cloud, scale it by its `QUANTILE` radius, and project outliers onto the sphere."""

    points = np.asarray(points, dtype=np.float32)
    points = points[np.isfinite(points).all(axis=1)]
    points = points - points.mean(axis=0)
    radius = np.percentile(np.linalg.norm(points, axis=1), QUANTILE)
    points = points / max(float(radius), np.finfo(np.float32).tiny)
    return points / np.maximum(np.linalg.norm(points, axis=1, keepdims=True), 1.0)


@torch.no_grad()
def embed(model: Uni3D, points: np.ndarray) -> torch.Tensor:
    """Return one cloud's L2-normalised Uni3D feature, resampled to `UNI3D_POINTS` points."""

    xyz = normalise(points)
    generator = np.random.default_rng(seed())
    xyz = xyz[generator.choice(len(xyz), UNI3D_POINTS, replace=len(xyz) < UNI3D_POINTS)]
    cloud = np.concatenate([xyz, np.full_like(xyz, UNI3D_COLOUR)], axis=1)
    feature = model.encode_pc(torch.as_tensor(cloud, device=device())[None])
    return torch.nn.functional.normalize(feature, dim=-1)[0]


def cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    """Cosine of two unit features, clipped to [-1, 1] against rounding error."""

    return float(np.clip(float(left.float() @ right.float()), -1.0, 1.0))


def moge_cloud(estimates: str | Path) -> np.ndarray:
    """The MoGe point map's valid points as a `(P, 3)` camera-space cloud."""

    with np.load(Path(estimates) / MOGE_FILENAME) as archive:
        points = archive[MOGE_POINTS_KEY].reshape(-1, 3)
        mask = archive[MOGE_MASK_KEY].reshape(-1) if MOGE_MASK_KEY in archive.files else None
    held = np.isfinite(points).all(axis=1)
    if mask is not None:
        held &= mask.astype(bool)
    return points[held].astype(np.float32)


def cloud_readings(submission: np.ndarray, point: np.ndarray | None, moge: np.ndarray | None,
                   model: Uni3D) -> dict:
    """Uni3D readings of the submission cloud against each available reference cloud."""

    def feature(value: torch.Tensor) -> np.ndarray:
        return value.detach().float().cpu().numpy()

    scene = embed(model, submission) if len(submission) else None
    arrays, readings = {}, {}
    if scene is not None:
        arrays["submission_scene"] = feature(scene)
    if point is not None:
        readings["uni3d_point_scene"] = None
        if scene is not None and len(point):
            reference = embed(model, point)
            arrays["reference_point_scene"] = feature(reference)
            readings["uni3d_point_scene"] = cosine(reference, scene)
    if moge is not None and len(moge) and scene is not None:
        reference = embed(model, moge)
        arrays["reference_moge_scene"] = feature(reference)
        readings["uni3d_moge_scene"] = cosine(reference, scene)
    readings["arrays"] = {"uni3d": arrays}
    return readings


__all__ = ["cloud_readings", "cosine", "embed", "load", "moge_cloud", "normalise",
           "point_tower"]
