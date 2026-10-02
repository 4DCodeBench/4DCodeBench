"""Z-buffer rasterisation of a world's per-frame geometry, on nvdiffrast.

Camera convention per `docs/spec.md`: `extrinsics` is camera-to-world, axes
+x right +y down +z forward, a world point projects proportionally to
`K R^T (p - t)`. `id_frames` rasterises every object present at a frame into
one z-buffer and labels each pixel with the id of its nearest surface. Requires CUDA.
"""

from __future__ import annotations

from functools import cache
from pathlib import Path

import numpy as np
import nvdiffrast.torch as dr
import torch

from config.world import (
    CAMERA_DTYPE,
    INDEX_DTYPE,
)

from .world import World

NEAR = 1e-3  # near clip plane, camera-space metres
MISS = -1    # face index of a pixel no triangle covers


def default_device() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("rasterisation needs a CUDA device")
    return torch.device("cuda")


@cache
def context(device: torch.device) -> dr.RasterizeCudaContext:
    return dr.RasterizeCudaContext(device=device)


def camera_points(verts: torch.Tensor, extrinsic: torch.Tensor) -> torch.Tensor:
    """`R^T (p - t)` for every world vertex."""

    return (verts - extrinsic[:3, 3]) @ extrinsic[:3, :3]


def clip_points(points: torch.Tensor, intrinsics: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """Camera-frame points as the `(x, y, z, w)` clip coordinates nvdiffrast draws from."""

    projected = points @ intrinsics.T
    z = points[:, 2:3]
    x = 2.0 * projected[:, 0:1] / width - z
    y = 2.0 * projected[:, 1:2] / height - z
    return torch.cat([x, y, z - 2.0 * NEAR, z], dim=1)


def rasterize(
    verts: np.ndarray,
    faces: np.ndarray,
    intrinsics: np.ndarray,
    extrinsic: np.ndarray,
    height: int,
    width: int,
    device: torch.device | str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """One frame's `(depth, face)`: camera z (`inf` on a miss) and winning face index (`-1`)."""

    device = torch.device(device) if device is not None else default_device()
    if not len(faces) or not len(verts):
        return (np.full((height, width), np.inf, dtype=np.float32),
                np.full((height, width), MISS, dtype=np.int32))

    vertices = torch.as_tensor(np.asarray(verts, dtype=np.float32), device=device)
    triangles = torch.as_tensor(np.asarray(faces, dtype=np.int32), device=device)
    matrix = torch.as_tensor(np.asarray(intrinsics, dtype=np.float32), device=device)
    pose = torch.as_tensor(np.asarray(extrinsic, dtype=np.float32), device=device)

    points = camera_points(vertices, pose)
    clip = clip_points(points, matrix, height, width)
    rast, _ = dr.rasterize(context(device), clip[None], triangles, (height, width))
    hit = rast[0, :, :, 3] > 0
    z, _ = dr.interpolate(points[None, :, 2:3].contiguous(), rast, triangles)
    depth = torch.where(hit, z[0, :, :, 0], torch.full_like(z[0, :, :, 0], float("inf")))
    face = torch.where(hit, rast[0, :, :, 3].int() - 1, torch.full_like(hit, MISS, dtype=torch.int32))
    return depth.cpu().numpy(), face.cpu().numpy()


def id_frame(world, frame: int, device=None) -> np.ndarray:
    """One frame's per-pixel object ids, 0 where no surface covers the pixel."""

    height, width = world.shape[1], world.shape[2]
    parts = world.parts(frame)
    if not parts:
        return np.zeros((height, width), dtype=INDEX_DTYPE)
    verts, faces, owner, base = [], [], [], 0
    for oid, v, f in parts:
        verts.append(v)
        faces.append(f + base)
        owner.append(np.full(len(f), oid, dtype=INDEX_DTYPE))
        base += len(v)
    owner = np.concatenate([np.zeros(1, dtype=INDEX_DTYPE), np.concatenate(owner)])
    _, face = rasterize(np.concatenate(verts), np.concatenate(faces),
                        world.intrinsics, world.extrinsics, height, width, device)
    return owner[face + 1]


def id_frames(world, device=None) -> np.ndarray:
    """The whole take's per-pixel object ids, `(F, H, W)` uint16."""

    return np.stack([id_frame(world, frame, device) for frame in range(len(world))])


class Geometry:
    """A world directory as a camera and one triangle soup per frame, at a fixed resolution."""

    def __init__(self, root: str | Path, height: int, width: int):
        self.root = Path(root)
        self.height, self.width = int(height), int(width)
        self._world = World(root)
        self.intrinsics = np.asarray(self._world.intrinsics, dtype=CAMERA_DTYPE)
        self.extrinsics = np.asarray(self._world.extrinsics, dtype=CAMERA_DTYPE)

    def __repr__(self) -> str:
        return f"Geometry({self.root}, F={len(self)}, {self.width}x{self.height})"

    def __len__(self) -> int:
        return len(self._world)

    def mesh(self, frame: int) -> tuple[np.ndarray, np.ndarray]:
        return self._world.mesh(frame)

    def depth(self, frame: int, device=None) -> tuple[np.ndarray, np.ndarray]:
        verts, faces = self.mesh(frame)
        return rasterize(verts, faces, self.intrinsics, self.extrinsics,
                         self.height, self.width, device)


def rasterize_frame(world: World, frame: int, device=None) -> tuple[np.ndarray, np.ndarray]:
    """`rasterize` of one frame of a `World`, at its own render resolution."""

    width, height = world.resolution
    verts, faces = world.mesh(frame)
    return rasterize(verts, faces, world.intrinsics, world.extrinsics, height, width, device)


__all__ = ["MISS", "NEAR", "Geometry", "camera_points", "clip_points", "default_device",
           "id_frame", "id_frames", "rasterize", "rasterize_frame"]
