"""Analytic optical flow of a world.

The flow of frame `t` is the screen-space displacement, `t -> t + k`, of the
surface point rasterised at each pixel at `t`, as Blender's Vector pass computes
it. `k` is the case's evaluation stride (`config.sampling`). The point's
position at `t + k` comes from one of two paths:

  exact    the object's mesh has the same vertices and faces on both frames: the
           pixel's face and barycentrics at `t` are evaluated on the `t + k`
           vertices (rigid bodies, skinned and constant-topology soft bodies);
  material the topology changes (a remeshed liquid, a fracture, granular matter):
           the point takes the Gaussian-weighted mean displacement of the
           object's material columns (`dynamics/<name>.npz` `pos`) within `REACH`
           typical surface-to-column distances.

A static object (no material lists it) has zero flow. A pixel is invalid when no
surface covers it, when its object is absent at `t + k`, or when no live column
is within reach of a material-path point. The camera is fixed for the take.
"""

from __future__ import annotations


import numpy as np
import torch

from ..world import World

SIGMA = 1.0      # the kernel's sigma, in typical surface-to-column distances
REACH = 3.0      # the kernel's support, in the same unit


def same_topology(a: tuple[np.ndarray, np.ndarray], b: tuple[np.ndarray, np.ndarray]) -> bool:
    """Whether two `(verts, faces)` meshes have equal vertex counts and identical faces."""

    return (a[0].shape == b[0].shape and a[1].shape == b[1].shape
            and np.array_equal(a[1], b[1]))


def _columns(world: World, oid: int, frame: int, stride: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """The object's material columns alive on both `frame` and `frame + stride`: `(now, next)`."""

    now, nxt = [], []
    for material in world.materials:
        if oid not in material.ids:
            continue
        a, b = material.pos[frame], material.pos[frame + stride]
        alive = np.isfinite(a).all(axis=1) & np.isfinite(b).all(axis=1)
        now.append(a[alive])
        nxt.append(b[alive])
    if not now:
        return np.zeros((0, 3), np.float32), np.zeros((0, 3), np.float32)
    return np.concatenate(now), np.concatenate(nxt)


BLOCK = 2 ** 27   # entries of one distance block on the GPU (512 MB of float32)
CELL = 0.25       # the lookup voxel, as a fraction of the typical surface-to-column distance


def _nearest(points: torch.Tensor, now: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    """The `k` nearest columns of every point, brute force on the GPU in blocks."""

    distance = torch.empty((len(points), k), device=points.device)
    index = torch.empty((len(points), k), dtype=torch.long, device=points.device)
    chunk = max(1024, BLOCK // len(now))
    for lo in range(0, len(points), chunk):
        block = torch.cdist(points[lo:lo + chunk], now)
        distance[lo:lo + chunk], index[lo:lo + chunk] = block.topk(k, dim=1, largest=False)
    return distance, index


def _borrow(points: torch.Tensor, now: torch.Tensor, nxt: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Each point moved by the Gaussian-weighted mean displacement of the columns around it.

    `typical` is the median distance from a probe of the points to their nearest
    column. The kernel has sigma `SIGMA * typical` and support `REACH * typical`.
    Points are quantised to voxels of edge `CELL * typical` and each voxel's mean
    point is moved once. Returns `(moved, reached)`; `reached` is False where no
    column lies within the support.
    """

    if len(now) < 2 or not len(points):
        return points, torch.zeros(len(points), dtype=torch.bool, device=points.device)
    probe = points[torch.randint(len(points), (min(len(points), 4096),), device=points.device)]
    typical = _nearest(probe, now, 1)[0].median().clamp(min=1e-9)
    cell = CELL * typical
    _, inverse = torch.unique(torch.floor(points / cell).to(torch.int64), dim=0, return_inverse=True)
    voxels = int(inverse.max()) + 1
    centre = torch.zeros((voxels, 3), device=points.device).index_add_(0, inverse, points)
    count = torch.zeros(voxels, device=points.device).index_add_(0, inverse, torch.ones_like(inverse, dtype=torch.float32))
    centre /= count[:, None]

    shift = nxt - now
    delta = torch.empty_like(centre)
    total = torch.empty(voxels, device=points.device)
    sigma, reach = SIGMA * typical, REACH * typical
    chunk = max(1024, BLOCK // len(now))
    for lo in range(0, voxels, chunk):
        distance = torch.cdist(centre[lo:lo + chunk], now)
        weight = torch.exp(-0.5 * (distance / sigma) ** 2) * (distance <= reach)
        total[lo:lo + chunk] = weight.sum(dim=1)
        delta[lo:lo + chunk] = (weight @ shift) / total[lo:lo + chunk].clamp(min=1e-12)[:, None]
    return points + delta[inverse], (total > 0)[inverse]


def frame_flow(world: World, frame: int, device: torch.device | str | None = None,
               stride: int = 1) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """One frame's `(flow, valid, detail)`: `(H, W, 2)` pixels `(u, v)`, `(H, W)` bool, on `device`.

    The displacement is `frame -> frame + stride`, computed in one step on both
    paths, not composed frame by frame. `detail` lists the object ids on each
    path (`exact`, `material`, `static`, `gone`) and counts the `unreached` pixels.
    """

    import nvdiffrast.torch as dr

    from ..raster import camera_points, clip_points, context, default_device

    device = torch.device(device) if device is not None else default_device()
    height, width = world.shape[1], world.shape[2]
    detail = {"exact": [], "material": [], "static": [], "gone": [], "unreached": 0}
    stride = max(1, int(stride))
    parts = world.parts(frame) if frame + stride < len(world) else []
    if not parts:
        return (torch.zeros((height, width, 2), device=device),
                torch.zeros((height, width), dtype=torch.bool, device=device), detail)
    after = {oid: world.object_mesh(oid, frame + stride) for oid, _, _ in parts}
    dynamic = None

    verts, nexts, faces, owner, base = [], [], [], [], 0
    for oid, v, f in parts:
        verts.append(v)
        faces.append(f + base)
        owner.append(np.full(len(f), oid, np.int32))
        base += len(v)
        exact = len(after[oid][1]) > 0 and same_topology((v, f), after[oid])
        nexts.append(after[oid][0] if exact else v)
        if exact:
            detail["exact"].append(oid)
        else:
            if dynamic is None:
                dynamic = {int(value) for value in world.dynamic_ids}
            if oid not in dynamic:
                detail["static"].append(oid)
            elif len(after[oid][1]) == 0:
                detail["gone"].append(oid)
            else:
                detail["material"].append(oid)
    owner = torch.as_tensor(np.concatenate([np.full(1, -1, np.int32), *owner]), device=device)
    verts, nexts, faces = np.concatenate(verts), np.concatenate(nexts), np.concatenate(faces)

    pose = torch.as_tensor(world.extrinsics, dtype=torch.float32, device=device)
    matrix = torch.as_tensor(world.intrinsics, dtype=torch.float32, device=device)
    tri = torch.as_tensor(faces.astype(np.int32), device=device)
    p_now = camera_points(torch.as_tensor(verts, dtype=torch.float32, device=device), pose)
    p_next = camera_points(torch.as_tensor(nexts, dtype=torch.float32, device=device), pose)
    with torch.no_grad():
        rast, _ = dr.rasterize(context(device), clip_points(p_now, matrix, height, width)[None],
                               tri, (height, width))
        now = dr.interpolate(p_now[None].contiguous(), rast, tri)[0][0]       # (H, W, 3) camera frame
        nxt = dr.interpolate(p_next[None].contiguous(), rast, tri)[0][0]
    pixel_owner = owner[rast[0, :, :, 3].long()]                           # -1 on a miss
    ok = pixel_owner >= 0

    rotation, translation = pose[:3, :3], pose[:3, 3]
    for oid in detail["material"]:
        where = pixel_owner == oid
        if not where.any():
            continue
        columns = _columns(world, oid, frame, stride)
        col_now = torch.as_tensor(columns[0], dtype=torch.float32, device=device)
        col_next = torch.as_tensor(columns[1], dtype=torch.float32, device=device)
        moved, reached = _borrow(now[where] @ rotation.T + translation, col_now, col_next)
        nxt[where] = (moved - translation) @ rotation
        detail["unreached"] += int((~reached).sum())
        ok[where] = reached
    for oid in detail["gone"]:
        ok &= pixel_owner != oid

    def project(points):
        z = points[..., 2]
        u = (matrix[0, 0] * points[..., 0] + matrix[0, 1] * points[..., 1]) / z + matrix[0, 2]
        v = (matrix[1, 1] * points[..., 1]) / z + matrix[1, 2]
        return u, v, z

    u0, v0, z0 = project(now)
    u1, v1, z1 = project(nxt)
    ok &= torch.isfinite(u1) & torch.isfinite(v1) & (z0 > 0) & (z1 > 0)
    flow = torch.stack([u1 - u0, v1 - v0], dim=-1)
    flow[~ok] = 0.0
    return flow, ok, detail


__all__ = ["BLOCK", "CELL", "REACH", "SIGMA", "frame_flow", "same_topology"]
