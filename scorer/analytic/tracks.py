"""Analytic point tracks of a world, and the query grid shared by both sides of Track2D.

A query is a pixel and a birth frame. The birth frame is rasterised; the face and
barycentrics at the query's pixel fix a surface point, which one of the two
paths of `scorer.analytic.flow` carries onward:

  exact    the object's mesh keeps its birth-frame vertices and faces: the point is
           the same face's barycentric interpolation on each later frame, and the
           path ends at the first frame the topology changes;
  material the topology changes at once (a remeshed liquid, a fracture): the point
           takes the displacement of the nearest of the object's material columns
           (`dynamics/<name>.npz` `pos`) at birth, starting on the rasterised
           surface. A point with no column within `REACH` typical
           surface-to-column distances gets no path.

Paths are projected through the world's camera. A point is visible on a frame
when it is inside the image, in front of the camera, and its depth matches the
rasterised depth at its pixel within `TOLERANCE` of its depth or `FLOOR` metres,
whichever is larger. Points are carried through every world frame; only the
frames of the sampled timeline (`config.sampling`) are projected and returned.
"""

from __future__ import annotations

import numpy as np
import nvdiffrast.torch as dr
import torch

from config.sampling import sampled_indices

from ..raster import (
    NEAR,
    camera_points,
    clip_points,
    context,
    default_device,
    rasterize,
)
from ..world import World
from .flow import BLOCK, same_topology

REACH = 3.0        # attachment radius, in typical surface-to-column distances
PROBE = 4096       # surface points sampled to compute the typical distance
TOLERANCE = 0.01   # visibility depth tolerance, as a fraction of the point's depth
FLOOR = 0.01       # minimum visibility depth tolerance, in metres


def _soup(world: World, frame: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    """One frame's geometry as one triangle soup, with each face's object and own index."""

    verts, faces, owner, local, base = [], [], [], [], 0
    for oid, v, f in world.parts(frame):
        verts.append(v)
        faces.append(f + base)
        owner.append(np.full(len(f), oid, np.int64))
        local.append(np.arange(len(f), dtype=np.int64))
        base += len(v)
    if not verts:
        return None
    return (np.concatenate(verts), np.concatenate(faces),
            np.concatenate(owner), np.concatenate(local))


def _surface(world: World, verts: np.ndarray, faces: np.ndarray,
             device: torch.device) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The frame's per-pixel face (-1 on a miss), barycentrics and world position."""

    height, width = world.shape[1], world.shape[2]
    vertices = torch.as_tensor(verts, dtype=torch.float32, device=device)
    tri = torch.as_tensor(faces.astype(np.int32), device=device)
    pose = torch.as_tensor(world.extrinsics, dtype=torch.float32, device=device)
    matrix = torch.as_tensor(world.intrinsics, dtype=torch.float32, device=device)
    with torch.no_grad():
        clip = clip_points(camera_points(vertices, pose), matrix, height, width)
        rast, _ = dr.rasterize(context(device), clip[None], tri, (height, width))
        point = dr.interpolate(vertices[None].contiguous(), rast, tri)[0][0]
    return (rast[0, :, :, 3].int().cpu().numpy() - 1, rast[0, :, :, :2].cpu().numpy(),
            point.cpu().numpy())


def _constant_topology(world: World, oid: int, birth: int) -> int:
    """The first frame after `birth` whose mesh is not the birth frame's, or the end."""

    mesh = world.object_mesh(oid, birth)
    stop = birth + 1
    while stop < len(world) and same_topology(mesh, world.object_mesh(oid, stop)):
        stop += 1
    return stop


def _columns(world: World, oid: int) -> np.ndarray:
    """Every material column that displays the object, `(F, N, 3)`."""

    found = [material.pos for material in world.materials if oid in material.ids]
    if not found:
        return np.zeros((len(world), 0, 3), np.float32)
    return found[0] if len(found) == 1 else np.concatenate(found, axis=1)


def _nearest(points: torch.Tensor, columns: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    """Each point's nearest column, brute force on the GPU in blocks: `(distance, index)`."""

    distance, index = [], []
    chunk = max(1024, BLOCK // max(len(columns), 1))
    for lo in range(0, len(points), chunk):
        block = torch.cdist(points[lo:lo + chunk], columns).min(dim=1)
        distance.append(block.values)
        index.append(block.indices)
    return (torch.cat(distance).cpu().numpy(), torch.cat(index).cpu().numpy())


def _exact_paths(world: World, oid: int, birth: int, stop: int, faces: np.ndarray,
                 weights: np.ndarray) -> np.ndarray:
    """The queries' points from `birth` to `stop`, the birth face evaluated on each frame."""

    corners = world.object_mesh(oid, birth)[1][faces]
    used = np.unique(corners)
    lookup = np.searchsorted(used, corners)
    verts = np.stack([world.object_mesh(oid, frame)[0][used] for frame in range(birth, stop)])
    return np.einsum("qc,tqcd->tqd", weights, verts[:, lookup])


def _material_paths(world: World, oid: int, birth: int, points: np.ndarray, surface: np.ndarray,
                    device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    """The queries' points over the take, each carried by the column nearest it at birth."""

    columns = _columns(world, oid)
    alive = np.flatnonzero(np.isfinite(columns[birth]).all(axis=1))
    if not len(alive):
        return np.zeros((len(world), 0, 3), np.float32), np.zeros(0, np.int64)
    live = torch.as_tensor(columns[birth][alive], dtype=torch.float32, device=device)
    probe = surface[np.random.default_rng(birth).choice(len(surface), min(len(surface), PROBE),
                                                        replace=False)]
    typical = float(np.median(_nearest(torch.as_tensor(probe, dtype=torch.float32, device=device),
                                       live)[0]))
    distance, index = _nearest(torch.as_tensor(points, dtype=torch.float32, device=device), live)
    kept = np.flatnonzero(distance <= REACH * typical)
    followed = columns[:, alive[index[kept]]]
    return points[kept] + (followed - followed[birth]), kept


def _paths(world: World, queries: np.ndarray, device: torch.device) -> np.ndarray:
    """Every query's surface point over the take, `(Q, F, 3)`, NaN where it has none."""

    frames = len(world)
    points = np.full((len(queries), frames, 3), np.nan, np.float32)
    pixels = np.round(queries[:, 1:]).astype(np.int64)
    born = queries[:, 0].astype(np.int64)
    for birth in np.unique(born):
        soup = _soup(world, int(birth))
        if soup is None:
            continue
        verts, faces, owner, local = soup
        face_map, bary_map, surface = _surface(world, verts, faces, device)
        pixel_owner = np.concatenate([np.full(1, -1, np.int64), owner])[face_map + 1]
        rows = np.flatnonzero(born == birth)
        rows = rows[face_map[pixels[rows, 1], pixels[rows, 0]] >= 0]
        face = face_map[pixels[rows, 1], pixels[rows, 0]]
        bary = bary_map[pixels[rows, 1], pixels[rows, 0]]
        weights = np.concatenate([bary, 1.0 - bary.sum(axis=1, keepdims=True)], axis=1).astype(np.float32)
        seen = surface[pixels[rows, 1], pixels[rows, 0]]
        for oid in np.unique(owner[face]):
            mine = np.flatnonzero(owner[face] == oid)
            stop = _constant_topology(world, int(oid), int(birth))
            if stop > birth + 1 or stop == frames:
                path = _exact_paths(world, int(oid), int(birth), stop, local[face[mine]], weights[mine])
                points[rows[mine], birth:stop] = np.moveaxis(path, 0, 1)
            else:
                path, kept = _material_paths(world, int(oid), int(birth), seen[mine],
                                             surface[pixel_owner == oid], device)
                points[rows[mine[kept]], birth:] = np.moveaxis(path[birth:], 0, 1)
    return points


def _project(world: World, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The paths on screen, `(Q, F, 2)` pixels, and their camera depth `(Q, F)`."""

    camera = world.camera(points)
    matrix, depth = world.intrinsics, camera[..., 2]
    with np.errstate(invalid="ignore", divide="ignore"):
        u = (matrix[0, 0] * camera[..., 0] + matrix[0, 1] * camera[..., 1]) / depth + matrix[0, 2]
        v = matrix[1, 1] * camera[..., 1] / depth + matrix[1, 2]
    return np.stack([u, v], axis=-1).astype(np.float32), depth


def track_points(world: World, queries: np.ndarray, device: torch.device | str | None = None,
                 stride: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """`(tracks (Q, F', 2) float32, visible (Q, F') bool)` for `(Q, 3)` `(t, x, y)` queries.

    `stride` is the case's evaluation stride `k`. Surface points are carried
    through every world frame; only the `F'` frames `0, k, 2k, ...` of the
    sampled timeline are projected, depth-tested and returned. A query's `t` is
    a world frame index.
    """

    device = torch.device(device) if device is not None else default_device()
    height, width = world.shape[1], world.shape[2]
    read = sampled_indices(len(world), max(1, int(stride)))
    tracks, depth = _project(world, _paths(world, np.asarray(queries, np.float32), device))
    tracks, depth = tracks[:, read], depth[:, read]
    with np.errstate(invalid="ignore"):
        inside = (np.isfinite(tracks).all(axis=2) & (depth > NEAR) & (tracks[..., 0] >= 0)
                  & (tracks[..., 0] < width) & (tracks[..., 1] >= 0) & (tracks[..., 1] < height))
    pixels = np.where(inside[..., None], tracks, 0.0).astype(np.int64)

    visible = np.zeros(inside.shape, bool)
    for step, frame in enumerate(read):
        here = np.flatnonzero(inside[:, step])
        if not len(here):
            continue
        drawn = rasterize(*world.mesh(int(frame)), world.intrinsics, world.extrinsics,
                          height, width, device)[0]
        seen = depth[here, step]
        gap = np.abs(seen - drawn[pixels[here, step, 1], pixels[here, step, 0]])
        visible[here, step] = gap <= np.maximum(TOLERANCE * seen, FLOOR)
    return tracks, visible


# ------------------------------------------------------------------------ the queries
#
# Track2D queries are born on frame 0 at the nodes of a centred grid of at most
# `COUNT` nodes; both sides track the same queries and are compared path by path.
# The grid step is `sqrt(H * W / COUNT)` rounded, widened by a pixel while the node
# count exceeds `COUNT`.
#
# With a dynamic mask, `DRIFT` nodes stay spread over the whole frame and the rest
# are laid densely inside the mask. Track2D subtracts the median displacement of
# the background paths as camera motion; `on_mask` (written by `scorer.prepare`)
# marks the paths on the mask.

COUNT = 2048         # queries the grid holds
DRIFT = 512          # of them spread over the whole frame, for the drift correction


def lattice(height: int, width: int, step: int) -> np.ndarray:
    """The centred lattice of one step, `(N, 2)` `(x, y)`: equal margins either side."""

    rows, cols = max(1, height // step), max(1, width // step)
    ys = (height - rows * step) / 2.0 + step / 2.0 + step * np.arange(rows)
    xs = (width - cols * step) / 2.0 + step / 2.0 + step * np.arange(cols)
    grid_y, grid_x = np.meshgrid(ys, xs, indexing="ij")
    return np.stack([grid_x.ravel(), grid_y.ravel()], axis=1)


def inside(points: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Which of `(N, 2)` `(x, y)` pixels the mask holds, `(N,)` bool."""

    height, width = mask.shape
    x = np.clip(points[:, 0].astype(np.int64), 0, width - 1)
    y = np.clip(points[:, 1].astype(np.int64), 0, height - 1)
    return np.asarray(mask, dtype=bool)[y, x]


def grid_step(height: int, width: int, count: int = COUNT,
              mask: np.ndarray | None = None) -> int:
    """The step that fits at most `count` nodes, over the frame or inside a mask."""

    if mask is None:
        step = max(1, round(float(np.sqrt(height * width / max(count, 1)))))
        while (height // step) * (width // step) > count:
            step += 1
        return step
    area = int(np.count_nonzero(mask))
    step = max(1, round(float(np.sqrt(max(area, 1) / max(count, 1)))))
    while int(inside(lattice(height, width, step), mask).sum()) > count:
        step += 1
    return step


def grid_queries(height: int, width: int, count: int = COUNT,
                 mask: np.ndarray | None = None) -> np.ndarray:
    """`(Q, 3)` float32 queries `(0, x, y)` on frame 0.

    Without a dynamic mask, a centred full-frame grid. With one, a `DRIFT`-node
    full-frame grid plus the remaining budget laid inside the mask at its own step.
    """

    plain = lattice(height, width, grid_step(height, width, count))
    if mask is None or not np.any(mask):
        return _queries(plain)
    coarse = lattice(height, width, grid_step(height, width, min(DRIFT, count)))
    fine = lattice(height, width, grid_step(height, width, max(count - DRIFT, 1), mask))
    points = np.concatenate([coarse, fine[inside(fine, mask)]])
    # Fall back to the plain grid when it puts at least as many nodes on the mask.
    if int(inside(points, mask).sum()) <= int(inside(plain, mask).sum()):
        return _queries(plain)
    return _queries(points)


def _queries(points: np.ndarray) -> np.ndarray:
    """`(N, 2)` pixels as `(N, 3)` float32 queries born on frame 0."""

    return np.concatenate([np.zeros((len(points), 1)), points], axis=1).astype(np.float32)


__all__ = ["COUNT", "DRIFT", "FLOOR", "PROBE", "REACH", "TOLERANCE",
           "grid_queries", "grid_step", "inside", "lattice", "track_points"]

