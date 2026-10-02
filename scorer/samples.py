"""Material samples: one voxel-thinned point set per frame.

A sample is a fixed linear combination of one material's vertices,
`p(s) = Σᵢ wᵢ vᵢ(s)`, with the weights fixed at the frame it is generated on:
- unconnected material: the column itself;
- tetrahedral mesh: barycentric coordinates of the tetrahedron a jittered voxel point lies in;
- shell: mean value coordinates (Ju, Schaefer & Warren 2005) over the connected
  component that contains the voxel point, so each component is sampled on its own;
- component with no interior: barycentric coordinates of an area-weighted surface point.
The union over the world's materials is thinned to one sample per voxel of edge `h`,
in an order fixed by the voxel and the position.

A scored pair calibrates `h` on the reference and culls to the reference camera.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from config.scorer import N

from .metrics.cloud import SCENE_POINTS, cloud, extent, subsample
from .metrics.distances import device, seed
from .raster import NEAR

VOXEL0 = 1.0 / 32.0  # first voxel edge, in reference radii
# Odd 64-bit multipliers -- 2^64/phi (SplitMix64's increment) and two xxHash64
# primes -- that hash a voxel's integer index into three seeds; see `_jitter`.
MIX = np.array(
    [0x9E3779B97F4A7C15, 0xC2B2AE3D27D4EB4F, 0x165667B19E3779F9], dtype=np.uint64
)
MAX_VOXELS = 4 * N  # most voxels probed in one box; a larger box is subsampled
SURFACE_DENSITY = 4.0  # surface samples per h² of a shell with no interior
INSIDE = 0.5  # |winding number| of a point inside a shell
MVC_PAIRS = 1 << 20  # (sample, triangle) pairs one mean-value block holds
WINDING_PAIRS = 1 << 24  # (point, triangle) pairs one winding-number block holds
MVC_WEIGHTS = 1 << 24  # weights materialised by one trajectory block
TOLERANCE = 1e-9  # barycentric and degeneracy tolerance


@dataclass(frozen=True)
class Weights:
    """One material's samples as rows of vertex weights.

    `index` `(S, D)` holds the vertices of each row's `D` weights. When `index` is
    `None`, each row is dense over the vertices in `columns`, or over every vertex
    when `columns` is `None` too.
    """

    index: np.ndarray | None
    weight: np.ndarray
    columns: np.ndarray | None = None

    def __len__(self) -> int:
        return len(self.weight)

    def take(self, rows: np.ndarray) -> Weights:
        return Weights(
            None if self.index is None else self.index[rows],
            self.weight[rows],
            self.columns,
        )

    def select(self, verts: np.ndarray) -> np.ndarray:
        """The vertices a dense block uses, selected along the last-but-one axis."""

        return verts if self.columns is None else verts[..., self.columns, :]

    def apply(self, verts: np.ndarray) -> np.ndarray:
        if self.index is None:
            return self.weight @ self.select(verts)
        return np.einsum("cd,cdk->ck", self.weight, verts[self.index])

    def alive(self, finite: np.ndarray) -> np.ndarray:
        """Rows whose every weighted vertex is finite."""

        if self.index is None:
            held = finite if self.columns is None else finite[..., self.columns]
            return ((self.weight == 0) | held).all(axis=1)
        return ((self.weight == 0) | finite[self.index]).all(axis=1)


@dataclass(frozen=True)
class VolumeWeights:
    """Samples inside one closed shell component, weighted by mean value coordinates.

    The weights are recomputed per block in `track` from the frame-of-origin
    `points`, `vertices` and component-local `faces`; `columns` maps the
    component's vertices to the material's columns.
    """

    points: np.ndarray
    vertices: np.ndarray
    faces: np.ndarray
    columns: np.ndarray

    def __len__(self) -> int:
        return len(self.points)

    def take(self, rows):
        return VolumeWeights(self.points[rows], self.vertices, self.faces, self.columns)

    def track(self, vertices):
        """Sample positions for `(T, N, 3)` material columns: `(T, S, 3)`, NaN where dead."""

        values = vertices[:, self.columns]
        finite = np.isfinite(values).all(axis=-1)
        flat = np.moveaxis(np.where(finite[..., None], values, 0.0), 0, 1).reshape(
            len(self.columns), -1
        )
        dev = device()
        history = torch.as_tensor(flat, dtype=torch.float64, device=dev)
        missing = (
            None
            if finite.all()
            else torch.as_tensor(~finite, dtype=torch.float32, device=dev).T
        )
        block = max(1, MVC_WEIGHTS // len(self.columns))
        out = np.empty((len(vertices), len(self.points), 3), np.float32)
        for start in range(0, len(self.points), block):
            weight = mean_value_coordinates_tensor(
                self.points[start : start + block], self.vertices, self.faces
            )
            points = (
                (weight.double() @ history)
                .reshape(len(weight), len(vertices), 3)
                .permute(1, 0, 2)
            )
            if missing is not None:
                dead = (weight != 0).float() @ missing
                points[dead.T > 0] = torch.nan
            out[:, start : start + len(weight)] = points.float().cpu().numpy()
        return out


@dataclass(frozen=True)
class Samples:
    """The samples of one frame, and where they sit at any frame."""

    common: list[np.ndarray]
    blocks: list[tuple[int, Weights | VolumeWeights]]
    origin_frame: int | None = None
    origin_points: np.ndarray | None = None

    def __len__(self) -> int:
        return sum(len(block) for _, block in self.blocks)

    def track(self, start: int, stop: int) -> np.ndarray:
        """Every sample over a range of frames, `(stop - start, samples, 3)`, NaN where dead."""

        found = []
        for material, block in self.blocks:
            verts = self.common[material][start:stop]
            if isinstance(block, VolumeWeights):
                found.append(block.track(verts))
                continue
            if block.index is None:
                verts = block.select(verts)
            else:
                verts = verts[:, block.index]
            finite = np.isfinite(verts).all(axis=-1)
            filled = np.where(finite[..., None], verts, 0.0).astype(np.float64)
            if block.index is None:
                flat = np.moveaxis(filled, 0, 1).reshape(verts.shape[1], -1)
                points = np.moveaxis(
                    (block.weight @ flat).reshape(-1, len(verts), 3), 0, 1
                )
                held = (block.weight != 0).astype(np.float32) @ (~finite).astype(
                    np.float32
                ).T
                points[held.T > 0] = np.nan
            else:
                points = np.einsum("cd,ncdk->nck", block.weight, filled)
                points[((block.weight != 0) & ~finite).any(axis=2)] = np.nan
            found.append(points)
        if not found:
            return np.zeros((stop - start, 0, 3), np.float32)
        return np.concatenate(found, axis=1).astype(np.float32)

    def positions(self, frame: int) -> np.ndarray:
        """Every sample at `frame`, NaN where it is not alive."""

        if frame == self.origin_frame:
            return self.origin_points
        return self.track(frame, frame + 1)[0]

    def subsample(self, cap: int, generator: np.random.Generator) -> Samples:
        total = len(self)
        if total <= cap:
            return self
        keep = np.zeros(total, dtype=bool)
        keep[generator.choice(total, cap, replace=False)] = True
        blocks, start = [], 0
        for material, block in self.blocks:
            rows = keep[start : start + len(block)]
            start += len(block)
            if rows.any():
                blocks.append((material, block.take(rows)))
        return Samples(
            self.common,
            blocks,
            self.origin_frame,
            None if self.origin_points is None else self.origin_points[keep],
        )


def common(world, transform) -> list[np.ndarray]:
    """Every material's positions in the registered common frame."""

    return [
        transform(world.camera(material.pos)).astype(np.float32)
        for material in world.materials
    ]


def _cells(points: np.ndarray, h: float) -> np.ndarray:
    return np.floor(np.asarray(points, dtype=np.float64) / h).astype(np.int64)


def _jitter(cell: np.ndarray) -> np.ndarray:
    """Three uniforms in [0, 1) per voxel, hashed from its integer index (SplitMix64).

    The offset inside a voxel depends on the voxel alone, so the same material
    yields the same points however it is split across archives.
    """

    key = (cell.view(np.uint64) * MIX).sum(axis=1)
    out = []
    for salt in MIX:
        z = key + salt
        z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        out.append((z ^ (z >> np.uint64(31))) >> np.uint64(11))
    return np.stack(out, axis=1).astype(np.float64) / float(1 << 53)


def _voxel_points(low, high, h, generator):
    """Each voxel of a box as its centre and one jittered point, with the flat grid indices.

    Returns `(flat, base, counts, centres, points)`. Callers test containment at
    the centre and sample at the jittered point. A box of more than `MAX_VOXELS`
    voxels is subsampled with `generator`.
    """

    base = _cells(low, h)
    counts = _cells(high, h) - base + 1
    total = int(np.prod(counts))
    flat = (
        np.arange(total)
        if total <= MAX_VOXELS
        else np.unique(generator.integers(0, total, MAX_VOXELS))
    )
    cell = base + np.stack(np.unravel_index(flat, tuple(counts)), axis=1)
    return flat, base, counts, (cell + 0.5) * h, (cell + _jitter(cell)) * h


def _unique_voxel(points: np.ndarray, h: float) -> np.ndarray:
    """One row per voxel, in an order fixed by the voxel and the position alone."""

    keys = _cells(points, h)
    order = np.lexsort(
        (points[:, 2], points[:, 1], points[:, 0], keys[:, 2], keys[:, 1], keys[:, 0])
    )
    keys = keys[order]
    first = np.ones(len(order), dtype=bool)
    first[1:] = (keys[1:] != keys[:-1]).any(axis=1)
    return order[first]


def _tets(verts, tets, low, high, h, generator):
    """Voxel points inside a tetrahedral mesh, weighted by barycentric coordinates.

    Each tetrahedron is tested only against the voxels of its own bounding box.
    """

    flat, base, counts, centres, points = _voxel_points(low, high, h, generator)
    # zero-volume tetrahedra have no barycentric coordinates
    volume = np.linalg.det(np.swapaxes(verts[tets][:, 1:] - verts[tets][:, :1], 1, 2))
    tets = tets[np.abs(volume) > TOLERANCE * h**3]
    corners = verts[tets]
    span_low = np.clip(_cells(corners.min(axis=1), h) - base, 0, counts - 1)
    span_high = np.clip(_cells(corners.max(axis=1), h) - base, 0, counts - 1)
    span = span_high - span_low + 1
    per = span.prod(axis=1)
    owner = np.repeat(np.arange(len(tets)), per)
    offset = np.arange(int(per.sum())) - np.repeat(np.cumsum(per) - per, per)
    plane, row = span[owner, 1] * span[owner, 2], span[owner, 2]
    cell = span_low[owner] + np.stack(
        [offset // plane, offset % plane // row, offset % row], axis=1
    )
    want = (cell[:, 0] * counts[1] + cell[:, 1]) * counts[2] + cell[:, 2]
    where = np.searchsorted(flat, want)
    held = flat[np.minimum(where, len(flat) - 1)] == want
    owner, where = owner[held], where[held]

    origin = corners[owner, 0]
    edges = np.swapaxes(corners[owner, 1:] - origin[:, None], 1, 2)
    bary = np.linalg.solve(edges, (centres[where] - origin)[..., None])[..., 0]
    inside = (bary >= -TOLERANCE).all(axis=1) & (bary.sum(axis=1) <= 1.0 + TOLERANCE)
    owner, where, origin, edges = (
        owner[inside],
        where[inside],
        origin[inside],
        edges[inside],
    )
    bary = np.linalg.solve(edges, (points[where] - origin)[..., None])[..., 0]
    weight = np.column_stack([1.0 - bary.sum(axis=1), bary]).astype(np.float32)
    kept = _unique_voxel(points[where], h)
    return Weights(tets[owner[kept]], weight[kept]), points[where[kept]]


def _surface(verts, faces, h, generator):
    """Area-weighted surface samples, one per voxel, weighted barycentrically."""

    corners = verts[faces]
    area = 0.5 * np.linalg.norm(
        np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0]), axis=1
    )
    total = float(area.sum())
    if total <= 0:
        return None
    count = int(min(MAX_VOXELS, max(1.0, SURFACE_DENSITY * total / (h * h))))
    owner = generator.choice(len(faces), count, p=area / total)
    u, v = generator.random(count), generator.random(count)
    folded = u + v > 1.0
    u, v = np.where(folded, 1.0 - u, u), np.where(folded, 1.0 - v, v)
    weight = np.column_stack([1.0 - u - v, u, v])
    points = np.einsum("cd,cdk->ck", weight, corners[owner])
    kept = _unique_voxel(points, h)
    return Weights(faces[owner[kept]], weight[kept].astype(np.float32)), points[kept]


def _single_faces(verts, faces, h, generator):
    """Surface samples of many one-triangle components at once, one per voxel per face."""

    corners = verts[faces]
    area = 0.5 * np.linalg.norm(
        np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0]), axis=1
    )
    count = np.minimum(
        MAX_VOXELS, np.maximum(1.0, SURFACE_DENSITY * area / (h * h))
    ).astype(np.int64)
    starts = np.r_[0, np.cumsum(count)[:-1]]
    owner = np.repeat(np.arange(len(faces)), count)
    row = np.arange(int(count.sum())) - starts[owner]
    random = generator.random(int(3 * count.sum()))
    u = random[3 * starts[owner] + count[owner] + row]
    v = random[3 * starts[owner] + 2 * count[owner] + row]
    folded = u + v > 1.0
    u, v = np.where(folded, 1.0 - u, u), np.where(folded, 1.0 - v, v)
    weight = np.column_stack([1.0 - u - v, u, v])
    points = np.einsum("cd,cdk->ck", weight, corners[owner])
    keys = _cells(points, h)
    order = np.lexsort(
        (
            points[:, 2],
            points[:, 1],
            points[:, 0],
            keys[:, 2],
            keys[:, 1],
            keys[:, 0],
            owner,
        )
    )
    first = np.ones(len(order), bool)
    first[1:] = (owner[order[1:]] != owner[order[:-1]]) | (
        keys[order[1:]] != keys[order[:-1]]
    ).any(axis=1)
    kept = order[first]
    return Weights(faces[owner[kept]], weight[kept].astype(np.float32)), points[kept]


def winding(triangles, centres, owners, starts, counts):
    """The generalised winding number of each centre with respect to its owning component.

    Each triangle contributes the solid angle it subtends at the point (Van Oosterom &
    Strackee 1983); the sum over a shell divided by 4π is 1 inside a closed shell and 0
    outside. `triangles` is a `(T, 3, 3)` tensor with the faces sorted by component;
    component `c` owns rows `starts[c]:starts[c] + counts[c]`. Centres are grouped by
    face count and run on the GPU in blocks.
    """

    out = np.empty(len(centres), np.float32)
    for count in np.unique(counts[owners]):
        rows = np.flatnonzero(counts[owners] == count)
        batch = max(1, WINDING_PAIRS // int(count))
        for begin in range(0, len(rows), batch):
            selected = rows[begin : begin + batch]
            first = torch.as_tensor(starts[owners[selected]], device=triangles.device)
            tri = triangles[
                first[:, None] + torch.arange(int(count), device=triangles.device)
            ]
            q = torch.as_tensor(
                centres[selected], dtype=torch.float32, device=triangles.device
            )
            a, b, c = (tri - q[:, None, None]).unbind(2)
            la, lb, lc = a.norm(dim=-1), b.norm(dim=-1), c.norm(dim=-1)
            numerator = (a * torch.linalg.cross(b, c)).sum(-1)
            denominator = (
                la * lb * lc
                + (a * b).sum(-1) * lc
                + (b * c).sum(-1) * la
                + (c * a).sum(-1) * lb
            )
            out[selected] = (
                (torch.atan2(numerator, denominator).sum(1) / (2 * np.pi)).cpu().numpy()
            )
    return out


def mean_value_coordinates_tensor(
    points: np.ndarray, verts: np.ndarray, faces: np.ndarray
) -> torch.Tensor:
    """Ju, Schaefer & Warren (2005) coordinates of `points` in a closed triangle shell.

    Returns `(P, V)` float32 weights on `device()`. Per point and triangle, the
    unit directions to the three corners span a spherical triangle whose angles
    give each corner its share; the shares are summed over the shell and
    normalised. A row where the formula degenerates (the point on a vertex, on a
    triangle, or in a triangle's plane) takes weight 1 on the nearest vertex.
    """

    dev = device()
    v = torch.as_tensor(verts, dtype=torch.float64, device=dev)
    f = torch.as_tensor(faces, dtype=torch.int64, device=dev)
    triangles, columns = v[f], f.reshape(-1)
    out = torch.zeros(len(points), len(v), dtype=torch.float64, device=dev)
    block = max(1, MVC_PAIRS // len(f))
    for start in range(0, len(points), block):
        x = torch.as_tensor(
            points[start : start + block], dtype=torch.float64, device=dev
        )
        offset = triangles[None] - x[:, None, None, :]  # (c, T, 3, 3)
        distance = offset.norm(dim=-1)
        unit = offset / distance.clamp_min(TOLERANCE).unsqueeze(-1)
        chord = (unit.roll(-1, dims=2) - unit.roll(-2, dims=2)).norm(dim=-1)
        theta = 2.0 * torch.asin((chord / 2.0).clamp(max=1.0))
        half = theta.sum(dim=-1, keepdim=True) / 2.0
        sine = torch.sin(theta)
        cosine = (
            2.0
            * torch.sin(half)
            * torch.sin(half - theta)
            / (sine.roll(-1, 2) * sine.roll(-2, 2))
        ) - 1.0
        signed = torch.sign(torch.linalg.det(unit)).unsqueeze(-1) * torch.sqrt(
            (1.0 - cosine * cosine).clamp_min(0.0)
        )
        share = (
            theta
            - cosine.roll(-1, 2) * theta.roll(-2, 2)
            - cosine.roll(-2, 2) * theta.roll(-1, 2)
        ) / (distance * sine.roll(-1, 2) * signed.roll(-2, 2))
        share = torch.where(torch.isfinite(share), share, torch.zeros_like(share))
        share[(signed.abs() < TOLERANCE).any(dim=-1)] = 0.0
        out[start : start + len(x)].index_add_(1, columns, share.reshape(len(x), -1))

    total = out.sum(dim=1, keepdim=True)
    coordinates = out / total
    broken = ~torch.isfinite(coordinates).all(dim=1) | (total[:, 0].abs() < TOLERANCE)
    if bool(broken.any()):
        used = torch.unique(columns)
        x = torch.as_tensor(points, dtype=torch.float64, device=dev)[broken]
        coordinates[broken] = 0.0
        coordinates[broken, used[torch.cdist(x, v[used]).argmin(dim=1)]] = 1.0
    return coordinates.float()


def _components(faces, count):
    """Each vertex's connected component, from the faces that share vertices."""

    rows = np.concatenate([faces[:, 0], faces[:, 1], faces[:, 2]])
    cols = np.concatenate([faces[:, 1], faces[:, 2], faces[:, 0]])
    graph = coo_matrix(
        (np.ones(len(rows), np.int8), (rows, cols)), shape=(count, count)
    )
    return connected_components(graph, directed=False)[1]


def _component(verts, faces, columns, h, generator):
    """One closed component's voxel points, weighted by its mean value coordinates.

    A voxel centre is inside when the winding number exceeds `INSIDE`. A
    component with no inside voxel at this `h` (an open sheet, or one thinner
    than a voxel) is sampled on its surface.
    """

    if len(faces) < 4:
        return _surface(verts, faces, h, generator)
    held = verts[columns]
    _, _, _, centres, points = _voxel_points(
        held.min(axis=0), held.max(axis=0), h, generator
    )
    local = np.searchsorted(columns, faces)
    triangles = torch.as_tensor(held, dtype=torch.float32, device=device())[
        torch.as_tensor(local, dtype=torch.int64, device=device())
    ]
    owners = np.zeros(len(centres), np.int64)  # one component owning every face
    values = winding(triangles, centres, owners, np.array([0]), np.array([len(faces)]))
    inside = np.isfinite(values) & (np.abs(values) > INSIDE)
    if not inside.any():
        return _surface(verts, faces, h, generator)
    points = points[inside]
    kept = _unique_voxel(points, h)
    points = points[kept]
    return VolumeWeights(points, held, local, columns), points


def _shell(verts, faces, h, generator):
    """Every connected component of a shell, each sampled on its own.

    Components of 4 to 64 faces and at most 64 vertices are batched through
    `small_components.sample`, runs of one-triangle components through
    `_single_faces`, and the rest go one by one through `_component`.
    """

    label = _components(faces, len(verts))
    vertex_order = np.argsort(label, kind="stable")
    vertex_edges = np.r_[0, np.bincount(label).cumsum()]
    face_label = label[faces[:, 0]]
    face_order = np.argsort(face_label, kind="stable")
    face_counts = np.bincount(face_label, minlength=len(vertex_edges) - 1)
    face_edges = np.r_[0, face_counts.cumsum()]
    found = []
    active = np.flatnonzero(face_counts)
    small = (face_counts >= 4) & (face_counts <= 64) & (np.diff(vertex_edges) <= 64)
    if small.any():
        low = np.minimum.reduceat(verts[vertex_order], vertex_edges[:-1], axis=0)
        high = np.maximum.reduceat(verts[vertex_order], vertex_edges[:-1], axis=0)
        totals = (_cells(high, h) - _cells(low, h) + 1).prod(1)
        small &= totals <= MAX_VOXELS
        sorted_faces = faces[face_order]
        corners = verts[sorted_faces]
        areas = 0.5 * np.linalg.norm(
            np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0]),
            axis=1,
        )
        triangles = torch.as_tensor(verts, dtype=torch.float32, device=device())[
            torch.as_tensor(sorted_faces, dtype=torch.int64, device=device())
        ]
    else:
        small[:] = False
    eligible = np.zeros(len(face_counts), bool)
    single = np.flatnonzero(face_counts == 1)
    if len(single):
        corners = verts[faces[face_order[face_edges[single]]]]
        area = np.linalg.norm(
            np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0]),
            axis=1,
        )
        voxels = (_cells(corners.max(1), h) - _cells(corners.min(1), h) + 1).prod(1)
        eligible[single] = (area > 0) & (voxels <= MAX_VOXELS)
    slot = 0
    while slot < len(active):
        component = active[slot]
        if small[component]:
            from .small_components import sample

            end = slot + 1
            queries = int(totals[component])
            while (
                end < len(active)
                and small[active[end]]
                and queries + totals[active[end]] <= (1 << 20)
            ):
                queries += int(totals[active[end]])
                end += 1
            found.extend(
                sample(
                    verts,
                    sorted_faces,
                    triangles,
                    areas,
                    active[slot:end],
                    face_edges,
                    face_counts,
                    vertex_order,
                    vertex_edges,
                    low,
                    high,
                    h,
                    generator,
                )
            )
            slot = end
            continue
        if eligible[component]:
            end = slot + 1
            while end < len(active) and eligible[active[end]]:
                end += 1
            selected = face_order[face_edges[active[slot:end]]]
            found.append(_single_faces(verts, faces[selected], h, generator))
            slot = end
            continue
        columns = vertex_order[vertex_edges[component] : vertex_edges[component + 1]]
        selected = face_order[face_edges[component] : face_edges[component + 1]]
        made = _component(verts, faces[selected], columns, h, generator)
        if made is not None and len(made[1]):
            found.append(made)
        slot += 1
    return found


def _candidates(piece, verts, finite, h, generator):
    """One material's candidate samples at one frame, as blocks of weights and positions."""

    if piece.faces is None and piece.tets is None:
        index = np.nonzero(finite)[0]
        return [
            (
                Weights(index[:, None], np.ones((len(index), 1), np.float32)),
                verts[index],
            )
        ]

    connectivity = piece.tets if piece.tets is not None else piece.faces
    connectivity = connectivity[finite[connectivity].all(axis=1)].astype(np.int64)
    if not len(connectivity):
        return []
    # faces that use a dead vertex are dropped; dead vertices are zero-filled
    points = np.where(finite[:, None], verts, 0.0).astype(np.float64)
    low, high = verts[finite].min(axis=0), verts[finite].max(axis=0)
    if piece.tets is not None:
        return [_tets(points, connectivity, low, high, h, generator)]
    return _shell(points, connectivity, h, generator)


def frame_samples(
    world,
    common_pos: list[np.ndarray],
    h: float,
    frame: int,
    generator: np.random.Generator,
) -> Samples:
    """Every material's candidates at `frame`, thinned to one sample per voxel."""

    found, points, bounds = [], [], [0]
    for index, piece in enumerate(world.materials):
        verts = common_pos[index][frame]
        finite = np.isfinite(verts).all(axis=1)
        if not finite.any():
            continue
        for block, made in _candidates(piece, verts, finite, h, generator):
            if not len(made):
                continue
            found.append((index, block))
            points.append(made)
            bounds.append(bounds[-1] + len(made))
    if not found:
        return Samples(common_pos, [])

    kept = np.sort(_unique_voxel(np.concatenate(points), h))
    edges = np.searchsorted(kept, bounds)
    blocks = []
    for (index, block), start, left, right in zip(
        found, bounds[:-1], edges[:-1], edges[1:], strict=True
    ):
        rows = kept[left:right] - start
        if len(rows):
            blocks.append((index, block.take(rows)))
    return Samples(
        common_pos, blocks, frame, np.concatenate(points)[kept].astype(np.float32)
    )


def _fullest_frame(common_pos: list[np.ndarray]) -> int:
    """The frame with the most live columns, on which `h` is calibrated."""

    if not common_pos:
        return 0
    live = sum(
        np.isfinite(positions).all(axis=2).sum(axis=1) for positions in common_pos
    )
    return int(np.argmax(live))


def _voxel_size(
    world, common_pos: list[np.ndarray], target: int, generator: np.random.Generator
) -> float | None:
    """The voxel edge that leaves about `target` samples on the reference's fullest frame.

    Two cube-root updates from `VOXEL0`; None when the frame yields no sample.
    """

    frame = _fullest_frame(common_pos)
    h = VOXEL0
    for _ in range(2):
        count = len(frame_samples(world, common_pos, h, frame, generator))
        if count == 0:
            return None
        h = float(h * (count / target) ** (1 / 3))
    return h


def visible(points: np.ndarray, alignment, gt) -> np.ndarray:
    """Mask of samples in front of the reference camera and inside its image.

    `points` are in reference radii; `alignment` maps them back to camera metres.
    """

    camera = np.asarray(points, dtype=np.float64) * alignment.radius + alignment.centre
    depth = camera[:, 2]
    pixel = (camera @ np.asarray(gt.intrinsics).T)[:, :2] / depth[:, None]
    width, height = gt.resolution
    with np.errstate(invalid="ignore"):
        return (
            (depth > NEAR)
            & (pixel[:, 0] >= 0)
            & (pixel[:, 0] < width)
            & (pixel[:, 1] >= 0)
            & (pixel[:, 1] < height)
        )


def dynamic_frames(frames: int, count: int) -> list[int]:
    """`count` evenly spaced frames in `[0, frames)` for the dynamic clouds, frame 0 first."""

    return [int(frame) for frame in np.linspace(0, frames - 1, count).astype(np.int64)]


@dataclass(frozen=True)
class Sampling:
    """Both worlds' materials in the common frame, and the voxel edge they share."""

    gt: list[np.ndarray]
    pred: list[np.ndarray]
    h: float | None


def material(gt, pred, alignment) -> Sampling:
    """Map both worlds into the common frame and calibrate `h` on the reference."""

    positions = common(gt, alignment.gt)
    return Sampling(
        positions,
        common(pred, alignment.pred),
        _voxel_size(gt, positions, N, np.random.default_rng([seed(), 0])),
    )


@dataclass(frozen=True)
class Unit:
    """One world's frame-0 cloud centre and RMS radius, in camera metres."""

    centre: np.ndarray
    radius: float

    def gt(self, points: np.ndarray) -> np.ndarray:
        """Camera metres to units of the world's own radius, as `Registration.gt`."""

        return (np.asarray(points, dtype=np.float64) - self.centre) / self.radius

    def metres(self, points: np.ndarray) -> np.ndarray:
        """The inverse of `gt`: radii to camera metres."""

        return np.asarray(points, dtype=np.float64) * self.radius + self.centre


def scene_unit(world) -> Unit:
    """The frame-0 visible cloud's camera-space unit of length."""

    centre, radius = extent(cloud(world, 0))
    return Unit(np.asarray(centre, dtype=np.float64), float(radius))


def lone_scene(world, unit: Unit) -> np.ndarray:
    """One world's frame-0 dense cloud, subsampled to `SCENE_POINTS`, in its own radii."""

    return unit.gt(
        subsample(cloud(world, 0), SCENE_POINTS, np.random.default_rng(seed()))
    )


__all__ = [
    "Samples",
    "Sampling",
    "Unit",
    "Weights",
    "common",
    "dynamic_frames",
    "frame_samples",
    "lone_scene",
    "material",
    "scene_unit",
    "visible",
]
