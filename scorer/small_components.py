"""Batched sampling of many small shell components at once.

The vectorised counterpart of `samples._component` and `samples._surface` for
components of up to 64 faces: winding numbers, surface samples and voxel
thinning run over all components of a batch together.
"""

import numpy as np

from .samples import (
    INSIDE,
    MAX_VOXELS,
    SURFACE_DENSITY,
    VolumeWeights,
    Weights,
    _cells,
    _jitter,
    _unique_voxel,
    winding,
)


def surfaces(verts, faces, areas, components, starts, counts, h, generator):
    """Area-weighted surface samples of each component, one per voxel.

    Returns `{component: (Weights, points)}`; components of zero area are left out.
    """

    if not len(components):
        return {}
    size = int(counts[components].max())
    mask = np.arange(size)[None] < counts[components, None]
    index = starts[components, None] + np.arange(size)[None]
    index = np.where(mask, index, 0)
    values = np.where(mask, areas[index], 0.0)
    total = np.empty(len(components))
    for n in np.unique(counts[components]):
        held = counts[components] == n
        total[held] = values[held, :n].sum(1)
    live = total > 0
    components, index, values, total = (
        components[live],
        index[live],
        values[live],
        total[live],
    )
    if not len(components):
        return {}
    number = np.minimum(
        MAX_VOXELS, np.maximum(1.0, SURFACE_DENSITY * total / (h * h))
    ).astype(np.int64)
    first = np.r_[0, np.cumsum(number)[:-1]]
    owner = np.repeat(np.arange(len(components)), number)
    row = np.arange(int(number.sum())) - first[owner]
    random = generator.random(int(3 * number.sum()))
    choice = random[3 * first[owner] + row]
    u = random[3 * first[owner] + number[owner] + row]
    v = random[3 * first[owner] + 2 * number[owner] + row]
    cdf = np.cumsum(values / total[:, None], axis=1)
    cdf /= cdf[np.arange(len(components)), counts[components] - 1, None]
    low = np.zeros(len(owner), np.int64)
    high = counts[components[owner]] - 1
    for _ in range(int(np.ceil(np.log2(size))) if size > 1 else 0):
        mid = (low + high) // 2
        right = choice >= cdf[owner, mid]
        low = np.where(right, mid + 1, low)
        high = np.where(right, high, mid)
    chosen = index[owner, low]
    folded = u + v > 1.0
    u, v = np.where(folded, 1.0 - u, u), np.where(folded, 1.0 - v, v)
    weight = np.column_stack([1.0 - u - v, u, v])
    points = np.einsum("cd,cdk->ck", weight, verts[faces[chosen]])
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
    keep = np.ones(len(order), bool)
    keep[1:] = (owner[order[1:]] != owner[order[:-1]]) | (
        keys[order[1:]] != keys[order[:-1]]
    ).any(1)
    kept = order[keep]
    length = np.bincount(owner[kept], minlength=len(components))
    edges = np.r_[0, length.cumsum()]
    out = {}
    for i, component in enumerate(components):
        selected = kept[edges[i] : edges[i + 1]]
        out[int(component)] = (
            Weights(faces[chosen[selected]], weight[selected].astype(np.float32)),
            points[selected],
        )
    return out


def sample(
    verts,
    faces,
    triangles,
    areas,
    components,
    starts,
    counts,
    vertex_order,
    vertex_edges,
    low,
    high,
    h,
    generator,
):
    """Samples of each component in `components`, as `(weights, points)` pairs.

    Voxel centres with winding number above 0.5 are inside and give
    `VolumeWeights`; a component with no inside voxel is sampled on its surface.
    """

    base = _cells(low[components], h)
    shape = _cells(high[components], h) - base + 1
    number = shape.prod(1)
    first = np.r_[0, number.cumsum()[:-1]]
    owner = np.repeat(np.arange(len(components)), number)
    row = np.arange(int(number.sum())) - first[owner]
    plane = shape[owner, 1] * shape[owner, 2]
    cell = base[owner] + np.column_stack(
        [row // plane, row % plane // shape[owner, 2], row % shape[owner, 2]]
    )
    centres = (cell + 0.5) * h
    points = (cell + _jitter(cell)) * h
    values = winding(triangles, centres, components[owner], starts, counts)
    inside = np.isfinite(values) & (np.abs(values) > INSIDE)
    inside_count = np.bincount(owner[inside], minlength=len(components))
    surface = surfaces(
        verts, faces, areas, components[inside_count == 0], starts, counts, h, generator
    )
    inside_points = points[inside]
    edges = np.r_[0, inside_count.cumsum()]
    out = []
    for i, component in enumerate(components):
        if not inside_count[i]:
            if int(component) in surface:
                out.append(surface[int(component)])
            continue
        columns = vertex_order[vertex_edges[component] : vertex_edges[component + 1]]
        selected = faces[starts[component] : starts[component] + counts[component]]
        local = np.searchsorted(columns, selected)
        point = inside_points[edges[i] : edges[i + 1]]
        point = point[_unique_voxel(point, h)]
        out.append((VolumeWeights(point, verts[columns], local, columns), point))
    return out
