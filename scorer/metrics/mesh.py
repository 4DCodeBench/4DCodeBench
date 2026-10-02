"""Mesh quality: watertight, manifold, clean-face and self-intersection checks.

Each object in each frame is checked per connected component, and passes a check
when all its components do. `mesh_<check>` is the fraction of (object, frame) rows
that pass, and `mesh_score` the fraction that pass all four.
"""

from __future__ import annotations

from dataclasses import dataclass

import igl
import networkx as nx
import numpy as np
import trimesh
from igl.copyleft.cgal import remesh_self_intersections

from ..world import World
from .distances import seed
from .frames import judge_frames

SURFACE = 0.75     # median |winding number| behind the faces below which an open part is a sheet
OFFSET = 1e-3      # inward offset of winding probes, as a fraction of the bounding-box diagonal
SAMPLES = 200      # faces probed per component by `is_surface`
CHECKS = ("watertight", "manifold", "clean_faces", "no_self_intersection")


def is_surface(part: trimesh.Trimesh, rng: np.random.Generator) -> bool:
    """True if an open component is a sheet rather than a leaky solid; sheets pass `watertight`."""

    pick = rng.choice(len(part.faces), min(SAMPLES, len(part.faces)), replace=False)
    diagonal = float(np.linalg.norm(part.extents))
    inward = part.triangles_center[pick] - part.face_normals[pick] * diagonal * OFFSET
    reading = igl.fast_winding_number(part.vertices, part.faces, inward)
    return float(np.median(np.abs(reading))) < SURFACE


def crossings(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Mask of faces intersecting another face beyond shared vertices or edges (CGAL).

    Zero-area faces are skipped; degenerate and duplicate faces fall under `clean_faces`.
    """

    valid = trimesh.triangles.area(vertices[faces]) > 0
    selection = np.zeros(len(faces), dtype=bool)
    faces = faces[valid]
    if len(faces) < 2:
        return selection
    _, _, pairs, _, _ = remesh_self_intersections(
        np.asarray(vertices, np.float64), np.asarray(faces, np.int64), detect_only=True)
    selection[np.flatnonzero(valid)[np.unique(pairs)]] = True
    return selection


def unclean_faces(vertices: np.ndarray, faces: np.ndarray, components=None) -> np.ndarray:
    """Indices of faces that are degenerate or duplicated after welding within each component."""

    if components is None:
        used, corners = np.unique(faces, return_inverse=True)
    else:
        grouped, corners = np.unique(np.column_stack((np.repeat(components, 3), faces.ravel())),
                                     axis=0, return_inverse=True)
        used = grouped[:, 1]
    digits = trimesh.util.decimal_to_digits(trimesh.constants.tol.merge)
    keys = (vertices[used] * 10**digits).round().astype(np.int64)
    if components is not None:
        keys = np.column_stack((grouped[:, 0], keys))
    first, inverse = trimesh.grouping.unique_rows(keys, keep_order=True)
    welded_faces = inverse[corners].reshape(-1, 3)
    welded_triangles = vertices[used[first]][welded_faces]
    triples = np.sort(welded_faces, axis=1)
    order = np.lexsort(triples.T)
    twin = np.zeros(len(triples), dtype=bool)
    same = (triples[order[1:]] == triples[order[:-1]]).all(axis=1)
    twin[order[:-1][same]] = twin[order[1:][same]] = True
    return np.flatnonzero(~(np.diff(triples, axis=1) > 0).all(axis=1) | twin
                         | ~(trimesh.triangles.area(welded_triangles) > 0))


def bad_elements(part: trimesh.Trimesh, rng: np.random.Generator) -> dict[str, np.ndarray]:
    """Return the failing geometry of one connected component, per check."""

    edges = np.sort(part.edges, axis=1)
    unique, shared = np.unique(edges, axis=0, return_counts=True)
    unclean = unclean_faces(part.vertices, part.faces)

    open_edges = np.zeros((0, 2, 3))
    if not part.is_watertight and not is_surface(part, rng):
        border = part.edges_sorted[trimesh.grouping.group_rows(part.edges_sorted, require_count=1)]
        open_edges = part.vertices[border]
    selection = crossings(part.vertices, part.faces)

    return {
        "watertight": open_edges,
        "manifold": part.vertices[unique[shared > 2]],
        "clean_faces": part.vertices[part.faces[unclean]] if len(unclean) else np.zeros((0, 3, 3)),
        "no_self_intersection": part.triangles[selection] if selection.any() else np.zeros((0, 3, 3)),
    }


def defects(part: trimesh.Trimesh, rng: np.random.Generator) -> dict[str, bool]:
    return {name: len(bad) == 0 for name, bad in bad_elements(part, rng).items()}


@dataclass(frozen=True)
class Topology:
    """Connected-component layout of one mesh.

    `labels` per face, `order` sorting faces by label, `offsets` into `order` per
    component, and `closed` per component (every edge shared by exactly two faces).
    """

    labels: np.ndarray
    order: np.ndarray
    offsets: np.ndarray
    closed: np.ndarray


def topology(mesh: trimesh.Trimesh) -> Topology:
    labels = trimesh.graph.connected_component_labels(mesh.face_adjacency, node_count=len(mesh.faces))
    order = np.argsort(labels, kind="stable")
    offsets = np.r_[0, np.cumsum(np.bincount(labels))]
    edges, counts = np.unique(np.column_stack((np.repeat(labels, 3), mesh.edges_sorted)),
                              axis=0, return_counts=True)
    closed = np.ones(len(offsets) - 1, dtype=bool)
    closed[edges[counts != 2, 0]] = False
    return Topology(labels, order, offsets, closed)


def component_batches(whole: trimesh.Trimesh, layout: Topology):
    """Yield components of 2-5 faces grouped by identical local connectivity.

    Each item is `(component ids, (N, V, 3) vertices, (K, 3) local faces)`.
    """

    sizes = np.diff(layout.offsets)
    for size in range(2, 6):
        ids = np.flatnonzero(sizes == size)
        if not len(ids):
            continue
        faces = whole.faces[layout.order[layout.offsets[ids, None] + np.arange(size)]]
        flat = np.asarray(faces).reshape(len(ids), -1)
        order = np.argsort(flat, axis=1)
        vertices = np.take_along_axis(flat, order, axis=1)
        first = np.c_[np.ones(len(ids), bool), np.diff(vertices, axis=1) != 0]
        ranks = np.cumsum(first, axis=1) - 1
        local = np.take_along_axis(ranks, np.argsort(order, axis=1), axis=1)
        patterns, assigned = np.unique(local, axis=0, return_inverse=True)
        for index, pattern in enumerate(patterns):
            take = assigned == index
            used = vertices[take][first[take]].reshape(np.count_nonzero(take), pattern.max() + 1)
            yield ids[take], np.asarray(whole.vertices[used]), pattern.reshape(-1, 3)


def noncoplanar(points: np.ndarray) -> np.ndarray:
    """True for each `(4, 3)` tetrahedron whose orientation determinant is certified nonzero."""

    a, b, c = np.moveaxis(points[:, :3] - points[:, 3:4], 1, 0)
    bc, cb = b[:, 0] * c[:, 1], c[:, 0] * b[:, 1]
    ca, ac = c[:, 0] * a[:, 1], a[:, 0] * c[:, 1]
    ab, ba = a[:, 0] * b[:, 1], b[:, 0] * a[:, 1]
    determinant = a[:, 2] * (bc - cb) + b[:, 2] * (ca - ac) + c[:, 2] * (ab - ba)
    permanent = ((np.abs(bc) + np.abs(cb)) * np.abs(a[:, 2])
                 + (np.abs(ca) + np.abs(ac)) * np.abs(b[:, 2])
                 + (np.abs(ab) + np.abs(ba)) * np.abs(c[:, 2]))
    # Shewchuk's orient3d error bound certifies nonzero volume before skipping CGAL.
    epsilon = np.finfo(np.float64).eps / 2
    return np.abs(determinant) > (7 + 56 * epsilon) * epsilon * permanent


def clean_batch(points: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """`clean_faces` verdict for each component of a batch sharing one local face list."""

    digits = trimesh.util.decimal_to_digits(trimesh.constants.tol.merge)
    keys = (points * 10**digits).round().astype(np.int64)
    same = (keys[:, :, None] == keys[:, None, :]).all(axis=3)
    welded = np.argmax(same, axis=2)[:, faces]
    triangles = points[np.arange(len(points))[:, None, None], welded]
    clean = (trimesh.triangles.area(triangles.reshape(-1, 3, 3)) > 0).reshape(len(points), -1).all(axis=1)
    triples = np.sort(welded, axis=2)
    clean &= (np.diff(triples, axis=2) > 0).all(axis=(1, 2))
    for a in range(len(faces)):
        for b in range(a):
            clean &= (triples[:, a] != triples[:, b]).any(axis=1)
    return clean


def small_components(whole: trimesh.Trimesh, layout: Topology) -> tuple[dict[str, bool], np.ndarray]:
    """Check components of 2-5 faces in vectorised batches.

    Returns the combined verdicts and a mask of the components handled; the rest are
    left to `object_checks`.
    """

    checks = dict.fromkeys(CHECKS, True)
    handled = np.zeros(len(layout.closed), dtype=bool)
    for ids, points, faces in component_batches(whole, layout):
        edges = faces[:, [[0, 1], [1, 2], [2, 0]]].reshape(-1, 2)
        unique, counts = np.unique(np.sort(edges, axis=1), axis=0, return_counts=True)
        closed = bool((counts == 2).all())
        if closed:
            if points.shape[1] != 4 or len(faces) != 4:
                continue
            if not np.array_equal(np.sort(np.sum(1 << faces, axis=1)), [7, 11, 13, 14]):
                continue
            accepted = noncoplanar(points)
            ids, points = ids[accepted], points[accepted]
            if not len(ids):
                continue
        elif len(faces) >= 4:
            boundary = unique[counts == 1]
            if len(boundary) >= 3:
                graph = nx.from_edgelist(boundary)
                if any(degree != 2 for _, degree in graph.degree):
                    continue
                if any(len(cycle) <= 4 for cycle in nx.connected_components(graph)):
                    continue

        checks["clean_faces"] &= bool(clean_batch(points, faces).all())
        checks["manifold"] &= bool((counts <= 2).all())
        handled[ids] = True
        if closed:
            continue

        triangles = points[:, faces]
        flat = triangles.reshape(-1, 3, 3)
        positive = (trimesh.triangles.area(flat) > 0).reshape(len(points), -1)
        normals, valid = trimesh.triangles.normals(triangles=flat)
        padded = np.zeros((len(flat), 3))
        padded[valid] = normals
        spans = np.array([np.linalg.norm(p.max(axis=0) - p.min(axis=0)) for p in points])
        inward = triangles.mean(axis=2) - padded.reshape(len(points), len(faces), 3) * spans[:, None, None] * OFFSET
        for index, vertices in enumerate(points):
            winding = igl.fast_winding_number(vertices, faces, inward[index])
            checks["watertight"] &= bool(not (counts == 1).any() or np.median(np.abs(winding)) < SURFACE)
            selected = faces[positive[index]]
            if len(selected) >= 2:
                _, _, pairs, _, _ = remesh_self_intersections(vertices, selected, detect_only=True)
                checks["no_self_intersection"] &= len(pairs) == 0
    return checks, handled


def object_checks(whole: trimesh.Trimesh, rng: np.random.Generator,
                  layout: Topology | None = None) -> dict[str, bool]:
    """The four verdicts of one object, each the AND over its connected components."""

    if not len(whole.faces):
        return defects(whole, rng)
    if layout is None:
        layout = topology(whole)
    checks, handled = small_components(whole, layout)
    sizes = np.diff(layout.offsets)
    single = (sizes == 1) & (whole.area_faces[layout.order[layout.offsets[:-1]]] > 0)
    batched = (layout.closed | single) & ~handled
    selected = batched[layout.labels]
    if selected.any():
        checks["clean_faces"] &= not len(unclean_faces(
            whole.vertices, whole.faces[selected], layout.labels[selected]))
    for component in np.flatnonzero(~(single | handled)):
        start, stop = layout.offsets[component:component + 2]
        faces = whole.faces[layout.order[start:stop]]
        used, local = np.unique(faces, return_inverse=True)
        vertices, faces = whole.vertices[used], local.reshape(-1, 3)
        if layout.closed[component]:
            if checks["no_self_intersection"]:
                checks["no_self_intersection"] = not crossings(vertices, faces).any()
        else:
            part = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
            if len(faces) >= 4:
                part.fill_holes()
            for name, passed in defects(part, rng).items():
                checks[name] &= passed
    return checks


def judge(world: World, frame: int, seed_base: int,
          fixed: dict[int, dict[str, bool]], layouts: dict[int, Topology]) -> list[dict[str, bool]]:
    """Return the verdicts of every object in one frame.

    Checks run per connected component, so components of one object may touch or cross.
    Objects in `fixed` reuse precomputed verdicts; `layouts` holds cached topologies.
    """

    rng = np.random.default_rng(seed_base + frame * 8191)
    rows = []
    for oid, verts, faces in world.parts(frame):
        if oid in fixed:
            rows.append(fixed[oid])
            continue
        whole = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
        rows.append(object_checks(whole, rng, layouts.get(oid)))
    return rows


def mesh_quality(world: World) -> dict:
    """Mesh-quality readings of one world over all frames.

    An object with the same mesh in every frame is checked once; one with fixed faces
    reuses a single topology.
    """

    frames = list(range(len(world)))
    base = seed() % (2**31)
    fixed, layouts = {}, {}
    for oid, history in world.meshes.items():
        if not history:
            continue
        first, *rest = history.values()
        verts, faces = first
        if len(faces) and all(np.array_equal(faces, f) for _, f in rest):
            whole = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
            if all(np.array_equal(verts, v) for v, _ in rest):
                fixed[oid] = object_checks(whole, np.random.default_rng(base))
            else:
                layouts[oid] = topology(whole)
    judged = judge_frames(lambda w, f: judge(w, f, base, fixed, layouts), world, frames)

    verdicts: list[dict[str, bool]] = []
    at: list[int] = []
    per_frame: list[dict[str, int]] = []
    for frame, checks in zip(frames, judged, strict=True):
        verdicts.extend(checks)
        at.extend([int(frame)] * len(checks))
        per_frame.append(
            {
                "frame": int(frame),
                "components": len(checks),
                "clean": int(sum(all(check.values()) for check in checks)),
            }
        )

    # One row per object and frame, with its four verdicts.
    kept = {"arrays": {"mesh": {
        "frame": np.asarray(at, dtype=np.int32),
        "checks": np.asarray([[verdict[check] for check in CHECKS] for verdict in verdicts],
                             dtype=bool).reshape(len(verdicts), len(CHECKS)),
        "check_names": np.asarray(CHECKS, dtype=np.str_),
    }}}
    if not verdicts:
        return {
            **kept,
            "mesh_components": 0,
            "mesh_frames": frames,
            "mesh_per_frame": per_frame,
            "mesh_failures": {},
        }

    clean = sum(all(check.values()) for check in verdicts)
    return {
        **kept,
        "mesh_components": len(verdicts),
        "mesh_frames": frames,
        "mesh_per_frame": per_frame,
        **{
            f"mesh_{check}": sum(verdict[check] for verdict in verdicts) / len(verdicts)
            for check in CHECKS
        },
        "mesh_failures": {
            check: int(sum(not verdict[check] for verdict in verdicts)) for check in CHECKS
        },
        "mesh_score": clean / len(verdicts),
    }
