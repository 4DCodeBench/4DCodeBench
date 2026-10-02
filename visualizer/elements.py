"""One frame of a world's geometry with per-component metric results, for the viewer.

The triangles are split into the components the metrics scored; each component
carries its buried fraction and mesh-check result.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np

from scorer.world import World

from .blob import Blob



@lru_cache(maxsize=8)
def verdicts(results: Path) -> dict[str, np.ndarray] | None:
    """The interpenetration and mesh results per component per frame, in `World.parts` order."""

    if not (results / "interpenetration.npz").is_file():
        return None
    with np.load(results / "interpenetration.npz") as data:
        buried, counts = data["buried"], data["components_per_frame"]
    found = {"buried": buried, "edges": np.concatenate([[0], np.cumsum(counts)])}
    if (results / "mesh.npz").is_file():
        with np.load(results / "mesh.npz") as data:
            found["frame"], found["clean"] = data["frame"], data["checks"].all(axis=1)
    return found


@lru_cache(maxsize=8)
def world_of(root: str) -> World:
    return World(root)


def geometry(name: str, world: str | Path, results: Path | None, frame: int) -> tuple[dict, bytes]:
    """Return one frame's geometry as a JSON manifest and a binary blob.

    Each vertex carries its component's slot; each component carries its object id,
    buried fraction and mesh result (1 clean, 0 failing, -1 not on disk). Slots index
    this frame's component list, so they change when objects appear or vanish.
    """

    parts = world_of(str(world)).parts(frame)
    verts = (np.concatenate([v.astype(np.float32) for _, v, _ in parts]) if parts
             else np.zeros((0, 3), np.float32))
    offsets = np.cumsum([0, *(len(v) for _, v, _ in parts)])
    faces = (np.concatenate([f.astype(np.int32) + offsets[i] for i, (_, _, f) in enumerate(parts)])
             if parts else np.zeros((0, 3), np.int32))
    part = np.repeat(np.arange(len(parts), dtype=np.int32), [len(v) for _, v, _ in parts])
    owner = np.asarray([oid for oid, _, _ in parts], dtype=np.int32)

    count = len(parts)
    buried = np.zeros(count, dtype=np.float32)
    clean = np.full(count, -1, dtype=np.int8)
    found = verdicts(results) if results is not None else None
    if found is not None and frame < len(found["edges"]) - 1:
        values = found["buried"][found["edges"][frame]:found["edges"][frame + 1]]
        buried[:len(values)] = values[:count]
        if "frame" in found:
            rows = found["clean"][found["frame"] == frame]
            if len(rows):
                clean[:len(rows)] = rows[:count].astype(np.int8)

    blob = Blob()
    blob.add("verts", verts)
    blob.add("faces", faces, dtype=np.int32)
    blob.add("part", part, dtype=np.int32)
    blob.add("owner", owner, dtype=np.int32)
    blob.add("buried", buried)
    blob.add("clean", clean, dtype=np.int8)
    return ({"world": name, "frame": int(frame), "vertices": len(verts),
             "triangles": len(faces), "components": count,
             "arrays": blob.manifest}, blob.bytes())



__all__ = ["geometry", "verdicts"]
