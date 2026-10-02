"""The scorer's view of a world directory (`docs/spec.md`).

`World` holds `camera.json`, `render.mp4`, the rendered geometry of `meshes/`
and the material columns of `dynamics/`, plus the tracer columns advected from
`solver/` for an identity-less liquid. Archives load on first access.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, replace
from fractions import Fraction
from functools import cached_property
from pathlib import Path

import numpy as np

from config.world import (
    CAMERA_FILENAME,
    DYNAMICS_DIRNAME,
    DYNAMICS_FACES_KEY,
    DYNAMICS_IDS_KEY,
    DYNAMICS_POS_KEY,
    DYNAMICS_TETS_KEY,
    FACES_DTYPE,
    INDEX_DTYPE,
    MESHES_DIRNAME,
    NPZ_SUFFIX,
    OBJECT_ID_DTYPE,
    POS_DTYPE,
    TRACER_POINTS,
    VIDEO_FILENAME,
)

EMPTY_VERTS = np.zeros((0, 3), dtype=POS_DTYPE)
EMPTY_FACES = np.zeros((0, 3), dtype=FACES_DTYPE)


def archive_paths(root: str | Path) -> dict[int, Path]:
    """Map each object id to its `meshes/<id>.npz`, ascending; raise on a bad or repeated name."""

    found = {}
    for path in sorted((Path(root) / MESHES_DIRNAME).iterdir()):
        if path.suffix != NPZ_SUFFIX or not path.stem.isdigit() or int(path.stem) <= 0:
            raise ValueError(f"{path}: name must be <positive id>.npz")
        oid = int(path.stem)
        if oid in found:
            raise ValueError(f"{path}: id {oid} is already declared")
        found[oid] = path
    return dict(sorted(found.items()))


def read_meshes(path: str | Path, frame_count: int | None = None) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """One mesh archive as `{frame: (vertices, faces)}`; raise `ValueError` on a violation."""

    from .validation import _read_mesh

    loaded, errors = _read_mesh(str(path), Path(path), frame_count)
    if errors:
        raise ValueError("; ".join(errors))
    return loaded


def video_stream(path: str | Path) -> tuple[int, int, int, Fraction]:
    """A video's `(F, H, W, fps)`, with F counted by decoding every frame."""

    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
         "-show_entries", "stream=nb_read_frames,height,width,r_frame_rate",
         "-of", "json", str(path)],
        capture_output=True, text=True, check=True,
    )
    stream = json.loads(probe.stdout)["streams"][0]
    return (int(stream["nb_read_frames"]), int(stream["height"]), int(stream["width"]),
            Fraction(stream["r_frame_rate"]))


@dataclass(frozen=True)
class Material:
    """One `dynamics/` archive: a piece of material and the ids that display it."""

    name: str
    pos: np.ndarray             # (F, N, 3) float32, NaN outside a column's lifetime
    ids: np.ndarray             # (M,) uint16
    faces: np.ndarray | None    # (T, 3) int32
    tets: np.ndarray | None     # (K, 4) int32


def read_material(path: Path, mesh_ids: set[int], frame_count: int) -> Material:
    """Load and validate one `dynamics/` archive; raise `ValueError` on a violation."""

    from .validation import _check_lifetimes

    allowed = {DYNAMICS_POS_KEY, DYNAMICS_IDS_KEY, DYNAMICS_FACES_KEY, DYNAMICS_TETS_KEY}
    with np.load(path) as archive:
        stray = set(archive.files) - allowed
        if stray:
            raise ValueError(f"{path}: unknown members {sorted(stray)}")
        arrays = {key: archive[key] for key in archive.files}
    pos, ids = arrays[DYNAMICS_POS_KEY], arrays[DYNAMICS_IDS_KEY]
    if pos.dtype != np.dtype(POS_DTYPE) or pos.ndim != 3 or pos.shape[2] != 3:
        raise ValueError(f"{path}: pos must be float32 (F,N,3)")
    if pos.shape[0] != frame_count:
        raise ValueError(f"{path}: pos frame count {pos.shape[0]} != video {frame_count}")
    error = _check_lifetimes(str(path), pos)
    if error:
        raise ValueError(error)
    if ids.dtype != np.dtype(INDEX_DTYPE) or ids.ndim != 1 or not len(ids):
        raise ValueError(f"{path}: ids must be a nonempty uint16 vector")
    missing = set(map(int, ids)) - mesh_ids
    if missing:
        raise ValueError(f"{path}: ids have no mesh archives: {sorted(missing)}")
    if DYNAMICS_FACES_KEY in arrays and DYNAMICS_TETS_KEY in arrays:
        raise ValueError(f"{path}: cannot contain both faces and tets")
    for key, width in ((DYNAMICS_FACES_KEY, 3), (DYNAMICS_TETS_KEY, 4)):
        if key not in arrays:
            continue
        index = arrays[key]
        if index.dtype != np.dtype(FACES_DTYPE) or index.ndim != 2 or index.shape[1] != width:
            raise ValueError(f"{path}: {key} must be int32 (T,{width})")
        if len(index) and ((index < 0).any() or index.max() >= pos.shape[1]):
            raise ValueError(f"{path}: {key} index outside pos")
    return Material(path.stem, pos, ids, arrays.get(DYNAMICS_FACES_KEY), arrays.get(DYNAMICS_TETS_KEY))


class World:
    """One world directory loaded for geometry scoring."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.video_path = self.root / VIDEO_FILENAME
        frames, height, width, self.fps = video_stream(self.video_path)
        self.shape = (frames, height, width)

    @cached_property
    def camera_data(self) -> tuple[np.ndarray, np.ndarray]:
        from .validation import _check_camera

        camera, errors = _check_camera(self.root / CAMERA_FILENAME)
        if errors:
            raise ValueError("; ".join(errors))
        return camera

    @property
    def intrinsics(self) -> np.ndarray:
        return self.camera_data[0]

    @property
    def extrinsics(self) -> np.ndarray:
        return self.camera_data[1]

    @cached_property
    def meshes(self) -> dict[int, dict[int, tuple[np.ndarray, np.ndarray]]]:
        return {oid: read_meshes(path, self.frames) for oid, path in archive_paths(self.root).items()}

    @cached_property
    def ids(self) -> np.ndarray:
        return np.asarray(list(self.meshes), dtype=OBJECT_ID_DTYPE)

    @cached_property
    def materials(self) -> list[Material]:
        """Every `dynamics/` archive, plus the liquid tracers when `solver/` supplies some.

        Also validates `solver/` and the static objects; raises `ValueError` on a violation.
        """

        from .validation import _check_solver, _check_static

        directory = self.root / DYNAMICS_DIRNAME
        if not directory.is_dir():
            raise ValueError(f"{directory}: is required")
        materials = [read_material(path, set(self.meshes), self.frames)
                     for path in sorted(directory.glob(f"*{NPZ_SUFFIX}"))]
        # tracers are extracted only for liquid ids that no dynamics/ archive lists
        moving = {int(oid) for material in materials for oid in material.ids}
        liquid, errors = _check_solver(self.root, set(self.meshes), moving)
        if errors:
            raise ValueError("; ".join(errors))
        liquid -= moving
        if liquid:
            from .liquid import prepare

            path = prepare(self.root)
            tracers = read_material(path, set(self.meshes), self.frames)
            if not liquid <= set(map(int, tracers.ids)) or tracers.pos.shape[1] > TRACER_POINTS:
                raise ValueError(f"{path}: tracer ids or count disagree with the solver contract")
            tracers = replace(tracers, ids=np.asarray(sorted(liquid), dtype=INDEX_DTYPE))
            materials.append(tracers)
        dynamic_ids = (np.unique(np.concatenate([m.ids for m in materials]))
                       if materials else np.zeros(0, dtype=INDEX_DTYPE))
        for oid in sorted(set(self.meshes) - set(map(int, dynamic_ids))):
            errors = _check_static(oid, self.meshes[oid])
            if errors:
                raise ValueError("; ".join(errors))
        return materials

    @cached_property
    def dynamic_ids(self) -> np.ndarray:
        materials = self.materials
        return (np.unique(np.concatenate([m.ids for m in materials]))
                if materials else np.zeros(0, dtype=INDEX_DTYPE))

    def __repr__(self) -> str:
        return f"World({self.root}, F={self.frames}, objects={len(self.ids)})"

    def __len__(self) -> int:
        return self.frames

    @property
    def frames(self) -> int:
        return self.shape[0]

    @property
    def resolution(self) -> tuple[int, int]:
        """The pixel resolution as `(W, H)`."""

        return self.shape[2], self.shape[1]


    def object_mesh(self, oid: int, frame: int) -> tuple[np.ndarray, np.ndarray]:
        """One object's `(verts, faces)` at one frame, empty when absent."""

        return self.meshes[int(oid)].get(int(frame), (EMPTY_VERTS, EMPTY_FACES))

    def parts(self, frame: int) -> list[tuple[int, np.ndarray, np.ndarray]]:
        """Every object present at one frame, as `(id, verts, faces)`."""

        found = []
        for oid in self.ids:
            verts, faces = self.object_mesh(int(oid), frame)
            if len(faces):
                found.append((int(oid), verts, faces))
        return found

    def mesh(self, frame: int) -> tuple[np.ndarray, np.ndarray]:
        """The whole frame as one anonymous triangle soup."""

        parts = self.parts(frame)
        if not parts:
            return EMPTY_VERTS, EMPTY_FACES
        verts, faces, base = [], [], 0
        for _, v, f in parts:
            verts.append(v)
            faces.append(f + base)
            base += len(v)
        return np.concatenate(verts), np.concatenate(faces)

    @cached_property
    def index(self) -> np.ndarray:
        """Per-pixel object ids of the whole take, rasterised by the scorer's own z-buffer."""

        from .raster import id_frames

        return id_frames(self)

    def camera(self, points: np.ndarray) -> np.ndarray:
        """World metres into the camera frame: `R^T (p - t)`."""

        return (np.asarray(points, dtype=np.float64) - self.extrinsics[:3, 3]) @ self.extrinsics[:3, :3]



__all__ = ["Material", "World", "archive_paths", "read_material",
           "read_meshes", "video_stream"]
