"""World-format validation (`check_world`, `check_links`) and the archive readers built on it."""

from __future__ import annotations

import json
import math
import os
import re
import subprocess
import zipfile
from pathlib import Path

import numpy as np

from .paths import solver_cache

from config.world import (
    CAMERA_EXTRINSIC_KEY,
    CAMERA_FILENAME,
    CAMERA_INTRINSICS_KEY,
    DYNAMICS_DIRNAME,
    DYNAMICS_FACES_KEY,
    DYNAMICS_IDS_KEY,
    DYNAMICS_POS_KEY,
    DYNAMICS_TETS_KEY,
    FACES_DTYPE,
    INDEX_DTYPE,
    MESH_FACES_KEY,
    MESH_VERTICES_KEY,
    MESHES_DIRNAME,
    NPZ_SUFFIX,
    POS_DTYPE,
    VIDEO_FILENAME,
)

Violation = str
UNREADABLE = (OSError, zipfile.BadZipFile, ValueError, KeyError)
MEMBER_RE = re.compile(r"^(vertices|faces)_(\d{4})$")
ORTHONORMAL_TOL = 1e-4      # max |R^T R - I| entry: float32 round-off in a 3x3 product


def _msg(name: str, field: str, text: str) -> str:
    return f"{name}: {field} {text}"


def check_world(root: str | Path) -> list[Violation]:
    """Return every format or internal-consistency violation in one world."""

    root = Path(root)
    violations: list[Violation] = []

    _, errors = _check_camera(root / CAMERA_FILENAME)
    violations.extend(errors)

    frame_count, errors = _check_video(root / VIDEO_FILENAME)
    violations.extend(errors)

    meshes, declared, errors = _check_meshes(root / MESHES_DIRNAME, frame_count)
    violations.extend(errors)
    if meshes and not any(len(faces) for frames in meshes.values() for _, faces in frames.values()):
        violations.append(_msg(MESHES_DIRNAME, "faces", "hold no triangle at any frame"))

    moving, errors = _check_dynamics(root / DYNAMICS_DIRNAME, declared, frame_count)
    violations.extend(errors)

    # The static rule needs every moving id, so it runs only when all of dynamics/ is readable.
    if moving is not None:
        liquid, errors = _check_solver(root, set(meshes), moving)
        violations.extend(errors)
        for oid in sorted(set(meshes) - moving - liquid):
            violations.extend(_check_static(oid, meshes[oid]))
    return violations


def _check_camera(path: Path) -> tuple[tuple[np.ndarray, np.ndarray] | None, list[Violation]]:
    """Return `(intrinsics, extrinsic)` when both are well formed, and the violations."""

    if not path.exists():
        return None, [_msg(CAMERA_FILENAME, "path", "is required")]
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return None, [_msg(CAMERA_FILENAME, "json", str(exc))]
    if not isinstance(payload, dict):
        return None, [_msg(CAMERA_FILENAME, "json", "must be an object")]

    violations: list[Violation] = []
    matrices: dict[str, np.ndarray] = {}
    for key, shape in ((CAMERA_INTRINSICS_KEY, (3, 3)), (CAMERA_EXTRINSIC_KEY, (4, 4))):
        if key not in payload:
            violations.append(_msg(CAMERA_FILENAME, key, "is required"))
            continue
        try:
            matrix = np.asarray(payload[key], dtype=np.float64)
        except (TypeError, ValueError) as exc:
            violations.append(_msg(CAMERA_FILENAME, key, f"is not numeric: {exc}"))
            continue
        if matrix.shape != shape:
            violations.append(_msg(CAMERA_FILENAME, key, f"shape {matrix.shape} is not {shape}"))
        elif not np.isfinite(matrix).all():
            violations.append(_msg(CAMERA_FILENAME, key, "must be finite"))
        else:
            matrices[key] = matrix
    if len(matrices) < 2:
        return None, violations

    rotation = matrices[CAMERA_EXTRINSIC_KEY][:3, :3]
    if (np.abs(rotation.T @ rotation - np.eye(3)).max() > ORTHONORMAL_TOL
            or np.linalg.det(rotation) < 0):
        violations.append(_msg(CAMERA_FILENAME, CAMERA_EXTRINSIC_KEY,
                               "rotation block must be a proper orthonormal matrix"))
        return None, violations
    return (matrices[CAMERA_INTRINSICS_KEY], matrices[CAMERA_EXTRINSIC_KEY]), violations


def _check_video(path: Path) -> tuple[int | None, list[Violation]]:
    """Return F, the frame count ffprobe decodes from render.mp4, and the violations."""

    if not path.exists():
        return None, [_msg(VIDEO_FILENAME, "path", "is required")]
    command = ["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
               "-show_entries", "stream=nb_read_frames", "-of", "json", str(path)]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        return None, [_msg(VIDEO_FILENAME, "ffprobe", result.stderr.strip())]
    try:
        frames = int(json.loads(result.stdout)["streams"][0]["nb_read_frames"])
    except (KeyError, IndexError, ValueError) as exc:
        return None, [_msg(VIDEO_FILENAME, "ffprobe", f"missing frame count: {exc}")]
    return frames, []


def _check_meshes(
    directory: Path, frame_count: int | None
) -> tuple[dict[int, dict[int, tuple[np.ndarray, np.ndarray]]], set[int], list[Violation]]:
    """Return the loaded archives, every id a file name declares, and the violations."""

    if not directory.is_dir():
        return {}, set(), [_msg(MESHES_DIRNAME, "path", "is required")]

    meshes: dict[int, dict[int, tuple[np.ndarray, np.ndarray]]] = {}
    declared: set[int] = set()
    violations: list[Violation] = []
    for path in sorted(directory.iterdir()):
        name = f"{MESHES_DIRNAME}/{path.name}"
        if path.suffix != NPZ_SUFFIX or not path.stem.isdigit() or int(path.stem) <= 0:
            violations.append(_msg(name, "name", "must be <positive id>.npz"))
            continue
        oid = int(path.stem)
        if oid in declared:
            violations.append(_msg(name, "id", f"{oid} is already declared"))
            continue
        declared.add(oid)
        frames, errors = _read_mesh(name, path, frame_count)
        violations.extend(errors)
        if frames is not None:
            meshes[oid] = frames
    return meshes, declared, violations


def _read_mesh(
    name: str, path: Path, frame_count: int | None
) -> tuple[dict[int, tuple[np.ndarray, np.ndarray]] | None, list[Violation]]:
    """Return one mesh archive as `{frame: (vertices, faces)}`, or None and its first violation."""

    try:
        with np.load(path) as data:
            members = list(data.files)
            stray = [key for key in members if not MEMBER_RE.match(key)]
            if stray:
                return None, [_msg(name, stray[0], "is not a vertices_TTTT/faces_TTTT member")]
            vertex_frames = {int(key[-4:]) for key in members if key.startswith("vertices_")}
            face_frames = {int(key[-4:]) for key in members if key.startswith("faces_")}
            if not vertex_frames and not face_frames:
                return None, [_msg(name, "members", "must hold at least one frame")]
            if vertex_frames != face_frames:
                odd = min(vertex_frames ^ face_frames)
                return None, [_msg(name, f"frame {odd:04d}", "must pair vertices with faces")]
            frames = sorted(vertex_frames)
            if frames != list(range(frames[0], frames[-1] + 1)):
                return None, [_msg(name, "frames", "must be one contiguous interval")]
            if frame_count is not None and frames[-1] >= frame_count:
                return None, [_msg(name, f"frame {frames[-1]:04d}",
                                   f"is outside the {frame_count} frame timeline")]
            loaded: dict[int, tuple[np.ndarray, np.ndarray]] = {}
            for frame in frames:
                pair, error = _check_frame_mesh(name, data, frame)
                if error is not None:
                    return None, [error]
                loaded[frame] = pair
    except UNREADABLE as exc:
        return None, [_msg(name, "npz", str(exc))]
    return loaded, []


def _check_frame_mesh(name: str, data, frame: int) -> tuple[tuple[np.ndarray, np.ndarray] | None,
                                                            Violation | None]:
    """Return one frame's `(vertices, faces)` with vertices cast to float32, or its violation."""

    vertices_key, faces_key = MESH_VERTICES_KEY.format(frame=frame), MESH_FACES_KEY.format(frame=frame)
    vertices = np.asarray(data[vertices_key], dtype=POS_DTYPE)
    faces = data[faces_key]
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        return None, _msg(name, vertices_key, f"shape {vertices.shape} is not (V,3)")
    if not np.isfinite(vertices).all():
        return None, _msg(name, vertices_key, "must be finite")
    if faces.dtype != np.dtype(FACES_DTYPE):
        return None, _msg(name, faces_key, f"dtype {faces.dtype} != int32")
    if faces.ndim != 2 or faces.shape[1] != 3:
        return None, _msg(name, faces_key, f"shape {faces.shape} is not (T,3)")
    if len(faces) and ((faces < 0).any() or faces.max() >= len(vertices)):
        return None, _msg(name, faces_key, "index outside the vertex array")
    return (vertices, faces), None


def _check_dynamics(
    directory: Path, mesh_ids: set[int], frame_count: int | None
) -> tuple[set[int] | None, list[Violation]]:
    """Return the ids listed by every archive, or None when one could not be read."""

    if not directory.is_dir():
        return None, [_msg(DYNAMICS_DIRNAME, "path", "is required")]

    moving: set[int] | None = set()
    violations: list[Violation] = []
    for path in sorted(directory.glob(f"*{NPZ_SUFFIX}")):
        ids, errors = _read_dynamics(f"{DYNAMICS_DIRNAME}/{path.name}", path, mesh_ids, frame_count)
        violations.extend(errors)
        if ids is None:
            moving = None
        elif moving is not None:
            moving.update(ids)
    return moving, violations


def _read_dynamics(
    name: str, path: Path, mesh_ids: set[int], frame_count: int | None
) -> tuple[set[int] | None, list[Violation]]:
    """Return the ids this archive lists, or None when they could not be read."""

    try:
        with np.load(path) as data:
            members = set(data.files)
            stray = sorted(members - {DYNAMICS_POS_KEY, DYNAMICS_IDS_KEY,
                                      DYNAMICS_FACES_KEY, DYNAMICS_TETS_KEY})
            if stray:
                return None, [_msg(name, stray[0], "is not a known member")]
            for key in (DYNAMICS_POS_KEY, DYNAMICS_IDS_KEY):
                if key not in members:
                    return None, [_msg(name, key, "is required")]
            if DYNAMICS_FACES_KEY in members and DYNAMICS_TETS_KEY in members:
                return None, [_msg(
                    name,
                    "connectivity",
                    "cannot contain both faces and tets; use faces for a surface mesh or tets for a tetrahedral mesh",
                )]
            arrays = {key: data[key] for key in members}
    except UNREADABLE as exc:
        return None, [_msg(name, "npz", str(exc))]

    violations: list[Violation] = []
    pos = arrays[DYNAMICS_POS_KEY]
    if pos.dtype != np.dtype(POS_DTYPE):
        violations.append(_msg(name, DYNAMICS_POS_KEY, f"dtype {pos.dtype} != float32"))
    elif pos.ndim != 3 or pos.shape[2] != 3:
        violations.append(_msg(name, DYNAMICS_POS_KEY, f"shape {pos.shape} is not (F,N,3)"))
    elif frame_count is not None and pos.shape[0] != frame_count:
        violations.append(_msg(name, DYNAMICS_POS_KEY,
                               f"F {pos.shape[0]} != {VIDEO_FILENAME} {frame_count}"))
    else:
        error = _check_lifetimes(name, pos)
        if error is not None:
            violations.append(error)

    ids = arrays[DYNAMICS_IDS_KEY]
    listed: set[int] | None = None
    if ids.dtype != np.dtype(INDEX_DTYPE):
        violations.append(_msg(name, DYNAMICS_IDS_KEY, f"dtype {ids.dtype} != uint16"))
    elif ids.ndim != 1 or len(ids) < 1:
        violations.append(_msg(name, DYNAMICS_IDS_KEY, f"shape {ids.shape} is not (M,) with M >= 1"))
    else:
        listed = {int(value) for value in ids}
        missing = sorted(listed - mesh_ids)
        if missing:
            violations.append(_msg(name, DYNAMICS_IDS_KEY,
                                   f"id {missing[0]} has no {MESHES_DIRNAME}/ archive"))

    columns = pos.shape[1] if pos.ndim == 3 else None
    for key, width in ((DYNAMICS_FACES_KEY, 3), (DYNAMICS_TETS_KEY, 4)):
        if key not in arrays:
            continue
        index = arrays[key]
        if index.dtype != np.dtype(FACES_DTYPE):
            violations.append(_msg(name, key, f"dtype {index.dtype} != int32"))
        elif index.ndim != 2 or index.shape[1] != width:
            violations.append(_msg(name, key, f"shape {index.shape} is not (T,{width})"))
        elif columns is not None and len(index) and ((index < 0).any() or index.max() >= columns):
            violations.append(_msg(name, key, f"index outside the {columns} columns of {DYNAMICS_POS_KEY}"))
    return listed, violations


def _check_lifetimes(name: str, pos: np.ndarray) -> Violation | None:
    """Every column is all finite or all NaN per frame, over one contiguous non-empty interval."""

    alive = np.isfinite(pos).all(axis=2)
    mixed = ~(alive | np.isnan(pos).all(axis=2))
    if mixed.any():
        frame, column = np.argwhere(mixed)[0]
        return _msg(name, f"{DYNAMICS_POS_KEY}[{frame},{column}]", "must be all finite or all NaN")

    empty = np.flatnonzero(~alive.any(axis=0))
    if len(empty):
        return _msg(name, f"{DYNAMICS_POS_KEY}[:,{int(empty[0])}]", "has empty lifetime")
    padded = np.zeros((len(alive) + 1, alive.shape[1]), dtype=bool)
    padded[1:] = alive
    starts = (padded[1:] & ~padded[:-1]).sum(axis=0)
    broken = np.flatnonzero(starts > 1)
    if len(broken):
        return _msg(name, f"{DYNAMICS_POS_KEY}[:,{int(broken[0])}]", "has non-contiguous lifetime")
    return None


SOLVER_DIRNAME = "solver"
SOLVER_MANIFEST = "manifest.json"
SOLVER_KEYS = ("cache", "fps", "time_scale", "metres_per_unit",
               "domain_min", "frames", "liquid_ids")
SOLVER_POSITIVE = ("fps", "time_scale", "metres_per_unit")   # each a positive finite number
SOLVER_CONFIG_DIRNAME = "config"
SOLVER_CONFIG_NAME = "config_{frame:04d}.uni"
DOMAIN_MIN_LENGTH = 3


def _number(value: object) -> bool:
    """Whether a value is a finite real scalar."""

    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value))


def _check_bake(name: str, cache: Path, span) -> list[Violation]:
    """Require a `config/` record in the bake for every frame of the manifest's `frames` span."""

    if not (isinstance(span, list) and len(span) == 2
            and all(isinstance(v, int) and not isinstance(v, bool) and v >= 0 for v in span)
            and span[0] <= span[1]):
        return []                             # `_check_solver` reports an invalid span
    configs = cache / SOLVER_CONFIG_DIRNAME
    frames = range(span[0], span[1] + 1)
    if frames[0] == 0 and not (configs / SOLVER_CONFIG_NAME.format(frame=0)).is_file():
        frames = range(1, frames[-1] + 2)     # video frame 0 can be Blender cache frame 1
    absent = [frame for frame in frames if not (configs / SOLVER_CONFIG_NAME.format(frame=frame)).is_file()]
    if not absent:
        return []
    more = f" and {len(absent) - 1} more" if len(absent) > 1 else ""
    return [_msg(name, "cache", f"has no {SOLVER_CONFIG_DIRNAME}/ record of frame {absent[0]:04d}{more}")]


def _check_solver(root: Path, mesh_ids: set[int], moving: set[int]) -> tuple[set[int], list[Violation]]:
    """Return the manifest's liquid mesh ids and the violations of `solver/`.

    The bake is checked only when some liquid id is listed by no `dynamics/` archive.
    """

    solver = root / SOLVER_DIRNAME
    if not solver.exists():
        return set(), []                      # solver/ is optional
    name = f"{SOLVER_DIRNAME}/{SOLVER_MANIFEST}"
    path = solver / SOLVER_MANIFEST
    if not path.is_file():
        return set(), [_msg(name, "path", "is required when solver/ exists")]
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return set(), [_msg(name, "json", f"is not readable: {exc}")]
    if not isinstance(manifest, dict):
        return set(), [_msg(name, "json", "must be an object")]

    violations = []
    for key in SOLVER_KEYS:
        if key not in manifest:
            violations.append(_msg(name, key, "is required"))
    cache = manifest.get("cache")
    baked = {oid for oid in manifest.get("liquid_ids") or [] if isinstance(oid, int)} - moving
    if isinstance(cache, str) and not solver_cache(root, cache).is_dir():
        violations.append(_msg(name, "cache", f"names {cache!r}, which is not a directory here"))
    elif isinstance(cache, str) and baked:
        violations.extend(_check_bake(name, solver_cache(root, cache), manifest.get("frames")))

    for key in SOLVER_POSITIVE:
        value = manifest.get(key)
        if key in manifest and not (_number(value) and value > 0):
            violations.append(_msg(name, key, f"must be a positive finite number, not {value!r}"))
    corner = manifest.get("domain_min")
    if "domain_min" in manifest and not (
            isinstance(corner, list) and len(corner) == DOMAIN_MIN_LENGTH
            and all(_number(v) for v in corner)):
        violations.append(_msg(name, "domain_min",
                               f"must be {DOMAIN_MIN_LENGTH} finite numbers, not {corner!r}"))
    span = manifest.get("frames")
    if "frames" in manifest and not (
            isinstance(span, list) and len(span) == 2
            and all(isinstance(v, int) and not isinstance(v, bool) and v >= 0 for v in span)
            and span[0] <= span[1]):
        violations.append(_msg(name, "frames",
                               f"must be two frame numbers, first no later than last, not {span!r}"))

    ids = manifest.get("liquid_ids")
    if not isinstance(ids, list) or not ids or not all(isinstance(v, int) for v in ids):
        violations.append(_msg(name, "liquid_ids", "must be a non-empty list of mesh ids"))
        return set(), violations
    missing = sorted(set(ids) - mesh_ids)
    if missing:
        violations.append(_msg(name, "liquid_ids", f"names ids with no mesh archive: {missing}"))
    return set(ids), violations


def _check_static(oid: int, frames: dict[int, tuple[np.ndarray, np.ndarray]]) -> list[Violation]:
    """An id no dynamics archive lists holds the same arrays at every frame."""

    name = f"{MESHES_DIRNAME}/{oid}{NPZ_SUFFIX}"
    ordered = sorted(frames)
    first_vertices, first_faces = frames[ordered[0]]
    for frame in ordered[1:]:
        vertices, faces = frames[frame]
        if not np.array_equal(vertices, first_vertices):
            return [_msg(name, MESH_VERTICES_KEY.format(frame=frame),
                         f"differs from frame {ordered[0]:04d} but no {DYNAMICS_DIRNAME}/ archive lists id {oid}")]
        if not np.array_equal(faces, first_faces):
            return [_msg(name, MESH_FACES_KEY.format(frame=frame),
                         f"differs from frame {ordered[0]:04d} but no {DYNAMICS_DIRNAME}/ archive lists id {oid}")]
    return []


def check_links(root: str | Path) -> list[Violation]:
    """Report symbolic links in a world directory."""

    root = Path(root)
    if root.is_symlink():
        return [f"{root.name}: must be a directory, not a symbolic link"]
    if not root.is_dir():
        return []
    violations = []
    for directory, subdirectories, files in os.walk(root):
        for entry in sorted(subdirectories + files):
            path = Path(directory) / entry
            if path.is_symlink():
                violations.append(f"{root.name}/{path.relative_to(root)}: symbolic links are not allowed; write the file itself")
    return violations


__all__ = ["check_links", "check_world"]
