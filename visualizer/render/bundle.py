"""The viewer's bundle: the point metrics' arrays, thinned and packed for WebGL.

A bundle is a manifest (`bundle.json`) plus one binary blob (`bundle.bin`) the
browser maps to buffer attributes. Points are anonymous; a point's colour is a
scalar of its own path.

A scored pair's clouds and trajectory paths come from `scene.npz` and
`trajectory.npz`, in the registered frame. Each side's world metres map into that
frame by one 4x4 matrix built from the recorded alignment; the manifest carries the
matrix per side. The registered frame is the reference's frame-0 camera, rotated by
that camera's rotation so the scene stands Z up.

A world without a scored pair (a reference world, or a run on a case with no
reference world) is bundled from its own material, in its own metres: its frame-0
cloud and paths sampled by the scorer's rules (voxel calibration, per-birth-frame
draw, camera culling) with itself as reference. A world with no visible material at
frame 0 has no unit of length and gets no paths.

Clouds and paths are thinned with fixed seeds; nothing here is a metric value.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from config.estimates import MOGE_FILENAME
from config.scorer import N

from scorer.metrics.cloud import cloud, extent
from scorer.metrics.trajectory import _paths
from scorer.metrics.uni3d import moge_cloud
from scorer.samples import Unit, _voxel_size
from scorer.world import World

from ..blob import Blob

from .. import Target

MAX_PATHS = 2048     # paths per side
MAX_CLOUD = 65536    # points per scene cloud
THIN_SEED = 0        # thinning seed
SAMPLE_SEED = 0      # sampling seed for a world without a scored pair


def thin(count: int, limit: int = MAX_PATHS, seed: int = THIN_SEED) -> np.ndarray:
    """Return `limit` sorted indices out of `count`, or all of them, with a fixed seed."""

    if count <= limit:
        return np.arange(count)
    return np.sort(np.random.default_rng(seed).choice(count, limit, replace=False))


def upright(reference: World) -> np.ndarray:
    """The reference camera's rotation, which stands the registered frame Z up."""

    return np.asarray(reference.extrinsics[:3, :3], dtype=np.float64)


def placement(side: str, world: World, alignment: dict, up: np.ndarray) -> np.ndarray:
    """The 4x4 from one side's world metres into the registered frame, rotated by `up`.

    Composes `scorer.metrics.registration.Registration` as one matrix: into the
    world's frame-0 camera, through the fitted similarity (submission only), centred
    and divided by the reference cloud's radius, then `up`.
    """

    stand = np.eye(4)
    stand[:3, :3] = up
    pose = world.extrinsics
    camera = np.eye(4)
    camera[:3, :3] = pose[:3, :3].T
    camera[:3, 3] = -pose[:3, :3].T @ pose[:3, 3]

    unit = np.eye(4)
    unit[:3, 3] = -np.asarray(alignment["centre"], dtype=np.float64)
    unit[:3] /= alignment["radius"]
    if side == "gt":
        return stand @ unit @ camera

    similarity = np.eye(4)
    similarity[:3, :3] = alignment["scale"] * np.asarray(alignment["rotation"], dtype=np.float64)
    similarity[:3, 3] = np.asarray(alignment["offset"], dtype=np.float64)
    return stand @ unit @ similarity @ camera


def objects(world: World) -> list[dict]:
    """The object id table."""

    return [{"id": int(oid)} for oid in world.ids]


def describe(world: World) -> dict:
    return {"frames": int(world.frames), "resolution": list(world.resolution),
            "objects": objects(world)}


def scored(worlds: dict[str, World], results: Path, blob: Blob, detail: dict) -> dict:
    """Build a scored pair's bundle payload from the metrics' arrays."""

    alignment = detail["alignment"]
    up = upright(worlds["gt"])
    matrices = {side: placement(side, world, alignment, up) for side, world in worlds.items()}
    payload: dict = {"pair": True, "sides": ["gt", "pred"], "alignment": alignment,
                     "placement": {side: [float(v) for v in matrix.reshape(-1)]
                                   for side, matrix in matrices.items()}}

    # the two frame-0 clouds Scene3D compares
    with np.load(results / "scene.npz") as data:
        payload["scene"] = {"sides": {}}
        for side in ("gt", "pred"):
            points = data[side]
            keep = thin(len(points), MAX_CLOUD)
            blob.add(f"scene_{side}", points[keep] @ up.T)
            payload["scene"]["sides"][side] = {"points": len(keep), "of": len(points)}

    payload["trajectory"] = trajectory(results, blob, up, tuple(worlds))
    return payload


def trajectory(results: Path, blob: Blob, up: np.ndarray, sides=("gt", "pred")) -> dict | None:
    """Add the trajectory metric's paths from `trajectory.npz`; `None` without paths.

    One blob array per side, `path_<side>` of shape `(P, F, 3)`, NaN outside a
    sample's lifetime or the reference camera's view. The payload carries the rows
    per side, the voxel edge and the matching errors present, in reference radii.
    """

    path = results / "trajectory.npz"
    if not path.is_file():
        return None
    with np.load(path) as data:
        if "gt" not in data.files:
            return None
        payload: dict = {"frames": int(data["gt"].shape[1]), "h": float(data["h"]),
                         "rows": {}, "error": {}}
        payload["error"]["dtw"] = float(data["dtw_error"])
        for side in sides:
            paths = data[side]
            keep = thin(len(paths))
            blob.add(f"path_{side}", (paths[keep] @ up.T).astype(np.float32))
            payload["rows"][side] = len(keep)
    return payload


def sampled_paths(world: World, blob: Blob) -> tuple[int, float | None]:
    """Sample a world's own trajectory paths by the scorer's rules.

    The world is its own reference: material in its camera frame, divided by its
    frame-0 cloud's radius, voxel edge `h` chosen for about `config.scorer.N`
    samples on the fullest frame, samples drawn per birth frame and followed to the
    end, NaN outside the camera's view. Paths are written in world metres. Returns
    the rows written and `h`; `(0, None)` when no material is visible at frame 0.
    """

    paths, h = None, None
    try:
        centre, radius = extent(cloud(world, 0))
    except ValueError:      # nothing visible at frame 0: no unit of length
        unit = None
    else:
        unit = Unit(np.asarray(centre, dtype=np.float64), float(radius))
    if unit is not None and world.materials:
        positions = [unit.gt(world.camera(material.pos)).astype(np.float32)
                     for material in world.materials]
        h = _voxel_size(world, positions, N, np.random.default_rng([SAMPLE_SEED, 0]))
        if h is not None:
            drawn = _paths(world, positions, h, unit, world,
                           np.random.default_rng([SAMPLE_SEED, 1]))
            if len(drawn):
                pose = world.extrinsics
                paths = unit.metres(drawn) @ pose[:3, :3].T + pose[:3, 3]

    if paths is None:
        blob.add("path_pred", np.zeros((0, world.frames, 3), dtype=np.float32))
        return 0, None

    rows = thin(len(paths))
    blob.add("path_pred", np.ascontiguousarray(paths[rows], dtype=np.float32))
    return len(rows), float(h)


def moge_scene(estimates: Path, world: World, blob: Blob) -> dict | None:
    """Add the MoGe point cloud in the world's metres, scaled as `uni3d_moge_scene` does.

    The estimate is centred, divided by its own radius and scaled by the radius of
    the world's frame-0 cloud. `None` without a point map, or when either cloud is empty.
    """

    path = Path(estimates) / MOGE_FILENAME
    if not path.is_file():
        return None
    points = moge_cloud(estimates)
    try:
        theirs = extent(points)
        ours = extent(cloud(world, 0))
    except ValueError:      # empty estimate or nothing visible
        return None
    unit = Unit(np.asarray(ours[0], dtype=np.float64), float(ours[1]))
    scaled = unit.metres((points - theirs[0]) / theirs[1])
    pose = world.extrinsics
    placed = scaled @ pose[:3, :3].T + pose[:3, 3]
    keep = thin(len(placed), MAX_CLOUD)
    blob.add("scene_moge", placed[keep].astype(np.float32))
    return {"points": len(keep), "of": len(points)}


def alone(world: World, blob: Blob, estimates: Path | None = None) -> dict:
    """Bundle a world without a scored pair, in world metres.

    Holds its frame-0 cloud, its sampled paths and, when the case has one, the MoGe cloud.
    """

    pose = world.extrinsics
    try:
        points = cloud(world, 0) @ pose[:3, :3].T + pose[:3, 3]
    except ValueError:      # nothing visible at frame 0
        points = np.zeros((0, 3), dtype=np.float32)
    keep = thin(len(points), MAX_CLOUD)
    blob.add("scene_pred", points[keep])
    sides = {"pred": {"points": len(keep), "of": len(points)}}
    if estimates is not None:
        moge = moge_scene(estimates, world, blob)
        if moge is not None:
            sides["moge"] = moge
    rows, h = sampled_paths(world, blob)
    return {
        "pair": False, "sides": ["pred"], "alignment": None, "placement": None,
        "scene": {"sides": sides},
        "trajectory": {"frames": int(world.frames), "h": h,
                       "rows": {"pred": rows}, "error": {}},
    }


def has_results(results: Path) -> bool:
    return all((results / name).is_file() for name in ("scene.npz", "reward.detail.json"))


def build(target: Target) -> tuple[Path, Path]:
    """Write `bundle.json` and `bundle.bin`. The server adds the pair's names."""

    world = World(target.pred)
    blob = Blob()
    scoring = target.gt is not None and target.results is not None and has_results(target.results)
    if scoring:
        detail = json.loads((target.results / "reward.detail.json").read_text(encoding="utf-8"))
        worlds = {"gt": World(target.gt), "pred": world}
        payload = scored(worlds, target.results, blob, detail)
    else:
        payload = alone(world, blob, target.estimates)
    payload.update({
        "frames": {"pred": int(world.frames)},
        "resolution": {"pred": list(world.resolution)},
        "objects": {"pred": objects(world)},
    })
    if payload["pair"]:
        side = describe(worlds["gt"])
        payload["frames"]["gt"] = side["frames"]
        payload["resolution"]["gt"] = side["resolution"]
        payload["objects"]["gt"] = side["objects"]
    payload["arrays"] = blob.manifest

    target.out.mkdir(parents=True, exist_ok=True)
    manifest_path, blob_path = target.out / "bundle.json", target.out / "bundle.bin"
    blob_path.write_bytes(blob.bytes())
    manifest_path.write_text(json.dumps(payload))
    return manifest_path, blob_path


def render(target: Target) -> str:
    build(target)
    return "bundle"


__all__ = ["Blob", "Unit", "build", "has_results", "objects",
           "placement", "render", "sampled_paths", "thin", "trajectory", "upright"]
