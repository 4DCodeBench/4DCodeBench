"""Metric scheduling and the `reward.json` / `reward.detail.json` writers.

Metrics are grouped into types (`TYPES`); each type owns a set of reward keys and,
through `OWNED_PREFIXES`, the detail keys it writes. Rescoring some types rewrites
only the keys those types own and keeps the rest of both files.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from config.estimates import REWARD_FILENAME
from config.models import GEOPHYS_MODELS

from .execution import TaskFailure, Timeouts, Worker
from .outcome import (
    DEGENERATE,
    ERROR,
    EXCLUDED,
    METRIC_FAILURE,
    NOT_APPLICABLE,
    UNCOMPUTABLE,
    applies,
    error_cause,
    exclusion,
    uncomputable_cause,
)
from .paths import Entry
from .storage import write_dense

SEMANTIC_KEYS = ("semantic_dinov3", "semantic_tips")
# `uni3d_point_scene` needs a reference world, `uni3d_moge_scene` the MoGe estimate
UNI3D_KEYS = ("uni3d_point_scene", "uni3d_moge_scene")
MESH_KEYS = ("mesh_watertight", "mesh_manifold", "mesh_clean_faces", "mesh_no_self_intersection")
# one GeoPhys key per backbone in `GEOPHYS_MODELS`
GEOPHYS_VIDEO_KEYS = tuple(f"geophys_{key}" for key in GEOPHYS_MODELS.values())
# reward key -> the detail key holding its number
REWARD_KEYS: dict[str, str] = {
    "dynamic_iou": "dynamic_iou",
    "scene_3d": "scene_score",
    "trajectory_dtw": "trajectory_dtw_score",
    "emd_step": "emd_step_score",
    "occupancy_dtw": "occupancy_dtw_score",
    "interpenetration": "interpenetration",
    **{key: key for key in MESH_KEYS},
    **{key: key for key in UNI3D_KEYS},
    "flow_distribution": "flow_distribution_score",
    "track2d_dtw": "track2d_dtw_score",
    **{key: key for key in SEMANTIC_KEYS},
    **{key: f"{key}_score" for key in GEOPHYS_VIDEO_KEYS},
    "depth_error": "depth_error",
}
WORLD_KEYS = ("interpenetration", *MESH_KEYS)   # computed from the submitted world alone
# cosine readings, reported on [0, 1] as (1 + cos) / 2; reward.detail.json keeps the cosine
COSINE_KEYS = (*SEMANTIC_KEYS, *UNI3D_KEYS)
# metric type (the values of `--type`) -> the reward keys it owns
TYPES: dict[str, tuple[str, ...]] = {
    "dynamic_iou": ("dynamic_iou",),
    "scene_3d": ("scene_3d",),
    "trajectory": ("trajectory_dtw",),
    "dynamics": ("emd_step",),
    "occupancy": ("occupancy_dtw",),
    "interpenetration": ("interpenetration",),
    "mesh": MESH_KEYS,
    "uni3d": UNI3D_KEYS,
    "flow": ("flow_distribution",),
    "track2d": ("track2d_dtw",),
    "semantic": SEMANTIC_KEYS,
    "geophys": GEOPHYS_VIDEO_KEYS,
    "depth": ("depth_error",),
}


# metric type or preparation step -> prefixes of the detail keys it owns
OWNED_PREFIXES: dict[str, tuple[str, ...]] = {
    "dynamic_iou": ("dynamic_",),
    "scene_3d": ("scene_",),
    "trajectory": ("trajectory",),
    "dynamics": ("emd_", "dynamics_"),
    "occupancy": ("occupancy",),
    "interpenetration": ("interpenetration",),
    "mesh": ("mesh",),
    "uni3d": ("uni3d_",),
    "flow": ("flow",),
    "track2d": ("track2d",),
    "semantic": ("semantic_",),
    "geophys": ("geophys_dinov",),
    "depth": ("depth_", "fps"),
}


RENDER_TYPES = {"semantic", "geophys"}   # computed from render.mp4 alone
GEOMETRY_TYPES = set(TYPES) - RENDER_TYPES
OWNED_PREFIXES.update({"registration": ("alignment", "registration_"),
                       "geometry": ("geometry_",), "reference": ("reference_",), "sampling": ("sampling_",), "paths": ("paths_",)})
for key in SEMANTIC_KEYS:
    OWNED_PREFIXES[key] = (key,)
for tag in GEOPHYS_MODELS.values():
    OWNED_PREFIXES[f"geophys_{tag}"] = (f"geophys_{tag}",)


@dataclass
class Scored:
    """One world being scored: the types still to compute and its merged detail."""

    entry: Entry
    wanted: set[str]
    detail: dict[str, Any] = field(default_factory=dict)
    failed: bool = False

    @property
    def readable(self) -> bool:
        """Whether the video gate passed."""
        return bool((self.detail.get("video_gate") or {}).get("ok"))


def read_json(path: Path) -> dict:
    return json.loads(path.read_text()) if path.is_file() else {}


def owners(key: str) -> set[str]:
    """Return the types and steps owning detail key `key` (longest matching prefix)."""
    found = [(len(prefix), name) for name, prefixes in OWNED_PREFIXES.items()
             for prefix in prefixes if key.startswith(prefix)]
    longest = max((length for length, _ in found), default=0)
    result = {name for length, name in found if length == longest}
    if result & set(SEMANTIC_KEYS):
        result.add("semantic")
    if any(name.startswith("geophys_dinov") for name in result):
        result.add("geophys")
    return result


def merged(old: dict, new: dict, types) -> dict:
    """Return `old` without the keys `types` own, updated with `new`."""
    return {key: value for key, value in old.items() if not owners(key) & set(types)} | new


FAMILY = {key: name for name, keys in TYPES.items() for key in keys}


def keys_of(name: str) -> tuple[str, ...]:
    if name in TYPES:
        return TYPES[name]
    return (name,) if name in (*SEMANTIC_KEYS, *GEOPHYS_VIDEO_KEYS) else ()


def failure_message(detail: dict) -> str:
    return " ".join(str(v) for k, v in detail.items() if k.endswith("_failure"))


def rewards(item: Scored, types, detail: dict, reported, error: bool, cause: str | None) -> tuple[dict, dict]:
    """Return the reward of every key `types` own, and the cause of each non-number.

    A reward is a number or one of `not_applicable`, `excluded`, `error`, `uncomputable`,
    in that precedence. Cosine keys are mapped to `(1 + cos) / 2`.
    """

    owned = {key for name in item.wanted for key in TYPES[name]}
    keys = [key for name in types for key in keys_of(name) if key in owned]
    reward, causes = {}, {}

    def unavailable(key):
        reward[key] = UNCOMPUTABLE
        causes[key] = uncomputable_cause(key, FAMILY[key], detail)

    for key in keys:
        if not applies(key, item.entry.case):
            reward[key] = NOT_APPLICABLE
        elif exclusion(key, item.entry.case):
            reward[key], causes[key] = EXCLUDED, exclusion(key, item.entry.case)
        elif error:
            reward[key], causes[key] = ERROR, cause or error_cause(failure_message(detail))
        elif key not in reported:
            unavailable(key)
        else:
            reading = detail.get(REWARD_KEYS[key])
            if reading == ERROR:
                reward[key], causes[key] = ERROR, cause or error_cause(failure_message(detail))
            elif reading is None:
                unavailable(key)
            elif not np.isfinite(reading):
                reward[key], causes[key] = ERROR, DEGENERATE
            else:
                reward[key] = (1 + float(reading)) / 2 if key in COSINE_KEYS else float(reading)
    return reward, causes


def write_result(item: Scored, types, detail: dict, reported=(), error: bool = False,
                 cause: str | None = None) -> None:
    """Merge the readings of `types` into `reward.json` and `reward.detail.json`."""

    path = item.entry.results / REWARD_FILENAME
    detail_path = detail_path_for(path)
    readings, causes = rewards(item, types, detail, reported, error, cause)
    reward = merged(read_json(path), readings, types)
    previous = read_json(detail_path)
    # `causes` keeps one cause per key whose reward is not a number
    detail["causes"] = {key: why for key, why in ({**previous.get("causes", {}), **causes}).items()
                        if reward.get(key) in {ERROR, UNCOMPUTABLE, EXCLUDED}}
    if "video_gate" in detail and detail["video_gate"]["ok"]:
        previous.pop("gate_failure", None)
    if "semantic_per_frame" in previous or "semantic_per_frame" in detail:
        held = {name: value for name, value in previous.get("semantic_per_frame", {}).items() if name not in types}
        detail["semantic_per_frame"] = held | detail.get("semantic_per_frame", {})
    write_json(path, reward)
    write_json(detail_path, merged(previous, detail, types))
    item.detail = merged(item.detail, detail, types)

def failure(item: Scored, name: str, error) -> None:
    item.failed = True
    write_result(item, {name}, {f"{name}_failure": str(error)}, error=True)
    print(f"{item.entry}: {name}: {error}", file=sys.stderr, flush=True)


def gate(entry: Entry, wanted: set[str], worker: Worker, limits: Timeouts) -> Scored:
    """Run the video and world gates and write `error` for the types a closed gate blocks.

    A closed video gate blocks every type; a closed world gate blocks `GEOMETRY_TYPES`.
    """

    item = Scored(entry, wanted)
    try:
        dimensions, video = worker.call("video_gate", entry, timeout=limits.seconds("gate", True))
    except TaskFailure as error:
        dimensions = None
        video = {"ok": False, "error": METRIC_FAILURE, "detail": [str(error)]}

    world = None
    if video["ok"]:
        try:
            world = worker.call("world_gate", entry, timeout=limits.seconds("gate", True))
        except TaskFailure as error:
            world = {"ok": False, "error": METRIC_FAILURE, "detail": [str(error)]}

    closed = video if not video["ok"] else world if not world["ok"] else None
    refused = set() if closed is None else wanted if closed is video else wanted & GEOMETRY_TYPES
    # `status` is the gate verdict: "ok" or "gate_failure"
    detail = {"dimensions": dimensions, "reference": str(entry.gt) if entry.gt.is_dir() else None,
              "checkpoints": str(entry.roots.checkpoints),
              "video_gate": video, "world_gate": world,
              "status": "ok" if closed is None else "gate_failure"}
    if closed is not None:
        detail["gate_failure"] = f"{closed['error']}: {'; '.join(closed['detail'])[:2000]}"
    item.failed = bool(refused)
    if refused:
        write_result(item, refused, detail, error=True, cause=closed["error"])
    else:
        write_result(item, set(), detail)
    # types with no key scored on this case are written `not_applicable` / `excluded` only
    unscored = {name for name in wanted - refused if keys_of(name) and not any(
        applies(key, entry.case) and not exclusion(key, entry.case) for key in keys_of(name))}
    if unscored:
        write_result(item, unscored, {})
    item.wanted = wanted - refused - unscored
    print(f"{entry}: {detail['status']}" + (f" ({closed['error']})" if refused else ""), flush=True)
    return item


def write_json(path: str | Path, payload: dict) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(target)


def write_arrays(out_dir, arrays) -> list[Path]:
    """Write each `{group: {name: array}}` atomically as `<group>.h5` or `<group>.npz`."""
    directory = Path(out_dir)
    directory.mkdir(parents=True, exist_ok=True)
    written = []
    for group, values in arrays.items():
        dense = group in {"flow", "raster_depth"}
        target = directory / f"{group}.{'h5' if dense else 'npz'}"
        temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
        if dense:
            write_dense(temporary, **values)
        else:
            uncompressed = group in {"uni3d", "sampling"} or group.startswith(("embeddings_", "geophys_"))
            save = np.savez if uncompressed else np.savez_compressed
            with temporary.open("wb") as handle:
                save(handle, **values)
        temporary.replace(target)
        written.append(target)
    return written


def detail_path_for(out) -> Path:
    path = Path(out)
    return path.with_name(f"{path.stem}.detail.json")


def commit(item, name, result):
    detail, reported = result
    write_result(item, {name}, detail, reported)
    print(f"{item.entry}: {name}: done", flush=True)


def geometry_pass(items: list[Scored], limits: Timeouts) -> None:
    """Compute the geometry types of each world in one `GeometryTasks` worker per world.

    Runs each type's preparation steps once per world; a failed step fails every type
    that needs it, and a dead worker fails the remaining types.
    """
    from .tasks import GeometryTasks

    for item in items:
        if not item.wanted & GEOMETRY_TYPES:
            continue
        write_result(item, {"geometry"}, {})
        try:
            with Worker(GeometryTasks, limits.seconds("worker", True)) as worker:
                worker.call("select", item.entry, item.detail["dimensions"], timeout=limits.seconds("geometry", True))
                prepared, blocked = set(), {}
                names = [name for name in TYPES if name in GEOMETRY_TYPES]
                for index, name in enumerate(names):
                    if name not in item.wanted:
                        continue
                    try:
                        needs = worker.call("prerequisites", name, timeout=limits.seconds("geometry", True))
                        for step in needs:
                            if step in blocked:
                                raise TaskFailure(blocked[step])
                            if step not in prepared:
                                try:
                                    detail = worker.call("prepare", step, timeout=limits.seconds(step, True))
                                except TaskFailure as error:
                                    blocked[step] = str(error)
                                    failure(item, step, error)
                                    raise
                                prepared.add(step)
                                write_result(item, {step}, detail)
                        if name == "uni3d":
                            worker.call("clouds", timeout=limits.seconds("sampling", True))
                        else:
                            result = worker.call("metric", name, timeout=limits.seconds(name))
                            commit(item, name, result)
                    except TaskFailure as error:
                        failure(item, name, error)
                        if not worker.alive:
                            failure(item, "geometry", f"interrupted during {name}")
                            for rest in names[index + 1:]:
                                if rest in item.wanted:
                                    failure(item, rest, f"interrupted during {name}")
                            break
        except TaskFailure as error:
            item.failed = True
            write_result(item, (GEOMETRY_TYPES & item.wanted) | {"geometry", "registration", "sampling", "paths"},
                         {"geometry_failure": str(error)}, error=True)
            print(f"{item.entry}: geometry: {error}", file=sys.stderr, flush=True)


def model_pass(items, stage, owner, limits):
    """Compute one model-backed `stage` for every world whose video gate passed.

    Reuses stored arrays when present, loads the model once, and fails every
    remaining world after a failed load.
    """
    from .tasks import ModelTasks

    worker = None
    unavailable = None
    try:
        for item in items:
            if not item.readable:
                continue
            try:
                if worker is None:
                    worker = Worker(ModelTasks, limits.seconds("worker", True))
                if unavailable:
                    raise TaskFailure(unavailable)
                if not worker.call("loaded", timeout=limits.seconds("model_load", True)):
                    try:
                        print(f"{stage}: loading model", flush=True)
                        worker.call("load", stage, item.entry.roots.checkpoints,
                                    timeout=limits.seconds("model_load", True))
                    except TaskFailure as error:
                        unavailable = f"model load failed: {error}"
                        raise
                result = worker.call("metric", stage, item.entry, item.detail["dimensions"],
                                     timeout=limits.seconds(owner))
                commit(item, owner, result)
            except TaskFailure as error:
                failure(item, owner, error)
                if worker is not None and not worker.alive:
                    worker.close()
                    worker = None
    finally:
        if worker is not None:
            worker.close()


def semantic_pass(items, limits):
    for name in SEMANTIC_KEYS:
        model_pass([item for item in items if "semantic" in item.wanted], name, name, limits)


def geophys_pass(items, limits):
    for tag in GEOPHYS_MODELS.values():
        name = f"geophys_{tag}"
        model_pass([item for item in items if "geophys" in item.wanted], name, name, limits)


def uni3d_pass(items, limits):
    selected = [item for item in items if "uni3d" in item.wanted
                and "uni3d_failure" not in item.detail and "geometry_failure" not in item.detail]
    model_pass(selected, "uni3d", "uni3d", limits)
