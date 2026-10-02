"""Worker-process tasks: geometry preparation and metrics, and model-backed metrics.

`GeometryTasks` and `ModelTasks` run inside a `Worker`; each method returns picklable
results to `scorer.score`. Each metric writes its arrays to the run's `results/`;
`uni3d_clouds.npz` carries the frame-0 clouds from the geometry worker to the model worker.
"""

from __future__ import annotations

from functools import cached_property

import numpy as np

from config.annotation import MASK_FILENAME
from config.estimates import (
    DEPTH_FILENAME,
    FLOW_FILENAME,
    GEOPHYS_FILENAME,
    MOGE_FILENAME,
    TRACKS_FILENAME,
)
from config.models import GEOPHYS_MODELS, READOUT, block_for
from config.world import VIDEO_FILENAME

from .score import REWARD_KEYS, TYPES, UNI3D_KEYS, write_arrays

REFERENCE_TYPES = {"scene_3d", "trajectory", "dynamics", "occupancy"}   # need a reference world
CLOUDS_FILENAME = "uni3d_clouds.npz"


def finish(entry, reading):
    """Write the reading's `arrays` to `results/` and return the rest."""
    write_arrays(entry.results, reading.pop("arrays", {}))
    return reading


def report(name, reading):
    """Return `(reading, reported keys)`; Flow, Track2D and Depth report all keys or none."""
    keys = TYPES.get(name, (name,))
    if name in {"flow", "track2d", "depth"}:
        return reading, list(keys) if reading else []
    return reading, [key for key in keys if REWARD_KEYS[key] in reading]


def uni3d_report(entry, reading):
    """Return the Uni3D reading and its keys, noting a missing reference as a failure."""
    present = {"point": entry.gt.is_dir(), "moge": (entry.estimates / MOGE_FILENAME).is_file()}
    keys = [key for key in UNI3D_KEYS if present["point"] or key != "uni3d_point_scene"]
    for key in keys:
        tag = key.split("_")[1]
        if reading.get(key) is None:
            reading[key] = None
            if not present[tag]:
                reading[f"uni3d_{tag}_failure"] = f"{tag} reference missing"
    return reading, keys


class GeometryTasks:
    """Gates and geometry metrics of one world, with the loaded worlds cached per world."""

    def video_gate(self, entry):
        from .gate import video_gate
        return video_gate(entry.world / VIDEO_FILENAME, entry.video)

    def world_gate(self, entry):
        from .gate import world_gate
        return world_gate(entry.world.parent)

    def select(self, entry, dimensions):
        self.entry, self.dimensions = entry, dimensions

    @cached_property
    def pred(self):
        from .world import World
        return World(self.entry.world)

    @cached_property
    def gt(self):
        from .world import World
        return World(self.entry.gt) if self.entry.gt.is_dir() else None

    @cached_property
    def alignment(self):
        from .metrics.registration import Registration, register
        data = register(self.gt, self.pred).as_dict()   # plain float64 values, as recorded in the detail
        return Registration(**{key: np.asarray(data[key]) if key in {"centre", "rotation", "offset"}
                               else data[key] for key in Registration.__dataclass_fields__})

    @cached_property
    def sampling(self):
        from .samples import material
        return material(self.gt, self.pred, self.alignment)

    @cached_property
    def paths(self):
        from .metrics.trajectory import trajectory_paths
        return trajectory_paths(self.gt, self.pred, self.alignment, self.sampling)

    def prepare(self, name):
        if name == "geometry":
            _ = self.pred.meshes
        elif name == "reference":
            _ = self.gt
        elif name == "registration":
            return {"alignment": self.alignment.as_dict()}
        elif name == "sampling":
            _ = self.sampling
        elif name == "paths":
            _ = self.paths
        return {}

    def prerequisites(self, name):
        """Return the preparation steps type `name` needs, in order."""
        reference = self.entry.gt.is_dir()
        if (name in REFERENCE_TYPES and not reference) or name == "depth":
            return []
        tasks = ["geometry"]
        if reference and name in {*REFERENCE_TYPES, "dynamic_iou", "uni3d"}:
            tasks.append("reference")
            if name != "dynamic_iou":
                tasks.append("registration")
            if name in {"trajectory", "dynamics", "occupancy"}:
                tasks.append("sampling")
            if name in {"trajectory", "dynamics"}:
                tasks.append("paths")
        return tasks

    def metric(self, name):
        reference = {"flow": FLOW_FILENAME, "track2d": TRACKS_FILENAME}.get(name)
        if reference is not None and not (self.entry.estimates / reference).is_file():
            return {}, []
        reading = finish(self.entry, self.calculate(name))
        # these types report every key they own
        keys = list(TYPES[name]) if name in {"mesh", "interpenetration"} or (
            self.entry.gt.is_dir() and name in {"dynamic_iou", *REFERENCE_TYPES}) else report(name, reading)[1]
        return reading, keys

    def calculate(self, name):
        entry = self.entry
        if name == "depth":
            from .metrics.depth import depth_error
            if not (entry.estimates / DEPTH_FILENAME).is_file():
                raise FileNotFoundError(entry.estimates / DEPTH_FILENAME)
            return depth_error(entry.world, entry.estimates)
        if name == "mesh":
            from .metrics.mesh import mesh_quality
            return mesh_quality(self.pred)
        if name == "interpenetration":
            from .metrics.interpenetration import interpenetration
            return interpenetration(self.pred)
        if name == "flow":
            return analytic_flow(entry, self.pred, self.dimensions)
        if name == "track2d":
            return analytic_tracks(entry, self.pred, self.dimensions)
        if name == "dynamic_iou":
            from .metrics.dynamic import Annotation, annotated_dynamic_iou, dynamic_iou
            if entry.gt.is_dir():
                return dynamic_iou(self.gt, self.pred)
            if (entry.annotation / MASK_FILENAME).is_file():
                return annotated_dynamic_iou(Annotation(entry.annotation), self.pred)
            return {}
        if not entry.gt.is_dir():
            return {}
        if name == "scene_3d":
            from .metrics.error import scene_error
            return scene_error(self.gt, self.pred, self.alignment)
        if name == "trajectory":
            from .metrics.trajectory import trajectory_reading
            return trajectory_reading(self.paths)
        if name == "dynamics":
            from .metrics.dynamics import dynamics_readings
            return dynamics_readings(self.paths.get("gt"), self.paths.get("pred"))
        if name == "occupancy":
            from .metrics.occupancy import occupancy_error
            return occupancy_error(self.gt, self.pred, self.alignment, self.sampling)
        raise ValueError(name)

    def clouds(self):
        """Write `uni3d_clouds.npz`: the frame-0 clouds the Uni3D metrics embed."""

        from .samples import lone_scene, scene_unit
        if self.entry.gt.is_dir():
            # the registered frame-0 clouds of Scene3D
            from .metrics.error import scene_error
            scene = scene_error(self.gt, self.pred, self.alignment)["arrays"]["scene"]
            a = {"submission_scene": scene["pred"], "point_scene": scene["gt"]}
        else:
            a = {"submission_scene": lone_scene(self.pred, scene_unit(self.pred))}
        write_arrays(self.entry.results, {CLOUDS_FILENAME.removesuffix(".npz"): a})


def clouds(entry):
    """Return the submission's frame-0 cloud and the reference world's, or None."""

    with np.load(entry.results / CLOUDS_FILENAME) as a:
        return a["submission_scene"], a["point_scene"] if "point_scene" in a.files else None


class ModelTasks:
    """Model-backed metrics (Semantic, GeoPhys, Uni3D) with one model loaded at a time."""

    def __init__(self):
        self.model = None

    def loaded(self):
        return self.model is not None

    def load(self, stage, root):
        self.root = root
        if stage.startswith("semantic_"):
            from .estimators import backbones
            from .metrics.semantic import MODELS
            self.model = MODELS[stage]
            self.model.load(backbones.DEVICE, root)
        elif stage.startswith("geophys_"):
            from .estimators import backbones
            self.model = {"geophys_dinov3": backbones.DINOV3}[stage]
            self.model.load(backbones.DEVICE, root)
        elif stage == "uni3d":
            from .metrics.uni3d import point_tower
            self.model = point_tower(root)
        else:
            raise ValueError(stage)

    def metric(self, stage, entry, dimensions):
        if stage.startswith("semantic_"):
            from .metrics.semantic import render_reading, short
            render, curve, stride = render_reading(stage, self.model, entry.world, entry.estimates, self.root)
            write_arrays(entry.results, {f"embeddings_{short(stage)}": {
                "render": render.astype(np.float32), "cosine": curve.astype(np.float32)}})
            detail, keys = semantic_result(stage, curve)
            return detail | {"semantic_stride": stride}, keys
        elif stage.startswith("geophys_"):
            from .metrics.semantic import frame_trajectory
            name = next(name for name, tag in GEOPHYS_MODELS.items() if stage == f"geophys_{tag}")
            tag = GEOPHYS_MODELS[name]
            with np.load(entry.estimates / GEOPHYS_FILENAME.format(name=tag)) as archive:
                reference, stride = archive["trajectory"], int(archive["stride"])
            render = frame_trajectory(self.model, entry.world / VIDEO_FILENAME, block=block_for(name),
                                      root=self.root, stride=stride)
            write_arrays(entry.results, {stage: {"render": render.astype(np.float32),
                                                 "stride": np.int64(stride), "layer": np.int64(READOUT[name])}})
            if len(reference) != len(render):
                raise ValueError("reference and render must have the same frame count")
            from .metrics.geophys import geophys_video_reading
            return report(stage, geophys_video_reading(stage, reference, render, layer=READOUT[name],
                                                       readout="spatial average pooling"))
        elif stage == "uni3d":
            from .metrics.uni3d import cloud_readings, moge_cloud
            submission, point = clouds(entry)
            moge = moge_cloud(entry.estimates) if (entry.estimates / MOGE_FILENAME).is_file() else None
            reading = cloud_readings(submission, point, moge, self.model)
        else:
            raise ValueError(stage)
        if stage == "uni3d":
            return uni3d_report(entry, finish(entry, reading))
        return report(stage, finish(entry, reading))


def analytic_flow(entry, pred, dimensions):
    """Flow of the submission's analytic flow against the reference RAFT flow."""

    from .analytic.flow import frame_flow
    from .metrics.flow import flow_reading
    from .raster import default_device
    device = default_device()

    def submission(step, stride):
        flow, valid, _ = frame_flow(pred, step * stride, device, stride)
        return flow.float(), valid

    return flow_reading(entry.estimates, dimensions[:3], submission, device)


def analytic_tracks(entry, pred, dimensions):
    """Track2D of the submission's surface points followed from the reference queries."""

    from .analytic.tracks import track_points
    from .metrics.track2d import track2d_reading
    return track2d_reading(entry.estimates, dimensions[:3],
                           lambda queries, stride: track_points(pred, queries, stride=stride), entry.video)


def semantic_result(name, curve):
    return {name: float(curve.mean()), "semantic_per_frame": {name: curve.tolist()},
            "semantic_frames": len(curve)}, [name]
