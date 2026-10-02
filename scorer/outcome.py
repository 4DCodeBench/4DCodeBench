"""The non-number readings and their causes.

A reading is a number, `not_applicable` (the metric is not computed on this kind
of case), `excluded` (the metric is unsound on this case), `uncomputable` (the
metric applies but the data has no samples to compare) or `error` (a gate is closed
or the submission is defective). `not_applicable` and `excluded` depend on the case
alone and take precedence, in that order, over the states the submission determines;
each `excluded`, `uncomputable` and `error` reading is stored with one of the causes below.
"""

from __future__ import annotations


ERROR = "error"
UNCOMPUTABLE = "uncomputable"
NOT_APPLICABLE = "not_applicable"
EXCLUDED = "excluded"

# causes of `excluded`
TRANSPARENT = "transparent"

# causes of `error`
NO_VIDEO = "no_video"
UNDECODABLE_VIDEO = "undecodable_video"
VIDEO_MISMATCH = "video_mismatch"
NO_WORLD = "no_world"
INVALID_WORLD = "invalid_world"
EMPTY_FIRST_FRAME = "empty_first_frame"
DEGENERATE = "degenerate"
METRIC_FAILURE = "metric_failure"

# causes of `uncomputable`
REFERENCE_EMPTY = "reference_empty"      # the reference side has no samples
SUBMISSION_EMPTY = "submission_empty"    # the submission side has no samples
NO_PAIRING = "no_pairing"                # both sides have samples, too few pair up

# keys that compare against a reference world, which only synthetic cases carry
REFERENCE_WORLD_KEYS = ("scene_3d", "trajectory_dtw", "emd_step", "occupancy_dtw", "uni3d_point_scene")


# keys that compare against estimates of the video, read on real cases only
REAL_KEYS = ("flow_distribution", "track2d_dtw", "depth_error", "uni3d_moge_scene")


def applies(key: str, case: str) -> bool:
    """Whether `key` is computed on `case` (`<kind>/<name>`).

    The reference-world keys apply to synthetic cases only; Flow, Track2D, Depth and
    Uni3D MoGe apply to real cases only.
    """

    real = case.split("/", 1)[0] == "real"
    if key in REFERENCE_WORLD_KEYS:
        return not real
    if key in REAL_KEYS:
        return real
    return True


# Readings that compare rasterised geometry with the case video. They are excluded on
# cases whose main object is transparent: the video shows the scene behind the surface
# the rasteriser draws.
RASTER_VIDEO_KEYS = ("dynamic_iou", "flow_distribution", "track2d_dtw", "depth_error", "uni3d_moge_scene")
TRANSPARENT_CASES = ("real/internet_01_purple_granular_sand", "real/internet_19_fabric_laundry_tumbles",
                     "real/internet_22_viscous_honey_dripping", "real/phys101_01_solid_object_dropped",
                     "real/wisa_80k_06_pouring_red_wine", "real/wisa_80k_46_wooden_pestle_crushing")


def exclusion(key: str, case: str) -> str | None:
    """The cause `key` is `excluded` on `case` for, or None when it is scored there."""

    return TRANSPARENT if key in RASTER_VIDEO_KEYS and case in TRANSPARENT_CASES else None


def error_cause(message: str) -> str:
    """The `error` cause for a metric's failure message."""

    if "no covered pixel" in message or "no visible geometry" in message:
        return EMPTY_FIRST_FRAME
    return METRIC_FAILURE


def uncomputable_cause(key: str, family: str, detail: dict) -> str:
    """The `uncomputable` cause for `key`, from the counts recorded in `detail`."""

    if family == "track2d":
        return REFERENCE_EMPTY if detail.get("track2d_moving") == 0 else SUBMISSION_EMPTY
    if family == "dynamics":
        counts = detail.get("trajectory_paths")
        if counts is not None and not counts[1]:
            return SUBMISSION_EMPTY
        return NO_PAIRING
    if key == "depth_error":
        return SUBMISSION_EMPTY
    if key.startswith("uni3d_") and detail.get(f"uni3d_{key.split('_')[1]}_failure"):
        return REFERENCE_EMPTY
    if family == "flow":
        return REFERENCE_EMPTY
    return NO_PAIRING


__all__ = ["ERROR", "UNCOMPUTABLE", "NOT_APPLICABLE", "EXCLUDED", "TRANSPARENT",
           "REFERENCE_WORLD_KEYS", "RASTER_VIDEO_KEYS", "TRANSPARENT_CASES", "applies", "exclusion",
           "error_cause", "uncomputable_cause",
           "NO_VIDEO", "UNDECODABLE_VIDEO", "VIDEO_MISMATCH", "NO_WORLD", "INVALID_WORLD",
           "EMPTY_FIRST_FRAME", "DEGENERATE",
           "METRIC_FAILURE", "REFERENCE_EMPTY", "SUBMISSION_EMPTY", "NO_PAIRING"]
