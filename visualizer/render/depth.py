"""Depth videos from the maps the depth metric compared."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from config.estimates import DEPTH_FILENAME, RASTER_DEPTH_FILENAME
from scorer.storage import open_dense

from .. import Target
from ..media import Writer, colourise

ERROR_SPAN = 3.0    # colour scale of the per-pixel error map
DEPTH_VIDEOS = ("reference_depth.mp4", "render_depth.mp4", "depth_error.mp4")
ARRAYS = (RASTER_DEPTH_FILENAME,)


def depth_videos(estimates: Path, results: Path, out_dir: Path, fps: float) -> dict[str, Path]:
    """Write the reference, render and error depth videos.

    The cached reference disparity is normalised with the per-frame median and
    deviation the depth metric stored. A pixel without geometry has error 1, as in
    the metric.
    """

    with open_dense(estimates / DEPTH_FILENAME) as cached, open_dense(results / RASTER_DEPTH_FILENAME) as data:
        reference = cached["disparity"]
        raster, covered = data["disparity"], data["valid"]
        median, deviation = data["median"], data["deviation"]

        names = dict(zip(("reference", "render", "error"), DEPTH_VIDEOS, strict=True))
        writers = {side: Writer(out_dir / name, fps) for side, name in names.items()}
        everywhere = np.ones(raster.shape[1:], dtype=bool)
        try:
            for frame in range(len(raster)):
                gt = (reference[frame].astype(np.float32) - median[frame]) / deviation[frame]
                pred = raster[frame].astype(np.float32)
                valid = covered[frame]
                writers["reference"].write(colourise(gt, everywhere))
                writers["render"].write(colourise(pred, valid))
                error = np.where(valid, np.abs(gt - pred), 1.0)
                writers["error"].write(colourise(error, everywhere, span=ERROR_SPAN, low=0.0))
        finally:
            for writer in writers.values():
                writer.close()
    return {side: out_dir / name for side, name in names.items()}


def render(target: Target) -> str:
    if not target.has_results(*ARRAYS):
        return "no depth arrays"
    if target.estimates is None or not (target.estimates / DEPTH_FILENAME).is_file():
        return "no cached reference depth"
    if target.has_output(*DEPTH_VIDEOS):
        return "depth exists"
    target.out.mkdir(parents=True, exist_ok=True)
    depth_videos(target.estimates, target.results, target.out, reading_fps(target.results))
    return "depth"


def reading_fps(results: Path) -> float:
    """Frame rate of the depth videos: the video's fps over the depth stride.

    Returns 24 without `reward.detail.json`.
    """

    detail = Path(results) / "reward.detail.json"
    if not detail.is_file():
        return 24.0
    read = json.loads(detail.read_text(encoding="utf-8"))
    return float(read.get("fps") or 24.0) / max(1, int(read.get("depth_stride") or 1))


__all__ = ["ARRAYS", "DEPTH_VIDEOS", "ERROR_SPAN", "depth_videos", "reading_fps", "render"]
