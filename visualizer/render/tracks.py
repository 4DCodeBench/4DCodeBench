"""The Track2D paths drawn on the videos they were tracked on.

The paths are the reference's moving queries that Track2D scores.
`reference_tracks.mp4` draws the reference side on the case video and
`render_tracks.mp4` the submission's side on its render, so row `i` is the same
query on both; a row without a position on a frame is not drawn there.
`matches.mp4` draws both sides on the case video, each pair joined by a segment.
The colour is a function of the query row, so a path keeps it across the videos.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from config.estimates import TRACKS_FILENAME
from scorer.metrics.track2d import _paths
from scorer.world import World

from .. import Target
from ..media import Writer, fps_of, frames_of, palette, stamp

TRACK_VIDEOS = ("reference_tracks.mp4", "render_tracks.mp4", "matches.mp4")
TRAIL = 24                       # frames of history drawn behind each dot
FONT = cv2.FONT_HERSHEY_SIMPLEX


def pair_inks(rows: int) -> np.ndarray:
    """One palette colour per query row."""

    return palette(rows)[1:]


def draw_paths(shot: np.ndarray, paths: np.ndarray, inks: np.ndarray, frame: int) -> np.ndarray:
    """Draw each path's dot at `frame` and its last `TRAIL` steps as a polyline.

    Steps with a NaN end are skipped.
    """

    canvas = shot.copy()
    history = min(TRAIL, frame)
    for row in range(len(paths)):
        ink = inks[row].tolist()
        for step in range(frame - history, frame):
            start, stop = paths[row, step], paths[row, step + 1]
            if np.isfinite(start).all() and np.isfinite(stop).all():
                cv2.line(canvas, tuple(np.round(start).astype(int)),
                         tuple(np.round(stop).astype(int)), ink, 2, cv2.LINE_AA)
    points = paths[:, frame]
    alive = np.isfinite(points).all(axis=1)
    stamp(canvas, points[alive], inks[alive], 2)
    return canvas


def caption(canvas: np.ndarray, lines: list[str]) -> np.ndarray:
    """Draw the caption lines in the top-left corner, dark then light, in place."""

    scale = max(0.42, min(1.0, canvas.shape[1] / 1280))
    step = int(round(30 * scale))
    for index, line in enumerate(lines):
        weight = 2 if index == 0 else 1
        spot = (int(round(12 * scale)), int(round(28 * scale)) + index * step)
        cv2.putText(canvas, line, spot, FONT, 0.62 * scale, (0, 0, 0),
                    weight + 3, cv2.LINE_AA)
        cv2.putText(canvas, line, spot, FONT, 0.62 * scale, (255, 255, 255),
                    weight, cv2.LINE_AA)
    return canvas


def track_videos(results: Path, world, reference: Path, out_dir: Path, fps: float,
                 title: str = "", cached: Path | None = None) -> dict[str, Path]:
    """Write the three track videos.

    Paths are stored in units of the image diagonal and converted to pixels at
    the submission's resolution. Videos are decoded on the sampled timeline and
    play at the video's fps over the stride.
    """

    out_dir.mkdir(parents=True, exist_ok=True)
    with np.load(results / "track2d.npz") as data:
        stride, moving = int(data["stride"]), data["moving"]
        if "gt" in data.files:
            gt, pred = data["gt"], data["pred"]
        elif moving.any():
            # no submission paths: draw the reference's moving queries alone
            with np.load(cached / TRACKS_FILENAME) as archive:
                tracks, visible = archive["tracks"], archive["visible"]
            gt = _paths(tracks[moving], visible[moving], float(np.hypot(*world.shape[1:])))[..., :2]
            pred = np.full_like(gt, np.nan)
        else:                                  # no reference mover
            gt = pred = np.zeros((0, 0, 2), np.float32)
    diagonal = float(np.hypot(*world.shape[1:]))
    gt, pred = gt * diagonal, pred * diagonal
    inks = pair_inks(len(gt))

    head = f"{title}   " if title else ""
    labels = {
        "reference_tracks.mp4": [f"{head}GT (reference video)", f"{len(gt)} moving paths"],
        "render_tracks.mp4": [f"{head}Reconstruction (render)", f"{len(gt)} moving paths"],
        "matches.mp4": [f"{head}Track2D pairs", f"{len(gt)} pairs, both sides on the reference video"],
    }
    writers = {name: Writer(out_dir / name, fps / stride) for name in TRACK_VIDEOS}
    try:
        videos = zip(frames_of(reference, stride), frames_of(world.video_path, stride), strict=True)
        for frame, (reference_shot, render_shot) in enumerate(videos):
            writers["reference_tracks.mp4"].write(caption(
                draw_paths(reference_shot, gt, inks, frame), labels["reference_tracks.mp4"]))
            writers["render_tracks.mp4"].write(caption(
                draw_paths(render_shot, pred, inks, frame), labels["render_tracks.mp4"]))
            both = draw_paths(draw_paths(reference_shot, gt, inks, frame), pred, inks, frame)
            for row in range(len(gt)):
                start, stop = gt[row, frame], pred[row, frame]
                if np.isfinite(start).all() and np.isfinite(stop).all():
                    cv2.line(both, tuple(np.round(start).astype(int)),
                             tuple(np.round(stop).astype(int)), inks[row].tolist(),
                             1, cv2.LINE_AA)
            writers["matches.mp4"].write(caption(both, labels["matches.mp4"]))
    finally:
        for writer in writers.values():
            writer.close()
    return {name: out_dir / name for name in TRACK_VIDEOS}


def render(target: Target) -> str:
    cached, video = target.estimates, target.reference_video
    if cached is None or not (cached / TRACKS_FILENAME).is_file() or video is None \
            or not target.has_results("track2d.npz"):
        return "no track2d arrays"
    if target.has_output(*TRACK_VIDEOS):
        return "tracks exists"
    world = World(target.pred)
    track_videos(target.results, world, video, target.out, fps_of(world.video_path),
                 target.entry.case.split("/")[-1], cached)
    return "track2d"


__all__ = [
    "FONT",
    "TRACK_VIDEOS",
    "TRAIL",
    "caption",
    "draw_paths",
    "pair_inks",
    "render",
    "track_videos",
]
