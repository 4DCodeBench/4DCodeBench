"""The scorer's per-pixel object id map, as a video and as an overlay on the render."""

from __future__ import annotations

from pathlib import Path

import cv2

from scorer.world import World

from .. import Target
from ..media import OVERLAY, Writer, fps_of, frames_of, palette

INDEX_VIDEOS = ("index.mp4", "index_overlay.mp4")


def index_videos(world, out_dir: Path, fps: float) -> dict[str, Path]:
    """Write the id map video, one colour per object id, and its overlay on the render."""

    out_dir.mkdir(parents=True, exist_ok=True)
    inks = palette(int(world.index.max()))
    writers = {name: Writer(out_dir / name, fps) for name in INDEX_VIDEOS}
    try:
        for frame, shot in enumerate(frames_of(world.video_path)):
            ids = inks[world.index[frame]]
            if ids.shape[:2] != shot.shape[:2]:
                ids = cv2.resize(ids, (shot.shape[1], shot.shape[0]), interpolation=cv2.INTER_NEAREST)
            writers["index.mp4"].write(ids)
            writers["index_overlay.mp4"].write(cv2.addWeighted(shot, 1 - OVERLAY, ids, OVERLAY, 0))
    finally:
        for writer in writers.values():
            writer.close()
    return {name: out_dir / name for name in INDEX_VIDEOS}


def render(target: Target) -> str:
    if target.has_output(*INDEX_VIDEOS):
        return "index exists"
    world = World(target.pred)
    index_videos(world, target.out, fps_of(world.video_path))
    return "index"


__all__ = ["INDEX_VIDEOS", "index_videos", "render"]
