"""The DynamicIoU masks as videos: one per side and their overlap."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np

from scorer.world import World

from .. import Target
from ..media import Writer, fps_of, frames_of, tinted

MASK_VIDEOS = ("reference_masks.mp4", "render_masks.mp4", "masks_overlap.mp4")
LONE_MASK = ("render_masks.mp4",)


@lru_cache(maxsize=4)
def dynamic_bits(results: Path) -> dict[str, np.ndarray]:
    """Load the packed masks DynamicIoU stored, cached per results directory."""

    with np.load(results / "dynamic.npz") as data:
        return {"shape": tuple(int(value) for value in data["shape"]),
                "gt": data["gt"], "pred": data["pred"]}


def dynamic_masks(results: Path, frame: int) -> dict[str, np.ndarray]:
    """Unpack both sides' dynamic masks of one frame."""

    bits = dynamic_bits(results)
    return {side: np.unpackbits(bits[side][frame], axis=-1, count=bits["shape"][2]).astype(bool)
            for side in ("gt", "pred")}


def mask_videos(results: Path | None, world, reference: Path | None, out_dir: Path,
                fps: float) -> dict[str, Path]:
    """Write the mask videos.

    With `results`, the reference and render videos are tinted by the masks
    DynamicIoU stored, and an overlap video colours reference-only, render-only
    and shared pixels. `reference` is the video the reference mask belongs to.
    Without `results`, only the render is written, tinted by its own dynamic ids.
    """

    out_dir.mkdir(parents=True, exist_ok=True)
    if results is None:
        ids = world.dynamic_ids
        writer = Writer(out_dir / "render_masks.mp4", fps)
        try:
            for frame, shot in enumerate(frames_of(world.video_path)):
                writer.write(tinted(shot, np.isin(world.index[frame], ids)))
        finally:
            writer.close()
        return {"render": out_dir / "render_masks.mp4"}

    writers = {name: Writer(out_dir / name, fps) for name in MASK_VIDEOS}
    try:
        videos = zip(frames_of(reference), frames_of(world.video_path), strict=True)
        for frame, (left, right) in enumerate(videos):
            masks = dynamic_masks(results, frame)
            both = np.zeros((*masks["gt"].shape, 3), np.uint8)
            both[masks["gt"] & ~masks["pred"]] = (166, 139, 111)     # BGR: reference only
            both[masks["pred"] & ~masks["gt"]] = (60, 136, 224)      # attempt only
            both[masks["gt"] & masks["pred"]] = (235, 235, 235)      # both
            writers["reference_masks.mp4"].write(tinted(left, masks["gt"]))
            writers["render_masks.mp4"].write(tinted(right, masks["pred"]))
            writers["masks_overlap.mp4"].write(both)
    finally:
        for writer in writers.values():
            writer.close()
    return {name: out_dir / name for name in MASK_VIDEOS}


def render(target: Target) -> str:
    # the reference mask is drawn on the reference world's render, or on the case video
    # when the case has no reference world
    left = World(target.gt).video_path if target.gt is not None else target.reference_video
    scored = target.has_results("dynamic.npz") and left is not None and left.is_file()
    wanted = MASK_VIDEOS if scored else LONE_MASK
    if target.has_output(*wanted):
        return "masks exists"
    world = World(target.pred)
    mask_videos(target.results if scored else None, world, left if scored else None,
                target.out, fps_of(world.video_path))
    return "masks"


__all__ = ["LONE_MASK", "MASK_VIDEOS", "dynamic_masks", "mask_videos", "render"]
