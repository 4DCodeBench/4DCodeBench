"""Compute each case's reference estimates of its `reference.mp4`.

    data/<kind>/<case>/estimates/flow.h5       RAFT optical flow of reference.mp4
    data/<kind>/<case>/estimates/tracks.npz     CoTracker3 tracks of a frame-0 query grid
    data/<kind>/<case>/estimates/moge.npz       MoGe-3 point map of frame 0
    data/<kind>/<case>/estimates/semantic_<name>.npz
                                                DINOv3's and TIPSv2's embeddings
    data/<kind>/<case>/estimates/geophys_dinov3.npz
                                                DINOv3's pooled trajectory at the
                                                GeoPhys readout layer
    data/<kind>/<case>/estimates/depth.h5      Video Depth Anything disparity, on the
                                                model's own output grid

All estimates except the point map are on the case's sampled timeline
(`config.sampling`) and store their `stride`; the point map covers frame 0.

    python -m scorer.prepare --manifest /manifest.json
                             [--only flow|tracks|moge|semantic|geophys|depth]...

The manifest lists `{"case": "<kind>/<case>"}` entries; the scorer's manifest also
works, and each case in it is prepared once. Groups run one at a time:
each loads its model once, writes every case, and frees the model. The roots
default to the container's mounts; the flags point them at another tree:

    python -m scorer.prepare --manifest m.json --cases cases --data data
"""

from __future__ import annotations

import argparse
import os
from functools import cache
from pathlib import Path

import numpy as np
import torch

from config.annotation import MASK_FILENAME, MASK_KEY
from config.estimates import (
    DEPTH_FILENAME,
    FLOW_FILENAME,
    GEOPHYS_FILENAME,
    MOGE_FILENAME,
    SEMANTIC_FILENAME,
    TRACKS_FILENAME,
)
from config.models import (COTRACKER, GEOPHYS_MODELS, MOGE, RAFT, READOUT,
                           VIDEO_DEPTH_ANYTHING, block_for, checkpoint)
from config.sampling import stride_for
from scorer.analytic.tracks import DRIFT, grid_queries, grid_step, inside
from scorer.estimators.cotracker import (
    load_model as load_cotracker,
)
from scorer.estimators.cotracker import (
    track,
    tracking_shape,
)
from scorer.estimators.moge import REFINE_STEPS, RESOLUTION_LEVEL, point_map
from scorer.estimators.moge import load_model as load_moge
from scorer.estimators.raft import ITERATIONS, stream_flow
from scorer.estimators.raft import load_model as load_raft
from scorer.estimators.video_depth import cached_disparity, frame_stream, stream_video_depth, video_info
from scorer.estimators.video_depth import load_model as load_video_depth
from scorer.metrics.semantic import MODELS as SEMANTIC_BACKBONES
from scorer.metrics.semantic import embed_video, frame_trajectory, release, short
from scorer.paths import (
    CASES_ROOT,
    CHECKPOINTS_DIR,
    DATA_ROOT,
    MANIFEST_PATH,
    Entry,
    Roots,
    load_manifest,
)
from scorer.world import video_stream
from scorer.storage import write_dense

def write_flow(video: Path, out: Path, model, device, stride: int) -> None:
    """Write `flow.h5`: float16 RAFT flow at full resolution per pair `t -> t + stride`."""

    flows, valids = [], []
    for flow, valid in stream_flow(video, model, device, stride=stride):
        flows.append(flow.astype(np.float16))
        valids.append(valid)

    write_dense(out / FLOW_FILENAME, flow=np.stack(flows), valid=np.stack(valids),
                        stride=np.int64(stride), model=RAFT, iterations=ITERATIONS)


def dynamic_mask(annotation: Path, height: int, width: int) -> np.ndarray | None:
    """Return frame 0 of the annotated dynamic mask as `(H, W)` bool, or None.

    Returns None when there is no annotation, its resolution differs from the
    video's, or frame 0 holds no dynamic pixel.
    """

    path = annotation / MASK_FILENAME
    if not path.is_file():
        return None
    with np.load(path) as held:
        first = held[MASK_KEY][0].astype(bool)
    if first.shape != (height, width) or not first.any():
        return None
    return first


def write_tracks(video: Path, out: Path, model, device, stride: int,
                 annotation: Path) -> None:
    """Write `tracks.npz`: CoTracker3 tracks of a query grid on frame 0.

    Without a dynamic mask the grid covers the whole image. With one,
    `grid_queries` keeps `DRIFT` queries over the whole image and places the rest
    inside the mask; `on_mask` marks the queries inside the mask, and Track2D
    estimates the shared drift from the others.
    """

    _, height, width, _ = video_info(video)
    mask = dynamic_mask(annotation, height, width)
    queries = grid_queries(height, width, mask=mask)
    on_mask = (inside(queries[:, 1:], mask) if mask is not None
               else np.zeros(len(queries), dtype=bool))
    step = (grid_step(height, width, max(len(queries) - DRIFT, 1), mask) if mask is not None
            else grid_step(height, width))
    tracks, visible = track(video, queries, model, device, stride=stride)

    np.savez_compressed(out / TRACKS_FILENAME, tracks=tracks.astype(np.float32), visible=visible,
                        queries=queries.astype(np.float32),
                        on_mask=on_mask,
                        shape=np.asarray([tracks.shape[1], height, width], dtype=np.int64),
                        step=np.int64(step),
                        stride=np.int64(stride),
                        model="cotracker3-offline",
                        tracking_resolution=np.asarray(tracking_shape(height, width), dtype=np.int64))


def write_moge(video: Path, out: Path, model, device) -> None:
    """Write `moge.npz`: MoGe-3's camera-space point map of frame 0."""

    frame = next(frame_stream(video), None)
    if frame is None:
        raise ValueError(f"{video}: no first frame to estimate a point map from")
    made = point_map(frame, model, device)
    points, mask = made["points"], made["mask"]

    np.savez_compressed(out / MOGE_FILENAME, points=points, mask=mask,
                        intrinsics=made["intrinsics"],
                        shape=np.asarray(mask.shape, dtype=np.int64),
                        model=MOGE, resolution_level=np.int64(RESOLUTION_LEVEL),
                        refine_steps=np.int64(REFINE_STEPS))


def write_semantic(video: Path, out: Path, name: str, model, checkpoints: Path | None,
                   stride: int) -> None:
    """Write `semantic_<name>.npz`: one backbone's float32 embeddings of the sampled frames."""

    embeddings = embed_video(model, video, root=checkpoints, stride=stride)
    np.savez(out / SEMANTIC_FILENAME.format(name=short(name)),
             embeddings=embeddings.astype(np.float32), stride=np.int64(stride))


def write_geophys(video: Path, out: Path, tag: str, model, block: int, layer: int,
                  checkpoints: Path | None, stride: int) -> None:
    """Write `geophys_<tag>.npz`: the pooled GeoPhys trajectory of the sampled frames.

    Each row is the spatial average of the frame's patch tokens at readout `block`
    (the paper's `z̄ₜ`), distinct from the class token in `semantic_<name>.npz`.
    """

    trajectory = frame_trajectory(model, video, block=block, root=checkpoints, stride=stride)
    np.savez(out / GEOPHYS_FILENAME.format(name=tag),
             trajectory=trajectory.astype(np.float32),
             layer=np.int64(layer), stride=np.int64(stride))


def write_depth(video: Path, out: Path, model, device, stride: int) -> None:
    """Write `depth.h5`: Video Depth Anything's float16 disparity of the sampled frames.

    Stores the model's output grid; the Depth metric upsamples to video resolution.
    """

    count, height, width, fps = video_info(video)
    maps = np.stack([cached_disparity(disparity, video, frame * stride)
                     for frame, disparity in enumerate(
                         stream_video_depth(video, model, device=device, stride=stride))])

    target = out / DEPTH_FILENAME
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    write_dense(temporary, disparity=maps,
                shape=np.asarray([count, height, width], dtype=np.int64),
                stride=np.int64(stride), fps=np.float64(fps))
    temporary.replace(target)


@cache
def stride_of(video: Path) -> int:
    """Return the sampled-timeline stride of `video`, from its frame count."""

    return stride_for(video_stream(video)[0])


def flow_group(entries: list[Entry], device, checkpoints: Path | None) -> None:
    model = load_raft(checkpoint(RAFT, checkpoints), device)
    for entry in entries:
        write_flow(entry.video, entry.estimates, model, device, stride_of(entry.video))
    del model
    torch.cuda.empty_cache()


def tracks_group(entries: list[Entry], device, checkpoints: Path | None) -> None:
    model = load_cotracker(checkpoint(COTRACKER, checkpoints), device)
    for entry in entries:
        write_tracks(entry.video, entry.estimates, model, device, stride_of(entry.video),
                     entry.annotation)
    del model
    torch.cuda.empty_cache()


def moge_group(entries: list[Entry], device, checkpoints: Path | None) -> None:
    model = load_moge(checkpoint(MOGE, checkpoints), device)
    for entry in entries:
        write_moge(entry.video, entry.estimates, model, device)
    del model
    torch.cuda.empty_cache()


def semantic_group(entries: list[Entry], device, checkpoints: Path | None) -> None:
    """Write the embeddings of every Semantic backbone, one backbone at a time."""

    for name, model in SEMANTIC_BACKBONES.items():
        try:
            for entry in entries:
                write_semantic(entry.video, entry.estimates, name, model, checkpoints,
                               stride_of(entry.video))
        finally:
            release(model)


def geophys_group(entries: list[Entry], device, checkpoints: Path | None) -> None:
    """Write the GeoPhys trajectories of every backbone in the store, one at a time."""

    from scorer.estimators import backbones

    models = {"dinov3": backbones.DINOV3}
    for name, tag in GEOPHYS_MODELS.items():
        if not checkpoint(name, checkpoints).exists():
            print(f"{name}: not in {checkpoints}, skipped", flush=True)
            continue
        model = models[tag]
        try:
            for entry in entries:
                write_geophys(entry.video, entry.estimates, tag, model, block_for(name),
                              READOUT[name], checkpoints, stride_of(entry.video))
        finally:
            release(model)


def depth_group(entries: list[Entry], device, checkpoints: Path | None) -> None:
    model = load_video_depth(checkpoint(VIDEO_DEPTH_ANYTHING, checkpoints),
                             device=device)
    for entry in entries:
        write_depth(entry.video, entry.estimates, model, device, stride_of(entry.video))
    del model
    torch.cuda.empty_cache()


GROUPS = {"flow": flow_group, "tracks": tracks_group, "moge": moge_group,
          "semantic": semantic_group, "geophys": geophys_group, "depth": depth_group}


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="python -m scorer.prepare", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH,
                        help="the cases to prepare, as the harness wrote them")
    parser.add_argument("--cases", type=Path, default=CASES_ROOT, help="the cases root")
    parser.add_argument("--data", type=Path, default=DATA_ROOT, help="the data root")
    parser.add_argument("--checkpoints", type=Path, default=CHECKPOINTS_DIR,
                        help="the model store the estimators load from")
    parser.add_argument("--only", action="append", default=None,
                        help=f"the estimate groups to compute, out of {list(GROUPS)}; "
                             "repeatable or comma-separated; all of them by default")
    parser.add_argument("--device", default=None)
    args = parser.parse_args(argv)
    wanted = [name for item in args.only or () for name in item.split(",") if name]
    unknown = sorted(set(wanted) - set(GROUPS))
    if unknown:
        parser.error(f"--only names {list(GROUPS)}, not {unknown}")
    args.only = [name for name in GROUPS if not wanted or name in wanted]
    return args


def main(argv=None) -> int:
    """Compute the selected estimate groups for every case in the manifest."""

    args = parse_args(argv)
    device = torch.device(args.device) if args.device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")
    roots = Roots(cases=args.cases, data=args.data, checkpoints=args.checkpoints)
    entries = list({entry.case: entry for entry in load_manifest(args.manifest, roots)}.values())
    for entry in entries:
        entry.estimates.mkdir(parents=True, exist_ok=True)
    print(f"{len(entries)} case(s), groups {args.only}", flush=True)
    for name in args.only:
        print(f"== {name}", flush=True)
        GROUPS[name](entries, device, args.checkpoints)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
