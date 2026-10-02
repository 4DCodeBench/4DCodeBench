"""Both sides' flow on one colour wheel, and the error between them."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from config.estimates import FLOW_FILENAME
from scorer.storage import open_dense
from scorer.world import World

from .. import Target
from ..media import Writer, colourise, fps_of

FLOW_VIDEOS = ("reference_flow.mp4", "render_flow.mp4", "flow_error.mp4")
RADIUS_FRACTION = 0.02    # the wheel saturates at this fraction of the image diagonal


def colour_wheel() -> np.ndarray:
    """RAFT's 55-colour wheel (Baker et al. 2011), `(55, 3)` uint8, RGB."""

    RY, YG, GC, CB, BM, MR = 15, 6, 4, 11, 13, 6
    wheel = np.zeros((RY + YG + GC + CB + BM + MR, 3))
    col = 0
    wheel[0:RY, 0] = 255
    wheel[0:RY, 1] = np.floor(255 * np.arange(0, RY) / RY)
    col += RY
    wheel[col:col + YG, 0] = 255 - np.floor(255 * np.arange(0, YG) / YG)
    wheel[col:col + YG, 1] = 255
    col += YG
    wheel[col:col + GC, 1] = 255
    wheel[col:col + GC, 2] = np.floor(255 * np.arange(0, GC) / GC)
    col += GC
    wheel[col:col + CB, 1] = 255 - np.floor(255 * np.arange(CB) / CB)
    wheel[col:col + CB, 2] = 255
    col += CB
    wheel[col:col + BM, 2] = 255
    wheel[col:col + BM, 0] = np.floor(255 * np.arange(0, BM) / BM)
    col += BM
    wheel[col:col + MR, 2] = 255 - np.floor(255 * np.arange(MR) / MR)
    wheel[col:col + MR, 0] = 255
    return wheel.astype(np.uint8)


WHEEL = colour_wheel()


def flow_to_image(flow, radius: float, valid=None, invalid_shade: int = 40) -> np.ndarray:
    """Paint flow on the colour wheel as `(H, W, 3)` uint8 RGB.

    Hue encodes direction and saturation magnitude, saturating at `radius`; pixels
    outside `valid` are `invalid_shade`.
    """

    flow = torch.as_tensor(flow).float()
    u, v = flow[..., 0], flow[..., 1]
    magnitude = (torch.sqrt(u * u + v * v) / max(radius, 1e-5)).clamp(max=1.0)
    angle = torch.atan2(-v, -u) / np.pi                     # RAFT's convention
    fk = (angle + 1) / 2 * (len(WHEEL) - 1)
    k0 = torch.floor(fk).long()
    k1 = (k0 + 1) % len(WHEEL)
    f = (fk - k0)[..., None]
    wheel = torch.as_tensor(WHEEL, device=flow.device).float() / 255.0
    col = (1 - f) * wheel[k0] + f * wheel[k1]
    col = 1 - magnitude[..., None] * (1 - col)              # white at zero motion
    image = torch.floor(255 * col).to(torch.uint8)
    if valid is not None:
        image[~torch.as_tensor(valid, device=flow.device)] = invalid_shade
    return image.cpu().numpy()


def flow_videos(estimates: Path, results: Path, out_dir: Path, fps: float) -> dict[str, Path]:
    """Write the reference, render and error flow videos.

    Both sides use the colour wheel with radius `RADIUS_FRACTION` of the image
    diagonal; invalid pixels are dark grey. The error video shows the endpoint error,
    in units of that radius, over the pixels the metric evaluated. One frame per
    sampled frame pair, played at the video's fps over the stride.
    """

    out_dir.mkdir(parents=True, exist_ok=True)
    with open_dense(results / FLOW_FILENAME) as data, open_dense(estimates / FLOW_FILENAME) as reference:
        render, render_valid, evaluated = data["flow"], data["valid"], data["evaluated"]
        height, width = render.shape[1:3]
        stride = int(data["stride"][()])
        gt, gt_valid = reference["flow"], reference["valid"]
        if (gt.shape[1:] != (height, width, 2) or render.shape[1:] != (height, width, 2)
                or gt_valid.shape != gt.shape[:-1] or render_valid.shape != render.shape[:-1]
                or evaluated.shape != render_valid.shape):
            raise ValueError("Flow arrays and masks must share the full-resolution grid")
        radius = RADIUS_FRACTION * float(np.hypot(height, width))
        writers = {name: Writer(out_dir / name, fps / stride) for name in FLOW_VIDEOS}
        try:
            for frame in range(min(len(render), len(gt))):
                for name, flow, valid in (("reference_flow.mp4", gt[frame], gt_valid[frame]),
                                          ("render_flow.mp4", render[frame], render_valid[frame])):
                    writers[name].write(np.ascontiguousarray(flow_to_image(flow, radius, valid)[..., ::-1]))
                gap = np.linalg.norm(gt[frame].astype(np.float32) - render[frame].astype(np.float32), axis=-1)
                writers["flow_error.mp4"].write(colourise(gap / radius, evaluated[frame], span=1.0, low=0.0))
        finally:
            for writer in writers.values():
                writer.close()
    return {name: out_dir / name for name in FLOW_VIDEOS}


def render(target: Target) -> str:
    cached = target.estimates
    if cached is None or not (cached / FLOW_FILENAME).is_file() or not target.has_results(FLOW_FILENAME):
        return "no flow arrays"
    if target.has_output(*FLOW_VIDEOS):
        return "flow exists"
    world = World(target.pred)
    flow_videos(cached, target.results, target.out, fps_of(world.video_path))
    return "flow"


__all__ = ["FLOW_VIDEOS", "flow_to_image", "flow_videos", "render"]
