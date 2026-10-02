"""Patch-feature PCA videos for one reference/render pair.

One PCA per pair, following the DINOv3 paper: frames are read at a long side of
`LONG_SIDE` snapped to whole patches, patch tokens are L2-normalised, three
components are mapped to RGB and each stretched between its 1% and 99% quantile,
and the grid is bilinearly upsampled to pixels. The basis is fitted on sampled
patches of both videos together, so a colour is comparable across them.
`LONG_SIDE` affects only these videos; the semantic metric reads each checkpoint
at its declared input size.

Frames follow the case's sampled timeline (`config.sampling`) and play at the
video's fps over the stride.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch

from config.sampling import sampled_length, stride_for
from scorer.estimators import backbones
from scorer.metrics.semantic import RENDER_FILENAME, batches, iter_frames, release

from .. import Target
from ..media import Writer

MODELS = {"dinov3": backbones.DINOV3, "tips": backbones.TIPS}
LONG_SIDE = 1024          # the long side the dense patch grid is read at
DENSE_BLOCK = None        # transformer block from the end; None is the top one
FIT_FRAMES = 12           # frames per video the colour basis is fitted on
SIDES = ("reference", "render")


def probe(path: str | Path) -> dict:
    """The frame count, size and rate of a video."""

    capture = cv2.VideoCapture(str(path))
    info = {
        "frames": int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
        "width": int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps": capture.get(cv2.CAP_PROP_FPS) or 24.0,
    }
    capture.release()
    return info


def fused_attention(model: torch.nn.Module) -> None:
    """Replace TIPS's attention forward with `scaled_dot_product_attention`.

    The fused kernel does not materialise the `(heads, n, n)` attention matrix,
    which does not fit in memory at `LONG_SIDE`.
    """

    blocks = getattr(getattr(model, "vision_encoder", None), "blocks", None)
    if blocks is None:
        return

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, tokens, channels = x.shape
        heads = self.num_heads
        qkv = self.qkv(x).reshape(batch, tokens, 3, heads, channels // heads).permute(2, 0, 3, 1, 4)
        query, key, value = qkv.unbind(0)
        attended = torch.nn.functional.scaled_dot_product_attention(query, key, value, scale=self.scale)
        return self.proj_drop(self.proj(attended.transpose(1, 2).reshape(batch, tokens, channels)))

    attention = type(blocks[0].attn)
    for owner in (attention, attention.__mro__[1]):
        owner.forward = forward


def dense_size(height: int, width: int, patch: int) -> tuple[int, int]:
    """The frame's aspect at a long side of `LONG_SIDE`, snapped to whole patches."""

    scale = LONG_SIDE / max(height, width)
    return tuple(max(patch, round(side * scale / patch) * patch) for side in (height, width))


def dense_tokens(model, frames: np.ndarray, size: tuple[int, int], root=None) -> torch.Tensor:
    """L2-normalised patch grids `(N, gh, gw, D)` at `DENSE_BLOCK`."""

    patches = model.tokens(frames, size=size, block=DENSE_BLOCK, root=root)[1]
    return torch.nn.functional.normalize(patches, dim=-1)


def sample_tokens(model, path: str | Path, size: tuple[int, int], stride: int, root=None,
                  step: int = 1) -> torch.Tensor:
    """Patch tokens of every `stride`-th frame, flattened to `(M, D)`."""

    kept = [frame for index, frame in enumerate(iter_frames(path, step)) if index % stride == 0]
    parts = [dense_tokens(model, batch, size, root).flatten(0, 2) for batch in batches(np.stack(kept), 1)]
    return torch.cat(parts)


def basis(features: torch.Tensor) -> dict:
    """A three-component colour basis with its 1%/99% stretch; seeded for reproducibility."""

    torch.manual_seed(0)
    mean = features.mean(0)
    _, _, vectors = torch.pca_lowrank(features - mean, q=7, center=False)
    axis = vectors[:, :3]
    projected = (features - mean) @ axis
    return {
        "mean": mean,
        "axis": axis,
        "low": torch.quantile(projected, 0.01, dim=0),
        "high": torch.quantile(projected, 0.99, dim=0),
    }


def fit_colours(model, reference, render, size: tuple[int, int], stride: int, root=None,
                step: int = 1) -> dict:
    """One colour basis fitted on the sampled patches of both videos together."""

    joint = torch.cat([sample_tokens(model, path, size, stride, root, step)
                       for path in (reference, render)])
    fit = basis(joint)
    del joint
    return fit


def colourise(grids: torch.Tensor, fit: dict) -> np.ndarray:
    """Map an `(N, gh, gw, D)` grid to RGB in `[0, 1]`."""

    rgb = ((grids - fit["mean"]) @ fit["axis"] - fit["low"]) / (fit["high"] - fit["low"])
    return rgb.clamp(0.0, 1.0).cpu().numpy()


def to_frame(rgb: np.ndarray, frame_size: tuple[int, int]) -> np.ndarray:
    """Convert a `(gh, gw, 3)` RGB grid in `[0, 1]` to a BGR uint8 frame at `(width, height)`."""

    pixels = cv2.cvtColor((rgb * 255).round().astype(np.uint8), cv2.COLOR_RGB2BGR)
    return cv2.resize(pixels, frame_size, interpolation=cv2.INTER_LINEAR)


def write_frame_video(model, path, size: tuple[int, int], fit: dict, out: Path, info: dict, root=None,
                      step: int = 1) -> Path:
    """Write one side's patch PCA video at the frame size, one frame per frame."""

    frame_size = (info["width"], info["height"])
    writer = Writer(out, info["fps"])
    try:
        for frame in iter_frames(path, step):
            writer.write(to_frame(colourise(dense_tokens(model, frame[None], size, root), fit)[0], frame_size))
    finally:
        writer.close()
    return out


def feature_videos(name: str, reference, render, out_dir: str | Path, root=None) -> dict[str, Path]:
    """Write one model's PCA video per side, `<side>_<name>.mp4`.

    Loads the model; the caller releases it.
    """

    out_dir = Path(out_dir)
    info = probe(reference)
    # the metric's sampled timeline, at the fps that preserves the video's duration
    step = stride_for(info["frames"])
    info = {**info, "frames": sampled_length(info["frames"], step), "fps": info["fps"] / step}
    model = MODELS[name]
    fused_attention(model.load(root=root))
    size = dense_size(info["height"], info["width"], model.PATCH_SIZE)
    fit = fit_colours(model, reference, render, size, max(1, info["frames"] // FIT_FRAMES), root, step)
    return {side: write_frame_video(model, path, size, fit, out_dir / f"{side}_{name}.mp4", info, root, step)
            for side, path in zip(SIDES, (reference, render), strict=True)}


def render(targets: list[Target]) -> list[str]:
    """Write the PCA videos of every model for every run, each backbone loaded once.

    Existing videos are kept; delete one to have it rebuilt.
    """

    status = []
    for name, model in MODELS.items():
        count = 0
        try:
            for target in targets:
                render_video = target.pred / RENDER_FILENAME
                if target.reference_video is None or not render_video.is_file():
                    continue
                if all((target.out / f"{side}_{name}.mp4").is_file() for side in SIDES):
                    continue
                target.out.mkdir(parents=True, exist_ok=True)
                feature_videos(name, target.reference_video, render_video, target.out,
                               target.checkpoints)
                count += 1
        finally:
            release(model)      # free the weights before the next backbone loads
        status.append(f"{name}: {count} world(s)")
    return status


__all__ = ["MODELS", "SIDES", "feature_videos", "probe", "render"]
