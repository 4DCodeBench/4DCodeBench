"""Semantic similarity: per-frame cosine of render and reference embeddings (DINOv3, TIPSv2).

Frame `t` of the render is compared with frame `t` of the reference on the case's
sampled timeline (`config.sampling`); both must have the same frame count.
`semantic_dinov3` and `semantic_tips` are the mean cosines, which `scorer.score` maps
to `(1 + cos) / 2`. The reference embeddings are cached in `estimates/` by
`python -m scorer.prepare`, so only the render is embedded here. `frame_trajectory`
extracts the per-frame features GeoPhys uses.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from config.estimates import SEMANTIC_FILENAME

from ..estimators import backbones, free
from ..prefetch import prefetch

MODELS = {"semantic_dinov3": backbones.DINOV3, "semantic_tips": backbones.TIPS}
BATCH = 8
RENDER_FILENAME = "render.mp4"


def short(name: str) -> str:
    """Return the on-disk backbone tag of a reading key: `semantic_tips` -> `tips`."""

    return name.removeprefix("semantic_")


def iter_frames(path: str | Path, stride: int = 1) -> Iterator[np.ndarray]:
    """Decode an mp4 into RGB uint8 frames in presentation order.

    With `stride > 1` only every `stride`-th frame is decoded; the others are grabbed
    and skipped.
    """

    stride = max(1, int(stride))
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"unreadable video: {path}")
    try:
        index = 0
        while True:
            if not capture.grab():
                return
            if index % stride == 0:
                ok, frame = capture.retrieve()
                if not ok:
                    return
                yield cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            index += 1
    finally:
        capture.release()


def batches(frames: Iterable[np.ndarray], size: int = BATCH) -> Iterator[np.ndarray]:
    """Group a frame stream into stacked `(n, H, W, 3)` batches of at most `size`."""

    batch: list[np.ndarray] = []
    for frame in frames:
        batch.append(frame)
        if len(batch) == size:
            yield np.stack(batch)
            batch = []
    if batch:
        yield np.stack(batch)


def release(model: Any) -> None:
    """Drop a model's cached weights and free the device memory they held."""

    model.load.cache_clear()
    free()


def cosine(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Positionally paired cosine similarity of two `(n, D)` embedding sets."""

    normalise = torch.nn.functional.normalize
    return (normalise(torch.as_tensor(left), dim=-1)
            * normalise(torch.as_tensor(right), dim=-1)).sum(-1).numpy()


def embed_video(model: Any, path: str | Path, root: str | Path | None = None,
                stride: int = 1) -> np.ndarray:
    """Return one model's `(n, D)` embeddings of a video's sampled frames."""

    with prefetch(batches(iter_frames(path, stride))) as prepared:
        parts = [model.embed(unit, root=root).cpu().numpy() for unit in prepared]
    if not parts:
        raise ValueError(f"{path}: decoded to zero frames")
    return np.concatenate(parts)


def frame_trajectory(model: Any, path: str | Path, block: int | None = None,
                     root: str | Path | None = None, stride: int = 1) -> np.ndarray:
    """Return `(n, D)`: the spatial average of patch tokens at `block`, per sampled frame."""

    with prefetch(batches(iter_frames(path, stride))) as prepared:
        parts = [backbones.pooled(model, unit, block=block, root=root).cpu().numpy() for unit in prepared]
    if not parts:
        raise ValueError(f"{path}: decoded to zero frames")
    return np.concatenate(parts)


def render_reading(name: str, model: Any, world_root: str | Path, estimates: str | Path,
                   root: str | Path | None = None) -> tuple[np.ndarray, np.ndarray, int]:
    """Embed a world's render and compare it with the cached reference embeddings.

    Returns the render embeddings `(n, D)`, the per-frame cosine curve, and the stride
    stored with the cached reference, which sets the render's sampling.
    """

    with np.load(Path(estimates) / SEMANTIC_FILENAME.format(name=short(name))) as archive:
        reference, stride = archive["embeddings"], int(archive["stride"])
    render = embed_video(model, Path(world_root) / RENDER_FILENAME, root=root, stride=stride)
    if len(render) != len(reference):
        raise ValueError("reference and render must have the same frame count")
    return render, cosine(reference, render), stride


__all__ = [
    "BATCH",
    "MODELS",
    "RENDER_FILENAME",
    "batches",
    "cosine",
    "embed_video",
    "frame_trajectory",
    "free",
    "iter_frames",
    "release",
    "render_reading",
    "short",
]
