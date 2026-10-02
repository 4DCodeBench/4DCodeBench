"""Depth error: normalised disparity error of the submitted geometry against monocular depth.

The reference is the Video Depth Anything disparity of `reference.mp4`, cached in the
case's `estimates/` by `python -m scorer.prepare`. The submission side is `1 / z` of the
z-buffer depth of `meshes/` through `camera.json`. On the covered pixels each side is
normalised by its median and mean absolute deviation (as in MiDaS), and the frame
error `E` is the mean absolute difference with the largest `TRIM` fraction of
residuals dropped. An uncovered pixel counts as 1, so at coverage `c` a frame's error
is `c * E + (1 - c)`. `depth_error` is the mean over the sampled timeline
(`config.sampling`), in [0, 2]; lower is better.

The stored arrays hold the normalised submission disparity on the cached reference
grid, with the reference's per-frame median and deviation.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from config.estimates import DEPTH_FILENAME
from config.sampling import sampled_indices

from ..raster import Geometry, default_device
from ..storage import open_dense

EPSILON = 1e-8
TRIM = 0.01      # fraction of the largest residuals dropped per frame, as in MiDaS


def normalise(values: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Return `(values - median) / deviation`, the median and the mean absolute deviation."""

    median = float(np.median(values))
    deviation = max(float(np.mean(np.abs(values - median))), EPSILON)
    return (values - median) / deviation, median, deviation


def frame_reading(reference: np.ndarray, prediction: np.ndarray,
                  valid: np.ndarray) -> tuple[float, np.ndarray, float, float]:
    """One frame's error, normalised prediction, and reference median and deviation.

    Both sides are normalised on `valid`. A frame with no valid pixel scores 1, and
    its reference is normalised on the whole frame.
    """

    _, median, deviation = normalise(reference[valid] if valid.any() else reference)
    attempt = np.zeros_like(prediction)
    if not valid.any():
        return 1.0, attempt, median, deviation
    attempt[valid] = normalise(prediction[valid])[0]
    residual = np.abs((reference[valid] - median) / deviation - attempt[valid])
    kept = residual.size - int(residual.size * TRIM)
    coverage = float(valid.mean())
    covered = float(np.partition(residual, kept - 1)[:kept].mean())
    return coverage * covered + (1.0 - coverage), attempt, median, deviation


def raster_disparity(geometry: Geometry, frame: int, device=None) -> tuple[np.ndarray, np.ndarray]:
    """Return one frame's inverse rasterised depth (0 where uncovered) and its coverage mask."""

    depth, _ = geometry.depth(frame, device)
    valid = np.isfinite(depth) & (depth > 0)
    disparity = np.zeros_like(depth)
    np.divide(1.0, depth, out=disparity, where=valid)
    return disparity, valid


def upsample(values: np.ndarray, height: int, width: int) -> np.ndarray:
    """Bilinearly resize a cached map to `(height, width)`, as the model's inference does."""

    grid = torch.from_numpy(values.astype(np.float32))[None, None]
    return F.interpolate(grid, (height, width), mode="bilinear", align_corners=True)[0, 0].numpy()


def to_grid(values: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Area-average a video-resolution map down to the cached reference grid."""

    return cv2.resize(values, (shape[1], shape[0]), interpolation=cv2.INTER_AREA)


def depth_error(world_root: str | Path, estimates: str | Path,
                device: torch.device | str | None = None) -> dict:
    """Compute the depth readings of one submitted world against the case's cached disparity.

    Returns scalar and per-frame readings, with arrays under `arrays["raster_depth"]`.
    """

    device = torch.device(device) if device is not None else default_device()

    with open_dense(Path(estimates) / DEPTH_FILENAME) as cached:
        reference_maps = cached["disparity"]
        count, height, width = (int(value) for value in cached["shape"][()])
        stride, fps = int(cached["stride"][()]), float(cached["fps"][()])
        geometry = Geometry(world_root, height, width)
        shape = (len(geometry), height, width)
        if (count, height, width) != shape:
            raise ValueError(f"cached depth of {count} frames at {(height, width)} does not match {shape}")
        read = sampled_indices(count, stride)
        grid = reference_maps.shape[1:]

        errors, coverage = [], []
        raster, covered, medians, deviations = [], [], [], []
        for step in range(min(len(reference_maps), len(read))):
            frame = int(read[step])
            reference = upsample(reference_maps[step], height, width)
            disparity, valid = raster_disparity(geometry, frame, device)
            error, attempt, median, deviation = frame_reading(reference, disparity, valid)
            errors.append(error)
            coverage.append(float(valid.mean()))
            raster.append(to_grid(attempt, grid).astype(np.float16))
            covered.append(to_grid(valid.astype(np.float32), grid) > 0.5)
            medians.append(median)
            deviations.append(deviation)

    return {
        "depth_error": float(np.mean(errors)),
        "depth_frames": len(errors),
        "depth_stride": stride,
        "depth_resolution": [height, width],
        "fps": fps,
        "depth_per_frame": [round(value, 6) for value in errors],
        "depth_valid_fraction": [round(value, 6) for value in coverage],
        "arrays": {
            "raster_depth": {"disparity": np.stack(raster), "valid": np.stack(covered),
                             "median": np.asarray(medians, dtype=np.float32),
                             "deviation": np.asarray(deviations, dtype=np.float32)},
        },
    }


__all__ = ["EPSILON", "TRIM", "depth_error", "frame_reading", "normalise", "raster_disparity",
           "to_grid", "upsample"]
