"""Shared rendering helpers: the mp4 writer, the colour scale and the object palette.

Every mp4 is H.264 yuv420p with even sides, one image per frame and one video per
side; the viewer lays the sides out and labels them. Only `tracks` draws a caption
on the image.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import cv2
import numpy as np

SPAN = 2.5          # the shared colour scale, in robustly normalised disparity units


class Writer:
    """An ffmpeg pipe that encodes BGR frames one at a time."""

    def __init__(self, path: str | Path, fps: float):
        self.path, self.fps, self.process, self.size = Path(path), fps, None, None

    def _start(self, frame: np.ndarray) -> None:
        height, width = frame.shape[:2]
        # yuv420p needs both sides even, so the canvas is rounded up once and reused
        self.size = (height + height % 2, width + width % 2)
        command = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-f", "rawvideo", "-pix_fmt", "bgr24",
            "-s", f"{self.size[1]}x{self.size[0]}", "-framerate", f"{self.fps:g}",
            "-i", "-", "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
            "-g", "12",     # a keyframe every 12 frames, for seeking
            str(self.path),
        ]
        self.process = subprocess.Popen(command, stdin=subprocess.PIPE)

    def write(self, frame: np.ndarray) -> None:
        if self.process is None:
            self._start(frame)
        canvas = np.zeros((*self.size, 3), dtype=np.uint8)
        canvas[: frame.shape[0], : frame.shape[1]] = frame
        self.process.stdin.write(canvas.tobytes())

    def close(self) -> None:
        if self.process is None:
            return
        self.process.stdin.close()
        if self.process.wait() != 0:
            raise RuntimeError(f"ffmpeg failed for {self.path}")

def colourise(values: np.ndarray, valid: np.ndarray, span: float = SPAN,
              low: float = -1.0) -> np.ndarray:
    """Turbo colours on the shared scale `[low * span, span]`; invalid pixels stay black."""

    scaled = (values - low * span) / (span - low * span)
    image = cv2.applyColorMap((np.clip(scaled, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    image[~valid] = 0
    return image

def fps_of(video: Path) -> float:
    capture = cv2.VideoCapture(str(video))
    rate = capture.get(cv2.CAP_PROP_FPS) or 24.0
    capture.release()
    return float(rate)


def frames_of(video: Path, stride: int = 1):
    """Yield a video's BGR frames in order.

    With `stride > 1`, only frames `0, k, 2k, ...` are decoded; the others are grabbed
    and skipped.
    """

    stride = max(1, int(stride))
    capture = cv2.VideoCapture(str(video))
    try:
        index = 0
        while True:
            if not capture.grab():
                return
            if index % stride == 0:
                ok, frame = capture.retrieve()
                if not ok:
                    return
                yield frame
            index += 1
    finally:
        capture.release()

TINT = np.array((60, 136, 224), np.uint8)   # BGR: a dynamic mask over a render
OVERLAY = 0.55                              # weight of the ids over the render


def tinted(shot: np.ndarray, mask: np.ndarray, ink: np.ndarray = TINT, weight: float = 0.55) -> np.ndarray:
    """Return the render tinted under `mask`, resizing the mask to the render if needed."""

    if mask.shape != shot.shape[:2]:
        mask = cv2.resize(mask.astype(np.uint8), (shot.shape[1], shot.shape[0]),
                          interpolation=cv2.INTER_NEAREST).astype(bool)
    out = shot.copy()
    out[mask] = ((1 - weight) * shot[mask] + weight * ink).astype(np.uint8)
    return out


# Object colour as a function of the id alone, so an object keeps it across frames
# and worlds: hue steps by the golden angle, value alternates. Id 0 (background) is dark.
GOLDEN_ANGLE = 0.618033988749895
EMPTY_INK = (26, 22, 20)


def palette(top: int) -> np.ndarray:
    """BGR ink for every id up to `top`, inclusive."""

    ids = np.arange(top + 1, dtype=np.float64)
    wheel = (ids * GOLDEN_ANGLE) % 1 * 6
    value = 255 * (1 - 0.22 * (ids % 2))
    low = value * (1 - 0.78)
    rise = low + (value - low) * (wheel % 1)
    fall = value + low - rise
    sixth = np.stack([np.stack([value, rise, low]), np.stack([fall, value, low]), np.stack([low, value, rise]),
                      np.stack([low, fall, value]), np.stack([rise, low, value]), np.stack([value, low, fall])])
    rgb = sixth[(wheel.astype(int) % 6), :, np.arange(top + 1)]
    inks = np.rint(rgb[:, ::-1]).astype(np.uint8)
    inks[0] = EMPTY_INK[::-1]
    return inks


def stamp(canvas: np.ndarray, points: np.ndarray, inks: np.ndarray, radius: int) -> None:
    """Draw a square of `inks` at every point, clipped to the canvas, in place."""

    height, width = canvas.shape[:2]
    x = np.clip(np.round(points[:, 0]).astype(np.int64), 0, width - 1)
    y = np.clip(np.round(points[:, 1]).astype(np.int64), 0, height - 1)
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            canvas[np.clip(y + dy, 0, height - 1), np.clip(x + dx, 0, width - 1)] = inks


__all__ = ["EMPTY_INK", "GOLDEN_ANGLE", "OVERLAY", "SPAN", "TINT", "Writer", "colourise",
           "fps_of", "frames_of", "palette", "stamp", "tinted"]
