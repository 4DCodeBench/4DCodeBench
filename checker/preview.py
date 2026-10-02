"""Draw one frame of the exported world the way the scorer will see it.

    python -m checker.preview            # frame 0 of /workspace/world -> /workspace/tmp/preview_0000.png

The scorer never looks at `render.mp4` to learn where the geometry is: it projects `meshes/`
through `camera.json`. This tool does the same projection with a plain z-buffer rasteriser
and writes one PNG with two panels: the frame of `render.mp4` on the left, the exported
geometry seen through the exported camera on the right (one colour per mesh id, lit by its
face normal). Look at it: if the right panel does not line up with the left one, `camera.json`
or `meshes/` do not describe what was rendered, and every metric will be scored against the
wrong geometry. It judges nothing and prints nothing.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from config.world import (
    CAMERA_EXTRINSIC_KEY,
    CAMERA_FILENAME,
    CAMERA_INTRINSICS_KEY,
    MESH_FACES_KEY,
    MESH_VERTICES_KEY,
    MESHES_DIRNAME,
    NPZ_SUFFIX,
    VIDEO_FILENAME,
)

from .paths import WORKSPACE_DIR, WORLD_DIR

PREVIEW_DIR = WORKSPACE_DIR / "tmp"
FRAME = 0
BACKGROUND = np.array([40, 40, 40], dtype=np.uint8)


def preview(world: Path, frame: int, out: Path) -> None:
    """Write the two-panel image for `frame`."""

    camera = json.loads((world / CAMERA_FILENAME).read_text(encoding="utf-8"))
    intrinsics = np.asarray(camera[CAMERA_INTRINSICS_KEY], dtype=np.float64)
    extrinsic = np.asarray(camera[CAMERA_EXTRINSIC_KEY], dtype=np.float64)

    rendered = _video_frame(world / VIDEO_FILENAME, frame)
    height, width = rendered.shape[:2]

    drawn = np.tile(BACKGROUND, (height, width, 1))
    depth = np.full((height, width), np.inf)
    for path in sorted((world / MESHES_DIRNAME).glob(f"*{NPZ_SUFFIX}")):
        oid = int(path.stem)
        with np.load(path) as data:
            vkey, fkey = MESH_VERTICES_KEY.format(frame=frame), MESH_FACES_KEY.format(frame=frame)
            if vkey not in data or not len(data[fkey]):
                continue
            vertices, faces = data[vkey].astype(np.float64), data[fkey]
        camera_points = (vertices - extrinsic[:3, 3]) @ extrinsic[:3, :3]
        _rasterise(camera_points, faces, intrinsics, _colour(oid), drawn, depth)

    out.parent.mkdir(parents=True, exist_ok=True)
    _write_png(np.concatenate([rendered, drawn], axis=1), out)


def _rasterise(points: np.ndarray, faces: np.ndarray, intrinsics: np.ndarray,
               colour: np.ndarray, drawn: np.ndarray, depth: np.ndarray) -> None:
    """Z-buffer every triangle in front of the camera into `drawn`."""

    height, width = depth.shape
    tri = points[faces]                                                  # (T, 3, 3)
    in_front = (tri[:, :, 2] > 1e-6).all(axis=1)
    tri = tri[in_front]
    if not len(tri):
        return
    normal = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    normal /= np.linalg.norm(normal, axis=1, keepdims=True) + 1e-12
    shade = 0.35 + 0.65 * np.abs(normal[:, 2])                          # headlight from the camera
    pixels = tri @ intrinsics.T
    uv = pixels[:, :, :2] / pixels[:, :, 2:3]                             # (T, 3, 2)
    z = tri[:, :, 2]
    lo = np.clip(np.floor(uv.min(axis=1)), 0, [width - 1, height - 1]).astype(int)
    hi = np.clip(np.ceil(uv.max(axis=1)), 0, [width - 1, height - 1]).astype(int)
    visible = ((uv.max(axis=1) >= 0) & (uv.min(axis=1) < [width, height])).all(axis=1)
    order = np.argsort(-z.mean(axis=1))                                  # far to near: fewer overdraw checks
    for t in order:
        if not visible[t]:
            continue
        (x0, y0), (x1, y1) = lo[t], hi[t]
        xs, ys = np.meshgrid(np.arange(x0, x1 + 1) + 0.5, np.arange(y0, y1 + 1) + 0.5)
        (ax, ay), (bx, by), (cx, cy) = uv[t]
        area = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)
        if abs(area) < 1e-12:
            continue
        w0 = ((bx - xs) * (cy - ys) - (by - ys) * (cx - xs)) / area
        w1 = ((cx - xs) * (ay - ys) - (cy - ys) * (ax - xs)) / area
        w2 = 1.0 - w0 - w1
        inside = (w0 >= 0) & (w1 >= 0) & (w2 >= 0)
        if not inside.any():
            continue
        # Screen-space weights interpolate inverse depth under perspective.
        inverse_depth = w0 / z[t, 0] + w1 / z[t, 1] + w2 / z[t, 2]
        zpix = np.full(w0.shape, np.inf, dtype=np.float32)
        np.divide(1.0, inverse_depth, out=zpix, where=inside)
        window = depth[y0:y1 + 1, x0:x1 + 1]
        nearer = inside & (zpix < window)
        window[nearer] = zpix[nearer]
        drawn[y0:y1 + 1, x0:x1 + 1][nearer] = (colour * shade[t]).astype(np.uint8)


def _colour(oid: int) -> np.ndarray:
    """A distinct, bright colour per id (golden-angle hue)."""

    hue = (oid * 0.618033988749895) % 1.0
    k = np.array([0.0, 2.0, 4.0])
    rgb = np.clip(np.abs(((hue * 6.0 + k) % 6.0) - 3.0) - 1.0, 0.0, 1.0)
    return (255 * (0.25 + 0.75 * rgb))


def _video_frame(path: Path, frame: int) -> np.ndarray:
    probe = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                            "stream=width,height", "-of", "csv=p=0", str(path)],
                           capture_output=True, text=True, check=True)
    width, height = (int(v) for v in probe.stdout.strip().split(",")[:2])
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-vf", f"select=eq(n\\,{frame})",
                          "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                         capture_output=True, check=True).stdout
    if len(raw) != width * height * 3:
        raise SystemExit(f"{path}: could not decode frame {frame}")
    return np.frombuffer(raw, dtype=np.uint8).reshape(height, width, 3)


def _write_png(image: np.ndarray, out: Path) -> None:
    height, width = image.shape[:2]
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
                    "-s", f"{width}x{height}", "-i", "-", str(out)],
                   input=np.ascontiguousarray(image).tobytes(), check=True)


def main() -> int:
    preview(WORLD_DIR, FRAME, PREVIEW_DIR / f"preview_{FRAME:04d}.png")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 1:
        raise SystemExit("checker.preview takes no arguments")
    raise SystemExit(main())
