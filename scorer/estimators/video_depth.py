"""Video Depth Anything: model assembly, checkpoint loading and windowed video inference.

Adapted from DepthAnything/Video-Depth-Anything (Apache-2.0); the vendored
`video_depth_anything/` package holds the network modules. The model outputs relative
disparity (larger is nearer) up to an unknown per-frame scale and shift. Windows are
stitched by taking each frame from the first window that covers it, with no
cross-window scale alignment, since the Depth metric normalises every frame on its own.
Inference is a generator over frames; peak memory depends on the window, not the length.

Also provides the OpenCV frame decoder (`frame_stream`, `video_info`) the other
estimators use.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


from .video_depth_anything.dinov2 import DINOv2
from .video_depth_anything.dpt_temporal import DPTHeadTemporal

INPUT_SIZE = 518
PATCH = 14
WINDOW = 32
OVERLAP = 8
ENCODER_CHUNK = 8
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
WIDEST_RATIO = 1.78

MODEL_CONFIGS = {
    "vits": dict(encoder="vits", features=64, out_channels=(48, 96, 192, 384)),
    "vitl": dict(encoder="vitl", features=256, out_channels=(256, 512, 1024, 1024)),
}
INTERMEDIATE_LAYERS = {"vits": [2, 5, 8, 11], "vitl": [4, 11, 17, 23]}


class VideoDepthAnything(nn.Module):
    def __init__(
        self,
        encoder: str = "vitl",
        features: int = 256,
        out_channels: list[int] = (256, 512, 1024, 1024),
        use_bn: bool = False,
        use_clstoken: bool = False,
        num_frames: int = WINDOW,
    ):
        super().__init__()
        self.encoder = encoder
        self.pretrained = DINOv2(encoder)
        self.head = DPTHeadTemporal(
            self.pretrained.embed_dim, features, use_bn, out_channels, use_clstoken, num_frames
        )

    def encode(self, images: torch.Tensor) -> tuple:
        """Return the four intermediate token layers, encoding `ENCODER_CHUNK` frames at a time.

        The encoder is per-frame, so chunking leaves the output unchanged.
        """

        layers = INTERMEDIATE_LAYERS[self.encoder]
        parts = [
            self.pretrained.get_intermediate_layers(images[begin : begin + ENCODER_CHUNK], layers)
            for begin in range(0, images.shape[0], ENCODER_CHUNK)
        ]
        return tuple(
            (torch.cat([part[i][0] for part in parts]), torch.cat([part[i][1] for part in parts]))
            for i in range(len(layers))
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map `(B, T, 3, H, W)` normalised frames to `(B, T, H, W)` disparity."""

        _, frames, _, height, width = x.shape
        tokens = self.encode(x.flatten(0, 1))
        depth = self.head(tokens, height // PATCH, width // PATCH, frames)
        return F.relu(depth).squeeze(1).unflatten(0, (-1, frames))


def default_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(
    directory: str | Path,
    encoder: str = "vitl",
    device: str | torch.device | None = None,
) -> VideoDepthAnything:
    """Load `video_depth_anything_<encoder>.pth` from `directory` onto `device`, float32, eval."""

    device = torch.device(device) if device is not None else default_device()
    path = Path(directory) \
        / f"video_depth_anything_{encoder}.pth"
    model = VideoDepthAnything(**MODEL_CONFIGS[encoder])
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True), strict=True)
    return model.to(device=device, dtype=torch.float32).eval()


def _multiple_of(value: float, minimum: int, multiple: int = PATCH) -> int:
    size = int(round(value / multiple) * multiple)
    return max(size, int(np.ceil(minimum / multiple) * multiple))


def input_resolution(width: int, height: int, input_size: int = INPUT_SIZE) -> tuple[int, int]:
    """Return the model input `(w, h)`: MiDaS `lower_bound` resize, sides multiples of `PATCH`."""

    ratio = max(width, height) / min(width, height)
    if ratio > WIDEST_RATIO:
        input_size = int(round(input_size * (WIDEST_RATIO - 0.003) / ratio / PATCH) * PATCH)
    scale = max(input_size / height, input_size / width)
    return _multiple_of(scale * width, input_size), _multiple_of(scale * height, input_size)




def preprocess(frames: Sequence[np.ndarray], input_size: int = INPUT_SIZE) -> torch.Tensor:
    """Convert a window of RGB `(H, W, 3)` uint8 frames to normalised `(T, 3, h, w)` input."""

    height, width = frames[0].shape[:2]
    target = input_resolution(width, height, input_size)
    resized = np.stack(
        [
            cv2.resize(frame.astype(np.float32) / 255.0, target, interpolation=cv2.INTER_CUBIC)
            for frame in frames
        ]
    )
    resized = (resized - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(resized.transpose(0, 3, 1, 2).copy())


def video_info(path: str | Path) -> tuple[int, int, int, float]:
    """Return `(frames, height, width, fps)` from the container metadata, without decoding."""

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"{path}: unreadable video")
    info = (
        int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
        int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
        float(capture.get(cv2.CAP_PROP_FPS)),
    )
    capture.release()
    return info


def frame_stream(path: str | Path, stride: int = 1) -> Iterator[np.ndarray]:
    """Yield frames `0, stride, 2 * stride, ...` as RGB `(H, W, 3)` uint8.

    Skipped frames are grabbed but not decoded.
    """

    stride = max(1, int(stride))
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"{path}: unreadable video")
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


def cached_disparity(disparity: np.ndarray, video: str | Path, frame: int) -> np.ndarray:
    """Return `disparity` as float16; raise if it or its float16 cast is non-finite."""

    if not np.isfinite(disparity).all():
        raise ValueError(f"{video}: frame {frame}: VDA disparity is non-finite")
    with np.errstate(over="ignore"):
        stored = disparity.astype(np.float16)
    if not np.isfinite(stored).all():
        raise ValueError(f"{video}: frame {frame}: VDA disparity overflows float16 storage")
    return stored


@torch.inference_mode()
def stream_video_depth(
    path: str | Path,
    model: VideoDepthAnything,
    device: str | torch.device | None = None,
    input_size: int = INPUT_SIZE,
    window: int = WINDOW,
    overlap: int = OVERLAP,
    stride: int = 1,
) -> Iterator[np.ndarray]:
    """Yield disparity `(h, w)` float32 per sampled frame, on the `input_resolution` grid.

    Each window of `window` frames starts with the previous window's last `overlap`
    frames as temporal context; each frame is yielded once. A window shorter than
    `window` is padded by repeating its last frame, as the released inference script
    does, so a video of any length, down to one frame, yields output.
    """

    device = torch.device(device) if device is not None else default_device()
    source = frame_stream(path, stride)
    frames: list[np.ndarray] = []
    context = 0

    while True:
        ended = False
        while len(frames) < window:
            frame = next(source, None)
            if frame is None:
                ended = True
                break
            frames.append(frame)
        fresh = len(frames) - context
        if fresh <= 0:
            return

        # padded frames are never yielded
        padded = frames + [frames[-1]] * (window - len(frames))
        batch = preprocess(padded, input_size).unsqueeze(0).to(device)
        disparity = model(batch)
        # moved to the CPU so a suspended generator holds no GPU memory
        disparity = disparity.float()[0].cpu()
        del batch
        for offset in range(context, context + fresh):
            yield disparity[offset].numpy().copy()
        del disparity, padded

        if ended:
            return
        frames = frames[-overlap:]
        context = len(frames)
