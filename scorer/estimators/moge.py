"""MoGe-3 point map of frame 0: the reference cloud of the Uni3D MoGe metric.

MoGe-3 ViT-L (microsoft/MoGe, `Ruicheng/moge-3-vitl`, official `moge` package)
predicts a camera-space point map and a validity mask from one RGB frame. The points
are in metres up to an unknown global scale; the Uni3D metric normalises each cloud
by its centre and radius before embedding it. Runs in `python -m scorer.prepare`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


WEIGHTS = "model.pt"
RESOLUTION_LEVEL = 9    # token budget level; 9 is the finest
REFINE_STEPS = 3        # sparse volumetric refinement passes (MoGe-3 default)


def default_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(directory: str | Path,
               device: str | torch.device | None = None):
    """Load MoGe-3 ViT-L from `directory` onto `device` in evaluation mode."""

    from moge.model.v3 import MoGeModel

    device = torch.device(device) if device is not None else default_device()
    path = Path(directory) / WEIGHTS
    return MoGeModel.from_pretrained(path).to(device).eval()


@torch.inference_mode()
def point_map(frame: np.ndarray, model, device: str | torch.device | None = None) -> dict:
    """Return `points`, `mask` and `intrinsics` for one RGB `(H, W, 3)` uint8 frame.

    `points` is `(H, W, 3)` float32 in camera axes (x right, y down, z forward), NaN
    outside `mask` `(H, W)` bool. `intrinsics` is the estimated `(3, 3)` matrix
    normalised to the unit image; it is stored as metadata only.
    """

    device = torch.device(device) if device is not None else default_device()
    image = torch.as_tensor(np.asarray(frame, dtype=np.float32) / 255.0,
                            device=device).permute(2, 0, 1)
    output = model.infer(image, resolution_level=RESOLUTION_LEVEL, refine_steps=REFINE_STEPS)
    mask = output["mask"].cpu().numpy().astype(bool)
    points = output["points"].float().cpu().numpy().astype(np.float32)
    points[~mask] = np.nan
    return {"points": points, "mask": mask,
            "intrinsics": output["intrinsics"].float().cpu().numpy().astype(np.float32)}


__all__ = ["REFINE_STEPS", "RESOLUTION_LEVEL", "WEIGHTS", "load_model", "point_map"]
