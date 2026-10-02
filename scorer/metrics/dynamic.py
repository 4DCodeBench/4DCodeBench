"""Dynamic IoU: per-frame IoU of the rasterised dynamic-object masks, averaged over frames.

An object is dynamic if a `dynamics/` archive lists its id. Each mask comes from the
scorer's z-buffer over the whole scene, so static geometry occludes dynamic geometry.
A band `BAND_FRACTION` of the image diagonal wide along the reference boundary is
excluded on both sides, as PASCAL VOC's void label. On a real case the reference mask
is a hand annotation (`Annotation`), and the submission mask is first closed with a
radius of `CLOSE_FRACTION` of the diagonal so that granular matter forms one region.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from config.annotation import MASK_FILENAME, MASK_KEY
from config.scorer import BAND_FRACTION, CLOSE_FRACTION

from ..world import World



def mean_iou(gt: np.ndarray, pred: np.ndarray) -> float:
    """Mean over frames of the IoU of two `(T, H, W)` masks; a frame with empty union scores 1."""

    intersection = np.count_nonzero(gt & pred, axis=(1, 2))
    union = np.count_nonzero(gt | pred, axis=(1, 2))
    return float(np.where(union == 0, 1.0, intersection / np.maximum(union, 1)).mean())


def close(mask: np.ndarray, radius: int) -> np.ndarray:
    """Morphologically close each frame of a `(T, H, W)` mask with a disc of `radius`."""

    element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    closed = np.empty(mask.shape, dtype=bool)
    for index, frame in enumerate(mask.view(np.uint8)):
        closed[index] = cv2.morphologyEx(frame, cv2.MORPH_CLOSE, element).astype(bool)
    return closed


def boundary_band(gt: np.ndarray, radius: int) -> np.ndarray:
    """Band of each frame's mask boundary: dilation minus erosion by a disc of `radius`."""

    element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    band = np.empty(gt.shape, dtype=bool)
    for index, frame in enumerate(gt.view(np.uint8)):
        band[index] = cv2.dilate(frame, element).astype(bool) & ~cv2.erode(frame, element).astype(bool)
    return band


def overlap(readings: dict, masks, shape: tuple[int, int, int]) -> dict:
    """Compute band-excluded Dynamic IoU from a stream of `(gt, pred)` frame masks.

    `readings` must hold the `dynamic_gt` and `dynamic_pred` id lists. The bit-packed
    gt, pred and band masks are stored under `arrays["dynamic"]`.
    """

    frames, height, width = shape
    radius = max(1, round(BAND_FRACTION * float(np.hypot(height, width))))
    packed = {name: np.empty((frames, height, (width + 7) // 8), dtype=np.uint8)
              for name in ("gt", "pred", "band")}
    pixels = np.zeros(2, dtype=np.int64)
    scores = []
    for frame, (gt, pred) in zip(range(frames), masks, strict=True):
        band = boundary_band(gt[None], radius)[0]
        pixels += [np.count_nonzero(gt), np.count_nonzero(pred)]
        scores.append(mean_iou((gt & ~band)[None], (pred & ~band)[None]))
        for name, mask in (("gt", gt), ("pred", pred), ("band", band)):
            packed[name][frame] = np.packbits(mask, axis=-1)
    readings["dynamic_pixels"] = pixels.tolist()
    readings["dynamic_band_px"] = radius
    readings["dynamic_iou"] = float(np.mean(scores))
    readings["arrays"] = {"dynamic": {
        **packed, "shape": np.asarray(shape, dtype=np.int64),
        "gt_ids": np.asarray(readings["dynamic_gt"]["ids"], dtype=np.uint16),
        "pred_ids": np.asarray(readings["dynamic_pred"]["ids"], dtype=np.uint16),
    }}
    return readings


def mask_frames(world: World):
    """Yield each frame's boolean mask of pixels showing a dynamic object."""

    from ..raster import id_frame

    for frame in range(len(world)):
        yield np.isin(id_frame(world, frame), world.dynamic_ids)


def dynamic_iou(gt: World, pred: World) -> dict:
    """Dynamic IoU of a submission against a reference world of the same `(F, H, W)`."""

    if gt.shape != pred.shape:
        raise ValueError(f"reference is {gt.shape} and the submission is {pred.shape}")
    readings = {f"dynamic_{side}": {"ids": world.dynamic_ids.tolist()}
                for side, world in (("gt", gt), ("pred", pred))}
    return overlap(readings, zip(mask_frames(gt), mask_frames(pred), strict=True), gt.shape)


class Annotation:
    """A real case's hand-annotated dynamic mask, `(F, H, W)` bool."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        with np.load(self.root / MASK_FILENAME) as data:
            self.mask = data[MASK_KEY].astype(bool, copy=False)

    def __repr__(self) -> str:
        return f"Annotation({self.root}, shape={self.mask.shape})"


def annotated_dynamic_iou(annotation: Annotation, pred: World) -> dict:
    """Dynamic IoU against an annotated mask, with the submission mask closed first."""

    if annotation.mask.shape != pred.shape:
        raise ValueError(f"annotation is {annotation.mask.shape} and the submission "
                         f"is {pred.shape}")
    readings: dict = {
        "dynamic_gt": {"ids": [1], "source": "annotation"},
        "dynamic_pred": {"ids": [int(value) for value in pred.dynamic_ids]},
    }
    height, width = annotation.mask.shape[1:]
    radius = max(1, round(CLOSE_FRACTION * float(np.hypot(height, width))))
    readings["dynamic_close_px"] = radius
    masks = ((truth, close(mask[None], radius)[0])
             for truth, mask in zip(annotation.mask, mask_frames(pred), strict=True))
    return overlap(readings, masks, pred.shape)


__all__ = ["Annotation", "annotated_dynamic_iou", "boundary_band", "dynamic_iou", "mean_iou",
           "overlap"]
