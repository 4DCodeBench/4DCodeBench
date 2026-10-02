"""Track2D: per-step DTW distance between corresponding reference and submission 2D tracks.

Both sides start from the same grid of queries on frame 0
(`scorer.analytic.tracks.grid_queries`), so path `i` of one side corresponds to path
`i` of the other. The reference paths are CoTracker3 tracks of `reference.mp4`, cached
in `estimates/`; the submission paths follow each query's surface point through the
4D world and project it with a z-buffer visibility test (`scorer.analytic.tracks`).
Both are on the sampled timeline (`config.sampling`).

A query is scored only if (1) its reference path, after subtracting the grid's median
displacement per frame, gets more than `MOTION` image diagonals from its start, and
(2) its seed pixel in reference frame 0 has a local grey-level standard deviation
above `TEXTURE` over a `WINDOW` patch. With `D_i` the per-step DTW distance in image
diagonals, between the reference path on its visible frames and the submission path
on those of them where it is defined,

    track2d_dtw_score = max(0, 1 - mean_i min(D_i, CAP) / CAP).

A pair with no answered submission frame costs `CAP`.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from config.estimates import TRACKS_FILENAME

from .distances import device
from .dtw import dtw_pairs

CAP = 0.03           # per-pair DTW cost cap, in image diagonals
MOTION = 0.02        # minimum drift-corrected travel of a scored reference path, in diagonals
TEXTURE = 0.5        # minimum local grey-level standard deviation at a scored seed pixel
WINDOW = 11          # side of the square patch `TEXTURE` is computed over, in pixels


def _drift(displacement: np.ndarray, visible: np.ndarray,
           background: np.ndarray | None = None) -> np.ndarray:
    """Median displacement per frame, `(F, 2)` px, over the paths visible on that frame.

    When `background` is given and non-empty, only those paths enter the median, so the
    dynamic object's own motion is not subtracted from its paths.
    """

    rows, seen_rows = displacement, visible
    if background is not None and bool(background.any()):
        rows, seen_rows = displacement[background], visible[background]
    seen = np.ma.masked_array(rows, ~np.repeat(seen_rows[..., None], 2, axis=2))
    return np.ma.median(seen, axis=0).filled(0.0)


def _travel(tracks: np.ndarray, visible: np.ndarray, origin: np.ndarray,
            background: np.ndarray | None = None) -> np.ndarray:
    """Maximum distance of each path from its query point over visible frames, `(Q,)` px.

    The grid's median displacement (`_drift`) is subtracted first, which removes camera
    motion shared by all paths.
    """

    if not tracks.shape[1]:
        return np.zeros(len(tracks))
    displacement = np.nan_to_num(tracks) - origin[:, None, :]
    gap = np.linalg.norm(displacement - _drift(displacement, visible, background)[None], axis=2)
    return np.where(visible, gap, 0.0).max(axis=1)


def _trackable(video: str | Path | None, queries: np.ndarray,
               count: int) -> np.ndarray | None:
    """`(Q,)` mask of queries whose seed pixel has local standard deviation above `TEXTURE`.

    The deviation is computed over a `WINDOW` box on the grey reference frame 0, where
    an untextured seed gives the tracker no gradient to follow. Returns None when the
    video is missing or unreadable, or when the queries do not fit its frame or count.
    """

    if video is None or not Path(video).is_file():
        return None
    capture = cv2.VideoCapture(str(video))
    try:
        ok, frame = capture.read()
    finally:
        capture.release()
    if not ok or frame is None:
        return None
    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
    mean = cv2.boxFilter(grey, -1, (WINDOW, WINDOW), borderType=cv2.BORDER_REFLECT)
    square = cv2.boxFilter(grey * grey, -1, (WINDOW, WINDOW), borderType=cv2.BORDER_REFLECT)
    deviation = np.sqrt(np.maximum(square - mean * mean, 0.0))
    height, width = grey.shape[:2]
    x, y = np.rint(queries[:, 1]).astype(int), np.rint(queries[:, 2]).astype(int)
    if len(x) != count or x.min() < 0 or y.min() < 0 or x.max() >= width or y.max() >= height:
        return None
    return deviation[y, x] > TEXTURE


def _paths(tracks: np.ndarray, alive: np.ndarray, diagonal: float) -> np.ndarray:
    """One side's paths in image diagonals, `(Q, F, 3)` with zero z, NaN where not `alive`."""

    flat = np.nan_to_num(tracks) / diagonal
    padded = np.concatenate([flat, np.zeros((*flat.shape[:2], 1), np.float32)], axis=2)
    return np.where(alive[..., None], padded, np.nan).astype(np.float32)


def _dtw(gt: np.ndarray, pred: np.ndarray, answered: np.ndarray) -> np.ndarray:
    """Per-pair DTW cost `(Q,)`, capped at `CAP`; `CAP` for a pair with no answered frame."""

    lives = answered.any(axis=1)
    cost = np.full(len(gt), CAP, np.float32)
    if lives.any():
        paired = dtw_pairs(gt[lives], pred[lives], device())
        cost[lives] = np.minimum(np.nan_to_num(paired, nan=CAP, posinf=CAP), CAP)
    return cost


def costs(gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """Per-pair DTW cost of `(Q, F, 2|3)` paths in diagonals; NaN frames are dropped per path."""

    answered = np.isfinite(pred[..., 0])
    flat = [np.concatenate([side[..., :2], side[..., :1] * 0], axis=2) for side in (gt, pred)]
    return _dtw(*flat, answered)


def track2d_reading(reference: str | Path, shape: tuple[int, int, int], submission,
                    video: str | Path | None) -> dict:
    """Compute the Track2D readings; `submission(queries, stride)` supplies the submission side.

    It returns `(tracks (Q, F', 2) float32, answered (Q, F') bool)` in pixels on the
    sampled timeline for the archive's `(Q, 3)` `(t, x, y)` queries. `video` is the
    reference video used by the texture gate.
    """

    diagonal = float(np.hypot(*shape[1:]))
    with np.load(Path(reference) / TRACKS_FILENAME) as archive:
        tracks, visible, queries = archive["tracks"], archive["visible"], archive["queries"]
        stride, on_mask = int(archive["stride"]), archive["on_mask"]

    background = ~np.asarray(on_mask, dtype=bool)
    # The texture gate removes a query from both sides.
    trackable = _trackable(video, queries, len(tracks))
    moving = _travel(tracks, visible, queries[:, 1:], background) > MOTION * diagonal
    ungated = int(moving.sum())
    if trackable is not None:
        moving &= trackable
    counts = [len(tracks), int(moving.sum())]
    answer, answered = submission(queries, stride)
    if answer.shape[:2] != tracks.shape[:2]:
        raise ValueError(f"Submission paths must be {tracks.shape[:2]}, not {answer.shape[:2]}")
    gate = {"track2d_trackable": None if trackable is None else int(trackable.sum()),
            "track2d_moving_ungated": ungated, "track2d_texture": TEXTURE,
            "track2d_extracted": bool(answered.any())}
    if not counts[1] or not answered.any():
        empty = {"moving": moving, "motion": np.float32(MOTION), "stride": np.int64(stride)}
        if trackable is not None:
            empty["trackable"] = trackable
            empty["texture"] = np.float32(TEXTURE)
        return {"arrays": {"track2d": empty},
                "track2d_queries": counts[0], "track2d_moving": counts[1],
                "track2d_stride": stride, **gate, "track2d_dtw_error": None}

    gt = _paths(tracks[moving], visible[moving], diagonal)
    attempt = _paths(answer[moving], (visible & answered)[moving], diagonal)

    arrays = {"gt": gt[..., :2], "pred": attempt[..., :2], "cap": np.float32(CAP), "moving": moving,
              "motion": np.float32(MOTION), "stride": np.int64(stride)}
    arrays["on_mask"] = np.asarray(on_mask, dtype=bool)[moving]
    if trackable is not None:
        arrays["trackable"] = trackable
        arrays["texture"] = np.float32(TEXTURE)
    cost = costs(gt, attempt)
    error = float(cost.mean() / CAP)
    arrays["cost_dtw"] = cost.astype(np.float32)
    readings = {"track2d_dtw_error": error, "track2d_dtw_score": float(max(0.0, 1.0 - error))}
    return {
        "arrays": {"track2d": arrays},
        "track2d_queries": counts[0],
        "track2d_moving": counts[1],
        "track2d_stride": stride,
        "track2d_frames": int(gt.shape[1]),
        **gate,
        **readings,
    }


__all__ = ["CAP", "MOTION", "TEXTURE", "WINDOW", "costs", "track2d_reading"]
