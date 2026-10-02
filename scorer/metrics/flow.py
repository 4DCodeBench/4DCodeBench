"""Flow distribution: sliced Wasserstein distance between reference and submission flow.

The reference is RAFT flow of `reference.mp4`, cached in the case's `estimates/`; the
submission flow is computed analytically from its geometry (`scorer.analytic.flow`).
Steps are consecutive frames of the sampled timeline (`config.sampling`). At step `k`
the compared pixels are those valid on both sides and moving more than `MOVING` px on
either, and their flow vectors are compared as 2D point sets:

    gap_k = min(DIST_CAP, SW(U_k, V_k) / (m(U_k) + m(V_k)))

with `m` the per-component RMS; `flow_distribution_score` is one minus the mean gap.
A step whose reference moves where the submission has no coverage counts as 1.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from config.estimates import FLOW_FILENAME
from config.sampling import sampled_length

from ..storage import open_dense
from .distances import seed, semd

MOVING = 0.5     # px: a pixel counts only if either side's flow exceeds this
SAMPLE = 4096    # flow vectors drawn per step, the same pixels on both sides
DIST_CAP = 1.0   # upper bound of one step's normalised gap


def frame_status(reference_valid, valid, reference_moving, count: int) -> str:
    """Classify one step as `ok`, `no_reference_coverage`, `no_prediction_coverage` or `stationary`.

    Only `ok` and `no_prediction_coverage` steps enter the score.
    """

    if count:
        return "ok"
    if not reference_valid.any():
        return "no_reference_coverage"
    if not (reference_valid & valid).any() or (reference_valid & reference_moving).any():
        return "no_prediction_coverage"
    return "stationary"


def summary(distribution, statuses):
    """Aggregate step gaps into `flow_distribution_score`; `no_prediction_coverage` counts as 1."""

    missing = statuses.count("no_prediction_coverage")
    count = len(distribution) + missing
    return {
        "flow_distribution_score": float(1 - (np.sum(distribution) + missing) / count) if count else None,
        "flow_distribution_frames": count,
        "flow_frame_status": statuses,
    }


def distribution_gap(reference_flow: torch.Tensor, flow: torch.Tensor,
                     read: torch.Tensor, sample: int = SAMPLE) -> float | None:
    """Normalised sliced Wasserstein distance between the two sides' flow vectors on one step.

    The vectors at the `read` mask are compared as 2D point sets, without pixel
    correspondence, and the distance is divided by the sum of the two sides'
    per-component RMS. Returns None when `read` is empty or both sides are static.
    """

    if not int(read.sum()):
        return None
    left, right = reference_flow[read], flow[read]
    if len(left) > sample:      # one seeded draw of pixels, shared by both sides
        draw = torch.Generator(device=left.device).manual_seed(seed())
        pick = torch.randperm(len(left), device=left.device, generator=draw)[:sample]
        left, right = left[pick], right[pick]
    # Normalise by the per-component RMS: `semd` compares one-dimensional projections.
    scale = float(left.pow(2).mean().sqrt() + right.pow(2).mean().sqrt())
    if scale <= 0.0:
        return None
    return min(DIST_CAP, semd(left.cpu().numpy(), right.cpu().numpy()) / scale)


def flow_reading(reference: str | Path, shape: tuple[int, int, int], submission,
                  device: torch.device) -> dict:
    """Compute the flow readings; `submission(step, stride)` supplies the submission side.

    It returns `(flow (H, W, 2) float32, valid (H, W) bool)` on `device` for frames
    `step * stride -> (step + 1) * stride`, and is called with increasing `step`.
    """

    with open_dense(Path(reference) / FLOW_FILENAME) as archive:
        stored, stored_valid = archive["flow"], archive["valid"]
        stride = int(archive["stride"][()])
        frames, height, width = shape
        if stored.shape[1:] != (height, width, 2) or stored_valid.shape != stored.shape[:-1]:
            raise ValueError(f"Flow arrays must match the submission resolution {(height, width)}: "
                             f"flow={stored.shape}, valid={stored_valid.shape}")
        steps = min(len(stored), sampled_length(frames, stride) - 1)

        evaluated_fraction, distribution, statuses = [], [], []
        kept = {"flow": [], "valid": [], "evaluated": []}
        for step in range(steps):
            reference_flow = torch.as_tensor(stored[step], dtype=torch.float32, device=device)
            reference_valid = torch.as_tensor(stored_valid[step], device=device)
            flow, valid = submission(step, stride)
            if tuple(flow.shape) != (height, width, 2):
                raise ValueError(f"Submission flow must be {(height, width, 2)}, not {tuple(flow.shape)}")
            reference_moving = reference_flow.norm(dim=-1) > MOVING
            moving = reference_moving | (flow.norm(dim=-1) > MOVING)
            read = reference_valid & valid & moving
            count = int(read.sum())
            statuses.append(frame_status(reference_valid, valid, reference_moving, count))
            gap = distribution_gap(reference_flow, flow, read)
            if gap is not None:
                distribution.append(gap)
            evaluated_fraction.append(count / (height * width))
            kept["flow"].append(flow.cpu().numpy().astype(np.float16))
            kept["valid"].append(valid.cpu().numpy())
            kept["evaluated"].append(read.cpu().numpy())

    arrays = {name: np.stack(values) for name, values in kept.items()}
    if distribution:
        arrays["distribution"] = np.asarray(distribution, np.float32)
    arrays["stride"] = np.int64(stride)
    return {
        "arrays": {"flow": arrays},
        **summary(distribution, statuses),
        "flow_evaluated_fraction": float(np.mean(evaluated_fraction)),
        "flow_frames": len(statuses),
        "flow_stride": stride,
    }


__all__ = ["MOVING", "flow_reading", "frame_status", "summary"]
