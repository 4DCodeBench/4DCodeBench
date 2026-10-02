"""Scene3D: capped symmetric Chamfer distance between registered frame-0 visible surfaces.

With `D` the Chamfer distance in reference radii, each point's distance capped at
`SCENE_CAP`, `scene_score = max(0, 1 - D / SCENE_CAP)`.
"""

from __future__ import annotations

import numpy as np

from ..world import World
from .cloud import SCENE_CAP, SCENE_POINTS, cloud, reference_cloud, subsample
from .distances import bounded_chamfer, seed
from .registration import Registration

def scene_readings(x: np.ndarray, y: np.ndarray) -> dict:
    """Capped Chamfer error and score."""

    error = bounded_chamfer(x, y, SCENE_CAP)
    return {"scene_error": error, "scene_score": max(0., 1 - error / SCENE_CAP)}


def scene_error(
    gt: World,
    pred: World,
    alignment: Registration,
) -> dict:
    """Scene3D readings of both frame-0 surfaces as seen from the reference camera."""

    gt_generator, pred_generator = (np.random.default_rng(seed()) for _ in range(2))
    x = alignment.gt(subsample(cloud(gt, 0), SCENE_POINTS, gt_generator))
    y = alignment.gt(subsample(reference_cloud(gt, pred, alignment), SCENE_POINTS, pred_generator))
    readings = scene_readings(x, y)
    # the compared clouds, in reference radii
    readings["arrays"] = {"scene": {"gt": x.astype(np.float32), "pred": y.astype(np.float32)}}
    return readings
