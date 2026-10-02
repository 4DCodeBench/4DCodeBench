"""Point-set distances, on the GPU when one is present."""

from __future__ import annotations

import numpy as np
import torch

from config.scorer import SEED

DIRECTIONS_K = 128
QUANTILES_Q = 8192


def seed() -> int:
    return SEED


def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def as_tensor(points: np.ndarray, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    tensor = torch.as_tensor(np.asarray(points), dtype=dtype, device=device())
    finite = tensor[torch.isfinite(tensor).all(dim=1)]
    if len(finite) == 0:
        raise ValueError("a distance was asked for on an empty cloud")
    return finite


def directions(dim: int) -> torch.Tensor:
    theta = np.random.default_rng(seed()).normal(size=(DIRECTIONS_K, dim))
    theta /= np.linalg.norm(theta, axis=1, keepdims=True)
    return torch.as_tensor(theta, dtype=torch.float32, device=device())


def quantiles(sorted_values: torch.Tensor, count: int = QUANTILES_Q) -> torch.Tensor:
    length = sorted_values.shape[0]
    if length == count:
        return sorted_values
    position = torch.linspace(0.0, length - 1.0, count, device=sorted_values.device)
    low, high = position.floor().long(), position.ceil().long()
    weight = (position - low).unsqueeze(1)
    return sorted_values[low] * (1.0 - weight) + sorted_values[high] * weight


def semd(x: np.ndarray, y: np.ndarray) -> float:
    """Sliced Wasserstein-2 distance between two point sets of equal dimension.

    Projects onto `DIRECTIONS_K` random unit directions seeded by `SEED`, resamples
    each sorted projection to `QUANTILES_Q` quantiles, and returns the RMS gap.
    Non-finite rows are dropped.
    """

    x, y = as_tensor(x), as_tensor(y)
    theta = directions(x.shape[1])
    projected_x = quantiles(torch.sort(x @ theta.T, dim=0).values)
    projected_y = quantiles(torch.sort(y @ theta.T, dim=0).values)
    return float(((projected_x - projected_y) ** 2).mean().sqrt())


def nearest(x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    from pytorch3d.ops import knn_points

    found = knn_points(x[None].contiguous(), y[None].contiguous(), K=1, return_sorted=False)
    return found.dists[0, :, 0].sqrt(), found.idx[0, :, 0]


def bounded_chamfer(x: np.ndarray, y: np.ndarray, cap: float) -> float:
    """Symmetric Chamfer distance, each point's distance capped at `cap`.

    Returns `0.5 * (mean forward + mean backward)`, or `cap` if either cloud is empty.
    """

    if not len(x) or not len(y):
        return cap
    a, b = as_tensor(x), as_tensor(y)
    forward, _ = nearest(a, b)
    backward, _ = nearest(b, a)
    return float(.5 * (forward.clamp_max(cap).mean() + backward.clamp_max(cap).mean()))

