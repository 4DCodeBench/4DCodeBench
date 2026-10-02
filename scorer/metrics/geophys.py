"""GeoPhys: similarity of five regularity statistics of two videos' feature trajectories.

Interno et al., *GeoPhys: The Geometry of Physical Plausibility* (arXiv:2606.20707).
Each frame is embedded as one vector -- the spatial average of DINOv3 patch tokens at
the readout layer (`config.models.READOUT`) -- giving a trajectory `z_t`. With steps
`v_t = z_{t+1} - z_t`, turning angles `theta_t`, second differences
`a_t = v_{t+1} - v_t` and linearity residuals `eps_t` (Sec. 3.1):

    phi_speed = std {||v_t||}          phi_curv  = mean {theta_t}
    phi_ang   = std {theta_t}          phi_accel = mean {||a_t||^2}
    phi_perr  = mean {||eps_t||}

Each statistic contributes the relative gap `|phi - phi'| / (phi + phi')` between
reference and render, and the score is one minus the mean gap over the finite ones.
"""

from __future__ import annotations

import numpy as np

SIGNALS = ("speed", "curv", "ang", "accel", "perr")
HISTORY = 6        # frames spanning the affine subspace of `perr`
EPS = 1e-9


def signals(trajectory: np.ndarray, history: int = HISTORY) -> dict[str, float]:
    """Return the five GeoPhys statistics of a `(T, d)` trajectory; NaN if `T < history + 2`."""

    z = np.asarray(trajectory, np.float64)
    if z.ndim != 2 or len(z) < history + 2:
        return {name: float("nan") for name in SIGNALS}
    step = np.diff(z, axis=0)
    length = np.linalg.norm(step, axis=1)
    turn = _angles(step)
    return {
        "speed": float(length.std()),
        "curv": float(turn.mean()) if len(turn) else float("nan"),
        "ang": float(turn.std()) if len(turn) else float("nan"),
        "accel": float((np.linalg.norm(np.diff(step, axis=0), axis=1) ** 2).mean()),
        "perr": float(_residual(z, history).mean()),
    }


def _angles(step: np.ndarray) -> np.ndarray:
    """Turning angles between consecutive steps, skipping pairs with a zero-length step."""

    cosine = (step[:-1] * step[1:]).sum(axis=1)
    norms = np.linalg.norm(step[:-1], axis=1) * np.linalg.norm(step[1:], axis=1)
    live = norms > EPS
    return np.arccos(np.clip(cosine[live] / norms[live], -1.0, 1.0))


def _residual(z: np.ndarray, history: int) -> np.ndarray:
    """Distance of each `z_{t+1}` from the affine span of the `history` frames before it.

    This is the paper's geometric form of the linear auto-regressive residual ("the
    component of z_{t+1} orthogonal to the H-step linear span"), fitted per window
    because a global least-squares predictor is underdetermined when `d` exceeds `T`.
    The span basis is the left singular vectors above a relative threshold of 1e-10,
    since `history` centred points span at most `history - 1` dimensions. The
    predicted frames are `H .. T-1`.
    """

    out = []
    for t in range(history - 1, len(z) - 1):
        window = z[t - history + 1:t + 1]
        centre = window.mean(axis=0)
        left, spread, _ = np.linalg.svd((window - centre).T, full_matrices=False)
        basis = left[:, spread > spread[0] * 1e-10] if spread[0] > 0 else left[:, :0]
        offset = z[t + 1] - centre
        out.append(np.linalg.norm(offset - basis @ (basis.T @ offset)))
    return np.asarray(out) if out else np.zeros(1)


def geophys_video_reading(key: str, reference: np.ndarray, render: np.ndarray,
                          layer: int | None = None, readout: str = "pooled") -> dict:
    """Return both videos' statistics, their relative gaps, and `<key>_score`.

    `layer` and `readout` are stored in the result for provenance only.
    """

    left, right = signals(reference), signals(render)
    gaps = {name: abs(left[name] - right[name]) / (left[name] + right[name] + EPS)
            for name in SIGNALS if np.isfinite(left[name]) and np.isfinite(right[name])}
    return {f"{key}_signals": {"gt": left, "pred": right, "gaps": gaps, "layer": layer, "readout": readout},
            f"{key}_score": float(max(0.0, 1.0 - np.mean(list(gaps.values())))) if gaps else None}


__all__ = ["EPS", "HISTORY", "SIGNALS", "geophys_video_reading", "signals"]
