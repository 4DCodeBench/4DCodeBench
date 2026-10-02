"""Dynamic time warping between sets of 3D paths, as a Triton GPU kernel.

The distance is the classic DTW recurrence (Sakoe & Chiba 1978; Berndt & Clifford
1994) over Euclidean gaps between samples, divided by the length of the optimal
warping path. Each kernel lane computes one path pair (the one-thread-per-DTW layout
of Sart et al. 2010). Lanes of a block share one path of the second set and take
consecutive paths of the first, and the current table row lives in a lane-major scratch buffer, so
each cell access is coalesced.

`dtw_matrix` computes the full `(n, m)` table (Trajectory); `dtw_pairs` computes only
`a[i]` against `b[i]`, `(n,)` (Track2D), selected by the kernel's `PAIRED` flag.
"""

from __future__ import annotations

import numpy as np
import torch
import triton
import triton.language as tl

LANES = 128          # lanes per program
CHUNK = 1 << 15      # pairs per kernel launch; width of the scratch buffers


def compact(paths: np.ndarray, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Pack each path's finite frames to the front: `(P, L, 3)` float32 and lengths `(P,)`."""

    alive = np.isfinite(paths).all(axis=2)
    lengths = alive.sum(axis=1)
    longest = int(lengths.max()) if len(lengths) else 0
    packed = np.zeros((len(paths), max(longest, 1), 3), np.float32)
    for row in range(len(paths)):
        packed[row, : lengths[row]] = paths[row][alive[row]]
    return (torch.as_tensor(packed, device=device).contiguous(),
            torch.as_tensor(lengths, dtype=torch.int32, device=device))


@triton.jit
def _dtw_kernel(x_ptr, y_ptr, la_ptr, lb_ptr, out_ptr, row_ptr, len_ptr,
                offset, total, n,
                LA: tl.constexpr, LB: tl.constexpr, STRIDE: tl.constexpr, LANES: tl.constexpr,
                PAIRED: tl.constexpr):
    lane = tl.program_id(0) * LANES + tl.arange(0, LANES)
    pair = offset + lane
    valid = pair < total
    if PAIRED:
        a = pair           # one lane per index pair (a[i], b[i])
        b = pair
    else:
        a = pair % n       # lanes of a block take consecutive paths of `x`
        b = pair // n      # and share one path of `y`
    la = tl.load(la_ptr + a, mask=valid, other=1)
    lb = tl.load(lb_ptr + b, mask=valid, other=1)
    x_base = x_ptr + a.to(tl.int64) * (LA * 3)
    y_base = y_ptr + b.to(tl.int64) * (LB * 3)
    inf = float("inf")
    result = tl.full([LANES], inf, tl.float32)
    for j in range(LB):
        tl.store(row_ptr + j * STRIDE + lane, tl.full([LANES], inf, tl.float32))
        tl.store(len_ptr + j * STRIDE + lane, tl.zeros([LANES], tl.float32))
    for i in range(LA):
        x0 = tl.load(x_base + i * 3 + 0, mask=valid, other=0.0)
        x1 = tl.load(x_base + i * 3 + 1, mask=valid, other=0.0)
        x2 = tl.load(x_base + i * 3 + 2, mask=valid, other=0.0)
        left = tl.full([LANES], inf, tl.float32)
        left_len = tl.zeros([LANES], tl.float32)
        diag = tl.full([LANES], inf, tl.float32)
        diag_len = tl.zeros([LANES], tl.float32)
        for j in range(LB):
            y0 = tl.load(y_base + j * 3 + 0, mask=valid, other=0.0)
            y1 = tl.load(y_base + j * 3 + 1, mask=valid, other=0.0)
            y2 = tl.load(y_base + j * 3 + 2, mask=valid, other=0.0)
            gap = tl.sqrt((x0 - y0) * (x0 - y0) + (x1 - y1) * (x1 - y1) + (x2 - y2) * (x2 - y2))
            up = tl.load(row_ptr + j * STRIDE + lane)
            up_len = tl.load(len_ptr + j * STRIDE + lane).to(tl.float32)
            best = tl.minimum(tl.minimum(left, up), diag)
            best_len = tl.where(best == left, left_len, tl.where(best == up, up_len, diag_len))
            value = gap + best
            length = best_len + 1.0
            start = (i == 0) & (j == 0)
            value = tl.where(start, gap, value)
            length = tl.where(start, 1.0, length)
            inside = (i < la) & (j < lb)
            value = tl.where(inside, value, inf)
            length = tl.where(inside, length, 0.0)
            tl.store(row_ptr + j * STRIDE + lane, value)
            tl.store(len_ptr + j * STRIDE + lane, length)
            diag = up
            diag_len = up_len
            left = value
            left_len = length
            done = (i == la - 1) & (j == lb - 1)
            result = tl.where(done, value / tl.maximum(length, 1.0), result)
    tl.store(out_ptr + pair, result, mask=valid)


def dtw_matrix_kernel(x: torch.Tensor, la: torch.Tensor, y: torch.Tensor, lb: torch.Tensor) -> torch.Tensor:
    """`(n, m)` length-normalised DTW distances between packed path sets."""

    n, longest_a, _ = x.shape
    m, longest_b, _ = y.shape
    total = n * m
    out = torch.empty(total, dtype=torch.float32, device=x.device)
    row = torch.empty((longest_b, CHUNK), dtype=torch.float32, device=x.device)
    length_dtype = torch.int16 if longest_a + longest_b - 1 <= 32767 else torch.int32
    length = torch.empty((longest_b, CHUNK), dtype=length_dtype, device=x.device)
    grid = (CHUNK // LANES,)
    for offset in range(0, total, CHUNK):
        _dtw_kernel[grid](x, y, la, lb, out, row, length, offset, total, n,
                          LA=longest_a, LB=longest_b, STRIDE=CHUNK, LANES=LANES, PAIRED=False)
    # pair = b * n + a
    return out.view(m, n).t().contiguous()


def dtw_pairs_kernel(x: torch.Tensor, la: torch.Tensor, y: torch.Tensor,
                     lb: torch.Tensor) -> torch.Tensor:
    """`(n,)` length-normalised DTW distances of `x[i]` against `y[i]`."""

    total, longest_a, _ = x.shape
    longest_b = y.shape[1]
    out = torch.empty(max(total, 1), dtype=torch.float32, device=x.device)
    row = torch.empty((longest_b, CHUNK), dtype=torch.float32, device=x.device)
    length_dtype = torch.int16 if longest_a + longest_b - 1 <= 32767 else torch.int32
    length = torch.empty((longest_b, CHUNK), dtype=length_dtype, device=x.device)
    grid = (CHUNK // LANES,)
    for offset in range(0, total, CHUNK):
        _dtw_kernel[grid](x, y, la, lb, out, row, length, offset, total, total,
                          LA=longest_a, LB=longest_b, STRIDE=CHUNK, LANES=LANES, PAIRED=True)
    return out[:total]


def dtw_matrix(a: np.ndarray, b: np.ndarray, device: torch.device) -> np.ndarray:
    """Per-step DTW distance of every path of `a` to every path of `b`, `(n, m)`."""

    x, la = compact(a, device)
    y, lb = compact(b, device)
    return dtw_matrix_kernel(x, la, y, lb).cpu().numpy()


def dtw_pairs(a: np.ndarray, b: np.ndarray, device: torch.device) -> np.ndarray:
    """Per-step DTW distance of `a[i]` to `b[i]`, `(n,)`, for two equally long sets."""

    if len(a) != len(b):
        raise ValueError(f"{len(a)} paths against {len(b)}: a paired reading needs one each")
    x, la = compact(a, device)
    y, lb = compact(b, device)
    return dtw_pairs_kernel(x, la, y, lb).cpu().numpy()


__all__ = ["compact", "dtw_matrix", "dtw_matrix_kernel", "dtw_pairs", "dtw_pairs_kernel"]
