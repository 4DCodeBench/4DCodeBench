"""Multi-start trimmed similarity ICP on per-frame dynamic-surface samples, batched on the GPU."""

import itertools

import numpy as np
import torch
from pytorch3d.ops import knn_points


def cube_rotations(device):
    rotations = []
    for permutation in itertools.permutations(range(3)):
        for signs in itertools.product((-1, 1), repeat=3):
            r = np.eye(3)[:, permutation] * signs
            if np.linalg.det(r) > 0:
                rotations.append(r)
    return torch.tensor(np.asarray(rotations), dtype=torch.float32, device=device)


def normalize(points):
    points = points.double()
    centre = points.mean(dim=(0, 1))
    radius = (points.sub(centre).square().sum(-1).mean()).sqrt()
    return ((points - centre) / radius).float(), centre, radius


def pca(points):
    flat = points.reshape(-1, 3)
    _, basis = torch.linalg.eigh(flat.T @ flat)
    basis = basis.clone()
    basis[:, -1] *= torch.linalg.det(basis).sign()
    return basis


def seeds(x, y):
    rotations = torch.cat(
        [
            torch.eye(3, device=x.device)[None],
            pca(x)[None] @ cube_rotations(x.device) @ pca(y).T[None],
        ]
    )
    result = (
        torch.ones(len(rotations), device=x.device),
        rotations,
        torch.zeros((len(rotations), 3), device=x.device),
    )
    centres = y.mean(1)
    if len(centres) > 1 and float((centres - centres.mean(0)).square().sum()) > 0:
        temporal = solve(centres[None], x.mean(1)[None])
        result = tuple(torch.cat([a, b]) for a, b in zip(result, temporal))
    return result


def moved(y, transform):
    s, r, t = transform
    return (
        s[:, None, None, None] * (y[None] @ r[:, None].transpose(-1, -2))
        + t[:, None, None]
    )


def gather(points, index):
    return torch.gather(points, -2, index[..., None].expand(*index.shape, 3))


def solve(a, b):
    ma, mb = a.mean(-2), b.mean(-2)
    a, b = a - ma[:, None], b - mb[:, None]
    u, singular, vh = torch.linalg.svd(
        b.transpose(-1, -2) @ a / a.shape[-2], full_matrices=False
    )
    sign = torch.linalg.det(u @ vh).sign()
    correction = torch.stack([torch.ones_like(sign), torch.ones_like(sign), sign], -1)
    rotation = (u * correction[:, None]) @ vh
    scale = (singular * correction).sum(-1) / a.square().sum(-1).mean(-1)
    offset = mb - scale[:, None] * (rotation @ ma[..., None]).squeeze(-1)
    return scale, rotation, offset


def iterate(x, y, transform, iterations):
    count = len(transform[0])
    xs, ys = x[None].expand(count, -1, -1, -1), y[None].expand(count, -1, -1, -1)
    keep = int(x.shape[0] * x.shape[1] * 0.8)
    for _ in range(iterations):
        f, fi, b, bi = nearest(x, y, transform)
        kf = f.flatten(1).topk(keep, largest=False, sorted=False).indices
        kb = b.flatten(1).topk(keep, largest=False, sorted=False).indices
        source = torch.cat(
            [gather(ys.flatten(1, 2), kf), gather(gather(ys, bi).flatten(1, 2), kb)], 1
        )
        target = torch.cat(
            [gather(gather(xs, fi).flatten(1, 2), kf), gather(xs.flatten(1, 2), kb)], 1
        )
        transform = solve(source, target)
    return transform


def objective(x, y, transform, cap):
    f, _, b, _ = nearest(x, y, transform)
    return 0.5 * (
        (f.sqrt() / cap).clamp_max(1).mean(dim=(1, 2))
        + (b.sqrt() / cap).clamp_max(1).mean(dim=(1, 2))
    )


def take(transform, index):
    return tuple(v[index] for v in transform)


def world_transform(transform, cg, rg, cp, rp):
    s, r, t = [v[0].double() for v in transform]
    scale = s * rg / rp
    offset = cg + rg * t - scale * (r @ cp)
    return {
        "scale": float(scale),
        "rotation": r.cpu().tolist(),
        "offset": offset.cpu().tolist(),
    }


def nearest(x, y, transform):
    points = moved(y, transform)
    batch, frames, count, _ = points.shape
    target = (
        x[None]
        .expand(batch, -1, -1, -1)
        .reshape(batch * frames, x.shape[1], 3)
        .contiguous()
    )
    source = points.reshape(batch * frames, count, 3).contiguous()
    forward = knn_points(source, target, K=1, return_sorted=False)
    backward = knn_points(target, source, K=1, return_sorted=False)
    return (
        forward.dists.reshape(batch, frames, count),
        forward.idx.reshape(batch, frames, count),
        backward.dists.reshape(batch, frames, x.shape[1]),
        backward.idx.reshape(batch, frames, x.shape[1]),
    )


def fit(gt, pred, radius):
    """Fit the similarity taking `pred` onto `gt`, both `(F, N, 3)` samples in world units.

    Seeds are the identity, the PCA alignment under each of the 24 proper cube
    rotations, and a fit of the per-frame centroids. All seeds run 20 trimmed ICP steps
    on 256 points of up to 4 frames, the best 4 run 20 more on 512 points of every
    frame, and the seed with the lowest Chamfer objective (capped at `0.5 * radius`) is
    returned as `{"scale", "rotation", "offset"}`.
    """

    x, cg, rg = normalize(torch.as_tensor(gt, dtype=torch.float64, device="cuda"))
    y, cp, rp = normalize(torch.as_tensor(pred, dtype=torch.float64, device="cuda"))
    if not float(rg) > 0 or not float(rp) > 0:
        raise ValueError("dynamic surface cloud has zero RMS radius")
    cap = (0.5 * radius / rg).float()
    transform = seeds(x, y)
    frames = torch.as_tensor(
        np.linspace(0, len(x) - 1, min(4, len(x))).astype(int), device=x.device
    )
    transform = iterate(x[frames, :256], y[frames, :256], transform, 20)
    cost = objective(x[frames, :256], y[frames, :256], transform, cap)
    transform = iterate(
        x[:, :512], y[:, :512], take(transform, cost.topk(4, largest=False).indices), 20
    )
    cost = objective(x, y, transform, cap)
    return world_transform(take(transform, cost.argmin().reshape(1)), cg, rg, cp, rp)
