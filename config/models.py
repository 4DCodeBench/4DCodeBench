"""Checkpoint names and layer choices of the pretrained models.

Each model is a directory under one checkpoint store, loaded offline. In the scorer
container the store is the `/checkpoints` mount; `scripts/download_checkpoints.sh`
fills one (docs/checkpoints.md).
"""

from __future__ import annotations

from pathlib import Path

DINOV3 = "dinov3-vitl16-pretrain-lvd1689m"
TIPS = "tipsv2-l14"
VIDEO_DEPTH_ANYTHING = "video-depth-anything-large"
RAFT = "raft-large"
COTRACKER = "cotracker3"
MOGE = "moge-3-vitl"

GEOPHYS_MODELS = {DINOV3: "dinov3"}   # GeoPhys backbone -> archive tag

# GeoPhys readout layer, counted from 1 over the transformer blocks, as selected by
# Interno et al. (Fig. 2, Table 3) for a ViT-L. `DEPTH` is the backbone's block count;
# `block_for` converts the layer to the from-the-end index of `backbones.*.tokens(block=)`.
READOUT = {DINOV3: 18}
DEPTH = {DINOV3: 24}


def block_for(name: str, layer: int | None = None) -> int:
    """Return the from-the-end `block=` index of `layer` (default: `READOUT[name]`)."""

    return (READOUT[name] if layer is None else layer) - DEPTH[name] - 1


def checkpoint(name: str, root: str | Path) -> Path:
    """Return the path of model `name` under the checkpoint store `root`."""

    return Path(root) / name

