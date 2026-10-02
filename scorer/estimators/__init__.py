"""Wrappers of the pretrained networks that process videos.

`raft` (optical flow), `cotracker` (point tracks), `moge` (monocular point map) and
`video_depth` (Video Depth Anything disparity) produce the reference estimates that
`python -m scorer.prepare` caches. `backbones` holds DINOv3 and TIPSv2, which embed
both the reference video (in prepare) and the submission's render (in the scorer).
All load weights offline from the checkpoint store named in `config.models`.
"""

from __future__ import annotations

import gc

import torch


def free() -> None:
    """Run the garbage collector and release cached CUDA memory."""

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


__all__ = ["free"]
