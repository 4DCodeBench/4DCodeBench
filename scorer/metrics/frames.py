"""Per-frame evaluation on processes forked from the scoring process.

Workers share the loaded world copy-on-write. The pool is recreated every
`FRAMES_PER_POOL` frames to return worker memory to the operating system. Results
do not depend on which worker evaluates which frame.
"""

from __future__ import annotations

import multiprocessing
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor

WORKERS = 4
FRAMES_PER_POOL = 32

_JOB: dict = {}


def _run(frame: int):
    return _JOB["judge"](_JOB["world"], frame)


def judge_frames(judge: Callable, world, frames: list[int], workers: int = WORKERS) -> list:
    """Return `judge(world, frame)` for every frame, in order, computed on forked workers."""

    _JOB.update(judge=judge, world=world)
    fork = multiprocessing.get_context("fork")
    out: list = []
    for start in range(0, len(frames), FRAMES_PER_POOL):
        block = frames[start : start + FRAMES_PER_POOL]
        with ProcessPoolExecutor(max_workers=min(workers, len(block)), mp_context=fork) as pool:
            out.extend(pool.map(_run, block))
    return out


__all__ = ["FRAMES_PER_POOL", "WORKERS", "judge_frames"]
