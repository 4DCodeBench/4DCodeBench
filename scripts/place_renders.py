#!/usr/bin/env python3
"""Place the released renders of the evaluated systems where the repository reads runs.

    hf download 4DCodeBench/Results --repo-type dataset --local-dir results
    python scripts/place_renders.py results

Moves

    results/<system>/<kind>/<case>.mp4

to

    runs/<kind>/<case>/<system>/1/workspace/world/render.mp4

so that `vlm_judge/judge.py pairwise` compares a new system against them. A render
already in place is left alone.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source", type=Path, help="the downloaded 4DCodeBench/Results")
    ap.add_argument("--runs", type=Path, default=ROOT / "runs")
    args = ap.parse_args()

    placed = kept = 0
    for video in sorted(args.source.glob("*/*/*.mp4")):
        system, kind, case = video.parent.parent.name, video.parent.name, video.stem
        target = args.runs / kind / case / system / "1" / "workspace" / "world" / "render.mp4"
        if target.exists():
            kept += 1
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(video, target)
        placed += 1
    print(f"{placed} renders placed under {args.runs}, {kept} already there")


if __name__ == "__main__":
    main()
