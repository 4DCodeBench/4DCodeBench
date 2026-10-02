"""Render the viewer media of every run in a manifest.

    python -m visualizer --manifest /manifest.json [--media flow,tracks,...]

Reads the scorer's roots and manifest and writes each run's media to the `viz/`
directory beside its `results/`. Model-free groups are rendered per run, then
`features` with each backbone loaded once. `--media` selects groups from
`visualizer.render.MEDIA`; a group with missing input is skipped, and the
scorer's gates limit which groups a run gets (see `visualizer.render`).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import Target
from .paths import (
    CASES_ROOT,
    CHECKPOINTS_DIR,
    DATA_ROOT,
    MANIFEST_PATH,
    RUNS_ROOT,
    Roots,
    load_manifest,
)
from .render import MEDIA, features, render


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="python -m visualizer", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH,
                        help="the runs to render, as the harness wrote it")
    parser.add_argument("--cases", type=Path, default=CASES_ROOT, help="the cases root")
    parser.add_argument("--data", type=Path, default=DATA_ROOT, help="the case data root")
    parser.add_argument("--runs", type=Path, default=RUNS_ROOT, help="the runs root")
    parser.add_argument("--checkpoints", type=Path, default=CHECKPOINTS_DIR,
                        help="the checkpoint directory for the feature videos")
    parser.add_argument("--media", action="append", default=None,
                        help=f"media groups to render, out of {list(MEDIA)}; "
                             "repeatable or comma-separated; everything by default")
    args = parser.parse_args(argv)
    if args.media is not None:
        args.media = [name for item in args.media for name in item.split(",") if name]
        unknown = sorted(set(args.media) - set(MEDIA))
        if unknown:
            parser.error(f"--media names {list(MEDIA)}, not {unknown}")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    roots = Roots(cases=args.cases, data=args.data, runs=args.runs, checkpoints=args.checkpoints)
    targets = [Target(entry) for entry in load_manifest(args.manifest, roots)]
    print(f"{len(targets)} world(s)", flush=True)
    for target in targets:
        target.out.mkdir(parents=True, exist_ok=True)
        print(f"{target}: {', '.join(render(target, args.media))}", flush=True)
    for status in features(targets, args.media):
        print(status, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
