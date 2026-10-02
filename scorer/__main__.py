"""`python -m scorer`: score every world a manifest names.

Runs the gates for every world, then the geometry, Semantic, GeoPhys and Uni3D
passes in turn. Each metric is written to `reward.json` when it completes. Exits
nonzero when any world had a failure.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .execution import Timeouts, Worker
from .paths import (
    CASES_ROOT,
    CHECKPOINTS_DIR,
    DATA_ROOT,
    MANIFEST_PATH,
    RUNS_ROOT,
    Roots,
    load_manifest,
)
from .score import (
    TYPES,
    gate,
    geometry_pass,
    geophys_pass,
    semantic_pass,
    uni3d_pass,
)
from .tasks import GeometryTasks


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="python -m scorer",
                                     description="Score every world a manifest names.")
    parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH,
                        help="the worlds to score, as the harness wrote them")
    parser.add_argument("--cases", type=Path, default=CASES_ROOT, help="the cases root")
    parser.add_argument("--data", type=Path, default=DATA_ROOT, help="the case data root")
    parser.add_argument("--runs", type=Path, default=RUNS_ROOT, help="the runs root")
    parser.add_argument("--checkpoints", type=Path, default=CHECKPOINTS_DIR,
                        help="the model store the semantic and Uni3D readings load from")
    parser.add_argument("--type", action="append", default=None,
                        help=f"metric types to read on their own, out of {list(TYPES)}; "
                             "repeatable or comma-separated; everything by default")
    parser.add_argument("--task-timeout", type=float, default=1800, help="seconds per metric or world inference")
    parser.add_argument("--prepare-timeout", type=float, default=600, help="seconds per geometry preparation or model load")
    parser.add_argument("--timeout", action="append", default=[], metavar="TASK=SECONDS",
                        help="override one task deadline; repeatable")
    args = parser.parse_args(argv)
    args.overrides = {}
    for value in args.timeout:
        try:
            name, seconds = value.split("=", 1)
            args.overrides[name] = float(seconds)
        except ValueError:
            parser.error("--timeout requires TASK=SECONDS")
    if any(value <= 0 for value in [args.task_timeout, args.prepare_timeout, *args.overrides.values()]):
        parser.error("timeouts must be positive")
    if args.type is not None:
        args.type = [name for item in args.type for name in item.split(",") if name]
        unknown = sorted(set(args.type) - set(TYPES))
        if unknown:
            parser.error(f"--type names {list(TYPES)}, not {unknown}")
    return args



def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    roots = Roots(cases=args.cases, data=args.data, runs=args.runs, checkpoints=args.checkpoints)
    entries = load_manifest(args.manifest, roots)
    wanted = set(args.type) if args.type else set(TYPES)
    limits = Timeouts(args.task_timeout, args.prepare_timeout, args.overrides)
    print(f"{len(entries)} world(s), types {sorted(wanted)}", flush=True)
    items = []
    worker = None
    try:
        for entry in entries:
            if worker is None:
                worker = Worker(GeometryTasks, limits.seconds("worker", True))
            items.append(gate(entry, wanted, worker, limits))
            if not worker.alive:
                worker.close()
                worker = None
    finally:
        if worker is not None:
            worker.close()
    geometry_pass(items, limits)
    semantic_pass(items, limits)
    geophys_pass(items, limits)
    uni3d_pass(items, limits)
    failed = sum(item.failed for item in items)
    print(f"{len(items) - failed}/{len(items)} completed without failures", flush=True)
    return int(bool(failed))


if __name__ == "__main__":
    raise SystemExit(main())
