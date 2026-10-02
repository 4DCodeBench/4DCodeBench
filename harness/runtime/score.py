"""Run the scoring stages over the runs a job list names.

    python harness/runtime/score.py --jobs J.toml --runtime R.toml [--stage prepare,score,viz]

`--stage` selects stages, which run in this order (default: `score`):

    prepare   `python -m scorer.prepare` writes each case's `estimates/` from its
              `reference.mp4`; one entry per case. `--only` limits the groups.
    score     `python -m scorer` writes each run's `results/`: `reward.json`,
              `reward.detail.json` and one `.npz` per metric. `--type` limits the metrics.
    viz       `python -m visualizer` renders each run's `viz/` videos. `--media` limits them.

`--sanity` adds each synthetic case's reference world to score and viz, written to
`data/<kind>/<case>/sanity/`.

Entries are sharded round-robin, one shard per device in `CUDA_VISIBLE_DEVICES`
(one when unset), and each shard runs in one scorer container. The container mounts
`/cases` (ro), `/data` (rw), `/runs` (rw, not for prepare), `/checkpoints` (ro) and
its shard at `/manifest.json`. Shard manifests and container logs go under
`<runs_root>/manifests/`. `[scorer] timeout_sec` bounds one container.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import launch
from common import (
    REPO,
    RUN_WORKSPACE,
    container_command,
    display,
    host_path,
    log,
    resolve_case,
    run_process,
)

sys.path.insert(0, str(REPO))

from checker.paths import WORLD_DIR
from config.estimates import SANITY_DIRNAME
from scorer.paths import (
    CASES_ROOT,
    CHECKPOINTS_DIR,
    DATA_ROOT,
    MANIFEST_PATH,
    RUNS_ROOT,
    WORLD_DIRNAME,
)

STAGES = launch.STAGES
MANIFEST_DIRNAME = "manifests"      # shard manifests and container logs
COMMANDS = {
    "prepare": ("python", "-m", "scorer.prepare"),
    "score": ("python", "-m", "scorer"),
    "viz": ("python", "-m", "visualizer"),
}
# per stage: the flag that narrows its work, and the option that supplies it
OPTIONS = {"prepare": ("--only", "only"), "score": ("--type", "type"), "viz": ("--media", "media")}


def stage_command(stage: str, runtime: dict, manifest: Path, flags: list[str]) -> list[str]:
    """Build the container command of one stage over one shard.

    Prepare gets no `/runs` mount. `/data` is writable, since prepare and sanity entries
    write there.
    """

    scorer = runtime["scorer"]
    store = scorer.get("checkpoints")
    if stage == "score":
        flags = [*flags, "--task-timeout", str(scorer.get("task_timeout_sec", 1800)),
                 "--prepare-timeout", str(scorer.get("task_prepare_timeout_sec", 600))]
        for name, seconds in scorer.get("task_timeouts", {}).items():
            flags.extend(["--timeout", f"{name}={seconds}"])
    runs = host_path(runtime["runs_root"]).resolve() if stage != "prepare" else None
    return container_command(runtime, scorer["environment"], scorer["image"][runtime["backend"]], [
        (host_path(runtime["cases_root"]).resolve(), CASES_ROOT, "ro"),
        (host_path(runtime["data_root"]).resolve(), DATA_ROOT, "rw"),
        (runs, RUNS_ROOT, "rw"),
        (host_path(store).resolve() if store else None, CHECKPOINTS_DIR, "ro"),
        (manifest.resolve(), MANIFEST_PATH, "ro"),
    ], [*COMMANDS[stage], "--manifest", str(MANIFEST_PATH), *flags])


def entries(stage: str, jobs: list[dict], runtime: dict, sanity: bool) -> list[dict]:
    """Return the manifest entries of a job list: one per case for prepare, else one per world.

    Each world entry names its `case`, its `world` and the `out` directory for its outputs.
    """

    cases_root = host_path(runtime["cases_root"])
    named = [job | {"case": resolve_case(cases_root, job["case"])} for job in jobs]
    cases = list(dict.fromkeys(job["case"] for job in named))
    if stage == "prepare":
        return [{"case": case} for case in cases]

    found = []
    for job in named:
        run = f"runs/{job['case']}/{job['agent']}-{job['model']}-{job['effort']}/{int(job['trial'])}"
        found.append({"case": job["case"], "world": f"{run}/{RUN_WORKSPACE}/{WORLD_DIR.name}",
                      "out": run})
    if sanity:
        data_root = host_path(runtime["data_root"])
        found += [{"case": case, "world": f"data/{case}/{WORLD_DIRNAME}",
                   "out": f"data/{case}/{SANITY_DIRNAME}"}
                  for case in cases if (data_root / case / WORLD_DIRNAME).is_dir()]
    return found


def shards(stage: str, jobs: list[dict], runtime: dict, options: dict, dry_run: bool) -> list[dict]:
    """Split the entries round-robin into one shard per device and write each manifest.

    Returns one item per non-empty shard: its manifest path, entry count and stage flags.
    """

    count = len(launch.gpus_available())
    flag, option = OPTIONS[stage]
    flags = [flag, options[option]] if options.get(option) else []
    directory = host_path(runtime["runs_root"]) / MANIFEST_DIRNAME
    if not dry_run:
        directory.mkdir(parents=True, exist_ok=True)
    listed = entries(stage, jobs, runtime, bool(options.get("sanity")))
    made = []
    for index in range(max(count, 1)):
        part = listed[index::count]
        if not part:
            continue
        manifest = directory / f"{stage}-{index}.json"
        if not dry_run:
            manifest.write_text(json.dumps(part, indent=2) + "\n", encoding="utf-8")
        made.append({"stage": stage, "shard": str(manifest), "worlds": len(part),
                     "flags": flags})
    return made


def one_shard(stage: str, item: dict, runtime: dict, dry_run: bool) -> dict:
    """Run one shard of one stage and return its record, with `exit` unless a dry run."""

    scorer = runtime["scorer"]
    manifest = Path(item["shard"])
    command = stage_command(stage, runtime, manifest, item["flags"])
    record = {"stage": stage, "shard": manifest.name, "worlds": item["worlds"],
              "command": display(command)}

    if dry_run:
        log("$", display(command))
        log("dry run, nothing executed")
        return record

    record["exit"] = run_process(command, manifest.with_suffix(".log"),
                                 int(scorer.get(f"{stage}_timeout_sec", scorer["timeout_sec"])))
    log(json.dumps({key: value for key, value in record.items() if key != "command"},
                   sort_keys=True))
    return record


def succeeded(record: dict) -> bool:
    return record.get("exit") == 0


def parse_stages(value: str) -> list[str]:
    """Return the comma-separated stages in their fixed run order; raise on an unknown one."""

    wanted = [name for name in value.split(",") if name]
    unknown = sorted(set(wanted) - set(STAGES))
    if unknown or not wanted:
        raise ValueError(f"--stage names {list(STAGES)}, not {value!r}")
    return [name for name in STAGES if name in wanted]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score a list of delivered runs.")
    parser.add_argument("--jobs", type=Path, required=True, help="the job list")
    parser.add_argument("--runtime", type=Path, required=True, help="the deployment configuration")
    parser.add_argument("--stage", default="score",
                        help=f"the stages to run, comma-separated, out of {list(STAGES)}; "
                             "always in that order, and only scoring by default")
    parser.add_argument("--type", default=None,
                        help="metric types to rescore on their own, comma-separated "
                             "(e.g. trajectory); the rest of results/ is left as it is")
    parser.add_argument("--media", default=None,
                        help="media groups the viz stage renders, comma-separated "
                             "(e.g. flow,tracks); everything by default")
    parser.add_argument("--only", default=None,
                        help="the estimate groups the prepare stage computes, comma-separated "
                             "(e.g. semantic,depth); all of them by default")
    parser.add_argument("--sanity", action="store_true",
                        help="score and render each case's own reference world too, into "
                             "data/<kind>/<case>/sanity/")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    options = {"prepare": {"only": args.only},
               "score": {"type": args.type, "sanity": args.sanity},
               "viz": {"media": args.media, "sanity": args.sanity}}
    return launch.stages(parse_stages(args.stage), args.jobs, args.runtime, args.dry_run, options)


if __name__ == "__main__":
    raise SystemExit(main())
