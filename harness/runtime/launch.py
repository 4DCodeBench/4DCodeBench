"""Run a job list through `infer` or the scoring stages on this machine.

`infer.py` calls `execute` and `score.py` calls `stages`. `infer` runs its jobs one
after another in this process. A scoring stage splits the list into shards and runs
them in parallel, one child process (`launch.py --index ...`) per visible GPU.
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import subprocess
import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import host_path, log, read_toml
from jobs import label, load_jobs

ACTIONS = ("infer", "prepare", "score", "viz")
STAGES = ("prepare", "score", "viz")   # handled by score.py; their items are shards


def action_module(action: str):
    """Import and return the module of one action: `infer` or `score`.

    Imported lazily, since both modules import this one.
    """

    if action not in ACTIONS:
        raise ValueError(f"unknown action {action!r}; known: {list(ACTIONS)}")
    return __import__("score" if action in STAGES else action)


def one(action: str, item: dict, runtime: dict, dry_run: bool) -> dict:
    """Run one item: one agent job, or one shard of one scoring stage."""

    module = action_module(action)
    if action in STAGES:
        return module.one_shard(action, item, runtime, dry_run)
    return module.one_infer(item, runtime, dry_run)


def gpus_available() -> list[str]:
    """Return the devices in `CUDA_VISIBLE_DEVICES`, one per worker slot; `[""]` when unset."""

    selection = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    return [item.strip() for item in selection.split(",") if item.strip()] or [""]


def sequential(action: str, items: list[dict], runtime: dict, dry_run: bool) -> int:
    """Run `items` one after another in this process and print a summary."""

    records = []
    for item in items:
        log(f"{action} {label(item)}")
        records.append(one(action, item, runtime, dry_run))
    keys = ("case", "agent", "model", "effort", "trial", "backend", "delivered",
            "stage", "shard", "worlds", "exit")
    summary = [{key: record[key] for key in keys if key in record} for record in records]
    print(json.dumps(summary, indent=2, sort_keys=True))
    if dry_run:
        return 0
    return int(not all(action_module(action).succeeded(record) for record in records))


def execute(action: str, jobs_path: Path, runtime_path: Path, dry_run: bool) -> int:
    """Run one action over the job list."""

    return stages([action], jobs_path, runtime_path, dry_run)


def stages(actions: list[str], jobs_path: Path, runtime_path: Path, dry_run: bool,
           options: dict[str, dict] | None = None) -> int:
    """Run the job list through each action in turn with `options[action]`.

    A scoring stage writes its shard list to `<action>-work.json` and runs it with `batch`.
    """

    loaded = load_jobs(host_path(str(jobs_path)))
    runtime = read_toml(host_path(str(runtime_path)))
    status = 0
    for action in actions:
        extra = (options or {}).get(action, {})
        if action not in STAGES:
            status |= sequential(action, [job | extra for job in loaded], runtime, dry_run)
            continue
        items = action_module(action).shards(action, loaded, runtime, extra, dry_run)
        if dry_run:
            status |= sequential(action, items, runtime, dry_run)
            continue
        manifest = Path(items[0]["shard"]).parent / f"{action}-work.json"
        manifest.write_text(json.dumps(items, indent=2) + "\n", encoding="utf-8")
        status |= batch(action, manifest, host_path(str(runtime_path)), manifest.parent)
    return status


def worker(action: str, manifest_path: Path, runtime_path: Path, index: int, gpu: str) -> int:
    """Run item `index` of the manifest on device `gpu`; an empty `gpu` pins nothing."""

    if gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    job = json.loads(manifest_path.read_text(encoding="utf-8"))[index]
    record = one(action, job, read_toml(runtime_path), False)
    return 0 if action_module(action).succeeded(record) else 1


def batch(action: str, manifest_path: Path, runtime_path: Path, log_dir: Path) -> int:
    """Run every item of the manifest, one child process per visible device at a time.

    Each child's output goes to `<log_dir>/<action>-<index>-<label>.out`. Returns 1 if any failed.
    """

    entries = json.loads(manifest_path.read_text(encoding="utf-8"))
    devices = gpus_available()
    todo: queue.SimpleQueue[int] = queue.SimpleQueue()
    for index in range(len(entries)):
        todo.put(index)
    failures: list[int] = []
    lock = threading.Lock()

    def slot(gpu: str) -> None:
        while True:
            try:
                index = todo.get_nowait()
            except queue.Empty:
                return
            name = label(entries[index])
            path = log_dir / f"{action}-{index}-{name}.out"
            with open(path, "w", encoding="utf-8") as sink:
                result = subprocess.run(
                    [sys.executable, __file__, "--action", action,
                     "--manifest", str(manifest_path), "--runtime", str(runtime_path),
                     "--index", str(index), "--gpu", gpu],
                    stdout=sink, stderr=subprocess.STDOUT, check=False,
                )
            with lock:
                if result.returncode != 0:
                    failures.append(index)
                print(f"[batch] gpu {gpu or 'all'} {action} {index} {name}: "
                      f"{'ok' if result.returncode == 0 else 'FAILED'} ({path.name})", flush=True)

    threads = [threading.Thread(target=slot, args=(gpu,)) for gpu in devices[:len(entries)]]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    print(f"[batch] {len(entries) - len(failures)}/{len(entries)} ok"
          + (f", failed indices {sorted(failures)}" if failures else ""), flush=True)
    return 1 if failures else 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one item of a work list (used by `batch`).")
    parser.add_argument("--action", choices=ACTIONS, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--index", type=int, required=True)
    parser.add_argument("--gpu", required=True)
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    raise SystemExit(worker(args.action, args.manifest, args.runtime, args.index, args.gpu))
