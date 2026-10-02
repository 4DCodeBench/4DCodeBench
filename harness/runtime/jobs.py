"""Load a jobs file and expand it into one entry per trial.

A jobs file holds `trials` and `[[job]]` entries of case, agent, model and effort.
Inference runs the expanded list one entry at a time; scoring shards it.
"""

from __future__ import annotations

from pathlib import Path

import tomllib

KEYS = ("case", "agent", "model", "effort")


def load_jobs(path: Path) -> list[dict]:
    """Return one dict per trial; a job with its own `trial` key yields that trial alone."""

    document = tomllib.loads(path.read_text(encoding="utf-8"))
    trials = int(document.get("trials", 1))
    if trials < 1:
        raise ValueError(f"{path}: trials must be at least 1")
    entries = document.get("job", [])
    if not entries:
        raise ValueError(f"{path}: contains no [[job]] entries")

    jobs = []
    for entry in entries:
        missing = [key for key in KEYS if key not in entry]
        if missing:
            raise ValueError(f"{path}: a job is missing {missing}")
        unknown = sorted(set(entry) - set(KEYS) - {"trial"})
        if unknown:
            raise ValueError(f"{path}: a job carries unknown keys {unknown}")
        if "trial" in entry:
            jobs.append({key: entry[key] for key in KEYS} | {"trial": int(entry["trial"])})
            continue
        for trial in range(1, trials + 1):
            jobs.append({key: entry[key] for key in KEYS} | {"trial": trial})

    keys = [tuple(job[key] for key in (*KEYS, "trial")) for job in jobs]
    if len(keys) != len(set(keys)):
        raise ValueError(f"{path}: duplicate case/agent/model/effort/trial entries")
    return jobs


def label(item: dict) -> str:
    """Return a flat, filesystem-safe name for one work item: a job or a shard."""

    if "shard" in item:
        return Path(item["shard"]).stem
    case = str(item["case"]).replace("/", "-")
    return f"{case}-{item['agent']}-{item['model']}-{item['effort']}-{item['trial']}"
