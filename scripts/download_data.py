#!/usr/bin/env python3
"""Download the benchmark data from Hugging Face into `cases/` and `data/`.

    python scripts/download_data.py                  # both kinds, videos and evaluation data
    python scripts/download_data.py --videos-only    # reference videos only, enough for inference
    python scripts/download_data.py --kind real

Sources:

    4DCodeBench/Dataset-Real-World   videos/<case>.mp4, annotations/<case>/dynamic_mask.npz,
                                     scripts/prepare_videos.py for the videos not redistributed
    4DCodeBench/Dataset-Synthetic    videos/<case>.mp4, worlds/<case>.h5

Output:

    cases/<kind>/<case>/reference.mp4
    data/real/<case>/annotation/dynamic_mask.npz
    data/synthetic/<case>/world/

Files already in place are skipped, so an interrupted download resumes by running again.
Real-World clips whose licence forbids redistribution are rebuilt from their public sources
by the dataset's own `prepare_videos.py`. Downloads go to a temporary directory.

Requires huggingface_hub, h5py, hdf5plugin, numpy and opencv-python, and ffmpeg on PATH.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from huggingface_hub import hf_hub_download, list_repo_files, snapshot_download

ROOT = Path(__file__).resolve().parent.parent
REAL = "4DCodeBench/Dataset-Real-World"
SYNTHETIC = "4DCodeBench/Dataset-Synthetic"


def copy(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)


def real(videos_only: bool, tmp: Path) -> None:
    patterns = ["metadata.jsonl", "videos/*", "scripts/*"] + ([] if videos_only else ["annotations/*"])
    local = Path(snapshot_download(REAL, repo_type="dataset", local_dir=tmp / "real",
                                   allow_patterns=patterns))
    cases = [json.loads(line)["case"]
             for line in (local / "metadata.jsonl").read_text().splitlines() if line.strip()]
    missing = [c for c in cases if not (ROOT / "cases/real" / c / "reference.mp4").exists()
               and not (local / "videos" / f"{c}.mp4").exists()]
    if missing:
        subprocess.run([sys.executable, str(local / "scripts/prepare_videos.py"),
                        "--downloads", str(tmp / "sources"), "--out", str(local / "videos"),
                        *missing], check=True)

    for case in cases:
        video = ROOT / "cases/real" / case / "reference.mp4"
        if not video.exists():
            copy(local / "videos" / f"{case}.mp4", video)
        mask = ROOT / "data/real" / case / "annotation/dynamic_mask.npz"
        if not videos_only and not mask.exists():
            copy(local / "annotations" / case / "dynamic_mask.npz", mask)
    print(f"real: {len(cases)} cases")


def synthetic(videos_only: bool, tmp: Path) -> None:
    files = list_repo_files(SYNTHETIC, repo_type="dataset")
    cases = sorted(Path(f).stem for f in files if f.startswith("videos/"))
    for case in cases:
        video = ROOT / "cases/synthetic" / case / "reference.mp4"
        if not video.exists():
            copy(Path(hf_hub_download(SYNTHETIC, f"videos/{case}.mp4", repo_type="dataset",
                                      local_dir=tmp / "synthetic")), video)
        world = ROOT / "data/synthetic" / case / "world"
        if not videos_only and not world.exists():
            packed = Path(hf_hub_download(SYNTHETIC, f"worlds/{case}.h5", repo_type="dataset",
                                          local_dir=tmp / "synthetic"))
            unpack(packed, world)
            packed.unlink()
    print(f"synthetic: {len(cases)} cases")


def unpack(packed: Path, world: Path) -> None:
    """Write a packed world back out as `world/`: `.npz` archives and plain files."""
    import h5py
    import hdf5plugin  # noqa: F401  (registers the Zstd filter)

    partial = world.with_name(world.name + ".partial")
    shutil.rmtree(partial, ignore_errors=True)
    with h5py.File(packed, "r") as h:
        for top in ("meshes", "dynamics"):
            for name in h.get(top, {}):
                target = partial / top / f"{name}.npz"
                target.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(target, **archive(h[top][name]))

        def write(name, item):
            if isinstance(item, h5py.Dataset) and name.split("/")[0] not in ("meshes", "dynamics"):
                target = partial / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(item[()].tobytes())
        h.visititems(write)
    partial.rename(world)


def archive(group) -> dict[str, np.ndarray]:
    """The arrays of one `.npz` archive. A per-frame series `<key>_TTTT` is stored as one
    `stack` (fixed shape) or one `flat` array with `lengths` (varying shape), plus `frames`."""
    arrays = {}
    for name, item in group.items():
        if not hasattr(item, "keys"):
            arrays[name] = item[()]
            continue
        frames, width = item["frames"][()], int(item.attrs.get("width", 4))
        if "flat" in item:
            flat, lengths = item["flat"][()], item["lengths"][()]
            edges = np.concatenate([[0], np.cumsum(lengths)])
            parts = [flat[edges[i]:edges[i + 1]] for i in range(len(frames))]
        else:
            parts = list(item["stack"][()])
        for frame, part in zip(frames, parts):
            arrays[f"{name}_{int(frame):0{width}d}"] = part
    return arrays


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kind", choices=("real", "synthetic"), help="one kind only (default: both)")
    ap.add_argument("--videos-only", action="store_true",
                    help="reference videos only, without the evaluation data")
    args = ap.parse_args()
    with tempfile.TemporaryDirectory() as tmp:
        if args.kind in (None, "real"):
            real(args.videos_only, Path(tmp))
        if args.kind in (None, "synthetic"):
            synthetic(args.videos_only, Path(tmp))

if __name__ == "__main__":
    main()
