"""Scorer container interface: four root trees and the manifest that indexes them.

A container mounts `/cases`, `/data`, `/runs` and `/checkpoints`, and
`/manifest.json` lists the worlds to process. Every input and output path is
derived from a root; an optional input (reference world, annotation, estimate)
is present or absent as a path.

Each entry names its case, its world and its output directory, as paths whose
first segment names the root they are relative to:

    {"case": "<kind>/<case>", "world": "runs/<run>/workspace/world", "out": "runs/<run>"}
    {"case": "<kind>/<case>", "world": "data/<kind>/<case>/world", "out": "data/<kind>/<case>/sanity"}

The first scores a delivered run, the second a case's reference world against
itself; both are processed identically. A prepare entry carries only `case`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from config.annotation import ANNOTATION_DIRNAME
from config.estimates import (
    ESTIMATES_DIRNAME,
    RESULTS_DIRNAME,
    VIZ_DIRNAME,
)

CASES_ROOT = Path("/cases")
DATA_ROOT = Path("/data")
RUNS_ROOT = Path("/runs")
CHECKPOINTS_DIR = Path("/checkpoints")
MANIFEST_PATH = Path("/manifest.json")

CASE_VIDEO = "reference.mp4"
WORLD_DIRNAME = "world"


def solver_cache(world: Path, name: str) -> Path:
    """Map a manifest `cache` name to a path under the saved `world` directory.

    `/workspace/world/...` is rebased onto `world`, `/workspace/...` onto its
    parent, and any other name is taken relative to `world/solver/`.
    """

    path = Path(name)
    if path.is_relative_to("/workspace/world"):
        return world / path.relative_to("/workspace/world")
    if path.is_relative_to("/workspace"):
        return world.parent / path.relative_to("/workspace")
    return world / "solver" / path


@dataclass(frozen=True)
class Roots:
    """The four root trees of a stage: the container mounts unless a flag overrides them."""

    cases: Path = CASES_ROOT
    data: Path = DATA_ROOT
    runs: Path = RUNS_ROOT
    checkpoints: Path = CHECKPOINTS_DIR

    def resolve(self, path: str) -> Path:
        """Resolve a manifest path, whose first segment names its root (`cases`, `data`, `runs`)."""

        name, _, rest = path.partition("/")
        trees = {"cases": self.cases, "data": self.data, "runs": self.runs}
        if name not in trees or not rest:
            raise ValueError(f"a manifest path is <{'|'.join(trees)}>/<path>, not {path!r}")
        return trees[name] / rest


@dataclass(frozen=True)
class Entry:
    """One manifest entry: a case, the world to score, and its output directory.

    `case` is `<kind>/<case>`, the path `cases/` and `data/` are both laid out on.
    `world` and `out` are already resolved; a prepare entry has neither.
    """

    case: str
    world: Path | None = None
    out: Path | None = None
    roots: Roots = field(default_factory=Roots)

    @property
    def video(self) -> Path:
        return self.roots.cases / self.case / CASE_VIDEO

    @property
    def gt(self) -> Path:
        return self.roots.data / self.case / WORLD_DIRNAME

    @property
    def annotation(self) -> Path:
        return self.roots.data / self.case / ANNOTATION_DIRNAME

    @property
    def estimates(self) -> Path:
        return self.roots.data / self.case / ESTIMATES_DIRNAME

    @property
    def results(self) -> Path:
        return self.out / RESULTS_DIRNAME

    @property
    def viz(self) -> Path:
        return self.out / VIZ_DIRNAME

    def __str__(self) -> str:
        return str(self.world or self.case)


def load_manifest(path: str | Path, roots: Roots | None = None) -> list[Entry]:
    """The manifest's entries, in order."""

    roots = roots or Roots()
    return [Entry(item["case"],
                  roots.resolve(item["world"]) if "world" in item else None,
                  roots.resolve(item["out"]) if "out" in item else None,
                  roots)
            for item in json.loads(Path(path).read_text(encoding="utf-8"))]


__all__ = ["CASES_ROOT", "CASE_VIDEO", "CHECKPOINTS_DIR", "DATA_ROOT", "MANIFEST_PATH",
           "RUNS_ROOT", "WORLD_DIRNAME", "Entry", "Roots", "load_manifest"]
