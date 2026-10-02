"""Write the viewer media of scored runs.

No metric is computed here. Every video and bundle is generated from the scorer's
arrays, the case's cached estimates or the videos on disk; the only model forward
pass is the patch PCA in `render.features`.

`python -m visualizer` renders every run in its manifest: the model-free media
groups per run, then the feature videos with each backbone loaded once. `--media`
selects the groups.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from scorer.paths import Entry


@dataclass(frozen=True)
class Target:
    """One run to render: its manifest entry and the paths read and written.

    `out` is the only directory written. The optional input paths are `None` when absent.
    """

    entry: Entry

    @property
    def pred(self) -> Path:
        return self.entry.world

    @property
    def out(self) -> Path:
        return self.entry.viz

    @property
    def checkpoints(self) -> Path:
        return self.entry.roots.checkpoints

    @property
    def gt(self) -> Path | None:
        return self.entry.gt if self.entry.gt.is_dir() else None

    @property
    def estimates(self) -> Path | None:
        return self.entry.estimates if self.entry.estimates.is_dir() else None

    @property
    def results(self) -> Path | None:
        return self.entry.results if self.entry.results.is_dir() else None

    @property
    def reference_video(self) -> Path | None:
        return self.entry.video if self.entry.video.is_file() else None

    @property
    def gates(self) -> tuple[bool, bool]:
        """`(video gate passed, world gate passed)` from `reward.detail.json`.

        Both are False for an unscored run.
        """

        path = self.entry.results / "reward.detail.json"
        if not path.is_file():
            return False, False
        detail = json.loads(path.read_text(encoding="utf-8"))
        return (bool((detail.get("video_gate") or {}).get("ok")),
                bool((detail.get("world_gate") or {}).get("ok")))

    def has_results(self, *names: str) -> bool:
        """Whether every named file exists in the scorer's results directory."""

        return self.results is not None and all((self.results / name).is_file() for name in names)

    def has_output(self, *names: str) -> bool:
        """Whether every named file exists in the output directory."""

        return bool(names) and all((self.out / name).is_file() for name in names)

    def __str__(self) -> str:
        return str(self.entry)


__all__ = ["Target"]
