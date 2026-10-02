"""Public structure checks for the submitted solution."""

from __future__ import annotations

import os
from pathlib import Path

Violation = str


def check_solution(root: str | Path) -> list[Violation]:
    """Return delivery-structure violations for one solution directory."""

    root = Path(root)
    if not root.exists():
        return [f"{root.name}: path is required"]
    if not root.is_dir():
        return [f"{root.name}: path must be a directory"]

    entrypoint = root / "build.sh"
    if not entrypoint.exists():
        return [f"{root.name}/build.sh: path is required"]
    if not entrypoint.is_file():
        return [f"{root.name}/build.sh: path must be a file"]
    violations = []
    if entrypoint.stat().st_mode & 0o111 == 0:
        violations.append(f"{root.name}/build.sh: must be executable")
    for path in sorted(path for path in root.rglob("*") if path.is_file()):
        if path.suffix not in {".py", ".sh"}:
            violations.append(f"{root.name}/{path.relative_to(root)}: only .py and .sh files are allowed")
    return violations


def check_links(root: str | Path) -> list[Violation]:
    """Symbolic links anywhere in a delivery directory, the directory itself included.

    The delivery must be real files: a link is not collected, so whatever it points
    at is missing from the returned world or solution.
    """

    root = Path(root)
    if root.is_symlink():
        return [f"{root.name}: must be a directory, not a symbolic link"]
    if not root.is_dir():
        return []
    violations = []
    for directory, subdirectories, files in os.walk(root):
        for entry in sorted(subdirectories + files):
            path = Path(directory) / entry
            if path.is_symlink():
                violations.append(f"{root.name}/{path.relative_to(root)}: symbolic links are not allowed; write the file itself")
    return violations


__all__ = ["check_links", "check_solution"]
