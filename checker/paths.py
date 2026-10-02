"""Fixed Agent container interface."""

from pathlib import Path

WORKSPACE_DIR = Path("/workspace")
WORLD_DIR = WORKSPACE_DIR / "world"
SOLUTION_DIR = WORKSPACE_DIR / "solution"
SOLUTION_ENTRYPOINT = SOLUTION_DIR / "build.sh"
VIDEO_PATH = Path("/input/reference.mp4")
HOME_DIR = Path("/root")

__all__ = [
    "HOME_DIR",
    "SOLUTION_DIR",
    "SOLUTION_ENTRYPOINT",
    "VIDEO_PATH",
    "WORKSPACE_DIR",
    "WORLD_DIR",
]
