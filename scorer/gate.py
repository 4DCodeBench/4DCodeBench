"""The two gates that decide which metrics a submission is scored on.

The video gate passes when `world/render.mp4` decodes and its (F, H, W, fps) match
the reference video. The world gate passes when the required files exist and
`check_world` reports no violation. A closed video gate marks every metric `error`;
a closed world gate marks the geometry metrics `error` and leaves the render-side
metrics computed.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from .outcome import INVALID_WORLD, NO_VIDEO, NO_WORLD, UNDECODABLE_VIDEO, VIDEO_MISMATCH

WORLD_FILES = ("world/render.mp4", "world/camera.json", "solution/build.sh")
WORLD_DIRECTORIES = ("world/meshes", "world/dynamics")


def decode_video(path):
    """Decode every frame of the first video stream; raise `ValueError` on any ffmpeg error."""

    decoded = subprocess.run(
        ["ffmpeg", "-v", "error", "-xerror", "-nostdin", "-threads", "1",
         "-i", str(path), "-map", "0:v:0", "-f", "null", "-"],
        capture_output=True, text=True, check=False,
    )
    if decoded.returncode or decoded.stderr.strip():
        raise ValueError(f"Video decoding failed: {decoded.stderr.strip()}")


def _gate(cause=None, detail=()):
    return {"ok": cause is None, "error": cause, "detail": list(detail)}


def video_gate(render: Path, reference: Path) -> tuple[list | None, dict]:
    """The submitted video's (F, H, W, fps) and the video gate record."""

    from .world import video_stream

    if not Path(render).is_file():
        return None, _gate(NO_VIDEO, [f"Missing {Path(render).name}"])
    try:
        decode_video(render)
    except ValueError as error:
        return None, _gate(UNDECODABLE_VIDEO, [str(error)[:2000]])
    got, want = video_stream(render), video_stream(reference)
    dimensions = [*got[:3], float(got[3])]
    if got != want:
        return dimensions, _gate(VIDEO_MISMATCH, [f"(F,H,W,fps) {got} != reference {want}"])
    return dimensions, _gate()


def world_gate(workspace: Path) -> dict:
    """The world gate record."""

    from .validation import check_links, check_world

    workspace = Path(workspace)
    missing = ([name for name in WORLD_FILES if not (workspace / name).is_file()]
               + [name + "/" for name in WORLD_DIRECTORIES if not (workspace / name).is_dir()])
    if missing:
        return _gate(NO_WORLD, ["Missing " + name for name in missing])
    world = workspace / "world"
    violations = check_links(world) or check_world(world)
    if violations:
        return _gate(INVALID_WORLD, violations)
    return _gate()


def gates(workspace: Path, reference: Path) -> tuple[list | None, dict, dict | None]:
    """Both gate records; the world gate is None when the video gate is closed."""

    dimensions, video = video_gate(Path(workspace) / "world/render.mp4", reference)
    return dimensions, video, world_gate(workspace) if video["ok"] else None


__all__ = ["decode_video", "gates", "video_gate", "world_gate", "WORLD_DIRECTORIES", "WORLD_FILES"]
