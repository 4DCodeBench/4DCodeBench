"""Shared runtime configuration and process helpers."""

from __future__ import annotations

import os
import shlex
import subprocess
import sys
from pathlib import Path

import tomllib

REPO = Path(__file__).resolve().parents[2]


def log(*items: object) -> None:
    print("[RUN]", *items)
    sys.stdout.flush()


CASE_VIDEO = "reference.mp4"
# the agent's task: this directory, rendered and mounted read-only at TASK_MOUNT
TASK_CONTEXT = REPO / "harness" / "agent_context"
TASK_ENTRYPOINT = "TASK.md"
TASK_MOUNT = Path("/task")
# one run's directory: runs/<kind>/<case>/<agent>-<model>-<effort>/<trial>/
RUN_WORKSPACE = "workspace"    # the collected world/ and solution/
RUN_LOGS = "logs"              # agent.log, the CLI's stdout and stderr
RUN_SESSION = "session"        # the CLI's own transcript
RUN_RECORD = "run.json"
AGENT_LOG = "agent.log"


def read_toml(path: Path) -> dict:
    return tomllib.loads(path.read_text(encoding="utf-8"))


def host_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO / path


def display(command: list[str]) -> str:
    return shlex.join(command)


def require_file(path: Path, label: str) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    return path


def resolve_case(cases_root: Path, case: str) -> str:
    """Return `<kind>/<case>` for a case given as `<case>` or `<kind>/<case>`.

    `runs/` and `data/` mirror this path under `cases/`. A bare name is looked up
    under each kind and must match exactly one; a case not on disk raises.
    """

    parts = Path(case).parts
    if len(parts) == 2:
        if not (cases_root / case).is_dir():
            raise FileNotFoundError(f"case does not exist: {cases_root / case}")
        return "/".join(parts)
    if len(parts) != 1:
        raise ValueError(f"a case is named <case> or <kind>/<case>, not {case!r}")
    found = [kind.name for kind in sorted(cases_root.iterdir())
             if kind.is_dir() and (kind / case).is_dir()]
    if len(found) != 1:
        raise FileNotFoundError(f"{case!r} names {len(found)} cases under {cases_root}")
    return f"{found[0]}/{case}"


def run_process(command: list[str], log_path: Path, timeout_sec: int) -> int:
    """Run `command` with its output in `log_path`; kill it and return 124 on timeout."""

    log("$", display(command))
    with log_path.open("w", encoding="utf-8") as handle:
        process = subprocess.Popen(command, stdout=handle, stderr=subprocess.STDOUT, text=True)
        try:
            return process.wait(timeout=timeout_sec)
        except subprocess.TimeoutExpired:
            process.kill()
            handle.write(f"\n[RUN] killed after {timeout_sec}s\n")
            return 124


def network_args(mode: str) -> list[str]:
    if mode == "default":
        return []
    if mode == "no-network":
        return ["--network", "none"]
    raise ValueError(f"unsupported network_mode {mode!r}")


def sif_network_args(mode: str) -> list[str]:
    if mode == "default":
        return []
    if mode == "no-network":
        return ["--net", "--network", "none"]
    raise ValueError(f"unsupported network_mode {mode!r}")


def gpu_args(count: int, *, backend: str) -> list[str]:
    """GPU flags for one container.

    Docker gets `--gpus device=$CUDA_VISIBLE_DEVICES` when that is set, else `--gpus <count>`.
    SIF gets `--nv`; `device_environment` passes the pinned device.
    """

    if count < 0:
        raise ValueError("gpus must be non-negative")
    if count == 0:
        return []
    if backend != "docker":
        return ["--nv"]
    selection = os.environ.get("CUDA_VISIBLE_DEVICES")
    return ["--gpus", f"device={selection}" if selection else str(count)]


# thread-pool sizes for OpenMP (torch), the BLAS behind numpy, numexpr and OpenCV
THREAD_VARIABLES = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                    "NUMEXPR_NUM_THREADS", "OPENCV_FOR_THREADS_NUM", "CV_NUM_THREADS")


def thread_environment(cpus: int) -> dict[str, str]:
    """Set every thread pool in THREAD_VARIABLES to the container's CPU count."""

    return {name: str(max(1, int(cpus))) for name in THREAD_VARIABLES}


def device_environment() -> dict[str, str]:
    """Return `CUDA_VISIBLE_DEVICES` for a SIF `--env` flag, or {} when unset.

    `--cleanenv` drops it, and the `*ENV_` passthrough prefix differs between
    Apptainer and SingularityCE.
    """

    selection = os.environ.get("CUDA_VISIBLE_DEVICES")
    return {"CUDA_VISIBLE_DEVICES": selection} if selection else {}


def docker_command(runtime: dict, arguments: list[str]) -> list[str]:
    return [runtime["docker"]["executable"], *arguments]


Mount = tuple[Path | None, Path | str, str]


def mount_args(flag: str, mounts: list[Mount]) -> list[str]:
    """Return `-v`/`-B` flag pairs for `mounts`, skipping a `None` source.

    Every mount names its mode explicitly, `ro` or `rw`.
    """

    arguments = []
    for source, target, mode in mounts:
        if source is None:
            continue
        if mode not in ("ro", "rw"):
            raise ValueError(f"a mount is ro or rw, not {mode!r}")
        arguments += [flag, f"{source}:{target}:{mode}"]
    return arguments


def container_command(runtime: dict, environment: dict, image: str,
                      mounts: list[Mount], command: list[str]) -> list[str]:
    """Build the Docker or SIF command that runs `command` in `image` with `mounts`.

    `command[0]` replaces the Docker entry point; SIF passes `command` to `exec`.
    Both backends take GPUs, network mode and thread counts from `environment`;
    Docker also applies its CPU and memory limits.
    """

    backend = runtime["backend"]
    gpus = int(environment["gpus"])
    threads = thread_environment(environment["cpus"])

    if backend == "docker":
        return docker_command(runtime, [
            "run",
            "--rm",
            f"--cpus={environment['cpus']}",
            f"--memory={environment['memory_mb']}m",
            *[argument for name, value in threads.items() for argument in ("--env", f"{name}={value}")],
            *gpu_args(gpus, backend=backend),
            *network_args(environment["network_mode"]),
            "--entrypoint",
            command[0],
            *mount_args("-v", mounts),
            image,
            *command[1:],
        ])

    if backend == "sif":
        variables = threads | device_environment()
        return [
            runtime["sif"]["executable"],
            "exec",
            *gpu_args(gpus, backend=backend),
            *sif_network_args(environment["network_mode"]),
            "--containall",
            "--no-home",
            "--cleanenv",
            *[argument for name, value in variables.items() for argument in ("--env", f"{name}={value}")],
            *mount_args("-B", mounts),
            image,
            *command,
        ]

    raise ValueError(f"unsupported runtime backend {backend!r}")
