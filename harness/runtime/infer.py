"""Run each job's agent on its case and collect the delivered world/ and solution/.

    python harness/runtime/infer.py --jobs J.toml --runtime R.toml [--dry-run]

Jobs run one after another, each in its own agent container. Each run goes to
`runs/<kind>/<case>/<agent>-<model>-<effort>/<trial>/`, replacing an existing one.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import launch
from agents import agent as get_agent
from common import (
    AGENT_LOG,
    CASE_VIDEO,
    REPO,
    RUN_LOGS,
    RUN_RECORD,
    RUN_SESSION,
    RUN_WORKSPACE,
    TASK_CONTEXT,
    TASK_ENTRYPOINT,
    TASK_MOUNT,
    display,
    docker_command,
    host_path,
    log,
    network_args,
    require_file,
    resolve_case,
    run_process,
    sif_network_args,
)
from cost import summarize_cost, summarize_steps

sys.path.insert(0, str(REPO))

from checker.paths import (
    HOME_DIR,
    SOLUTION_DIR,
    SOLUTION_ENTRYPOINT,
    VIDEO_PATH,
    WORKSPACE_DIR,
    WORLD_DIR,
)

STIRRUP_DIR = Path(__file__).resolve().parents[1] / "stirrup"   # bound at /opt/stirrup
PLACEHOLDER = re.compile(r"{{([a-z_]+(?:\.[a-z_]+)*)}}")


def agent_gpu_args(backend: str) -> list[str]:
    """GPU flags for the agent container: `--nv` on SIF; on Docker, the pinned devices or all."""

    if backend == "sif":
        return ["--nv"]
    if backend == "docker":
        visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        return ["--gpus", f"device={visible}" if visible else "all"]
    raise ValueError(f"unsupported runtime backend {backend!r}")


def prepare_agent_context(target: Path) -> None:
    """Copy the agent context to `target` and fill the `{{paths.*}}` placeholders of TASK.md."""

    shutil.copytree(TASK_CONTEXT, target)
    entrypoint = target / TASK_ENTRYPOINT
    template = entrypoint.read_text(encoding="utf-8")
    context = {
        "paths": {
            "task": TASK_MOUNT,
            "video": VIDEO_PATH,
            "world": WORLD_DIR,
            "solution": SOLUTION_DIR,
        },
    }

    def replace(match: re.Match) -> str:
        value: object = context
        for key in match.group(1).split("."):
            value = value[key]
        if isinstance(value, (dict, list)):
            raise TypeError(f"task placeholder {match.group(0)} is not scalar")
        return str(value)

    rendered = PLACEHOLDER.sub(replace, template)
    if "{{" in rendered or "}}" in rendered:
        raise ValueError("task template contains an invalid placeholder")
    entrypoint.write_text(rendered, encoding="utf-8")


def agent_command(
    runtime: dict,
    run_dir: Path,
    case_dir: Path,
    workspace_source: Path,
    context_source: Path,
    name: str,
    model: str,
    effort: str,
) -> list[str]:
    """Build the container command of one agent run.

    Mounts a fresh home, the empty workspace, the task (ro), the video (ro), the
    session directory and the agent's credential file (ro).
    """

    backend = runtime["backend"]
    agent_spec = runtime["agent"]
    environment = agent_spec["environment"]
    cli = get_agent(name)
    settings = agent_spec.get(name, {})
    credential_mount: list[str] = []
    if "credential" in settings:
        credential = require_file(host_path(runtime["auth_root"]) / settings["credential"], f"{name} credential")
        credential_mount = [f"{credential}:{settings['container']}:ro"]
    variables = {key: str(value) for key, value in settings.get("env", {}).items()}
    task_path = TASK_MOUNT / TASK_ENTRYPOINT
    inner = cli.command(
        str(task_path), str(WORKSPACE_DIR), model, effort, int(agent_spec["timeout_sec"])
    )
    session_source = run_dir / RUN_SESSION / cli.session_dir
    session_source.mkdir(parents=True, exist_ok=True)
    session_target = HOME_DIR / cli.session_dir
    video_source = case_dir / CASE_VIDEO
    require_file(video_source, "case input video")
    image = agent_spec["image"][backend]
    # Both backends mount a fresh, empty home in place of the image's.
    home_source = workspace_source.parent / "home"
    home_source.mkdir()
    # The session mount point must exist in the host home, since a bind cannot
    # create its destination inside another bind.
    (home_source / cli.session_dir).mkdir(parents=True, exist_ok=True)

    if backend == "docker":
        arguments = [
            "run",
            "--rm",
            *agent_gpu_args(backend),
            *network_args(environment["network_mode"]),
            *(item for key, value in variables.items() for item in ("--env", f"{key}={value}")),
            "-v",
            f"{home_source}:{HOME_DIR}",
            "-v",
            f"{workspace_source}:{WORKSPACE_DIR}",
            "-v",
            f"{context_source}:{TASK_MOUNT}:ro",
            "-v",
            f"{video_source}:{VIDEO_PATH}:ro",
            "-v",
            f"{session_source}:{session_target}",
            *(("-v", f"{STIRRUP_DIR}:/opt/stirrup:ro") if name == "stirrup" else ()),
            *(item for mount in credential_mount for item in ("-v", mount)),
            "-w",
            str(WORKSPACE_DIR),
            image,
            *inner,
        ]
        return docker_command(runtime, arguments)

    if backend == "sif":
        pack_env: list[str] = []
        pack_bind: list[str] = []
        # NVIDIA's EGL vendor file, so glvnd dispatches EEVEE to the library `--nv` binds.
        egl_icd = Path(__file__).resolve().parents[1] / "apptainer" / "10_nvidia.json"
        if egl_icd.is_file():
            pack_bind += ["-B", f"{egl_icd}:/usr/share/glvnd/egl_vendor.d/10_nvidia.json:ro"]
        # nvidia-smi is bound directly: `--nv` places it via an underlay that fails with many binds
        if Path("/usr/bin/nvidia-smi").is_file():
            pack_bind += ["-B", "/usr/bin/nvidia-smi:/usr/bin/nvidia-smi:ro"]
        # /tmp and /var/tmp live in this run's staging directory.
        scratch_root = workspace_source.parent / "scratch"
        for sub, dest in (("tmp", "/tmp"), ("vartmp", "/var/tmp")):
            (scratch_root / sub).mkdir(parents=True, exist_ok=True)
            pack_bind += ["-B", f"{scratch_root / sub}:{dest}"]
        # `--cleanenv` drops the GPU pin, so it is passed explicitly.
        if "CUDA_VISIBLE_DEVICES" in os.environ:
            pack_env += ["--env", f"CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}"]
        if name == "stirrup":
            pack_bind += ["-B", f"{STIRRUP_DIR}:/opt/stirrup:ro"]
        return [
            runtime["sif"]["executable"],
            "exec",
            *agent_gpu_args(backend),
            *sif_network_args(environment["network_mode"]),
            *(item for key, value in variables.items() for item in ("--env", f"{key}={value}")),
            *pack_env,
            "--containall",
            "--cleanenv",
            # own PID namespace, so an agent's `pkill` cannot reach other processes on the node
            "--pid",
            "--home",
            f"{home_source}:{HOME_DIR}",
            "--writable-tmpfs",
            "--pwd",
            str(WORKSPACE_DIR),
            "-B",
            f"{workspace_source}:{WORKSPACE_DIR}",
            "-B",
            f"{context_source}:{TASK_MOUNT}:ro",
            "-B",
            f"{video_source}:{VIDEO_PATH}:ro",
            "-B",
            f"{session_source}:{session_target}",
            *pack_bind,
            *(item for mount in credential_mount for item in ("-B", mount)),
            image,
            *inner,
        ]

    raise ValueError(f"unsupported runtime backend {backend!r}")


def collect_delivery(source: Path, target: Path, dropped: list[str] | None = None) -> list[str]:
    """Copy world/ and solution/ from `source` to `target`; return the names copied.

    Symbolic links are neither followed nor copied, since they would resolve on the
    host; each one's workspace path is appended to `dropped`.
    """

    dropped = [] if dropped is None else dropped

    def skip_links(directory: str, entries: list[str]) -> list[str]:
        links = [entry for entry in entries if os.path.islink(os.path.join(directory, entry))]
        dropped.extend(os.path.relpath(os.path.join(directory, entry), source) for entry in links)
        return links

    names = []
    for name in (WORLD_DIR.name, SOLUTION_DIR.name):
        directory = source / name
        if directory.is_symlink():
            dropped.append(name)
            continue
        if directory.is_dir():
            target.mkdir(parents=True, exist_ok=True)
            shutil.copytree(directory, target / name, symlinks=True, ignore=skip_links)
            names.append(name)
    return names


def ownership_release_command(runtime: dict, workspace: Path, session: Path) -> list[str] | None:
    """Return the Docker command that chowns the workspace and session to the host user.

    None on the SIF backend, whose files already belong to the host user.
    """

    if runtime["backend"] == "sif":
        return None
    if runtime["backend"] != "docker":
        raise ValueError(f"unsupported runtime backend {runtime['backend']!r}")
    return docker_command(
        runtime,
        [
            "run",
            "--rm",
            "--network",
            "none",
            "-v",
            f"{workspace}:{WORKSPACE_DIR}",
            "-v",
            f"{session}:/session",
            runtime["agent"]["image"]["docker"],
            "chown",
            "-R",
            f"{os.getuid()}:{os.getgid()}",
            str(WORKSPACE_DIR),
            "/session",
        ],
    )


def one_infer(job: dict, runtime: dict, dry_run: bool) -> dict:
    """Run one job, write its run.json, and return the record."""

    case, name = job["case"], job["agent"]
    model, effort = job["model"], job["effort"]
    trial = int(job["trial"])
    cases_root = host_path(runtime["cases_root"])
    case = resolve_case(cases_root, case)     # `<kind>/<case>`
    case_dir = cases_root / case
    cli = get_agent(name)
    if cli.efforts and effort not in cli.efforts:   # empty: any value
        raise ValueError(f"{name} has no effort {effort!r}; it takes {list(cli.efforts)}")

    final_dir = host_path(runtime["runs_root"]) / case / f"{name}-{model}-{effort}" / str(trial)

    with tempfile.TemporaryDirectory(prefix="4dcb-run-") as temporary:
        staging = Path(temporary)
        if dry_run:
            run_dir = staging / "run"      # a dry run leaves runs/ untouched
        else:
            run_dir = final_dir
            if run_dir.exists():
                shutil.rmtree(run_dir)
        (run_dir / RUN_LOGS).mkdir(parents=True, exist_ok=True)
        workspace_source = staging / "workspace"
        context_source = staging / "task"
        workspace_source.mkdir()
        prepare_agent_context(context_source)
        agent_cmd = agent_command(
            runtime,
            run_dir,
            case_dir,
            workspace_source,
            context_source,
            name,
            model,
            effort,
        )
        record = {
            "case": case,
            "agent": name,
            "model": model,
            "effort": effort,
            "trial": trial,
            "backend": runtime["backend"],
            "session": RUN_SESSION,
            "agent_command": display(agent_cmd),
        }
        if dry_run:
            log("$", record["agent_command"])
            log("dry run, nothing executed")
            return record

        try:
            started = time.time()
            record["agent_exit"] = run_process(
                agent_cmd,
                run_dir / RUN_LOGS / AGENT_LOG,
                int(runtime["agent"]["timeout_sec"]),
            )
            record["agent_seconds"] = round(time.time() - started, 1)
            try:
                cost = summarize_cost(run_dir / RUN_LOGS / AGENT_LOG, name, model)
                if cost is not None:
                    record.update(cost)
            except Exception as exc:  # a parse failure is recorded, not raised
                record["cost_error"] = str(exc)[:200]
            try:
                steps = summarize_steps(run_dir / RUN_SESSION, name)
                if steps is not None:
                    record.update(steps)
            except Exception as exc:
                record["steps_error"] = str(exc)[:200]
            collection = run_dir / RUN_WORKSPACE
            dropped: list[str] = []
            record["artifacts"] = collect_delivery(workspace_source, collection, dropped)
            if dropped:
                record["symlinks_dropped"] = dropped
                log("symlinks left out of the delivery:", ", ".join(dropped))
        finally:
            release = ownership_release_command(runtime, workspace_source, run_dir / RUN_SESSION)
            if release is not None:
                subprocess.run(release, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT, check=True)

        world = collection / WORLD_DIR.name
        solution = collection / SOLUTION_DIR.name
        entrypoint = solution / SOLUTION_ENTRYPOINT.name
        record["delivered"] = world.is_dir() and solution.is_dir() and entrypoint.is_file()
        if record["delivered"]:
            log("world and solution delivered")
        else:
            log("incomplete delivery")

        (run_dir / RUN_RECORD).write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        log(json.dumps({key: value for key, value in record.items() if not key.endswith("_command")}, sort_keys=True))

    return record


def succeeded(record: dict) -> bool:
    return record.get("agent_exit") == 0 and bool(record.get("delivered"))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a list of inference jobs.")
    parser.add_argument("--jobs", type=Path, required=True, help="the job list")
    parser.add_argument("--runtime", type=Path, required=True, help="the deployment configuration")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    return launch.execute("infer", args.jobs, args.runtime, args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
