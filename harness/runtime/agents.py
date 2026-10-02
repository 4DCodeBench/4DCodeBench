"""The headless command line of each agent, and the agent registry.

All agents run in one image. `infer.py` adds each agent's environment and mounts
only that agent's credential file.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Agent:
    """One agent: its name, accepted efforts (empty: any), and transcript dir under $HOME."""

    name: str
    efforts: tuple[str, ...]
    session_dir: str

    def command(self, task_path: str, workspace_path: str, model: str, effort: str, timeout_sec: int) -> list[str]:
        raise NotImplementedError


@dataclass(frozen=True)
class ClaudeCode(Agent):
    def command(self, task_path: str, workspace_path: str, model: str, effort: str, timeout_sec: int) -> list[str]:
        return [
            "bash", "-c",
            (
                f'claude -p "$(cat {task_path})" --model {model} --effort {effort} '
                f"--permission-mode dontAsk --allowedTools Bash Edit Write Read Glob Grep NotebookEdit "
                f"--output-format json"
            ),
        ]


@dataclass(frozen=True)
class Codex(Agent):
    def command(self, task_path: str, workspace_path: str, model: str, effort: str, timeout_sec: int) -> list[str]:
        return [
            "bash", "-c",
            (
                f'codex exec "$(cat {task_path})" --model {model} '
                f'-c model_reasoning_effort="{effort}" '
                # plugin sync is off: it copies bundles into ~/.codex at startup and can block exit
                f"--disable plugins --disable remote_plugin "
                f"--dangerously-bypass-approvals-and-sandbox --skip-git-repo-check -C {workspace_path} --json"
            ),
        ]


@dataclass(frozen=True)
class Antigravity(Agent):
    def command(self, task_path: str, workspace_path: str, model: str, effort: str, timeout_sec: int) -> list[str]:
        return [
            "bash", "-c",
            (
                f'agy -p "$(cat {task_path})" --model {model} --effort {effort} '
                f"--dangerously-skip-permissions --output-format json --print-timeout {timeout_sec}s"
            ),
        ]


@dataclass(frozen=True)
class Stirrup(Agent):
    """Artificial Analysis' Stirrup agent loop: harness/stirrup/runner.py at /opt/stirrup."""

    def command(self, task_path: str, workspace_path: str, model: str, effort: str, timeout_sec: int) -> list[str]:
        return ["bash", "-c", f"python3 /opt/stirrup/runner.py {task_path} {workspace_path} {model} {effort} {timeout_sec}"]


AGENTS: dict[str, Agent] = {
    # stirrup takes any effort and sends it as reasoning_effort
    "stirrup": Stirrup(name="stirrup", efforts=(), session_dir=".stirrup"),
    "claude": ClaudeCode(
        name="claude",
        efforts=("low", "medium", "high", "xhigh", "max"),
        session_dir=".claude/projects",
    ),
    "codex": Codex(
        name="codex",
        efforts=("low", "medium", "high", "xhigh", "max"),
        session_dir=".codex/sessions",
    ),
    "antigravity": Antigravity(
        name="antigravity",
        efforts=("low", "medium", "high"),
        session_dir=".gemini/antigravity-cli/brain",
    ),
}


def agent(name: str) -> Agent:
    if name not in AGENTS:
        raise KeyError(f"unknown agent {name!r}; known: {sorted(AGENTS)}")
    return AGENTS[name]
