#!/usr/bin/env python3
"""Run the task with Artificial Analysis' Stirrup agent loop inside the agent container.

    runner.py <task_path> <workspace> <model> <effort> <timeout_sec>

Calls an OpenAI-compatible endpoint (`OPENAI_BASE_URL`, `OPENAI_API_KEY`) with the
tools code_exec (bash in the workspace), view_image and finish. Each event is
appended to $HOME/.stirrup/session/<run>.jsonl. Limits are set by `FDCB_*` variables.
"""

import asyncio
import inspect
import json
import os
import re
import sys
import time
import urllib.request
from pathlib import Path

# requests go to the endpoint directly, never through a proxy
os.environ.setdefault("no_proxy", "*")
os.environ.setdefault("NO_PROXY", "*")
for variable in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
    os.environ.pop(variable, None)

from local_backend import LocalCodeExecToolProvider  # adapted copy of Stirrup's local backend
from stirrup import Agent
from stirrup.clients.chat_completions_client import ChatCompletionsClient
from stirrup.core.exceptions import OutputTokenLimitError
from stirrup.core.models import ToolResult
from stirrup.tools import SIMPLE_FINISH_TOOL
from stirrup.tools.view_image import ViewImageToolProvider
from stirrup.utils.logging import AgentLoggerBase

task_path, workspace, model, effort, timeout_sec = (
    sys.argv[1], Path(sys.argv[2]), sys.argv[3], sys.argv[4], int(sys.argv[5]))
BASE_URL = os.environ.get("OPENAI_BASE_URL", "http://127.0.0.1:8000/v1")
MAX_TURNS = 500
CONTEXT_TOKENS = 262144
MAX_TOKENS = 65536           # output budget of one response
MAX_TOKENS_RETRY = 131072    # output budget of the one retry after a truncated response
REQUEST_TIMEOUT = 1800       # seconds per request
SHELL_TIMEOUT = 36000        # seconds per shell command
SUMMARIZE_AT = 0.7           # share of the context window at which Stirrup summarises
RECONNECT_MINUTES = 45       # how long `generate` retries an unreachable endpoint
# The job's effort is sent as `reasoning_effort`; "none" omits the parameter.
EFFORT = None if effort.strip().lower() in ("", "none", "off") else effort
CONNECTION_ERRORS = ("APIConnectionError", "APITimeoutError", "ConnectError", "ReadTimeout", "RemoteProtocolError")
CONNECTION_MESSAGES = ("Connection error", "Connection refused", "Error code: 502", "Error code: 503", "Error code: 504")
# context-overflow errors of SGLang (includes the prompt size) and vLLM (does not)
SGLANG_OVERFLOW = re.compile(r"maximum context length of (\d+) tokens\. You requested a total of (\d+) tokens: "
                             r"(\d+) tokens from the input")
VLLM_OVERFLOW = re.compile(r"maximum context length is (\d+) tokens.*?requested (\d+) output tokens", re.S)


def is_connection_error(exc: Exception) -> bool:
    return type(exc).__name__ in CONNECTION_ERRORS or any(text in str(exc) for text in CONNECTION_MESSAGES)


async def wait_healthy(deadline: float) -> None:
    """Poll the endpoint's /health until it returns 200 or `deadline` passes."""

    def healthy() -> bool:
        try:
            return urllib.request.urlopen(BASE_URL.rsplit("/v1", 1)[0] + "/health", timeout=10).status == 200
        except Exception:  # noqa: BLE001
            return False

    while time.time() < deadline and not await asyncio.to_thread(healthy):
        await asyncio.sleep(15)


class Client(ChatCompletionsClient):
    """Stirrup's client with retries on connection errors and on output-budget overflows."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # copy vLLM's `message.reasoning` into `reasoning_content`, the field Stirrup reads
        completions = getattr(getattr(getattr(self, "_client", None), "chat", None), "completions", None)
        if completions is None:
            return
        create = completions.create

        async def create_with_reasoning(*a, **kw):
            response = await create(*a, **kw)
            for choice in getattr(response, "choices", None) or []:
                message = choice.message
                if not getattr(message, "reasoning_content", None) and getattr(message, "reasoning", None):
                    message.reasoning_content = message.reasoning
            return response

        completions.create = create_with_reasoning

    async def generate(self, messages, tools=None, **kw):
        """Retry connection errors for RECONNECT_MINUTES, waiting 30 s and for /health each time."""

        deadline = time.time() + 60 * RECONNECT_MINUTES
        while True:
            try:
                return await self._generate_fitted(messages, tools, **kw)
            except Exception as exc:  # noqa: BLE001
                if not is_connection_error(exc) or time.time() > deadline:
                    raise
                print(f"[stirrup] endpoint unreachable ({type(exc).__name__}); retrying", file=sys.stderr)
                await asyncio.sleep(30)
                await wait_healthy(deadline)

    async def _generate_fitted(self, messages, tools=None, **kw):
        """Run one step, retrying with a smaller output budget on a context overflow.

        A response cut off at the output budget is retried once with MAX_TOKENS_RETRY.
        """

        saved = self._max_tokens
        try:
            try:
                return await super().generate(messages, tools, **kw)
            except OutputTokenLimitError:
                self._max_tokens = MAX_TOKENS_RETRY
                print(f"[stirrup] output budget exhausted at {saved}; retrying with {MAX_TOKENS_RETRY}",
                      file=sys.stderr)
                return await super().generate(messages, tools, **kw)
            except Exception as exc:  # noqa: BLE001
                sglang = SGLANG_OVERFLOW.search(str(exc))
                if not sglang and not VLLM_OVERFLOW.search(str(exc)):
                    raise
                # SGLang's error gives the room left in the window; vLLM's does not
                fitted = [max(1024, int(sglang.group(1)) - int(sglang.group(3)) - 512)] if sglang else []
                last = exc
                for budget in [*fitted, 32768, 16384, 8192, 4096, 2048]:
                    if budget >= saved:
                        continue
                    self._max_tokens = budget
                    print(f"[stirrup] prompt near the window: retrying with output budget {budget}", file=sys.stderr)
                    try:
                        return await super().generate(messages, tools, **kw)
                    except Exception as retry:  # noqa: BLE001
                        last = retry
                        if not re.search(r"requested (\d+) output tokens|You requested a total of", str(retry)):
                            raise
                raise last
        finally:
            self._max_tokens = saved


class WorkspaceExec(LocalCodeExecToolProvider):
    """Stirrup's local shell, rooted at the persistent workspace instead of a temporary dir."""

    async def __aenter__(self):
        self._temp_dir = workspace
        workspace.mkdir(parents=True, exist_ok=True)
        return self.get_code_exec_tool(description=self._description)

    async def __aexit__(self, *args):
        return None

    def get_view_image_tool(self, *args, **kwargs):
        """Return view_image with failures reported as a short text result."""

        tool = super().get_view_image_tool(*args, **kwargs)
        executor = tool.executor

        async def guarded(params):
            path = getattr(params, "path", None) or getattr(params, "image_path", None) or ""
            try:
                result = executor(params)
                return await result if inspect.isawaitable(result) else result
            except Exception as exc:  # noqa: BLE001
                return ToolResult(content=f"[view_image] {path}: cannot be shown ({type(exc).__name__}: "
                                          f"{str(exc)[:200]}). Convert it to a PNG/JPG first.")

        tool.executor = guarded
        return tool

    def _check_absolute_paths(self, argv, cmd):
        # Stirrup's check rejects /tmp/, /home/, /var/ and /etc/ paths; the container is the sandbox
        return None


def _strip(value):
    """Return a pydantic dump made JSON-safe, image and binary payloads replaced by their size."""

    if isinstance(value, dict):
        if value.get("kind") == "image_content_block" or isinstance(value.get("data"), (bytes, bytearray)):
            return {"kind": "image", "bytes": len(value.get("data") or b""), "mime": value.get("mime_type")}
        return {key: _strip(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_strip(item) for item in value]
    if isinstance(value, (bytes, bytearray)):
        return f"<{len(value)} bytes>"
    return value


class JsonlLogger(AgentLoggerBase):
    """Logger that appends one flushed JSON line per event, in place of Stirrup's console logger."""

    def __init__(self, path: Path):
        try:
            super().__init__()
        except TypeError:
            super().__init__(depth=0)
        if not hasattr(self, "depth"):
            self.depth = 0
        path.parent.mkdir(parents=True, exist_ok=True)
        self.file = path.open("a", encoding="utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.file.close()

    def _write(self, **record):
        record["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        self.file.write(json.dumps(record, default=str) + "\n")
        self.file.flush()

    def _dump(self, message):
        try:
            return _strip(message.model_dump())
        except Exception:  # noqa: BLE001
            return str(message)[:2000]

    def on_step(self, step, tool_calls=0, input_tokens=0, output_tokens=0):
        self._write(type="step", step=step, tool_calls=tool_calls, input_tokens=input_tokens,
                    output_tokens=output_tokens)

    def assistant_message(self, turn, max_turns, assistant_message):
        self._write(type="assistant", turn=turn, message=self._dump(assistant_message))

    def user_message(self, user_message):
        self._write(type="user", message=self._dump(user_message))

    def task_message(self, task):
        self._write(type="task", task=task if isinstance(task, str) else _strip(task))

    def tool_result(self, tool_message):
        self._write(type="tool", message=self._dump(tool_message))

    def context_summarization_start(self, pct_used, cutoff):
        self._write(type="summarization_start", pct_used=pct_used, cutoff=cutoff)

    def context_summarization_complete(self, summary, bridge):
        self._write(type="summarization_complete", summary=summary[:4000], bridge=bridge[:2000])

    def debug(self, message, *args):
        pass

    def info(self, message, *args):
        try:
            self._write(type="info", text=(message % args) if args else str(message))
        except Exception:  # noqa: BLE001
            self._write(type="info", text=str(message))

    def warning(self, message, *args):
        self._write(type="warning", text=str(message))

    def error(self, message, *args):
        self._write(type="error", text=str(message))


async def main() -> int:
    task = Path(task_path).read_text(encoding="utf-8")
    client = Client(
        model=model,
        base_url=BASE_URL,
        api_key=os.environ.get("OPENAI_API_KEY", "local"),
        max_tokens=MAX_TOKENS,
        context_window_tokens=CONTEXT_TOKENS,
        reasoning_effort=EFFORT,
        timeout=REQUEST_TIMEOUT,
        max_retries=0,   # `Client.generate` does the retrying
    )
    shell = WorkspaceExec(temp_base_dir=workspace, shell_timeout=SHELL_TIMEOUT)
    home = Path(os.environ.get("HOME", "/root"))
    run_id = time.strftime("%Y%m%d-%H%M%S")
    logger = JsonlLogger(home / ".stirrup" / "session" / f"{run_id}.jsonl")
    agent = Agent(
        client=client,
        name="4dcb",
        max_turns=MAX_TURNS,
        tools=[shell, ViewImageToolProvider(shell)],
        finish_tool=SIMPLE_FINISH_TOOL,
        # summarise at `SUMMARIZE_AT` of the window; unwind turns on overflow
        context_summarization_cutoff=SUMMARIZE_AT,
        recover_from_context_overflow=True,
        logger=logger,
    )
    async with agent.session(output_dir=str(home / ".stirrup" / "output" / run_id)) as session:
        try:
            finish, history, _ = await asyncio.wait_for(session.run(task), timeout=timeout_sec)
            logger.info("finished: %s", str(finish)[:500])
        except asyncio.TimeoutError:
            logger.info("timeout")
            print("[stirrup] timeout", file=sys.stderr)
            return 124
        except Exception as exc:  # noqa: BLE001
            message = str(exc)
            if "context length" in message or "context window" in message.lower() or "Input length" in message:
                # an exhausted context window ends the run with exit 0
                logger.error("context_limit: " + message[:500])
                print(f"[stirrup] context limit reached ({message[:200]})", file=sys.stderr)
                return 0
            raise
    print(f"[stirrup] finished turns={len(history)} finish={finish}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
