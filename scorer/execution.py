"""A persistent worker process that runs method calls serially, each with a deadline."""

from __future__ import annotations

import multiprocessing as mp
import os
import signal
from dataclasses import dataclass, field


class TaskFailure(RuntimeError):
    """A worker call raised, timed out, or lost its process."""

    pass


def serve(connection, factory):
    """Worker loop: build `factory()`, then run each `(method, args)` request received."""
    os.setsid()
    state = factory()
    connection.send((True, None))
    while True:
        method, args = connection.recv()
        try:
            result = getattr(state, method)(*args)
        except Exception as error:  # noqa: BLE001 — task failures cross the process boundary
            connection.send((False, f"{type(error).__name__}: {error}"))
        else:
            connection.send((True, result))


@dataclass
class Timeouts:
    """Per-call deadlines in seconds: `metric` or `prepare`, unless `overrides` names the task."""

    metric: float = 1800
    prepare: float = 600
    overrides: dict[str, float] = field(default_factory=dict)

    def seconds(self, name: str, preparation: bool = False) -> float:
        return self.overrides.get(name, self.prepare if preparation else self.metric)


class Worker:
    """A spawned process holding one `factory()` object across calls.

    The process leads its own process group; `close` kills the whole group. A
    timed-out or broken call closes the worker and raises `TaskFailure`.
    """

    def __init__(self, factory, timeout: float):
        context = mp.get_context("spawn")
        self.connection, child = context.Pipe()
        self.process = context.Process(target=serve, args=(child, factory))
        self.process.start()
        child.close()
        try:
            self.receive(timeout)
        except BaseException:
            self.close()
            raise

    def receive(self, timeout: float):
        if not self.connection.poll(timeout):
            self.close()
            raise TaskFailure(f"timeout after {timeout:g}s")
        try:
            ok, value = self.connection.recv()
        except EOFError:
            self.close()
            raise TaskFailure("worker exited without a result") from None
        if not ok:
            raise TaskFailure(value)
        return value

    def call(self, method: str, *args, timeout: float):
        try:
            self.connection.send((method, args))
        except (BrokenPipeError, OSError):
            self.close()
            raise TaskFailure("worker is no longer running") from None
        return self.receive(timeout)

    @property
    def alive(self) -> bool:
        return self.process.is_alive()

    def close(self):
        if self.process.pid is not None:
            try:
                os.killpg(self.process.pid, signal.SIGKILL)
            except ProcessLookupError:
                if self.process.is_alive():
                    self.process.kill()
            self.process.join()
        self.connection.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
