"""Decode one upcoming batch while the caller processes the current batch."""

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager


@contextmanager
def prefetch(source):
    """Yield an iterator over `source` that fetches the next item on a worker thread.

    `source` is closed on exit.
    """

    end = object()

    def ready(pool):
        pending = pool.submit(next, source, end)
        while (batch := pending.result()) is not end:
            pending = pool.submit(next, source, end)
            yield batch

    try:
        with ThreadPoolExecutor(max_workers=1, thread_name_prefix="video-decode") as pool:
            yield ready(pool)
    finally:
        source.close()
