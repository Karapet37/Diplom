from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Callable


class Scheduler:
    def __init__(self, max_workers: int = 4):
        self._pool = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="rv-worker")

    def submit(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Future[Any]:
        return self._pool.submit(fn, *args, **kwargs)

    def shutdown(self, wait: bool = True) -> None:
        self._pool.shutdown(wait=wait)
