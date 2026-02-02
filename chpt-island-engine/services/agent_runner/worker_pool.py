"""
WORKER POOL
===========
Thread pool wrapper + safe shutdown.

One-liner idea:
    "Parallelize work, serialize commits."
"""

from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor, Future
from typing import Callable, Any


class WorkerPool:
    def __init__(self, max_workers: int = 4):
        self._ex = ThreadPoolExecutor(max_workers=max_workers)

    def submit(self, fn: Callable[..., Any], *args, **kwargs) -> Future:
        return self._ex.submit(fn, *args, **kwargs)

    def shutdown(self):
        self._ex.shutdown(wait=True)
