"""Per-process registry of in-flight chat-run asyncio.Tasks.

The chat_runs DB row is just state; the actual work is an asyncio.Task.
When the staleness sweeper flips a run to 'failed' (or a user explicitly
cancels), the in-memory Task keeps running and racks up cost — the
"orphaned cells" bug. This registry bridges the two: the agent registers
its Task on start, the sweeper / user-cancel path looks it up by run_id
and calls .cancel().

In-process only. Workers are single-process today; if we ever split, this
becomes a Redis pub/sub or similar.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Dict, Optional


log = logging.getLogger(__name__)


class ChatRunTaskRegistry:
    def __init__(self) -> None:
        self._tasks: Dict[str, asyncio.Task] = {}
        self._lock = asyncio.Lock()

    async def register(self, run_id: str, task: asyncio.Task) -> None:
        async with self._lock:
            self._tasks[run_id] = task
        log.debug("chat-run: registered task for run_id=%s", run_id)

    async def unregister(self, run_id: str, task: asyncio.Task) -> None:
        """Best-effort: only remove if the registered task is still `task`
        — guards against a newer overlapping run de-registering itself
        on the older run's completion."""
        async with self._lock:
            if self._tasks.get(run_id) is task:
                self._tasks.pop(run_id, None)

    async def cancel(self, run_id: str, reason: str = "cancelled") -> bool:
        async with self._lock:
            task = self._tasks.get(run_id)
        if task is None or task.done():
            return False
        task.cancel()
        log.info("chat-run: cancelled task run_id=%s reason=%s", run_id, reason)
        return True

    def cancel_sync(self, run_id: str, reason: str = "cancelled") -> bool:
        """Sync variant for callers outside an event loop (staleness
        sweeper runs from a Celery beat job / scheduler thread). Skips
        the lock — best-effort, but safe since dict.get is atomic in
        CPython and Task.cancel() is thread-safe to call."""
        task = self._tasks.get(run_id)
        if task is None or task.done():
            return False
        # Schedule cancellation on the task's own loop. asyncio Task.cancel
        # is documented as thread-safe; loop.call_soon_threadsafe is the
        # belt-and-suspenders version.
        try:
            loop = task.get_loop()
            loop.call_soon_threadsafe(task.cancel)
        except Exception:
            try:
                task.cancel()
            except Exception:
                return False
        log.info("chat-run: cancelled task (sync) run_id=%s reason=%s", run_id, reason)
        return True

    def is_running(self, run_id: str) -> bool:
        task = self._tasks.get(run_id)
        return bool(task and not task.done())


REGISTRY = ChatRunTaskRegistry()
