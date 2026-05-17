"""In-process cancel registry for REST-driven enrichment runs.

Use case: the Stop button in the table tabs bar. The chat run is
cancellable via SSE disconnect; the REST endpoints
(/v2/projects/.../enrichments/.../run) aren't because they're plain HTTP
requests. This registry tracks the asyncio.Task each REST run spawns so
the Stop button can call `.cancel()` and the cell-loop's CancelledError
handler unwinds cleanly.

Keyed by (project_id, enrichment_id). If a user kicks off two overlapping
runs on the same enrichment (e.g. "Run first 10" while "Run unfilled" is
still running), the latest task overwrites the registry entry; the older
task is still running but no longer cancellable via this path. v1
limitation — acceptable since the common case is one run per enrichment
at a time.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Dict, Tuple


log = logging.getLogger(__name__)


class CancelRegistry:
    """Project-scoped task registry for cancellable REST enrichment runs."""

    def __init__(self) -> None:
        self._tasks: Dict[Tuple[str, str], asyncio.Task] = {}
        self._lock = asyncio.Lock()

    async def register(self, project_id: str, enrichment_id: str, task: asyncio.Task) -> None:
        key = (project_id, enrichment_id)
        async with self._lock:
            self._tasks[key] = task
        log.debug("cancel: registered %s", key)

    async def unregister(self, project_id: str, enrichment_id: str, task: asyncio.Task) -> None:
        """Remove the task from the registry only if it's still the current
        one. Prevents a newer overlapping run from being de-registered by
        the older run's completion."""
        key = (project_id, enrichment_id)
        async with self._lock:
            if self._tasks.get(key) is task:
                self._tasks.pop(key, None)

    async def cancel_enrichment(self, project_id: str, enrichment_id: str) -> bool:
        """Cancel the active task for (project, enrichment). Returns True
        if a task was found and cancel was requested."""
        key = (project_id, enrichment_id)
        async with self._lock:
            task = self._tasks.get(key)
        if task is None or task.done():
            return False
        task.cancel()
        log.info("cancel: signalled %s", key)
        return True

    async def list_running(self, project_id: str) -> list[str]:
        """Return enrichment_ids with an active task for this project."""
        async with self._lock:
            return [eid for (pid, eid), t in self._tasks.items() if pid == project_id and not t.done()]


# Singleton — REST endpoint registers tasks; cancel endpoint looks them up.
REGISTRY = CancelRegistry()
