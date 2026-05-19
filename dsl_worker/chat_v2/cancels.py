"""In-process cancel registry for REST-driven enrichment runs.

Use case: the Stop button in the table tabs bar. The chat run is
cancellable via SSE disconnect; the REST endpoints
(/v2/projects/.../enrichments/.../run) aren't because they're plain HTTP
requests. This registry tracks the asyncio.Tasks each REST run spawns so
the Stop button can call `.cancel()` and the cell-loop's CancelledError
handler unwinds cleanly.

Keyed by (project_id, enrichment_id) → set of in-flight Tasks. Two
overlapping runs on the same enrichment (e.g. user clicks "Run first 10"
then immediately "Run unfilled") both register here; cancel_enrichment
cancels all of them.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Dict, Set, Tuple


log = logging.getLogger(__name__)


class CancelRegistry:
    """Project-scoped task registry for cancellable REST enrichment runs."""

    def __init__(self) -> None:
        # Set of tasks per (project, enrichment). Multiple overlapping
        # runs all register; Stop cancels every one.
        self._tasks: Dict[Tuple[str, str], Set[asyncio.Task]] = {}
        self._lock = asyncio.Lock()

    async def register(self, project_id: str, enrichment_id: str, task: asyncio.Task) -> None:
        key = (project_id, enrichment_id)
        async with self._lock:
            self._tasks.setdefault(key, set()).add(task)
        log.debug("cancel: registered %s", key)

    async def unregister(self, project_id: str, enrichment_id: str, task: asyncio.Task) -> None:
        """Drop this specific task from its (project, enrichment) bucket
        on completion. Other in-flight tasks for the same key keep
        their cancellability."""
        key = (project_id, enrichment_id)
        async with self._lock:
            bucket = self._tasks.get(key)
            if bucket is None:
                return
            bucket.discard(task)
            if not bucket:
                self._tasks.pop(key, None)

    async def cancel_enrichment(self, project_id: str, enrichment_id: str) -> bool:
        """Cancel ALL in-flight tasks for (project, enrichment). Returns
        True if at least one task was signalled."""
        key = (project_id, enrichment_id)
        async with self._lock:
            bucket = list(self._tasks.get(key) or ())
        signalled = False
        for task in bucket:
            if not task.done():
                task.cancel()
                signalled = True
        if signalled:
            log.info("cancel: signalled %d task(s) for %s", len(bucket), key)
        return signalled

    async def list_running(self, project_id: str) -> list[str]:
        """Return enrichment_ids with at least one active task for this project."""
        async with self._lock:
            out: list[str] = []
            for (pid, eid), bucket in self._tasks.items():
                if pid != project_id:
                    continue
                if any(not t.done() for t in bucket):
                    out.append(eid)
            return out


# Singleton — REST endpoint registers tasks; cancel endpoint looks them up.
REGISTRY = CancelRegistry()
