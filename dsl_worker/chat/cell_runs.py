"""In-process registry of cells currently being filled by an enrichment.

Bridge for the FE's per-cell pending state across page refreshes:

- handleRunCells fires N parallel jobs over the REST run endpoint.
  Each job runs in an asyncio task that survives client disconnect, so
  enrichment continues to run even if the user refreshes mid-fill.
- But pendingCells (the FE map driving the per-cell spinner) is React
  state and resets on refresh. Without a way to ask the server "what
  cells are you currently working on?", the user sees an empty table
  with column-header spinners but no per-cell affordance.

This registry tracks (project_id) → list of (enrichment_id, sample_id,
columns) currently being processed. The REST endpoint /cells/running
surfaces it so the FE can rebuild pendingCells on mount and refresh
the local view on a short poll while runs are active.

In-process only. Worker restart drops the registry; the rows the cell
agent was filling will still receive their values via the DB write
inside the cell loop (the task continues across the registry drop —
it lives on the event loop, not the registry), but the FE loses
visibility into them until the next refresh shows the filled values.
Persistent durability (resume after worker crash) is a bigger
architectural piece — chat's ChatRun + ChatRunEvent model is the
analog there; enrichment isn't on that model yet.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, List, Set, Tuple


log = logging.getLogger(__name__)


# Entry key: (enrichment_id, sample_id, frozenset(columns)). Sample-only
# would collide when two enrichments fill different columns on the same
# sample (the common "stress test" the user hit).
_EntryKey = Tuple[str, str, frozenset]


class RunningCellsRegistry:
    """Project-scoped registry of cells currently being processed by an
    enrichment run. Add when the cell agent picks up a row; remove when
    the row's write commits (or errors)."""

    def __init__(self) -> None:
        self._entries: Dict[str, Set[_EntryKey]] = {}
        self._lock = asyncio.Lock()

    async def add(
        self,
        project_id: str,
        enrichment_id: str,
        sample_id: str,
        columns: List[str],
    ) -> None:
        key: _EntryKey = (enrichment_id, sample_id, frozenset(columns or []))
        async with self._lock:
            self._entries.setdefault(project_id, set()).add(key)

    async def remove(
        self,
        project_id: str,
        enrichment_id: str,
        sample_id: str,
        columns: List[str],
    ) -> None:
        key: _EntryKey = (enrichment_id, sample_id, frozenset(columns or []))
        async with self._lock:
            bucket = self._entries.get(project_id)
            if bucket is None:
                return
            bucket.discard(key)
            if not bucket:
                self._entries.pop(project_id, None)

    async def list_for_project(self, project_id: str) -> List[Dict[str, Any]]:
        async with self._lock:
            bucket = self._entries.get(project_id)
            if not bucket:
                return []
            return [
                {
                    "enrichment_id": eid,
                    "sample_id": sid,
                    "columns": sorted(cols),
                }
                for (eid, sid, cols) in bucket
            ]


REGISTRY = RunningCellsRegistry()
