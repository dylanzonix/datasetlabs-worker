"""In-process registry for `plan_options` tool calls.

Pattern (mirrors approvals.py — same Future-based blocking shape, but
the result is the chosen option key, not a yes/no bool):

  1. Agent calls `plan_options(question, options)` when there's a
     genuine choice the user needs to make before the agent can
     continue (e.g. "you said 'GSA auctions' — that could be
     Treasury / IRS / US Marshals; pick one").
  2. The handler registers a pending pick, emits a
     ``plan_options_required`` SSE event, and awaits the Future.
  3. FE shows a small button-row card. User clicks one of the
     options.
  4. POST /plan_option_picks/{id}/respond resolves the Future with
     the chosen key.
  5. Handler returns ``{"chosen": "<key>"}`` to the agent, which
     proceeds with that choice.

Storage is in-process — picks do NOT survive a worker restart. If
the user reloads mid-pick the chat run will hang until the staleness
sweeper kills it, same as approvals.py. Acceptable for v1.

Why separate from approvals.py: approvals carry bool, picks carry
str. Tagging the registry by result-type and forking the resolve
endpoint is cleaner than overloading PendingApproval.future.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


log = logging.getLogger(__name__)


@dataclass
class PendingOptionPick:
    """One in-flight plan-options call: the Future the handler is
    awaiting + metadata captured at request time so the FE card can
    render the question + buttons."""
    id: str
    project_id: str
    question: str
    options: List[Dict[str, Any]]
    future: asyncio.Future = field(default_factory=asyncio.Future)


class OptionPickRegistry:
    """Single per-process registry of pending option picks. Lock
    serializes access to the dict so concurrent requests / resolves
    don't race on membership."""

    def __init__(self) -> None:
        self._pending: Dict[str, PendingOptionPick] = {}
        self._lock = asyncio.Lock()

    async def request(
        self,
        project_id: str,
        question: str,
        options: List[Dict[str, Any]],
    ) -> PendingOptionPick:
        loop = asyncio.get_running_loop()
        pending = PendingOptionPick(
            id=str(uuid.uuid4()),
            project_id=project_id,
            question=question,
            options=options,
            future=loop.create_future(),
        )
        async with self._lock:
            self._pending[pending.id] = pending
        log.info("plan_options requested: %s n=%d", pending.id, len(options))
        return pending

    async def resolve(self, pick_id: str, chosen: str) -> bool:
        """Resolve a pending pick with the chosen option key. Returns
        True if the pick was found, False if not. Idempotent."""
        async with self._lock:
            pending = self._pending.get(pick_id)
        if pending is None:
            return False
        if pending.future.done():
            log.debug("plan_options %s already resolved; skipping", pick_id)
            return True
        pending.future.set_result(chosen)
        log.info("plan_options resolved: %s chosen=%s", pick_id, chosen)
        async with self._lock:
            self._pending.pop(pick_id, None)
        return True

    async def peek(self, pick_id: str) -> Optional[PendingOptionPick]:
        async with self._lock:
            return self._pending.get(pick_id)

    async def list_for_project(self, project_id: str) -> List[Dict[str, Any]]:
        async with self._lock:
            return [
                {
                    "id": p.id,
                    "question": p.question,
                    "options": p.options,
                }
                for p in self._pending.values()
                if p.project_id == project_id
            ]

    async def cleanup_chat_run(self, project_id: str) -> None:
        """Cancel any pending picks for a project whose chat run is
        ending. Resolves with empty-string so the awaiting handler
        unwinds cleanly with a "no choice made" tool result."""
        async with self._lock:
            stale_ids = [p.id for p in self._pending.values() if p.project_id == project_id]
            for pid in stale_ids:
                p = self._pending.get(pid)
                if p and not p.future.done():
                    p.future.set_result("")
                self._pending.pop(pid, None)
        if stale_ids:
            log.info(
                "cleared %d pending plan_options on chat-run end (project=%s)",
                len(stale_ids), project_id,
            )


REGISTRY = OptionPickRegistry()
