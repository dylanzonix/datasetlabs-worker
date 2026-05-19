"""Glue between chat_v2 row-write sites and the email verification module.

The chat_api/email_verify module is path-agnostic — it takes an async
progress_cb and calls it with `{"type": <event>, ...}` dicts. chat_v2's
SSE layer wants `legacy_runs.emit_event(db, run, type, payload)` on a
DB session that ISN'T shared with the orchestrator's `ctx.db` (verify
tasks run concurrently; a shared session would race on the tags JSON
write).

This module bridges the two. `schedule_for_row` builds the adapter and
hands the verify module the contract it expects.

Email-only: the URL verification path was removed in a9bd552 after it
starved the event loop. Email verification is safe to keep:
  • Scrubby's client serializes submits at 1/sec via an asyncio lock,
    so 99 concurrent verifies pace themselves at the network gate.
  • Cell-agent enrichment fills ~1 email column per row, not ~5 URLs,
    so the event volume is ~5× lower for the worst case.
  • Provider emails (Apollo / FullEnrich) skip Scrubby entirely.

Tasks are pinned in `_BACKGROUND_TASKS` so fire-and-forget callers can
drop the returned list without risk of GC cancelling mid-await.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Awaitable, Callable, Dict, List, Optional, Set

from dsl_api.db import SessionLocal
from dsl_api.models import ChatRun

from dsl_worker.chat_api import email_verify
from dsl_worker.chat_api import runs as legacy_runs


log = logging.getLogger(__name__)


_BACKGROUND_TASKS: set = set()


def _register_background(task: asyncio.Task) -> None:
    _BACKGROUND_TASKS.add(task)
    task.add_done_callback(_BACKGROUND_TASKS.discard)


def _email_columns_from_defs(columns: List[Dict[str, Any]]) -> Set[str]:
    out: Set[str] = set()
    for c in columns or []:
        if not isinstance(c, dict):
            continue
        name = c.get("name")
        if not isinstance(name, str) or not name:
            continue
        if email_verify.is_email_column(c, name):
            out.add(name)
    return out


def _make_event_emitter(
    run_id: Optional[Any],
) -> Optional[Callable[[Dict[str, Any]], Awaitable[None]]]:
    if not run_id:
        return None

    async def emit(event: Dict[str, Any]) -> None:
        try:
            event_type = event.get("type")
            if not isinstance(event_type, str) or not event_type:
                return
            payload = {k: v for k, v in event.items() if k != "type"}
            db = SessionLocal()
            try:
                run_obj = db.query(ChatRun).filter(ChatRun.id == run_id).first()
                if run_obj is None:
                    return
                legacy_runs.emit_event(db, run_obj, event_type, payload)
                db.commit()
            finally:
                db.close()
        except Exception:
            log.debug("email_verify_hook emit failed (suppressed)", exc_info=True)

    return emit


def schedule_for_row(
    *,
    run_id: Optional[Any],
    sample_id: str,
    written_values: Dict[str, Any],
    columns: List[Dict[str, Any]],
) -> List[asyncio.Task]:
    """Fire Scrubby verification for a row's just-written email values.

    Returns the spawned tasks (also pinned in `_BACKGROUND_TASKS`).
    Caller can drop the list — pinning keeps tasks alive past the
    function that scheduled them.
    """
    if not written_values:
        return []
    email_cols = _email_columns_from_defs(columns)
    if not email_cols & set(written_values.keys()):
        return []
    progress_cb = _make_event_emitter(run_id)
    tasks = email_verify.schedule_verifications(
        sample_id=sample_id,
        written_values=written_values,
        email_columns=email_cols,
        provider_emails=set(),
        progress_cb=progress_cb,
    )
    for t in tasks:
        _register_background(t)
    return tasks
