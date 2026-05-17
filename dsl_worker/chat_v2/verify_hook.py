"""Glue between chat_v2 row-write sites and the verify modules.

The verify modules themselves (email_verify, url_verify) are agnostic —
they take an async progress_cb and call it with `{"type": <event>, ...}`
dicts. chat_v2's SSE layer wants `legacy_runs.emit_event(db, run, type,
payload)` instead, and the events must be written from a session that
isn't shared with the orchestrator's `ctx.db` (the verify tasks run
concurrently and a shared session would race).

This module bridges the two: `schedule_for_row` constructs an async
adapter that opens a fresh SessionLocal per event, loads the ChatRun,
and writes the event row. The verify modules see the same async
progress_cb contract they expect.
"""

from __future__ import annotations

import asyncio
import logging
import re
from typing import Any, Awaitable, Callable, Dict, List, Optional, Set, Tuple

from dsl_api.db import SessionLocal
from dsl_api.models import ChatRun

from dsl_worker.chat_api import email_verify
from dsl_worker.chat_api import runs as legacy_runs
from dsl_worker.infra import url_verify


log = logging.getLogger(__name__)


# Strong-reference registry for fire-and-forget verify tasks. asyncio
# only weak-refs tasks via its _all_tasks set, so a background task
# could be GC'd mid-await if nothing else holds it. We keep each task
# here and self-remove on completion.
_BACKGROUND_TASKS: set = set()


def _register_background(task: asyncio.Task) -> None:
    _BACKGROUND_TASKS.add(task)
    task.add_done_callback(_BACKGROUND_TASKS.discard)


def _email_columns_from_defs(columns: List[Dict[str, Any]]) -> Set[str]:
    """Email columns must be classified at the column level because we
    can't sniff "this string is an email" from value alone with high
    confidence (false positives on free-text fields would burn Scrubby
    credits). The column-def detector covers explicit type markers AND
    a name match for common patterns."""
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


_VALUE_URL_RE = re.compile(r"^https?://", re.IGNORECASE)


def _url_columns_from_values(written_values: Dict[str, Any]) -> Set[str]:
    """URL columns are detected by VALUE, not column type. Apify /
    web_harvest / many user-defined columns default to `type: "text"`
    even when they store URLs, so the column-type detector was missing
    a lot of real URL fields. Value-based detection has zero false
    negatives at the cost of occasionally fetching a URL that happens
    to live in a non-URL column (e.g. a "description" cell containing
    a link) — that's a few cents in the worst case and still tells the
    user whether the page resolves."""
    out: Set[str] = set()
    for k, v in (written_values or {}).items():
        if not isinstance(k, str):
            continue
        if isinstance(v, str) and _VALUE_URL_RE.match(v):
            out.add(k)
    return out


def _make_event_emitter(
    run_id: Optional[Any],
) -> Optional[Callable[[Dict[str, Any]], Awaitable[None]]]:
    """Build the async progress_cb the verify modules expect."""
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
                run_obj = (
                    db.query(ChatRun)
                    .filter(ChatRun.id == run_id)
                    .first()
                )
                if run_obj is None:
                    return
                # emit_event handles its own (run_id, seq) IntegrityError
                # retry loop, so concurrent verify tasks don't clobber
                # each other's seq.
                legacy_runs.emit_event(db, run_obj, event_type, payload)
                db.commit()
            finally:
                db.close()
        except Exception:
            log.debug("verify_hook emit failed (suppressed)", exc_info=True)

    return emit


def schedule_for_row(
    *,
    run_id: Optional[Any],
    sample_id: str,
    written_values: Dict[str, Any],
    columns: List[Dict[str, Any]],
    row_snapshot: Dict[str, Any],
) -> List[asyncio.Task]:
    """Fire email + URL verifications for a row's just-written values.

    Caller awaits the returned tasks before its tool returns so SSE
    events (which the FE consumes) get flushed within the same tool
    invocation. Empty list when no qualifying columns were written.
    """
    if not written_values:
        return []
    email_cols = _email_columns_from_defs(columns)
    url_cols = _url_columns_from_values(written_values)
    write_keys = set(written_values.keys())
    email_hits = email_cols & write_keys
    url_hits = url_cols & write_keys
    if not (email_hits or url_hits):
        # Loud-but-once log: if a row had values but nothing tripped
        # either detector, log enough to diagnose (column names + first
        # 60 chars of each value). Was the silent-skip the user noticed.
        if log.isEnabledFor(logging.DEBUG):
            sample = {
                k: (v[:60] if isinstance(v, str) else type(v).__name__)
                for k, v in list(written_values.items())[:6]
            }
            log.debug(
                "verify_hook: no email/url columns detected for sample %s — values=%s",
                sample_id, sample,
            )
        return []
    log.info(
        "verify_hook: sample %s — email cols=%s url cols=%s",
        sample_id, sorted(email_hits), sorted(url_hits),
    )
    progress_cb = _make_event_emitter(run_id)
    tasks: List[asyncio.Task] = []
    if email_hits:
        tasks.extend(
            email_verify.schedule_verifications(
                sample_id=sample_id,
                written_values=written_values,
                email_columns=email_cols,
                provider_emails=set(),
                progress_cb=progress_cb,
            )
        )
    if url_hits:
        tasks.extend(
            url_verify.schedule_verifications(
                sample_id=sample_id,
                written_values=written_values,
                url_columns=url_cols,
                row_snapshot=row_snapshot,
                progress_cb=progress_cb,
            )
        )
    # Pin every task so fire-and-forget callers can drop the returned
    # list without risk of GC cancelling mid-await.
    for t in tasks:
        _register_background(t)
    return tasks
