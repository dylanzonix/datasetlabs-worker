"""Glue between chat row-write sites and the URL verification module.

Mirrors email_verify_hook: same _BACKGROUND_TASKS pinning pattern,
same fresh-SessionLocal-per-event emitter, same one-task-per-batch
shape. The verify module itself is path-agnostic and takes an async
progress_cb; this module bridges it to chat's SSE layer.

Why a different module from email_verify_hook: URL verification has
fundamentally different cost shape (firecrawl is paid per page +
slower) so we sample instead of verifying every value. The hook
collects all URLs across the batch, groups by column, and hands the
verify module ONE batch per column. Detection is value-based
(`^https?://`) because most user-defined columns store URLs under
generic `text` type.

Tasks are pinned in `_BACKGROUND_TASKS` so fire-and-forget callers
can drop the returned reference without risk of GC cancelling
mid-await.
"""

from __future__ import annotations

import asyncio
import logging
import re
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple

from dsl_api.db import SessionLocal
from dsl_api.models import ChatRun

from dsl_worker.chat import run_state
from dsl_worker.infra import url_verify


log = logging.getLogger(__name__)


_BACKGROUND_TASKS: set = set()


def _register_background(task: asyncio.Task) -> None:
    _BACKGROUND_TASKS.add(task)
    task.add_done_callback(_BACKGROUND_TASKS.discard)


_VALUE_URL_RE = re.compile(r"^https?://", re.IGNORECASE)


def _url_columns_from_values(rows: List[Tuple[str, Dict[str, Any]]]) -> List[str]:
    """URL columns are detected by VALUE across the batch.

    Apify / web_harvest / many user-defined columns default to
    `type: "text"` even when they store URLs, so the column-type
    detector was missing real URL fields in the old code. A column is
    a URL column if ANY row in the batch has a URL-shaped value in it.
    Stable-ordered for deterministic batches in logs/tests.
    """
    seen: Dict[str, None] = {}  # preserve first-seen order
    for _sid, written in rows:
        if not isinstance(written, dict):
            continue
        for k, v in written.items():
            if not isinstance(k, str) or k in seen:
                continue
            if isinstance(v, str) and _VALUE_URL_RE.match(v):
                seen[k] = None
    return list(seen.keys())


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
                run_state.emit_event(db, run_obj, event_type, payload)
                db.commit()
            finally:
                db.close()
        except Exception:
            log.debug("url_verify_hook emit failed (suppressed)", exc_info=True)

    return emit


def schedule_bulk_for_rows(
    *,
    run_id: Optional[Any],
    rows: List[Tuple[str, Dict[str, Any]]],
) -> Optional[asyncio.Task]:
    """Fire ONE URL-verify task per detected URL column across the batch.

    `rows`: list of (sample_id, written_values) tuples — same shape
    email_verify_hook.schedule_bulk_for_rows accepts.

    Returns the spawned task (pinned in `_BACKGROUND_TASKS`), or None
    if firecrawl isn't configured or there are no URL cells in the
    batch.
    """
    if not rows:
        return None
    url_cols = _url_columns_from_values(rows)
    if not url_cols:
        return None

    # Group: rows_by_column[col] = [(sample_id, url, row_dict), ...]
    rows_by_column: Dict[str, List[Tuple[str, str, Dict[str, Any]]]] = {}
    for sample_id, written in rows:
        if not isinstance(written, dict):
            continue
        for col in url_cols:
            val = written.get(col)
            if not isinstance(val, str) or not _VALUE_URL_RE.match(val):
                continue
            rows_by_column.setdefault(col, []).append(
                (str(sample_id), val, written)
            )
    if not rows_by_column:
        return None

    progress_cb = _make_event_emitter(run_id)
    log.info(
        "url_verify: scheduling bulk verify — %d column(s) across %d row(s)",
        len(rows_by_column), len(rows),
    )
    task = asyncio.create_task(
        url_verify.verify_batch(
            rows_by_column=rows_by_column,
            url_columns=url_cols,
            progress_cb=progress_cb,
        )
    )
    _register_background(task)
    return task
