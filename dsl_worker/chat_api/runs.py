"""Run lifecycle for chat agent execution.

Decouples the agent loop from the HTTP request that triggered it. A
`ChatRun` row carries the state machine (queued → running → paused →
completed | failed | cancelled). Events are persisted to
`ChatRunEvent` so reconnecting clients can replay from a cursor.

Architecture:
- `start_run()`         creates the ChatRun row + schedules the
                        background asyncio.Task. Returns run_id.
- `_run_agent_task()`   is the background coroutine. Acquires the
                        per-project lock so queued runs serialize.
                        Calls into `streaming.run_agent_loop()` with
                        an `emit` callable that writes events.
- `tail_events()`       async-generates events for a subscriber: first
                        replays persisted events from the cursor,
                        then tails new ones via the in-process bus
                        until the run reaches a terminal status.
- `pause_run()`         flips status to pause_requested. The agent
                        loop polls status between rounds + between
                        tool calls and exits cleanly.
- `cancel_run()`        flips status to cancelled. Same poll behavior.
- `resume_run()`        creates a NEW run with content='Continue'
                        (Claude Code style — resume is just another
                        message). Returns the new run_id.
"""
from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from typing import Any, AsyncGenerator, Awaitable, Callable, Dict, List, Optional
from uuid import UUID

from sqlalchemy.orm import Session

from dsl_api.db import SessionLocal
from dsl_api.models import ChatMessage, ChatRun, ChatRunEvent, Project
from dsl_api.models.chat_run import (
    RUN_ACTIVE_STATUSES,
    RUN_STATUS_CANCELLED,
    RUN_STATUS_COMPLETED,
    RUN_STATUS_FAILED,
    RUN_STATUS_PAUSE_REQUESTED,
    RUN_STATUS_PAUSED,
    RUN_STATUS_QUEUED,
    RUN_STATUS_RUNNING,
    RUN_TERMINAL_STATUSES,
)

log = logging.getLogger(__name__)


# ---- In-process pubsub for live event fanout -----------------------------
# The DB is the source of truth — every event is persisted to
# ChatRunEvent before fanout. The bus exists so live subscribers don't
# have to poll the DB. On process restart, subscribers reconnect and
# replay from their last-seen cursor, bypassing the bus entirely.

class _RunBus:
    def __init__(self) -> None:
        self._subs: Dict[str, List[asyncio.Queue]] = defaultdict(list)
        # Cumulative assistant `token` content per run. Built up as
        # the agent streams; used to bootstrap mid-stream reconnects
        # without per-token DB writes. Reset on run cleanup.
        self._content: Dict[str, str] = defaultdict(str)

    def subscribe(self, run_id: UUID) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=1024)
        self._subs[str(run_id)].append(q)
        return q

    def subscribe_with_snapshot(self, run_id: UUID) -> tuple:
        """Atomic subscribe + snapshot. Single-threaded asyncio means
        no awaits between these two ops — a delta cannot fire between
        the snapshot read and the queue registration.

        Returns (queue, content_snapshot). The caller should yield the
        snapshot as a `text_checkpoint` event before draining the
        queue, so a reconnecting subscriber gets the in-progress
        round's text without waiting for the next round-boundary
        checkpoint.
        """
        q: asyncio.Queue = asyncio.Queue(maxsize=1024)
        self._subs[str(run_id)].append(q)
        snap = self._content.get(str(run_id), "")
        return q, snap

    def unsubscribe(self, run_id: UUID, q: asyncio.Queue) -> None:
        key = str(run_id)
        try:
            self._subs[key].remove(q)
        except ValueError:
            pass
        if not self._subs[key]:
            self._subs.pop(key, None)

    def publish(self, run_id: UUID, event: Dict[str, Any]) -> None:
        # Best-effort fanout. If a subscriber's queue is full the event
        # is dropped for that subscriber only — they'll see it on the
        # next DB replay (subscribers track their own cursor).
        for q in list(self._subs.get(str(run_id), [])):
            try:
                q.put_nowait(event)
            except asyncio.QueueFull:
                log.warning("run %s subscriber queue full, dropping event", run_id)

    def publish_delta(self, run_id: UUID, kind: str, content: str) -> None:
        """Live-only delta — fanout to subscribers, no DB write.

        For `token` deltas, also append to the per-run cumulative
        content accumulator so reconnects can bootstrap mid-round.
        For `thinking` deltas, fanout only (thinking is per-round
        ephemeral display; reconnect mid-round won't show prior
        thinking, which is acceptable).

        Live deltas have no `seq` — `tail_events` lets them through
        the seq guard (seq is the cursor for persisted events only).
        """
        if not content:
            return
        if kind == "token":
            self._content[str(run_id)] += content
        ev = {"type": kind, "content": content}
        for q in list(self._subs.get(str(run_id), [])):
            try:
                q.put_nowait(ev)
            except asyncio.QueueFull:
                log.warning("run %s subscriber queue full, dropping delta", run_id)

    def cleanup_run(self, run_id: UUID) -> None:
        """Drop in-memory state for a run. Called when the run reaches
        a terminal status."""
        key = str(run_id)
        self._content.pop(key, None)
        # Subscribers' queues are cleaned up on unsubscribe in tail_events.


_BUS = _RunBus()


# ---- Per-project serialization lock --------------------------------------
# Auto-queue: while one run on a project is executing, others wait.
# In-process only (per worker process). On restart, queued/running
# rows are recovered by `recover_orphan_runs()` at startup.
_PROJECT_LOCKS: Dict[str, asyncio.Lock] = {}


def _project_lock(project_id: UUID) -> asyncio.Lock:
    key = str(project_id)
    lock = _PROJECT_LOCKS.get(key)
    if lock is None:
        lock = asyncio.Lock()
        _PROJECT_LOCKS[key] = lock
    return lock


# ---- Event emit (persist + fanout) ---------------------------------------
def emit_event(
    db: Session,
    run: ChatRun,
    event_type: str,
    payload: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Persist a ChatRunEvent row and fan out to live subscribers.
    Returns the full {type, seq, ...payload} dict callers may forward.

    Retries on `(run_id, seq)` unique violations: another writer (e.g.
    a second worker process that recovered this run as an orphan,
    uvicorn-reload spawn, etc.) may have advanced the seq counter
    while we were holding our local copy. On conflict we rollback,
    refresh `run.next_event_seq` from the DB, and retry with the new
    seq. Up to 5 attempts so a real bug surfaces rather than masking
    forever.
    """
    from sqlalchemy.exc import IntegrityError

    payload = dict(payload or {})
    last_err: Optional[Exception] = None
    for attempt in range(5):
        seq = run.next_event_seq
        run.next_event_seq = seq + 1
        db.add(ChatRunEvent(
            run_id=run.id,
            seq=seq,
            type=event_type,
            payload=payload,
        ))
        try:
            db.commit()
        except IntegrityError as e:
            last_err = e
            try:
                db.rollback()
            except Exception:
                log.exception("rollback after seq conflict failed")
            try:
                # Re-read seq from DB so the next attempt picks up the
                # advanced value committed by the other writer.
                db.refresh(run, attribute_names=["next_event_seq"])
            except Exception:
                log.exception("refresh after seq conflict failed")
            log.warning(
                "emit_event seq conflict for run %s (attempt %d, seq=%d) — retrying",
                run.id, attempt + 1, seq,
            )
            continue

        full = {"type": event_type, "seq": seq, **payload}
        _BUS.publish(run.id, full)
        return full

    # Out of retries — propagate the last error.
    log.error("emit_event giving up after 5 seq-conflict retries for run %s", run.id)
    raise last_err if last_err else RuntimeError("emit_event failed")


def update_run_phase(db: Session, run: ChatRun, phase: Optional[str]) -> None:
    """Cheap hint update — no event emitted; the FE polls or reads
    when it cares (e.g. on reconnect) via the run row itself."""
    run.current_phase = phase
    db.commit()


# ---- Live deltas + checkpoint persistence --------------------------------
# OpenAI's response stream fires `token` and `thinking` deltas at 30-50/s.
# Per-token DB commit round-trips to Supabase (~30-50ms each via the
# pooler) throttled the agent loop and made streaming feel sluggish.
#
# Architecture: live deltas go through the in-process bus only — no DB.
# At round boundaries (when an OpenAI stream completes) we persist a
# `text_checkpoint` event carrying the FULL accumulated assistant
# content. ~5-10 commits per turn instead of thousands. Reconnects
# replay checkpoints from DB then bootstrap from the bus's per-run
# in-memory accumulator (atomic subscribe + snapshot).


def publish_token_delta(run_id: UUID, content: str) -> None:
    """Emit a live token delta — bus only, no DB. Updates the per-run
    content accumulator so reconnects can bootstrap mid-round."""
    _BUS.publish_delta(run_id, "token", content)


def publish_thinking_delta(run_id: UUID, content: str) -> None:
    """Emit a live thinking delta — bus only, no DB. Thinking is
    per-round ephemeral; not included in the content accumulator."""
    _BUS.publish_delta(run_id, "thinking", content)


def emit_text_checkpoint(db: Session, run: ChatRun) -> Dict[str, Any]:
    """Persist a `text_checkpoint` event with the full accumulated
    assistant content for this run. Called at round boundaries — gives
    reconnects a durable snapshot of the assistant text up to that
    point. The FE handles `text_checkpoint` by REPLACING (not appending)
    the message content; idempotent for live subscribers (their content
    already matches) and rebuilding for reconnects."""
    full = _BUS._content.get(str(run.id), "")
    return emit_event(db, run, "text_checkpoint", {"full_content": full})


# ---- Pause / cancel polling ----------------------------------------------
def check_should_stop(db: Session, run: ChatRun) -> Optional[str]:
    """Re-read run.status. Returns 'pause' or 'cancel' if the agent
    loop should exit cleanly, else None. Called between tool rounds
    and inside long-running tools (rows_fill cell loop).

    Cheap: single indexed PK lookup. Caller is expected to call this
    every few seconds at most, not in tight loops.
    """
    db.refresh(run, attribute_names=["status"])
    if run.status == RUN_STATUS_PAUSE_REQUESTED:
        return "pause"
    if run.status == RUN_STATUS_CANCELLED:
        return "cancel"
    return None


# ---- Start a run ---------------------------------------------------------
async def start_run(
    project_id: UUID,
    user_id: UUID,
    user_content: str,
    effort: Optional[str] = None,
) -> ChatRun:
    """Create a ChatRun and the user ChatMessage; schedule the
    background task. Returns the ChatRun (detached — caller should
    not assume the row is still attached to its session)."""
    db = SessionLocal()
    try:
        project = (
            db.query(Project)
            .filter(
                Project.id == project_id,
                Project.user_id == user_id,
                Project.deleted_at.is_(None),
            )
            .first()
        )
        if project is None:
            raise ValueError("Project not found")
        if project.mode != "chat":
            raise ValueError("Project is not in chat mode")

        run = ChatRun(
            project_id=project_id,
            status=RUN_STATUS_QUEUED,
        )
        db.add(run)
        db.flush()

        user_msg = ChatMessage(
            project_id=project_id,
            role="user",
            content=user_content,
            run_id=run.id,
        )
        db.add(user_msg)
        db.flush()

        run.triggering_message_id = user_msg.id
        db.commit()
        db.refresh(run)
        run_id = run.id
    finally:
        db.close()

    asyncio.create_task(
        _run_agent_task(run_id, user_id, user_content, effort),
        name=f"chat-run-{run_id}",
    )

    # Re-fetch detached so caller can read fields without triggering
    # the closed session's lazy-load.
    db = SessionLocal()
    try:
        run = db.query(ChatRun).filter(ChatRun.id == run_id).first()
        db.expunge(run)  # type: ignore[arg-type]
        return run  # type: ignore[return-value]
    finally:
        db.close()


# ---- Background task (per-project lock + agent dispatch) -----------------
async def _run_agent_task(
    run_id: UUID,
    user_id: UUID,
    user_content: str,
    effort: Optional[str],
) -> None:
    """Background coroutine for a single ChatRun. Acquires the
    per-project lock so multiple runs on the same project serialize.

    Dispatches into `streaming.run_agent_loop()` which does the actual
    OpenAI / tool work. The lock is released after the run reaches a
    terminal status."""
    # Late import to avoid a circular dep with streaming.py.
    from dsl_worker.chat_api.streaming import run_agent_loop

    # Resolve project_id from the run row up-front so we know which
    # lock to acquire even if the run was cancelled before we started.
    db = SessionLocal()
    try:
        run = db.query(ChatRun).filter(ChatRun.id == run_id).first()
        if run is None:
            log.warning("run %s vanished before task start", run_id)
            return
        project_id = run.project_id
        # If the run was cancelled in the brief window between
        # start_run() commit and task scheduling, just exit.
        if run.status == RUN_STATUS_CANCELLED:
            return
    finally:
        db.close()

    async with _project_lock(project_id):
        # Re-check status after acquiring the lock — could have been
        # cancelled while waiting in the queue.
        db = SessionLocal()
        try:
            run = db.query(ChatRun).filter(ChatRun.id == run_id).first()
            if run is None or run.status in RUN_TERMINAL_STATUSES:
                return
            run.status = RUN_STATUS_RUNNING
            from sqlalchemy.sql import func
            run.started_at = func.now()
            db.commit()
        finally:
            db.close()

        # Now run the agent loop. It opens its own session and emits
        # events via the bus + ChatRunEvent. Errors are caught here
        # and persisted as a final "failed" status.
        try:
            await run_agent_loop(
                run_id=run_id,
                user_id=user_id,
                user_content=user_content,
                effort=effort,
            )
        except Exception as e:
            log.exception("run %s crashed", run_id)
            _mark_run_failed(run_id, str(e)[:500])


def _mark_run_failed(run_id: UUID, error: str) -> None:
    db = SessionLocal()
    try:
        run = db.query(ChatRun).filter(ChatRun.id == run_id).first()
        if run is None or run.status in RUN_TERMINAL_STATUSES:
            return
        from sqlalchemy.sql import func
        run.status = RUN_STATUS_FAILED
        # run.error keeps the raw text for offline diagnosis; FE-facing
        # events get a generic message so SQL/frame leaks don't ship
        # to the client.
        run.error = error
        run.completed_at = func.now()
        user_safe_msg = "Something went wrong on our end. Please try again."
        emit_event(db, run, "error", {"message": user_safe_msg})
        emit_event(db, run, "done", {"stopped": True, "error": user_safe_msg})
    finally:
        db.close()
    _BUS.cleanup_run(run_id)


# ---- Public terminal-marking helpers (called from streaming) -------------
def mark_run_completed(db: Session, run: ChatRun, payload: Dict[str, Any]) -> None:
    from sqlalchemy.sql import func
    run.status = RUN_STATUS_COMPLETED
    run.completed_at = func.now()
    run.current_phase = None
    emit_event(db, run, "done", payload)
    _BUS.cleanup_run(run.id)


def mark_run_paused(db: Session, run: ChatRun, payload: Dict[str, Any]) -> None:
    from sqlalchemy.sql import func
    run.status = RUN_STATUS_PAUSED
    run.paused_at = func.now()
    run.current_phase = None
    emit_event(db, run, "paused", payload)
    _BUS.cleanup_run(run.id)


def mark_run_cancelled(db: Session, run: ChatRun, payload: Dict[str, Any]) -> None:
    from sqlalchemy.sql import func
    run.status = RUN_STATUS_CANCELLED
    run.completed_at = func.now()
    run.current_phase = None
    emit_event(db, run, "cancelled", payload)
    _BUS.cleanup_run(run.id)


# ---- Subscriber-facing API: tail events ----------------------------------
async def tail_events(
    run_id: UUID,
    cursor: int = 0,
    is_disconnected: Optional[Callable[[], Awaitable[bool]]] = None,
) -> AsyncGenerator[Dict[str, Any], None]:
    """Async-generate events for a subscriber.

    1. Replay persisted events from `cursor` (exclusive of seq < cursor)
       in batches.
    2. Subscribe to the bus for live tail.
    3. Stop when the run reaches a terminal status AND the DB has no
       events newer than what's been emitted (drain).

    `is_disconnected` is an optional callback (e.g. `request.is_disconnected`)
    checked between yields. If it returns True the generator exits early
    — the run keeps running in the background; client may reconnect later.
    """
    last_seq = cursor - 1  # next event to emit must have seq > last_seq

    # ---- Step 1: replay from cursor -----------------------------------
    while True:
        db = SessionLocal()
        try:
            rows = (
                db.query(ChatRunEvent)
                .filter(
                    ChatRunEvent.run_id == run_id,
                    ChatRunEvent.seq > last_seq,
                )
                .order_by(ChatRunEvent.seq.asc())
                .limit(500)
                .all()
            )
        finally:
            db.close()
        if not rows:
            break
        for row in rows:
            event = {"type": row.type, "seq": row.seq, **(row.payload or {})}
            last_seq = row.seq
            yield event
            if is_disconnected is not None and await is_disconnected():
                return

    # ---- Step 2: subscribe + tail -------------------------------------
    # Check terminal status BEFORE subscribing so we don't miss the
    # boundary case where the run completed during replay.
    db = SessionLocal()
    try:
        run = db.query(ChatRun).filter(ChatRun.id == run_id).first()
        if run is None:
            return
        is_terminal_now = run.status in RUN_TERMINAL_STATUSES or run.status == RUN_STATUS_PAUSED
    finally:
        db.close()

    if is_terminal_now:
        # Drain any events that landed after our last batch.
        db = SessionLocal()
        try:
            rows = (
                db.query(ChatRunEvent)
                .filter(
                    ChatRunEvent.run_id == run_id,
                    ChatRunEvent.seq > last_seq,
                )
                .order_by(ChatRunEvent.seq.asc())
                .all()
            )
        finally:
            db.close()
        for row in rows:
            yield {"type": row.type, "seq": row.seq, **(row.payload or {})}
        return

    # Atomic subscribe + snapshot. The snapshot is the cumulative
    # token content the bus has seen this run; yielding it as a
    # `text_checkpoint` lets a reconnecting subscriber bootstrap the
    # in-progress round's text without waiting for the next round
    # boundary. Idempotent for live subscribers (their content already
    # equals the snapshot — set is a no-op).
    q, content_snapshot = _BUS.subscribe_with_snapshot(run_id)
    if content_snapshot:
        yield {"type": "text_checkpoint", "full_content": content_snapshot}
    try:
        while True:
            if is_disconnected is not None and await is_disconnected():
                return
            try:
                event = await asyncio.wait_for(q.get(), timeout=15.0)
            except asyncio.TimeoutError:
                # Heartbeat / re-check terminal status. The bus does
                # not signal end-of-stream, so we poll the run row.
                db = SessionLocal()
                try:
                    run = db.query(ChatRun).filter(ChatRun.id == run_id).first()
                    if run is None:
                        return
                    if run.status in RUN_TERMINAL_STATUSES or run.status == RUN_STATUS_PAUSED:
                        # Drain any events the bus dropped before exiting.
                        rows = (
                            db.query(ChatRunEvent)
                            .filter(
                                ChatRunEvent.run_id == run_id,
                                ChatRunEvent.seq > last_seq,
                            )
                            .order_by(ChatRunEvent.seq.asc())
                            .all()
                        )
                        for row in rows:
                            yield {"type": row.type, "seq": row.seq, **(row.payload or {})}
                        return
                finally:
                    db.close()
                # Emit a heartbeat so proxies don't drop the SSE.
                yield {"type": "heartbeat", "seq": last_seq}
                continue

            seq = event.get("seq")
            # Live deltas (token/thinking) have no seq — let them pass.
            # Persisted events have seq; dedupe against last_seq.
            if seq is not None:
                if seq <= last_seq:
                    continue
                last_seq = seq
            yield event
            if event.get("type") in ("done", "paused", "cancelled", "error"):
                return
    finally:
        _BUS.unsubscribe(run_id, q)


# ---- Pause / cancel / resume control surface -----------------------------
def request_pause(run_id: UUID) -> bool:
    """Flip status → pause_requested. Returns True if the request was
    accepted (run was in an active non-terminal state)."""
    db = SessionLocal()
    try:
        run = db.query(ChatRun).filter(ChatRun.id == run_id).first()
        if run is None:
            return False
        if run.status == RUN_STATUS_QUEUED:
            # Cancel a queued run outright — nothing's running yet.
            from sqlalchemy.sql import func
            run.status = RUN_STATUS_CANCELLED
            run.completed_at = func.now()
            emit_event(db, run, "cancelled", {"reason": "paused-before-start"})
            return True
        if run.status == RUN_STATUS_RUNNING:
            run.status = RUN_STATUS_PAUSE_REQUESTED
            db.commit()
            return True
        return False
    finally:
        db.close()


def request_cancel(run_id: UUID) -> bool:
    """Flip status → cancelled (terminal). The agent loop picks up the
    new status on its next poll and exits without saving."""
    db = SessionLocal()
    try:
        run = db.query(ChatRun).filter(ChatRun.id == run_id).first()
        if run is None or run.status in RUN_TERMINAL_STATUSES:
            return False
        if run.status in (RUN_STATUS_QUEUED, RUN_STATUS_PAUSED):
            from sqlalchemy.sql import func
            run.status = RUN_STATUS_CANCELLED
            run.completed_at = func.now()
            emit_event(db, run, "cancelled", {"reason": "user-cancel"})
            return True
        # Running / pause_requested → mark cancelled; the loop will
        # see it on next poll and exit cleanly.
        run.status = RUN_STATUS_CANCELLED
        db.commit()
        return True
    finally:
        db.close()


async def resume_run(
    project_id: UUID,
    user_id: UUID,
    paused_run_id: Optional[UUID] = None,
    content: str = "Continue.",
    effort: Optional[str] = None,
) -> ChatRun:
    """Resume = create a new run with the given content (default
    'Continue.'). Claude Code style: the model picks up where it left
    off because chat history (with the prior tool_log) is replayed
    into context every turn.

    If `paused_run_id` is given and is in `paused` state, mark it
    completed first so the active-run query stops returning it.
    """
    if paused_run_id is not None:
        db = SessionLocal()
        try:
            paused = db.query(ChatRun).filter(ChatRun.id == paused_run_id).first()
            if paused is not None and paused.status == RUN_STATUS_PAUSED:
                from sqlalchemy.sql import func
                paused.status = RUN_STATUS_COMPLETED
                paused.completed_at = func.now()
                paused.resumed_at = func.now()
                db.commit()
        finally:
            db.close()

    return await start_run(project_id, user_id, content, effort=effort)


# ---- Active-run discovery (for FE reattach on mount) ---------------------
def get_active_run(db: Session, project_id: UUID) -> Optional[ChatRun]:
    """Return the most recent non-terminal run for a project, if any.
    The FE hits this on mount to know whether to reattach to a stream."""
    return (
        db.query(ChatRun)
        .filter(
            ChatRun.project_id == project_id,
            ChatRun.status.in_(list(RUN_ACTIVE_STATUSES) + [RUN_STATUS_PAUSED]),
        )
        .order_by(ChatRun.created_at.desc())
        .first()
    )


# ---- TTL cleanup (called periodically) -----------------------------------
EVENTS_TTL_DAYS = 30


def purge_old_events() -> int:
    """Delete ChatRunEvent rows older than EVENTS_TTL_DAYS. Returns the
    number of rows removed. Called on a schedule from a long-running
    asyncio task started at app startup."""
    from sqlalchemy import text
    db = SessionLocal()
    try:
        result = db.execute(
            text(
                "DELETE FROM chat_run_events "
                "WHERE created_at < now() - make_interval(days => :days)"
            ),
            {"days": EVENTS_TTL_DAYS},
        )
        db.commit()
        return int(result.rowcount or 0)
    finally:
        db.close()


async def run_ttl_cleanup_loop(interval_seconds: int = 3600) -> None:
    """Background task: purge old events on an interval. Logs each pass.
    Started by app startup; runs until the process exits."""
    while True:
        try:
            n = await asyncio.to_thread(purge_old_events)
            if n:
                log.info("chat_run_events TTL: purged %d row(s)", n)
        except Exception:
            log.exception("chat_run_events TTL pass failed")
        await asyncio.sleep(interval_seconds)


# ---- Crash recovery (called at app startup) ------------------------------
def recover_orphan_runs() -> int:
    """Mark runs that look truly stale as failed.

    Heartbeat-based: a run is only considered orphaned if it (a)
    started more than 2 minutes ago AND (b) has not emitted any
    ChatRunEvent in the last 60 seconds. Otherwise a freshly-started
    second worker would murder another worker's live run via this
    path, causing `(run_id, seq)` collisions on emit_event when both
    write to the same event log.

    Returns count fixed. Safe to call on a polling schedule, not just
    at startup.
    """
    from sqlalchemy.sql import func
    from sqlalchemy import select

    db = SessionLocal()
    try:
        # Candidates: any non-terminal run started > 2 minutes ago.
        candidates = (
            db.query(ChatRun)
            .filter(
                ChatRun.status.in_([
                    RUN_STATUS_QUEUED,
                    RUN_STATUS_RUNNING,
                    RUN_STATUS_PAUSE_REQUESTED,
                ]),
                ChatRun.started_at < func.now() - func.make_interval(0, 0, 0, 0, 0, 2, 0),
            )
            .all()
        )

        fixed = 0
        for run in candidates:
            # Per-run heartbeat: latest ChatRunEvent timestamp. If
            # something was emitted in the last 60s, another process
            # is alive on this run — leave it alone.
            last_evt_at = (
                db.query(func.max(ChatRunEvent.created_at))
                .filter(ChatRunEvent.run_id == run.id)
                .scalar()
            )
            if last_evt_at is not None:
                # Compare in DB to avoid timezone fudging in Python.
                stale = db.execute(
                    select(func.now() - last_evt_at > func.make_interval(0, 0, 0, 0, 0, 1, 0))
                ).scalar()
                if not stale:
                    continue
            elif run.started_at is None:
                continue

            run.status = RUN_STATUS_FAILED
            run.error = "Worker process restarted; run was orphaned (no heartbeat)."
            run.completed_at = func.now()
            try:
                emit_event(db, run, "error", {"message": run.error})
                emit_event(db, run, "done", {"stopped": True, "error": run.error})
            except Exception:
                log.exception("orphan recovery: emit failed for run %s", run.id)
                try:
                    db.rollback()
                except Exception:
                    pass
                continue
            fixed += 1
        return fixed
    finally:
        db.close()
