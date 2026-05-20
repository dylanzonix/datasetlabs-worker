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
import os
from collections import defaultdict
from typing import Any, AsyncGenerator, Awaitable, Callable, Dict, List, Optional, Tuple
from uuid import UUID

import httpx
from sqlalchemy import text as sa_text
from sqlalchemy.orm import Session

from dsl_api.config import settings as dsl_api_settings
from dsl_api.db import SessionLocal
from dsl_api.models import Account, ChatMessage, ChatRun, ChatRunEvent, Project
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
        # Cumulative reasoning summary text per run-round. Persisted
        # at each round boundary as `thinking_checkpoint` so we can
        # diagnose why the model made decisions (e.g. why it bailed
        # apify_call_actor → web_search). Reset at round end.
        self._thinking: Dict[str, str] = defaultdict(str)

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

        Both `token` and `thinking` deltas accumulate per-run so
        round-boundary checkpoints can persist the full text. Tokens
        carry across rounds (assistant content keeps growing); thinking
        is reset each round end (per-round display).

        Live deltas have no `seq` — `tail_events` lets them through
        the seq guard (seq is the cursor for persisted events only).
        """
        if not content:
            return
        if kind == "token":
            self._content[str(run_id)] += content
        elif kind == "thinking":
            self._thinking[str(run_id)] += content
        ev = {"type": kind, "content": content}
        for q in list(self._subs.get(str(run_id), [])):
            try:
                q.put_nowait(ev)
            except asyncio.QueueFull:
                log.warning("run %s subscriber queue full, dropping delta", run_id)

    def reset_thinking(self, run_id: UUID) -> str:
        """Pop and return the run's accumulated thinking text. Called
        at round boundaries — the popped string is persisted as a
        thinking_checkpoint, then the buffer is empty for next round."""
        return self._thinking.pop(str(run_id), "")

    def cleanup_run(self, run_id: UUID) -> None:
        """Drop in-memory state for a run. Called when the run reaches
        a terminal status."""
        key = str(run_id)
        self._content.pop(key, None)
        self._thinking.pop(key, None)
        # Subscribers' queues are cleaned up on unsubscribe in tail_events.


_BUS = _RunBus()


# ---- Per-project serialization lock --------------------------------------
# Auto-queue: while one run on a project is executing, others wait.
# In-process only (per worker process). On restart, queued/running
# rows are recovered by `recover_orphan_runs()` at startup.
_PROJECT_LOCKS: Dict[str, asyncio.Lock] = {}

# ---- Global concurrency cap ----------------------------------------------
# Per-project lock prevents same-project pile-up; this prevents N projects
# from spawning N concurrent worker tasks and exhausting the DB pool /
# OpenAI rate limits. Excess runs queue at the asyncio task level.
# Sized for ~10-20 concurrent active users — safe for the Supabase pool
# (5 + 10 overflow) since each run holds at most 1 connection at a time.
_GLOBAL_RUN_CAP = int(os.getenv("CHAT_GLOBAL_RUN_CAP", "20"))
_GLOBAL_RUN_SEMAPHORE: Optional[asyncio.Semaphore] = None


def _global_run_semaphore() -> asyncio.Semaphore:
    """Lazy-init so the semaphore is bound to the running event loop."""
    global _GLOBAL_RUN_SEMAPHORE
    if _GLOBAL_RUN_SEMAPHORE is None:
        _GLOBAL_RUN_SEMAPHORE = asyncio.Semaphore(_GLOBAL_RUN_CAP)
    return _GLOBAL_RUN_SEMAPHORE


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
    from time import perf_counter_ns

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
            mono_ns=perf_counter_ns(),
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


def emit_events_batch(
    db: Session,
    run: ChatRun,
    events: List[Tuple[str, Dict[str, Any]]],
) -> List[Dict[str, Any]]:
    """Persist many events in a single commit, then fan out to subscribers.

    `events` is a list of `(event_type, payload)` tuples; they receive
    contiguous seq values and land in one transaction. Use when a hot
    loop would otherwise emit N events one at a time — each individual
    emit_event commits, so N events = N round-trips. Batching cuts that
    to one.

    Falls back to per-event emit on seq conflict to keep the retry
    behavior identical to emit_event.
    """
    from sqlalchemy.exc import IntegrityError
    from time import perf_counter_ns

    if not events:
        return []

    rows: List[ChatRunEvent] = []
    out: List[Dict[str, Any]] = []
    for event_type, raw_payload in events:
        payload = dict(raw_payload or {})
        seq = run.next_event_seq
        run.next_event_seq = seq + 1
        rows.append(ChatRunEvent(
            run_id=run.id, seq=seq, type=event_type,
            payload=payload, mono_ns=perf_counter_ns(),
        ))
        out.append({"type": event_type, "seq": seq, **payload})
    db.add_all(rows)
    try:
        db.commit()
    except IntegrityError:
        # Another writer raced us on next_event_seq. Roll back, refresh,
        # then fall back to per-event emit which already has its own
        # retry loop. We lose the batching speedup on this call but the
        # output stays correct.
        log.warning("emit_events_batch seq conflict for run %s — falling back to per-event emit", run.id)
        try:
            db.rollback()
        except Exception:
            log.exception("rollback after seq conflict failed")
        try:
            db.refresh(run, attribute_names=["next_event_seq"])
        except Exception:
            log.exception("refresh after seq conflict failed")
        return [emit_event(db, run, t, p) for t, p in events]

    for full in out:
        _BUS.publish(run.id, full)
    return out


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


def emit_thinking_checkpoint(
    db: Session, run: ChatRun, round_num: Optional[int] = None
) -> Optional[Dict[str, Any]]:
    """Persist the run's accumulated reasoning text for the just-completed
    round, then clear the buffer. The diagnose CLI reads these to show
    why the model made decisions; the FE doesn't need them (live tail
    already showed the deltas, end-of-round display is just the answer).

    Returns None if the buffer was empty (no reasoning emitted this
    round) so we don't pollute the event log with empty rows."""
    text = _BUS.reset_thinking(run.id)
    if not text:
        return None
    payload: Dict[str, Any] = {"content": text}
    if round_num is not None:
        payload["round"] = round_num
    return emit_event(db, run, "thinking_checkpoint", payload)


def replace_text_content(
    db: Session, run: ChatRun, content: str
) -> Dict[str, Any]:
    """Overwrite the live content accumulator with `content` and persist
    a text_checkpoint carrying it. Used when we need to scrub partial
    output already streamed to subscribers — e.g. when an OpenAI stream
    dies mid-sentence and we want to show a clean warning instead of the
    half-thought the model managed to emit before being cut off. The FE
    treats text_checkpoint as a replacement, so live subscribers see the
    new content overwrite whatever they had displayed."""
    _BUS._content[str(run.id)] = content
    return emit_event(db, run, "text_checkpoint", {"full_content": content})


def trim_token_content(run_id: UUID, suffix: str) -> bool:
    """Remove `suffix` from the end of the run's accumulated token
    content if present. Returns True if trimmed.

    Used when a mid-iteration text segment finalizes: the deltas were
    already streamed (and the accumulator grew), but the text actually
    belongs to a separate `text_segment` event for durability — not the
    final-text snapshot. Trimming keeps the accumulator equal to "what
    is in the FINAL text segment" so reconnect snapshots don't double-
    render the mid-text alongside its durable text_segment event."""
    key = str(run_id)
    current = _BUS._content.get(key, "")
    if suffix and current.endswith(suffix):
        _BUS._content[key] = current[: -len(suffix)]
        return True
    return False


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
    budget_cap_override_cents: Optional[int] = None,
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

        existing_message_count = (
            db.query(ChatMessage)
            .filter(ChatMessage.project_id == project_id)
            .count()
        )
        is_first_message = existing_message_count == 0

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

        first_message_email: Optional[str] = None
        if is_first_message:
            auth_row = db.execute(
                sa_text("SELECT email FROM auth.users WHERE id = :user_id"),
                {"user_id": str(user_id)},
            ).fetchone()
            if auth_row and auth_row[0]:
                first_message_email = auth_row[0]
            else:
                account = (
                    db.query(Account)
                    .filter(Account.user_id == str(user_id))
                    .first()
                )
                first_message_email = account.email if account else None
    finally:
        db.close()

    if is_first_message:
        asyncio.create_task(
            _post_first_chat_to_slack(
                project_id=str(project_id),
                email=first_message_email,
                message=user_content,
            ),
            name=f"slack-first-msg-{project_id}",
        )

    asyncio.create_task(
        _run_agent_task(
            run_id, user_id, user_content, effort, budget_cap_override_cents
        ),
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
    budget_cap_override_cents: Optional[int] = None,
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

    # Acquire BOTH the per-project lock (serializes runs on the same
    # project) AND the global cap (bounds total concurrent agent
    # tasks across the worker process). Order matters: global semaphore
    # OUTSIDE so a queued run on an idle project can still progress
    # without holding the project lock while waiting for capacity.
    async with _global_run_semaphore():
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
                    budget_cap_override_cents=budget_cap_override_cents,
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
        # Belt-and-suspenders: if streaming.run_agent_loop crashed
        # before getting a chance to persist its own assistant message,
        # write a minimal stub here so chat history always shows
        # something for the failed turn. This is the last line of
        # defense — the in-loop exception path tries first via the
        # main session, then falls back to a fresh session via
        # _force_persist_assistant_message. We only land here if BOTH
        # of those failed (or run_agent_loop died before reaching its
        # own except block).
        if run.assistant_message_id is None:
            try:
                # Reconstruct what we can from chat_run_events. The
                # latest text_checkpoint has the cumulative content;
                # tool_call/tool_result events compose the tool_log;
                # the most recent cost_update has the running total
                # the FE renders under the assistant bubble.
                content = ""
                tool_log: List[Dict[str, Any]] = []
                tool_log_idx: Dict[str, int] = {}
                last_cost: Optional[float] = None
                evs = (
                    db.query(ChatRunEvent)
                    .filter(ChatRunEvent.run_id == run.id)
                    .order_by(ChatRunEvent.seq.asc())
                    .all()
                )
                for e in evs:
                    pl = e.payload or {}
                    if e.type == "text_checkpoint":
                        content = pl.get("full_content") or content
                    elif e.type == "tool_call":
                        cid = pl.get("id") or f"_evt_{e.seq}"
                        tool_log_idx[cid] = len(tool_log)
                        tool_log.append({
                            "id": cid,
                            "name": pl.get("name", "?"),
                            "args_preview": pl.get("args_preview", ""),
                        })
                    elif e.type == "tool_result":
                        cid = pl.get("id")
                        if cid and cid in tool_log_idx:
                            tool_log[tool_log_idx[cid]].update({
                                "summary": pl.get("summary"),
                                "cost": pl.get("cost"),
                                "duration_ms": pl.get("duration_ms"),
                            })
                    elif e.type == "cost_update":
                        v = pl.get("total_cost_usd")
                        if isinstance(v, (int, float)):
                            last_cost = float(v)
                ac: Dict[str, Any] = {"error": error[:500], "interrupted": True}
                if tool_log:
                    ac["tool_log"] = tool_log
                if last_cost is not None:
                    # Preserve the cumulative cost so the FE's "$X spent"
                    # line under the assistant message survives a crash —
                    # without this the assistant bubble for a failed run
                    # shows no cost at all, even though credits were used.
                    ac["total_cost_usd"] = last_cost
                msg = ChatMessage(
                    project_id=run.project_id,
                    role="assistant",
                    content=content,
                    applied_changes=ac,
                    version_id=run.version_id,
                    run_id=run.id,
                )
                db.add(msg)
                db.flush()
                run.assistant_message_id = msg.id
            except Exception:
                log.exception("mark_run_failed: stub-msg recovery failed for run %s", run_id)
        user_safe_msg = "Something went wrong on our end. Please try again."
        emit_event(db, run, "error", {"message": user_safe_msg})
        emit_event(db, run, "done", {"stopped": True, "error": user_safe_msg})
    finally:
        db.close()
    _BUS.cleanup_run(run_id)


# ---- Public terminal-marking helpers (called from streaming) -------------
def mark_run_completed(db: Session, run: ChatRun, payload: Dict[str, Any]) -> None:
    """Mark a run completed, but only if the reaper hasn't already
    claimed it as failed/cancelled.

    The orphan reaper runs concurrently with active workers. If a worker
    legitimately took a long time on a single tool call (browser_use can
    run 2+ minutes silently), the reaper might wrongly mark the run as
    failed. If the worker then finishes normally and we blindly set
    status=completed, we'd produce the inconsistent state
    `status=completed` + `error="orphaned"` (and the FE has already seen
    the reaper's `done` event, so silently flipping back to completed
    is the worse outcome anyway). CAS: only flip if not already terminal.
    """
    from sqlalchemy.sql import func
    from sqlalchemy import text as sa_text

    result = db.execute(
        sa_text(
            "UPDATE chat_runs SET status=:s, completed_at=now(), current_phase=NULL "
            "WHERE id=:id AND status NOT IN ('failed', 'cancelled', 'completed')"
        ),
        {"s": RUN_STATUS_COMPLETED, "id": str(run.id)},
    )
    if result.rowcount == 0:
        # Reaper or someone else got here first. Don't emit the `done`
        # event — they emitted their own terminal event already and the
        # FE has moved on. Just clean up local state.
        log.warning(
            "mark_run_completed: run %s was already terminal — skipping",
            run.id,
        )
        _BUS.cleanup_run(run.id)
        return
    # Reflect CAS result back into the ORM instance so callers reading
    # `run.status` see the new value.
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
            last_seq = row.seq
            yield {"type": row.type, "seq": row.seq, **(row.payload or {})}
        # Grace tail — background Scrubby (and other fire-and-forget
        # tasks pinned in `_BACKGROUND_TASKS`) fire row_merged AFTER the
        # run is marked done. Without polling here those late events
        # land in chat_run_events but no subscriber sees them — the user
        # has to refresh to pick up cleared INVALID cells / late badges.
        # 120s covers Scrubby's typical 30-60s bulk roundtrip plus
        # padding for the worst case.
        async for ev in _grace_tail(run_id, last_seq, is_disconnected):
            yield ev
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
                            last_seq = row.seq
                            yield {"type": row.type, "seq": row.seq, **(row.payload or {})}
                        # Background Scrubby (and other fire-and-forget
                        # tasks) fire row_merged events AFTER the agent
                        # marks done. Keep tailing for a grace window so
                        # those late events reach the FE without forcing
                        # a page refresh.
                        async for ev in _grace_tail(run_id, last_seq, is_disconnected):
                            yield ev
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
                # Don't exit yet — late row_merged from background
                # Scrubby verifications still need a delivery window.
                async for ev in _grace_tail(run_id, last_seq, is_disconnected):
                    yield ev
                return
    finally:
        _BUS.unsubscribe(run_id, q)


async def _grace_tail(
    run_id: UUID,
    after_seq: int,
    is_disconnected: Optional[Callable[[], Awaitable[bool]]],
    duration_s: float = 120.0,
    poll_interval_s: float = 3.0,
) -> AsyncGenerator[Dict[str, Any], None]:
    """Poll chat_run_events for new rows after the run goes terminal.

    Scrubby's bulk verify path (and other tasks pinned in
    `_BACKGROUND_TASKS`) finish AFTER the agent emits `done`. They
    write row_merged + email_verified into chat_run_events but, before
    this grace window existed, no subscriber was listening so the FE
    silently missed them — the user had to refresh to see verified
    badges / cleared INVALID cells.
    """
    deadline = asyncio.get_event_loop().time() + duration_s
    last_seq = after_seq
    while True:
        if is_disconnected is not None and await is_disconnected():
            return
        if asyncio.get_event_loop().time() >= deadline:
            return
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
            last_seq = row.seq
            yield {"type": row.type, "seq": row.seq, **(row.payload or {})}
        await asyncio.sleep(poll_interval_s)


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
    from sqlalchemy import text as sa_text

    # Staleness threshold: chat_v2 emits a heartbeat event every 30s
    # while the run is alive (see chat_v2.runs._heartbeat_loop). A
    # missed-heartbeat window of 60 minutes is FAR longer than any
    # legitimate gap — if we don't see ANY event for an hour
    # we can be confident the worker actually died. The sweeper exists
    # to clean up zombies, NOT to police long-running tasks; killing
    # a live run is much worse than slow visibility on a dead one.
    # (DB audit showed 82% of recent failures were watchdog false-positives
    # caused by event-loop saturation from a now-deleted url-verify hook —
    # 60min gives breathing room for any future saturation regression.)
    STALE_THRESHOLD = "60 minutes"
    # Minimum run age: don't even consider a run brand-new.
    MIN_AGE = "5 minutes"

    db = SessionLocal()
    try:
        # Atomic CAS: only flip status if the run is STILL non-terminal
        # AND STILL stale at UPDATE time. If a fresh event landed between
        # us picking candidates and us updating, the staleness check
        # re-evaluates inside the UPDATE and the WHERE clause no-ops.
        # RETURNING gives us the rows we successfully flipped so we know
        # which ones to emit terminal events for. This is the only safe
        # path that won't race the worker — never `SELECT then UPDATE`
        # without re-checking the predicate inside the UPDATE itself.
        cas_sql = sa_text("""
            UPDATE chat_runs
            SET status = 'failed',
                error = 'Worker process appears dead — no heartbeat in 60+ minutes.',
                completed_at = now(),
                current_phase = NULL
            WHERE status IN ('queued', 'running', 'pause_requested')
              AND started_at IS NOT NULL
              AND started_at < now() - cast(:min_age AS interval)
              AND COALESCE(
                    (SELECT max(created_at) FROM chat_run_events WHERE run_id = chat_runs.id),
                    started_at
                  ) < now() - cast(:stale AS interval)
            RETURNING id
        """)
        result = db.execute(cas_sql, {"min_age": MIN_AGE, "stale": STALE_THRESHOLD})
        flipped_ids = [row[0] for row in result.fetchall()]
        db.commit()

        # Emit terminal events ONLY for runs we actually claimed via
        # the CAS. If `flipped_ids` is empty, we touched nothing — no
        # events to emit, no FE state to disturb.
        fixed = 0
        for run_id in flipped_ids:
            run = db.query(ChatRun).get(run_id)
            if run is None:
                continue
            try:
                emit_event(db, run, "error", {"message": run.error})
                emit_event(db, run, "done", {"stopped": True, "error": run.error})
                db.commit()
            except Exception:
                log.exception("orphan recovery: emit failed for run %s", run.id)
                try:
                    db.rollback()
                except Exception:
                    pass
                continue
            # CRITICAL: also cancel the in-process asyncio.Task driving
            # this run. Without this the DB row says "failed" but cell
            # tasks keep churning in memory — racking up real $ on BU /
            # Apollo / FE calls (orphaned-cells bug). Best-effort: if
            # the registry has no task (worker restarted or task already
            # finished), this no-ops.
            try:
                from dsl_worker.chat_v2.runs import cancel_v2_run as _cancel_v2
                _cancel_v2(run.id)
            except Exception:
                log.exception("orphan recovery: cancel failed for run %s", run.id)
            fixed += 1
        return fixed
    finally:
        db.close()


async def orphan_recovery_loop(interval_seconds: int = 30) -> None:
    """Background task: periodically reap orphaned runs.

    Without this, runs orphaned by a worker crash / restart would only
    be cleaned up when the chat worker process next restarts — which
    can be hours, leaving the FE showing a phantom spinner. Ran on a
    30s interval, the worst-case visibility delay is ~30s + the 60s
    heartbeat threshold inside recover_orphan_runs.
    """
    while True:
        try:
            n = await asyncio.to_thread(recover_orphan_runs)
            if n:
                log.warning("orphan recovery: marked %d run(s) failed", n)
        except Exception:
            log.exception("orphan recovery pass failed")
        await asyncio.sleep(interval_seconds)


async def _post_first_chat_to_slack(
    *,
    project_id: str,
    email: Optional[str],
    message: str,
) -> None:
    if not dsl_api_settings.SLACK_PROJECTS_WEBHOOK_URL:
        return

    email_line = email or "_(unknown)_"
    snippet = message if len(message) <= 1500 else message[:1500] + "…"
    quoted = "\n".join(f"> {line}" for line in (snippet.splitlines() or [""]))
    text_msg = (
        f":sparkles: *New project* — {email_line}\n"
        f"{quoted}\n"
        f"• project_id: `{project_id}`"
    )

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.post(
                dsl_api_settings.SLACK_PROJECTS_WEBHOOK_URL,
                json={"text": text_msg},
            )
            resp.raise_for_status()
    except Exception:
        log.exception("Failed to post first-chat notification to Slack for %s", project_id)
