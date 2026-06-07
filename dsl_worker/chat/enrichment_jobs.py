"""Durable enrichment jobs — coordinator + cell writer.

The REST entry point (`routes_enrichment_jobs.py`) creates an
EnrichmentJob + N EnrichmentTask rows and pg_notifies the coordinator.
The coordinator (started in chat-api lifespan) claims tasks via
SELECT … FOR UPDATE SKIP LOCKED under a global Semaphore(25), runs the
cell agent for each, and writes the result. Refresh, network drops,
and worker restarts are all survivable because state lives in Postgres.

The agent's chat-tool path (`enrichment.enrichment_run`) is unchanged —
it still owns the chat-SSE event emission, missing-deps batching, and
the rich tool-result shape. This module is for user-button-triggered
runs only.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import socket
import time
import uuid
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy import text as sa_text
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session

from dsl_api.db import SessionLocal

log = logging.getLogger(__name__)


# Tuning knobs ---------------------------------------------------------------

# Max in-flight cells per coordinator process. Cells beyond this wait at
# the semaphore. The chat-tool path uses its own semaphore(25) — they
# don't share, but in practice only one or the other runs on a project
# at a time so total concurrent cells stays bounded.
GLOBAL_CONCURRENCY = 25

# Classify-tier cells use a SEPARATE, higher concurrency cap.
# They run gpt-5.4-nano with no tools, take ~3s wall-time, cost
# ~$0.0005/cell. The 25-slot global cap was sized for research/deep
# cells that hit FullEnrich/Apollo/BU.
#
# Was 100 — but 100 concurrent streaming OpenAI calls saturate the
# AsyncOpenAI httpx connection pool (default max 100) AND the single
# event loop, so under a 1000-cell run every in-flight read stalls and
# times out at once ("Request timed out" flood) while table/poll
# requests can't get serviced. 40 stays well under the pool ceiling,
# keeps the loop responsive, and still clears 1000 nano cells in a few
# minutes. RPM/TPM is not the limit (tier 5); connection+loop pressure is.
CLASSIFY_CONCURRENCY = 40

# Coordinator claim loop: ask Postgres for up to this many queued tasks
# in one round-trip. Keeps the loop responsive without thrashing the DB.
CLAIM_BATCH = 50

# Watchdog: a task with status='running' and last_heartbeat_at older
# than this is assumed dead (worker crash) and gets reset to queued.
HEARTBEAT_STALE_SECONDS = 300

# Hard ceiling on a single cell agent run, enforced by the coordinator.
# FE's bulk_enrich budget is 600s + retries can add ~200s + a few
# web_searches; legitimate worst-case is around 12-13min. 900s gives
# headroom and circuit-breaks the rest. Past this the cell is marked
# failed with a 'cell agent exceeded hard timeout' error so the slot
# frees up and the user can retry.
CELL_AGENT_HARD_TIMEOUT_S = 900

# How often the watchdog runs.
WATCHDOG_INTERVAL_SECONDS = 60

# How often the coordinator's safety poll runs (in case a LISTEN
# notification was missed during reconnect).
COORDINATOR_POLL_SECONDS = 5


WORKER_ID = f"{socket.gethostname()}:{os.getpid()}:{uuid.uuid4().hex[:8]}"


# ---------------------------------------------------------------------------
# Pubsub for SSE tails — coordinator publishes, SSE endpoints subscribe.
# Listeners are keyed on job_id and only see events for their job.
# ---------------------------------------------------------------------------

_subscribers: Dict[str, "list[asyncio.Queue[dict]]"] = {}

# Main event loop, captured at startup. emit_event runs in asyncio.to_thread
# (the coordinator persists off the loop), so _publish is called from a
# worker thread — asyncio.Queue.put_nowait is not thread-safe, so the fanout
# is marshaled back onto the loop via call_soon_threadsafe.
_event_loop: "Optional[asyncio.AbstractEventLoop]" = None

# Terminal-ish events that must never be dropped from a subscriber queue —
# dropping a cell_done / job_done leaves the FE cell spinner stuck until a
# manual refresh (happens when a backgrounded tab drains its SSE queue slowly
# and it fills). For these, evict the oldest queued event to make room.
_CRITICAL_JOB_EVENTS = {
    "cell_done", "cell_failed", "job_done", "job_failed", "job_cancelled",
}


def set_event_loop(loop: "asyncio.AbstractEventLoop") -> None:
    """Register the main loop so off-loop publishers (the coordinator's
    to_thread emits) marshal fanout safely. Call once at app startup."""
    global _event_loop
    _event_loop = loop


def subscribe(job_id: str) -> "asyncio.Queue[dict]":
    q: "asyncio.Queue[dict]" = asyncio.Queue(maxsize=512)
    _subscribers.setdefault(job_id, []).append(q)
    return q


def unsubscribe(job_id: str, q: "asyncio.Queue[dict]") -> None:
    lst = _subscribers.get(job_id) or []
    try:
        lst.remove(q)
    except ValueError:
        pass
    if not lst and job_id in _subscribers:
        _subscribers.pop(job_id, None)


def _publish(job_id: str, event: dict) -> None:
    """Fan out a job event to live SSE subscribers. Thread-safe: when called
    from a worker thread (coordinator persists in to_thread), marshals the
    fanout onto the loop. Terminal events evict an old event rather than drop."""
    loop = _event_loop
    if loop is not None:
        try:
            on_loop = asyncio.get_running_loop() is loop
        except RuntimeError:
            on_loop = False
        if not on_loop:
            loop.call_soon_threadsafe(_fanout_job_event, job_id, event)
            return
    _fanout_job_event(job_id, event)


def _fanout_job_event(job_id: str, event: dict) -> None:
    critical = event.get("kind") in _CRITICAL_JOB_EVENTS
    for q in list(_subscribers.get(job_id) or []):
        try:
            q.put_nowait(event)
        except asyncio.QueueFull:
            if critical:
                try:
                    q.get_nowait()
                    q.put_nowait(event)
                    continue
                except Exception:
                    pass


# ---------------------------------------------------------------------------
# Event log writer — every coordinator action that the FE cares about
# lands here. SSE clients read from this table on reconnect (replay from
# cursor=last seen id) and tail via pubsub for live events.
# ---------------------------------------------------------------------------


def emit_event(db: Session, job_id: str, kind: str, payload: dict) -> int:
    """Insert an enrichment_event row + publish to in-process subscribers.

    Returns the inserted event id. Caller owns the transaction —
    most callers commit immediately after to make the event durable
    before notifying subscribers.
    """
    eid = db.execute(
        sa_text(
            "INSERT INTO enrichment_events (job_id, kind, payload) "
            "VALUES (CAST(:jid AS uuid), :kind, CAST(:payload AS jsonb)) "
            "RETURNING id"
        ),
        {"jid": job_id, "kind": kind, "payload": json.dumps(payload)},
    ).scalar()
    db.commit()
    _publish(job_id, {"id": int(eid), "kind": kind, "payload": payload})
    return int(eid)


# ---------------------------------------------------------------------------
# Per-cell write — clean replacement for the inline write block in
# enrichment._run_enrichment_on_rows. Same semantics (advisory lock,
# fresh-read, merge deltas, sidecars, sources, tag merge) without the
# chat-SSE event coupling. Returns the fresh row + tags so the caller
# can build an enrichment_event payload.
# ---------------------------------------------------------------------------


def _column_diff(
    old_row: Dict[str, Any],
    new_row: Dict[str, Any],
    column_names: List[str],
) -> List[Tuple[str, Optional[str], Optional[str]]]:
    """Compute (column, old, new) tuples for columns whose value changed.

    Sidecar keys (__cell_status__, __cell_cost__) and unknown columns
    are ignored. Values are stringified the same way the materialize
    table stores them so the deltas line up with the count rows.
    """
    out: List[Tuple[str, Optional[str], Optional[str]]] = []
    for col in column_names:
        ov = _stringify(old_row.get(col))
        nv = _stringify(new_row.get(col))
        if ov != nv:
            out.append((col, ov, nv))
    return out


MAX_DISTINCT_VALUE_LEN = 500


def _stringify(v: Any) -> Optional[str]:
    """Canonical form for distinct-value indexing.

    None and empty string both collapse to None (no row in
    table_column_value). Lists/dicts get JSON-encoded with sorted keys
    so equal values hash to the same key.

    Values longer than MAX_DISTINCT_VALUE_LEN return None — Postgres
    btree can't index rows over ~2700 bytes, and a 500-char filter
    dropdown entry is the limit of what's useful in the UI anyway.
    Filters on those columns just fall through to the live-scan path.
    """
    if v is None:
        return None
    if isinstance(v, str):
        if not v:
            return None
        if len(v) > MAX_DISTINCT_VALUE_LEN:
            return None
        return v
    if isinstance(v, (int, float, bool)):
        return str(v)
    try:
        s = json.dumps(v, sort_keys=True, separators=(",", ":"))
    except Exception:
        s = str(v)
    if len(s) > MAX_DISTINCT_VALUE_LEN:
        return None
    return s


def _apply_column_value_deltas(
    db: Session,
    table_id: str,
    deltas: List[Tuple[str, Optional[str], Optional[str]]],
) -> None:
    """Update table_column_value counts for the changed columns.

    Each (column, old, new) decrements the old row's count (if any) and
    increments the new row's count (if any). Counts that reach 0 get
    deleted so the top-N read doesn't have to filter them out.
    """
    for col, old, new in deltas:
        if old is not None:
            db.execute(
                sa_text(
                    "UPDATE table_column_value SET count = count - 1, "
                    "updated_at = now() "
                    "WHERE table_id = CAST(:tid AS uuid) "
                    "  AND column_name = :col AND value = :v"
                ),
                {"tid": table_id, "col": col, "v": old},
            )
            db.execute(
                sa_text(
                    "DELETE FROM table_column_value "
                    "WHERE table_id = CAST(:tid AS uuid) "
                    "  AND column_name = :col AND value = :v AND count <= 0"
                ),
                {"tid": table_id, "col": col, "v": old},
            )
        if new is not None:
            db.execute(
                sa_text(
                    "INSERT INTO table_column_value "
                    "(table_id, column_name, value, count, updated_at) "
                    "VALUES (CAST(:tid AS uuid), :col, :v, 1, now()) "
                    "ON CONFLICT (table_id, column_name, value) "
                    "DO UPDATE SET count = table_column_value.count + 1, "
                    "              updated_at = now()"
                ),
                {"tid": table_id, "col": col, "v": new},
            )


def commit_cell_result(
    db: Session,
    *,
    sample_id: str,
    table_id: str,
    target_cols: List[str],
    new_fields: Dict[str, Any],
    new_sources: Dict[str, List[Dict[str, Any]]],
    cost_usd: float,
    status: str,
    overwrite: bool,
) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
    """Apply a single cell-agent result onto samples.row + samples.tags.

    Mirrors enrichment.py's per-row write block but is callable from the
    new coordinator. Holds an advisory xact lock on the sample so two
    enrichments running on the same row serialize cleanly.

    Returns (final_row, fresh_tags). final_row is what the FE should
    render; fresh_tags is the merged tags dict (or None if no tag
    changes happened).
    """
    db.execute(
        sa_text("SELECT pg_advisory_xact_lock(hashtextextended(:sid, 0))"),
        {"sid": str(sample_id)},
    )
    fresh = db.execute(
        sa_text("SELECT row FROM samples WHERE id=:sid"),
        {"sid": sample_id},
    ).fetchone()
    if not fresh:
        db.rollback()
        return {}, None
    fresh_row = fresh[0] if isinstance(fresh[0], dict) else json.loads(fresh[0] or "{}")
    if not isinstance(fresh_row, dict):
        fresh_row = {}

    # Build deltas
    value_delta: Dict[str, Any] = {}
    if isinstance(new_fields, dict):
        for cn in target_cols:
            if cn in new_fields:
                value_delta[cn] = new_fields[cn]

    status_set: Dict[str, str] = {}
    status_clear: List[str] = []
    if status == "hit_budget":
        for cn in target_cols:
            if not (isinstance(new_fields, dict) and new_fields.get(cn)):
                status_set[cn] = "hit_budget"
    elif status == "filled" and isinstance(new_fields, dict):
        for cn in target_cols:
            v = new_fields.get(cn)
            if v not in (None, ""):
                status_clear.append(cn)
            else:
                status_set[cn] = "not_found"

    cost_delta: Dict[str, float] = {}
    if cost_usd > 0:
        keys = list(new_fields.keys()) if isinstance(new_fields, dict) and new_fields else list(target_cols)
        for cn in keys:
            cost_delta[cn] = float(cost_usd) * 10.0

    sources_to_persist: Dict[str, List[Dict[str, Any]]] = {}
    if isinstance(new_sources, dict):
        for cn, citations in new_sources.items():
            if (
                isinstance(new_fields, dict)
                and new_fields.get(cn) not in (None, "")
                and isinstance(citations, list)
                and citations
            ):
                sources_to_persist[cn] = citations

    final = dict(fresh_row)
    for k, v in value_delta.items():
        if not overwrite and fresh_row.get(k) not in (None, ""):
            continue
        final[k] = v

    final_status = final.get("__cell_status__") if isinstance(final.get("__cell_status__"), dict) else {}
    final_status = dict(final_status) if isinstance(final_status, dict) else {}
    for cn, s in status_set.items():
        final_status[cn] = s
    for cn in status_clear:
        final_status.pop(cn, None)
    if final_status:
        final["__cell_status__"] = final_status
    elif "__cell_status__" in final:
        final.pop("__cell_status__")

    final_cost = final.get("__cell_cost__") if isinstance(final.get("__cell_cost__"), dict) else {}
    final_cost = dict(final_cost) if isinstance(final_cost, dict) else {}
    for cn, c in cost_delta.items():
        final_cost[cn] = c
    if final_cost:
        final["__cell_cost__"] = final_cost

    # Compute distinct-value deltas BEFORE writing so we count exactly
    # the columns whose value changed.
    distinct_deltas = _column_diff(fresh_row, final, target_cols)

    if sources_to_persist:
        db.execute(
            sa_text(
                "UPDATE samples "
                "SET row=CAST(:row AS jsonb), "
                "    tags=jsonb_set("
                "      COALESCE(tags, '{}'::jsonb), "
                "      '{sources}', "
                "      COALESCE(tags->'sources', '{}'::jsonb) || CAST(:srcs AS jsonb)"
                "    ) "
                "WHERE id=:sid"
            ),
            {
                "row": json.dumps(final),
                "srcs": json.dumps(sources_to_persist),
                "sid": sample_id,
            },
        )
    else:
        db.execute(
            sa_text("UPDATE samples SET row=CAST(:row AS jsonb) WHERE id=:sid"),
            {"row": json.dumps(final), "sid": sample_id},
        )

    if distinct_deltas:
        _apply_column_value_deltas(db, table_id, distinct_deltas)

    db.commit()

    fresh_tags_row = db.execute(
        sa_text("SELECT tags FROM samples WHERE id=:sid"),
        {"sid": sample_id},
    ).fetchone()
    fresh_tags: Optional[Dict[str, Any]] = None
    if fresh_tags_row and fresh_tags_row[0] is not None:
        ft = fresh_tags_row[0]
        fresh_tags = ft if isinstance(ft, dict) else json.loads(ft or "{}")

    return final, fresh_tags


# ---------------------------------------------------------------------------
# Coordinator — one async loop per chat-api process. Claims tasks,
# runs them under a global semaphore, writes events.
# ---------------------------------------------------------------------------


class Coordinator:
    """Singleton-per-process. Started by app.lifespan.

    Two cooperating tasks:
      • _claim_loop: wakes on pg NOTIFY or every COORDINATOR_POLL_SECONDS,
        claims up to (GLOBAL_CONCURRENCY - in_flight) tasks, schedules
        them.
      • _watchdog_loop: every WATCHDOG_INTERVAL_SECONDS, resets tasks
        whose heartbeat went stale (worker died mid-task).

    Stopping is cooperative — `stop()` cancels both loops and waits for
    in-flight cell work to finish.
    """

    def __init__(self) -> None:
        self._sem = asyncio.Semaphore(GLOBAL_CONCURRENCY)
        self._classify_sem = asyncio.Semaphore(CLASSIFY_CONCURRENCY)
        self._wake = asyncio.Event()
        self._stop = asyncio.Event()
        self._tasks: List[asyncio.Task] = []
        self._in_flight: set[str] = set()
        # Tier breakdown so we can size the claim-batch capacity per
        # semaphore without overshooting either.
        self._in_flight_classify: set[str] = set()

    async def start(self) -> None:
        self._tasks = [
            asyncio.create_task(self._claim_loop(), name="enrichment-coordinator"),
            asyncio.create_task(self._watchdog_loop(), name="enrichment-watchdog"),
            asyncio.create_task(self._listen_loop(), name="enrichment-listen"),
        ]
        log.info("enrichment coordinator started; worker_id=%s", WORKER_ID)

    async def stop(self) -> None:
        self._stop.set()
        self._wake.set()
        for t in self._tasks:
            t.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)

    def wake(self) -> None:
        """Synchronous wake — safe to call from any thread."""
        try:
            loop = asyncio.get_running_loop()
            loop.call_soon_threadsafe(self._wake.set)
        except RuntimeError:
            self._wake.set()

    # ---- claim loop --------------------------------------------------

    async def _claim_loop(self) -> None:
        while not self._stop.is_set():
            try:
                claimed = await asyncio.to_thread(self._claim_batch)
                if claimed:
                    for task in claimed:
                        asyncio.create_task(
                            self._run_task(task), name=f"enrichment-task-{task['id'][:8]}"
                        )
                else:
                    # No work — wait for a wake or the safety poll.
                    try:
                        await asyncio.wait_for(
                            self._wake.wait(), timeout=COORDINATOR_POLL_SECONDS
                        )
                    except asyncio.TimeoutError:
                        pass
                    self._wake.clear()
            except asyncio.CancelledError:
                break
            except Exception:
                log.exception("enrichment coordinator claim loop iteration failed")
                await asyncio.sleep(2)

    def _claim_batch(self) -> List[Dict[str, Any]]:
        """Atomically claim up to (CLAIM_BATCH, capacity) queued tasks.

        Synchronous because we use psycopg2 row-level locks. Runs in a
        thread so the event loop stays free.

        Capacity is the sum of headroom on both semaphores — heavy lane
        (GLOBAL_CONCURRENCY, all non-classify tiers) and classify lane
        (CLASSIFY_CONCURRENCY). The actual semaphore acquire in
        _run_task throttles each tier to its own ceiling; this just
        caps how many tasks we'll claim per round so we don't grab 100
        heavy tasks when only 25 can actually run.
        """
        in_flight_heavy = len(self._in_flight) - len(self._in_flight_classify)
        capacity_heavy = max(0, GLOBAL_CONCURRENCY - in_flight_heavy)
        capacity_classify = max(0, CLASSIFY_CONCURRENCY - len(self._in_flight_classify))
        capacity = capacity_heavy + capacity_classify
        if capacity == 0:
            return []
        limit = min(CLAIM_BATCH, capacity)

        db = SessionLocal()
        try:
            rows = db.execute(
                sa_text(
                    "WITH claimed AS ("
                    "  SELECT id FROM enrichment_tasks "
                    "  WHERE status='queued' "
                    "  ORDER BY created_at "
                    "  LIMIT :n "
                    "  FOR UPDATE SKIP LOCKED"
                    ") "
                    "UPDATE enrichment_tasks t "
                    "SET status='running', "
                    "    attempts = t.attempts + 1, "
                    "    worker_id = :wid, "
                    "    started_at = now(), "
                    "    last_heartbeat_at = now() "
                    "FROM claimed "
                    "WHERE t.id = claimed.id "
                    "RETURNING t.id::text, t.job_id::text, t.project_id::text, "
                    "          t.enrichment_id::text, t.sample_id::text, "
                    "          (SELECT LOWER(COALESCE(e.action->>'research', e.action->>'tier', '')) "
                    "           FROM enrichments e WHERE e.id = t.enrichment_id) AS research_tier"
                ),
                {"n": limit, "wid": WORKER_ID},
            ).fetchall()
            db.commit()
            # Track in-flight before returning so a fast wake doesn't
            # double-claim past capacity.
            out = []
            for r in rows:
                tid = r[0]
                tier = (r[5] or "").lower()
                self._in_flight.add(tid)
                if tier == "classify":
                    self._in_flight_classify.add(tid)
                out.append({
                    "id": tid,
                    "job_id": r[1],
                    "project_id": r[2],
                    "enrichment_id": r[3],
                    "sample_id": r[4],
                    "research_tier": tier,
                })
            # Mark job started_at on first claimed task per job.
            if out:
                job_ids = sorted({r["job_id"] for r in out})
                for jid in job_ids:
                    db.execute(
                        sa_text(
                            "UPDATE enrichment_jobs SET status='running', "
                            "started_at = COALESCE(started_at, now()) "
                            "WHERE id = CAST(:jid AS uuid) AND status = 'queued'"
                        ),
                        {"jid": jid},
                    )
                db.commit()
            return out
        finally:
            db.close()

    # ---- per-task run -------------------------------------------------

    async def _run_task(self, task: Dict[str, Any]) -> None:
        task_id = task["id"]
        # Pick the semaphore by tier — classify gets the high-concurrency
        # lane, everything else stays on the 25-slot heavy lane.
        is_classify = (task.get("research_tier") or "").lower() == "classify"
        sem = self._classify_sem if is_classify else self._sem
        async with sem:
            heartbeat = asyncio.create_task(
                self._heartbeat(task_id), name=f"hb-{task_id[:8]}"
            )
            try:
                await self._execute_task(task)
            except Exception:
                log.exception("enrichment task %s crashed", task_id)
                await asyncio.to_thread(self._mark_task_failed, task, "crash")
            finally:
                heartbeat.cancel()
                self._in_flight.discard(task_id)
                if is_classify:
                    self._in_flight_classify.discard(task_id)
                # Wake the claim loop so it can fill the now-free slot.
                self._wake.set()

    async def _heartbeat(self, task_id: str) -> None:
        try:
            while True:
                await asyncio.sleep(30)
                await asyncio.to_thread(self._bump_heartbeat, task_id)
        except asyncio.CancelledError:
            return

    def _bump_heartbeat(self, task_id: str) -> None:
        db = SessionLocal()
        try:
            db.execute(
                sa_text(
                    "UPDATE enrichment_tasks SET last_heartbeat_at = now() "
                    "WHERE id = CAST(:tid AS uuid)"
                ),
                {"tid": task_id},
            )
            db.commit()
        finally:
            db.close()

    async def _execute_task(self, task: Dict[str, Any]) -> None:
        """Run one cell agent + write the result + emit events.

        Defensive: any failure is caught, logged, persisted as a task
        failure + cell_failed event, and credits are NOT charged for
        the failed cell.
        """
        from dsl_worker.chat.tools import ToolContext
        from dsl_worker.chat.enrichment import _execute_action, _scrub_failed_values
        from dsl_worker.chat.cell_runs import REGISTRY as CELL_RUNS

        # Load job + enrichment + row state in a short transaction. We
        # release the connection before the cell agent runs so we don't
        # pin a session for 30-60s while FullEnrich / browser_use churn.
        state = await asyncio.to_thread(self._load_task_state, task)
        if state is None:
            await asyncio.to_thread(
                self._mark_task_failed, task, "job or enrichment missing"
            )
            return

        job = state["job"]
        enrichment = state["enrichment"]
        sample = state["sample"]

        all_cols = [c["name"] for c in enrichment["columns"]]
        # Column-scoped run (e.g. retry ONE cell): job scope may carry a subset
        # of column names. Narrow target_cols so the cell agent fills only those
        # and never re-researches or overwrites already-filled siblings.
        scope_cols = job["scope"].get("columns") if isinstance(job.get("scope"), dict) else None
        target_cols = [c for c in all_cols if c in scope_cols] if scope_cols else all_cols
        if not target_cols:
            target_cols = all_cols
        overwrite = bool(job["scope"].get("overwrite", False))

        # Already-filled check (only when overwrite=false). The job-create
        # endpoint may have queued the task before another run filled it.
        if not overwrite:
            row_data = sample["row"]
            if isinstance(row_data, dict) and all(
                row_data.get(c) not in (None, "") for c in target_cols
            ):
                await asyncio.to_thread(
                    self._mark_task_skipped, task, "already filled"
                )
                return

        # Register the in-flight cell so the FE /cells/running endpoint
        # still works for FE pieces that haven't switched to enrichment_events.
        await CELL_RUNS.add(
            task["project_id"], task["enrichment_id"],
            task["sample_id"], target_cols,
        )

        ctx = ToolContext(
            db=None,  # cell agent re-opens its own session as needed
            project_id=task["project_id"],
            user_id=str(job["user_id"]),
            run_id=None,
        )
        # cell_agent (and a few tool implementations) read ctx.db. Give
        # it a fresh session that we close after the agent returns.
        ctx_db = SessionLocal()
        ctx.db = ctx_db

        # cell_start event for SSE subscribers.
        await asyncio.to_thread(
            self._emit_in_session,
            task["job_id"], "cell_start",
            {
                "task_id": task_id_short(task["id"]),
                "row_id": task["sample_id"],
                "enrichment_id": task["enrichment_id"],
                "columns": target_cols,
            },
        )

        raw_row = _scrub_failed_values(sample["raw_row"] or {}, sample["tags"])
        # When column-scoped, force the cell agent's columns_to_fill to the
        # subset. Everything downstream (commit, status, cost, events) already
        # keys off target_cols, so siblings stay exactly as they were.
        eff_action = enrichment["action"]
        if scope_cols:
            eff_action = {**eff_action, "columns_to_fill": target_cols}
        try:
            new_fields, new_sources, cost, status = await asyncio.wait_for(
                _execute_action(
                    eff_action,
                    sample["row"] or {},
                    enrichment["per_row_credit_cap"],
                    enrichment["columns"],
                    ctx,
                    enrichment_id=task["enrichment_id"],
                    sample_id=task["sample_id"],
                    raw_row=raw_row,
                ),
                timeout=CELL_AGENT_HARD_TIMEOUT_S,
            )
        except asyncio.TimeoutError:
            log.warning(
                "cell agent hard-timeout on task %s (sample %s) after %ds",
                task["id"], task["sample_id"], CELL_AGENT_HARD_TIMEOUT_S,
            )
            await asyncio.to_thread(
                self._mark_task_failed, task,
                f"cell agent exceeded {CELL_AGENT_HARD_TIMEOUT_S}s hard timeout",
            )
            try:
                ctx_db.close()
            finally:
                await CELL_RUNS.remove(
                    task["project_id"], task["enrichment_id"],
                    task["sample_id"], target_cols,
                )
            return
        except asyncio.CancelledError:
            await asyncio.to_thread(
                self._mark_task_failed, task, "cancelled mid-flight"
            )
            try:
                ctx_db.close()
            finally:
                await CELL_RUNS.remove(
                    task["project_id"], task["enrichment_id"],
                    task["sample_id"], target_cols,
                )
            raise
        except Exception as e:
            log.exception(
                "cell agent raised for task %s sample %s",
                task["id"], task["sample_id"],
            )
            await asyncio.to_thread(
                self._mark_task_failed, task, f"cell agent crashed: {e!r}"
            )
            try:
                ctx_db.close()
            finally:
                await CELL_RUNS.remove(
                    task["project_id"], task["enrichment_id"],
                    task["sample_id"], target_cols,
                )
            return
        finally:
            try:
                ctx_db.close()
            except Exception:
                pass

        # Commit the row write in a fresh session.
        try:
            final, fresh_tags = await asyncio.to_thread(
                self._commit_in_session,
                task["sample_id"], enrichment["table_id"], target_cols,
                new_fields, new_sources, float(cost or 0.0), status, overwrite,
            )
        except Exception as e:
            log.exception(
                "commit_cell_result failed for task %s sample %s",
                task["id"], task["sample_id"],
            )
            await asyncio.to_thread(
                self._mark_task_failed, task, f"commit failed: {e!r}"
            )
            await CELL_RUNS.remove(
                task["project_id"], task["enrichment_id"],
                task["sample_id"], target_cols,
            )
            return

        # Successful cell — record cost on the task + bump job counters
        # + emit cell_done.
        produced_value = (
            status == "filled"
            and isinstance(new_fields, dict)
            and any(v not in (None, "") for v in new_fields.values())
        )
        charge_usd = float(cost or 0.0) if produced_value else float(cost or 0.0) * 0.10

        # Emit cell_done BEFORE marking the task done. _mark_task_done emits
        # job_done once the LAST task is terminal; if it ran first, job_done
        # would get a lower event id than this cell_done — and the SSE stream
        # closes on job_done, dropping the trailing cell_done. The value
        # lands in the DB but the FE never receives the event that clears the
        # spinner + paints the value (spinner spins forever; value only shows
        # on manual refresh). Ordering cell_done first keeps every per-cell
        # terminal event strictly before the job terminal event.
        await asyncio.to_thread(
            self._emit_in_session,
            task["job_id"], "cell_done",
            {
                "task_id": task_id_short(task["id"]),
                "row_id": task["sample_id"],
                "enrichment_id": task["enrichment_id"],
                "columns": target_cols,
                "new_fields": new_fields if isinstance(new_fields, dict) else None,
                "status": status,
                "cost_credits": float(cost or 0.0) * 10.0,
                "row": final,
                "tags": fresh_tags,
            },
        )
        await asyncio.to_thread(
            self._mark_task_done,
            task, float(cost or 0.0), charge_usd, status,
        )
        await CELL_RUNS.remove(
            task["project_id"], task["enrichment_id"],
            task["sample_id"], target_cols,
        )

        # Email verify hook: kept consistent with the chat-tool path. If
        # an email landed, Scrubby verifies async; updates flow through
        # the chat-run event stream (not us), but the row tags update so
        # the next FE poll/refresh sees the verification badge.
        if isinstance(new_fields, dict) and new_fields:
            try:
                from dsl_worker.chat import email_verify_hook
                email_verify_hook.schedule_for_row(
                    run_id=None,
                    sample_id=task["sample_id"],
                    written_values=new_fields,
                    columns=enrichment["columns"],
                )
            except Exception:
                log.exception("email_verify_hook.schedule_for_row raised; suppressed")

    # ---- task state helpers (sync, run in threads) -------------------

    def _load_task_state(self, task: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        db = SessionLocal()
        try:
            job = db.execute(
                sa_text(
                    "SELECT user_id::text, scope, status "
                    "FROM enrichment_jobs WHERE id=CAST(:jid AS uuid)"
                ),
                {"jid": task["job_id"]},
            ).fetchone()
            if not job:
                return None
            if job[2] in ("cancelled", "failed"):
                return None
            enrichment = db.execute(
                sa_text(
                    "SELECT table_id::text, columns, action, per_row_credit_cap "
                    "FROM enrichments WHERE id=CAST(:eid AS uuid) AND deleted_at IS NULL"
                ),
                {"eid": task["enrichment_id"]},
            ).fetchone()
            if not enrichment:
                return None
            sample = db.execute(
                sa_text(
                    "SELECT row, raw_row, tags FROM samples WHERE id=CAST(:sid AS uuid) "
                    "AND deleted_at IS NULL"
                ),
                {"sid": task["sample_id"]},
            ).fetchone()
            if not sample:
                return None

            columns = enrichment[1] if isinstance(enrichment[1], list) else json.loads(enrichment[1] or "[]")
            action = enrichment[2] if isinstance(enrichment[2], dict) else json.loads(enrichment[2] or "{}")
            scope = job[1] if isinstance(job[1], dict) else json.loads(job[1] or "{}")

            return {
                "job": {"user_id": job[0], "scope": scope},
                "enrichment": {
                    "table_id": enrichment[0],
                    "columns": columns,
                    "action": action,
                    "per_row_credit_cap": enrichment[3],
                },
                "sample": {
                    "row": sample[0] if isinstance(sample[0], dict) else json.loads(sample[0] or "{}"),
                    "raw_row": sample[1] if isinstance(sample[1], dict) else (json.loads(sample[1]) if sample[1] else {}),
                    "tags": sample[2] if isinstance(sample[2], dict) else (json.loads(sample[2]) if sample[2] else {}),
                },
            }
        finally:
            db.close()

    def _commit_in_session(
        self,
        sample_id: str,
        table_id: str,
        target_cols: List[str],
        new_fields: Dict[str, Any],
        new_sources: Dict[str, List[Dict[str, Any]]],
        cost_usd: float,
        status: str,
        overwrite: bool,
    ) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
        db = SessionLocal()
        try:
            return commit_cell_result(
                db,
                sample_id=sample_id,
                table_id=table_id,
                target_cols=target_cols,
                new_fields=new_fields,
                new_sources=new_sources,
                cost_usd=cost_usd,
                status=status,
                overwrite=overwrite,
            )
        finally:
            db.close()

    def _emit_in_session(self, job_id: str, kind: str, payload: dict) -> int:
        db = SessionLocal()
        try:
            return emit_event(db, job_id, kind, payload)
        finally:
            db.close()

    def _mark_task_failed(self, task: Dict[str, Any], reason: str) -> None:
        db = SessionLocal()
        try:
            db.execute(
                sa_text(
                    "UPDATE enrichment_tasks SET status='failed', "
                    "error=:err, ended_at=now() "
                    "WHERE id=CAST(:tid AS uuid)"
                ),
                {"tid": task["id"], "err": reason[:1000]},
            )
            db.execute(
                sa_text(
                    "UPDATE enrichment_jobs "
                    "SET failed_tasks = failed_tasks + 1 "
                    "WHERE id=CAST(:jid AS uuid)"
                ),
                {"jid": task["job_id"]},
            )
            db.commit()
            emit_event(
                db, task["job_id"], "cell_failed",
                {
                    "task_id": task_id_short(task["id"]),
                    "row_id": task["sample_id"],
                    "enrichment_id": task["enrichment_id"],
                    "error": reason,
                },
            )
            self._maybe_finalize_job(db, task["job_id"])
        finally:
            db.close()

    def _mark_task_skipped(self, task: Dict[str, Any], reason: str) -> None:
        db = SessionLocal()
        try:
            db.execute(
                sa_text(
                    "UPDATE enrichment_tasks SET status='skipped', "
                    "error=:err, ended_at=now() "
                    "WHERE id=CAST(:tid AS uuid)"
                ),
                {"tid": task["id"], "err": reason[:1000]},
            )
            db.execute(
                sa_text(
                    "UPDATE enrichment_jobs "
                    "SET done_tasks = done_tasks + 1 "
                    "WHERE id=CAST(:jid AS uuid)"
                ),
                {"jid": task["job_id"]},
            )
            db.commit()
            self._maybe_finalize_job(db, task["job_id"])
        finally:
            db.close()

    def _mark_task_done(
        self,
        task: Dict[str, Any],
        cost_usd: float,
        charge_usd: float,
        status: str,
    ) -> None:
        db = SessionLocal()
        try:
            db.execute(
                sa_text(
                    "UPDATE enrichment_tasks "
                    "SET status='done', cost_usd=:c, ended_at=now() "
                    "WHERE id=CAST(:tid AS uuid)"
                ),
                {"tid": task["id"], "c": cost_usd},
            )
            db.execute(
                sa_text(
                    "UPDATE enrichment_jobs "
                    "SET done_tasks = done_tasks + 1, "
                    "    cost_usd = cost_usd + :c "
                    "WHERE id=CAST(:jid AS uuid)"
                ),
                {"jid": task["job_id"], "c": charge_usd},
            )
            db.commit()
            # Charge credits immediately so the user sees the balance move
            # in real time. Best-effort — a settle pass at job finalize
            # reconciles any drift.
            self._charge_credits(db, task["job_id"], charge_usd)
            self._maybe_finalize_job(db, task["job_id"])
        finally:
            db.close()

    def _charge_credits(self, db: Session, job_id: str, charge_usd: float) -> None:
        if charge_usd <= 0:
            return
        try:
            job = db.execute(
                sa_text(
                    "SELECT user_id::text, project_id::text "
                    "FROM enrichment_jobs WHERE id=CAST(:jid AS uuid)"
                ),
                {"jid": job_id},
            ).fetchone()
            if not job:
                return
            from dsl_api.models import Account
            from dsl_api.credits import consume_credits
            from dsl_api.plans import CENTS_PER_CREDIT
            user_id, project_id = job[0], job[1]
            account = db.query(Account).filter(Account.user_id == user_id).first()
            if not account:
                return
            spend_cents = int(round(charge_usd * 100))
            if spend_cents <= 0:
                return
            consume_credits(
                db, account, spend_cents / CENTS_PER_CREDIT,
                project_id=project_id, reason="enrichment_job",
            )
            db.commit()
        except Exception:
            log.exception("credit charge failed for job %s; suppressed", job_id)
            try:
                db.rollback()
            except Exception:
                pass

    def _maybe_finalize_job(self, db: Session, job_id: str) -> None:
        """If all tasks are terminal, mark the job done and emit job_done.

        Uses a single atomic UPDATE...RETURNING with the terminal check
        in the WHERE clause. The previous "SELECT FOR UPDATE then UPDATE"
        pattern deadlocked when multiple workers (pc1 + nlpfollower on
        dev, or two replicas on prod) finalized the same job at the same
        moment: each had already touched the job row (counter increment
        in _mark_task_done) and held a foreign-key shared lock from the
        task UPDATE, then both raced for the FOR UPDATE → cycle.

        Atomic UPDATE eliminates the lock-acquisition window. Either we
        win the race (the row matched, RETURNING gives us the snapshot,
        emit job_done) or we lose (no row returned, someone else
        finalized — silent no-op). PG still occasionally raises a
        DeadlockDetected on related FK/index locks, so retry up to 3x.
        """
        for attempt in range(3):
            try:
                row = db.execute(
                    sa_text(
                        "UPDATE enrichment_jobs "
                        "SET status='done', ended_at=now() "
                        "WHERE id=CAST(:jid AS uuid) "
                        "  AND status NOT IN ('done', 'failed', 'cancelled') "
                        "  AND total_tasks > 0 "
                        "  AND (done_tasks + failed_tasks) >= total_tasks "
                        "RETURNING total_tasks, done_tasks, failed_tasks"
                    ),
                    {"jid": job_id},
                ).fetchone()
                db.commit()
                if row:
                    total, done, failed = row
                    emit_event(
                        db, job_id, "job_done",
                        {"total": total, "done": done, "failed": failed},
                    )
                return
            except OperationalError as e:
                # PG deadlock detector picked us as the victim. Roll back
                # the half-committed work and retry with fresh locks.
                msg = str(e).lower()
                if "deadlock" not in msg or attempt >= 2:
                    log.exception("finalize_job failed for %s", job_id)
                    try:
                        db.rollback()
                    except Exception:
                        pass
                    return
                try:
                    db.rollback()
                except Exception:
                    pass
                # Tiny jitter so we don't immediately re-collide with
                # the same peer.
                time.sleep(0.05 * (attempt + 1))

    # ---- watchdog ----------------------------------------------------

    async def _watchdog_loop(self) -> None:
        while not self._stop.is_set():
            try:
                await asyncio.sleep(WATCHDOG_INTERVAL_SECONDS)
                n = await asyncio.to_thread(self._reap_stale)
                if n:
                    log.warning("watchdog reset %d stale enrichment task(s)", n)
                    self._wake.set()
            except asyncio.CancelledError:
                break
            except Exception:
                log.exception("watchdog iteration failed")

    def _reap_stale(self) -> int:
        db = SessionLocal()
        try:
            res = db.execute(
                sa_text(
                    "UPDATE enrichment_tasks "
                    "SET status='queued', worker_id=NULL, "
                    "    last_heartbeat_at=NULL, started_at=NULL "
                    "WHERE status='running' "
                    "  AND last_heartbeat_at < now() - make_interval(secs => :secs) "
                    "RETURNING id"
                ),
                {"secs": HEARTBEAT_STALE_SECONDS},
            )
            n = len(res.fetchall())
            db.commit()
            return n
        finally:
            db.close()

    # ---- LISTEN/NOTIFY -----------------------------------------------

    async def _listen_loop(self) -> None:
        """Subscribe to pg_notify('enrichment_wake') so new jobs wake us
        instantly instead of waiting for the safety poll. Uses a raw
        psycopg2 connection in a thread because asyncio + LISTEN don't
        play nicely with SQLAlchemy's pool.
        """
        try:
            await asyncio.to_thread(self._listen_blocking)
        except asyncio.CancelledError:
            return
        except Exception:
            log.exception("listen loop crashed; falling back to poll-only")

    def _listen_blocking(self) -> None:
        import select
        import psycopg2
        from dsl_api.config import settings

        while not self._stop.is_set():
            try:
                dsn = settings.DATABASE_URL
                conn = psycopg2.connect(dsn)
                conn.set_isolation_level(0)  # AUTOCOMMIT
                cur = conn.cursor()
                cur.execute("LISTEN enrichment_wake;")
                while not self._stop.is_set():
                    if select.select([conn], [], [], 30) == ([], [], []):
                        continue
                    conn.poll()
                    while conn.notifies:
                        conn.notifies.pop(0)
                        self.wake()
            except Exception:
                log.exception("LISTEN connection dropped; reconnecting in 5s")
                time.sleep(5)


# ---------------------------------------------------------------------------
# Module-level singleton + helpers used by app.lifespan and endpoints.
# ---------------------------------------------------------------------------


_COORDINATOR: Optional[Coordinator] = None


def get_coordinator() -> Coordinator:
    global _COORDINATOR
    if _COORDINATOR is None:
        _COORDINATOR = Coordinator()
    return _COORDINATOR


def notify_new_work(db: Session) -> None:
    """pg_notify so the coordinator wakes immediately. Wrap any
    job-creation transaction with this (after commit)."""
    try:
        db.execute(sa_text("NOTIFY enrichment_wake"))
        db.commit()
    except Exception:
        log.debug("NOTIFY enrichment_wake failed; safety poll will catch it")


def task_id_short(tid: str) -> str:
    """Stable short handle for an in-flight task — used in SSE payloads
    so the FE can pin a UI row to a task without showing a full UUID."""
    return tid[:8]
