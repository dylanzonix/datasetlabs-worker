"""Background task lifecycle for slow chat tools.

Backs `wait=false` on `table_create`, `table_extend`, `enrichment_run`
(and any future slow tool). The shape:

    bg_task = spawn(handler=enrichment_run, args={...}, ctx=ctx,
                    kind="enrichment_run", task_key=enrichment_id)
    → returns {status: "running", task_id: "bt3", ...} immediately.

The agent monitors via `task_status` (instant peek) and `task_wait`
(blocks on the underlying asyncio.Task until done or timeout). The DB
row in chat_background_tasks is the durable record — survives a worker
restart in terms of *visibility* (the row still exists with status=running
even if the asyncio.Task is gone, so a sweeper would mark zombies as
'cancelled' on startup; not built yet).

Cost flow: the spawned handler still returns (result_dict, cost_usd)
like any other tool. The wrapper around it settles credits via
`_charge_run_credits` once the task completes (mirrors the REST flow
in routes_actions.py:233-247). No double-charging because the
synchronous agent loop didn't add this tool's cost — it only added the
trivial "started" return.
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple

from sqlalchemy import text as sa_text
from sqlalchemy.orm import Session

from dsl_api.db import SessionLocal


log = logging.getLogger(__name__)


# Handler signature: same as every other tool — async (args, ctx) -> (result, cost_usd).
HandlerFn = Callable[
    [Dict[str, Any], Any],  # Any = ToolContext (forward ref to avoid cycle)
    Awaitable[Tuple[Dict[str, Any], float]],
]


@dataclass
class BackgroundTask:
    """In-process handle to a spawned asyncio.Task.

    The DB row is the durable state. This object is the live handle so
    task_wait can `asyncio.wait()` on it and the run-level cancel can
    `task.cancel()` it. Cleaned out of REGISTRY when the task completes.
    """

    id: str  # uuid str
    short_id: str  # bt1, bt2, ...
    project_id: str
    run_id: Optional[str]
    user_id: str
    kind: str  # tool name
    task_key: Optional[str]  # entity id (table_id / enrichment_id) the task is operating on
    args: Dict[str, Any]
    task: asyncio.Task = field(repr=False)
    done_event: asyncio.Event = field(repr=False)


class BackgroundTaskRegistry:
    """Project-scoped registry of in-process bg tasks.

    Keyed by task UUID. The DB row outlives this entry — the registry
    only tracks what's still actively running in THIS worker process.
    """

    def __init__(self) -> None:
        self._tasks: Dict[str, BackgroundTask] = {}
        self._lock = asyncio.Lock()

    async def register(self, bg: BackgroundTask) -> None:
        async with self._lock:
            self._tasks[bg.id] = bg

    async def unregister(self, task_id: str) -> None:
        async with self._lock:
            self._tasks.pop(task_id, None)

    async def lookup_by_short_or_uuid(
        self, project_id: str, key: str
    ) -> Optional[BackgroundTask]:
        """Look up a task by either its short_id (bt3) or its uuid."""
        async with self._lock:
            for bg in self._tasks.values():
                if bg.project_id != project_id:
                    continue
                if bg.id == key or bg.short_id == key:
                    return bg
            return None

    async def snapshot_for_run(
        self, run_id: str
    ) -> List[BackgroundTask]:
        async with self._lock:
            return [bg for bg in self._tasks.values() if bg.run_id == run_id]

    async def cancel_run(self, run_id: str) -> int:
        """Cancel all tasks tied to a chat run. Used when the user hits
        Stop — every spawned background task for that turn is cancelled
        so spend doesn't keep accruing after the user said stop."""
        bgs = await self.snapshot_for_run(run_id)
        n = 0
        for bg in bgs:
            if not bg.task.done():
                bg.task.cancel()
                n += 1
        return n


REGISTRY = BackgroundTaskRegistry()


def _next_bt_short_id(db: Session, project_id: str) -> str:
    """Next 'bt<N>' for this project. Per-project advisory lock keyed
    salt 3 (tables=1, enrichments=2, version_id=0) so parallel spawns
    can't both pick the same short_id and trip the unique index."""
    db.execute(
        sa_text("SELECT pg_advisory_xact_lock(hashtextextended(:pid, 3))"),
        {"pid": str(project_id)},
    )
    row = db.execute(
        sa_text(
            "SELECT short_id FROM chat_background_tasks WHERE project_id=:pid "
            "ORDER BY (CASE WHEN short_id ~ '^bt[0-9]+$' THEN CAST(substring(short_id, 3) AS int) ELSE 0 END) DESC LIMIT 1"
        ),
        {"pid": project_id},
    ).fetchone()
    if not row or not row[0] or not row[0].startswith("bt"):
        return "bt1"
    try:
        return f"bt{int(row[0][2:]) + 1}"
    except (ValueError, IndexError):
        return "bt1"


def _emit_run_event(
    db: Session, run_id: Optional[str], etype: str, payload: Dict[str, Any]
) -> None:
    """Fire-and-forget SSE event emit. Opens its own ChatRun lookup
    because the caller's session may be in a state we can't trust."""
    if not run_id:
        return
    try:
        from dsl_worker.chat import run_state
        from dsl_api.models import ChatRun

        run_obj = db.query(ChatRun).filter(ChatRun.id == run_id).first()
        if run_obj is not None:
            run_state.emit_event(db, run_obj, etype, payload)
    except Exception:
        log.exception("%s emit failed for run=%s", etype, run_id)


async def spawn(
    *,
    handler: HandlerFn,
    args: Dict[str, Any],
    ctx: Any,  # ToolContext (forward ref)
    kind: str,
    task_key: Optional[str] = None,
    summary: str = "",
) -> Dict[str, Any]:
    """Spawn a background task that runs `handler(args, ctx_clone)`.

    Returns the {status, task_id, ...} payload the agent reads
    *immediately* — the actual work runs in an asyncio.Task that
    settles credits + updates the DB row on completion.

    Cancellation:
      - The asyncio.Task is registered in REGISTRY and CANCELS
        (legacy registry, keyed by (project_id, task_uuid)). Either
        path cancels it.
      - On cancel, the handler typically raises CancelledError; we
        capture partial cost from `task_ctx.partial_cost_usd`, mark
        the row 'cancelled', and emit a `background_task_done` event.

    Errors:
      - Any non-cancel exception inside the handler is caught, logged,
        and the row is marked 'error' with a truncated message. The
        SSE event still fires so the FE clears its running indicator.
    """
    task_id = str(uuid.uuid4())
    short_id = _next_bt_short_id(ctx.db, ctx.project_id)
    ctx.db.execute(
        sa_text(
            """
            INSERT INTO chat_background_tasks (
                id, project_id, run_id, user_id, kind, task_key, short_id,
                status, args, started_at
            ) VALUES (
                :id, :pid, :rid, :uid, :kind, :tk, :sid,
                'running', CAST(:args AS jsonb), now()
            )
            """
        ),
        {
            "id": task_id,
            "pid": ctx.project_id,
            "rid": ctx.run_id,
            "uid": ctx.user_id,
            "kind": kind,
            "tk": task_key,
            "sid": short_id,
            "args": json.dumps(args, default=str),
        },
    )
    ctx.db.commit()

    done_event = asyncio.Event()

    async def _runner() -> None:
        # Fresh session per background task so it doesn't share state
        # with the caller's session (which may close immediately after
        # spawn returns). ToolContext clone gets its own partial_cost
        # so cancel can attribute spend correctly.
        from dsl_worker.chat.tools import ToolContext as _Ctx

        task_db = SessionLocal()
        task_ctx = _Ctx(
            db=task_db,
            project_id=ctx.project_id,
            user_id=ctx.user_id,
            run_id=ctx.run_id,
            emit_progress=ctx.emit_progress,
            emit_event=None,
            cancel_event=ctx.cancel_event,
            partial_cost_usd=0.0,
        )
        result: Optional[Dict[str, Any]] = None
        err: Optional[str] = None
        cost_usd = 0.0
        final_status = "complete"
        try:
            result, cost_usd = await handler(args, task_ctx)
        except asyncio.CancelledError:
            final_status = "cancelled"
            # Capture any partial spend that accrued before the cancel
            # landed (e.g. an apify actor that was aborted mid-poll
            # still bills for compute units burned).
            cost_usd = float(task_ctx.partial_cost_usd or 0.0)
            raise
        except Exception as e:
            log.exception("background task %s (%s) raised: %s", task_id, kind, e)
            final_status = "error"
            err = str(e)[:2000]
            try:
                task_db.rollback()
            except Exception:
                pass
        finally:
            # Separate session for the final UPDATE because task_db may
            # be in an aborted-tx state after a handler exception.
            settle_db = SessionLocal()
            try:
                # cost_credits column stores credits (1 credit = $0.10).
                cost_credits = float(cost_usd) * 10.0 if cost_usd else 0.0
                settle_db.execute(
                    sa_text(
                        """
                        UPDATE chat_background_tasks
                        SET status = :status,
                            result = CAST(:result AS jsonb),
                            error = :error,
                            cost_credits = :cost,
                            finished_at = now()
                        WHERE id = :id
                        """
                    ),
                    {
                        "status": final_status,
                        "result": json.dumps(result, default=str) if result is not None else None,
                        "error": err,
                        "cost": cost_credits,
                        "id": task_id,
                    },
                )
                settle_db.commit()

                # Settle credits — same code path as the REST enrichment
                # endpoint. _charge_run_credits is a no-op for spend<=0.
                if cost_usd and cost_usd > 0:
                    try:
                        from dsl_worker.chat.runs import _charge_run_credits

                        spend_cents = int(round(cost_usd * 100))
                        if spend_cents > 0:
                            _charge_run_credits(
                                settle_db,
                                ctx.user_id,
                                spend_cents,
                                ctx.project_id,
                                reason=f"bg_{kind}",
                            )
                            settle_db.commit()
                    except Exception:
                        log.exception(
                            "background task %s credit settle failed", task_id
                        )
                        try:
                            settle_db.rollback()
                        except Exception:
                            pass

                _emit_run_event(
                    settle_db,
                    ctx.run_id,
                    "background_task_done",
                    {
                        "task_id": short_id,
                        "task_uuid": task_id,
                        "kind": kind,
                        "task_key": task_key,
                        "status": final_status,
                        "cost_credits": cost_credits,
                        "error": err,
                    },
                )
            except Exception:
                log.exception("background task %s settle failed", task_id)
            finally:
                settle_db.close()
                try:
                    task_db.close()
                except Exception:
                    pass
                done_event.set()
                await REGISTRY.unregister(task_id)

    task = asyncio.create_task(_runner(), name=f"bg-{kind}-{short_id}")
    bg = BackgroundTask(
        id=task_id,
        short_id=short_id,
        project_id=ctx.project_id,
        run_id=ctx.run_id,
        user_id=ctx.user_id,
        kind=kind,
        task_key=task_key,
        args=args,
        task=task,
        done_event=done_event,
    )
    await REGISTRY.register(bg)

    # Started event so the FE can render a "task running" chip
    # immediately, before the handler does any work.
    _emit_run_event(
        ctx.db,
        ctx.run_id,
        "background_task_started",
        {
            "task_id": short_id,
            "task_uuid": task_id,
            "kind": kind,
            "task_key": task_key,
            "summary": summary,
        },
    )

    return {
        "status": "running",
        "task_id": short_id,
        "task_uuid": task_id,
        "kind": kind,
        "summary": summary,
    }


def read_task_rows(
    db: Session, project_id: str, ids: List[str]
) -> List[Dict[str, Any]]:
    """Read current state of one or more tasks. `ids` may be a mix of
    short_ids (bt1) and uuids. Caller passes its own session."""
    if not ids:
        return []
    rows = db.execute(
        sa_text(
            """
            SELECT id::text, short_id, kind, task_key, status,
                   cost_credits, partial_cost_credits,
                   started_at, finished_at, result, error
            FROM chat_background_tasks
            WHERE project_id = :pid
              AND (short_id = ANY(:keys) OR id::text = ANY(:keys))
            """
        ),
        {"pid": project_id, "keys": ids},
    ).fetchall()
    out: List[Dict[str, Any]] = []
    for r in rows:
        (tid, sid, kind, tk, status, cost, partial,
         started_at, finished_at, result, error) = r
        out.append({
            "task_id": sid,
            "task_uuid": tid,
            "kind": kind,
            "task_key": tk,
            "status": status,
            "cost_credits": float(cost) if cost is not None else None,
            "partial_cost_credits": float(partial) if partial is not None else 0.0,
            "started_at": started_at.isoformat() if started_at else None,
            "finished_at": finished_at.isoformat() if finished_at else None,
            "result": result,
            "error": error,
        })
    return out


def list_running_rows(
    db: Session, project_id: str
) -> List[Dict[str, Any]]:
    """All tasks currently 'running' for this project — backs the
    /background-tasks REST endpoint for FE refresh-recovery."""
    rows = db.execute(
        sa_text(
            """
            SELECT id::text, short_id, kind, task_key, started_at,
                   partial_cost_credits, run_id::text
            FROM chat_background_tasks
            WHERE project_id = :pid AND status = 'running'
            ORDER BY started_at ASC
            """
        ),
        {"pid": project_id},
    ).fetchall()
    out: List[Dict[str, Any]] = []
    for r in rows:
        (tid, sid, kind, tk, started_at, partial, rid) = r
        out.append({
            "task_id": sid,
            "task_uuid": tid,
            "kind": kind,
            "task_key": tk,
            "started_at": started_at.isoformat() if started_at else None,
            "partial_cost_credits": float(partial) if partial is not None else 0.0,
            "run_id": rid,
        })
    return out


async def wait_inproc(
    project_id: str,
    ids: List[str],
    timeout_s: Optional[float] = None,
    mode: str = "all",
) -> None:
    """Await the in-process asyncio.Tasks for these ids until the chosen
    condition (all done OR first done) holds, or until timeout. Tasks
    not in REGISTRY (already finished + unregistered) are skipped — the
    caller re-queries the DB afterwards to read final state.
    """
    bgs: List[BackgroundTask] = []
    for key in ids or []:
        bg = await REGISTRY.lookup_by_short_or_uuid(project_id, key)
        if bg is not None:
            bgs.append(bg)
    if not bgs:
        return
    tasks = [bg.task for bg in bgs]
    return_when = (
        asyncio.ALL_COMPLETED if mode != "any" else asyncio.FIRST_COMPLETED
    )
    # asyncio.wait swallows exceptions in the gathered tasks (they
    # propagate via task.exception() / await task). We just want to
    # block until the condition holds, so swallow is fine here.
    await asyncio.wait(tasks, timeout=timeout_s, return_when=return_when)


# ---------------------------------------------------------------------------
# Tool handlers — task_status / task_wait
# ---------------------------------------------------------------------------


def _summarize_result(result: Any) -> Optional[str]:
    """Short preview of a task's result that's safe to surface back to
    the model — full payload is in `result` already, this is just so
    the agent's task_status output isn't a wall of JSON when there are
    many tasks."""
    if result is None:
        return None
    try:
        s = json.dumps(result, default=str)
    except Exception:
        return None
    return s if len(s) <= 300 else s[:300] + "…"


async def task_status(
    args: Dict[str, Any], ctx: Any
) -> Tuple[Dict[str, Any], float]:
    """Instant peek at one or more background tasks.

    Args: {task_ids: ['bt1', 'bt2', ...]}. ids may be short_ids or uuids.
    Returns per-task current state — status, cost_credits (settled if
    done, partial if running), error, and a short result preview.
    """
    raw_ids = args.get("task_ids") or args.get("ids") or []
    if isinstance(raw_ids, str):
        raw_ids = [raw_ids]
    ids = [str(t).strip() for t in raw_ids if t]
    if not ids:
        return {"error": "task_ids is required (list of bt<N> short_ids or uuids)"}, 0.0

    rows = read_task_rows(ctx.db, ctx.project_id, ids)
    found_keys = {r["task_id"] for r in rows} | {r["task_uuid"] for r in rows}
    missing = [t for t in ids if t not in found_keys]

    tasks_out = []
    for r in rows:
        tasks_out.append({
            "task_id": r["task_id"],
            "kind": r["kind"],
            "task_key": r["task_key"],
            "status": r["status"],
            "cost_credits": (
                r["cost_credits"]
                if r["cost_credits"] is not None
                else r["partial_cost_credits"]
            ),
            "started_at": r["started_at"],
            "finished_at": r["finished_at"],
            "result_preview": _summarize_result(r["result"]),
            "error": r["error"],
        })
    return {
        "tasks": tasks_out,
        "not_found": missing,
    }, 0.0


async def task_wait(
    args: Dict[str, Any], ctx: Any
) -> Tuple[Dict[str, Any], float]:
    """Block until the chosen condition over a set of tasks holds, or
    until timeout. Args:
      task_ids: ['bt1', 'bt2', ...]   (short_ids or uuids)
      mode: 'all' (default) | 'any'   ('all' = wait until every task is done;
                                       'any' = return as soon as one finishes)
      timeout_s: number?              (default 300s; max 600s — agent should
                                       re-call rather than wait indefinitely)

    Returns the SAME shape as task_status — the agent reads per-task state
    after the wait completes. Tasks that finished BEFORE this tool was
    called are returned immediately with their final state (read from DB).
    """
    raw_ids = args.get("task_ids") or args.get("ids") or []
    if isinstance(raw_ids, str):
        raw_ids = [raw_ids]
    ids = [str(t).strip() for t in raw_ids if t]
    if not ids:
        return {"error": "task_ids is required (list of bt<N> short_ids or uuids)"}, 0.0

    mode = (args.get("mode") or "all").lower()
    if mode not in ("all", "any"):
        return {"error": f"mode must be 'all' or 'any'; got {mode!r}"}, 0.0

    timeout_in = args.get("timeout_s")
    if timeout_in is None:
        timeout_s: Optional[float] = 300.0
    else:
        try:
            timeout_s = float(timeout_in)
        except (TypeError, ValueError):
            return {"error": "timeout_s must be a number (seconds)"}, 0.0
        if timeout_s < 0:
            return {"error": "timeout_s must be >= 0"}, 0.0
        # Cap at 10 minutes so the agent doesn't sit indefinitely on a
        # zombie task that never finishes.
        timeout_s = min(timeout_s, 600.0)

    await wait_inproc(ctx.project_id, ids, timeout_s=timeout_s, mode=mode)
    # Re-read DB state for the canonical answer. Tasks that finished
    # while we waited will already have rows in the right status.
    rows = read_task_rows(ctx.db, ctx.project_id, ids)
    found_keys = {r["task_id"] for r in rows} | {r["task_uuid"] for r in rows}
    missing = [t for t in ids if t not in found_keys]

    all_done = all(r["status"] != "running" for r in rows) if rows else True
    any_done = any(r["status"] != "running" for r in rows) if rows else False
    timed_out = (mode == "all" and not all_done) or (mode == "any" and not any_done)

    tasks_out = []
    for r in rows:
        tasks_out.append({
            "task_id": r["task_id"],
            "kind": r["kind"],
            "task_key": r["task_key"],
            "status": r["status"],
            "cost_credits": (
                r["cost_credits"]
                if r["cost_credits"] is not None
                else r["partial_cost_credits"]
            ),
            "started_at": r["started_at"],
            "finished_at": r["finished_at"],
            "result_preview": _summarize_result(r["result"]),
            "error": r["error"],
        })
    return {
        "tasks": tasks_out,
        "not_found": missing,
        "mode": mode,
        "timeout_s": timeout_s,
        "timed_out": timed_out,
        "all_done": all_done,
    }, 0.0


HANDLERS: Dict[str, Any] = {
    "task_status": task_status,
    "task_wait": task_wait,
}
