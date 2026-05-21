"""Phase-level timing instrumentation for chat runs.

`emit_event` / `emit_events_batch` already stamp `mono_ns` on every event,
so any two events in the same run can be compared for elapsed wall time
without trusting `created_at`. This module adds *intentional* phase
markers on top:

  - `phase_marker(phase, **meta)` writes a single `phase` event whose
    payload includes `phase` (dotted path like `enrichment_run/setup`)
    and any kwargs. Cheap.
  - `phase_span(phase, **meta)` is the context-manager form: emits a
    `phase_start` on enter and a `phase_end` on exit, with `dur_ms` and
    `error` set on the end event.

Both are best-effort and never raise — instrumentation never breaks
the work it's measuring.

To analyze a run, use `scripts/analyze_run.py <run_id>` (sister file).
"""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager, contextmanager
from typing import Any, Dict, Optional


log = logging.getLogger(__name__)


def _emit_phase(ctx: Any, event_type: str, payload: Dict[str, Any]) -> None:
    """Best-effort emit through run_state.emit_event. Looks up the run
    from ctx.run_id + ctx.db so callers don't have to thread the run obj
    through every layer."""
    run_id = getattr(ctx, "run_id", None)
    if not run_id:
        return
    try:
        from dsl_worker.chat import run_state
        from dsl_api.models import ChatRun
        run_obj = ctx.db.query(ChatRun).filter(ChatRun.id == run_id).first()
        if run_obj is not None:
            run_state.emit_event(ctx.db, run_obj, event_type, payload)
    except Exception:
        log.debug("phase emit %s failed; continuing", event_type, exc_info=True)


def phase_marker(ctx: Any, phase: str, **meta: Any) -> None:
    """Emit a one-shot phase marker — a `phase` event with the given path
    and metadata. Use for single milestones ("setup_done", "first_cell_ready")
    where you don't have an enclosing block to time.
    """
    payload: Dict[str, Any] = {"phase": phase, **meta}
    _emit_phase(ctx, "phase", payload)


@contextmanager
def phase_span(ctx: Any, phase: str, **meta: Any):
    """Sync context manager. Emits `phase_start` on enter and `phase_end`
    with `dur_ms` on exit. Exceptions are recorded as `error=<str>` on the
    end event but not suppressed.

    Use inside synchronous code (the bulk of tool handlers run on the
    asyncio thread but call sync DB code through SQLAlchemy sessions —
    that's still fine to wrap with this).
    """
    t0 = time.perf_counter_ns()
    _emit_phase(ctx, "phase_start", {"phase": phase, **meta})
    err: Optional[BaseException] = None
    try:
        yield
    except BaseException as e:
        err = e
        raise
    finally:
        dur_ms = (time.perf_counter_ns() - t0) / 1_000_000
        end_payload: Dict[str, Any] = {"phase": phase, "dur_ms": round(dur_ms, 2), **meta}
        if err is not None:
            end_payload["error"] = type(err).__name__
        _emit_phase(ctx, "phase_end", end_payload)


@asynccontextmanager
async def phase_span_async(ctx: Any, phase: str, **meta: Any):
    """Async context manager. Same shape as phase_span but for `async with`."""
    t0 = time.perf_counter_ns()
    _emit_phase(ctx, "phase_start", {"phase": phase, **meta})
    err: Optional[BaseException] = None
    try:
        yield
    except BaseException as e:
        err = e
        raise
    finally:
        dur_ms = (time.perf_counter_ns() - t0) / 1_000_000
        end_payload: Dict[str, Any] = {"phase": phase, "dur_ms": round(dur_ms, 2), **meta}
        if err is not None:
            end_payload["error"] = type(err).__name__
        _emit_phase(ctx, "phase_end", end_payload)


@contextmanager
def time_commit(ctx: Any, label: str, threshold_ms: float = 100.0):
    """Wrap a `ctx.db.commit()` (or any sync block) and emit a `slow_commit`
    event if it took longer than `threshold_ms`. Cheap default — most
    commits are <10 ms and emit nothing.

    Usage:
        with time_commit(ctx, "samples_upsert"):
            ctx.db.execute(stmt, params)
            ctx.db.commit()
    """
    t0 = time.perf_counter_ns()
    try:
        yield
    finally:
        dur_ms = (time.perf_counter_ns() - t0) / 1_000_000
        if dur_ms >= threshold_ms:
            _emit_phase(ctx, "slow_commit", {"label": label, "dur_ms": round(dur_ms, 2)})
