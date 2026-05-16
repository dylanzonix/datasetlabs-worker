"""Durable chat run lifecycle for chat_v2.

Mirrors legacy `dsl_worker/chat_api/runs.py:start_run` + `_run_agent_task`
but dispatches to the chat_v2 Responses-API agent. Reuses the legacy
primitives for everything else:

  - chat_runs / chat_run_events tables (durable state)
  - per-project lock + global semaphore (concurrency)
  - emit_event / mark_run_completed / _mark_run_failed (lifecycle)
  - tail_events (SSE replay + live tail + heartbeat + reattach)

Net effect: a turn runs as a background asyncio task that survives
client disconnect; the FE can reattach on refresh via the existing
chat_runs URLs. Assistant message + tool_log + cost are persisted at
run completion regardless of SSE connection state.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any, Dict, List, Optional
from uuid import UUID

from sqlalchemy import text as sa_text
from sqlalchemy.sql import func

from dsl_api.db import SessionLocal
from dsl_api.models import ChatMessage, ChatRun, Project
from dsl_api.models.balance_ledger import BalanceLedger
from dsl_api.models.chat_run import (
    RUN_STATUS_CANCELLED,
    RUN_STATUS_FAILED,
    RUN_STATUS_PAUSED,
    RUN_STATUS_QUEUED,
    RUN_STATUS_RUNNING,
    RUN_TERMINAL_STATUSES,
)

# Legacy primitives we reuse wholesale.
from dsl_worker.chat_api import runs as legacy_runs


log = logging.getLogger(__name__)


# Registry of in-flight asyncio.Task objects keyed by run_id. Populated
# by start_v2_run, drained by _run_chat_v2_task's finally, read by
# cancel_v2_run. Lives only inside the worker process — if the worker
# restarts, the orphan reaper handles stale runs. The registry is what
# makes cancellation instantaneous: cancel_v2_run does task.cancel(),
# which raises CancelledError into the agent's current await on the
# next event-loop tick, and the loop's CancelledError path persists
# whatever cost was already incurred before exiting.
_active_tasks: Dict[UUID, asyncio.Task] = {}


# Per-run queue of user messages injected via the inject endpoint while
# the agent's turn is mid-flight. The agent loop drains this at each
# iteration boundary (right before its next LLM call) and prepends the
# messages as user-role input items so the model integrates them into
# its next decision. Survives only within the worker process; if the
# worker restarts mid-turn the queue is lost (the persisted ChatMessage
# rows still record the user's intent for the next run).
_pending_injections: Dict[UUID, List[Dict[str, str]]] = {}


def inject_message(
    run_id: UUID,
    project_id: UUID,
    user_id: UUID,
    content: str,
) -> Optional[str]:
    """Inject a user message into a running v2 turn. Returns the
    injection id (also the persisted ChatMessage.id) so the FE can
    dedupe its optimistic balloon against the SSE echo.

    Validates the run exists, belongs to the caller, and is still
    running. Persists a ChatMessage with `run_id` set + a marker in
    applied_changes so a refresh re-renders the inline injection in
    the right turn. Pushes onto the in-memory queue for the live
    agent loop to consume."""
    content = (content or "").strip()
    if not content:
        return None

    db = SessionLocal()
    try:
        run = (
            db.query(ChatRun)
            .filter(ChatRun.id == run_id, ChatRun.project_id == project_id)
            .first()
        )
        if run is None or run.status in RUN_TERMINAL_STATUSES:
            return None
        # Persist the injection as a ChatMessage so a refresh during
        # the same turn renders it back. mapHistoryMessage will sort it
        # into the right assistant bubble using `applied_changes.inject_target`.
        injected_msg = ChatMessage(
            project_id=project_id,
            role="user",
            content=content,
            run_id=run_id,
            applied_changes={"injected": True, "run_id": str(run_id)},
        )
        db.add(injected_msg)
        db.flush()
        msg_id = str(injected_msg.id)
        db.commit()

        # Echo immediately to any connected SSE subscriber so the
        # injecting client sees the persisted id (dedupe target) and
        # other tabs see the new balloon without polling.
        try:
            legacy_runs.emit_event(db, run, "user_injection", {
                "id": msg_id,
                "content": content,
            })
            db.commit()
        except Exception:
            log.exception("user_injection emit failed for %s", run_id)

        _pending_injections.setdefault(run_id, []).append({
            "id": msg_id,
            "content": content,
        })
        return msg_id
    finally:
        db.close()


def _drain_injections(run_id: UUID) -> List[Dict[str, str]]:
    """Atomically pop all pending injections for a run. Called by the
    agent loop at each iteration boundary. Empty list if none pending."""
    queue = _pending_injections.get(run_id)
    if not queue:
        return []
    items = list(queue)
    queue.clear()
    return items


_DEFAULT_PROJECT_NAMES = {"New Dataset", "Untitled", "", None}


async def _auto_name_project(project_id: UUID, user_content: str) -> None:
    """One-shot mini LLM call to generate a 3-5 word project name from the
    first message. Subsidized — bypasses TrackedOpenAIClient + balance_ledger
    on purpose; this isn't user-facing OpenAI usage worth charging for.
    Only runs if the project is still on the default name; idempotent for
    subsequent turns."""
    import os
    db = SessionLocal()
    try:
        proj = db.query(Project).filter(Project.id == project_id).first()
        if proj is None or proj.name not in _DEFAULT_PROJECT_NAMES:
            return
    finally:
        db.close()

    try:
        from openai import AsyncOpenAI
        client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        model = os.getenv("OPENAI_MODEL_MINI", "gpt-5.4-mini")
        resp = await client.responses.create(
            model=model,
            input=(
                "Generate a 3-5 word project name for this dataset request. "
                "Title Case. No quotes, no punctuation. Just the name.\n\n"
                f"Request: {user_content[:400]}"
            ),
        )
        name = (resp.output_text or "").strip().strip('"').strip("'").splitlines()[0][:80]
        if not name:
            return
    except Exception:
        log.exception("auto-name LLM call failed for project %s", project_id)
        return

    db = SessionLocal()
    try:
        db.execute(
            sa_text(
                "UPDATE projects SET name=:n, updated_at=now() "
                "WHERE id=:p AND name = ANY(:defaults)"
            ),
            {"n": name, "p": str(project_id), "defaults": ["New Dataset", "Untitled", ""]},
        )
        db.commit()
    except Exception:
        log.exception("auto-name UPDATE failed for project %s", project_id)
    finally:
        db.close()


async def start_v2_run(
    project_id: UUID,
    user_id: UUID,
    user_content: str,
) -> ChatRun:
    """Create a ChatRun + user ChatMessage; fire the background task.

    Returns the detached ChatRun row. Identical signature/contract to
    legacy `runs.start_run` so the route can substitute it.
    """
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

    task = asyncio.create_task(
        _run_chat_v2_task(run_id, user_id, user_content),
        name=f"chat-v2-run-{run_id}",
    )
    _active_tasks[run_id] = task

    # Fire-and-forget project naming. Independent of the orchestrator
    # so a slow LLM here can't block the run. Only names projects that
    # are still on the default name; subsequent turns won't re-trigger.
    asyncio.create_task(
        _auto_name_project(project_id, user_content),
        name=f"chat-v2-name-{project_id}",
    )

    # Re-fetch detached so the caller can read fields without lazy-load.
    db = SessionLocal()
    try:
        run = db.query(ChatRun).filter(ChatRun.id == run_id).first()
        db.expunge(run)  # type: ignore[arg-type]
        return run  # type: ignore[return-value]
    finally:
        db.close()


async def _run_chat_v2_task(
    run_id: UUID,
    user_id: UUID,
    user_content: str,
) -> None:
    """Background coroutine for one chat_v2 run.

    Acquires the same global semaphore + per-project lock the legacy
    runs use, so chat_v2 turns and legacy turns serialize correctly on
    the same project.
    """
    try:
        # Resolve project + early-exit on cancel.
        db = SessionLocal()
        try:
            run = db.query(ChatRun).filter(ChatRun.id == run_id).first()
            if run is None:
                log.warning("v2 run %s vanished before start", run_id)
                return
            project_id = run.project_id
            if run.status == RUN_STATUS_CANCELLED:
                return
        finally:
            db.close()

        async with legacy_runs._global_run_semaphore():
            async with legacy_runs._project_lock(project_id):
                db = SessionLocal()
                try:
                    run = db.query(ChatRun).filter(ChatRun.id == run_id).first()
                    if run is None or run.status in RUN_TERMINAL_STATUSES:
                        return
                    run.status = RUN_STATUS_RUNNING
                    run.started_at = func.now()
                    db.commit()
                finally:
                    db.close()

                heartbeat = asyncio.create_task(
                    _heartbeat_loop(run_id),
                    name=f"chat-v2-heartbeat-{run_id}",
                )
                try:
                    await _drive_agent(run_id, user_id, project_id, user_content)
                except asyncio.CancelledError:
                    # _drive_agent already persisted partial state + marked
                    # the run cancelled before re-raising. Swallow here so
                    # the outer task ends normally and the registry pop in
                    # the outermost finally runs cleanly.
                    log.info("v2 run %s cancelled cleanly", run_id)
                except Exception as e:
                    log.exception("v2 run %s crashed", run_id)
                    legacy_runs._mark_run_failed(run_id, str(e)[:500])
                finally:
                    heartbeat.cancel()
                    try:
                        await heartbeat
                    except (asyncio.CancelledError, Exception):
                        pass
    finally:
        _active_tasks.pop(run_id, None)
        _pending_injections.pop(run_id, None)


def cancel_v2_run(run_id: UUID) -> bool:
    """Instantaneous cancel for a chat_v2 run.

    Sends CancelledError into the task's current await. The agent's
    CancelledError handler in _drive_agent flushes whatever cost was
    incurred so far to the balance ledger and persists a partial
    assistant message + cancelled-status event before exiting — so the
    user can't dodge billing by cancelling mid-call.

    Falls back to a DB-only status flip via legacy.request_cancel if
    the registry has no task (worker restarted between start and cancel,
    or the run already finished). In that case the orphan reaper picks
    up the stale state.
    """
    task = _active_tasks.get(run_id)
    if task is not None and not task.done():
        task.cancel()
        return True
    return legacy_runs.request_cancel(run_id)


async def _heartbeat_loop(run_id: UUID) -> None:
    """Emit a heartbeat event every 30s so the staleness sweeper never
    false-positives an actively-running task. Cancelled by the caller
    when the run finishes — until then, this is the proof of life."""
    while True:
        try:
            await asyncio.sleep(30)
        except asyncio.CancelledError:
            return
        db = SessionLocal()
        try:
            run = db.query(ChatRun).filter(ChatRun.id == run_id).first()
            if run is None or run.status in RUN_TERMINAL_STATUSES:
                return
            legacy_runs.emit_event(db, run, "heartbeat", {})
            db.commit()
        except Exception:
            log.debug("heartbeat emit failed; continuing", exc_info=True)
        finally:
            db.close()


async def _drive_agent(
    run_id: UUID,
    user_id: UUID,
    project_id: UUID,
    user_content: str,
) -> None:
    """The actual chat_v2 turn: load history, call run_turn with an
    event callback that fans into chat_run_events, persist assistant
    message + ledger entry, mark run completed."""
    # Late import to keep this module light on import.
    from dsl_worker.chat_v2.agent import run_turn

    started_monotonic = time.monotonic()

    db = SessionLocal()
    try:
        run = db.query(ChatRun).filter(ChatRun.id == run_id).first()
        if run is None:
            return

        # Load conversation history (prior user + assistant messages).
        # Exclude the run's triggering message — agent receives it as
        # the new user message in run_turn.
        prior = (
            db.query(ChatMessage)
            .filter(
                ChatMessage.project_id == project_id,
                ChatMessage.id != run.triggering_message_id,
                ChatMessage.role.in_(["user", "assistant"]),
            )
            .order_by(ChatMessage.created_at.asc())
            .all()
        )
        history: List[Dict[str, str]] = [
            {"role": m.role, "content": m.content or ""} for m in prior if m.content
        ]

        # Emit an early "status" event so the FE shows a thinking
        # indicator immediately, before the LLM call returns its first
        # reasoning token.
        legacy_runs.emit_event(db, run, "status", {"content": "Thinking…"})

        tool_log: List[Dict[str, Any]] = []
        total_cost_ref = {"value": 0.0}
        final_text_ref = {"value": ""}

        async def on_event(evt: Dict[str, Any]) -> None:
            etype = evt.get("type")
            # Open a fresh session per emit since this callback may fire
            # while the outer `db` session is mid-statement on the LLM
            # call. Legacy emit_event needs a clean session.
            ldb = SessionLocal()
            try:
                lrun = ldb.query(ChatRun).filter(ChatRun.id == run_id).first()
                if lrun is None:
                    return
                if etype == "reasoning":
                    # Stream as a thinking delta — FE appends to shimmer.
                    legacy_runs.publish_thinking_delta(lrun.id, evt.get("text") or "")
                elif etype == "tool_call_start":
                    tc_id = evt.get("tool_call_id") or ""
                    name = evt.get("name") or "?"
                    args_preview = json.dumps(evt.get("args") or {}, default=str)[:200]
                    tool_log.append({
                        "id": tc_id,
                        "name": name,
                        "args_preview": args_preview,
                    })
                    legacy_runs.emit_event(ldb, lrun, "tool_call", {
                        "id": tc_id,
                        "name": name,
                        "args_preview": args_preview,
                    })
                elif etype == "tool_call_result":
                    tc_id = evt.get("tool_call_id") or ""
                    summary = evt.get("result_preview") or ""
                    cost = float(evt.get("cost_usd") or 0.0)
                    for entry in tool_log:
                        if entry.get("id") == tc_id:
                            entry["summary"] = summary
                            entry["cost"] = cost
                            break
                    legacy_runs.emit_event(ldb, lrun, "tool_result", {
                        "id": tc_id,
                        "name": evt.get("name"),
                        "summary": summary,
                        "cost": cost,
                    })
                elif etype == "final_message":
                    text = evt.get("text") or ""
                    final_text_ref["value"] = text
                    # Stream the full text via token-delta then persist
                    # via text_checkpoint so a reconnecting subscriber
                    # gets the assistant content even after the live
                    # bus emit is gone.
                    legacy_runs.publish_token_delta(lrun.id, text)
                    legacy_runs.emit_text_checkpoint(ldb, lrun)
                elif etype == "text_segment":
                    # Mid-iteration text from the agent — emit as its
                    # own SSE event so the FE can render it between
                    # tool batches as a discrete segment rather than
                    # appending to the final content blob.
                    legacy_runs.emit_event(ldb, lrun, "text_segment", {
                        "iteration": evt.get("iteration"),
                        "text": evt.get("text") or "",
                    })
                elif etype == "cost_update":
                    # Running cost from agent.py after each LLM call or
                    # tool. Keeps total_cost_ref up to date so the
                    # CancelledError path always has the most recent
                    # total to bill — AND forward to the FE so the
                    # per-turn cost chip's hover tooltip reflects the
                    # live spend while the agent is still working
                    # (matches the user's anti-abuse-pausing ask: cost
                    # visible AS it accrues, not just at the end).
                    new_total = float(
                        evt.get("total_cost_usd") or total_cost_ref["value"]
                    )
                    total_cost_ref["value"] = new_total
                    legacy_runs.emit_event(ldb, lrun, "cost_update", {
                        "total_cost_usd": new_total,
                    })
                elif etype == "turn_complete":
                    total_cost_ref["value"] = float(evt.get("total_cost_usd") or 0.0)
                elif etype == "error":
                    legacy_runs.emit_event(ldb, lrun, "error", {
                        "message": evt.get("message") or "unknown error",
                    })
            except Exception:
                log.exception("v2 on_event failed for %s", etype)
            finally:
                ldb.close()

        cancelled = False
        try:
            result = await run_turn(
                db=db,
                project_id=str(project_id),
                user_id=str(user_id),
                run_id=str(run_id),
                user_message=user_content,
                history=history,
                on_event=on_event,
                drain_injections=lambda: _drain_injections(run_id),
            )
        except asyncio.CancelledError:
            # User cancelled mid-turn. The on_event closure has been
            # accumulating tool_log / total_cost_ref / final_text_ref
            # all along — synthesize a partial result from those so the
            # downstream persist + ledger flush runs the same code path
            # as a normal completion. Re-raise after persisting.
            cancelled = True
            log.info("v2 run %s cancelled mid-turn; flushing partial state", run_id)
            result = {
                "final_message": final_text_ref["value"],
                "tool_calls_made": [],
                "total_cost_usd": total_cost_ref["value"],
                "iterations": None,
            }

        final_text = result.get("final_message") or final_text_ref["value"] or ""
        total_cost_usd = float(result.get("total_cost_usd") or total_cost_ref["value"])
        thinking_duration_s = round(time.monotonic() - started_monotonic, 1)

        # Persist assistant ChatMessage with tool_log + cost.
        ac: Dict[str, Any] = {
            "tool_log": tool_log,
            "total_cost_usd": total_cost_usd,
            "iterations": result.get("iterations"),
            "thinking_duration": thinking_duration_s,
        }
        if result.get("error"):
            ac["error"] = result["error"]
        if cancelled:
            # FE renders an inline "Cancelled." annotation under the
            # assistant body when stop_reason is set, AND the
            # mapHistoryMessage path picks this up so a refresh after
            # cancel still shows it.
            ac["stopped"] = True
            ac["stop_reason"] = "cancel"

        # Capture table_card_added payloads emitted during the run so a
        # page refresh still renders the "table created" chips inline
        # with the assistant message.
        try:
            from dsl_api.models import ChatRunEvent
            cards = (
                db.query(ChatRunEvent.payload)
                .filter(
                    ChatRunEvent.run_id == run_id,
                    ChatRunEvent.type == "table_card_added",
                )
                .order_by(ChatRunEvent.seq.asc())
                .all()
            )
            if cards:
                ac["table_cards"] = [c[0] for c in cards if c[0]]
        except Exception:
            log.exception("collecting table_cards for applied_changes failed")

        assistant_msg = ChatMessage(
            project_id=project_id,
            role="assistant",
            content=final_text,
            run_id=run_id,
            applied_changes=ac,
        )
        db.add(assistant_msg)
        db.flush()

        # Link assistant message id back to the run so the FE / chat
        # history endpoint can pair them.
        run.assistant_message_id = assistant_msg.id

        # Charge balance_ledger. cost is USD; ledger.amount is cents-of-USD
        # (negative = charge). Credit-to-dollar markup happens upstream when
        # users top up — we store raw USD-cents and let billing apply its
        # own pricing.
        spend_cents = int(round(total_cost_usd * 100))
        if spend_cents > 0:
            db.add(BalanceLedger(
                user_id=user_id,
                amount=-spend_cents,
                reason="chat_v2_run",
                project_id=project_id,
            ))

        db.commit()
        db.refresh(run)

        terminal_payload = {
            "total_cost_usd": total_cost_usd,
            "iterations": result.get("iterations"),
            "thinking_duration": thinking_duration_s,
        }
        if cancelled:
            # Use the cancelled-event helper so the FE's `cancelled`
            # branch in consumeRunEvents fires (and the bus is cleaned).
            # Carry the stop_reason in the same payload as the legacy
            # streaming path so the FE's `done`+stop_reason inline note
            # logic works for v2 too via the matching applied_changes.
            terminal_payload["stopped"] = True
            terminal_payload["stop_reason"] = "cancel"
            legacy_runs.mark_run_cancelled(db, run, terminal_payload)
        else:
            # Final lifecycle event. thinking_duration drives the FE's
            # "Took X" label — without it the meta line above the
            # assistant message stays hidden because the FE gates on
            # `evt.thinking_duration`.
            legacy_runs.mark_run_completed(db, run, terminal_payload)
    finally:
        db.close()

    if cancelled:
        # Re-raise so the outer _run_chat_v2_task's except-block sees
        # the cancellation (it swallows + logs) and the asyncio.Task
        # ends in the cancelled state — which is the truthful outcome
        # and lets cancel_v2_run's `if task.done()` check work next time.
        raise asyncio.CancelledError()
