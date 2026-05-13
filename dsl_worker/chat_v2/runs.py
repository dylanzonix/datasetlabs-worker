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


CENTS_PER_CREDIT = 1  # 1 credit = 1 cent. Matches legacy cost_tracker default.


_DEFAULT_PROJECT_NAMES = {"New Dataset", "Untitled", "", None}


async def _auto_name_project(project_id: UUID, user_content: str) -> None:
    """One-shot LLM call to generate a 3-5 word project name from the
    first message. Only runs if the project is still on the default
    name; idempotent for subsequent turns."""
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

    asyncio.create_task(
        _run_chat_v2_task(run_id, user_id, user_content),
        name=f"chat-v2-run-{run_id}",
    )

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

            try:
                await _drive_agent(run_id, user_id, project_id, user_content)
            except Exception as e:
                log.exception("v2 run %s crashed", run_id)
                legacy_runs._mark_run_failed(run_id, str(e)[:500])


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

        result = await run_turn(
            db=db,
            project_id=str(project_id),
            user_id=str(user_id),
            run_id=str(run_id),
            user_message=user_content,
            history=history,
            on_event=on_event,
        )

        final_text = result.get("final_message") or final_text_ref["value"] or ""
        total_cost_usd = float(result.get("total_cost_usd") or total_cost_ref["value"])

        # Persist assistant ChatMessage with tool_log + cost.
        ac: Dict[str, Any] = {
            "tool_log": tool_log,
            "total_cost_usd": total_cost_usd,
            "iterations": result.get("iterations"),
        }
        if result.get("error"):
            ac["error"] = result["error"]

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

        # Charge balance_ledger. cost is USD; ledger amounts are cents
        # (negative = charge). 1 credit = 1 cent, so we just convert.
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

        # Final lifecycle event.
        legacy_runs.mark_run_completed(db, run, {
            "total_cost_usd": total_cost_usd,
            "iterations": result.get("iterations"),
        })
    finally:
        db.close()
