"""Durable chat run lifecycle.

Turn-level entry points (`start_run`, `_run_chat_task`, `cancel_run`,
`inject_message`, `_drive_agent`). Builds on `run_state.py` for the
infrastructure primitives:

  - chat_runs / chat_run_events tables (durable state)
  - per-project lock + global semaphore (concurrency)
  - emit_event / mark_run_completed / _mark_run_failed (lifecycle)
  - tail_events (SSE replay + live tail + heartbeat + reattach)

Net effect: a turn runs as a background asyncio task that survives
client disconnect; the FE can reattach on refresh via the chat-run
URLs. Assistant message + tool_log + cost are persisted at run
completion regardless of SSE connection state.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any, Dict, List, Optional
from uuid import UUID

import httpx
from sqlalchemy import text as sa_text
from sqlalchemy.sql import func

from dsl_api.config import settings as dsl_api_settings
from dsl_api.credits import consume_credits
from dsl_api.db import SessionLocal
from dsl_api.models import Account, ChatMessage, ChatRun, Project
from dsl_api.plans import CENTS_PER_CREDIT
from dsl_api.models.chat_run import (
    RUN_STATUS_CANCELLED,
    RUN_STATUS_FAILED,
    RUN_STATUS_PAUSED,
    RUN_STATUS_QUEUED,
    RUN_STATUS_RUNNING,
    RUN_TERMINAL_STATUSES,
)

# Legacy primitives we reuse wholesale.
from dsl_worker.chat import run_state


log = logging.getLogger(__name__)


# Registry of in-flight asyncio.Task objects keyed by run_id. Populated
# by start_run, drained by _run_chat_task's finally, read by
# cancel_run. Lives only inside the worker process — if the worker
# restarts, the orphan reaper handles stale runs. The registry is what
# makes cancellation instantaneous: cancel_run does task.cancel(),
# which raises CancelledError into the agent's current await on the
# next event-loop tick, and the loop's CancelledError path persists
# whatever cost was already incurred before exiting.
_active_tasks: Dict[UUID, asyncio.Task] = {}


def _charge_run_credits(
    db, user_id, spend_cents: int, project_id, reason: str
) -> None:
    """Deduct credits via consume_credits — decrements Account pools AND
    writes the BalanceLedger entries. Doing only the ledger write (the
    historical mistake) leaves Account.subscription_credits / daily /
    rollover untouched, so the user's balance display stays flat while
    the audit log says they spent.
    """
    if spend_cents <= 0:
        return
    account = db.query(Account).filter(Account.user_id == user_id).first()
    if not account:
        log.warning("charge_run_credits: no account for user %s", user_id)
        return
    consume_credits(
        db,
        account,
        spend_cents / CENTS_PER_CREDIT,
        project_id=project_id,
        reason=reason,
    )


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
            run_state.emit_event(db, run, "user_injection", {
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


async def _auto_name_project(project_id: UUID, user_content: str, run_id: Optional[UUID] = None) -> None:
    """One-shot mini LLM call to generate a 3-5 word project name from the
    first message. Subsidized — bypasses TrackedOpenAIClient + balance_ledger
    on purpose; this isn't user-facing OpenAI usage worth charging for.
    Only runs if the project is still on the default name; idempotent for
    subsequent turns.

    When run_id is provided, emits a `project_name` SSE event after the
    DB update so the FE swaps the title in place (useChat handles it at
    case "project_name"). Without this event the new name only shows
    after a page refresh.
    """
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
        result = db.execute(
            sa_text(
                "UPDATE projects SET name=:n, updated_at=now() "
                "WHERE id=:p AND name = ANY(:defaults)"
            ),
            {"n": name, "p": str(project_id), "defaults": ["New Dataset", "Untitled", ""]},
        )
        db.commit()
        if result.rowcount and run_id is not None:
            # Tell the FE so the header title swaps in place. Without
            # this the rename only shows on next page refresh.
            run = db.query(ChatRun).filter(ChatRun.id == run_id).first()
            if run is not None:
                try:
                    run_state.emit_event(db, run, "project_name", {"name": name})
                    db.commit()
                except Exception:
                    log.exception("project_name event emit failed for project %s", project_id)
    except Exception:
        log.exception("auto-name UPDATE failed for project %s", project_id)
    finally:
        db.close()


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
    except (httpx.ConnectTimeout, httpx.ConnectError, httpx.ReadTimeout) as e:
        # Network unreachable — common on dev machines where outbound
        # hooks.slack.com is blocked. Non-blocking, just noisy.
        # Single-line warning instead of a stack trace.
        log.warning("Slack webhook unreachable for %s: %s", project_id, type(e).__name__)
    except Exception:
        log.exception("Failed to post first-chat notification to Slack for %s", project_id)


async def start_run(
    project_id: UUID,
    user_id: UUID,
    user_content: str,
) -> ChatRun:
    """Create a ChatRun + user ChatMessage; fire the background task.

    Returns the detached ChatRun row. The first user message on a
    project also fires a Slack notification so we see new projects in
    real time."""
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

        is_first_message = (
            db.query(ChatMessage)
            .filter(ChatMessage.project_id == project_id)
            .count()
            == 0
        )

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
                sa_text("SELECT email FROM auth.users WHERE id = :uid"),
                {"uid": str(user_id)},
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

    task = asyncio.create_task(
        _run_chat_task(run_id, user_id, user_content),
        name=f"chat-run-{run_id}",
    )
    _active_tasks[run_id] = task

    # Fire-and-forget project naming. Independent of the orchestrator
    # so a slow LLM here can't block the run. Only names projects that
    # are still on the default name; subsequent turns won't re-trigger.
    # Pass run_id so the helper can emit a `project_name` SSE event
    # after the rename — that's what swaps the FE title in place.
    asyncio.create_task(
        _auto_name_project(project_id, user_content, run_id=run_id),
        name=f"chat-name-{project_id}",
    )

    # Re-fetch detached so the caller can read fields without lazy-load.
    db = SessionLocal()
    try:
        run = db.query(ChatRun).filter(ChatRun.id == run_id).first()
        db.expunge(run)  # type: ignore[arg-type]
        return run  # type: ignore[return-value]
    finally:
        db.close()


async def _run_chat_task(
    run_id: UUID,
    user_id: UUID,
    user_content: str,
) -> None:
    """Background coroutine for one chat run.

    Acquires the same global semaphore + per-project lock the legacy
    runs use, so chat turns and legacy turns serialize correctly on
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

        async with run_state._global_run_semaphore():
            async with run_state._project_lock(project_id):
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
                    name=f"chat-heartbeat-{run_id}",
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
                    run_state._mark_run_failed(run_id, str(e)[:500])
                finally:
                    heartbeat.cancel()
                    try:
                        await heartbeat
                    except (asyncio.CancelledError, Exception):
                        pass
    finally:
        _active_tasks.pop(run_id, None)
        _pending_injections.pop(run_id, None)


def _force_persist_terminal(
    *,
    run_id: UUID,
    project_id: UUID,
    user_id: UUID,
    final_text: str,
    applied_changes: Dict[str, Any],
    spend_cents: int,
    charged_cents: int,
    terminal_payload: Dict[str, Any],
    cancelled: bool,
) -> None:
    """Belt-and-suspenders persist for chat turn completion.

    Called from _drive_agent when the shared session's commit raises
    (e.g. an in-flight tool transaction was cut short by a CancelledError
    and left the session in a poisoned state). Opens a fresh session,
    writes the assistant ChatMessage, the BalanceLedger entry, and the
    terminal cancelled/done event, then drops the bus subscriber.
    Idempotent against the run already being terminal (a concurrent
    request_cancel may have flipped it first).
    """
    fresh = SessionLocal()
    try:
        run = fresh.query(ChatRun).filter(ChatRun.id == run_id).first()
        if run is None:
            log.warning("force_persist_v2: run %s vanished", run_id)
            return
        if run.assistant_message_id is None:
            msg = ChatMessage(
                project_id=project_id,
                role="assistant",
                content=final_text,
                run_id=run_id,
                applied_changes=applied_changes,
            )
            fresh.add(msg)
            fresh.flush()
            run.assistant_message_id = msg.id
            # Settle the residual — incremental cost_update charges
            # already debited charged_cents mid-turn on a separate
            # session, so this fresh-session flush only owes the delta.
            residual_cents = spend_cents - charged_cents
            _charge_run_credits(
                fresh, user_id, residual_cents, project_id, reason="chat_run"
            )
            fresh.commit()
            fresh.refresh(run)

        if run.status not in RUN_TERMINAL_STATUSES:
            if cancelled:
                run_state.mark_run_cancelled(fresh, run, terminal_payload)
            else:
                run_state.mark_run_completed(fresh, run, terminal_payload)
        else:
            log.info(
                "force_persist_v2: run %s already terminal (%s) — skipping terminal-event emit",
                run_id, run.status,
            )
    except Exception:
        log.exception("force_persist_v2 failed for run %s", run_id)
    finally:
        fresh.close()


def cancel_run(run_id: UUID) -> bool:
    """Instantaneous cancel for a chat run.

    Sends CancelledError into the task's current await. The agent's
    CancelledError handler in _drive_agent flushes whatever cost was
    incurred so far to the balance ledger and persists a partial
    assistant message + cancelled-status event before exiting — so the
    user can't dodge billing by cancelling mid-call.

    Also cancels any background tasks (wait=false table_creates,
    enrichment_runs, etc.) spawned by this run — Stop = stop everything,
    so the user doesn't keep accruing spend on backgrounded work after
    clicking Stop. Background tasks have their own CancelledError handlers
    that capture partial cost and update chat_background_tasks.status.

    Falls back to a DB-only status flip via legacy.request_cancel if
    the registry has no task (worker restarted between start and cancel,
    or the run already finished). In that case the orphan reaper picks
    up the stale state.
    """
    # Cancel any backgrounded tasks tied to this chat run first. They
    # run independently of the main agent task; without this, Stop would
    # only halt the LLM loop and leave the bg tasks burning credits.
    # Fire-and-forget on the running loop — the cancel itself is async
    # (REGISTRY's lock) but we don't need to await it from this sync
    # function; the bg tasks' CancelledError handlers settle their own
    # cost into the chat_background_tasks row.
    try:
        from dsl_worker.chat.background_tasks import REGISTRY as _BG
        asyncio.create_task(_BG.cancel_run(str(run_id)))
    except RuntimeError:
        # No running loop (cancel called from a non-async context — e.g.
        # the orphan reaper). Bg tasks can still finish on their own;
        # the row will flip to 'complete' or 'error' when they do.
        pass
    except Exception:
        log.exception("cancel_run: bg cancel failed for %s", run_id)

    task = _active_tasks.get(run_id)
    if task is not None and not task.done():
        task.cancel()
        return True
    return run_state.request_cancel(run_id)


async def _heartbeat_loop(run_id: UUID) -> None:
    """Emit a heartbeat event every 30s so the staleness sweeper never
    false-positives an actively-running task. Cancelled by the caller
    when the run finishes — until then, this is the proof of life.

    DB write runs via asyncio.to_thread so a slow commit doesn't block
    the event loop and starve other coroutines (which would in turn
    starve THIS coroutine on the next iteration — a known regression
    mode that caused 82% of run failures last week before the
    url-verify hook removal)."""
    def _emit_sync() -> bool:
        """Returns True to keep looping, False if run is terminal."""
        db = SessionLocal()
        try:
            run = db.query(ChatRun).filter(ChatRun.id == run_id).first()
            if run is None or run.status in RUN_TERMINAL_STATUSES:
                return False
            run_state.emit_event(db, run, "heartbeat", {})
            db.commit()
            return True
        except Exception:
            log.debug("heartbeat emit failed; continuing", exc_info=True)
            return True
        finally:
            db.close()

    while True:
        try:
            await asyncio.sleep(30)
        except asyncio.CancelledError:
            return
        try:
            keep = await asyncio.to_thread(_emit_sync)
        except Exception:
            log.debug("heartbeat to_thread failed; continuing", exc_info=True)
            continue
        if not keep:
            return


async def _drive_agent(
    run_id: UUID,
    user_id: UUID,
    project_id: UUID,
    user_content: str,
) -> None:
    """The actual chat turn: load history, call run_turn with an
    event callback that fans into chat_run_events, persist assistant
    message + ledger entry, mark run completed."""
    # Late import to keep this module light on import.
    from dsl_worker.chat.agent import run_turn

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
        run_state.emit_event(db, run, "status", {"content": "Thinking…"})

        tool_log: List[Dict[str, Any]] = []
        # Tool names we deliberately hide from the chat UI — their pills
        # never appear in the live SSE stream OR the persisted assistant
        # message. Use for internal mechanics whose name/args/result would
        # confuse or leak signal to the user. load_skill exposes
        # internal playbook names + bodies; those are agent IP, not
        # user-facing content.
        _HIDDEN_TOOLS_FROM_CHAT = {"load_skill"}
        total_cost_ref = {"value": 0.0}
        # Cents already deducted from the account during this turn via
        # incremental cost_update events. The end-of-turn flush charges
        # only the residual (final spend minus what's already settled).
        # Decoupled from total_cost_ref so we can read it both at normal
        # completion and on the cancel/force-persist path.
        charged_cents_ref = {"value": 0}
        final_text_ref = {"value": ""}

        async def on_event(evt: Dict[str, Any]) -> None:
            etype = evt.get("type")
            # Fast path for live token deltas: skip the DB session +
            # ChatRun lookup (publish_token_delta only needs run_id) and
            # yield to the event loop so SSE subscribers drain BEFORE
            # the next delta lands. Without the explicit sleep(0) the
            # OpenAI stream loop can fire dozens of output_text.delta
            # events between scheduler ticks — the bus enqueues them
            # all and the consumer flushes them in one burst at the
            # end, making the FE see the full text "all at once" after
            # the LLM call completes.
            if etype == "text_delta":
                delta = evt.get("text") or ""
                if delta:
                    run_state.publish_token_delta(run_id, delta)
                    await asyncio.sleep(0)
                return
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
                    run_state.publish_thinking_delta(lrun.id, evt.get("text") or "")
                elif etype == "tool_call_start":
                    tc_id = evt.get("tool_call_id") or ""
                    name = evt.get("name") or "?"
                    # Suppress hidden tools entirely — no tool_log entry,
                    # no SSE pill, no DB row. The internal agent loop
                    # still runs them; we just don't surface them.
                    if name in _HIDDEN_TOOLS_FROM_CHAT:
                        return
                    args_preview = json.dumps(evt.get("args") or {}, default=str)
                    tool_log.append({
                        "id": tc_id,
                        "name": name,
                        "args_preview": args_preview,
                    })
                    run_state.emit_event(ldb, lrun, "tool_call", {
                        "id": tc_id,
                        "name": name,
                        "args_preview": args_preview,
                    })
                elif etype == "tool_call_result":
                    if (evt.get("name") or "") in _HIDDEN_TOOLS_FROM_CHAT:
                        return
                    tc_id = evt.get("tool_call_id") or ""
                    summary = evt.get("result_preview") or ""
                    cost = float(evt.get("cost_usd") or 0.0)
                    duration_ms = int(evt.get("duration_ms") or 0)
                    for entry in tool_log:
                        if entry.get("id") == tc_id:
                            entry["summary"] = summary
                            entry["cost"] = cost
                            entry["duration_ms"] = duration_ms
                            break
                    run_state.emit_event(ldb, lrun, "tool_result", {
                        "id": tc_id,
                        "name": evt.get("name"),
                        "summary": summary,
                        "cost": cost,
                        "duration_ms": duration_ms,
                    })
                elif etype == "llm_call_complete":
                    # Surface to SSE so a live FE timing overlay (or a
                    # future per-iteration timing chip) can render
                    # without re-deriving from logs.
                    run_state.emit_event(ldb, lrun, "llm_call_complete", {
                        "iteration": evt.get("iteration"),
                        "duration_ms": int(evt.get("duration_ms") or 0),
                        "cost_usd": float(evt.get("cost_usd") or 0.0),
                    })
                elif etype == "final_message":
                    text = evt.get("text") or ""
                    final_text_ref["value"] = text
                    # Deltas already streamed live via text_delta. Use
                    # replace_text_content to overwrite the bus
                    # accumulator with the canonical final text (handles
                    # any drift between concat'd deltas and the
                    # authoritative response text, e.g. citation-cleaned
                    # variants) AND persist a text_checkpoint for
                    # reconnects.
                    run_state.replace_text_content(ldb, lrun, text)
                elif etype == "text_segment":
                    # Mid-iteration text from the agent — the iteration
                    # had both message text AND function calls. The
                    # deltas already streamed live into the final-text
                    # slot, so we (1) trim them out of the bus
                    # accumulator (keeps the cumulative snapshot equal
                    # to "what's in the FINAL segment") and (2) emit the
                    # durable text_segment event. The FE handler
                    # demotes the currently-streaming final-text segment
                    # to a mid-iteration segment in place.
                    seg_text = evt.get("text") or ""
                    if seg_text:
                        run_state.trim_token_content(lrun.id, seg_text)
                    run_state.emit_event(ldb, lrun, "text_segment", {
                        "iteration": evt.get("iteration"),
                        "text": seg_text,
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

                    # Incremental balance debit: subtract the delta since
                    # the last cost_update from the user's account pools
                    # right now, not at end-of-turn. Long turns (multi-
                    # step agent loops, BU sessions) previously hid spend
                    # until the very end. ldb is a fresh session so this
                    # commit is independent of the main turn's state.
                    new_cumulative_cents = int(round(new_total * 100))
                    delta_cents = new_cumulative_cents - charged_cents_ref["value"]
                    if delta_cents > 0:
                        try:
                            _charge_run_credits(
                                ldb, user_id, delta_cents, project_id,
                                reason="chat_run",
                            )
                            ldb.commit()
                            charged_cents_ref["value"] = new_cumulative_cents
                        except Exception:
                            log.exception(
                                "incremental credit charge failed (run=%s, "
                                "delta_cents=%d) — end-of-turn flush will "
                                "settle the residual",
                                run_id, delta_cents,
                            )
                            try:
                                ldb.rollback()
                            except Exception:
                                pass

                    run_state.emit_event(ldb, lrun, "cost_update", {
                        "total_cost_usd": new_total,
                    })
                elif etype == "turn_complete":
                    total_cost_ref["value"] = float(evt.get("total_cost_usd") or 0.0)
                elif etype == "error":
                    run_state.emit_event(ldb, lrun, "error", {
                        "message": evt.get("message") or "unknown error",
                    })
                elif etype == "approval_required":
                    # Pass-through for the approval gate (agent.py emits
                    # this and then awaits the user's decision). FE
                    # useChat picks it up, Project.tsx mounts the approval
                    # card above the chat input. Without this branch the
                    # event was silently dropped — agent would block on
                    # the pending Future forever.
                    run_state.emit_event(ldb, lrun, "approval_required", {
                        "approval_id": evt.get("approval_id"),
                        "tool": evt.get("tool"),
                        "args": evt.get("args"),
                        "estimated_cost_credits": evt.get("estimated_cost_credits"),
                        "summary": evt.get("summary"),
                    })
                elif etype == "approval_resolved":
                    run_state.emit_event(ldb, lrun, "approval_resolved", {
                        "approval_id": evt.get("approval_id"),
                        "approved": evt.get("approved"),
                    })
                elif etype == "phase":
                    # Pass-through for timing instrumentation. Carries
                    # `phase` (dotted path like "agent/iteration_start")
                    # plus arbitrary meta from the emitter.
                    run_state.emit_event(ldb, lrun, "phase", {
                        k: v for k, v in evt.items() if k != "type"
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
            # If the cancel hit before the final_message event fired
            # (e.g. mid-stream of the final iteration), fall back to the
            # bus accumulator so we persist whatever text was streamed
            # to the user — otherwise refresh would show an empty
            # assistant bubble for a turn that visibly produced text.
            partial_text = final_text_ref["value"] or run_state._BUS._content.get(str(run_id), "")
            result = {
                "final_message": partial_text,
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

        # Same for enrichment_card_added — mirrors the table_cards block
        # so a refresh keeps the "Enrichment created" chips visible.
        try:
            from dsl_api.models import ChatRunEvent
            ecards = (
                db.query(ChatRunEvent.payload)
                .filter(
                    ChatRunEvent.run_id == run_id,
                    ChatRunEvent.type == "enrichment_card_added",
                )
                .order_by(ChatRunEvent.seq.asc())
                .all()
            )
            if ecards:
                ac["enrichment_cards"] = [c[0] for c in ecards if c[0]]
        except Exception:
            log.exception("collecting enrichment_cards for applied_changes failed")

        # Same trick for suggest_replies chips: replay the SSE events
        # we emitted live so a refresh re-renders them. Multiple emits
        # in one turn collapse into a single ordered items list.
        try:
            from dsl_api.models import ChatRunEvent
            sg_rows = (
                db.query(ChatRunEvent.payload)
                .filter(
                    ChatRunEvent.run_id == run_id,
                    ChatRunEvent.type == "suggestions",
                )
                .order_by(ChatRunEvent.seq.asc())
                .all()
            )
            sg_items: List[Dict[str, Any]] = []
            for (p,) in sg_rows:
                if isinstance(p, dict):
                    for it in (p.get("items") or []):
                        if isinstance(it, dict) and it.get("label") and it.get("message"):
                            sg_items.append(it)
            if sg_items:
                ac["suggestions"] = {"items": sg_items}
        except Exception:
            log.exception("collecting suggestions for applied_changes failed")

        # Rebuild the interleaved segments list (mid-iteration text +
        # tool batches) from chat_run_events in seq order, mirroring how
        # the FE useChat live-stream consumer builds m.segments. Without
        # this, refreshing after a turn loses every mid-iteration narration
        # line — the ChatMessage.content only carries the FINAL iteration's
        # text, and the toolLog renders flat without the interleaved text
        # the user already saw live. Matches the live segment shape:
        # {kind:"text", content, final?} | {kind:"tools", toolIds}.
        try:
            from dsl_api.models import ChatRunEvent
            ord_rows = (
                db.query(ChatRunEvent.type, ChatRunEvent.payload)
                .filter(
                    ChatRunEvent.run_id == run_id,
                    ChatRunEvent.type.in_(["text_segment", "tool_call"]),
                )
                .order_by(ChatRunEvent.seq.asc())
                .all()
            )
            segments: List[Dict[str, Any]] = []
            for etype, payload in ord_rows:
                if not isinstance(payload, dict):
                    continue
                if etype == "text_segment":
                    txt = (payload.get("text") or "").strip()
                    if not txt:
                        continue
                    segments.append({"kind": "text", "content": payload.get("text") or "", "final": False})
                elif etype == "tool_call":
                    tcid = payload.get("id")
                    if not tcid:
                        continue
                    # Group consecutive tool_calls into one "tools" segment
                    # so the FE renders the parallel batch as a single row
                    # of chips (matches the live build at useChat.ts:1466).
                    if segments and segments[-1]["kind"] == "tools":
                        segments[-1]["toolIds"].append(tcid)
                    else:
                        segments.append({"kind": "tools", "toolIds": [tcid]})
            # Append the final assistant text as the trailing final-text
            # segment if we have any narration / tools above it. The FE
            # also has the standalone ChatMessage.content to fall back on,
            # but stashing the final segment here keeps the segments list
            # self-contained (FE doesn't need to splice on m.content).
            if final_text:
                segments.append({"kind": "text", "content": final_text, "final": True})
            if segments:
                ac["segments"] = segments
        except Exception:
            log.exception("rebuilding segments for applied_changes failed")

        # Persist the assistant message + ledger + terminal event. On
        # cancel paths the shared `db` session may be in an inconsistent
        # state (an in-flight tool's transaction was cut short between
        # its execute and its commit), so wrap the whole flush and fall
        # back to a fresh-session force-persist if anything raises. The
        # alternative — letting the exception propagate — leaves the
        # user with their input message and no assistant bubble after
        # refresh (the exact "last message disappeared" report).
        spend_cents = int(round(total_cost_usd * 100))
        terminal_payload = {
            "total_cost_usd": total_cost_usd,
            "iterations": result.get("iterations"),
            "thinking_duration": thinking_duration_s,
        }
        if cancelled:
            terminal_payload["stopped"] = True
            terminal_payload["stop_reason"] = "cancel"

        try:
            assistant_msg = ChatMessage(
                project_id=project_id,
                role="assistant",
                content=final_text,
                run_id=run_id,
                applied_changes=ac,
            )
            db.add(assistant_msg)
            db.flush()
            run.assistant_message_id = assistant_msg.id

            # Settle the residual — incremental cost_update charges
            # already debited charged_cents_ref["value"] mid-turn.
            residual_cents = spend_cents - charged_cents_ref["value"]
            _charge_run_credits(
                db, user_id, residual_cents, project_id, reason="chat_run"
            )

            db.commit()
            db.refresh(run)

            if cancelled:
                run_state.mark_run_cancelled(db, run, terminal_payload)
            else:
                run_state.mark_run_completed(db, run, terminal_payload)
        except Exception:
            log.exception(
                "v2 run %s flush failed on shared session — falling back "
                "to force-persist on a fresh session", run_id,
            )
            try:
                db.rollback()
            except Exception:
                log.exception("v2 run %s rollback failed", run_id)
            _force_persist_terminal(
                run_id=run_id,
                project_id=project_id,
                user_id=user_id,
                final_text=final_text,
                applied_changes=ac,
                spend_cents=spend_cents,
                charged_cents=charged_cents_ref["value"],
                terminal_payload=terminal_payload,
                cancelled=cancelled,
            )
    finally:
        db.close()

    if cancelled:
        # Re-raise so the outer _run_chat_task's except-block sees
        # the cancellation (it swallows + logs) and the asyncio.Task
        # ends in the cancelled state — which is the truthful outcome
        # and lets cancel_run's `if task.done()` check work next time.
        raise asyncio.CancelledError()
