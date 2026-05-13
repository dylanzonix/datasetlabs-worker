"""v-next orchestrator agent — minimal dispatch loop.

For v1 this is a non-streaming "run a turn to completion" function. The
streaming SSE wrapper that surfaces tool calls and progress to the FE is a
separate layer (see streaming_v2.py — not yet written; for now the chat-api
routes can call run_turn directly and return when the agent stops).

Loop:
  1. Build system prompt + project state banner (auto-injected as user prefix)
  2. Append conversation history + the new user message
  3. Call OpenAI Responses API with the 15-tool surface
  4. For each tool call: dispatch to HANDLERS, append result back
  5. Stop when assistant emits text-only (no tool calls) — that's the reply
  6. Persist messages and return
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import uuid
from typing import Any, AsyncIterator, Awaitable, Callable, Dict, List, Optional

from openai import AsyncOpenAI
from sqlalchemy.orm import Session
from sqlalchemy import text as sa_text

from dsl_worker.chat_v2 import (
    HANDLERS,
    TOOL_DEFS,
    ToolContext,
    build_project_state,
    build_system_prompt,
)


log = logging.getLogger(__name__)


MAX_TURN_ITERATIONS = 30  # hard safety cap on tool-call rounds per turn


# Event types emitted by the streaming turn runner. The FE renders these
# as live tool log entries + final message; see ChatSidebar / useChat.
StreamEvent = Dict[str, Any]
EventCallback = Optional[Callable[[StreamEvent], Awaitable[None]]]


async def run_turn(
    db: Session,
    project_id: str,
    user_id: str,
    run_id: Optional[str],
    user_message: str,
    history: Optional[List[Dict[str, str]]] = None,
    on_event: EventCallback = None,
) -> Dict[str, Any]:
    """Run one chat turn end-to-end. Returns the assistant's final text +
    metadata.

    Args:
        db: SQLAlchemy session.
        project_id: project the turn is for.
        user_id: user owning the project.
        run_id: optional chat_run id (for event emission); None = ad-hoc.
        user_message: the user's input for this turn.
        history: prior turn pairs as OpenAI messages; None = fresh.

    Returns:
        {final_message, tool_calls_made, total_cost_usd, iterations}
    """
    client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    model = os.getenv("OPENAI_MODEL", "gpt-5.4")

    ctx = ToolContext(
        db=db, project_id=project_id, user_id=user_id, run_id=run_id, emit_event=None
    )

    system_prompt = build_system_prompt()
    project_state = build_project_state(db, project_id)

    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
    ]
    if history:
        messages.extend(history)
    # Project state injected as a user-visible prefix to the new message —
    # keeps the agent grounded each turn without polluting history.
    messages.append({"role": "user", "content": f"{project_state}\n\n{user_message}"})

    tool_calls_made: List[Dict[str, Any]] = []
    total_cost_usd = 0.0
    final_text = ""

    async def emit(evt: StreamEvent) -> None:
        if on_event is None:
            return
        try:
            await on_event(evt)
        except Exception:
            log.exception("on_event callback raised; continuing")

    await emit({"type": "turn_started", "project_id": project_id})

    for iteration in range(MAX_TURN_ITERATIONS):
        try:
            resp = await client.chat.completions.create(
                model=model,
                messages=messages,
                tools=TOOL_DEFS,
                tool_choice="auto",
                temperature=0.0,
            )
        except Exception as e:
            log.exception("LLM call failed: %s", e)
            await emit({"type": "error", "message": str(e)})
            return {
                "final_message": f"(error: {e})",
                "tool_calls_made": tool_calls_made,
                "total_cost_usd": total_cost_usd,
                "iterations": iteration,
                "error": str(e),
            }

        msg = resp.choices[0].message
        # No tool calls — assistant is done
        if not msg.tool_calls:
            final_text = msg.content or ""
            messages.append({"role": "assistant", "content": final_text})
            await emit({"type": "final_message", "text": final_text})
            break

        # Record assistant tool-call message
        messages.append({
            "role": "assistant",
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                }
                for tc in msg.tool_calls
            ],
        })

        # Dispatch each tool call
        for tc in msg.tool_calls:
            name = tc.function.name
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            await emit({"type": "tool_call_start", "tool_call_id": tc.id, "name": name, "args": args})
            handler = HANDLERS.get(name)
            if not handler:
                tool_result = {"error": f"unknown tool {name}"}
                cost = 0.0
            else:
                try:
                    tool_result, cost = await handler(args, ctx)
                except Exception as e:
                    log.exception("tool %s raised: %s", name, e)
                    tool_result = {"error": str(e)[:300]}
                    cost = 0.0

            preview = json.dumps(tool_result, default=str)[:300]
            tool_calls_made.append({
                "name": name,
                "args": args,
                "result_preview": preview,
                "cost_usd": cost,
            })
            total_cost_usd += cost
            await emit({
                "type": "tool_call_result",
                "tool_call_id": tc.id,
                "name": name,
                "result_preview": preview,
                "cost_usd": cost,
            })
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": json.dumps(tool_result, default=str)[:8000],
            })
    else:
        log.warning("agent loop hit MAX_TURN_ITERATIONS=%d for project %s", MAX_TURN_ITERATIONS, project_id)

    await emit({
        "type": "turn_complete",
        "total_cost_usd": total_cost_usd,
        "iterations": iteration + 1,
    })

    return {
        "final_message": final_text,
        "tool_calls_made": tool_calls_made,
        "total_cost_usd": total_cost_usd,
        "iterations": iteration + 1,
    }


async def stream_turn(
    db: Session,
    project_id: str,
    user_id: str,
    run_id: Optional[str],
    user_message: str,
    history: Optional[List[Dict[str, str]]] = None,
) -> AsyncIterator[StreamEvent]:
    """Run a turn and yield events as they happen. Backs the SSE endpoint."""
    queue: asyncio.Queue[StreamEvent] = asyncio.Queue()
    SENTINEL: StreamEvent = {"type": "__end__"}

    async def emit(evt: StreamEvent) -> None:
        await queue.put(evt)

    async def runner() -> None:
        try:
            result = await run_turn(
                db=db,
                project_id=project_id,
                user_id=user_id,
                run_id=run_id,
                user_message=user_message,
                history=history,
                on_event=emit,
            )
            await queue.put({"type": "turn_result", **{k: v for k, v in result.items() if k != "tool_calls_made"}})
        except Exception as e:
            log.exception("stream_turn runner failed")
            await queue.put({"type": "error", "message": str(e)})
        finally:
            await queue.put(SENTINEL)

    task = asyncio.create_task(runner())
    try:
        while True:
            evt = await queue.get()
            if evt is SENTINEL:
                break
            yield evt
    finally:
        if not task.done():
            task.cancel()
