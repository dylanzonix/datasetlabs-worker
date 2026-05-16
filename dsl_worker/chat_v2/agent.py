"""v-next orchestrator agent — Responses-API loop with reasoning + real cost.

Reuses the legacy infra:
  - TrackedOpenAIClient → real token-level cost + cached-input handling
  - ResilientClient (inside Tracked*) → retries with backoff
  - RateLimiter (inside Tracked*) → TPM/RPM throttle
  - Responses API → reasoning_effort + native web_search + structured items
  - prompt_cache_key → routing for prompt cache hits

We keep custom control over the iteration loop because we need to emit fine-
grained SSE events (tool_call_start/result, reasoning summaries) into the
chat stream. The wider agents/base.py:AgentConversation is a great pattern
for non-streaming flows but doesn't expose mid-turn callbacks at that
granularity.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
from typing import Any, AsyncIterator, Awaitable, Callable, Dict, List, Optional

from openai import AsyncOpenAI
from sqlalchemy.orm import Session

from dsl_worker.billing.tracked_client import TrackedOpenAIClient
from dsl_worker.chat_v2 import (
    HANDLERS,
    TOOL_DEFS,
    ToolContext,
    build_project_state,
    build_system_prompt,
)


log = logging.getLogger(__name__)


MAX_TURN_ITERATIONS = 30  # hard safety cap on tool-call rounds per turn


StreamEvent = Dict[str, Any]
EventCallback = Optional[Callable[[StreamEvent], Awaitable[None]]]


def _flatten_tool_defs() -> List[Dict[str, Any]]:
    """chat.completions shape {type, function:{name,description,parameters}}
    → Responses shape {type, name, description, parameters}."""
    out: List[Dict[str, Any]] = []
    for t in TOOL_DEFS:
        if t.get("type") == "function" and "function" in t:
            fn = t["function"]
            out.append({
                "type": "function",
                "name": fn["name"],
                "description": fn.get("description", ""),
                "parameters": fn.get("parameters") or {"type": "object", "properties": {}},
            })
        else:
            out.append(t)
    return out


_TOOLS_PAYLOAD = _flatten_tool_defs()


def _build_client() -> TrackedOpenAIClient:
    raw = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    return TrackedOpenAIClient(raw)


async def run_turn(
    db: Session,
    project_id: str,
    user_id: str,
    run_id: Optional[str],
    user_message: str,
    history: Optional[List[Dict[str, str]]] = None,
    on_event: EventCallback = None,
    drain_injections: Optional[Callable[[], List[Dict[str, str]]]] = None,
) -> Dict[str, Any]:
    """Run one chat turn end-to-end via the Responses API.

    Returns {final_message, tool_calls_made, total_cost_usd, iterations}.
    Emits via on_event: turn_started, reasoning, tool_call_start,
    tool_call_result, final_message, turn_complete, error.

    drain_injections, if provided, is called at the top of each model
    iteration (before the LLM call) to pull any user messages that
    were posted via the inject endpoint since the last iteration. The
    items are prepended as user-role input items so the model sees the
    new content on its next inference. Long-running tools are not
    interrupted — injections are only consumed between major model
    decisions, matching the Claude Code behavior.
    """
    client = _build_client()
    model = os.getenv("OPENAI_MODEL", "gpt-5.4")
    effort = os.getenv("CHAT_V2_REASONING_EFFORT", "medium")

    async def _emit_progress(msg: str) -> None:
        if on_event:
            await emit({"type": "progress", "text": msg})

    ctx = ToolContext(
        db=db,
        project_id=project_id,
        user_id=user_id,
        run_id=run_id,
        emit_progress=_emit_progress,
        emit_event=None,
    )

    system_prompt = build_system_prompt()
    project_state = build_project_state(db, project_id)

    # System message stays stable across the turn for prompt cache hits.
    system_msg: Dict[str, Any] = {"role": "system", "content": system_prompt}

    # Pre-seed history (prior user/assistant text), then volatile
    # project_state + the new user message.
    input_items: List[Dict[str, Any]] = []
    if history:
        for m in history:
            role = m.get("role")
            content = m.get("content", "")
            if role in ("user", "assistant") and content:
                input_items.append({"role": role, "content": content})
    input_items.append({
        "role": "user",
        "content": f"{project_state}\n\n{user_message}",
    })

    # Stable cache key per identical system prompt so requests route to the
    # same backend → better prompt-cache hit rates.
    cache_key = hashlib.sha256(system_prompt.encode()).hexdigest()[:32]

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

    iteration = 0
    for iteration in range(MAX_TURN_ITERATIONS):
        # Mid-turn user-message drain. Anything the user typed while
        # this turn was thinking lands in input_items as a regular
        # user-role message — same shape as the initial user_message
        # so the model has no special-case to learn. Echoes back via
        # on_event so any connected SSE subscriber can render the
        # inline balloon for tabs that didn't post the inject themselves.
        if drain_injections is not None:
            try:
                pending = drain_injections()
            except Exception:
                log.exception("drain_injections raised; continuing with empty drain")
                pending = []
            for inj in pending:
                inj_content = (inj.get("content") or "").strip()
                if not inj_content:
                    continue
                input_items.append({
                    "role": "user",
                    "content": inj_content,
                })
                await emit({
                    "type": "user_injection",
                    "id": inj.get("id") or "",
                    "content": inj_content,
                })

        try:
            response, cost = await client.responses_create(
                model=model,
                input=[system_msg] + input_items,
                tools=_TOOLS_PAYLOAD,
                reasoning={"effort": effort, "summary": "detailed"},
                prompt_cache_key=cache_key,
            )
            total_cost_usd += cost.total_cost_usd
            # Emit running total after each LLM call so the FE can show
            # live cost growth AND so a mid-turn cancellation has the
            # latest charge captured server-side. Without this, the
            # per-turn ledger entry on cancel would miss everything past
            # the last completed tool call.
            await emit({"type": "cost_update", "total_cost_usd": total_cost_usd})
        except asyncio.CancelledError:
            raise
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

        text_parts: List[str] = []
        function_calls: List[Any] = []

        for item in response.output:
            itype = getattr(item, "type", None)
            if itype == "reasoning":
                summary_items = list(item.summary) if item.summary else []
                summary_text = "\n".join(getattr(s, "text", "") for s in summary_items)
                if summary_text:
                    await emit({"type": "reasoning", "text": summary_text[:1200]})
                input_items.append({
                    "type": "reasoning",
                    "id": item.id,
                    "summary": [
                        {"type": s.type, "text": s.text} for s in summary_items
                    ],
                })
            elif itype == "message":
                for c in item.content:
                    if hasattr(c, "text"):
                        text_parts.append(c.text)
                input_items.append(item.model_dump(exclude_none=True))
            elif itype == "function_call":
                function_calls.append(item)
                input_items.append(item.model_dump(exclude_none=True))
            elif itype == "web_search_call":
                # Server-side tool — already executed. Keep in history to
                # preserve reasoning-item pairing.
                input_items.append(item.model_dump(exclude_none=True))
            else:
                # Unknown / future item type — keep it in history defensively.
                try:
                    input_items.append(item.model_dump(exclude_none=True))
                except Exception:
                    log.warning("unknown item type %r, skipping", itype)

        if not function_calls:
            final_text = "".join(text_parts)
            await emit({"type": "final_message", "text": final_text})
            break

        # Mid-iteration text: the model produced a message AND function
        # calls in the same response. Emit it as a `text_segment` event
        # so the FE renders it as its own segment between tool batches
        # (tool — text — tool — text — final). Previously this text
        # was silently dropped because we only emit on `final_message`
        # when no tools are called.
        mid_text = "".join(text_parts).strip()
        if mid_text:
            await emit({
                "type": "text_segment",
                "iteration": iteration,
                "text": mid_text,
            })

        # Dispatch each tool call sequentially. Parallel-tool execution is
        # a follow-up — for now agent rarely emits >1 fn call per turn at
        # this surface size.
        for fc in function_calls:
            name = fc.name
            try:
                args = json.loads(fc.arguments or "{}")
            except json.JSONDecodeError:
                args = {}

            await emit({
                "type": "tool_call_start",
                "tool_call_id": fc.call_id,
                "name": name,
                "args": args,
            })

            handler = HANDLERS.get(name)
            if not handler:
                tool_result: Dict[str, Any] = {"error": f"unknown tool {name}"}
                h_cost = 0.0
            else:
                # Long-running tools (apify) bump ctx.partial_cost_usd
                # as external cost is incurred. Reset to 0 before each
                # handler call so the running tally is per-call. On
                # CancelledError, we sum partial_cost_usd into the
                # turn total below before re-raising.
                ctx.partial_cost_usd = 0.0
                try:
                    tool_result, h_cost = await handler(args, ctx)
                except asyncio.CancelledError:
                    # User cancelled while this tool was in flight.
                    # Capture any partial external cost the tool
                    # accumulated (e.g. apify aborted mid-run still
                    # bills for compute units burned) into the turn
                    # total + emit a cost_update so _drive_agent's
                    # cancel handler flushes the right amount to the
                    # balance ledger.
                    total_cost_usd += ctx.partial_cost_usd
                    await emit({"type": "cost_update", "total_cost_usd": total_cost_usd})
                    raise
                except Exception as e:
                    log.exception("tool %s raised: %s", name, e)
                    tool_result = {"error": str(e)[:300]}
                    h_cost = 0.0

            preview = json.dumps(tool_result, default=str)[:300]
            tool_calls_made.append({
                "name": name,
                "args": args,
                "result_preview": preview,
                "cost_usd": h_cost,
            })
            total_cost_usd += h_cost

            await emit({
                "type": "tool_call_result",
                "tool_call_id": fc.call_id,
                "name": name,
                "result_preview": preview,
                "cost_usd": h_cost,
            })
            # Same running-total emit as after the LLM call — keeps the
            # cancellation safety net up-to-date as tools accumulate cost.
            await emit({"type": "cost_update", "total_cost_usd": total_cost_usd})

            input_items.append({
                "type": "function_call_output",
                "call_id": fc.call_id,
                "output": json.dumps(tool_result, default=str)[:8000],
            })
    else:
        log.warning(
            "agent loop hit MAX_TURN_ITERATIONS=%d for project %s",
            MAX_TURN_ITERATIONS, project_id,
        )

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
            await queue.put({
                "type": "turn_result",
                **{k: v for k, v in result.items() if k != "tool_calls_made"},
            })
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
