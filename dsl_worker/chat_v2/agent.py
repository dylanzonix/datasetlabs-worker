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
) -> Dict[str, Any]:
    """Run one chat turn end-to-end via the Responses API.

    Returns {final_message, tool_calls_made, total_cost_usd, iterations}.
    Emits via on_event: turn_started, reasoning, tool_call_start,
    tool_call_result, final_message, turn_complete, error.
    """
    client = _build_client()
    model = os.getenv("OPENAI_MODEL", "gpt-5.4")
    effort = os.getenv("CHAT_V2_REASONING_EFFORT", "medium")

    ctx = ToolContext(
        db=db, project_id=project_id, user_id=user_id, run_id=run_id, emit_event=None
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
        try:
            response, cost = await client.responses_create(
                model=model,
                input=[system_msg] + input_items,
                tools=_TOOLS_PAYLOAD,
                reasoning={"effort": effort, "summary": "detailed"},
                prompt_cache_key=cache_key,
            )
            total_cost_usd += cost.total_cost_usd
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
                try:
                    tool_result, h_cost = await handler(args, ctx)
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
