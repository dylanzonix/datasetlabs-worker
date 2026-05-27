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
import time
from typing import Any, AsyncIterator, Awaitable, Callable, Dict, List, Optional

from openai import AsyncOpenAI
from sqlalchemy.orm import Session

from dsl_worker.billing.pricing import get_pricing_config
from dsl_worker.billing.tracked_client import TrackedOpenAIClient
from dsl_worker.chat import (
    HANDLERS,
    TOOL_DEFS,
    ToolContext,
    build_project_state,
    build_system_prompt,
)


log = logging.getLogger(__name__)


MAX_TURN_ITERATIONS = 30  # hard safety cap on tool-call rounds per turn

# Fn calls that are pure UI side-effects (no information for the LLM to
# respond to). When an iteration's fn_calls are ALL of these, the turn
# is semantically over — don't re-run the LLM or it'll regenerate the
# same text it already streamed.
_TERMINATOR_FN_CALLS = {"suggest_replies"}


StreamEvent = Dict[str, Any]
EventCallback = Optional[Callable[[StreamEvent], Awaitable[None]]]


def _flatten_tool_defs() -> List[Dict[str, Any]]:
    """chat.completions shape {type, function:{name,description,parameters}}
    → Responses shape {type, name, description, parameters}.

    Also injects the OpenAI hosted web_search tool. It's a server-side
    tool that runs as part of the same Responses call (no sidecar LLM
    round-trip). Cost is OpenAI's flat per-search fee + the tokens it
    returns to the model.
    """
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
    # Native web_search — invoked server-side by the model when it
    # decides a query is needed. We see it come back as web_search_call
    # items in response.output (handled in the agent loop below).
    out.append({"type": "web_search"})
    return out


_TOOLS_PAYLOAD = _flatten_tool_defs()

# Per-call cost for hosted web_search — see dsl_worker/billing/web_search.py
# for the full billing model (advertised rate, sub-search multiplier,
# why we use 0.025 and not 0.010). Re-exported here so existing imports
# of WEB_SEARCH_CALL_COST_USD from agent.py keep working.
from dsl_worker.billing.web_search import (
    WEB_SEARCH_CALL_COST_USD,  # noqa: F401  (re-export)
    web_search_cost_usd,  # noqa: F401
)


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
    model = os.getenv("OPENAI_MODEL", "gpt-5.5")
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
    # Enrichment-run approvals are non-blocking: registered when the
    # agent calls enrichment_run, then emitted as a single batch at
    # end-of-turn so the user sees one summary card instead of a
    # mid-iteration freeze. table_delete / row_delete keep the old
    # blocking pattern — those are fast yes/no decisions, not
    # multi-minute cost approvals.
    pending_enrichment_chips: List[Dict[str, Any]] = []
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
    # Phase markers via the same emit channel. We don't import
    # instrumentation.phase_marker here because `emit()` already routes
    # through the run's event sink — emit a phase event directly so the
    # event carries the same mono_ns timestamp as everything else and we
    # avoid needing a ctx-shaped object in this scope.
    async def _phase(name: str, **meta: Any) -> None:
        await emit({"type": "phase", "phase": name, **meta})

    for iteration in range(MAX_TURN_ITERATIONS):
        await _phase("agent/iteration_start", iteration=iteration)
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

        llm_started = time.perf_counter()
        # IDs of web_search_call items we've already emitted live during
        # the stream — the post-call response.output loop skips emit for
        # these (still bills + adds to history).
        streamed_search_ids: set[str] = set()
        try:
            # Stream the Responses API so hosted web_search calls and
            # reasoning summaries flow to the FE live, instead of all
            # landing as one blob at the end of a multi-minute call.
            # Function-call dispatch is unchanged — we still iterate
            # response.output after the stream completes.
            stream_kwargs = {
                "model": model,
                "input": [system_msg] + input_items,
                "tools": _TOOLS_PAYLOAD,
                "reasoning": {"effort": effort, "summary": "concise"},
                "prompt_cache_key": cache_key,
            }
            raw = client.raw_client
            response = None
            async with raw.responses.stream(**stream_kwargs) as stream:
                async for event in stream:
                    etype = getattr(event, "type", None)
                    if etype == "response.reasoning_summary_text.delta":
                        delta = getattr(event, "delta", "") or ""
                        if delta:
                            await emit({"type": "reasoning", "text": delta})
                    elif etype == "response.output_text.delta":
                        # Live assistant-text delta. Stream straight to
                        # subscribers as a `text_delta` event; the runs.py
                        # router publishes a token delta on the bus. We
                        # still walk response.output post-stream to make
                        # the final-vs-mid-iteration decision (does this
                        # iteration also have function calls?).
                        delta = getattr(event, "delta", "") or ""
                        if delta:
                            await emit({"type": "text_delta", "iteration": iteration, "text": delta})
                    elif etype == "response.output_item.added":
                        added = getattr(event, "item", None)
                        if added is not None and getattr(added, "type", None) == "web_search_call":
                            item_id = getattr(added, "id", None) or ""
                            if item_id and item_id not in streamed_search_ids:
                                streamed_search_ids.add(item_id)
                                q = ""
                                action = getattr(added, "action", None)
                                if action is not None:
                                    q = getattr(action, "query", "") or ""
                                await emit({
                                    "type": "tool_call_start",
                                    "tool_call_id": item_id,
                                    "name": "web_search",
                                    "args": {"query": q} if q else {},
                                })
                    elif etype == "response.output_item.done":
                        done = getattr(event, "item", None)
                        if done is not None and getattr(done, "type", None) == "web_search_call":
                            item_id = getattr(done, "id", None) or ""
                            if item_id and item_id in streamed_search_ids:
                                q = ""
                                action = getattr(done, "action", None)
                                if action is not None:
                                    q = getattr(action, "query", "") or ""
                                await emit({
                                    "type": "tool_call_result",
                                    "tool_call_id": item_id,
                                    "name": "web_search",
                                    "result_preview": f"native web_search: {q[:120]}" if q else "native web_search",
                                    "cost_usd": WEB_SEARCH_CALL_COST_USD,
                                    "duration_ms": 0,
                                })
                response = await stream.get_final_response()

            if response is None:
                raise RuntimeError("stream ended without final response")

            # Cost compute mirrors TrackedOpenAIClient.responses_create:
            # input/output/cached tokens off response.usage, priced via
            # the loaded pricing config.
            pricing = get_pricing_config()
            usage = getattr(response, "usage", None)
            in_tok = getattr(usage, "input_tokens", 0) if usage else 0
            out_tok = getattr(usage, "output_tokens", 0) if usage else 0
            cached_tok = 0
            details = getattr(usage, "input_tokens_details", None) if usage else None
            if details is not None:
                cached_tok = getattr(details, "cached_tokens", 0) or 0
            non_cached = max(in_tok - cached_tok, 0)
            cost = pricing.calculate_cost(
                model=model,
                input_tokens=non_cached,
                output_tokens=out_tok,
                cached_input_tokens=cached_tok,
            )

            llm_ms = int((time.perf_counter() - llm_started) * 1000)
            log.info(
                "[chat timing] llm_call project=%s iter=%d duration_ms=%d cost_usd=%.6f",
                project_id, iteration, llm_ms, cost.total_cost_usd,
            )
            total_cost_usd += cost.total_cost_usd
            # Emit running total after each LLM call so the FE can show
            # live cost growth AND so a mid-turn cancellation has the
            # latest charge captured server-side. Without this, the
            # per-turn ledger entry on cancel would miss everything past
            # the last completed tool call.
            await emit({"type": "cost_update", "total_cost_usd": total_cost_usd})
            await emit({
                "type": "llm_call_complete",
                "iteration": iteration,
                "duration_ms": llm_ms,
                "cost_usd": cost.total_cost_usd,
            })
        except asyncio.CancelledError:
            raise
        except Exception as e:
            llm_ms = int((time.perf_counter() - llm_started) * 1000)
            log.info(
                "[chat timing] llm_call_failed project=%s iter=%d duration_ms=%d err=%s",
                project_id, iteration, llm_ms, str(e)[:120],
            )
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
                # Hosted tool — OpenAI ran it server-side inside this
                # same Responses call. The live tool_call_start /
                # tool_call_result events were emitted during the stream
                # (see the stream loop above), so we don't re-emit here.
                # We still bill the per-call hosted-tool fee and preserve
                # the item in history so reasoning-item pairing holds.
                query = ""
                try:
                    action = getattr(item, "action", None)
                    if action is not None:
                        query = getattr(action, "query", "") or ""
                except Exception:
                    pass
                call_id = getattr(item, "id", "") or f"web_search_{iteration}"
                total_cost_usd += WEB_SEARCH_CALL_COST_USD
                tool_calls_made.append({
                    "name": "web_search",
                    "args": {"query": query},
                    "result_preview": f"native web_search (status={getattr(item, 'status', '?')})",
                    "cost_usd": WEB_SEARCH_CALL_COST_USD,
                })
                # If the stream missed the item for some reason, emit
                # the synthesized events as a fallback so the FE still
                # renders the row in the tool log.
                if call_id not in streamed_search_ids:
                    await emit({
                        "type": "tool_call_start",
                        "tool_call_id": call_id,
                        "name": "web_search",
                        "args": {"query": query},
                    })
                    await emit({
                        "type": "tool_call_result",
                        "tool_call_id": call_id,
                        "name": "web_search",
                        "result_preview": f"native web_search: {query[:120]}",
                        "cost_usd": WEB_SEARCH_CALL_COST_USD,
                    })
                await emit({"type": "cost_update", "total_cost_usd": total_cost_usd})
                input_items.append(item.model_dump(exclude_none=True))
            else:
                # Unknown / future item type — keep it in history defensively.
                try:
                    input_items.append(item.model_dump(exclude_none=True))
                except Exception:
                    log.warning("unknown item type %r, skipping", itype)

        await _phase("agent/iteration_llm_done", iteration=iteration, fn_calls=len(function_calls))

        if not function_calls:
            final_text = "".join(text_parts)
            await emit({"type": "final_message", "text": final_text})
            break

        # Mid-iteration text: the model produced a message AND function
        # calls in the same response. Emit it as a `text_segment` event
        # so the FE renders it as its own segment between tool batches
        # (tool — text — tool — text — final). The raw (un-stripped)
        # text is sent so runs.py can trim the exact deltas it streamed
        # out of the bus accumulator — keeping the cumulative snapshot
        # equal to "what's in the FINAL text segment".
        mid_text_raw = "".join(text_parts)
        if mid_text_raw.strip():
            await emit({
                "type": "text_segment",
                "iteration": iteration,
                "text": mid_text_raw,
            })

        # Dispatch all tool calls concurrently when the model emits a batch
        # of independent function_calls in one response. Each branch gets
        # its own SessionLocal + cloned ToolContext so SQLAlchemy sessions
        # never overlap. Approval cards still serialize on the FE through
        # `approval_lock` — only one card visible at a time. The single
        # end-of-batch `cost_update` avoids the out-of-order debit race in
        # _drive_agent's incremental ledger flusher (sibling branches can
        # finish in any order; emitting only the final sum keeps the
        # cumulative monotonic).
        from dsl_api.db import SessionLocal as _SessionLocal
        from dsl_worker.chat.approvals import (
            APPROVAL_REQUIRED,
            REGISTRY as APPROVALS,
            estimate_enrichment_run_cost,
        )

        branch_dbs: List[Any] = []
        branch_ctxs: List[ToolContext] = []
        for _ in function_calls:
            _bdb = _SessionLocal()
            branch_dbs.append(_bdb)
            branch_ctxs.append(ToolContext(
                db=_bdb,
                project_id=ctx.project_id,
                user_id=ctx.user_id,
                run_id=ctx.run_id,
                emit_progress=ctx.emit_progress,
                emit_event=None,
                cancel_event=ctx.cancel_event,
                partial_cost_usd=0.0,
            ))

        # Per-iteration lock: only one approval card visible at a time on
        # the FE. Non-approval tools never touch it and run fully in
        # parallel with the approval await.
        approval_lock = asyncio.Lock()

        async def _dispatch_call(fc: Any, bctx: ToolContext) -> Dict[str, Any]:
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

            tool_started = time.perf_counter()
            handler = HANDLERS.get(name)
            tool_result: Dict[str, Any] = {}
            h_cost = 0.0
            approval_denied = False

            if not handler:
                tool_result = {"error": f"unknown tool {name}"}
            elif name == "enrichment_run":
                # NON-BLOCKING enrichment approval. Register the pending
                # approval, attach the chip metadata to the per-turn
                # list, and return a "scheduled" stub to the agent. The
                # actual run only fires when the user clicks Approve on
                # the end-of-turn chip — handled in respond_to_approval.
                # Skipping the mid-turn await + emit is what unblocks
                # the agent loop and prevents the "worker hangs on user
                # decision" wedge that caused the earlier crash.
                est_cost, summary = estimate_enrichment_run_cost(
                    args, bctx.db, bctx.project_id
                )
                pending = await APPROVALS.request(
                    project_id=bctx.project_id,
                    tool=name,
                    args=args,
                    estimated_cost_credits=est_cost,
                    summary=summary,
                )
                pending_enrichment_chips.append({
                    "approval_id": pending.id,
                    "tool": name,
                    "args": args,
                    "estimated_cost_credits": est_cost,
                    "summary": summary,
                })
                tool_result = {
                    "scheduled": True,
                    "approval_id": pending.id,
                    "estimated_cost_credits": est_cost,
                    "summary": summary,
                    "note": (
                        "Enrichment queued — pending user approval. It "
                        "will NOT run during this turn. Don't claim "
                        "results; phrase your reply as 'I've queued X — "
                        "approve below to run.'"
                    ),
                }
            elif name in APPROVAL_REQUIRED:
                # Blocking approval for the rest of APPROVAL_REQUIRED
                # (table_delete / row_delete). These are fast yes/no
                # decisions, so awaiting the future is fine.
                async with approval_lock:
                    est_cost, summary = 0.0, f"Run {name}"
                    pending = await APPROVALS.request(
                        project_id=bctx.project_id,
                        tool=name,
                        args=args,
                        estimated_cost_credits=est_cost,
                        summary=summary,
                    )
                    await emit({
                        "type": "approval_required",
                        "approval_id": pending.id,
                        "tool": name,
                        "args": args,
                        "estimated_cost_credits": est_cost,
                        "summary": summary,
                    })
                    approved = await pending.future
                    await emit({
                        "type": "approval_resolved",
                        "approval_id": pending.id,
                        "approved": approved,
                    })
                if not approved:
                    approval_denied = True
                    tool_result = {
                        "error": "denied",
                        "message": "User denied this action. Acknowledge and propose an alternative or wait for direction.",
                    }
                else:
                    try:
                        tool_result, h_cost = await handler(args, bctx)
                    except asyncio.CancelledError:
                        # Partial cost already in bctx.partial_cost_usd.
                        # Outer cancel handler aggregates across branches.
                        raise
                    except Exception as e:
                        log.exception("tool %s raised: %s", name, e)
                        tool_result = {"error": str(e)[:300]}
                        h_cost = 0.0
                        try:
                            bctx.db.rollback()
                        except Exception:
                            log.exception("post-tool-error rollback failed for %s", name)
            else:
                try:
                    tool_result, h_cost = await handler(args, bctx)
                except asyncio.CancelledError:
                    raise
                except Exception as e:
                    log.exception("tool %s raised: %s", name, e)
                    tool_result = {"error": str(e)[:300]}
                    h_cost = 0.0
                    try:
                        bctx.db.rollback()
                    except Exception:
                        log.exception("post-tool-error rollback failed for %s", name)

            tool_ms = int((time.perf_counter() - tool_started) * 1000)
            log.info(
                "[chat timing] tool project=%s tool=%s duration_ms=%d cost_usd=%.6f",
                project_id, name, tool_ms, h_cost,
            )

            preview = json.dumps(tool_result, default=str)
            await emit({
                "type": "tool_call_result",
                "tool_call_id": fc.call_id,
                "name": name,
                "result_preview": preview,
                "cost_usd": h_cost,
                "duration_ms": tool_ms,
            })

            return {
                "fc": fc,
                "name": name,
                "args": args,
                "tool_result": tool_result,
                "h_cost": h_cost,
                "tool_ms": tool_ms,
                "preview": preview,
                "approval_denied": approval_denied,
            }

        # Spawn one task per call; gather waits for the whole batch.
        # return_exceptions=False so CancelledError surfaces here and we
        # can sum partial costs from every branch's bctx before re-raising.
        dispatch_tasks = [
            asyncio.create_task(_dispatch_call(fc, bctx))
            for fc, bctx in zip(function_calls, branch_ctxs)
        ]
        try:
            results = await asyncio.gather(*dispatch_tasks)
        except asyncio.CancelledError:
            # Stop-button cancel landed mid-batch. Each branch raised
            # CancelledError out of its handler (or hadn't started yet);
            # bctx.partial_cost_usd carries any spend the in-flight tool
            # accrued before the cancel cut it off. Cancel any siblings
            # still running, drain them, then sum partials so the outer
            # _drive_agent flushes the right amount to the ledger.
            for t in dispatch_tasks:
                if not t.done():
                    t.cancel()
            for t in dispatch_tasks:
                try:
                    await t
                except (asyncio.CancelledError, Exception):
                    pass
            for bctx_i in branch_ctxs:
                total_cost_usd += bctx_i.partial_cost_usd
            await emit({"type": "cost_update", "total_cost_usd": total_cost_usd})
            for bdb in branch_dbs:
                try:
                    bdb.close()
                except Exception:
                    log.exception("branch db close failed during cancel")
            raise

        # Walk results in the model's emission order so input_items keeps
        # the canonical fc/output pairing for the next iteration.
        results_by_call_id = {r["fc"].call_id: r for r in results}
        for fc in function_calls:
            r = results_by_call_id[fc.call_id]
            entry: Dict[str, Any] = {
                "name": r["name"],
                "args": r["args"],
                "result_preview": r["preview"],
                "cost_usd": r["h_cost"],
                "duration_ms": r["tool_ms"],
            }
            if r["approval_denied"]:
                entry["approval"] = "denied"
            tool_calls_made.append(entry)
            total_cost_usd += r["h_cost"]
            input_items.append({
                "type": "function_call_output",
                "call_id": fc.call_id,
                "output": json.dumps(r["tool_result"], default=str)[:8000],
            })

        # One cost_update at the end of the batch — keeps _drive_agent's
        # incremental debit monotonic (sibling branches finishing out of
        # order would otherwise emit decreasing totals).
        await emit({"type": "cost_update", "total_cost_usd": total_cost_usd})

        for bdb in branch_dbs:
            try:
                bdb.close()
            except Exception:
                log.exception("branch db close failed")

        # UI-only side-effect tools (suggest_replies) don't represent
        # work the LLM should respond to — they just emit SSE events for
        # the FE. When the only fn_calls in this iteration were those,
        # the turn is semantically over. Letting the loop run again just
        # gives the LLM a free turn with no new info and it regenerates
        # the same intro text (observed in prod, project 70e437bc).
        # UNLESS the model produced zero text alongside the tool call —
        # in that case let the loop continue so it gets the tool result
        # and has another chance to produce an actual reply.
        if all(fc.name in _TERMINATOR_FN_CALLS for fc in function_calls):
            if mid_text_raw.strip():
                final_text = mid_text_raw
                break
            log.warning(
                "agent: terminator-only response with no text for project %s iter %d — continuing",
                project_id, iteration,
            )
    else:
        log.warning(
            "agent loop hit MAX_TURN_ITERATIONS=%d for project %s",
            MAX_TURN_ITERATIONS, project_id,
        )

    # Emit deferred enrichment approval chips at end-of-turn. The agent
    # already saw `{scheduled: true}` for each one and shaped its reply
    # around that; now the FE renders the chips as a turn summary so the
    # user can approve / decline without the worker holding a Future
    # open across the user's decision window.
    for chip in pending_enrichment_chips:
        await emit({
            "type": "approval_required",
            "approval_id": chip["approval_id"],
            "tool": chip["tool"],
            "args": chip["args"],
            "estimated_cost_credits": chip["estimated_cost_credits"],
            "summary": chip["summary"],
        })

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
