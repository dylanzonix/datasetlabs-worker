"""rows_fill — per-cell mini-agent for chat-mode.

When the user (via the chat agent) calls `rows_fill(columns, where, ...)`,
each matching row spawns a small bounded subagent that:
  - sees the row's existing column values
  - sees the goal (one or more column names + their `format` + `description`)
  - has access to all source tools (FE / Apollo / Apify / Google Maps /
    code_exec / browser_use / web_search built-in)
  - has a `set_values(values)` tool to commit one or more cell values
  - has a `give_up(reason)` tool to bail
  - is capped at ~5 turns and ~$max_cost per cell

Concurrency: up to N cells run in parallel (asyncio.Semaphore). Each
cell holds its own SQLAlchemy session so writes don't collide.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple

from sqlalchemy.orm import Session

from dsl_api.config import settings as _api_settings
from dsl_api.db import SessionLocal
from dsl_api.models import Project
from dsl_api.models.sample import Sample

from dsl_worker import skills as skills_loader
from dsl_worker.chat_api import cell_traces, sources

ProgressCallback = Callable[[Dict[str, Any]], Awaitable[None]]


log = logging.getLogger(__name__)


# Concurrent cell agents. Continuous via asyncio.Semaphore + gather —
# one straggler doesn't block the next slot. Bumped from 5 to 10 to
# get more parallelism on long fills; per-source rate limits (Apollo,
# FullEnrich) are still the real ceiling.
_CELL_CONCURRENCY = 10

# Per-cell budget tiers, keyed off the user's effort selection in the chat
# input. The cell agent stops calling tools once cost_usd >= max_cost, so
# this is purely a safety cap — most cells finish well under it.
#
# fast: cheap lookups (existing-fields-derive, one quick web_search)
# balanced: typical, room for ~10 web_searches OR one Apollo enrichment
# highest: covers FE phones (~$0.55) and other expensive single calls
_TIER_MAX_COST = {
    "auto": 0.30,
    "fast": 0.10,
    "balanced": 0.30,
    "highest": 1.00,
}

# Hard ceiling on per-cell turns regardless of cost. The cost cap usually
# fires first; this just prevents pathological infinite loops.
_HARD_TURN_LIMIT = 10


def tier_default_max_cost(effort: Optional[str]) -> float:
    """Resolve the per-cell budget from the user's effort tier.

    If a cheaper mini model is configured for cell agents
    (OPENAI_MODEL_MINI), bump the budget — the same per-cell dollar cap
    translates to more model + tool calls, which is the actual lever for
    success rate on hard enrichments (more web_search retries, more
    candidate sources tried). Net effect: same or lower turn-level spend,
    higher fill success.
    """
    base = _TIER_MAX_COST.get(effort or "balanced", _TIER_MAX_COST["balanced"])
    if _api_settings.OPENAI_MODEL_MINI:
        return base * 2.5
    return base

# Subset of the chat-mode tool surface that the cell agent gets. All the
# row/column management tools are excluded — a cell agent should NOT
# create rows or columns. It only researches and commits its assigned
# cells.
_CELL_TOOL_NAMES = {
    "fullenrich_search_people",
    "fullenrich_search_companies",
    "fullenrich_enrich_contacts",
    "apollo_search_companies",
    "apollo_enrich_person",
    "apollo_enrich_company",
    "apify_search_actors",
    "apify_actor_details",
    "apify_call_actor",
    "google_maps_search_places",
    "google_maps_place_details",
    "code_exec",
    "browser_use",
}
# web_harvest is intentionally NOT in the cell-agent toolset. It's a
# multi-search discovery tool meant for orchestrator-level "iterate
# across the web to find entities", not per-cell research. Per-cell
# use would just burn budget on broad searches when targeted
# enrichment APIs (FE/Apollo/Apify/GMaps) or web_search exist.


_CELL_AGENT_SYSTEM_PROMPT = """\
You are a research subagent. Your job is to fill specific column values
for ONE row of a dataset. The row's existing fields are your context;
the column goals tell you what to find.

Process:
1. Look at the row's existing fields and figure out the simplest path to
   the answer.
2. If the answer is already obvious from existing fields (e.g. you can
   derive a domain from a website URL), call set_values with the result
   and exit immediately.
3. Otherwise, use the source tools — pick the cheapest / most direct
   one. Rough hierarchy:
     - FullEnrich for verified contact info (email/phone/LinkedIn)
     - Apollo as fallback for people/company enrichment
     - Apify for site-specific scraping (22k+ actors, prefer over web)
     - Google Maps for local businesses with a physical address
     - web_search for quick factual lookups
     - browser_use only as a true last resort (slow, $0.10-0.50/call;
       use only when no source/Apify actor fits and the page needs JS
       rendering or login).
4. Once you have the value(s), call set_values to commit and stop.

Critical rules:
- Real data only. A confident null beats a wrong value. If you can't
  find a real answer, call set_values with null (or omit the field)
  and explain in 'reason'. Don't fabricate.
- Match the column's `format` exactly when stated.
- One or two tool calls per turn max. Don't chain a research odyssey.
- You are CAPPED at a tight budget. Don't waste calls.
- **Bail early on dead ends.** If your first tool call returns an
  error or empty data, switch tactics ONCE. If your second attempt
  also returns nothing useful, call `give_up` immediately — don't
  keep paying for retries that won't yield. A clean give_up after 2
  bad turns is better than burning the full per-cell budget producing
  nothing.
- **Always cite per-cell sources.** When you call set_values, include
  the `sources` arg mapping each filled column to the URL(s) that
  justified the value (the web_search result URL, the page
  browser_use loaded, the Apollo profile URL, etc.). The frontend
  renders these as audit citations under each cell. Skip sources for
  columns you set to null. Don't fabricate URLs — only cite ones you
  actually saw in tool results.
- **Don't fan out org-level info into per-person columns.** If the
  Org has a generic phone like "(555) 123-4567" or a generic email
  like "info@example.com", do NOT paste it into Contact 1 / Contact 2
  / Contact 3 phone/email slots. Per-person columns require
  per-person evidence (their name in a bio, a personal email like
  jane.doe@..., a direct line on a staff page). When in doubt,
  null > guess.
- ALWAYS finish by calling either `set_values` or `give_up`. Don't trail
  off.
"""


@dataclass
class CellFillResult:
    row_id: str
    values: Dict[str, Any] = field(default_factory=dict)
    # Per-column citations. Maps column_name -> list of source dicts in
    # the same shape rows_add uses: [{"type": "url", "value": "..."}].
    # Populated from the cell agent's set_values(sources=...) arg and
    # persisted to sample.tags["sources"][col] so the FE can render
    # per-cell "where did this come from" tooltips.
    sources: Dict[str, List[Dict[str, str]]] = field(default_factory=dict)
    # filled — agent committed real values
    # null_legitimate — agent looked, returned null deliberately
    # error — agent crashed / source errored
    # budget_exhausted — agent ran but hit its OWN per-cell cost cap
    # no_op — agent ended without committing
    # deferred — system skipped this cell because of a TURN-level
    #            decision (sample-and-project projection too high,
    #            or cumulative turn cost crossed the soft cap before
    #            this cell could start). Distinct from budget_exhausted
    #            so the FE can label it "Left empty for now" — these
    #            cells weren't tried, just postponed.
    status: str = "filled"
    reason: Optional[str] = None
    cost_usd: float = 0.0
    turns: int = 0
    # Forensic transcript of this cell's run — populated as the agent
    # progresses through turns. Persisted by fill_rows to a per-batch
    # blob (cell_traces module). The chat agent never sees the full
    # trace inline; it inspects via cell_traces_inspect on demand.
    trace: Optional[cell_traces.CellTrace] = None


def _cell_tools_for_columns(target_columns: List[str]) -> List[Dict[str, Any]]:
    """Build the cell agent's tool list: source tools + set_values + give_up."""
    source_tools = [
        t for t in sources.SOURCE_TOOLS
        if t.get("name") in _CELL_TOOL_NAMES
    ]
    # OpenAI built-in web search
    builtins: List[Dict[str, Any]] = [
        {"type": "web_search", "search_context_size": "low"},
    ]
    set_values_schema = {
        "type": "function",
        "name": "set_values",
        "description": (
            "Commit final values for the cell(s) you were assigned. Pass a "
            "dict of column_name -> value. Use null for legitimate "
            "no-result cases. After calling this, stop — the run is done.\n\n"
            "**ALSO pass `sources`** — for every non-null value, list the "
            "URL(s) where you found it (the web_search result, "
            "browser_use page, Apollo profile, etc.). The frontend "
            "renders these as per-cell citations so the user can audit "
            "where each value came from."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "values": {
                    "type": "object",
                    "properties": {
                        col: {"description": f"Value for the {col!r} column. Use null if you couldn't find a real value."}
                        for col in target_columns
                    },
                    "additionalProperties": True,
                },
                "sources": {
                    "type": "object",
                    "description": (
                        "Per-column citations. Map column_name -> list "
                        "of URLs that justify the value. REQUIRED for "
                        "every non-null value in `values`. Skip the key "
                        "for columns set to null. Use only URLs you "
                        "actually visited or saw in tool results — "
                        "don't fabricate citations."
                    ),
                    "properties": {
                        col: {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": f"URLs supporting the {col!r} value.",
                        }
                        for col in target_columns
                    },
                    "additionalProperties": True,
                },
                "reason": {"type": "string", "description": "Optional one-line note on what you did or why a value is null."},
            },
            "required": ["values"],
        },
    }
    give_up_schema = {
        "type": "function",
        "name": "give_up",
        "description": "Bail on this row entirely — call when you cannot proceed (e.g. all source calls erroring). Stop after.",
        "parameters": {
            "type": "object",
            "properties": {"reason": {"type": "string"}},
            "required": ["reason"],
        },
    }
    return source_tools + builtins + [set_values_schema, give_up_schema]


def _row_context(row_data: Dict[str, Any], target_columns: List[str], target_specs: Dict[str, Dict[str, str]]) -> str:
    """Render the per-cell user message: row fields + column goals."""
    parts = ["# Existing row fields"]
    if not row_data:
        parts.append("(none)")
    else:
        for k, v in row_data.items():
            if v is None or v == "":
                continue
            s = json.dumps(v, default=str)
            if len(s) > 200:
                s = s[:200] + "..."
            parts.append(f"- {k}: {s}")
    parts.append("")
    parts.append(f"# Columns to fill ({len(target_columns)})")
    for col in target_columns:
        spec = target_specs.get(col, {})
        line = f"- {col}"
        if spec.get("format"):
            line += f" — format: {spec['format']}"
        if spec.get("description"):
            line += f" — {spec['description']}"
        parts.append(line)
    return "\n".join(parts)


async def _run_cell_agent(
    *,
    row_id: str,
    row_data: Dict[str, Any],
    target_columns: List[str],
    target_specs: Dict[str, Dict[str, str]],
    max_cost: float,
    extra_system: Optional[str] = None,
    skills_applied: Optional[List[str]] = None,
    progress_cb: Optional[ProgressCallback] = None,
    cell_idx: Optional[int] = None,
    cell_total: Optional[int] = None,
) -> CellFillResult:
    """Spawn the bounded subagent for one row and return its CellFillResult.

    `extra_system`: optional snippet appended to the cell-agent system
    prompt. Used to inject matched skills (per-fill, based on the target
    columns). Kept short — skills are markdown bullet rules, not essays.
    """
    from openai import AsyncOpenAI

    result = CellFillResult(row_id=row_id)
    result.trace = cell_traces.new_trace(row_id=row_id, columns=target_columns)
    if skills_applied:
        result.trace.skills_applied = list(skills_applied)
    client = AsyncOpenAI(api_key=_api_settings.OPENAI_API_KEY)

    def _done() -> CellFillResult:
        # Mirror terminal CellFillResult fields onto the trace so the
        # forensic record matches what the chat agent ultimately sees.
        # Called from every return path in this function — keeps trace
        # finalization in one place instead of scattered through six
        # exits.
        if result.trace is not None:
            result.trace.status = result.status
            result.trace.reason = result.reason
            result.trace.values = dict(result.values)
            result.trace.cost_usd = result.cost_usd
            result.trace.turns_used = result.turns
            result.trace.ended_at = datetime.now(timezone.utc).isoformat()
        return result

    user_msg = _row_context(row_data, target_columns, target_specs)
    cell_tools = _cell_tools_for_columns(target_columns)
    next_input: Any = [{"role": "user", "content": user_msg}]
    previous_response_id: Optional[str] = None
    # Tracks consecutive turns where every source-tool call returned an
    # error (or no source tool was called at all). When this hits 2, we
    # force-bail the cell — see the early-exit block at the bottom of
    # the loop. Cells that can't get a single productive tool result in
    # 2 turns aren't going to yield, and continuing burns budget.
    unproductive_streak = 0
    # One-shot escape hatch when the model returns text instead of a
    # tool call. Empirically this happens A LOT — project a7b01689 saw
    # 47 of 60 cells (78%) bail as no_op on turn 1, each costing ~$0.07
    # of reasoning + output for zero output. The re-prompt converts most
    # of those into a real set_values(null) or give_up, so the cost
    # actually buys a status badge instead of being burned.
    no_op_retried = False

    for turn_idx in range(_HARD_TURN_LIMIT):
        result.turns = turn_idx + 1

        if result.cost_usd >= max_cost:
            result.status = "budget_exhausted"
            cap_credits = max_cost / 0.1
            cap_str = (
                f"{round(cap_credits)} credits"
                if cap_credits >= 10
                else f"{cap_credits:.1f} credits"
            )
            result.reason = (
                f"Couldn't finish researching this cell in time — "
                f"used the per-cell budget ({cap_str}) without finding "
                f"a confident answer."
            )
            break

        try:
            # Cell agents use the cheaper mini model when configured.
            # Falls back to the main OPENAI_MODEL when OPENAI_MODEL_MINI
            # is empty so behavior is unchanged unless the env is set.
            cell_model = _api_settings.OPENAI_MODEL_MINI or _api_settings.OPENAI_MODEL
            kwargs: Dict[str, Any] = {
                "model": cell_model,
                "input": next_input,
                "tools": cell_tools,
                "max_output_tokens": 2000,
            }
            if turn_idx == 0:
                instructions = _CELL_AGENT_SYSTEM_PROMPT
                if extra_system:
                    instructions = instructions + "\n\n" + extra_system.strip() + "\n"
                kwargs["instructions"] = instructions
            else:
                kwargs["previous_response_id"] = previous_response_id
            resp = await client.responses.create(**kwargs)
        except Exception as e:
            result.status = "error"
            result.reason = f"{type(e).__name__}: {e}"
            if result.trace is not None:
                result.trace.turn_log.append(cell_traces.CellTraceTurn(
                    turn=turn_idx + 1,
                    kind="tool_call",
                    name="responses.create",
                    error=f"{type(e).__name__}: {e}",
                ))
            return _done()

        result.cost_usd += sources._response_cost(resp, model=cell_model)
        previous_response_id = resp.id

        function_calls: List[Any] = []
        for item in resp.output:
            if item.type == "web_search_call":
                # Cell agent uses search_context_size="low" (see _cell_tools_for_columns).
                result.cost_usd += sources.WEB_SEARCH_USD_BY_TIER["low"]
                if result.trace is not None:
                    result.trace.add_web_search(
                        turn=turn_idx + 1,
                        cost_usd=sources.WEB_SEARCH_USD_BY_TIER["low"],
                    )
                # Surface to the toolLog so the user sees the cell did a
                # web search this turn. Timing is retroactive — the
                # built-in already ran inside the OpenAI response — but
                # most cells web_search before they call set_values, so
                # this still flashes before the next per-cell event.
                if progress_cb is not None and cell_idx is not None:
                    try:
                        await progress_cb({
                            "type": "cell_tool",
                            "row_id": str(row_id),
                            "index": cell_idx,
                            "total": cell_total,
                            "tool": "web_search",
                        })
                    except Exception:
                        log.exception("progress_cb cell_tool (web_search) raised; suppressing")
            elif item.type == "function_call":
                function_calls.append(item)

        if not function_calls:
            # Model returned text without committing. Give it ONE more
            # turn with an explicit nudge before bailing — cheaper than
            # eating the no_op cost for a non-result. After re-prompt,
            # bail for real if the model still won't call a tool.
            if not no_op_retried:
                no_op_retried = True
                next_input = [
                    {
                        "role": "user",
                        "content": (
                            "You returned text without calling any tool. "
                            "You MUST commit by calling `set_values` "
                            "(use null for any value you couldn't find) "
                            "or `give_up` if you genuinely can't proceed. "
                            "Do not respond with text — call a tool now."
                        ),
                    }
                ]
                previous_response_id = resp.id
                continue
            result.status = "no_op"
            result.reason = (
                "agent ended without calling set_values "
                "(re-prompted once, still wouldn't commit)"
            )
            if result.trace is not None:
                result.trace.add_no_op(
                    turn=turn_idx + 1,
                    note="agent returned text (no tool call) after re-prompt",
                )
            return _done()

        # Process tool calls
        tool_outputs: List[Dict[str, Any]] = []
        terminated = False
        # True once any source tool returns a non-error result this turn.
        # Drives the unproductive_streak counter; cells where every call
        # errors or short-circuits to budget_exhausted don't bump this.
        had_productive_source_call = False
        for fc in function_calls:
            try:
                fc_args = json.loads(fc.arguments) if fc.arguments else {}
            except json.JSONDecodeError:
                tool_outputs.append({"type": "function_call_output", "call_id": fc.call_id, "output": "Error: invalid arguments"})
                if result.trace is not None:
                    result.trace.add_tool_call(
                        turn=turn_idx + 1,
                        name=fc.name,
                        args=fc.arguments,
                        result=None,
                        error="invalid JSON arguments",
                    )
                continue

            if fc.name == "set_values":
                values = fc_args.get("values") or {}
                if not isinstance(values, dict):
                    tool_outputs.append({"type": "function_call_output", "call_id": fc.call_id, "output": "Error: values must be an object"})
                    if result.trace is not None:
                        result.trace.add_tool_call(
                            turn=turn_idx + 1,
                            name="set_values",
                            args=fc_args,
                            result=None,
                            error="values must be an object",
                        )
                    continue
                # Filter to target columns only — the agent might emit extras
                clean = {k: v for k, v in values.items() if k in target_columns}
                result.values = clean
                result.reason = fc_args.get("reason") or None
                # Capture per-cell sources. Agent passes
                # {column_name: ["url1", "url2", ...]}; normalize to the
                # rows_add wire shape so persistence is uniform across
                # rows_fill and rows_add.
                raw_sources = fc_args.get("sources") or {}
                if isinstance(raw_sources, dict):
                    for col, urls in raw_sources.items():
                        if col not in target_columns:
                            continue
                        # Skip sources for columns we set to null —
                        # storing citations for empty cells is noise.
                        if clean.get(col) is None or clean.get(col) == "":
                            continue
                        if not isinstance(urls, list):
                            continue
                        normed = [
                            {"type": "url", "value": str(u).strip()}
                            for u in urls
                            if isinstance(u, str) and u.strip()
                        ]
                        if normed:
                            result.sources[col] = normed
                # Determine status from values
                non_null_count = sum(1 for v in clean.values() if v is not None and v != "")
                result.status = "filled" if non_null_count > 0 else "null_legitimate"
                terminated = True
                tool_outputs.append({"type": "function_call_output", "call_id": fc.call_id, "output": "ok"})
                if result.trace is not None:
                    result.trace.add_tool_call(
                        turn=turn_idx + 1,
                        name="set_values",
                        args={
                            "values": clean,
                            "reason": result.reason,
                            "sources": result.sources or None,
                        },
                        result="ok",
                    )

            elif fc.name == "give_up":
                result.status = "error"
                result.reason = fc_args.get("reason") or "agent gave up"
                terminated = True
                tool_outputs.append({"type": "function_call_output", "call_id": fc.call_id, "output": "ok"})
                if result.trace is not None:
                    result.trace.add_tool_call(
                        turn=turn_idx + 1,
                        name="give_up",
                        args=fc_args,
                        result="ok",
                    )

            elif fc.name in sources._HANDLERS:
                # Source tool — execute and feed result back
                if result.cost_usd >= max_cost:
                    tool_outputs.append({"type": "function_call_output", "call_id": fc.call_id, "output": "budget_exhausted; finish via set_values or give_up"})
                    if result.trace is not None:
                        result.trace.add_tool_call(
                            turn=turn_idx + 1,
                            name=fc.name,
                            args=fc_args,
                            result="skipped — budget_exhausted",
                        )
                    continue
                # Surface what the cell agent is about to do — source
                # tools (apollo_enrich, browser_use, fullenrich) take
                # 10-60s and would otherwise show as dead air between
                # cell_start and cell_done. The FE renders this as
                # "cell N/M → tool_name…" in the toolLog summary.
                if progress_cb is not None and cell_idx is not None:
                    try:
                        await progress_cb({
                            "type": "cell_tool",
                            "row_id": str(row_id),
                            "index": cell_idx,
                            "total": cell_total,
                            "tool": fc.name,
                        })
                    except Exception:
                        log.exception("progress_cb cell_tool raised; suppressing")
                try:
                    out_text, tool_cost = await sources.execute_source_tool(fc.name, fc_args)
                    result.cost_usd += tool_cost
                    tool_outputs.append({"type": "function_call_output", "call_id": fc.call_id, "output": out_text[:6000]})
                    had_productive_source_call = True
                    if result.trace is not None:
                        result.trace.add_tool_call(
                            turn=turn_idx + 1,
                            name=fc.name,
                            args=fc_args,
                            result=out_text,
                            cost_usd=tool_cost,
                        )
                except Exception as e:
                    tool_outputs.append({"type": "function_call_output", "call_id": fc.call_id, "output": f"error: {type(e).__name__}: {e}"})
                    if result.trace is not None:
                        result.trace.add_tool_call(
                            turn=turn_idx + 1,
                            name=fc.name,
                            args=fc_args,
                            result=None,
                            error=f"{type(e).__name__}: {e}",
                        )

            else:
                tool_outputs.append({"type": "function_call_output", "call_id": fc.call_id, "output": f"unknown tool: {fc.name}"})
                if result.trace is not None:
                    result.trace.add_tool_call(
                        turn=turn_idx + 1,
                        name=fc.name,
                        args=fc_args,
                        result=None,
                        error="unknown tool",
                    )

        if terminated:
            return _done()

        # Early-bail backstop. The cell-agent prompt already tells the
        # model to give_up after 2 dead-end turns; this is the safety
        # net for when it doesn't. After 2 consecutive turns where no
        # source tool returned a non-error result, force-bail rather
        # than burn the rest of the per-cell budget on calls that
        # aren't going to yield. Reset on any productive turn.
        if had_productive_source_call:
            unproductive_streak = 0
        else:
            unproductive_streak += 1
        if unproductive_streak >= 2:
            result.status = "error"
            result.reason = (
                "Bailed early: 2 consecutive turns with no useful tool "
                "results — stopped to avoid burning the per-cell budget "
                "on a cell that won't yield."
            )
            return _done()

        next_input = tool_outputs

    # Out of turns without termination
    if result.status == "filled":
        return _done()  # was set in the last turn
    result.status = "budget_exhausted"
    result.reason = result.reason or f"reached turn ceiling without calling set_values"
    return _done()


async def fill_rows(
    *,
    project: Project,
    target_columns: List[str],
    where_sql: str,
    where_params: Dict[str, Any],
    limit: Optional[int],
    max_cost: float = 0.30,
    concurrency: int = _CELL_CONCURRENCY,
    progress_cb: Optional[ProgressCallback] = None,
    retry_failed: bool = False,
) -> Tuple[Dict[str, Any], float]:
    """Run cell agents over all matching rows.

    Returns (summary_dict, total_cost_usd).

    No programmatic budget enforcement here. All matching rows run to
    completion, subject only to per-cell `max_cost` (the cell agent's
    own budget cap) and the user's actual balance (out_of_credits at
    the meter level). The agent is expected to call `confirm_budget`
    BEFORE this when it suspects the fanout will be expensive — see
    the chat agent's prompt.
    """
    # One run_id per fill batch — drives the cell trace filename so the
    # chat agent can inspect this exact run via cell_traces_inspect. Also
    # surfaced in the summary so the LLM has a stable handle.
    run_id = uuid.uuid4().hex[:12]

    # Build column specs from project.columns
    project_columns = {
        c.get("name"): c for c in (project.columns or []) if isinstance(c, dict)
    }
    missing = [c for c in target_columns if c not in project_columns]
    if missing:
        return (
            {"error": f"columns not found: {missing}. Add them with columns_add first."},
            0.0,
        )

    target_specs: Dict[str, Dict[str, str]] = {}
    for col in target_columns:
        spec = project_columns.get(col, {}) or {}
        target_specs[col] = {
            "format": spec.get("format") or "",
            "description": spec.get("description") or "",
        }

    # Skills matching: once per fill batch (same target columns for every
    # row). Builds the prompt extension and a list of skill names to record
    # in each cell trace, so we can correlate "which skills were active"
    # with which cells succeeded/failed when iterating on the playbook.
    skill_columns_for_match = [
        {
            "name": col,
            "description": target_specs[col]["description"],
            "format": target_specs[col]["format"],
        }
        for col in target_columns
    ]
    try:
        matched_skills = skills_loader.match_skills("cell_agent", skill_columns_for_match)
        skills_extra_system = skills_loader.render_skills(matched_skills)
        skills_applied_names = [s.name for s in matched_skills]
    except Exception:
        log.exception("skills loader failed (continuing without skills)")
        matched_skills = []
        skills_extra_system = ""
        skills_applied_names = []

    # Fetch rows in a short-lived session — we'll then spawn per-cell sessions.
    db = SessionLocal()
    try:
        from sqlalchemy import text
        version_id = project.current_version_id
        if not version_id:
            return ({"error": "project has no version yet"}, 0.0)
        sql = (
            f"SELECT id, row, tags FROM samples WHERE version_id = :vid "
            f"AND deleted_at IS NULL AND ({where_sql}) "
            f"ORDER BY seq"
        )
        if limit is not None:
            sql += f" LIMIT {int(limit)}"
        rows = db.execute(text(sql), {"vid": version_id, **where_params}).all()
    finally:
        db.close()

    if not rows:
        return ({"matched_rows": 0, "filled": 0, "summary": "no rows match"}, 0.0)

    # Pre-filter: skip rows where ALL target columns are already filled,
    # and per-row determine the SUBSET of target_columns the cell agent
    # actually needs to fill. Cells are write-once at this layer — to
    # re-fill, the caller clears values via rows_update or
    # columns_delete + re-add. This makes rows_fill idempotent (re-runs
    # are no-ops on already-filled cells) and removes the footgun where
    # the model called rows_fill twice with no `where` and re-classified
    # the same first 80 rows (project 051e8704). Lifts state-tracking
    # off the model and into the system.
    #
    # Second pre-filter: when retry_failed=False (default), drop columns
    # whose tags.fill_status[col].status indicates a prior terminal-fail
    # attempt (null_legitimate). Without this, calling rows_fill twice
    # over the same window with the same strategy retries the SAME
    # rows that already failed, producing the SAME nulls — that was
    # the project f34982fd regression where the second bulk_first call
    # re-attempted 6 already-failed rows for $1.03 of waste. Allow
    # retry by setting retry_failed=true (different approach in mind).
    def _is_empty(v: Any) -> bool:
        if v is None:
            return True
        if isinstance(v, str) and v.strip() == "":
            return True
        return False

    work_items: List[Tuple[Any, Dict[str, Any], List[str]]] = []
    rows_skipped_already_filled = 0
    rows_skipped_prior_fail = 0
    for row_id, row_data, tags in rows:
        existing = dict(row_data or {})
        fill_status = (tags or {}).get("fill_status") or {}
        unfilled: List[str] = []
        per_row_skipped_fail = 0
        for c in target_columns:
            if not _is_empty(existing.get(c)):
                continue
            if not retry_failed:
                prior = fill_status.get(c) or {}
                if isinstance(prior, dict) and prior.get("status") == "null_legitimate":
                    per_row_skipped_fail += 1
                    continue
            unfilled.append(c)
        if unfilled:
            work_items.append((row_id, existing, unfilled))
        elif per_row_skipped_fail > 0:
            rows_skipped_prior_fail += 1
        else:
            rows_skipped_already_filled += 1

    if not work_items:
        # Either every matched row was already filled, or every row was
        # skipped because of prior-fail markers. Distinct notes — the
        # model needs to know which so it doesn't re-call.
        if rows_skipped_prior_fail and not rows_skipped_already_filled:
            note = (
                "All matched rows have a prior null_legitimate "
                "fill_status on the target column(s) — they were "
                "already attempted and yielded null. Skipping retries "
                "by default to avoid re-paying for the same null. To "
                "force a retry (e.g. you've changed approach), call "
                "with retry_failed=true. To advance to a different "
                "row window, pass start_seq/end_seq."
            )
        elif rows_skipped_prior_fail:
            note = (
                f"{rows_skipped_already_filled} matched rows already "
                f"have values; the remaining "
                f"{rows_skipped_prior_fail} were skipped because of "
                f"prior null_legitimate fill_status on the target "
                f"column(s). Pass retry_failed=true to retry them, "
                f"or start_seq/end_seq to target a fresh row window."
            )
        else:
            note = (
                "All matched rows already have values in the target "
                "columns. To re-fill, clear the existing values first "
                "(rows_update with the column → null, or "
                "columns_delete + columns_add)."
            )
        summary = {
            "matched_rows": len(rows),
            "rows_skipped_already_filled": rows_skipped_already_filled,
            "rows_skipped_prior_fail": rows_skipped_prior_fail,
            "processed": 0,
            "cells_filled": 0,
            "by_status": {},
            "note": note,
        }
        return (summary, 0.0)

    sem = asyncio.Semaphore(concurrency)
    total_cost = 0.0
    results: List[CellFillResult] = []

    async def _emit(event: Dict[str, Any]) -> None:
        if progress_cb is None:
            return
        try:
            await progress_cb(event)
        except Exception:
            log.exception("progress_cb raised; suppressing")

    if progress_cb is not None:
        await _emit({
            "type": "fill_start",
            "total": len(work_items),
            "columns": list(target_columns),
        })

    async def _process(
        row_id: Any, row_data: Dict[str, Any], unfilled_cols: List[str], idx: int,
    ) -> CellFillResult:
        async with sem:
            await _emit({
                "type": "cell_start",
                "row_id": str(row_id),
                "index": idx,
                "total": len(work_items),
                # FE marks each (row_id, column) as "processing" so the
                # table can show per-cell spinners while the cell agent
                # runs. cell_done clears these. Only the unfilled subset
                # for this row.
                "columns": list(unfilled_cols),
            })
            # Per-row specs are sliced to the unfilled subset so the
            # cell agent only researches what's actually needed.
            row_specs = {c: target_specs[c] for c in unfilled_cols if c in target_specs}
            res = await _run_cell_agent(
                row_id=str(row_id),
                row_data=row_data,
                target_columns=unfilled_cols,
                target_specs=row_specs,
                max_cost=max_cost,
                extra_system=skills_extra_system or None,
                skills_applied=skills_applied_names or None,
                progress_cb=progress_cb,
                cell_idx=idx,
                cell_total=len(work_items),
            )
            # Persist this cell's values immediately so:
            #  (a) the table updates live as cells fill, not in one
            #      batch at the end, and
            #  (b) if rows_fill is interrupted, partial progress is in
            #      the DB — re-running with where={col: null} resumes
            #      naturally.
            #
            # Also persist a per-column fill_status entry under tags for
            # any column that was attempted but ended up empty (legit
            # null, error, budget_exhausted, no_op). The frontend reads
            # these to render a "couldn't fill — here's why" badge so an
            # empty cell isn't silently empty.
            updated_row: Optional[Dict[str, Any]] = None
            failed_cols: Dict[str, Dict[str, Any]] = {}
            for col in unfilled_cols:
                v = res.values.get(col) if res.values else None
                if v is None or v == "":
                    failed_cols[col] = {
                        "status": res.status,
                        "reason": res.reason or None,
                        "cost": round(res.cost_usd, 4),
                        # Strategy tag drives the skip-prior-fail
                        # pre-filter on subsequent calls. "per_cell"
                        # here; bulk_browser writes "bulk_browser".
                        "strategy": "per_cell",
                    }
            # Tracks which columns the agent committed real values for —
            # a re-fill that succeeded clears any stale "couldn't fill"
            # badge for those columns, so a retry doesn't leave both a
            # value and a leftover failure tooltip.
            succeeded_cols = (
                {
                    k for k, v in res.values.items()
                    if v is not None and v != ""
                }
                if res.values
                else set()
            )
            needs_persist = bool(res.values) or bool(failed_cols)
            if needs_persist:
                write_db = SessionLocal()
                try:
                    sample = write_db.query(Sample).filter(Sample.id == res.row_id).first()
                    if sample is not None:
                        if res.values:
                            d = dict(sample.row or {})
                            for k, v in res.values.items():
                                d[k] = v
                            sample.row = d
                        # Reconcile fill_status in one pass so partial
                        # successes (some cols filled, some not) leave
                        # exactly the right markers behind.
                        existing_tags = dict(sample.tags or {})
                        existing_status = dict(existing_tags.get("fill_status") or {})
                        status_changed = False
                        for col in succeeded_cols:
                            if col in existing_status:
                                del existing_status[col]
                                status_changed = True
                        if failed_cols:
                            existing_status.update(failed_cols)
                            status_changed = True
                        if status_changed:
                            if existing_status:
                                existing_tags["fill_status"] = existing_status
                            else:
                                existing_tags.pop("fill_status", None)
                            sample.tags = existing_tags
                        # Per-cell sources from set_values(sources=...).
                        # Only write entries for columns the agent
                        # actually filled this turn; merge with existing
                        # so a partial re-fill on the same row keeps
                        # earlier citations intact. Same wire shape as
                        # rows_add's _sources path:
                        #   tags["sources"][col] = [{"type":"url","value":"..."}, ...]
                        if res.sources:
                            existing_sources = dict(existing_tags.get("sources") or {})
                            sources_changed = False
                            for col, srcs in res.sources.items():
                                if col not in succeeded_cols:
                                    continue
                                if srcs:
                                    existing_sources[col] = srcs
                                    sources_changed = True
                            if sources_changed:
                                existing_tags["sources"] = existing_sources
                                sample.tags = existing_tags
                        write_db.commit()
                        write_db.refresh(sample)
                        updated_row = {
                            "_id": str(sample.id),
                            "_seq": sample.seq,
                            "_tags": sample.tags or {},
                            **(sample.row or {}),
                        }
                except Exception:
                    log.exception("per-cell persist failed for row %s", res.row_id)
                    try:
                        write_db.rollback()
                    except Exception:
                        pass
                finally:
                    write_db.close()
            # Billing gate: cells that didn't end "filled" are charged
            # at FAILED_FILL_CHARGE_RATE * cost (default 0.1). Set the
            # rate to 1.0 for full charge, 0.0 for full waive. The
            # actual cost stays on res.cost_usd / the trace for audit;
            # only the user-facing meter reading is scaled.
            billable_cost = (
                _api_settings.FAILED_FILL_CHARGE_RATE * res.cost_usd
                if res.status != "filled"
                else res.cost_usd
            )
            await _emit({
                "type": "cell_done",
                "row_id": str(row_id),
                "index": idx,
                "total": len(work_items),
                "status": res.status,
                "cost": round(billable_cost, 4),
                "filled": [
                    k for k, v in res.values.items()
                    if v is not None and v != ""
                ],
            })
            # Push a row_merged so the frontend updates the visible row
            # in the table immediately, same channel as rows_add uses.
            if updated_row is not None:
                await _emit({"type": "row_merged", "row": updated_row})
            return res

    tasks = [
        _process(rid, rdata, unfilled, i + 1)
        for i, (rid, rdata, unfilled) in enumerate(work_items)
    ]
    completed = await asyncio.gather(*tasks, return_exceptions=True)
    voided_cost = 0.0
    for r in completed:
        if isinstance(r, Exception):
            continue
        results.append(r)
        # Mirror the cell_done billing gate: non-filled cells contribute
        # only `rate * cost` to total_cost (so streaming's residual calc
        # doesn't double-bill the waived portion); the (1 - rate) * cost
        # delta accumulates as voided_cost for the audit summary.
        if r.status != "filled":
            charged = _api_settings.FAILED_FILL_CHARGE_RATE * r.cost_usd
            total_cost += charged
            voided_cost += r.cost_usd - charged
        else:
            total_cost += r.cost_usd

    # Persist forensic traces — one JSONL line per cell, keyed by run_id.
    # Best-effort: any blob error is logged inside write_traces and the
    # fill summary still returns normally. The chat agent inspects via
    # cell_traces_inspect(run_id=..., filter=...).
    traces = [r.trace for r in results if r.trace is not None]
    trace_persist_info: Optional[Dict[str, Any]] = None
    if traces:
        try:
            trace_persist_info = cell_traces.write_traces(
                project.id,
                run_id,
                traces,
                target_columns=list(target_columns),
            )
        except Exception:
            log.exception("cell_traces persist failed (continuing)")

    # Aggregate summary
    by_status: Dict[str, int] = {}
    for r in results:
        by_status[r.status] = by_status.get(r.status, 0) + 1
    filled_total = sum(1 for r in results if any(v is not None and v != "" for v in r.values.values()))

    # Top failure reasons: bag-of-similar reasons across non-filled cells
    # so the chat agent sees clustering ("8 of 12 cells failed: name not
    # findable on X — try checking bio links") at a glance, without
    # paging through cell_traces. Reasons are normalized loosely: lower-
    # cased + first 100 chars. Same reason text from many cells collapses
    # naturally; truly distinct reasons stay distinct.
    failure_buckets: Dict[str, int] = {}
    for r in results:
        if r.status == "filled":
            continue
        key = (r.reason or f"({r.status})").strip().lower()[:100]
        failure_buckets[key] = failure_buckets.get(key, 0) + 1
    top_failures = sorted(
        ({"reason": k, "count": v} for k, v in failure_buckets.items()),
        key=lambda x: x["count"],
        reverse=True,
    )[:5]

    summary: Dict[str, Any] = {
        "matched_rows": len(rows),
        "rows_skipped_already_filled": rows_skipped_already_filled,
        "rows_skipped_prior_fail": rows_skipped_prior_fail,
        "processed": len(results),
        "cells_filled": filled_total,
        "by_status": by_status,
        "avg_cost_per_row": round(total_cost / max(1, len(results)), 4),
        "samples": [
            {"row_id": r.row_id, "values": r.values, "status": r.status, "reason": r.reason, "cost": round(r.cost_usd, 4)}
            for r in results[:5]
        ],
        "run_id": run_id,
        "top_failure_reasons": top_failures,
    }
    if voided_cost > 0:
        # Internal-audit field — surfaces compute we ate (didn't bill
        # the user) via the (1 - FAILED_FILL_CHARGE_RATE) discount on
        # non-filled cells. The chat agent SEES this and can use it to
        # gauge wasted-OpenAI-spend without changing how it talks about
        # cost to the user.
        summary["voided_cost_usd"] = round(voided_cost, 4)
    if trace_persist_info and trace_persist_info.get("persisted"):
        summary["trace_file"] = trace_persist_info.get("file")
    if skills_applied_names:
        summary["skills_applied"] = list(skills_applied_names)
    if rows_skipped_already_filled and len(results) == 0:
        summary["note"] = (
            "All matched rows already have values in the target columns."
        )
    return (summary, total_cost)


# Escalation thresholds for rows_fill_with_escalation. These are
# intentionally hardcoded — the chat agent's only decisional input is
# whether to set the flag at all; tuning the thresholds would push more
# tunables into the prompt without much benefit.
_ESCALATION_MIN_NULL_ROWS = 5
_ESCALATION_YIELD_THRESHOLD = 0.7

# Thresholds for surfacing the "use bulk_first next time" hint. Two
# independent signals — either is enough to fire the hint:
#
#   (A) cost-per-fill: phase 2 produced fills ≥30% cheaper than phase 1.
#       Phase 1 and phase 2 run on different row slices (phase 2 only
#       sees still-null rows) so cost-per-row isn't comparable; cost-
#       per-fill is.
#
#   (B) phase 1 yield was poor AND phase 2 still recovered a meaningful
#       fraction of its slice. Captures the "phase 1 burned web_search
#       on dead ends" pattern even when its cost-per-fill on the rows
#       it DID catch was fine. The point isn't that phase 2 was
#       cheaper, it's that phase 1 spent most of its budget on cells
#       that ended null.
#
# Both signals require phase 2 to have filled enough cells to be a
# meaningful sample (small wins on 1-2 rows are noise).
_BULK_FIRST_HINT_COST_PER_FILL_RATIO = 0.7
_BULK_FIRST_HINT_LOW_P1_YIELD = 0.5
_BULK_FIRST_HINT_MIN_P2_RECOVERY = 0.3
_BULK_FIRST_HINT_MIN_P2_FILLS = 3


def _bulk_first_hint(
    p1: Dict[str, Any],
    p2: Dict[str, Any],
    target_columns: List[str],
) -> Optional[str]:
    """Return a short recommendation string when phase 1 was clearly
    pulling its weight badly enough to skip next time.

    See thresholds above for the two independent signals. Returns None
    when phase 2 didn't run, filled too few cells, or phase 1 was
    already competitive.
    """
    p2_filled = int(p2.get("cells_filled", 0) or 0)
    if p2_filled < _BULK_FIRST_HINT_MIN_P2_FILLS:
        return None
    p1_filled = int(p1.get("cells_filled", 0) or 0)

    p1_avg = float(p1.get("avg_cost_per_row", 0) or 0)
    p2_avg = float(p2.get("avg_cost_per_row", 0) or 0)
    p1_processed = max(1, int(p1.get("processed", 0) or 0))
    p2_processed = max(1, int(p2.get("processed", 0) or 0))
    p1_total = p1_avg * p1_processed
    p2_total = p2_avg * p2_processed

    p1_yield = p1_filled / p1_processed
    p2_recovery = p2_filled / p2_processed

    # Signal A — phase 2 cost-per-fill beat phase 1 by ≥30%.
    cost_signal = False
    if p1_filled > 0:
        p1_cpf = p1_total / p1_filled
        p2_cpf = p2_total / p2_filled
        cost_signal = p2_cpf < p1_cpf * _BULK_FIRST_HINT_COST_PER_FILL_RATIO
    else:
        # Phase 1 filled nothing — fire automatically.
        cost_signal = True

    # Signal B — phase 1 yield was low AND phase 2 recovered enough.
    yield_signal = (
        p1_yield < _BULK_FIRST_HINT_LOW_P1_YIELD
        and p2_recovery >= _BULK_FIRST_HINT_MIN_P2_RECOVERY
    )

    if not (cost_signal or yield_signal):
        return None

    cols = ", ".join(target_columns) if target_columns else "this column"
    if cost_signal and p1_filled > 0:
        lead = (
            f"Bulk browser_use produced fills "
            f"{(p1_total / p1_filled) / (p2_total / p2_filled):.1f}x "
            f"cheaper than per-cell"
        )
        detail = (
            f"phase 1: {p1_filled}/{p1_processed} filled at "
            f"${p1_total / p1_filled:.3f}/fill; "
            f"phase 2: {p2_filled}/{p2_processed} at "
            f"${p2_total / p2_filled:.3f}/fill"
        )
    else:
        # Yield-driven hint (or phase-1 zero-fills) — phase 1 didn't
        # pull its weight even if its cost-per-fill on the rows it
        # caught was OK.
        lead = "Per-cell phase 1 was a poor first pass"
        detail = (
            f"phase 1 yield {p1_yield:.0%} ({p1_filled}/{p1_processed}); "
            f"bulk phase 2 recovered {p2_filled}/{p2_processed} of the "
            f"still-null rows"
        )
    return (
        f"{lead} on '{cols}' in this run ({detail}). For follow-up "
        f"rows_fill calls on this column, prefer `bulk_first=true` to "
        f"skip the per-cell phase."
    )


def _merge_fill_summaries(
    p1: Dict[str, Any], p2: Dict[str, Any]
) -> Dict[str, Any]:
    """Combine phase-1 (per-cell) and phase-2 (bulk_browser) summaries.

    Phase 2 runs over the same where clause; its pre-filter automatically
    skips rows whose target columns are now all filled, so its
    `cells_filled` is additive to phase 1's.

    The merged shape stays familiar to the chat agent — same top-level
    keys as phase 1 — but adds a `phase2` block so the LLM can see the
    breakdown.
    """
    p1_filled = int(p1.get("cells_filled", 0) or 0)
    p2_filled = int(p2.get("cells_filled", 0) or 0)
    total_filled = p1_filled + p2_filled

    p1_status = p1.get("by_status", {}) or {}
    p2_status = p2.get("by_status", {}) or {}

    # Approximate by_status: start from phase 1, decrement
    # null_legitimate / error by phase 2's wins, increment filled. Not
    # row-perfect (we don't track per-row before/after status here),
    # but close enough for the LLM to understand the rollup. The
    # samples + per-phase blocks below are the source of truth for
    # detail.
    by_status: Dict[str, int] = {k: int(v) for k, v in p1_status.items()}
    if p2_filled:
        for src_key in ("null_legitimate", "error", "no_op", "budget_exhausted"):
            if p2_filled <= 0:
                break
            current = by_status.get(src_key, 0)
            if current <= 0:
                continue
            decrement = min(current, p2_filled)
            by_status[src_key] = current - decrement
            by_status["filled"] = by_status.get("filled", 0) + decrement
            p2_filled -= decrement

    merged: Dict[str, Any] = {
        "matched_rows": p1.get("matched_rows", 0),
        "rows_skipped_already_filled": p1.get("rows_skipped_already_filled", 0),
        "processed": p1.get("processed", 0),
        "cells_filled": total_filled,
        "by_status": by_status,
        "escalated": True,
        # Phase 2's residual top reasons are more useful (these are the
        # rows BU also couldn't get). Fall back to phase 1's if phase 2
        # cleared everything.
        "top_failure_reasons": (
            p2.get("top_failure_reasons")
            or p1.get("top_failure_reasons", [])
        ),
        "phase1": {
            "cells_filled": int(p1.get("cells_filled", 0) or 0),
            "by_status": p1_status,
            "avg_cost_per_row": p1.get("avg_cost_per_row", 0),
            "run_id": p1.get("run_id"),
            "trace_file": p1.get("trace_file"),
        },
        "phase2": {
            "processed": p2.get("processed", 0),
            "cells_filled": int(p2.get("cells_filled", 0) or 0),
            "batches_run": p2.get("batches_run", 0),
            "batches_failed": p2.get("batches_failed", 0),
            "avg_cost_per_row": p2.get("avg_cost_per_row", 0),
            "run_id": p2.get("run_id"),
            "trace_file": p2.get("trace_file"),
        },
    }
    p1_skills = p1.get("skills_applied") or []
    p2_skills = p2.get("skills_applied") or []
    if p1_skills or p2_skills:
        merged["skills_applied"] = sorted(set(list(p1_skills) + list(p2_skills)))
    # Sum voided cost across phases (the (1 - FAILED_FILL_CHARGE_RATE)
    # share of cost on non-filled cells, eaten by us not billed).
    voided = float(p1.get("voided_cost_usd", 0) or 0) + float(
        p2.get("voided_cost_usd", 0) or 0
    )
    if voided > 0:
        merged["voided_cost_usd"] = round(voided, 4)
    # Use phase 2's samples (most recent post-escalation state) when
    # available; the chat agent's "show a couple sample results" needs
    # the latest values.
    if p2.get("samples"):
        merged["samples"] = p2["samples"]
    elif p1.get("samples"):
        merged["samples"] = p1["samples"]
    return merged


async def fill_rows_with_escalation(
    *,
    project: Project,
    target_columns: List[str],
    where_sql: str,
    where_params: Dict[str, Any],
    limit: Optional[int],
    max_cost: float = 0.30,
    concurrency: int = _CELL_CONCURRENCY,
    progress_cb: Optional[ProgressCallback] = None,
    escalate_via_browser_use: bool = False,
    bulk_first: bool = False,
    retry_failed: bool = False,
) -> Tuple[Dict[str, Any], float]:
    """rows_fill dispatcher across three strategies:

    - `bulk_first=True`: skip per-cell entirely, run bulk browser_use as
      the only phase. For columns where per-cell web_search is known to
      be ineffective (X handles, niche social profiles). No automatic
      fallback — the chat agent re-calls with `bulk_first=False` on the
      still-null rows if it wants a per-cell second pass.
    - `bulk_first=False, escalate_via_browser_use=True`: per-cell is
      phase 1; bulk is phase 2 if phase 1 yield was poor (>=5 still-null
      OR <70%).
    - `bulk_first=False, escalate_via_browser_use=False`: per-cell only.

    The summary may include a `next_call_hint` when phase 2 outperformed
    phase 1 by a wide margin, suggesting the chat agent pass
    `bulk_first=True` on the next fill of the same column.
    """
    if bulk_first:
        from dsl_worker.chat_api import bulk_browser

        summary, total_cost = await bulk_browser.bulk_fill_rows(
            project=project,
            target_columns=target_columns,
            where_sql=where_sql,
            where_params=where_params,
            limit=limit,
            progress_cb=progress_cb,
            retry_failed=retry_failed,
        )
        if not summary.get("error"):
            summary["strategy"] = "bulk_first"
        return (summary, total_cost)

    # Phase 1 — cheap per-cell pass.
    summary, total_cost = await fill_rows(
        project=project,
        target_columns=target_columns,
        where_sql=where_sql,
        where_params=where_params,
        limit=limit,
        max_cost=max_cost,
        concurrency=concurrency,
        progress_cb=progress_cb,
        retry_failed=retry_failed,
    )

    if not escalate_via_browser_use or summary.get("error"):
        return (summary, total_cost)

    processed = int(summary.get("processed", 0) or 0)
    p1_filled = int(summary.get("cells_filled", 0) or 0)
    still_unfilled = max(0, processed - p1_filled)
    yield_pct = (p1_filled / processed) if processed else 1.0

    # Skip escalation when phase 1 already covered the work. The bar is
    # deliberately lenient (>=5 OR <70%) so the LLM doesn't pay for
    # browser_use on a column where the cheap pass nailed it.
    if still_unfilled < _ESCALATION_MIN_NULL_ROWS and yield_pct >= _ESCALATION_YIELD_THRESHOLD:
        summary["escalated"] = False
        summary["escalation_skipped_reason"] = (
            f"Phase 1 yield {yield_pct:.0%} with {still_unfilled} rows "
            f"still null — under the escalation threshold."
        )
        return (summary, total_cost)

    # Phase 2 — browser_use bulk fallback over the same where clause.
    # Lazy import to avoid a circular import at module load.
    from dsl_worker.chat_api import bulk_browser

    summary2, cost2 = await bulk_browser.bulk_fill_rows(
        project=project,
        target_columns=target_columns,
        where_sql=where_sql,
        where_params=where_params,
        limit=limit,
        progress_cb=progress_cb,
        retry_failed=retry_failed,
    )
    total_cost += cost2

    if summary2.get("error"):
        # Phase 2 blew up before doing anything useful — surface phase 1
        # results plus an error breadcrumb so the chat agent can decide
        # whether to retry or give up.
        summary["escalated"] = True
        summary["escalation_error"] = summary2.get("error")
        return (summary, total_cost)

    merged = _merge_fill_summaries(summary, summary2)
    hint = _bulk_first_hint(summary, summary2, target_columns)
    if hint:
        merged["next_call_hint"] = hint
    return (merged, total_cost)
