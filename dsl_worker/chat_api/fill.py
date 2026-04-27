"""rows_fill — per-cell mini-agent for chat-mode.

When the user (via the chat agent) calls `rows_fill(columns, where, ...)`,
each matching row spawns a small bounded subagent that:
  - sees the row's existing column values
  - sees the goal (one or more column names + their `format` + `description`)
  - has access to all source tools (FE / Apollo / Apify / Google Maps /
    code_exec / web_harvest / browser_use / web_search built-in)
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
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple

from sqlalchemy.orm import Session

from dsl_api.config import settings as _api_settings
from dsl_api.db import SessionLocal
from dsl_api.models import Project
from dsl_api.models.sample import Sample

from dsl_worker.chat_api import sources

ProgressCallback = Callable[[Dict[str, Any]], Awaitable[None]]


log = logging.getLogger(__name__)


# Concurrent cell agents. Same default as V13 row-gens.
_CELL_CONCURRENCY = 5

# Per-cell defaults. The first prod test (a16z Speedrun founders → Twitter
# handles via web research) hit the $0.10 cap on 4/5 cells before finding a
# match — bumping up so the agent has enough room to do a couple of web
# searches per cell.
_DEFAULT_MAX_COST = 0.20
_DEFAULT_MAX_TURNS = 7

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
    "web_harvest",
    "browser_use",
}


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
   one. (FullEnrich for verified contact info, Apollo for fallback,
   Apify for site-specific scraping, Google Maps for local biz, web
   research only as a last resort.)
4. Once you have the value(s), call set_values to commit and stop.

Critical rules:
- Real data only. If you can't find a real answer, call set_values with
  null (or omit the field) and explain in 'reason'. Don't fabricate.
- Match the column's `format` exactly when stated.
- One or two tool calls per turn max. Don't chain a research odyssey.
- You are CAPPED at a tight budget. Don't waste calls.
- ALWAYS finish by calling either `set_values` or `give_up`. Don't trail
  off.
"""


@dataclass
class CellFillResult:
    row_id: str
    values: Dict[str, Any] = field(default_factory=dict)
    status: str = "filled"   # filled | null_legitimate | error | budget_exhausted | no_op
    reason: Optional[str] = None
    cost_usd: float = 0.0
    turns: int = 0


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
            "no-result cases. After calling this, stop — the run is done."
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
    max_turns: int,
) -> CellFillResult:
    """Spawn the bounded subagent for one row and return its CellFillResult."""
    from openai import AsyncOpenAI

    result = CellFillResult(row_id=row_id)
    client = AsyncOpenAI(api_key=_api_settings.OPENAI_API_KEY)

    user_msg = _row_context(row_data, target_columns, target_specs)
    cell_tools = _cell_tools_for_columns(target_columns)
    next_input: Any = [{"role": "user", "content": user_msg}]
    previous_response_id: Optional[str] = None

    for turn_idx in range(max_turns):
        result.turns = turn_idx + 1

        if result.cost_usd >= max_cost:
            result.status = "budget_exhausted"
            result.reason = f"hit ${max_cost:.2f} cap after {turn_idx} turn(s)"
            break

        try:
            kwargs: Dict[str, Any] = {
                "model": _api_settings.OPENAI_MODEL,
                "input": next_input,
                "tools": cell_tools,
                "max_output_tokens": 2000,
            }
            if turn_idx == 0:
                kwargs["instructions"] = _CELL_AGENT_SYSTEM_PROMPT
            else:
                kwargs["previous_response_id"] = previous_response_id
            resp = await client.responses.create(**kwargs)
        except Exception as e:
            result.status = "error"
            result.reason = f"{type(e).__name__}: {e}"
            return result

        result.cost_usd += sources._response_cost(resp)
        previous_response_id = resp.id

        function_calls: List[Any] = []
        for item in resp.output:
            if item.type == "web_search_call":
                result.cost_usd += sources._WEB_SEARCH_USD_PER_CALL
            elif item.type == "function_call":
                function_calls.append(item)

        if not function_calls:
            # Model returned text without committing. Mark as no-op.
            result.status = "no_op"
            result.reason = "agent ended without calling set_values"
            return result

        # Process tool calls
        tool_outputs: List[Dict[str, Any]] = []
        terminated = False
        for fc in function_calls:
            try:
                fc_args = json.loads(fc.arguments) if fc.arguments else {}
            except json.JSONDecodeError:
                tool_outputs.append({"type": "function_call_output", "call_id": fc.call_id, "output": "Error: invalid arguments"})
                continue

            if fc.name == "set_values":
                values = fc_args.get("values") or {}
                if not isinstance(values, dict):
                    tool_outputs.append({"type": "function_call_output", "call_id": fc.call_id, "output": "Error: values must be an object"})
                    continue
                # Filter to target columns only — the agent might emit extras
                clean = {k: v for k, v in values.items() if k in target_columns}
                result.values = clean
                result.reason = fc_args.get("reason") or None
                # Determine status from values
                non_null_count = sum(1 for v in clean.values() if v is not None and v != "")
                result.status = "filled" if non_null_count > 0 else "null_legitimate"
                terminated = True
                tool_outputs.append({"type": "function_call_output", "call_id": fc.call_id, "output": "ok"})

            elif fc.name == "give_up":
                result.status = "error"
                result.reason = fc_args.get("reason") or "agent gave up"
                terminated = True
                tool_outputs.append({"type": "function_call_output", "call_id": fc.call_id, "output": "ok"})

            elif fc.name in sources._HANDLERS:
                # Source tool — execute and feed result back
                if result.cost_usd >= max_cost:
                    tool_outputs.append({"type": "function_call_output", "call_id": fc.call_id, "output": "budget_exhausted; finish via set_values or give_up"})
                    continue
                try:
                    out_text, tool_cost = await sources.execute_source_tool(fc.name, fc_args)
                    result.cost_usd += tool_cost
                    tool_outputs.append({"type": "function_call_output", "call_id": fc.call_id, "output": out_text[:6000]})
                except Exception as e:
                    tool_outputs.append({"type": "function_call_output", "call_id": fc.call_id, "output": f"error: {type(e).__name__}: {e}"})

            else:
                tool_outputs.append({"type": "function_call_output", "call_id": fc.call_id, "output": f"unknown tool: {fc.name}"})

        if terminated:
            return result

        next_input = tool_outputs

    # Out of turns without termination
    if result.status == "filled":
        return result  # was set in the last turn
    result.status = "budget_exhausted"
    result.reason = result.reason or f"reached max_turns={max_turns} without calling set_values"
    return result


async def fill_rows(
    *,
    project: Project,
    target_columns: List[str],
    where_sql: str,
    where_params: Dict[str, Any],
    limit: Optional[int],
    max_cost: float = _DEFAULT_MAX_COST,
    max_turns: int = _DEFAULT_MAX_TURNS,
    concurrency: int = _CELL_CONCURRENCY,
    progress_cb: Optional[ProgressCallback] = None,
) -> Tuple[Dict[str, Any], float]:
    """Run cell agents over all matching rows.

    Returns (summary_dict, total_cost_usd).
    """
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

    # Fetch rows in a short-lived session — we'll then spawn per-cell sessions.
    db = SessionLocal()
    try:
        from sqlalchemy import text
        version_id = project.current_version_id
        if not version_id:
            return ({"error": "project has no version yet"}, 0.0)
        sql = (
            f"SELECT id, row FROM samples WHERE version_id = :vid AND ({where_sql}) "
            f"ORDER BY seq"
        )
        if limit is not None:
            sql += f" LIMIT {int(limit)}"
        rows = db.execute(text(sql), {"vid": version_id, **where_params}).all()
    finally:
        db.close()

    if not rows:
        return ({"matched_rows": 0, "filled": 0, "summary": "no rows match"}, 0.0)

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
            "total": len(rows),
            "columns": list(target_columns),
        })

    async def _process(row_id: Any, row_data: Any, idx: int) -> CellFillResult:
        async with sem:
            await _emit({
                "type": "cell_start",
                "row_id": str(row_id),
                "index": idx,
                "total": len(rows),
            })
            res = await _run_cell_agent(
                row_id=str(row_id),
                row_data=dict(row_data or {}),
                target_columns=target_columns,
                target_specs=target_specs,
                max_cost=max_cost,
                max_turns=max_turns,
            )
            await _emit({
                "type": "cell_done",
                "row_id": str(row_id),
                "index": idx,
                "total": len(rows),
                "status": res.status,
                "cost": round(res.cost_usd, 4),
                "filled": [
                    k for k, v in res.values.items()
                    if v is not None and v != ""
                ],
            })
            return res

    tasks = [
        _process(rid, rdata, i + 1)
        for i, (rid, rdata) in enumerate(rows)
    ]
    completed = await asyncio.gather(*tasks, return_exceptions=True)

    # Persist values to the DB. Each result writes only its own cells.
    write_db = SessionLocal()
    try:
        for r in completed:
            if isinstance(r, Exception):
                continue
            results.append(r)
            total_cost += r.cost_usd
            if not r.values:
                continue
            sample = write_db.query(Sample).filter(Sample.id == r.row_id).first()
            if sample is None:
                continue
            d = dict(sample.row or {})
            for k, v in r.values.items():
                d[k] = v
            sample.row = d
        write_db.commit()
    finally:
        write_db.close()

    # Aggregate summary
    by_status: Dict[str, int] = {}
    for r in results:
        by_status[r.status] = by_status.get(r.status, 0) + 1
    filled_total = sum(1 for r in results if any(v is not None and v != "" for v in r.values.values()))

    return (
        {
            "matched_rows": len(rows),
            "processed": len(results),
            "cells_filled": filled_total,
            "by_status": by_status,
            "avg_cost_per_row": round(total_cost / max(1, len(results)), 4),
            "samples": [
                {"row_id": r.row_id, "values": r.values, "status": r.status, "reason": r.reason, "cost": round(r.cost_usd, 4)}
                for r in results[:5]
            ],
        },
        total_cost,
    )
