"""Cell agent — per-row enrichment runner with a focused toolset.

Spawned per row when an enrichment's action.type == "cell_agent". Each cell
agent gets:
  - The full current row data (all already-filled columns)
  - The action's `prompt` (natural-language goal)
  - `columns_to_fill` — which column names to produce
  - 11 tools (see CELL_TOOL_HANDLERS)
  - A budget cap (per_row_credit_cap)

Loop terminates when:
  - Cell agent emits a `final_result` tool call (or final JSON message)
  - Budget cap is reached
  - Hard iteration limit hit (safety)
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Awaitable, Callable, Dict, List, Tuple

from openai import AsyncOpenAI

from dsl_worker.chat_v2.tools import ToolContext


log = logging.getLogger(__name__)


# Per-row hard iteration cap so a misbehaving cell agent can't loop forever.
MAX_CELL_ITERATIONS = 8


# Coarse credit estimates per tool call. Server's actual cost accounting
# uses balance_ledger; this is just for the cell agent's local budget tracking
# so it stops at per_row_credit_cap.
TOOL_COST_ESTIMATES = {
    "fullenrich_enrich_email": 1.0,    # 1 credit on success
    "fullenrich_enrich_phone": 10.0,   # ~10x email
    "fullenrich_enrich_company": 0.5,
    "apollo_org_enrich": 1.0,
    "google_maps_place_details": 0.3,
    "apify_search_actors": 0.0,
    "apify_actor_details": 0.0,
    "apify_call_actor": 1.0,
    "web_search": 0.0,
    "browser_use": 5.0,                 # session is real money
    "code_exec": 0.0,
}


# ---------------------------------------------------------------------------
# Cell-agent-facing tool handlers — same shape as orchestrator handlers
# ---------------------------------------------------------------------------


async def _fullenrich_enrich_email(args: Dict[str, Any], ctx: ToolContext) -> Tuple[Dict[str, Any], float]:
    """Call FE enrich-people for verified email. Returns {email, verification_status}."""
    import httpx
    api_key = os.getenv("FULLENRICH_API_KEY")
    if not api_key:
        return {"error": "FULLENRICH_API_KEY not configured"}, 0.0
    body = {
        "first_name": args.get("first_name", ""),
        "last_name": args.get("last_name", ""),
        "domain": args.get("domain") or args.get("company_domain", ""),
        "company_name": args.get("company", ""),
        "include_email": True,
        "include_phone": False,
    }
    async with httpx.AsyncClient(timeout=60) as client:
        r = await client.post(
            "https://app.fullenrich.com/api/v1/contact/enrich",
            headers={"Authorization": f"Bearer {api_key}"},
            json=body,
        )
        if r.status_code != 200:
            return {"error": f"FE HTTP {r.status_code}"}, 0.0
        data = r.json() or {}
    contact = (data.get("data") or {})
    return {
        "email": contact.get("email"),
        "verification_status": contact.get("email_status"),
    }, 1.0  # ~1 credit per successful email


async def _fullenrich_enrich_phone(args: Dict[str, Any], ctx: ToolContext) -> Tuple[Dict[str, Any], float]:
    import httpx
    api_key = os.getenv("FULLENRICH_API_KEY")
    if not api_key:
        return {"error": "FULLENRICH_API_KEY not configured"}, 0.0
    body = {
        "first_name": args.get("first_name", ""),
        "last_name": args.get("last_name", ""),
        "domain": args.get("domain") or args.get("company_domain", ""),
        "company_name": args.get("company", ""),
        "include_email": False,
        "include_phone": True,
    }
    async with httpx.AsyncClient(timeout=90) as client:
        r = await client.post(
            "https://app.fullenrich.com/api/v1/contact/enrich",
            headers={"Authorization": f"Bearer {api_key}"},
            json=body,
        )
        if r.status_code != 200:
            return {"error": f"FE HTTP {r.status_code}"}, 0.0
        data = r.json() or {}
    contact = (data.get("data") or {})
    return {"phone": contact.get("phone")}, 10.0


async def _fullenrich_enrich_company(args: Dict[str, Any], ctx: ToolContext) -> Tuple[Dict[str, Any], float]:
    import httpx
    api_key = os.getenv("FULLENRICH_API_KEY")
    if not api_key:
        return {"error": "FULLENRICH_API_KEY not configured"}, 0.0
    domain = args.get("domain")
    if not domain:
        return {"error": "domain is required"}, 0.0
    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.post(
            "https://app.fullenrich.com/api/v1/company/enrich",
            headers={"Authorization": f"Bearer {api_key}"},
            json={"domain": domain},
        )
        if r.status_code != 200:
            return {"error": f"FE HTTP {r.status_code}"}, 0.0
        data = (r.json() or {}).get("data") or {}
    return data, 0.5


async def _apollo_org_enrich(args: Dict[str, Any], ctx: ToolContext) -> Tuple[Dict[str, Any], float]:
    import httpx
    api_key = os.getenv("APOLLO_API_KEY")
    if not api_key:
        return {"error": "APOLLO_API_KEY not configured"}, 0.0
    domain = args.get("domain")
    if not domain:
        return {"error": "domain is required"}, 0.0
    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.get(
            "https://api.apollo.io/api/v1/organizations/enrich",
            params={"domain": domain},
            headers={"X-Api-Key": api_key},
        )
        if r.status_code != 200:
            return {"error": f"apollo HTTP {r.status_code}"}, 0.0
        org = (r.json() or {}).get("organization") or {}
    # Return a curated subset to avoid token-blowing the cell agent
    return {
        "name": org.get("name"),
        "estimated_num_employees": org.get("estimated_num_employees"),
        "industry": org.get("industry"),
        "annual_revenue_printed": org.get("annual_revenue_printed"),
        "total_funding_printed": org.get("total_funding_printed"),
        "latest_funding_stage": org.get("latest_funding_stage"),
        "latest_funding_round_date": org.get("latest_funding_round_date"),
        "founded_year": org.get("founded_year"),
        "linkedin_url": org.get("linkedin_url"),
        "short_description": (org.get("short_description") or "")[:500],
        "current_technologies": [t.get("name") for t in (org.get("current_technologies") or [])][:30],
        "departmental_head_count": org.get("departmental_head_count"),
    }, 1.0


async def _google_maps_place_details(args: Dict[str, Any], ctx: ToolContext) -> Tuple[Dict[str, Any], float]:
    import httpx
    api_key = os.getenv("GOOGLE_API_KEY")
    place_id = args.get("place_id")
    if not (api_key and place_id):
        return {"error": "GOOGLE_API_KEY + place_id required"}, 0.0
    async with httpx.AsyncClient(timeout=15) as client:
        r = await client.get(
            "https://maps.googleapis.com/maps/api/place/details/json",
            params={"key": api_key, "place_id": place_id},
        )
        if r.status_code != 200:
            return {"error": f"gmaps HTTP {r.status_code}"}, 0.0
        result = (r.json() or {}).get("result") or {}
    return result, 0.3


async def _apify_call_actor(args: Dict[str, Any], ctx: ToolContext) -> Tuple[Dict[str, Any], float]:
    """Per-row Apify scrape with tight scope (maxItems=1-5)."""
    import httpx
    api_key = os.getenv("APIFY_API_KEY")
    actor_id = args.get("actor_id")
    actor_input = args.get("input") or {}
    if not (api_key and actor_id):
        return {"error": "APIFY_API_KEY + actor_id required"}, 0.0
    # Cap maxItems for cell-level calls
    actor_input.setdefault("maxItems", 3)
    if actor_input.get("maxItems", 0) > 5:
        actor_input["maxItems"] = 5
    aid = actor_id.replace("/", "~")
    async with httpx.AsyncClient(timeout=180) as client:
        r = await client.post(
            f"https://api.apify.com/v2/acts/{aid}/run-sync-get-dataset-items",
            params={"token": api_key, "format": "json"},
            json=actor_input,
        )
        if r.status_code != 200:
            return {"error": f"apify HTTP {r.status_code}"}, 0.0
        items = r.json() or []
    return {"items": items[:5]}, 1.0


# Reuse light wrappers from orchestrator
from dsl_worker.chat_v2.light_tools import (
    apify_search_actors as _apify_search_actors,
    apify_actor_details as _apify_actor_details,
    web_search as _web_search,
    code_exec as _code_exec,
)


async def _browser_use(args: Dict[str, Any], ctx: ToolContext) -> Tuple[Dict[str, Any], float]:
    """Single BU session, bounded by task scope. Wraps existing infra."""
    try:
        from dsl_worker.infra.bu_client import bu_extract_rows
    except ImportError:
        return {"error": "bu_client not available"}, 0.0
    url = args.get("url")
    task = args.get("task")
    if not (url and task):
        return {"error": "url + task required"}, 0.0
    rows, cost = await bu_extract_rows(url=url, task=task, candidate_description=args.get("candidate_description", ""))
    return {"rows": rows[:10]}, cost


CELL_TOOL_HANDLERS: Dict[str, Callable[[Dict[str, Any], ToolContext], Awaitable[Tuple[Dict[str, Any], float]]]] = {
    "fullenrich_enrich_email": _fullenrich_enrich_email,
    "fullenrich_enrich_phone": _fullenrich_enrich_phone,
    "fullenrich_enrich_company": _fullenrich_enrich_company,
    "apollo_org_enrich": _apollo_org_enrich,
    "google_maps_place_details": _google_maps_place_details,
    "apify_search_actors": _apify_search_actors,
    "apify_actor_details": _apify_actor_details,
    "apify_call_actor": _apify_call_actor,
    "web_search": _web_search,
    "browser_use": _browser_use,
    "code_exec": _code_exec,
}


# ---------------------------------------------------------------------------
# Cell agent loop
# ---------------------------------------------------------------------------


CELL_SYSTEM_PROMPT = """You are a cell agent: your job is to fill in specific columns for ONE row of a table.

You will be given:
  - The current row's data (already-filled fields)
  - The columns you must fill (column names + types)
  - The prompt describing what to find
  - A toolset to use
  - A credit budget per row — when you're near zero, stop and return what you have

Rules:
  - When you have a final answer, call `final_result` with JSON of the columns to fill.
    e.g., final_result({values: {twitter_url: "https://twitter.com/foo", confidence: "high"}})
  - If you genuinely can't find a value for a column, set it to null. That's fine.
  - "Not finding" is not "not working" — some rows legitimately don't have a value.
  - Don't make up information. If web_search has nothing, return null.
  - Be efficient. Don't burn budget on dead ends; pick the most direct tool for the data.
"""


def _final_result_tool_def() -> Dict[str, Any]:
    """OpenAI tool definition for the final_result terminator."""
    return {
        "type": "function",
        "function": {
            "name": "final_result",
            "description": "Emit the final filled column values. Call this exactly once when done.",
            "parameters": {
                "type": "object",
                "properties": {
                    "values": {
                        "type": "object",
                        "description": "Map of column_name → value to fill on this row.",
                    }
                },
                "required": ["values"],
            },
        },
    }


def _tool_defs_for_cell() -> List[Dict[str, Any]]:
    """OpenAI function definitions for cell-agent tools. Minimal — adapter
    callers learn specifics from the prompt's per-source filter cards (which
    aren't in v1 cell-agent context). For v1, we provide loose typing and let
    the agent pass through whatever args make sense."""
    defs = [_final_result_tool_def()]
    generic = {
        "type": "object",
        "properties": {
            "first_name": {"type": "string"},
            "last_name": {"type": "string"},
            "company": {"type": "string"},
            "domain": {"type": "string"},
            "query": {"type": "string"},
            "place_id": {"type": "string"},
            "url": {"type": "string"},
            "task": {"type": "string"},
            "actor_id": {"type": "string"},
            "input": {"type": "object"},
            "code": {"type": "string"},
            "files": {"type": "array", "items": {"type": "string"}},
        },
        "additionalProperties": True,
    }
    for name in CELL_TOOL_HANDLERS:
        defs.append({
            "type": "function",
            "function": {
                "name": name,
                "description": f"Cell-level wrapper for {name}",
                "parameters": generic,
            },
        })
    return defs


async def run_cell_agent(
    action: Dict[str, Any],
    row_data: Dict[str, Any],
    per_row_cap: float,
    columns: List[Dict[str, str]],
    ctx: ToolContext,
) -> Tuple[Dict[str, Any], float]:
    """Mini-LLM per-row loop. Returns (new_fields_dict, total_cost_credits)."""
    prompt = action.get("prompt", "")
    columns_to_fill = action.get("columns_to_fill") or [c["name"] for c in columns]

    client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    user_payload = {
        "current_row": row_data,
        "columns_to_fill": columns_to_fill,
        "instruction": prompt,
        "budget_credits_remaining": per_row_cap,
    }

    messages = [
        {"role": "system", "content": CELL_SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps(user_payload, default=str)},
    ]
    tool_defs = _tool_defs_for_cell()
    total_cost = 0.0
    final_values: Dict[str, Any] = {}

    model = os.getenv("OPENAI_MODEL_MINI", "gpt-5.4-mini")

    for iteration in range(MAX_CELL_ITERATIONS):
        try:
            resp = await client.chat.completions.create(
                model=model,
                messages=messages,
                tools=tool_defs,
                tool_choice="auto",
                temperature=0.0,
            )
        except Exception as e:
            log.warning("cell agent LLM call failed: %s", e)
            break

        msg = resp.choices[0].message
        if msg.tool_calls:
            messages.append({"role": "assistant", "tool_calls": [
                {"id": tc.id, "type": "function", "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                for tc in msg.tool_calls
            ]})

            for tc in msg.tool_calls:
                name = tc.function.name
                try:
                    args = json.loads(tc.function.arguments or "{}")
                except json.JSONDecodeError:
                    args = {}

                if name == "final_result":
                    final_values = args.get("values") or {}
                    return final_values, total_cost

                handler = CELL_TOOL_HANDLERS.get(name)
                if not handler:
                    tool_result = {"error": f"unknown tool {name}"}
                    tool_cost = 0.0
                else:
                    try:
                        tool_result, tool_cost = await handler(args, ctx)
                    except Exception as e:
                        log.exception("cell tool %s raised: %s", name, e)
                        tool_result = {"error": str(e)}
                        tool_cost = 0.0

                total_cost += tool_cost
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": json.dumps(tool_result, default=str)[:8000],
                })

                # Budget check — if remaining < min_step_cost, stop
                if total_cost >= per_row_cap:
                    return final_values, total_cost
        else:
            # No tool calls — try to parse JSON from the assistant message as
            # the final result.
            content = (msg.content or "").strip()
            if content:
                try:
                    data = json.loads(content)
                    if isinstance(data, dict) and "values" in data:
                        return data["values"], total_cost
                    if isinstance(data, dict):
                        return data, total_cost
                except json.JSONDecodeError:
                    pass
            # Otherwise stop — no more tool calls and no parseable result.
            break

    return final_values, total_cost
