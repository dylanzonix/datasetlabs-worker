"""Cell agent — per-row enrichment runner with research-level routing.

Spawned per row for every enrichment. Each cell agent gets:
  - The full current row data (already-filled columns + hidden source fields)
  - The action's `prompt` (natural-language goal)
  - `columns_to_fill` — which column names to produce
  - A toolset (depends on research level)
  - A credit budget per row (enforced programmatically; NOT shown to the LLM)

Research levels — four flat values, picked per enrichment:

  - "classify" gpt-5.4-nano  | no tools — pure classify-from-text
  - "light"    gpt-5.4-mini  | all tools, cheap — agent picks if it
                              | needs to look anything up
  - "standard" gpt-5.5       | all tools, normal depth
  - "deep"     gpt-5.5 high  | all tools, multi-step / chained

Only "classify" restricts tool access. The other three can all call
web_search / FE / Apollo / browser_use; tier picks the model + effort +
default budget rather than what's available.

Legacy aliases:
  fast/classify → "classify"   (nano name churn from earlier rename pass)
  smart         → "light"      (mini, was no-tools; now has tools)
  lookup        → "standard"
  research      → "deep"
  expert        → "standard"

Loop terminates when:
  - Cell agent emits `final_result` (or a parseable JSON message)
  - per_row_credit_cap is reached — server kills the loop without notice
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
import uuid
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple

from openai import AsyncOpenAI
from sqlalchemy import text as sa_text

from dsl_worker.billing.tracked_client import TrackedOpenAIClient
from dsl_worker.chat_v2.tools import ToolContext


log = logging.getLogger(__name__)


# Coarse credit estimates per tool call. Server's actual cost accounting
# uses balance_ledger; this is just for the cell agent's local budget tracking
# so it stops at per_row_credit_cap.
TOOL_COST_ESTIMATES = {
    "fullenrich_enrich_email": 1.0,
    "fullenrich_enrich_phone": 10.0,
    "fullenrich_enrich_company": 0.5,
    "apollo_org_enrich": 1.0,
    "google_maps_place_details": 0.3,
    "apify_search_actors": 0.0,
    "apify_actor_details": 0.0,
    "apify_call_actor": 1.0,
    "web_search": 0.0,
    "browser_use": 5.0,
    "code_exec": 0.0,
}


# ---------------------------------------------------------------------------
# Tier configuration
# ---------------------------------------------------------------------------


RESEARCH_CONFIG = {
    "classify": {"model": "gpt-5.4-nano", "effort": "medium", "default_cap": 0.3, "tools": []},
    "light":    {"model": "gpt-5.4-mini", "effort": "medium", "default_cap": 1.0, "tools": "all"},
    "standard": {"model": "gpt-5.5",      "effort": "medium", "default_cap": 2.0, "tools": "all"},
    "deep":     {"model": "gpt-5.5",      "effort": "high",   "default_cap": 8.0, "tools": "all"},
}

# Old → new. Lets pre-rename enrichments keep running.
LEGACY_ALIASES = {
    # v0 → v1: original three-tier ladder
    "lookup":   "standard",
    "research": "deep",
    # v1 → v2: renamed nano tier
    "fast":     "classify",
    # v1 → v2: mini was no-tools "smart", v2 promotes it to "light" (with tools)
    "smart":    "light",
    # v1 → v2: 5.5-no-tools "expert" tier dropped; fold into "standard"
    "expert":   "standard",
}


def _resolve_research(action: Dict[str, Any], per_row_cap: Optional[float]) -> Dict[str, Any]:
    """Return resolved config: {model, effort, cap, tools, name}.

    Reads `research` (or legacy `tier`) from the action. Cap defaults from
    the research level when caller passes None — fixes the prior bug where
    enrichment.py always passed 5 and trampled per-tier defaults.
    """
    requested = (action.get("research") or action.get("tier") or "standard").lower()
    requested = LEGACY_ALIASES.get(requested, requested)
    if requested not in RESEARCH_CONFIG:
        log.warning("cell_agent: unknown research %r, defaulting to standard", requested)
        requested = "standard"
    cfg = RESEARCH_CONFIG[requested].copy()
    cap = float(per_row_cap) if per_row_cap and per_row_cap > 0 else cfg["default_cap"]
    cfg["cap"] = cap
    cfg["name"] = requested
    return cfg


# Back-compat alias for any external callers still importing the old name.
TIER_CONFIG = RESEARCH_CONFIG
_resolve_tier = _resolve_research


# ---------------------------------------------------------------------------
# Cell-agent-facing tool handlers — same shape as orchestrator handlers
# ---------------------------------------------------------------------------


async def _fullenrich_enrich_email(args: Dict[str, Any], ctx: ToolContext) -> Tuple[Dict[str, Any], float]:
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
    }, 1.0


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
    import httpx
    api_key = os.getenv("APIFY_API_KEY")
    actor_id = args.get("actor_id")
    actor_input = args.get("input") or {}
    if not (api_key and actor_id):
        return {"error": "APIFY_API_KEY + actor_id required"}, 0.0
    actor_input.setdefault("maxItems", 3)
    if actor_input.get("maxItems", 0) > 5:
        actor_input["maxItems"] = 5
    aid = actor_id.replace("/", "~")
    import asyncio as _asyncio
    async with httpx.AsyncClient(timeout=180) as client:
        # Async pattern: POST /runs → poll until terminal → fetch items +
        # cost. The /run-sync-get-dataset-items endpoint doesn't expose
        # run ID; this is the only path that gives us real billing.
        start = await client.post(
            f"https://api.apify.com/v2/acts/{aid}/runs",
            params={"token": api_key},
            json=actor_input,
        )
        if start.status_code >= 400:
            return {"error": f"apify start HTTP {start.status_code}"}, 0.0
        run_data = (start.json() or {}).get("data") or {}
        run_id = run_data.get("id")
        dataset_id = run_data.get("defaultDatasetId")
        if not (run_id and dataset_id):
            return {"error": "apify: no run id"}, 0.0
        cost_usd = 0.0
        terminal = {"SUCCEEDED", "FAILED", "ABORTED", "TIMED-OUT"}
        t0 = _asyncio.get_event_loop().time()
        while True:
            if _asyncio.get_event_loop().time() - t0 > 150:
                break
            rr = await client.get(
                f"https://api.apify.com/v2/actor-runs/{run_id}",
                params={"token": api_key},
            )
            if rr.status_code == 200:
                rd = (rr.json() or {}).get("data") or {}
                if rd.get("status") in terminal:
                    from dsl_worker.sources_v2.apify_actor import _apify_run_cost_usd_from_data
                    cost_usd = _apify_run_cost_usd_from_data(rd)
                    break
            await _asyncio.sleep(2.0)
        items = []
        items_resp = await client.get(
            f"https://api.apify.com/v2/datasets/{dataset_id}/items",
            params={"token": api_key, "format": "json", "limit": 5},
        )
        if items_resp.status_code == 200:
            items = items_resp.json() or []
    return {"items": items[:5]}, cost_usd * 10.0


from dsl_worker.chat_v2.light_tools import (
    apify_search_actors as _apify_search_actors,
    apify_actor_details as _apify_actor_details,
    web_search as _web_search,
    code_exec as _code_exec,
)


async def _browser_use(args: Dict[str, Any], ctx: ToolContext) -> Tuple[Dict[str, Any], float]:
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


CELL_SYSTEM_PROMPT = """You are a cell agent: fill specific columns for ONE row of a table.

Inputs you'll receive (JSON):
  - row_visible_to_user: the row's already-filled fields as shown in the user's table
  - row_hidden_source_fields (optional): fields the source returned but the
    orchestrator didn't surface as columns. Use these as additional context
    when reasoning. If a column you need is hiding here, return its value
    via final_result — you cannot create new columns, only fill the ones in
    columns_to_fill.
  - columns_to_fill: column names you must produce values for
  - instruction: what to find or compute

Rules:
  - Always finish with `final_result({values: {col_name: value, ...}})`
    where col_name is the EXACT column name from columns_to_fill — do NOT
    invent your own keys like "label", "value", "answer", "result".
  - Set a column to null when the value genuinely doesn't exist. Null is fine.
  - Don't fabricate. If nothing was found, return null.
  - Output format obeys the instruction exactly:
    * Yes/No → enum-style "Yes" / "No" (Title Case, never booleans)
    * Numbers → plain numeric (5000000, never "$5M")
    * Dates → ISO 8601 ("2026-05-15" or "2026-05-15T10:30:00Z")
    * URLs → only commit URLs you actually visited and verified; don't
      construct URLs from name slugs or guess identifiers
  - If you have tools, use them when the answer needs lookup or search.
    If you have no tools, the answer must come from the row context alone —
    reason carefully and emit final_result directly.
"""


def _final_result_tool_def() -> Dict[str, Any]:
    return {
        "type": "function",
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
    }


def _tool_defs_for_tier(tier_cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Responses-API tool definitions, scoped to the research level."""
    defs: List[Dict[str, Any]] = [_final_result_tool_def()]
    if tier_cfg["tools"] == "all":
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
                "name": name,
                "description": f"Cell-level wrapper for {name}",
                "parameters": generic,
            })
    return defs


_cell_client: Optional[TrackedOpenAIClient] = None


def _get_client() -> TrackedOpenAIClient:
    global _cell_client
    if _cell_client is None:
        raw = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        _cell_client = TrackedOpenAIClient(raw)
    return _cell_client


_GENERIC_VALUE_KEYS = {"value", "label", "answer", "result", "output", "v"}


def _coerce_value_keys(
    raw: Dict[str, Any],
    columns_to_fill: List[str],
) -> Dict[str, Any]:
    """Map raw final_result keys onto the columns_to_fill names.

    Small models (especially nano) often emit {label: X} or {value: X} or
    {answer: X} instead of using the actual column name. Without this map,
    enrichment.py merges those bogus keys into samples.row and the user's
    actual column stays empty — looks like the cell agent "ran and
    returned nothing." Returns the cleaned dict.
    """
    if not isinstance(raw, dict) or not raw:
        return {}
    if not columns_to_fill:
        return raw
    # Exact-match keys → keep as-is. Anything else is a candidate for remap.
    exact = {k: v for k, v in raw.items() if k in columns_to_fill}
    leftovers = {k: v for k, v in raw.items() if k not in columns_to_fill}

    # Case: model returned generic key(s) and there's exactly one column to
    # fill (the most common nano failure mode). Use the leftover value.
    if not exact and leftovers and len(columns_to_fill) == 1:
        target = columns_to_fill[0]
        # Prefer a generic-named key if present; otherwise take the first.
        for k in _GENERIC_VALUE_KEYS:
            if k in leftovers:
                return {target: leftovers[k]}
        # Fallback: first value
        first_val = next(iter(leftovers.values()))
        return {target: first_val}

    # Case: model returned positional keys matching a sensible order
    # (label/value, etc.) for multi-column. Best-effort: if leftover count
    # equals missing-column count and leftover keys are all generic, fill
    # in column order.
    missing = [c for c in columns_to_fill if c not in exact]
    if leftovers and len(leftovers) == len(missing) and all(
        k in _GENERIC_VALUE_KEYS or k.lower() in _GENERIC_VALUE_KEYS
        for k in leftovers.keys()
    ):
        for col, val in zip(missing, leftovers.values()):
            exact[col] = val
        return exact

    # Case-insensitive / underscore-collapsed match: e.g. column "Founder Email"
    # and model returned "founder_email" or "founderEmail".
    def _normalize(s: str) -> str:
        return "".join(ch for ch in s.lower() if ch.isalnum())
    cols_norm = {_normalize(c): c for c in columns_to_fill if c not in exact}
    for k, v in leftovers.items():
        nk = _normalize(k)
        if nk in cols_norm:
            exact[cols_norm[nk]] = v

    # Anything that still didn't map gets dropped (logged).
    unmapped = [k for k in leftovers.keys() if k not in exact and _normalize(k) not in cols_norm]
    if unmapped:
        log.warning(
            "cell_agent: dropped unmapped keys %s; columns_to_fill=%s",
            unmapped, columns_to_fill,
        )
    return exact


def _persist_cell_trace(
    ctx: ToolContext,
    enrichment_id: Optional[str],
    sample_id: Optional[str],
    tier: str,
    model: str,
    tool_calls: List[Dict[str, Any]],
    final_values: Optional[Dict[str, Any]],
    error: Optional[str],
    cost_credits: float,
    duration_ms: int,
) -> None:
    """Write a cell_traces row. Best-effort — never raises into the caller."""
    if not (enrichment_id and sample_id and getattr(ctx, "db", None)):
        return
    try:
        ctx.db.execute(
            sa_text(
                """
                INSERT INTO cell_traces
                  (id, enrichment_id, sample_id, tier, model, tool_calls,
                   final_values, error, cost_credits, duration_ms, created_at)
                VALUES
                  (:id, :eid, :sid, :tier, :model, CAST(:tc AS jsonb),
                   CAST(:fv AS jsonb), :err, :cost, :dur, now())
                """
            ),
            {
                "id": str(uuid.uuid4()),
                "eid": enrichment_id,
                "sid": sample_id,
                "tier": tier,
                "model": model,
                "tc": json.dumps(tool_calls, default=str),
                "fv": json.dumps(final_values, default=str) if final_values is not None else None,
                "err": error,
                "cost": cost_credits,
                "dur": duration_ms,
            },
        )
        ctx.db.commit()
    except Exception as e:
        log.warning("cell trace persist failed: %s", e)


async def run_cell_agent(
    action: Dict[str, Any],
    row_data: Dict[str, Any],
    per_row_cap: Optional[float],
    columns: List[Dict[str, str]],
    ctx: ToolContext,
    *,
    enrichment_id: Optional[str] = None,
    sample_id: Optional[str] = None,
    raw_row: Optional[Dict[str, Any]] = None,
) -> Tuple[Dict[str, Any], float, str]:
    """Per-row Responses-API loop with research-level routing.

    Returns (new_fields_dict, total_cost_credits, status).
      status ∈ {"filled", "hit_budget", "error"}
        - filled: cell agent emitted final_result (values may still be null
          if the answer genuinely didn't exist)
        - hit_budget: programmatic kill — per_row_credit_cap reached before
          final_result. FE renders a "hit budget" badge.
        - error: LLM call failed or no parseable result. Cell left untouched.

    Budget is NEVER surfaced to the LLM (no budget_credits_remaining in
    the user payload). The server kills the loop silently when cap is hit.
    If enrichment_id + sample_id are supplied, writes a cell_traces row.
    """
    prompt = action.get("prompt", "")
    columns_to_fill = action.get("columns_to_fill") or [c["name"] for c in columns]
    tier_cfg = _resolve_research(action, per_row_cap)

    system_prompt = CELL_SYSTEM_PROMPT

    # Build a hidden-fields view: source data that isn't currently shown
    # as a visible column. The cell agent gets to see everything the
    # source returned, with a clear marker of what's visible-to-user vs
    # hidden-but-available.
    hidden_fields: Dict[str, Any] = {}
    if isinstance(raw_row, dict):
        visible_keys = set(row_data.keys()) if isinstance(row_data, dict) else set()
        for k, v in raw_row.items():
            if k not in visible_keys:
                hidden_fields[k] = v

    user_payload: Dict[str, Any] = {
        "row_visible_to_user": row_data,
        "columns_to_fill": columns_to_fill,
        "instruction": prompt,
    }
    if hidden_fields:
        user_payload["row_hidden_source_fields"] = hidden_fields
        user_payload["note"] = (
            "row_visible_to_user is what's shown in the user's table. "
            "row_hidden_source_fields are extra fields the source returned "
            "that aren't currently mapped to a column — you can read these "
            "as additional context for reasoning, but you can't return them "
            "as values without the orchestrator adding columns."
        )

    input_items: List[Dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": json.dumps(user_payload, default=str)},
    ]
    tool_defs = _tool_defs_for_tier(tier_cfg)
    total_cost = 0.0
    final_values: Dict[str, Any] = {}
    tool_calls_log: List[Dict[str, Any]] = []
    error_str: Optional[str] = None
    t0 = time.monotonic()

    client = _get_client()
    cache_key = hashlib.sha256(
        f"{tier_cfg['name']}::{system_prompt}".encode()
    ).hexdigest()[:32]

    HARD_TURN_LIMIT = 40
    iteration = 0
    status = "error"  # default; flipped to "filled" or "hit_budget" on exit
    try:
        while iteration < HARD_TURN_LIMIT:
            iteration += 1

            # Cap check before next LLM call — reasoning-only loops would
            # otherwise never trip the post-tool check below.
            if total_cost >= tier_cfg["cap"]:
                error_str = "budget cap reached"
                status = "hit_budget"
                log.info(
                    "cell agent budget hit (research=%s cost=%.2f cap=%.2f) — stopping",
                    tier_cfg["name"], total_cost, tier_cfg["cap"],
                )
                return final_values, total_cost, status

            try:
                response, cost = await client.responses_create(
                    model=tier_cfg["model"],
                    input=input_items,
                    tools=tool_defs,
                    reasoning={"effort": tier_cfg["effort"]},
                    prompt_cache_key=cache_key,
                )
                total_cost += cost.total_cost_usd
            except Exception as e:
                error_str = f"LLM call failed: {e}"[:500]
                log.warning("cell agent LLM call failed (research=%s): %s", tier_cfg["name"], e)
                return final_values, total_cost, "error"

            function_calls: List[Any] = []
            text_parts: List[str] = []
            for item in response.output:
                itype = getattr(item, "type", None)
                if itype == "function_call":
                    function_calls.append(item)
                    input_items.append(item.model_dump(exclude_none=True))
                elif itype == "reasoning":
                    input_items.append(item.model_dump(exclude_none=True))
                elif itype == "message":
                    for c in item.content:
                        if hasattr(c, "text"):
                            text_parts.append(c.text)
                    input_items.append(item.model_dump(exclude_none=True))
                else:
                    try:
                        input_items.append(item.model_dump(exclude_none=True))
                    except Exception:
                        pass

            if not function_calls:
                content = "".join(text_parts).strip()
                if content:
                    try:
                        data = json.loads(content)
                        if isinstance(data, dict) and "values" in data:
                            final_values = data["values"]
                            return final_values, total_cost, "filled"
                        if isinstance(data, dict):
                            final_values = data
                            return final_values, total_cost, "filled"
                    except json.JSONDecodeError:
                        pass
                error_str = error_str or "no function call and no parseable message"
                return final_values, total_cost, "error"

            for fc in function_calls:
                name = fc.name
                try:
                    args = json.loads(fc.arguments or "{}")
                except json.JSONDecodeError:
                    args = {}

                if name == "final_result":
                    if isinstance(args.get("values"), dict):
                        raw_values = args["values"]
                    elif isinstance(args, dict) and args:
                        raw_values = args
                    else:
                        raw_values = {}
                    # Small models often invent sensible labels like
                    # {label, value, answer, result} instead of using the
                    # actual column name. _coerce_value_keys maps those back.
                    final_values = _coerce_value_keys(raw_values, columns_to_fill)
                    tool_calls_log.append({
                        "name": "final_result",
                        "args": args,
                        "coerced_values": final_values,
                        "cost": 0.0,
                    })
                    return final_values, total_cost, "filled"

                handler = CELL_TOOL_HANDLERS.get(name)
                if not handler:
                    tool_result: Dict[str, Any] = {"error": f"unknown tool {name}"}
                    tool_cost = 0.0
                else:
                    try:
                        tool_result, tool_cost = await handler(args, ctx)
                    except Exception as e:
                        log.exception("cell tool %s raised: %s", name, e)
                        tool_result = {"error": str(e)[:300]}
                        tool_cost = 0.0

                total_cost += tool_cost
                tool_calls_log.append({
                    "name": name,
                    "args": args,
                    "result_preview": json.dumps(tool_result, default=str)[:400],
                    "cost": tool_cost,
                })
                input_items.append({
                    "type": "function_call_output",
                    "call_id": fc.call_id,
                    "output": json.dumps(tool_result, default=str)[:8000],
                })

                if total_cost >= tier_cfg["cap"]:
                    error_str = "budget cap reached before final_result"
                    log.info(
                        "cell agent budget hit (research=%s cost=%.2f cap=%.2f) — stopping",
                        tier_cfg["name"], total_cost, tier_cfg["cap"],
                    )
                    return final_values, total_cost, "hit_budget"

        error_str = f"hit HARD_TURN_LIMIT={HARD_TURN_LIMIT}"
        log.warning(
            "cell agent hit HARD_TURN_LIMIT=%d (research=%s) — emergency stop",
            HARD_TURN_LIMIT, tier_cfg["name"],
        )
        return final_values, total_cost, "error"
    finally:
        duration_ms = int((time.monotonic() - t0) * 1000)
        _persist_cell_trace(
            ctx,
            enrichment_id,
            sample_id,
            tier_cfg["name"],
            tier_cfg["model"],
            tool_calls_log,
            final_values if final_values else None,
            error_str,
            total_cost,
            duration_ms,
        )
