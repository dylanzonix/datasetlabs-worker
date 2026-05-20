"""Cell agent — per-row enrichment runner with research-level routing.

Spawned per row for every enrichment. Each cell agent gets:
  - The full current row data (already-filled columns + hidden source fields)
  - The action's `prompt` (natural-language goal)
  - `columns_to_fill` — which column names to produce
  - A toolset (depends on research level)
  - A credit budget per row (enforced programmatically; NOT shown to the LLM)

Research levels — two values, picked per enrichment:

  - "classify"  gpt-5.4-nano  | no tools — decide a label from row text
  - "research"  gpt-5.5       | all tools — go find more data (web /
                                FE / Apollo / browser_use / etc.)

The split is binary: does the cell agent need to look OUTSIDE this row's
data? If yes → research. If no → classify. The middle tiers (mini, etc.)
proved awkward in practice — for paid integrations the LLM cost is a
rounding error vs the tool fee, so picking 5.5 everywhere we research
buys reliability at imperceptible cost.

Legacy aliases cover every prior rename pass (none/low/medium/high,
classify/lookup/search/investigate, fast/smart/expert, etc.) — all
collapse into one of the two canonical tiers above.

Loop terminates when:
  - Cell agent emits `final_result` (or a parseable JSON message)
  - per_row_credit_cap is reached — server kills the loop without notice
"""

from __future__ import annotations

import asyncio
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


# Tools fall into three categories for budget enforcement:
#
#   FIXED_COST_TOOLS  — billing is deterministic per successful call.
#       We pre-check (would total+cost exceed cap?) and refuse pre-call
#       if so, returning a "skipped" tool result instead of burning.
#
#   CAPPED_TOOLS      — tool supports an explicit max-cost / timeout
#       parameter we plumb the remaining budget into. Tool self-limits.
#
#   FREE_TOOLS        — non-billing (final_result, discovery, code_exec,
#       web_search). No pre-check.
FIXED_COST_TOOLS = {
    "fullenrich_enrich_email":  0.5,   # FE charges per successful contact
    "fullenrich_enrich_phone":  5.0,   # FE phone is expensive
    "fullenrich_enrich_company": 0.5,
    # apollo_org_enrich removed — organizations/enrich is request-quota
    # limited (Apollo's response headers confirm: x-rate-limit-* not
    # credit-* ). Treat as free; pre-call budget gate doesn't refuse it.
    "google_maps_place_details": 0.3,
}

# Hard floor — refuse to call BU if remaining budget is below this, since
# even a one-step BU session typically costs ~$0.10-0.30 and the overhead
# wouldn't get you anything useful.
BU_MIN_BUDGET = 0.30

# Same idea for apify — actor runs need time + a few CU to be worth it.
APIFY_MIN_BUDGET = 0.50

CAPPED_TOOLS = {"browser_use", "apify_call_actor"}

# Legacy reference — kept so external imports don't break.
TOOL_COST_ESTIMATES = dict(FIXED_COST_TOOLS)
TOOL_COST_ESTIMATES.update({
    "browser_use": 5.0,
    "apify_call_actor": 1.0,
    "apify_search_actors": 0.0,
    "apify_actor_details": 0.0,
    "web_search": 0.0,
    "code_exec": 0.0,
})


# ---------------------------------------------------------------------------
# Tier configuration
# ---------------------------------------------------------------------------


RESEARCH_CONFIG = {
    # Two tiers. The split is binary: "process row data only" vs "go find
    # more data via web/integrations". Mini tier was awkward in practice —
    # for paid integrations the LLM cost is a rounding error vs the tool
    # fee, so we just use the smart model everywhere we do research and
    # buy reliability for $0.005/row.
    "classify": {"model": "gpt-5.4-nano", "effort": "medium", "default_cap": 0.3, "tools": []},
    "research": {"model": "gpt-5.5",      "effort": "medium", "default_cap": 5.0, "tools": "all"},
}

# Every old name (across every prior rename pass + the latest collapse)
# normalizes to one of the two canonical tiers. Old enrichment rows in
# the DB keep working, agent slips are tolerated.
LEGACY_ALIASES = {
    # v4 (none/low/medium/high) — current rename. Anything that needed
    # tools collapses to "research"; tool-less stays "classify".
    "none":   "classify",
    "low":    "research",
    "medium": "research",
    "high":   "research",
    # v3 (classify/lookup/search/investigate)
    "lookup":      "research",
    "search":      "research",
    "investigate": "research",
    # v2 (light)
    "light":       "research",
    # v1 (fast/smart/standard/deep/expert)
    "fast":     "classify",
    "smart":    "research",
    "expert":   "research",
    "standard": "research",
    "deep":     "research",
}


def _resolve_research(action: Dict[str, Any], per_row_cap: Optional[float]) -> Dict[str, Any]:
    """Return resolved config: {model, effort, cap, tools, name}.

    Reads `research` (or legacy `tier`) from the action. Cap defaults from
    the research level when caller passes None — fixes the prior bug where
    enrichment.py always passed 5 and trampled per-tier defaults.
    """
    requested = (action.get("research") or action.get("tier") or "research").lower()
    requested = LEGACY_ALIASES.get(requested, requested)
    if requested not in RESEARCH_CONFIG:
        log.warning("cell_agent: unknown research %r, defaulting to research", requested)
        requested = "research"
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


async def _fullenrich_bulk_enrich(
    api_key: str,
    contact: Dict[str, Any],
    enrich_fields: list[str],
    timeout_s: int = 120,
    poll_interval_s: int = 3,
) -> Tuple[Dict[str, Any], float]:
    """Shared FE waterfall path. Submits a 1-contact bulk job, polls until
    FINISHED (typical: 30-60s), returns the contact's enriched fields +
    the credits FE actually billed. Returns 0.0 credits on miss /
    timeout / error so the user only pays for successful matches.

    FE's single-contact /v1 endpoint we used to hit doesn't exist (404 on
    every call). The bulk endpoint is what's actually live.
    """
    import httpx
    BASE = "https://app.fullenrich.com"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    body = {
        "name": "cell_enrich",
        "data": [{**contact, "enrich_fields": enrich_fields}],
    }
    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.post(f"{BASE}/api/v2/contact/enrich/bulk", headers=headers, json=body)
    if r.status_code != 200:
        return {"error": f"FE submit HTTP {r.status_code}: {r.text[:200]}"}, 0.0
    eid = (r.json() or {}).get("enrichment_id")
    if not eid:
        return {"error": "FE returned no enrichment_id"}, 0.0
    # Poll. Single-contact bulk runs typically finish in 30-60s. Beyond
    # timeout_s, give up and charge nothing (user shouldn't pay for our
    # poll budget running out).
    elapsed = 0
    last_result: Dict[str, Any] = {}
    while elapsed < timeout_s:
        await asyncio.sleep(poll_interval_s)
        elapsed += poll_interval_s
        async with httpx.AsyncClient(timeout=15) as client:
            g = await client.get(f"{BASE}/api/v2/contact/enrich/bulk/{eid}", headers=headers)
        if g.status_code != 200:
            continue
        last_result = g.json() or {}
        status = last_result.get("status", "")
        if status == "FINISHED":
            break
        if status in ("CANCELED", "CREDITS_INSUFFICIENT", "RATE_LIMIT"):
            return {"error": f"FE {status}"}, 0.0
    else:
        return {"error": f"FE timed out after {timeout_s}s", "enrichment_id": eid}, 0.0

    items = last_result.get("data") or []
    if not items:
        return {"contact_info": {}}, 0.0
    item = items[0]
    # FE's response nests results under `contact_info`, not `contact`.
    # Was reading the wrong key; every successful lookup looked empty.
    contact_info = item.get("contact_info") or {}
    # FE returns its cost as `cost.credits` — those are FE's INTERNAL
    # credits (not USD, not our credits). Pro plan: ~$55 per 1000
    # credits → $0.055/credit. Cell agent's total_cost is denominated in
    # USD, so we MUST convert here. Before this conversion existed,
    # every FE email lookup was being billed as $1 USD (raw credit
    # count) — a 20x overcharge confirmed against project c978ed19
    # where 10 enrichments all clustered at $1.25-$1.50 because the
    # $1.00 FE line dominated.
    raw_fe_credits = float(((last_result.get("cost") or {}).get("credits") or 0))
    fe_credit_to_usd = float(os.getenv("FULLENRICH_COST_PER_CREDIT", "0.055"))
    cost_usd = raw_fe_credits * fe_credit_to_usd
    return {"contact_info": contact_info}, cost_usd


async def _fullenrich_enrich_email(args: Dict[str, Any], ctx: ToolContext) -> Tuple[Dict[str, Any], float]:
    api_key = os.getenv("FULLENRICH_API_KEY")
    if not api_key:
        return {"error": "FULLENRICH_API_KEY not configured"}, 0.0
    contact = {
        "first_name": args.get("first_name", ""),
        "last_name": args.get("last_name", ""),
        "domain": args.get("domain") or args.get("company_domain", ""),
        "company_name": args.get("company", ""),
    }
    # LinkedIn URL gates FE's deeper waterfall. Many contacts return
    # empty without it but DELIVERABLE with it (e.g. Malik Shamsuddin
    # @ coldemailhustle.com → second@example.com once linkedin is sent).
    linkedin_url = args.get("linkedin_url") or args.get("professional_network_url")
    if linkedin_url:
        contact["linkedin_url"] = linkedin_url
    result, credits = await _fullenrich_bulk_enrich(
        api_key, contact, ["contact.emails"], timeout_s=120,
    )
    if "error" in result:
        return result, credits
    ci = result.get("contact_info") or {}
    # Prefer the highest-confidence single answer FE picked, fall back
    # to the first entry in work_emails[].
    mp = ci.get("most_probable_work_email") or {}
    email = mp.get("email")
    status = mp.get("status")
    if not email:
        for e in (ci.get("work_emails") or []):
            if isinstance(e, dict) and e.get("email"):
                email = e.get("email")
                status = e.get("status")
                break
    return {
        "email": email,
        "verification_status": status,
    }, credits


async def _fullenrich_enrich_phone(args: Dict[str, Any], ctx: ToolContext) -> Tuple[Dict[str, Any], float]:
    api_key = os.getenv("FULLENRICH_API_KEY")
    if not api_key:
        return {"error": "FULLENRICH_API_KEY not configured"}, 0.0
    contact = {
        "first_name": args.get("first_name", ""),
        "last_name": args.get("last_name", ""),
        "domain": args.get("domain") or args.get("company_domain", ""),
        "company_name": args.get("company", ""),
    }
    # Same lever as email: LinkedIn URL gates FE's deeper waterfall. Many
    # contacts return empty without it but a real phone with it.
    linkedin_url = args.get("linkedin_url") or args.get("professional_network_url")
    if linkedin_url:
        contact["linkedin_url"] = linkedin_url
    result, credits = await _fullenrich_bulk_enrich(
        api_key, contact, ["contact.phones"], timeout_s=180,
    )
    if "error" in result:
        return result, credits
    ci = result.get("contact_info") or {}
    mp = ci.get("most_probable_work_phone") or {}
    phone = mp.get("phone")
    if not phone:
        for p in (ci.get("work_phones") or []) + (ci.get("personal_phones") or []):
            if isinstance(p, dict) and p.get("phone"):
                phone = p.get("phone")
                break
    return {"phone": phone}, credits


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
    }, 0.0  # organizations/enrich is request-quota-limited, not credit-billed


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
    # Remaining-budget cap stuffed into args by the cell-agent loop. Apify's
    # /runs endpoint accepts timeout (seconds) as a query param — we set it
    # proportional to the remaining USD so a stuck actor can't keep billing
    # past the cap. At ~$0.40/CU/hr a conservative 90 sec/dollar bounds the
    # worst-case spend.
    max_cost_usd = args.get("__max_cost_usd")
    timeout_secs = None
    if max_cost_usd is not None and max_cost_usd > 0:
        timeout_secs = max(30, min(300, int(max_cost_usd * 90)))
    # Heartbeat the chat_run so a multi-minute actor poll doesn't trip the
    # staleness sweeper.
    heartbeat = asyncio.create_task(_heartbeat_emitter(ctx, "apify_call_actor"))
    cost_usd = 0.0
    items: List[Dict[str, Any]] = []
    apify_run_id: Optional[str] = None
    try:
        async with httpx.AsyncClient(timeout=180) as client:
            # Async pattern: POST /runs → poll until terminal → fetch items +
            # cost. The /run-sync-get-dataset-items endpoint doesn't expose
            # run ID; this is the only path that gives us real billing.
            run_params: Dict[str, Any] = {"token": api_key}
            if timeout_secs is not None:
                run_params["timeout"] = timeout_secs
            start = await client.post(
                f"https://api.apify.com/v2/acts/{aid}/runs",
                params=run_params,
                json=actor_input,
            )
            if start.status_code >= 400:
                return {"error": f"apify start HTTP {start.status_code}"}, 0.0
            run_data = (start.json() or {}).get("data") or {}
            run_id = run_data.get("id")
            dataset_id = run_data.get("defaultDatasetId")
            if not (run_id and dataset_id):
                return {"error": "apify: no run id"}, 0.0
            # Track at function scope so the CancelledError handler in the
            # outer try can abort the actor and capture partial CU cost.
            apify_run_id = run_id
            terminal = {"SUCCEEDED", "FAILED", "ABORTED", "TIMED-OUT"}
            poll_cap = timeout_secs if timeout_secs is not None else 150
            t0 = asyncio.get_event_loop().time()
            while True:
                if asyncio.get_event_loop().time() - t0 > poll_cap:
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
                await asyncio.sleep(2.0)
            items_resp = await client.get(
                f"https://api.apify.com/v2/datasets/{dataset_id}/items",
                params={"token": api_key, "format": "json", "limit": 5},
            )
            if items_resp.status_code == 200:
                items = items_resp.json() or []
    except asyncio.CancelledError:
        # Abort the actor on the Apify side so it stops consuming CUs,
        # then fetch the final cost (CUs used up to the abort) and
        # bill it to the turn ledger via ctx.partial_cost_usd. Without
        # this, a user cancel mid-actor would (a) leak compute on
        # Apify's servers and (b) hide the cost from billing.
        if apify_run_id:
            try:
                async with httpx.AsyncClient(timeout=10) as abort_client:
                    from dsl_worker.sources_v2.apify_actor import _abort_apify_run_and_get_cost
                    partial_usd = await asyncio.shield(
                        _abort_apify_run_and_get_cost(abort_client, api_key, apify_run_id)
                    )
                    if partial_usd > 0:
                        try:
                            ctx.partial_cost_usd = float(
                                getattr(ctx, "partial_cost_usd", 0.0)
                            ) + partial_usd * 10.0
                        except Exception:
                            pass
            except Exception:
                log.debug("apify abort-on-cancel failed", exc_info=True)
        raise
    finally:
        heartbeat.cancel()
        try:
            await heartbeat
        except asyncio.CancelledError:
            pass
    return {"items": items[:5]}, cost_usd * 10.0


from dsl_worker.chat_v2.light_tools import (
    apify_search_actors as _apify_search_actors,
    apify_actor_details as _apify_actor_details,
    code_exec as _code_exec,
)


async def _heartbeat_emitter(ctx: ToolContext, tool_name: str, interval: float = 60.0) -> None:
    """Loop forever emitting tool_heartbeat events into chat_run_events.

    Used as a background task while a long-running tool (BU, apify) is in
    flight. Keeps the staleness sweeper from flipping the chat_run to
    failed during legitimate multi-minute tool calls. Cancelled by the
    caller when the tool returns.
    """
    run_id = getattr(ctx, "run_id", None)
    if not run_id:
        return
    try:
        while True:
            await asyncio.sleep(interval)
            try:
                from dsl_worker.chat_api import runs as legacy_runs
                from dsl_api.models import ChatRun
                run_obj = ctx.db.query(ChatRun).filter(ChatRun.id == run_id).first()
                if run_obj is not None:
                    legacy_runs.emit_event(ctx.db, run_obj, "tool_heartbeat", {
                        "tool": tool_name,
                    })
            except Exception:
                log.debug("tool_heartbeat emit failed; continuing", exc_info=True)
    except asyncio.CancelledError:
        return


async def _browser_use(args: Dict[str, Any], ctx: ToolContext) -> Tuple[Dict[str, Any], float]:
    try:
        from dsl_worker.infra.bu_client import bu_extract_rows
    except ImportError:
        return {"error": "bu_client not available"}, 0.0
    url = args.get("url")
    task = args.get("task")
    if not (url and task):
        return {"error": "url + task required"}, 0.0
    # max_cost_usd was stuffed into args by the cell-agent loop before
    # dispatch — it's the remaining per-row budget, so BU self-limits
    # inside its session instead of us best-effort capping after the fact.
    max_cost_usd = args.get("__max_cost_usd")
    heartbeat = asyncio.create_task(_heartbeat_emitter(ctx, "browser_use"))

    # On CancelledError, BUClient stops the cloud session and fetches
    # the partial cost via this callback. We attribute it to ctx so the
    # agent's CancelledError handler bills it to the turn ledger —
    # otherwise the user's BU spend up to the abort would be free,
    # which we DO have to pay for on BU's side.
    def _bill_partial(usd: float) -> None:
        try:
            ctx.partial_cost_usd = float(getattr(ctx, "partial_cost_usd", 0.0)) + usd
        except Exception:
            pass

    try:
        rows, cost = await bu_extract_rows(
            url=url,
            task=task,
            candidate_description=args.get("candidate_description", ""),
            max_cost_usd=max_cost_usd,
            on_partial_cost=_bill_partial,
            # Per-cell budget is much tighter than the table-level
            # extraction budget — we're answering one field for one
            # row, not crawling a directory. 5 actions covers
            # "open page → find the cell value → return".
            action_budget=5,
            # If the cell value is a URL, BU must have actually visited
            # it (not just spotted it in a link list) before returning.
            # Guards against the 404-y URLs the cell agent used to
            # surface from search snippets.
            include_url_check=True,
        )
    finally:
        heartbeat.cancel()
        try:
            await heartbeat
        except asyncio.CancelledError:
            pass
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
    # web_search is the OpenAI hosted tool — the model invokes it
    # server-side as part of its own Responses call. We don't dispatch
    # it ourselves; web_search_call items in the response output are
    # handled in the cell loop directly (billing + tool_calls_log).
    "browser_use": _browser_use,
    "code_exec": _code_exec,
}


# ---------------------------------------------------------------------------
# Cell agent loop
# ---------------------------------------------------------------------------


CELL_SYSTEM_PROMPT = """You are a cell agent: fill specific columns for ONE row of a table.

# Inputs (JSON)

- `row_visible_to_user` — the row's already-filled fields as shown in the user's table.
- `row_hidden_source_fields` (optional) — fields the source returned but the orchestrator didn't surface as columns. Use as extra context when reasoning. If the value you need is hiding here, return it via `final_result` for the column in `columns_to_fill`.
- `columns_to_fill` — column names you must produce values for.
- `instruction` — what to find or compute.

# Finishing

Always end with `final_result({values: {col_name: value, ...}})` where `col_name` is the EXACT name from `columns_to_fill`. Do NOT invent keys like `label`, `value`, `answer`, `result`.

Set a column to `null` when the value genuinely doesn't exist. Null is fine. Don't fabricate.

When you used an external tool to find a value, also pass `sources`: `final_result({values: {...}, sources: {col_name: [citation, ...]}})`.

Two citation shapes — pick the one that matches the tool:

- **Web hits** (`web_search`, browser_use pages you visited): `{type: "url", value: "https://example.com/full-page-url"}`. Use the actual page URL. NEVER use OpenAI's internal annotation pointers like `turn1search3` — those are useless to the user.
- **Paid services** (`fullenrich_enrich_phone`, `apollo_org_enrich`, `google_maps_place_details`, `apify_call_actor`, etc.): `{source: "<tool_name>"}` — just the tool name, no value/field needed.

Omit `sources` entirely when the value came purely from reasoning over `row_visible_to_user` / `row_hidden_source_fields`.

# Output format

- **Yes/No** → enum-style `"Yes"` / `"No"` (Title Case, never booleans).
- **Numbers** → plain numeric, e.g. `5000000`, never `"$5M"`. The column's format renders it pretty.
- **Dates** → ISO 8601: `"2026-05-15"` or `"2026-05-15T10:30:00Z"`.
- **URLs** → only commit a URL you actually visited and verified. Don't construct URLs from name slugs or guess identifiers; if you didn't open and read the page, return null.

# Picking a tool

If you have no tools at all, the answer must come from `row_visible_to_user` + `row_hidden_source_fields` alone. Reason carefully and call `final_result` directly.

## Web data — strict escalation order

Pick by *accessibility*, not by what the instruction sounds like. The rule is empirical: try the cheaper tool, if it gets the answer you're done, if not escalate.

1. **`web_search`** — ALWAYS try this first. Native OpenAI web search, cheap, fast. Works for the vast majority of public web pages including most static / server-rendered content. Don't try to predict whether it'll work — just call it.
2. **`apify_call_actor`** — only if `web_search` didn't return the data you need AND there's a platform-specific actor that covers the source (Reddit, LinkedIn, Twitter/X, Instagram, etc.). Use `apify_search_actors` + `apify_actor_details` to discover. Bounded to `maxItems=5` at cell level — for per-row lookups, not bulk fetching.
3. **`browser_use`** — last resort. Only after both `web_search` and apify have failed (or there's no apify actor for the source). Real headless browser, expensive ($0.50–$3+ per session, sometimes more). This is for the cases where nothing else can reach the data — JS-only content with no server fallback, login walls, form interactions, infinite-scroll pages.

Do NOT pick `browser_use` predictively. Even if the instruction mentions a URL, even if the page sounds JS-heavy — try `web_search` first. It's cheap, and most of the time it works.

## Known per-row API calls

Use these when the row already has the inputs and the column wants the matching field. They're direct lookups, not searches:

- **`fullenrich_enrich_email`** — verified business email. Inputs: `first_name`, `last_name`, `domain`. ~0.5 cr.
- **`fullenrich_enrich_phone`** — verified phone. Same inputs. **~5 cr — expensive**, only when the column explicitly asks for phone.
- **`fullenrich_enrich_company`** — company-level enrichment. Input: `domain`. ~0.5 cr.
- **`apollo_org_enrich`** — Apollo company data (headcount, revenue, funding, tech stack, etc.). Input: `domain`. ~1 cr.
- **`google_maps_place_details`** — local business info. Input: `place_id` (must already be on the row from a prior Google Maps fetch).

## Computation / parsing

- **`code_exec`** — Python sandbox. For string parsing, math, regex, transforms on row data. No external network — pure compute only.
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
                },
                "sources": {
                    "type": "object",
                    "description": (
                        "Optional map of column_name → list of source citations "
                        "describing where each value came from. Each citation is "
                        "{source, source_field?}: `source` is the tool you used "
                        "(e.g. fullenrich_enrich_phone, apollo_org_enrich, "
                        "web_search, browser_use); `source_field` is the JSON "
                        "path into that tool's result (e.g. "
                        "\"phone_numbers[0].sanitized\"). Omit if no external "
                        "tool produced the value."
                    ),
                },
            },
            "required": ["values"],
        },
    }


def _tool_defs_for_tier(tier_cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Responses-API tool definitions, scoped to the research level.

    Each function-style tool gets its own param schema + a "when to use"
    description. Hosted web_search is appended as {"type": "web_search"}
    so the model invokes it server-side as part of its own Responses
    call (no sidecar round-trip).
    """
    defs: List[Dict[str, Any]] = [_final_result_tool_def()]
    if tier_cfg["tools"] != "all":
        return defs
    # web_search first in the list — reinforces the STEP-1 escalation
    # framing in the system prompt at the tool-picker.
    defs.append({"type": "web_search"})
    defs.extend(_CELL_TOOL_DEFS)
    return defs


# Per-call cost for hosted web_search — see dsl_worker/billing/web_search.py
# for the full billing model. Imported (not re-defined) so the orchestrator,
# cell agent, and web_harvest all read from the same source.
from dsl_worker.billing.web_search import WEB_SEARCH_CALL_COST_USD  # noqa: E402


_CELL_TOOL_DEFS: List[Dict[str, Any]] = [
    # web_search is the OpenAI hosted tool — added in _tool_defs_for_tier
    # as {"type": "web_search"}, not as a function we dispatch. The model
    # invokes it server-side as part of its own Responses call; results
    # come back inline so it has them in its own context window. Cost is
    # OpenAI's per-call fee (added manually below — TrackedClient only
    # computes token cost).
    {
        "type": "function",
        "name": "apify_search_actors",
        "description": "Discover Apify actors that cover a platform (Reddit, LinkedIn, Twitter/X, Instagram, etc.). Use when web_search didn't return the data you need and you're escalating to apify.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Platform or site name to find actors for."},
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "apify_actor_details",
        "description": "Read an Apify actor's input schema + pricing before calling apify_call_actor.",
        "parameters": {
            "type": "object",
            "properties": {
                "actor_id": {"type": "string", "description": "Actor ID, e.g. 'apify/web-scraper'."},
            },
            "required": ["actor_id"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "apify_call_actor",
        "description": (
            "STEP 2 of the web-access escalation: try this when web_search "
            "couldn't get the data AND a platform-specific actor covers the "
            "source. Bounded to maxItems=5 at cell level — for per-row "
            "lookups, not bulk fetches. Costs ~1 cr typical; varies by actor. "
            "Do NOT use as a first step."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "actor_id": {"type": "string"},
                "input": {
                    "type": "object",
                    "description": "Actor-specific input. Read apify_actor_details first.",
                    "additionalProperties": True,
                },
            },
            "required": ["actor_id", "input"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "browser_use",
        "description": (
            "STEP 3 of the web-access escalation: LAST RESORT only. Use ONLY "
            "after both web_search and apify have failed (or there's no apify "
            "actor for the source). Real headless browser session; expensive "
            "($0.50–$3+ typical, sometimes more). Don't pick this predictively "
            "from the task description — escalate to it only after the cheaper "
            "tools demonstrably can't reach the data."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "Starting URL for the browser session."},
                "task": {"type": "string", "description": "What to do on the page (extract X, click Y, fill form Z)."},
                "candidate_description": {
                    "type": "string",
                    "description": "Optional: shape of each row to extract, e.g. '{name, role, headshot_url}'.",
                },
            },
            "required": ["url", "task"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "fullenrich_enrich_email",
        "description": (
            "Verified business email lookup via FullEnrich. Use when the row has "
            "first_name + last_name + domain and the column wants email. "
            "~1 cr per successful match (no charge on miss). "
            "**Pass `linkedin_url` whenever the row has it** — FE's waterfall is "
            "dramatically better with a LinkedIn URL; many contacts return empty "
            "without it but DELIVERABLE with it."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "first_name": {"type": "string"},
                "last_name": {"type": "string"},
                "domain": {"type": "string", "description": "Company domain like 'anthropic.com'."},
                "company": {"type": "string", "description": "Optional fallback when domain isn't on the row."},
                "linkedin_url": {
                    "type": "string",
                    "description": "Person's LinkedIn profile URL if known. Improves hit rate substantially — include it whenever the row has it.",
                },
            },
            "required": ["first_name", "last_name"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "fullenrich_enrich_phone",
        "description": (
            "Verified phone lookup via FullEnrich. EXPENSIVE — ~5 cr per "
            "successful match. Only use when the column explicitly asks for "
            "a phone number. Same inputs as email; pass linkedin_url when the "
            "row has it for a better hit rate."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "first_name": {"type": "string"},
                "last_name": {"type": "string"},
                "domain": {"type": "string"},
                "company": {"type": "string"},
                "linkedin_url": {"type": "string"},
            },
            "required": ["first_name", "last_name"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "fullenrich_enrich_company",
        "description": "Company-level enrichment from FullEnrich. Input: domain. ~0.5 cr.",
        "parameters": {
            "type": "object",
            "properties": {
                "domain": {"type": "string"},
            },
            "required": ["domain"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "apollo_org_enrich",
        "description": (
            "Company info from Apollo: headcount, revenue, funding stage, tech "
            "stack, industry, LinkedIn URL, etc. Input: domain. Effectively "
            "free — uses Apollo's request quota, not credit balance. Use when "
            "the column wants company-level data and the row has a domain."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "domain": {"type": "string"},
            },
            "required": ["domain"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "google_maps_place_details",
        "description": (
            "Local business detail lookup. Requires a Google Maps place_id that's "
            "already on the row from a prior Google Maps fetch. ~0.3 cr."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "place_id": {"type": "string"},
            },
            "required": ["place_id"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "code_exec",
        "description": (
            "Python sandbox. For parsing, string transforms, math, regex on the "
            "row data. No external network — pure compute only. Useful when the "
            "instruction wants you to transform a value (e.g. extract domain "
            "from URL, parse a date, normalize a number)."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "code": {"type": "string", "description": "Python source. Last expression's value is returned."},
                "files": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional file paths to make available in the sandbox.",
                },
            },
            "required": ["code"],
            "additionalProperties": False,
        },
    },
]


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


# Tool names that don't represent an external data source — pure compute or
# control flow. Inference skips these when guessing the citation for a value.
_NON_SOURCE_TOOLS = {"final_result", "code_exec"}

# Map raw tool names → canonical sources_v2 kinds. The FE's sourceDisplay()
# only renders nice labels + favicons for canonical kinds; raw tool names
# fall through to ugly "fullenrich enrich phone" text. Anything not in this
# map passes through unchanged.
_TOOL_TO_SOURCE_KIND: Dict[str, str] = {
    "fullenrich_enrich_phone": "fullenrich_people",
    "fullenrich_enrich_email": "fullenrich_people",
    "fullenrich_enrich_company": "fullenrich_people",
    "apollo_org_enrich": "apollo_companies",
    "google_maps_place_details": "google_maps",
    "browser_use": "browser_use",
    "apify_call_actor": "apify_actor",
    "web_search": "web_search",
}


def _normalize_tool_to_kind(tool_name: str) -> str:
    """Map a cell-agent tool name to a canonical sources_v2 kind for FE display."""
    return _TOOL_TO_SOURCE_KIND.get(tool_name, tool_name)


def _looks_like_url(s: Any) -> bool:
    return isinstance(s, str) and (s.startswith("http://") or s.startswith("https://"))


def _infer_source_from_tool_calls(
    tool_calls_log: List[Dict[str, Any]],
) -> Optional[Dict[str, str]]:
    """Walk tool_calls_log backwards; return the most recent data-producing
    tool as a source_record citation. Returns None if no such tool ran.

    Used as a fallback when the cell agent fills a value but forgets to
    declare `sources` in `final_result`.

    Special case: when the source tool is apify_call_actor, append the
    specific actor_id so the FE can render a clickable link to
    apify.com/{actor_id} instead of a generic "Apify actor" tag with
    no drill-through.
    """
    for entry in reversed(tool_calls_log):
        name = entry.get("name") or ""
        if not name or name in _NON_SOURCE_TOOLS:
            continue
        cost = entry.get("cost") or 0.0
        result_preview = entry.get("result_preview") or ""
        if cost <= 0 and "error" in result_preview.lower():
            continue
        kind = _normalize_tool_to_kind(name)
        # Apify: append the actor_id so the FE can resolve a specific
        # actor URL. Citation becomes "apify_actor:username/actor-name".
        if name == "apify_call_actor":
            actor_id = (entry.get("args") or {}).get("actor_id")
            if isinstance(actor_id, str) and actor_id:
                kind = f"apify_actor:{actor_id}"
        return {"type": "source_record", "source": kind}
    return None


def _coerce_sources_keys(
    raw_sources: Any,
    canonical_columns: List[str],
    tool_calls_log: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, List[Dict[str, Any]]]:
    """Normalize the LLM's `sources` arg into {canonical_col: [citation, ...]}.

    Output shapes (matching frontend/src/components/project/CellDetailPanel.tsx):
      - {type: "url", value: "https://..."}              — web hits, real URLs
      - {type: "source_record", source: "<kind>"}        — paid APIs, service-level

    For apify_call_actor sources we resolve to "apify_actor:<actor_id>"
    using the most recent apify call in tool_calls_log, so the FE can
    render a clickable link to that specific actor on apify.com instead
    of a generic "Apify actor" tag.

    `source_field` is intentionally dropped from source_record citations: the
    FE's drill-through panel queries the sample's raw_row, which doesn't
    exist for enrichment. Cell-trace drill-through is future work.
    """
    if not isinstance(raw_sources, dict) or not raw_sources or not canonical_columns:
        return {}

    # Pre-resolve the most recent apify actor_id once so we don't walk
    # tool_calls_log for every column × every citation.
    last_apify_actor_id: Optional[str] = None
    for entry in reversed(tool_calls_log or []):
        if entry.get("name") == "apify_call_actor":
            aid = (entry.get("args") or {}).get("actor_id")
            if isinstance(aid, str) and aid:
                last_apify_actor_id = aid
                break

    def _norm(s: str) -> str:
        return "".join(ch for ch in s.lower() if ch.isalnum())

    canon_by_norm = {_norm(c): c for c in canonical_columns}
    out: Dict[str, List[Dict[str, Any]]] = {}
    for k, v in raw_sources.items():
        col = k if k in canonical_columns else canon_by_norm.get(_norm(k))
        if not col:
            continue
        citations = v if isinstance(v, list) else [v]
        normalized: List[Dict[str, Any]] = []
        for c in citations:
            # Allow bare strings: URL → url citation, else → service name.
            if isinstance(c, str):
                if _looks_like_url(c):
                    c = {"type": "url", "value": c}
                else:
                    c = {"type": "source_record", "source": c}
            if not isinstance(c, dict):
                continue
            declared_type = c.get("type")
            # URL citation — pass through if the value is actually a URL.
            url_value = c.get("value")
            if (declared_type == "url" or _looks_like_url(url_value)) and _looks_like_url(url_value):
                normalized.append({"type": "url", "value": url_value})
                continue
            # source_record citation — normalize the source name; drop
            # source_field (no drill-through endpoint for enrichment yet).
            src = c.get("source")
            if src:
                kind = _normalize_tool_to_kind(src)
                # Apify: append the specific actor_id from the run log
                # so the FE links to apify.com/{actor_id}. The LLM may
                # also include actor_id directly on the citation —
                # honor that if present.
                if kind == "apify_actor":
                    explicit_aid = c.get("actor_id") if isinstance(c.get("actor_id"), str) else None
                    aid = explicit_aid or last_apify_actor_id
                    if aid:
                        kind = f"apify_actor:{aid}"
                normalized.append({
                    "type": "source_record",
                    "source": kind,
                })
        if normalized:
            out[col] = normalized
    return out


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
) -> Tuple[Dict[str, Any], Dict[str, List[Dict[str, Any]]], float, str]:
    """Per-row Responses-API loop with research-level routing.

    Returns (new_fields_dict, sources_per_column, total_cost_credits, status).
      status ∈ {"filled", "hit_budget", "error"}
        - filled: cell agent emitted final_result (values may still be null
          if the answer genuinely didn't exist)
        - hit_budget: programmatic kill — per_row_credit_cap reached before
          final_result. FE renders a "hit budget" badge.
        - error: LLM call failed or no parseable result. Cell left untouched.

    sources_per_column maps {col_name → [source_record citation, ...]} for
    each filled value. Same shape as the fetch-side sources stored in
    samples.tags.sources. Populated either from the agent's declared
    `sources` arg in final_result, or inferred from tool_calls_log when the
    agent omits it.

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
    final_sources: Dict[str, List[Dict[str, Any]]] = {}
    tool_calls_log: List[Dict[str, Any]] = []
    error_str: Optional[str] = None
    t0 = time.monotonic()

    def _build_sources_with_fallback(
        declared: Dict[str, List[Dict[str, Any]]],
        values: Dict[str, Any],
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Fill in implicit citations for columns the agent set but didn't
        cite. Uses the last informative tool call as the fallback source."""
        if not values:
            return declared
        fallback = _infer_source_from_tool_calls(tool_calls_log)
        if not fallback:
            return declared
        out = dict(declared)
        for col, v in values.items():
            if v in (None, "") or col in out:
                continue
            out[col] = [fallback]
        return out

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
                return final_values, final_sources, total_cost, status

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
                return final_values, final_sources, total_cost, "error"

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
                elif itype == "web_search_call":
                    # Hosted tool already ran server-side. Bill the
                    # per-call fee (TrackedClient only sums token cost)
                    # and log it to tool_calls_log so cell_traces shows
                    # it just like any other tool the cell agent used.
                    query = ""
                    try:
                        action = getattr(item, "action", None)
                        if action is not None:
                            query = getattr(action, "query", "") or ""
                    except Exception:
                        pass
                    total_cost += WEB_SEARCH_CALL_COST_USD
                    tool_calls_log.append({
                        "name": "web_search",
                        "args": {"query": query},
                        "result_preview": f"native (status={getattr(item, 'status', '?')})",
                        "cost": WEB_SEARCH_CALL_COST_USD,
                    })
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
                            final_sources = _build_sources_with_fallback(
                                _coerce_sources_keys(data.get("sources"), list(final_values.keys()), tool_calls_log),
                                final_values,
                            )
                            return final_values, final_sources, total_cost, "filled"
                        if isinstance(data, dict):
                            final_values = data
                            final_sources = _build_sources_with_fallback({}, final_values)
                            return final_values, final_sources, total_cost, "filled"
                    except json.JSONDecodeError:
                        pass
                error_str = error_str or "no function call and no parseable message"
                return final_values, final_sources, total_cost, "error"

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
                    declared_sources = _coerce_sources_keys(
                        args.get("sources"), list(final_values.keys()), tool_calls_log
                    )
                    final_sources = _build_sources_with_fallback(
                        declared_sources, final_values
                    )
                    tool_calls_log.append({
                        "name": "final_result",
                        "args": args,
                        "coerced_values": final_values,
                        "coerced_sources": final_sources,
                        "cost": 0.0,
                    })
                    return final_values, final_sources, total_cost, "filled"

                handler = CELL_TOOL_HANDLERS.get(name)
                if not handler:
                    tool_result: Dict[str, Any] = {"error": f"unknown tool {name}"}
                    tool_cost = 0.0
                else:
                    # PRE-TOOL BUDGET GATE — strict, programmatic.
                    # Three policies depending on the tool category.
                    remaining = tier_cfg["cap"] - total_cost
                    tool_result, tool_cost = None, 0.0
                    skip_reason: Optional[str] = None

                    if name in FIXED_COST_TOOLS:
                        # Single-call paid APIs (Apollo, FE, gmaps). Their
                        # success-billing cost is deterministic — refuse pre-
                        # call if we wouldn't be able to afford the result.
                        est = FIXED_COST_TOOLS[name]
                        if remaining < est:
                            skip_reason = (
                                f"skipped: {name} costs ~{est} cr but only "
                                f"{remaining:.2f} cr remaining of per-row cap"
                            )
                    elif name == "browser_use":
                        if remaining < BU_MIN_BUDGET:
                            skip_reason = (
                                f"skipped: browser_use needs at least "
                                f"{BU_MIN_BUDGET} cr; {remaining:.2f} remaining"
                            )
                        else:
                            args["__max_cost_usd"] = float(remaining)
                    elif name == "apify_call_actor":
                        if remaining < APIFY_MIN_BUDGET:
                            skip_reason = (
                                f"skipped: apify_call_actor needs at least "
                                f"{APIFY_MIN_BUDGET} cr; {remaining:.2f} remaining"
                            )
                        else:
                            args["__max_cost_usd"] = float(remaining)

                    if skip_reason is not None:
                        tool_result = {"error": "budget", "message": skip_reason}
                        tool_cost = 0.0
                        log.info("cell agent pre-tool skip: %s", skip_reason)
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
                    return final_values, final_sources, total_cost, "hit_budget"

        error_str = f"hit HARD_TURN_LIMIT={HARD_TURN_LIMIT}"
        log.warning(
            "cell agent hit HARD_TURN_LIMIT=%d (research=%s) — emergency stop",
            HARD_TURN_LIMIT, tier_cfg["name"],
        )
        return final_values, final_sources, total_cost, "error"
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
