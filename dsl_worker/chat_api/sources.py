"""Source tools for chat-mode projects.

Wraps the worker's source-tool clients (FullEnrich, Apollo, Apify, Google
Maps, sandbox, browser_use) as flat OpenAI function-tools the chat agent
can call. Each handler returns data inline as JSON for the agent to
inspect, plus a cost figure that the chat handler charges to the user's
credit balance at the end of the turn.

Filter-translation logic for FullEnrich is duplicated locally rather than
imported from worker/dsl_worker/agents/integrations/fullenrich.py so we
can iterate independently of V13.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple

from dsl_worker.infra.fullenrich_client import FullEnrichClient
from dsl_worker.infra.apollo_client import ApolloClient
from dsl_worker.infra.apify_client import ApifyClient
from dsl_worker.infra.google_maps_client import GoogleMapsClient
from dsl_worker.agents.integrations.apollo import _simplify_person, _simplify_company, _select_companies
from sandbox_service import SandboxClient
from openai import AsyncOpenAI

from dsl_api.config import settings as _api_settings

from dsl_worker.chat_api import candidates


def _persist_candidates(
    project_id: Optional[Any],
    tool: str,
    items: List[Dict[str, Any]],
    *,
    cost_usd: float = 0.0,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Write items to a candidates blob file (if project_id given) and
    return the canonical tool-result dict (file path, count, fields,
    preview). When project_id is None — only happens in tests/dev — we
    fall back to a degenerate "preview-only" shape so the tool still
    runs end-to-end without blob writes."""
    if project_id is None:
        fields = sorted({k for it in items if isinstance(it, dict) for k in it.keys()})
        return {
            "candidates_file": None,
            "tool": tool,
            "items_count": len(items),
            "fields": fields,
            "cost_usd": round(cost_usd, 4),
            "preview": items[:candidates.PREVIEW_ITEMS],
            "run_metadata": extra or {},
        }
    try:
        meta = candidates.write_candidates(
            project_id, tool, items, cost_usd=cost_usd, extra=extra
        )
    except Exception as e:
        log.exception("write_candidates failed for %s", tool)
        return {
            "error": f"failed to persist candidates: {type(e).__name__}: {e}",
            "items_count_attempted": len(items),
            "preview": items[:candidates.PREVIEW_ITEMS],
        }
    return candidates.make_tool_result(
        meta=meta, preview_items=items[:candidates.PREVIEW_ITEMS]
    )


log = logging.getLogger(__name__)


# Cost — FE bills in credits; 1 credit per work email, 10 per phone, ~0.001
# credits per search-people result. We bill the user at our own
# COMPUTE_COST_PER_CREDIT downstream; this is the raw $ to us.
_FE_COST_PER_CREDIT_USD = float(os.getenv("FULLENRICH_COST_PER_CREDIT", "0.055"))
_APOLLO_COST_PER_CREDIT_USD = 0.024  # ~$24/1000 credits


# ---------------------------------------------------------------------------
# Lazy client singletons
# ---------------------------------------------------------------------------


_fe_client: Optional[FullEnrichClient] = None
_apollo_client: Optional[ApolloClient] = None
_apify_client: Optional[ApifyClient] = None
_gmaps_client: Optional[GoogleMapsClient] = None


def _fe() -> Optional[FullEnrichClient]:
    global _fe_client
    if _fe_client is None:
        key = os.getenv("FULLENRICH_API_KEY")
        if not key:
            log.warning("FULLENRICH_API_KEY not set — fullenrich tools disabled")
            return None
        _fe_client = FullEnrichClient(api_key=key)
    return _fe_client


def _apollo() -> Optional[ApolloClient]:
    global _apollo_client
    if _apollo_client is None:
        key = os.getenv("APOLLO_API_KEY")
        if not key:
            log.warning("APOLLO_API_KEY not set — apollo tools disabled")
            return None
        _apollo_client = ApolloClient(api_key=key)
    return _apollo_client


def _apify() -> Optional[ApifyClient]:
    global _apify_client
    if _apify_client is None:
        key = os.getenv("APIFY_API_KEY")
        if not key:
            log.warning("APIFY_API_KEY not set — apify tools disabled")
            return None
        _apify_client = ApifyClient(api_key=key)
    return _apify_client


def _gmaps() -> Optional[GoogleMapsClient]:
    global _gmaps_client
    if _gmaps_client is None:
        key = os.getenv("GOOGLE_API_KEY")
        if not key:
            log.warning("GOOGLE_API_KEY not set — google_maps tools disabled")
            return None
        _gmaps_client = GoogleMapsClient(api_key=key)
    return _gmaps_client


# ---------------------------------------------------------------------------
# FE filter helpers (mirror worker's build_filter / build_range_filter)
# ---------------------------------------------------------------------------


def _build_filter(
    values: List[str], exact_match: bool = False, exclude: bool = False
) -> List[Dict[str, Any]]:
    return [{"value": v, "exact_match": exact_match, "exclude": exclude} for v in values]


def _build_range_filter(
    min_val: Optional[int] = None,
    max_val: Optional[int] = None,
    exclude: bool = False,
) -> List[Dict[str, Any]]:
    entry: Dict[str, Any] = {"exclude": exclude}
    if min_val is not None:
        entry["min"] = min_val
    if max_val is not None:
        entry["max"] = max_val
    return [entry]


# ---------------------------------------------------------------------------
# Tool handlers — each returns (result_text_for_llm, cost_usd)
# ---------------------------------------------------------------------------


async def _fe_search_people(args: Dict[str, Any], *, project_id: Optional[Any] = None) -> Tuple[str, float]:
    client = _fe()
    if client is None:
        return json.dumps({"error": "FULLENRICH_API_KEY not configured"}), 0.0

    filters: Dict[str, Any] = {}
    string_params = [
        ("titles", "current_position_titles"),
        ("locations", "person_locations"),
        ("industries", "current_company_industries"),
        ("seniority", "current_position_seniority_level"),
        ("skills", "person_skills"),
        ("universities", "person_universities"),
        ("company_names", "current_company_names"),
        ("company_domains", "current_company_domains"),
        ("company_specialties", "current_company_specialties"),
        ("company_types", "current_company_types"),
        ("company_hq", "current_company_headquarters"),
        ("names", "person_names"),
        ("linkedin_urls", "person_linkedin_urls"),
        ("past_companies", "past_company_names"),
        ("past_titles", "past_position_titles"),
    ]
    for param, api_key in string_params:
        v = args.get(param)
        if not v:
            continue
        if isinstance(v, list):
            filters[api_key] = _build_filter(v)
        elif isinstance(v, str):
            filters[api_key] = _build_filter([v])

    range_params = [
        ("company_headcount_min", "company_headcount_max", "current_company_headcounts"),
        ("years_in_position_min", "years_in_position_max", "current_position_years_in"),
        ("years_at_company_min", "years_at_company_max", "current_company_years_at"),
        ("company_founded_min", "company_founded_max", "current_company_founded_years"),
        ("days_since_job_change_min", "days_since_job_change_max", "current_company_days_since_last_job_change"),
    ]
    for pmin, pmax, api_key in range_params:
        mn = args.get(pmin)
        mx = args.get(pmax)
        if mn is not None or mx is not None:
            filters[api_key] = _build_range_filter(mn, mx)

    limit = int(args.get("limit", 25))
    offset = int(args.get("offset", 0))

    try:
        result = await client.search_people(filters=filters, limit=limit, offset=offset)
    except Exception as e:
        return json.dumps({"error": f"{type(e).__name__}: {e}"}), 0.0

    if "error" in result:
        return json.dumps(result), 0.0

    # FE returns {"people": [...], "metadata": {...}}.
    items = result.get("people", []) or result.get("data", []) or []
    metadata = result.get("metadata", {}) or {}
    total = metadata.get("total", result.get("total", len(items)))
    credits = float(metadata.get("credits_used", result.get("credits_used", 0)) or 0)
    cost_usd = credits * _FE_COST_PER_CREDIT_USD

    persisted = _persist_candidates(
        project_id,
        "fullenrich_search_people",
        items,
        cost_usd=cost_usd,
        extra={
            "total_in_db": total,
            "credits_used": credits,
            "next_offset": offset + len(items) if len(items) >= limit else None,
        },
    )
    return json.dumps(persisted, default=str), cost_usd


async def _fe_search_companies(args: Dict[str, Any], *, project_id: Optional[Any] = None) -> Tuple[str, float]:
    client = _fe()
    if client is None:
        return json.dumps({"error": "FULLENRICH_API_KEY not configured"}), 0.0

    filters: Dict[str, Any] = {}
    string_params = [
        ("industries", "industries"),
        ("locations", "headquarters_locations"),
        ("specialties", "specialties"),
        ("keywords", "keywords"),
        ("names", "names"),
        ("domains", "domains"),
        ("types", "types"),
    ]
    for param, api_key in string_params:
        v = args.get(param)
        if not v:
            continue
        if isinstance(v, list):
            filters[api_key] = _build_filter(v)
        elif isinstance(v, str):
            filters[api_key] = _build_filter([v])

    range_params = [
        ("headcount_min", "headcount_max", "headcounts"),
        ("founded_min", "founded_max", "founded_years"),
    ]
    for pmin, pmax, api_key in range_params:
        mn = args.get(pmin)
        mx = args.get(pmax)
        if mn is not None or mx is not None:
            filters[api_key] = _build_range_filter(mn, mx)

    limit = int(args.get("limit", 25))
    offset = int(args.get("offset", 0))

    try:
        result = await client.search_companies(filters=filters, limit=limit, offset=offset)
    except Exception as e:
        return json.dumps({"error": f"{type(e).__name__}: {e}"}), 0.0

    if "error" in result:
        return json.dumps(result), 0.0

    # FE returns {"companies": [...], "metadata": {...}}.
    items = result.get("companies", []) or result.get("data", []) or []
    metadata = result.get("metadata", {}) or {}
    total = metadata.get("total", result.get("total", len(items)))
    credits = float(metadata.get("credits_used", result.get("credits_used", 0)) or 0)
    cost_usd = credits * _FE_COST_PER_CREDIT_USD

    persisted = _persist_candidates(
        project_id,
        "fullenrich_search_companies",
        items,
        cost_usd=cost_usd,
        extra={
            "total_in_db": total,
            "credits_used": credits,
            "next_offset": offset + len(items) if len(items) >= limit else None,
        },
    )
    return json.dumps(persisted, default=str), cost_usd


async def _fe_enrich_contacts(args: Dict[str, Any], *, project_id: Optional[Any] = None) -> Tuple[str, float]:
    """Enrich a small batch of contacts (≤25 recommended) with verified
    work emails and/or phones via FullEnrich's waterfall enrichment.

    Each contact dict needs at minimum: `linkedin_url` OR
    (`first_name` + `last_name` + `domain`).
    """
    client = _fe()
    if client is None:
        return json.dumps({"error": "FULLENRICH_API_KEY not configured"}), 0.0

    contacts = args.get("contacts") or []
    if not isinstance(contacts, list) or not contacts:
        return json.dumps({"error": "`contacts` must be a non-empty array"}), 0.0
    if len(contacts) > 25:
        return json.dumps({"error": "max 25 contacts per call; batch into multiple calls"}), 0.0

    fields = args.get("fields") or ["emails", "phones"]
    enrich_fields: List[str] = []
    for f in fields:
        f = str(f).lower()
        if "email" in f and "personal" not in f:
            enrich_fields.append("contact.emails")
        if "phone" in f:
            enrich_fields.append("contact.phones")
        if "personal" in f:
            enrich_fields.append("contact.personal_emails")
    if not enrich_fields:
        enrich_fields = ["contact.emails", "contact.phones"]

    try:
        result = await client.enrich_contacts(
            contacts=contacts,
            name=f"chat_enrich_{len(contacts)}",
            enrich_fields=enrich_fields,
        )
    except Exception as e:
        return json.dumps({"error": f"{type(e).__name__}: {e}"}), 0.0

    if "error" in result:
        return json.dumps(result), 0.0

    data = result.get("data", []) or []
    credits = float(result.get("cost", {}).get("credits", 0) or 0)
    cost_usd = credits * _FE_COST_PER_CREDIT_USD

    persisted = _persist_candidates(
        project_id,
        "fullenrich_enrich_contacts",
        data,
        cost_usd=cost_usd,
        extra={"credits_used": credits, "enriched": len(data)},
    )
    return json.dumps(persisted, default=str), cost_usd


# ---------------------------------------------------------------------------
# Apollo tool handlers
# ---------------------------------------------------------------------------


async def _apollo_search_companies(args: Dict[str, Any], *, project_id: Optional[Any] = None) -> Tuple[str, float]:
    client = _apollo()
    if client is None:
        return json.dumps({"error": "APOLLO_API_KEY not configured"}), 0.0

    page = int(args.get("page", 1))
    per_page = int(args.get("per_page", 25))
    try:
        orgs, total = await client.search_companies(
            organization_keywords=args.get("keywords") or None,
            organization_name=args.get("name") or None,
            organization_locations=args.get("locations") or None,
            organization_not_locations=args.get("not_locations") or None,
            organization_num_employees_ranges=args.get("employee_ranges") or None,
            organization_revenue_ranges=args.get("revenue_ranges") or None,
            website_urls=args.get("domains") or None,
            industry_tag_ids=args.get("industries") or None,
            technology_uids=args.get("technologies") or None,
            per_page=per_page,
            page=page,
        )
    except Exception as e:
        return json.dumps({"error": f"{type(e).__name__}: {e}"}), 0.0

    simplified = [_simplify_company(o) for o in (orgs or [])]
    persisted = _persist_candidates(
        project_id,
        "apollo_search_companies",
        simplified,
        cost_usd=0.0,
        extra={
            "total": total,
            "page": page,
            "per_page": per_page,
            "next_page": page + 1 if total and (page * per_page) < total else None,
        },
    )
    return json.dumps(persisted, default=str), 0.0  # Apollo company search has no per-result cost


async def _apollo_enrich_person(args: Dict[str, Any], *, project_id: Optional[Any] = None) -> Tuple[str, float]:
    client = _apollo()
    if client is None:
        return json.dumps({"error": "APOLLO_API_KEY not configured"}), 0.0
    try:
        person = await client.enrich_person(
            apollo_id=args.get("apollo_id") or None,
            first_name=args.get("first_name") or None,
            last_name=args.get("last_name") or None,
            name=args.get("name") or None,
            email=args.get("email") or None,
            organization_name=args.get("company") or None,
            domain=args.get("domain") or None,
            linkedin_url=args.get("linkedin_url") or None,
        )
    except Exception as e:
        return json.dumps({"error": f"{type(e).__name__}: {e}"}), 0.0
    if not person:
        return json.dumps({"matched": False}), 0.0
    return (
        json.dumps({"matched": True, "person": _simplify_person(person)}, default=str),
        _APOLLO_COST_PER_CREDIT_USD,
    )


async def _apollo_enrich_company(args: Dict[str, Any], *, project_id: Optional[Any] = None) -> Tuple[str, float]:
    client = _apollo()
    if client is None:
        return json.dumps({"error": "APOLLO_API_KEY not configured"}), 0.0
    domain = args.get("domain")
    name = args.get("name")
    if not domain and not name:
        return json.dumps({"error": "provide domain or name"}), 0.0
    try:
        org = await client.enrich_company(domain=domain or None, name=name or None)
    except Exception as e:
        return json.dumps({"error": f"{type(e).__name__}: {e}"}), 0.0
    if not org:
        return json.dumps({"matched": False}), 0.0
    return (
        json.dumps({"matched": True, "company": _simplify_company(org)}, default=str),
        0.0,  # Apollo company enrichment is free in our tier
    )


# ---------------------------------------------------------------------------
# Apify tool handlers
# ---------------------------------------------------------------------------


async def _apify_search_actors(args: Dict[str, Any], *, project_id: Optional[Any] = None) -> Tuple[str, float]:
    client = _apify()
    if client is None:
        return json.dumps({"error": "APIFY_API_KEY not configured"}), 0.0
    query = args.get("query")
    if not query:
        return json.dumps({"error": "query is required"}), 0.0
    limit = int(args.get("limit", 10))
    try:
        actors = await client.search_actors(query=query, limit=limit)
    except Exception as e:
        return json.dumps({"error": f"{type(e).__name__}: {e}"}), 0.0
    return (
        json.dumps({"results": actors, "count": len(actors or [])}, default=str),
        0.0,
    )


async def _apify_actor_details(args: Dict[str, Any], *, project_id: Optional[Any] = None) -> Tuple[str, float]:
    client = _apify()
    if client is None:
        return json.dumps({"error": "APIFY_API_KEY not configured"}), 0.0
    actor_id = args.get("actor_id")
    if not actor_id:
        return json.dumps({"error": "actor_id is required"}), 0.0
    try:
        details = await client.get_actor_details(actor_id)
    except Exception as e:
        return json.dumps({"error": f"{type(e).__name__}: {e}"}), 0.0
    if details is None:
        return json.dumps({"matched": False}), 0.0
    return json.dumps(details, default=str), 0.0


async def _apify_call_actor(args: Dict[str, Any], *, project_id: Optional[Any] = None) -> Tuple[str, float]:
    client = _apify()
    if client is None:
        return json.dumps({"error": "APIFY_API_KEY not configured"}), 0.0
    actor_id = args.get("actor_id")
    if not actor_id:
        return json.dumps({"error": "actor_id is required"}), 0.0
    actor_input = args.get("input") or {}
    # max_items applies a server-side cap when paging the Apify dataset.
    # None = unbounded; the agent can pull large fetches without truncation
    # because we land them in a candidates file, not the LLM context.
    max_items = args.get("max_items")
    if max_items is not None:
        max_items = int(max_items)
    timeout_secs = int(args.get("timeout_secs", 300))
    try:
        run_info = await client.run_actor(
            actor_id=actor_id,
            run_input=actor_input,
            timeout=timeout_secs,
            max_items=max_items,
        )
        if not run_info or run_info.get("status") != "SUCCEEDED":
            return json.dumps({"ok": False, "run_info": run_info}, default=str), 0.0
        items = run_info.get("items", []) or []
    except Exception as e:
        return json.dumps({"error": f"{type(e).__name__}: {e}"}), 0.0

    # Apify reports the run's billed-USD via usageTotalUsd. Pass it through
    # so the chat handler bills the user's credits at COMPUTE_COST_PER_CREDIT.
    cost_usd = float(run_info.get("cost_usd") or 0.0)
    extra = {
        "actor_id": actor_id,
        "status": run_info.get("status"),
        "stats": run_info.get("stats"),
        "usage": run_info.get("usage"),
    }
    result = _persist_candidates(
        project_id, "apify_call_actor", items, cost_usd=cost_usd, extra=extra
    )
    return json.dumps(result, default=str), cost_usd


# ---------------------------------------------------------------------------
# Google Maps tool handlers
# ---------------------------------------------------------------------------


async def _gmaps_search_places(args: Dict[str, Any], *, project_id: Optional[Any] = None) -> Tuple[str, float]:
    client = _gmaps()
    if client is None:
        return json.dumps({"error": "GOOGLE_API_KEY not configured"}), 0.0
    query = args.get("query")
    if not query:
        return json.dumps({"error": "query is required"}), 0.0
    try:
        places = await client.text_search(
            query=query,
            page_token=args.get("page_token") or None,
        )
    except Exception as e:
        return json.dumps({"error": f"{type(e).__name__}: {e}"}), 0.0
    # GoogleMapsClient returns {"results": [...], "next_page_token": ...}
    items = places.get("results", []) or places.get("places", []) or []
    next_page_token = places.get("next_page_token") or places.get("nextPageToken")
    persisted = _persist_candidates(
        project_id,
        "google_maps_search_places",
        items,
        cost_usd=0.032,
        extra={"next_page_token": next_page_token, "query": query},
    )
    return (
        json.dumps(persisted, default=str),
        # Google charges roughly $0.032 per text-search request
        0.032,
    )


async def _gmaps_place_details(args: Dict[str, Any], *, project_id: Optional[Any] = None) -> Tuple[str, float]:
    client = _gmaps()
    if client is None:
        return json.dumps({"error": "GOOGLE_API_KEY not configured"}), 0.0
    place_id = args.get("place_id")
    if not place_id:
        return json.dumps({"error": "place_id is required"}), 0.0
    try:
        details = await client.place_details(place_id)
    except Exception as e:
        return json.dumps({"error": f"{type(e).__name__}: {e}"}), 0.0
    if not details:
        return json.dumps({"matched": False}), 0.0
    return json.dumps(details, default=str), 0.020


# ---------------------------------------------------------------------------
# Sandbox-backed code_exec
# ---------------------------------------------------------------------------


async def _code_exec(args: Dict[str, Any], *, project_id: Optional[Any] = None) -> Tuple[str, float]:
    """Run a Python snippet in the remote sandbox.

    Stateless — fresh session per call. Use for parsing/processing data
    (inline in the snippet, OR loaded from candidate files via the
    `files` arg).

    When `files=[...]` is passed, each candidate file is fetched from
    blob and uploaded into the sandbox before the snippet runs, so the
    snippet can `open(filename)` or json-iter through it just like a
    local file. Use this to filter / aggregate / transform large
    candidate sets without holding them in LLM context.
    """
    code = args.get("code")
    if not code or not isinstance(code, str):
        return json.dumps({"error": "code (string) is required"}), 0.0

    sandbox_url = os.getenv("SANDBOX_SERVICE_URL")
    if not sandbox_url:
        return json.dumps({"error": "SANDBOX_SERVICE_URL not configured"}), 0.0

    timeout = int(args.get("timeout_secs", 60))
    file_names = args.get("files") or []
    if not isinstance(file_names, list):
        return json.dumps({"error": "files must be a list of candidate file names"}), 0.0

    try:
        async with SandboxClient(sandbox_url, timeout=timeout + 30) as pool:
            session = await pool.create_session()
            session_id = session.session_id
            try:
                # Stage requested candidate files into the sandbox workspace.
                for fn in file_names:
                    if not isinstance(fn, str) or not fn:
                        continue
                    if project_id is None:
                        return json.dumps({
                            "error": "files= requires a project context "
                                     "(set when called via the chat agent)"
                        }), 0.0
                    try:
                        blob_bytes = candidates.read_candidates_bytes(project_id, fn)
                    except FileNotFoundError as e:
                        return json.dumps({"error": str(e)}), 0.0
                    await session.upload_content(blob_bytes, fn)

                result = await session.exec_python(code, timeout=timeout)
                return (
                    json.dumps(
                        {
                            "success": result.success,
                            "exit_code": result.exit_code,
                            "stdout": (result.stdout or "")[:8000],
                            "stderr": (result.stderr or "")[:4000],
                            "duration_ms": result.duration_ms,
                            "staged_files": [
                                fn for fn in file_names if isinstance(fn, str) and fn
                            ],
                        },
                        default=str,
                    ),
                    0.0,
                )
            finally:
                try:
                    await pool.destroy_session(session_id)
                except Exception:
                    pass
    except Exception as e:
        return json.dumps({"error": f"{type(e).__name__}: {e}"}), 0.0


# ---------------------------------------------------------------------------
# Browser Use — last-resort cloud browser session
# ---------------------------------------------------------------------------


_bu_client = None  # BUClient lazy singleton


def _bu():
    global _bu_client
    if _bu_client is None:
        key = os.getenv("BROWSER_USE_API_KEY")
        if not key:
            log.warning("BROWSER_USE_API_KEY not set — browser_use disabled")
            return None
        from dsl_worker.infra.bu_client import BUClient
        _bu_client = BUClient(
            api_key=key,
            proxy_country=os.getenv("BROWSER_USE_PROXY_COUNTRY", "us"),
        )
    return _bu_client


async def _browser_use(args: Dict[str, Any], *, project_id: Optional[Any] = None) -> Tuple[str, float]:
    """Run a Browser Use cloud session for a single tightly-scoped task.

    Last resort: only when (a) Apify has no actor for the site AND (b) the
    page needs JS rendering, anti-bot bypass, or interactive flow. Slow
    (~30-180s) and pricey ($0.10–$0.50). Use for ONE specific URL + ONE
    extraction task.
    """
    client = _bu()
    if client is None:
        return json.dumps({"error": "BROWSER_USE_API_KEY not configured"}), 0.0
    task = args.get("task")
    if not task or not isinstance(task, str):
        return json.dumps({"error": "task (string) is required"}), 0.0
    timeout = int(args.get("timeout_secs", 300))
    try:
        items, cost, session_id, summary = await client.extract(
            task=task, timeout=timeout,
        )
    except Exception as e:
        return json.dumps({"error": f"{type(e).__name__}: {e}"}), 0.0
    items = items or []
    persisted = _persist_candidates(
        project_id,
        "browser_use",
        items,
        cost_usd=float(cost or 0),
        extra={
            "summary": (summary or "")[:1000],
            "session_id": session_id,
            "task": task[:200],
        },
    )
    return json.dumps(persisted, default=str), float(cost or 0)


# ---------------------------------------------------------------------------
# web_harvest — bounded research subagent
# ---------------------------------------------------------------------------

# Cost — approximate OpenAI Responses pricing for the subagent's LLM.
# (Mirrors chat.py's _INPUT_COST / _OUTPUT_COST.)
_LLM_INPUT_USD = 0.0000025
_LLM_CACHED_INPUT_USD = 0.00000025
_LLM_OUTPUT_USD = 0.000015
# Cost per OpenAI built-in web_search call (rough; updates over time).
# Mirrors streaming._WEB_SEARCH_USD_PER_CALL — keep in sync.
_WEB_SEARCH_USD_PER_CALL = 0.025


def _response_cost(resp: Any) -> float:
    usage = getattr(resp, "usage", None)
    if not usage:
        return 0.0
    inp = usage.input_tokens or 0
    outp = usage.output_tokens or 0
    cached = 0
    details = getattr(usage, "input_tokens_details", None)
    if details:
        cached = getattr(details, "cached_tokens", 0) or 0
    non_cached = max(0, inp - cached)
    return non_cached * _LLM_INPUT_USD + cached * _LLM_CACHED_INPUT_USD + outp * _LLM_OUTPUT_USD


_subagent_client: Optional[AsyncOpenAI] = None


def _get_subagent_client() -> AsyncOpenAI:
    global _subagent_client
    if _subagent_client is None:
        _subagent_client = AsyncOpenAI(api_key=_api_settings.OPENAI_API_KEY)
    return _subagent_client


_WEB_HARVEST_SYSTEM_PROMPT = """\
You are a web-research subagent. Given a target description, use web_search
to find matching candidates and yield each one with `yield_candidate`.

Rules:
- Real data only. Don't make up entries — every candidate must come from
  search results you actually saw.
- Target ~10-30 candidates depending on target richness, then stop.
- Yield candidates incrementally as you find them — don't batch.
- Each candidate is a JSON object with whatever fields are relevant
  (name, url, location, description — whatever the target asks for).
- Vary search queries to get coverage; don't keep firing the same one.
- If you've done 3-4 search rounds and aren't finding new candidates,
  stop.
"""


async def _web_harvest(args: Dict[str, Any], *, project_id: Optional[Any] = None) -> Tuple[str, float]:
    """Spawn a small bounded web-research agent. Returns yielded
    candidates inline.

    Use sparingly — slower and pricier than a direct API source. Only
    when no Apify actor or structured source matches the target.
    """
    query = args.get("query")
    if not query:
        return json.dumps({"error": "query is required"}), 0.0
    description = (
        args.get("candidate_description")
        or args.get("find")
        or query
    )
    max_candidates = int(args.get("max_candidates", 25))
    max_turns = int(args.get("max_turns", 6))

    client = _get_subagent_client()

    harvested: List[Dict[str, Any]] = []
    total_cost = 0.0
    web_search_calls = 0
    previous_response_id: Optional[str] = None

    initial_user_msg = (
        f"Target: {description}\n\nQuery hint: {query}\n\n"
        f"Find up to {max_candidates} candidates."
    )
    next_input: Any = [{"role": "user", "content": initial_user_msg}]

    tools = [
        {"type": "web_search", "search_context_size": "medium"},
        {
            "type": "function",
            "name": "yield_candidate",
            "description": "Save one candidate. Pass `data` as an object with whatever fields are relevant.",
            "parameters": {
                "type": "object",
                "properties": {
                    "data": {
                        "type": "object",
                        "description": "Candidate fields (name, url, location, etc.)",
                        "additionalProperties": True,
                    },
                },
                "required": ["data"],
            },
        },
    ]

    for turn_idx in range(max_turns):
        try:
            kwargs: Dict[str, Any] = {
                "model": _api_settings.OPENAI_MODEL,
                "input": next_input,
                "tools": tools,
                "max_output_tokens": 4000,
            }
            if turn_idx == 0:
                kwargs["instructions"] = _WEB_HARVEST_SYSTEM_PROMPT
            else:
                kwargs["previous_response_id"] = previous_response_id
            resp = await client.responses.create(**kwargs)
        except Exception as e:
            return (
                json.dumps({"error": f"{type(e).__name__}: {e}", "candidates": harvested}),
                total_cost,
            )

        total_cost += _response_cost(resp)
        previous_response_id = resp.id

        function_calls: List[Any] = []
        for item in resp.output:
            if item.type == "web_search_call":
                web_search_calls += 1
            elif item.type == "function_call":
                function_calls.append(item)

        if not function_calls:
            break

        tool_outputs: List[Dict[str, Any]] = []
        for fc in function_calls:
            try:
                fc_args = json.loads(fc.arguments) if fc.arguments else {}
            except json.JSONDecodeError:
                tool_outputs.append({
                    "type": "function_call_output",
                    "call_id": fc.call_id,
                    "output": "Error: invalid arguments",
                })
                continue
            if fc.name == "yield_candidate":
                data = fc_args.get("data") or {}
                if isinstance(data, dict) and data:
                    harvested.append(data)
                    tool_outputs.append({
                        "type": "function_call_output",
                        "call_id": fc.call_id,
                        "output": f"Saved #{len(harvested)}",
                    })
                else:
                    tool_outputs.append({
                        "type": "function_call_output",
                        "call_id": fc.call_id,
                        "output": "Error: data must be a non-empty object",
                    })
            else:
                tool_outputs.append({
                    "type": "function_call_output",
                    "call_id": fc.call_id,
                    "output": f"unknown tool: {fc.name}",
                })

        if len(harvested) >= max_candidates:
            break

        next_input = tool_outputs

    total_cost += web_search_calls * _WEB_SEARCH_USD_PER_CALL

    persisted = _persist_candidates(
        project_id,
        "web_harvest",
        harvested,
        cost_usd=total_cost,
        extra={
            "query": query,
            "candidate_description": description,
            "web_search_calls": web_search_calls,
            "turns_used": turn_idx + 1,
        },
    )
    return json.dumps(persisted, default=str), total_cost


# ---------------------------------------------------------------------------
# Tool schemas (OpenAI Responses API format)
# ---------------------------------------------------------------------------


SOURCE_TOOLS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "name": "fullenrich_search_people",
        "description": (
            "Search FullEnrich's LinkedIn-derived people database. Best for "
            "finding professional / B2B / tech / knowledge-work targets "
            "('founders of B2B SaaS', 'engineers at Y', 'marketing leads in "
            "fintech'). Returns inline results with all fields. Use the "
            "result with rows_add to commit selected people to the table. "
            "Cost is fractional credits per result (~$0.001 each)."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "titles": {"type": "array", "items": {"type": "string"}, "description": "e.g. ['Founder', 'Co-Founder', 'CEO']"},
                "seniority": {"type": "array", "items": {"type": "string"}, "description": "e.g. ['Owner', 'C-Level']"},
                "industries": {"type": "array", "items": {"type": "string"}},
                "locations": {"type": "array", "items": {"type": "string"}},
                "skills": {"type": "array", "items": {"type": "string"}},
                "universities": {"type": "array", "items": {"type": "string"}},
                "company_names": {"type": "array", "items": {"type": "string"}},
                "company_domains": {"type": "array", "items": {"type": "string"}},
                "company_specialties": {"type": "array", "items": {"type": "string"}},
                "company_types": {"type": "array", "items": {"type": "string"}},
                "company_hq": {"type": "array", "items": {"type": "string"}},
                "names": {"type": "array", "items": {"type": "string"}},
                "linkedin_urls": {"type": "array", "items": {"type": "string"}},
                "past_companies": {"type": "array", "items": {"type": "string"}},
                "past_titles": {"type": "array", "items": {"type": "string"}},
                "company_headcount_min": {"type": "integer"},
                "company_headcount_max": {"type": "integer"},
                "company_founded_min": {"type": "integer"},
                "company_founded_max": {"type": "integer"},
                "limit": {"type": "integer", "description": "Max results to return (default 25). Start small to validate filters."},
                "offset": {"type": "integer", "description": "Pagination offset. Use the next_offset from a prior call to continue."},
            },
        },
    },
    {
        "type": "function",
        "name": "fullenrich_search_companies",
        "description": (
            "Search FullEnrich's company database. Filter by industry, "
            "location, size, specialties, keywords, etc. Pair with "
            "fullenrich_search_people(company_names=[...]) to find people "
            "at the matched companies. Cost is fractional credits per result."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "industries": {"type": "array", "items": {"type": "string"}},
                "locations": {"type": "array", "items": {"type": "string"}},
                "specialties": {"type": "array", "items": {"type": "string"}},
                "keywords": {"type": "array", "items": {"type": "string"}},
                "names": {"type": "array", "items": {"type": "string"}},
                "domains": {"type": "array", "items": {"type": "string"}},
                "types": {"type": "array", "items": {"type": "string"}},
                "headcount_min": {"type": "integer"},
                "headcount_max": {"type": "integer"},
                "founded_min": {"type": "integer"},
                "founded_max": {"type": "integer"},
                "limit": {"type": "integer"},
                "offset": {"type": "integer"},
            },
        },
    },
    {
        "type": "function",
        "name": "fullenrich_enrich_contacts",
        "description": (
            "Find verified work emails (and/or phones) for a small batch of "
            "people. Each contact needs at minimum: linkedin_url OR "
            "(first_name + last_name + domain). Cost: ~1 credit per email "
            "found, ~10 credits per phone found. ≤25 contacts per call — "
            "batch larger lists across multiple calls."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "contacts": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "first_name": {"type": "string"},
                            "last_name": {"type": "string"},
                            "domain": {"type": "string"},
                            "company_name": {"type": "string"},
                            "linkedin_url": {"type": "string"},
                        },
                    },
                },
                "fields": {
                    "type": "array",
                    "items": {"type": "string", "enum": ["emails", "phones", "personal_emails"]},
                    "description": "Default: ['emails', 'phones'].",
                },
            },
            "required": ["contacts"],
        },
    },
    # ── Apollo ───────────────────────────────────────────────────────────
    {
        "type": "function",
        "name": "apollo_search_companies",
        "description": (
            "Search Apollo's company DB. Filters: keywords, name, locations, "
            "employee_ranges (e.g. ['1,10','11,50']), revenue_ranges, "
            "domains, industries (Apollo industry tag IDs), technologies. "
            "Returns simplified company records inline. Free per result."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "keywords": {"type": "array", "items": {"type": "string"}},
                "name": {"type": "string"},
                "locations": {"type": "array", "items": {"type": "string"}},
                "not_locations": {"type": "array", "items": {"type": "string"}},
                "employee_ranges": {"type": "array", "items": {"type": "string"}, "description": "e.g. ['1,10', '11,50']"},
                "revenue_ranges": {"type": "array", "items": {"type": "string"}},
                "domains": {"type": "array", "items": {"type": "string"}},
                "industries": {"type": "array", "items": {"type": "string"}, "description": "Apollo industry tag IDs"},
                "technologies": {"type": "array", "items": {"type": "string"}},
                "page": {"type": "integer"},
                "per_page": {"type": "integer"},
            },
        },
    },
    {
        "type": "function",
        "name": "apollo_enrich_person",
        "description": (
            "Look up a single person's contact info in Apollo. Provide one or "
            "more identifiers: linkedin_url, email, name+company, or "
            "first_name+last_name+domain. Costs 1 credit (~$0.024) per match."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "apollo_id": {"type": "string"},
                "first_name": {"type": "string"},
                "last_name": {"type": "string"},
                "name": {"type": "string"},
                "email": {"type": "string"},
                "company": {"type": "string"},
                "domain": {"type": "string"},
                "linkedin_url": {"type": "string"},
            },
        },
    },
    {
        "type": "function",
        "name": "apollo_enrich_company",
        "description": "Look up a single company in Apollo by domain or name. Free.",
        "parameters": {
            "type": "object",
            "properties": {
                "domain": {"type": "string"},
                "name": {"type": "string"},
            },
        },
    },
    # ── Apify ────────────────────────────────────────────────────────────
    {
        "type": "function",
        "name": "apify_search_actors",
        "description": (
            "Search Apify's marketplace for pre-built scrapers. Use this to "
            "find an actor for a specific site (e.g. 'reddit posts', "
            "'linkedin company', 'zillow listings') before you reach for "
            "web_harvest. Returns up to N actor records inline."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "limit": {"type": "integer", "minimum": 1, "maximum": 25},
            },
            "required": ["query"],
        },
    },
    {
        "type": "function",
        "name": "apify_actor_details",
        "description": (
            "Fetch full details for an Apify actor (input schema, pricing, "
            "stats, README). Call this after apify_search_actors to figure "
            "out how to invoke a specific actor."
        ),
        "parameters": {
            "type": "object",
            "properties": {"actor_id": {"type": "string"}},
            "required": ["actor_id"],
        },
    },
    {
        "type": "function",
        "name": "apify_call_actor",
        "description": (
            "Run an Apify actor and write results to a candidates file. "
            "Use apify_actor_details first to understand the actor's input "
            "schema. Returns the candidates_file path, items_count, fields, "
            "and a small preview — NOT the full result set. To work with "
            "items use candidates_inspect / candidates_to_rows / code_exec. "
            "Apify run cost is billed separately on our account."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "actor_id": {"type": "string"},
                "input": {"type": "object", "description": "Actor-specific input. Check apify_actor_details for the schema."},
                "max_items": {"type": "integer", "minimum": 1, "description": "Cap items returned by the actor. Omit for unbounded."},
                "timeout_secs": {"type": "integer", "description": "Default 300."},
            },
            "required": ["actor_id"],
        },
    },
    # ── Google Maps ──────────────────────────────────────────────────────
    {
        "type": "function",
        "name": "google_maps_search_places",
        "description": (
            "Google Places text search. Best for local businesses ("
            "restaurants, dentists, schools, churches, anything with a "
            "physical address). Bake the location into the `query` string "
            "directly (e.g. 'pizza near Brooklyn NY'). Returns up to 20 "
            "places per page; use next_page_token to paginate. ~$0.032 per "
            "request to us."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "e.g. 'pizza near Brooklyn NY' or 'dentists in Austin'"},
                "page_token": {"type": "string"},
            },
            "required": ["query"],
        },
    },
    {
        "type": "function",
        "name": "google_maps_place_details",
        "description": "Fetch full details for a Google Place by place_id.",
        "parameters": {
            "type": "object",
            "properties": {"place_id": {"type": "string"}},
            "required": ["place_id"],
        },
    },
    # ── Last-resort cloud browser ────────────────────────────────────────
    {
        "type": "function",
        "name": "browser_use",
        "description": (
            "Open a Browser Use cloud browser to extract data from ONE URL "
            "with ONE specific task. LAST RESORT: only when Apify has no "
            "actor and the page needs JS rendering / anti-bot / interactive "
            "navigation. Slow (~30-180s) and pricey ($0.10-0.50). Always "
            "tightly scope: the task should be one URL + one extraction "
            "goal."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "task": {
                    "type": "string",
                    "description": "ONE URL + ONE extraction task. e.g. 'Go to https://example.com/listings and extract names + phones from the table.'",
                },
                "timeout_secs": {"type": "integer", "minimum": 30, "maximum": 600},
            },
            "required": ["task"],
        },
    },
    # ── Last-resort web research ─────────────────────────────────────────
    {
        "type": "function",
        "name": "web_harvest",
        "description": (
            "Spawn a small bounded research subagent that uses web_search "
            "to find candidates and yields them as structured JSON. Slower "
            "and pricier than direct API sources — use ONLY when there's "
            "no Apify actor for the target site and no structured source "
            "(FE / Apollo / Google Maps) covers the target. Good for "
            "discovery tasks like 'find indie hackers building X', "
            "'startups in this niche cohort', etc. Returns up to "
            "max_candidates inline."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query / target hint."},
                "candidate_description": {
                    "type": "string",
                    "description": "What you want each candidate to look like — fields, criteria, anything specific.",
                },
                "max_candidates": {"type": "integer", "minimum": 1, "maximum": 50},
                "max_turns": {"type": "integer", "minimum": 2, "maximum": 10},
            },
            "required": ["query"],
        },
    },
    # ── Sandbox ──────────────────────────────────────────────────────────
    {
        "type": "function",
        "name": "code_exec",
        "description": (
            "Run a Python snippet in a remote sandbox. Stateless — each call "
            "gets a fresh session. Use for parsing/processing data (inline "
            "or loaded from candidate files), mapping nested JSON to flat "
            "dicts, pure algorithmic Python, or quick calculations. The "
            "sandbox has the Python stdlib + httpx + json + re. It does NOT "
            "have access to the project DB; if you want to commit results, "
            "print them to stdout, then call rows_add or candidates_to_rows "
            "separately. Default timeout 60s.\n\n"
            "Pass `files=['name.jsonl', ...]` to stage candidate files from "
            "blob into the sandbox workspace before running. Inside the "
            "snippet they appear as local files: `for line in open('name.jsonl')` "
            "or `[json.loads(l) for l in open('name.jsonl')]`. Use this to "
            "filter, transform, or aggregate large fetches without holding "
            "them in LLM context."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "code": {"type": "string", "description": "Python source to execute."},
                "files": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Candidate file names (from candidates_list) to stage into the sandbox before running. Each becomes openable as a local file by that name.",
                },
                "timeout_secs": {"type": "integer", "minimum": 1, "maximum": 300},
            },
            "required": ["code"],
        },
    },
]


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------


_HandlerType = Callable[..., Awaitable[Tuple[str, float]]]

_HANDLERS: Dict[str, _HandlerType] = {
    "fullenrich_search_people": _fe_search_people,
    "fullenrich_search_companies": _fe_search_companies,
    "fullenrich_enrich_contacts": _fe_enrich_contacts,
    "apollo_search_companies": _apollo_search_companies,
    "apollo_enrich_person": _apollo_enrich_person,
    "apollo_enrich_company": _apollo_enrich_company,
    "apify_search_actors": _apify_search_actors,
    "apify_actor_details": _apify_actor_details,
    "apify_call_actor": _apify_call_actor,
    "google_maps_search_places": _gmaps_search_places,
    "google_maps_place_details": _gmaps_place_details,
    "code_exec": _code_exec,
    "web_harvest": _web_harvest,
    "browser_use": _browser_use,
}


def is_source_tool(name: str) -> bool:
    return name in _HANDLERS


async def execute_source_tool(
    name: str,
    args: Dict[str, Any],
    *,
    project_id: Optional[Any] = None,
) -> Tuple[str, float]:
    """Run a source tool by name. Returns (result_json_text, cost_usd).

    Cost is in USD (raw provider cost to us). The chat handler converts
    to user-facing credits at COMPUTE_COST_PER_CREDIT.

    project_id is threaded to handlers that write candidates files. Tools
    that don't need it ignore the kwarg.
    """
    handler = _HANDLERS.get(name)
    if handler is None:
        return json.dumps({"error": f"unknown source tool: {name}"}), 0.0
    try:
        return await handler(args, project_id=project_id)
    except Exception as e:
        log.exception("source tool %s failed", name)
        return json.dumps({"error": f"{type(e).__name__}: {e}"}), 0.0
