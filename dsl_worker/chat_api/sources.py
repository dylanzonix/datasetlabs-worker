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
import re
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

    # Default emails-only. Phones cost 10 credits each on FE — when both
    # email and phone enrichment are enabled and a phone is found, the
    # caller pays ~10x what they would for emails alone. Most fills want
    # email; phones are an explicit opt-in.
    fields = args.get("fields") or ["emails"]
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
        enrich_fields = ["contact.emails"]

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
    if not domain:
        return json.dumps({"error": "provide domain (call apollo_search_companies if you only have a name)"}), 0.0
    try:
        org = await client.enrich_company(domain=domain)
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

    # Heuristic warning against the over-perfection failure mode: if the
    # agent already called apify_call_actor on the SAME actor in this
    # run AND hasn't committed any rows since, attach a warning to the
    # result so the agent gets reminded. We let the call proceed (vs
    # refusing) so legit edge cases — agent has good reason to refetch,
    # rare patterns we haven't anticipated — aren't broken. Different
    # actors don't trigger the warning (multi-source harvests are fine).
    over_iteration_warning: Optional[str] = None
    if project_id is not None:
        try:
            from dsl_api.db import SessionLocal
            from dsl_api.models import ChatRun, ChatRunEvent
            from sqlalchemy import desc
            db = SessionLocal()
            try:
                run = (
                    db.query(ChatRun)
                    .filter(
                        ChatRun.project_id == project_id,
                        ChatRun.status == "running",
                    )
                    .order_by(desc(ChatRun.started_at))
                    .first()
                )
                if run is not None:
                    events = (
                        db.query(ChatRunEvent)
                        .filter(ChatRunEvent.run_id == run.id)
                        .order_by(desc(ChatRunEvent.seq))
                        .limit(500)
                        .all()
                    )
                    rows_added_since = 0
                    prior_same_actor_items: Optional[int] = None
                    prior_same_actor_errored = False
                    same_actor_call_count = 0
                    for e in events:
                        if e.type == "row_added":
                            rows_added_since += 1
                            continue
                        if e.type == "tool_result":
                            p = e.payload or {}
                            if p.get("name") == "apify_call_actor":
                                call_id = p.get("id")
                                # Search BOTH summary and result_text — summary
                                # is structured bits ("returned=40, ok") and
                                # rarely contains actor_id; result_text is the
                                # full JSON dict (truncated to 4000 chars but
                                # actor_id is lifted to the top of the dict
                                # so it survives).
                                searchable = (p.get("summary") or "") + "\n" + (p.get("result_text") or "")
                                m_actor = re.search(r'"actor_id":\s*"([^"]+)"', searchable)
                                prior_actor_id = m_actor.group(1) if m_actor else None
                                if prior_actor_id is None and call_id:
                                    matching_call = (
                                        db.query(ChatRunEvent)
                                        .filter(
                                            ChatRunEvent.run_id == run.id,
                                            ChatRunEvent.type == "tool_call",
                                        )
                                        .filter(ChatRunEvent.payload["id"].astext == str(call_id))
                                        .first()
                                    )
                                    if matching_call:
                                        prior_args = (matching_call.payload or {}).get("args_full") or {}
                                        prior_actor_id = prior_args.get("actor_id") if isinstance(prior_args, dict) else None
                                if prior_actor_id == actor_id:
                                    same_actor_call_count += 1
                                    m = re.search(r'"items_count":\s*(\d+)', searchable)
                                    if prior_same_actor_items is None:
                                        prior_same_actor_items = int(m.group(1)) if m else None
                                        prior_same_actor_errored = '"ok": false' in searchable or '"error"' in searchable
                    if (
                        prior_same_actor_items is not None
                        and prior_same_actor_items > 0
                        and not prior_same_actor_errored
                        and rows_added_since == 0
                    ):
                        over_iteration_warning = (
                            f"You already called this actor ('{actor_id}') "
                            f"{same_actor_call_count}× in this turn and got "
                            f"candidates back (most recent: {prior_same_actor_items} "
                            "items), but no rows have been committed yet. This is "
                            "the over-perfection pattern — fetching more without "
                            "committing the previous batch. Strongly consider "
                            "committing what you already have (candidates_to_rows "
                            "or code_exec with add_rows, filter inline if needed) "
                            "before refining the keywords further. Calling a "
                            "different actor is fine, but refining the same actor's "
                            "input over and over rarely yields better results than "
                            "committing the first batch and letting the user steer "
                            "from there."
                        )
            finally:
                db.close()
        except Exception:
            log.exception("apify_call_actor over-iteration check failed; proceeding")

    # Validate input BEFORE running. Mirrors apify-mcp-server's behavior:
    # an empty or invalid input typically means the actor falls back to
    # the publisher's placeholder defaults (e.g. query="helloworld" or
    # the publisher's username) and returns garbage that confuses the
    # agent into bailing to web_search. Catch this up front and return
    # the schema so the model can self-correct.
    actor_input = args.get("input")
    details = None
    try:
        details = await client.get_actor_details(actor_id)
    except Exception as e:
        log.warning("apify_call_actor: get_actor_details failed: %s", e)
    input_schema = (details or {}).get("input_schema") if details else None

    if not actor_input or not isinstance(actor_input, dict):
        return json.dumps({
            "error": "input is required and must be a non-empty object",
            "hint": (
                "Build `input` from the actor's input_schema.properties — "
                "required fields are listed in input_schema.required. "
                "Calling without input makes the actor fall back to the "
                "publisher's placeholder defaults, which return garbage."
            ),
            "input_schema": input_schema,
        }), 0.0

    if input_schema and isinstance(input_schema, dict):
        try:
            import jsonschema
            jsonschema.validate(instance=actor_input, schema=input_schema)
        except jsonschema.ValidationError as ve:
            return json.dumps({
                "error": f"input failed validation against actor's input_schema: {ve.message}",
                "validation_path": list(ve.absolute_path),
                "input_schema": input_schema,
                "your_input": actor_input,
                "hint": (
                    "Fix `input` to match input_schema.properties (types, "
                    "enums, required fields) and call again."
                ),
            }), 0.0
        except Exception:
            # If validation itself errors (broken schema, jsonschema bug,
            # etc.), don't block the call — let Apify reject it instead.
            log.exception("apify_call_actor: schema validation crashed; proceeding")

    # max_items applies a server-side cap when paging the Apify dataset.
    # None = unbounded normally; the agent can pull large fetches because
    # we land them in a candidates file, not the LLM context.
    max_items = args.get("max_items")
    if max_items is not None:
        max_items = int(max_items)
    timeout_secs = int(args.get("timeout_secs", 600))

    # Small-first cap: when the project has 0 rows committed, force a
    # tight cap on the first apify_call_actor batch. Prevents the
    # over-perfection failure mode where the agent fetches 500 candidates
    # chasing "scope coverage" and never commits anything (saw it in runs
    # 88328425 and 3fc103bc — 5 calls, 180+ candidates, 0 rows landed).
    SMALL_FIRST_CAP = 50
    small_first_clamped = False
    if project_id is not None:
        try:
            from dsl_api.db import SessionLocal
            from dsl_api.models.sample import Sample
            from sqlalchemy import func
            db = SessionLocal()
            try:
                n_rows = db.query(func.count(Sample.id)).filter(
                    Sample.project_id == project_id,
                    Sample.deleted_at.is_(None),
                ).scalar() or 0
            finally:
                db.close()
        except Exception:
            n_rows = 0
        if n_rows == 0:
            if max_items is None or max_items > SMALL_FIRST_CAP:
                max_items = SMALL_FIRST_CAP
                small_first_clamped = True
            # Strip the input-level cap too — actors honor their own
            # input field, not our download-side max_items.
            for key in ("maxItems", "max_items", "max_results", "maxResults", "limit"):
                if key in actor_input and (
                    not isinstance(actor_input[key], int)
                    or actor_input[key] > SMALL_FIRST_CAP
                ):
                    actor_input[key] = SMALL_FIRST_CAP
                    small_first_clamped = True

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
    raw_result = _persist_candidates(
        project_id, "apify_call_actor", items, cost_usd=cost_usd, extra=extra
    )
    # Lift actor_id to the TOP of the result so it survives the 4000-char
    # truncation in agent.format_tool_result and is grep-able by the
    # over-iteration check on the next call. Python dicts preserve insertion
    # order, so {actor_id: ..., **raw_result} puts it first.
    result = {"actor_id": actor_id, **raw_result}
    if small_first_clamped:
        # Stamp at top level so the agent reads it before run_metadata
        # gets any chance to be skipped.
        result["small_first_clamped"] = True
        result["next_step_hint"] = (
            f"FIRST BATCH WAS CAPPED AT {SMALL_FIRST_CAP} (project had 0 "
            "rows). Your next move is to COMMIT these candidates as rows "
            "(candidates_to_rows or code_exec with add_rows). Filter "
            "inline as you commit if needed. Do NOT call apify_call_actor "
            "again before committing — that's the over-perfection failure "
            "mode the cap exists to prevent. The user expects to see rows "
            "land in the table now and refine via follow-up turns; they "
            "don't expect you to scan the entire search space first."
        )
    if over_iteration_warning:
        result["over_iteration_warning"] = over_iteration_warning
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


def _uploaded_file_urls(project_id: Optional[Any]) -> Dict[str, str]:
    """Generate short-lived SAS read-URLs for every uploaded file on
    this project. Returns ``{filename: sas_url}`` (possibly empty).

    Mirrors the v13 pipeline's `_generate_file_urls` (job_processor.py)
    so chat-mode runs see the same set of user uploads at the sandbox's
    `/workspace/uploads/<filename>` path.
    """
    if project_id is None:
        return {}
    from dsl_api.azure.blob import create_download_url
    from dsl_api.db import SessionLocal
    from dsl_api.models.project_file import ProjectFile

    db = SessionLocal()
    try:
        rows = (
            db.query(ProjectFile)
            .filter(
                ProjectFile.project_id == project_id,
                ProjectFile.deleted_at.is_(None),
                ProjectFile.status == "uploaded",
            )
            .all()
        )
    finally:
        db.close()

    urls: Dict[str, str] = {}
    for f in rows:
        if not f.filename or not f.blob_path:
            continue
        try:
            grant = create_download_url(f.blob_path)
            urls[f.filename] = grant.upload_url
        except Exception as e:
            log.warning("Failed to generate SAS URL for %s: %s", f.filename, e)
    return urls


async def _code_exec(args: Dict[str, Any], *, project_id: Optional[Any] = None) -> Tuple[str, float]:
    """Run a Python snippet in the remote sandbox.

    Stateless — fresh session per call. User-uploaded files are
    auto-staged at `/workspace/uploads/<filename>`. Optional
    `files=[...]` additionally stages named candidate files at the
    workspace root.

    The sandbox is OFFLINE (no network). To mutate the project from
    Python, the snippet uses `dsl_tools` helpers (`add_rows`,
    `add_columns`, etc.) which append intent records to
    `/workspace/_dsl_ops.jsonl`. After exec, this function reads that
    file and returns the ops embedded in the result; `agent.execute_tool`
    applies them through the canonical chat-mode handlers and replaces
    this payload with a small LLM-facing envelope.
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

    upload_urls = _uploaded_file_urls(project_id)

    try:
        async with SandboxClient(sandbox_url, timeout=timeout + 30) as pool:
            session = await pool.create_session()
            session_id = session.session_id
            try:
                # Inject the dsl_tools module so the snippet can do
                # `from dsl_tools import add_rows, add_columns, ...` —
                # the bulk-write primitives that make data manipulation
                # in code_exec viable. Without this upload, every
                # snippet trying to use dsl_tools hits ModuleNotFoundError
                # and falls back to pasting rows through tokens.
                # (v13 does this in SandboxSession.upload_workspace; the
                # chat-api path uses the raw session client and has to
                # do it explicitly.)
                try:
                    from dsl_worker.infra.dsl_tools_module import DSL_TOOLS_SOURCE
                    await session.upload_content(DSL_TOOLS_SOURCE, "dsl_tools.py")
                except Exception as e:
                    log.warning("Failed to upload dsl_tools.py to sandbox: %s", e)

                # Auto-stage user uploads into /workspace/uploads/.
                staged_uploads: List[str] = []
                for upload_name, sas_url in upload_urls.items():
                    try:
                        await session.fetch_from_url(
                            sas_url, f"uploads/{upload_name}"
                        )
                        staged_uploads.append(upload_name)
                    except Exception as e:
                        log.warning(
                            "Failed to stage upload %s into sandbox: %s",
                            upload_name, e,
                        )

                # Stage requested candidate files at the workspace root.
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

                # Drain `/workspace/_dsl_ops.jsonl` if the snippet
                # emitted any project-mutation ops via dsl_tools. Each
                # line is a JSON op dict ({"op": "add_rows", ...}). The
                # sandbox is offline; ops haven't been applied yet —
                # we hand them to execute_tool() which holds the DB
                # context and applies via _tool_* handlers.
                ops: List[Dict[str, Any]] = []
                ops_read_error: Optional[str] = None
                try:
                    raw_ops = await session.read_file("_dsl_ops.jsonl")
                except Exception:
                    raw_ops = ""  # no ops emitted is the common case
                if raw_ops:
                    if len(raw_ops) > 25 * 1024 * 1024:
                        ops_read_error = (
                            f"ops log {len(raw_ops)} bytes > 25MB cap; "
                            "split work across multiple code_exec calls"
                        )
                    else:
                        for ln in raw_ops.splitlines():
                            ln = ln.strip()
                            if not ln:
                                continue
                            try:
                                op = json.loads(ln)
                            except Exception:
                                continue
                            if isinstance(op, dict) and op.get("op"):
                                ops.append(op)

                payload: Dict[str, Any] = {
                    "success": result.success,
                    "exit_code": result.exit_code,
                    "stdout": (result.stdout or "")[:8000],
                    "stderr": (result.stderr or "")[:4000],
                    "duration_ms": result.duration_ms,
                    "staged_files": [
                        fn for fn in file_names if isinstance(fn, str) and fn
                    ],
                    "staged_uploads": staged_uploads,
                    # `_pending_ops` is consumed by agent.execute_tool —
                    # never goes back to the LLM directly. The dispatcher
                    # applies them and replaces this whole payload with
                    # a small envelope before serialization.
                    "_pending_ops": ops,
                }
                if ops_read_error:
                    payload["_pending_ops_error"] = ops_read_error
                return (json.dumps(payload, default=str), 0.0)
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

# Per-model token rates. Keys are model name prefixes; longest match wins
# so "gpt-5.4-mini" resolves before "gpt-5.4". Unknown models fall back
# to the full-model rates — over-reporting cost is safer than
# under-reporting (we'd rather display a slightly high number than
# silently bill the user more than they were told).
#
# Rates in USD per token. Update when OpenAI changes pricing.
_MODEL_RATES: Dict[str, Dict[str, float]] = {
    # gpt-5.4 (full): $2.50 input / $0.25 cached / $15.00 output per 1M
    "gpt-5.4": {
        "input":  0.0000025,
        "cached": 0.00000025,
        "output": 0.000015,
    },
    # gpt-5.4-mini: $0.75 input / $0.075 cached / $4.50 output per 1M
    "gpt-5.4-mini": {
        "input":  0.00000075,
        "cached": 0.000000075,
        "output": 0.0000045,
    },
}

# Backwards-compat aliases — kept so any external code importing these
# constants directly still works. The canonical path is `_response_cost`
# with a `model=` kwarg.
_LLM_INPUT_USD = _MODEL_RATES["gpt-5.4"]["input"]
_LLM_CACHED_INPUT_USD = _MODEL_RATES["gpt-5.4"]["cached"]
_LLM_OUTPUT_USD = _MODEL_RATES["gpt-5.4"]["output"]

# Per-call rates for OpenAI's built-in web_search, by search_context_size.
# OpenAI bills at ~$25/$30/$50 per 1K calls for low/medium/high. Update if
# OpenAI changes their pricing.
WEB_SEARCH_USD_BY_TIER = {
    "low": 0.025,
    "medium": 0.030,
    "high": 0.050,
}


def _resolve_rates(model: Optional[str]) -> Dict[str, float]:
    """Pick the rate table for a model name. Longest-prefix match.

    Falls back to full-model rates when the model name is unknown so
    cost is over-reported rather than under-reported. We log a warning
    once per unknown model so the rate table can be kept current.
    """
    if not model:
        return _MODEL_RATES["gpt-5.4"]
    if model in _MODEL_RATES:
        return _MODEL_RATES[model]
    for key in sorted(_MODEL_RATES.keys(), key=len, reverse=True):
        if model.startswith(key):
            return _MODEL_RATES[key]
    log.warning(
        "sources._resolve_rates: unknown model %r — falling back to "
        "gpt-5.4 rates. Add an entry to _MODEL_RATES.",
        model,
    )
    return _MODEL_RATES["gpt-5.4"]


def _response_cost(resp: Any, model: Optional[str] = None) -> float:
    """Compute the dollar cost of a single Responses API response.

    Pass `model` so per-model rates are applied. The cell agent calls
    gpt-5.4-mini; the chat agent calls gpt-5.4 (full). Without `model`,
    full-model rates are used as a safe fallback — that overstates
    mini-generated costs and was the source of the "cell costs ~1
    credit" bug (mini tokens were being priced at full rates).
    """
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
    rates = _resolve_rates(model)
    return (
        non_cached * rates["input"]
        + cached * rates["cached"]
        + outp * rates["output"]
    )


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

        total_cost += _response_cost(resp, model=_api_settings.OPENAI_MODEL)
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

    # web_harvest uses search_context_size=medium (line ~814), so bill at
    # the medium rate, not the low default.
    total_cost += web_search_calls * WEB_SEARCH_USD_BY_TIER["medium"]

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
            "Find verified work emails for a small batch of people. Each "
            "contact needs at minimum: linkedin_url OR (first_name + "
            "last_name + domain). Adding linkedin_url improves the match "
            "rate significantly — pass it when you have one. ≤25 contacts "
            "per call.\n\n"
            "Cost: ~1 credit per verified email (~$0.055). Defaults to "
            "EMAILS ONLY. Phones cost ~10 credits each (~$0.55) — only "
            "pass `fields=['emails','phones']` when the target column is "
            "a phone column. Don't enable phones 'just in case' — it "
            "silently costs 10x."
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
                    "description": (
                        "Defaults to ['emails']. Add 'phones' ONLY if "
                        "the target is a phone column — phone enrichment "
                        "is ~10x the cost of email enrichment."
                    ),
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
        "description": (
            "Look up a single company in Apollo by domain. Free. "
            "If you only have a name, call apollo_search_companies first to get the domain."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "domain": {"type": "string"},
            },
            "required": ["domain"],
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
            "Run an Apify actor and write results to a candidates file.\n\n"
            "WORKFLOW:\n"
            "1. apify_search_actors → find a relevant actor.\n"
            "2. apify_actor_details(actor_id) → read the actor's input_schema.\n"
            "3. apify_call_actor(actor_id, input=<object matching the schema>).\n\n"
            "The `input` argument is REQUIRED and must contain real values for "
            "the actor's filters / queries / URLs / etc — DO NOT call this with "
            "an empty input or only actor_id. Many actors silently fall back to "
            "their author's placeholder defaults (e.g. query='helloworld') and "
            "return garbage results. Construct `input` from input_schema."
            "properties — required fields are listed in input_schema.required."
            "\n\n"
            "Returns the candidates_file path, items_count, fields, and a small "
            "preview — NOT the full result set. To work with items use "
            "candidates_inspect / candidates_to_rows / code_exec. Apify run "
            "cost is billed separately on our account."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "actor_id": {"type": "string"},
                "input": {
                    "type": "object",
                    "description": (
                        "Actor-specific input object matching the actor's "
                        "input_schema. REQUIRED — never omit or pass {}. "
                        "Get the schema from apify_actor_details first."
                    ),
                },
                "max_items": {
                    "type": "integer",
                    "minimum": 1,
                    "description": (
                        "Hard cap on items downloaded. **ALWAYS pass this.** "
                        "First call on a new query: 10–25. Going higher "
                        "requires the user to have given an explicit count "
                        "('get me 100 X'). Server enforces a max of 50 on "
                        "first calls when the project has zero rows — "
                        "fetching more before committing anything is the "
                        "over-perfection failure mode. Refine via follow-up "
                        "turns, not bigger first fetches."
                    ),
                },
                "timeout_secs": {
                    "type": "integer",
                    "description": (
                        "How long to wait for the actor to finish (default "
                        "600). On timeout we abort the Apify run so you "
                        "stop getting billed for it."
                    ),
                },
            },
            "required": ["actor_id", "input"],
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


def derive_default_source(
    tool: str, item: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    """Default cell-source citation for a candidate item from a source tool.

    candidates_to_rows uses this to auto-attach `_sources` so rows committed
    via bulk mapping carry the same provenance the agent would have added
    via `_sources` on `rows_add`. Returns `{"type": "url", "value": ...}` or
    None when no useful source URL can be derived.
    """
    if not isinstance(item, dict) or not tool:
        return None
    if tool.startswith("google_maps"):
        # text_search items don't include `url`; build the canonical
        # place-id deep link. place_details items do include `url`.
        url = item.get("url")
        if url:
            return {"type": "url", "value": str(url)}
        place_id = item.get("place_id")
        if place_id:
            return {
                "type": "url",
                "value": f"https://www.google.com/maps/place/?q=place_id:{place_id}",
            }
        return None
    if tool.startswith("fullenrich"):
        url = item.get("linkedin_url") or item.get("linkedinUrl")
        if url:
            return {"type": "url", "value": str(url)}
        domain = item.get("domain") or item.get("website")
        if domain:
            v = str(domain)
            if not v.startswith("http"):
                v = f"https://{v}"
            return {"type": "url", "value": v}
        return None
    if tool.startswith("apollo"):
        url = item.get("linkedin_url") or item.get("website_url")
        if url:
            return {"type": "url", "value": str(url)}
        domain = item.get("primary_domain") or item.get("domain")
        if domain:
            v = str(domain)
            if not v.startswith("http"):
                v = f"https://{v}"
            return {"type": "url", "value": v}
        return None
    if tool.startswith("apify") or tool == "browser_use" or tool == "web_harvest":
        url = (
            item.get("url")
            or item.get("link")
            or item.get("href")
            or item.get("source_url")
            or item.get("profile_url")
        )
        if url:
            return {"type": "url", "value": str(url)}
        return None
    return None


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
