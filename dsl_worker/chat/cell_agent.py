"""Cell agent — per-row enrichment runner with research-level routing.

Spawned per row for every enrichment. Each cell agent gets:
  - The full current row data (already-filled columns + hidden source fields)
  - The action's `prompt` (natural-language goal)
  - `columns_to_fill` — which column names to produce
  - A toolset (depends on research level)
  - A credit budget per row (enforced programmatically; NOT shown to the LLM)

Research levels — picked per enrichment:

  - "classify"  gpt-5.4-nano  | no tools — decide a label from row text
  - "research"  gpt-5.4-mini  | all tools — go find more data (web /
                                FE / Apollo / browser_use / etc.)
  - "deep"      gpt-5.5       | all tools — harder multi-step lookups

The split is binary: does the cell agent need to look OUTSIDE this row's
data? If yes → research/deep. If no → classify.

Provider is a runtime toggle for the tool-using tiers (research/deep), via
the ENRICHMENT_LLM_PROVIDER env var:

  - "openai"    (default) → OpenAI Responses path (unchanged)
  - "anthropic"           → Claude via the Messages API (_anthropic_cell_loop):
                            research = Haiku 4.5, deep = Sonnet 4.6. Server-side
                            web_search cap (max_uses), exact search-count
                            billing, real citation URLs, prompt-cached system.

classify always stays on gpt-5.4-nano (OpenAI). Default is OpenAI, so nothing
changes unless you opt in; flip back any time by unsetting the var.

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
import random
import time
import uuid
from collections import OrderedDict
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple

import httpx
from openai import AsyncOpenAI
from sqlalchemy import text as sa_text

from dsl_worker.billing.pricing import get_pricing_config
from dsl_worker.billing.tracked_client import TrackedOpenAIClient
from dsl_worker.chat.tools import ToolContext

# NOTE: the Anthropic SDK + Claude helpers are imported LAZILY inside the
# anthropic-only functions below (not at module top), so the default OpenAI
# enrichment path carries zero new import dependencies — a broken Claude path
# can never break normal enrichments.


log = logging.getLogger(__name__)


class _FERateLimiter:
    """Process-wide pacing gate for FullEnrich API calls.

    FE limits a workspace to 60 API calls/min across ALL endpoints — and,
    per FE support, the GET poll counts just like the submit. With ~25
    concurrent cells each doing a 1-contact submit + a poll loop, we trip
    the limiter constantly: the submit 429s, the cell falls back to
    web_search and commits a null email even though FE would have found it.

    This spaces every FE call (submit AND poll) to stay just under the
    workspace limit, so calls queue and pace instead of failing. Strict
    spacing: each caller reserves the next slot (min_interval apart) under a
    short lock, then sleeps outside the lock until its slot — so concurrent
    callers form an orderly FIFO at the target rate rather than bursting.

    Configurable via FULLENRICH_RPM (default 55, safely under 60). When FE
    raises us to 300/min, bump the env var — no code change. If we ever run
    >1 chat-worker replica sharing one FE workspace, set RPM to 55/N since
    the limit is per-workspace and each process keeps its own bucket.
    """

    def __init__(self, rpm: int) -> None:
        self._min_interval = 60.0 / max(1, rpm)
        self._lock = asyncio.Lock()
        self._next_at = 0.0

    async def acquire(self) -> None:
        async with self._lock:
            now = time.monotonic()
            scheduled = max(now, self._next_at)
            self._next_at = scheduled + self._min_interval
        delay = scheduled - time.monotonic()
        if delay > 0:
            await asyncio.sleep(delay)


_fe_limiter = _FERateLimiter(int(os.getenv("FULLENRICH_RPM", "55")))


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
#
# All values are USD — same unit as total_cost. Tuned to typical actual
# bills observed in cell_traces:
#   FE email      ~$0.06 per call (1 FE-cr × $0.055; verified 5/21+ batch)
#   FE phone      ~$0.55 per call (10 FE-cr × $0.055 per FE docs)
#   FE company    ~$0.06 per call (1 FE-cr per email match)
#   gmaps         ~$0.017 per place_details
# Pre-2026-05-21 we observed FE billing 18 credits per email call (~$1)
# but that appears to have been an FE-side billing bug they fixed; if it
# recurs, raise these estimates back up + audit cell_traces.cost_credits.
# Apollo people/match consumes ~1 export credit per MATCHED person
# (no charge on miss). Paid plans price an export credit at roughly a
# cent; tune via env if the plan changes.
APOLLO_MATCH_COST_USD = float(os.getenv("APOLLO_MATCH_COST_USD", "0.01"))

# Apollo phone reveals consume credits per delivered number, reported back
# in the webhook payload (observed live: 8 credits per mobile, 0 on a
# failed reveal). Billed dynamically at credits_consumed × this rate; the
# FIXED_COST_TOOLS entry is only the pre-call budget gate's estimate.
APOLLO_CREDIT_COST_USD = float(os.getenv("APOLLO_CREDIT_COST_USD", "0.0125"))

FIXED_COST_TOOLS = {
    "fullenrich_enrich_email":  0.07,
    "fullenrich_enrich_phone":  0.60,
    "fullenrich_enrich_company": 0.07,
    "apollo_enrich_person": APOLLO_MATCH_COST_USD,
    "apollo_reveal_phone": 8 * APOLLO_CREDIT_COST_USD,
    # apollo_org_enrich removed — organizations/enrich is request-quota
    # limited (Apollo's response headers confirm: x-rate-limit-* not
    # credit-* ). Treat as free; pre-call budget gate doesn't refuse it.
    "google_maps_place_details": 0.05,
}

# Hard floor — refuse to call BU if remaining budget is below this, since
# even a one-step BU session typically costs ~$0.10-0.30 and the overhead
# wouldn't get you anything useful.
BU_MIN_BUDGET = 0.30

# Same idea for apify — actor runs need time + a few CU to be worth it.
# USD-denominated to match BU_MIN_BUDGET and the tier_cfg.cap units.
# Was 0.50 which is $0.50 floor — wildly over for cheap actors like
# harvestapi/linkedin-company ($0.004/item). Project db529ab4 had every
# call skipped with "needs at least 0.5 cr; 0.04 remaining" because
# the floor was 12x typical per_row_credit_cap. 0.03 lets cheap actors
# run while still blocking pointless apify attempts on near-empty budgets.
APIFY_MIN_BUDGET = 0.03

CAPPED_TOOLS = {"browser_use", "apify_call_actor"}

# Legacy reference — kept so external imports don't break. USD-denominated
# now that total_cost is USD; numbers tuned to typical observed bills.
TOOL_COST_ESTIMATES = dict(FIXED_COST_TOOLS)
TOOL_COST_ESTIMATES.update({
    "browser_use": 0.50,
    "apify_call_actor": 0.10,
    "apify_search_actors": 0.0,
    "apify_actor_details": 0.0,
    "web_search": 0.0,
    "code_exec": 0.0,
})


# ---------------------------------------------------------------------------
# Tier configuration
# ---------------------------------------------------------------------------


RESEARCH_CONFIG = {
    # Three tiers. The first split is binary ("process row data only" vs
    # "go find more data"); the second split inside research is "default
    # mini" vs "smarter 5.5 for nuance/multi-step." `default_cap` is in
    # CREDITS — the orchestrator hands per_row_credit_cap in credits and
    # we convert to USD at the boundary in _resolve_research so all
    # internal math uses USD (matches total_cost). Per-enrichment caps
    # passed at action time override these defaults.
    "classify": {"model": "gpt-5.4-nano", "provider": "openai", "effort": "medium", "default_cap": 0.05, "tools": []},
    "research": {"model": "gpt-5.4-mini", "provider": "openai", "effort": "medium", "default_cap": 2.0, "tools": "all"},
    "deep":     {"model": "gpt-5.5",      "provider": "openai", "effort": "medium", "default_cap": 5.0, "tools": "all"},
}

# Provider for the tool-using enrichment tiers (research/deep).
#
#   ENRICHMENT_LLM_PROVIDER unset / anything ≠ "openai" → Claude (DEFAULT)
#   ENRICHMENT_LLM_PROVIDER = "openai"                  → OpenAI path
#
# Anthropic is the default: research → ENRICHMENT_ANTHROPIC_MODEL (Haiku 4.5, the
# cheap workhorse) and deep → ENRICHMENT_ANTHROPIC_DEEP_MODEL (Sonnet 4.6, the
# smarter model for nuanced multi-step lookups). Set the var to "openai" to opt
# OUT and run the OpenAI path instead. `classify` ALWAYS stays on gpt-5.4-nano
# (OpenAI) — it never web-searches, so Claude would be pure overhead and nano is
# ~20x cheaper per token. If Anthropic is selected but ANTHROPIC_API_KEY is
# missing, we log and fall back to OpenAI rather than failing every cell —
# see _resolve_research.
ENRICHMENT_LLM_PROVIDER = os.getenv("ENRICHMENT_LLM_PROVIDER", "anthropic").strip().lower()
ENRICHMENT_ANTHROPIC_MODEL = os.getenv("ENRICHMENT_ANTHROPIC_MODEL", "claude-haiku-4-5").strip()
ENRICHMENT_ANTHROPIC_DEEP_MODEL = os.getenv("ENRICHMENT_ANTHROPIC_DEEP_MODEL", "claude-sonnet-4-6").strip()

# Every old name (across every prior rename pass + the latest collapse)
# normalizes to one of the three canonical tiers. Old enrichment rows
# in the DB keep working, agent slips are tolerated. `deep` is now a
# real tier (not aliased to research) — same for `expert` / `investigate`
# which route to deep so the smarter model handles them.
LEGACY_ALIASES = {
    # v4 (none/low/medium/high) — current rename. Anything that needed
    # tools collapses to "research"; tool-less stays "classify";
    # explicit-high routes to deep since the user asked for more depth.
    "none":   "classify",
    "low":    "research",
    "medium": "research",
    "high":   "deep",
    # v3 (classify/lookup/search/investigate)
    "lookup":      "research",
    "search":      "research",
    "investigate": "deep",
    # v2 (light)
    "light":       "research",
    # v1 (fast/smart/standard/deep/expert)
    "fast":     "classify",
    "smart":    "research",
    "expert":   "deep",
    "standard": "research",
}


def _resolve_research(action: Dict[str, Any], per_row_cap: Optional[float]) -> Dict[str, Any]:
    """Return resolved config: {model, effort, cap, cap_credits, tools, name}.

    Reads `research` (or legacy `tier`) from the action. `per_row_cap` is
    in CREDITS (1 cr = $0.10) — that's how the orchestrator names it
    (per_row_credit_cap). Internal cap is USD so it can be compared to
    total_cost.
    """
    requested = (action.get("research") or action.get("tier") or "research").lower()
    requested = LEGACY_ALIASES.get(requested, requested)
    if requested not in RESEARCH_CONFIG:
        log.warning("cell_agent: unknown research %r, defaulting to research", requested)
        requested = "research"
    cfg = RESEARCH_CONFIG[requested].copy()
    # Provider for the tool-using tiers (research/deep): Anthropic by default,
    # OpenAI only when ENRICHMENT_LLM_PROVIDER is explicitly "openai". classify
    # always stays on nano/OpenAI. Falls back to OpenAI when the key is absent
    # so a missing key can't break every research cell.
    if requested in ("research", "deep") and ENRICHMENT_LLM_PROVIDER != "openai":
        if os.getenv("ANTHROPIC_API_KEY"):
            cfg["provider"] = "anthropic"
            cfg["model"] = ENRICHMENT_ANTHROPIC_DEEP_MODEL if requested == "deep" else ENRICHMENT_ANTHROPIC_MODEL
        else:
            log.warning(
                "ENRICHMENT_LLM_PROVIDER defaults to anthropic but ANTHROPIC_API_KEY "
                "is unset — using OpenAI for the %s tier", requested,
            )
    cap_credits = float(per_row_cap) if per_row_cap and per_row_cap > 0 else cfg["default_cap"]
    cfg["cap_credits"] = cap_credits
    cfg["cap"] = cap_credits * 0.10
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
    poll_interval_s: int = 8,
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
    # FE's submit endpoint occasionally takes >30s to ack — observed in
    # prod logs with httpx.ReadTimeout. Retry once with a longer ceiling
    # before giving up; otherwise the cell agent sees an opaque timeout,
    # retries at the LLM level (burns web_search budget on fallback), and
    # eventually hits its per-row cap with no email found. Single retry
    # bounds the slow path to ~2×timeout = ~60s worst case.
    #
    # Also handles HTTP 429 (rate limit). FE limits the SUBMIT endpoint
    # aggressively when many cells fire at once (25 concurrent cells × N
    # enrichment_run jobs = bursts that trip the limiter). Until 2026-05-26
    # we returned the 429 directly to the cell agent, which fell back to
    # web_search and committed null even though the email was findable on
    # a normal call. Now we honor Retry-After (capped at 70s) and retry
    # up to 3 times before giving up.
    r = None
    last_err: Optional[Exception] = None
    for attempt in range(3):
        try:
            await _fe_limiter.acquire()
            async with httpx.AsyncClient(timeout=60) as client:
                r = await client.post(
                    f"{BASE}/api/v2/contact/enrich/bulk",
                    headers=headers, json=body,
                )
            if r.status_code == 429:
                # FE message looks like: "Too many requests. Try again in 1m"
                # Retry-After header is the canonical signal but FE doesn't
                # always send it; default to 65s which covers the 1m bucket.
                retry_after_s = 65
                ra = r.headers.get("Retry-After")
                if ra:
                    try:
                        retry_after_s = min(70, max(5, int(ra)))
                    except ValueError:
                        pass
                log.warning(
                    "FE submit HTTP 429 on attempt %d — waiting %ds then retrying",
                    attempt + 1, retry_after_s,
                )
                if attempt < 2:
                    await asyncio.sleep(retry_after_s)
                    continue
                # final attempt also failed → fall through to error path below
            break
        except (httpx.ReadTimeout, httpx.ConnectTimeout, httpx.PoolTimeout) as e:
            last_err = e
            log.warning(
                "FE submit timeout on attempt %d: %s — retrying" if attempt < 2
                else "FE submit timeout on attempt %d: %s — giving up",
                attempt + 1, e,
            )
            await asyncio.sleep(2)
    if r is None:
        return {"error": f"FE submit timeout after retry: {last_err}"}, 0.0
    if r.status_code != 200:
        return {"error": f"FE submit HTTP {r.status_code}: {r.text[:200]}"}, 0.0
    eid = (r.json() or {}).get("enrichment_id")
    if not eid:
        return {"error": "FE returned no enrichment_id"}, 0.0
    # Poll. Single-contact bulk runs typically finish in 30-60s. Beyond
    # timeout_s, give up and charge nothing (user shouldn't pay for our
    # poll budget running out).
    #
    # Budget against WALL TIME via time.monotonic(), not iteration count.
    # The old `elapsed += poll_interval_s` undercounted whenever a poll
    # hit the 15s httpx timeout: each iteration took up to 20s wall time
    # but only credited 5s of the budget. With FE lagging, that meant
    # cells could run for 30-40min before the 600s "budget" caught up.
    # Confirmed against the 11 cells in job 9620671a stuck at 13-18min
    # (elapsed counter was at ~200s when reaped).
    import time as _time
    deadline = _time.monotonic() + timeout_s
    last_result: Dict[str, Any] = {}
    while _time.monotonic() < deadline:
        await asyncio.sleep(poll_interval_s)
        try:
            await _fe_limiter.acquire()
            async with httpx.AsyncClient(timeout=15) as client:
                g = await client.get(f"{BASE}/api/v2/contact/enrich/bulk/{eid}", headers=headers)
        except (httpx.ReadTimeout, httpx.ConnectTimeout, httpx.PoolTimeout) as e:
            # FE's poll endpoint sometimes lags past the 15s socket timeout
            # — that's a transient backend slowdown, not a real failure.
            # Without this catch the entire bulk enrich died on a single
            # slow poll: stack bubbled out as 'cell tool
            # fullenrich_enrich_email raised: httpx.ReadTimeout', cell
            # agent treated it as a tool failure, model burned web_searches
            # trying to recover. Just skip this poll and try again next
            # iteration; the bulk job is still running FE-side.
            remaining = max(0, int(deadline - _time.monotonic()))
            log.warning(
                "FE poll timeout (eid=%s, %ds remaining) — retrying next interval: %s",
                eid, remaining, e,
            )
            continue
        if g.status_code != 200:
            continue
        last_result = g.json() or {}
        status = last_result.get("status", "")
        if status == "FINISHED":
            break
        if status in ("CANCELED", "CREDITS_INSUFFICIENT", "RATE_LIMIT"):
            return {"error": f"FE {status}"}, 0.0
    else:
        return {
            "error": (
                f"FE timed out after {timeout_s}s — FullEnrich is stuck on this "
                f"contact and a retry will hang the same way. Do NOT call this "
                f"tool again for this row; fall back to another source or commit "
                f"null."
            ),
            "enrichment_id": eid,
        }, 0.0

    items = last_result.get("data") or []
    if not items:
        return {"contact_info": {}, "_raw_payload": last_result}, 0.0
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
    # TEMP diagnostic: dump the FE response shape so we can see exactly
    # what `cost.credits` represents. The user's plan ($79.50 / 1500
    # credits = ~5¢/credit, advertised as "1500 emails") implies 1 credit
    # per successful email match. We're measuring 18 credits/call on
    # calls that returned email=null — that's a contradiction worth
    # resolving. Remove this log once we know the answer.
    # Confirmed: cost.credits == 1.0 on a successful email match (contact_info
    # has work_emails / most_probable_work_email), 0.0 on a miss. Kept at
    # debug so it stays available without flooding the worker log.
    log.debug(
        "[FE_DIAG] cost.credits=%s contact_info_keys=%s top_level_keys=%s",
        raw_fe_credits,
        sorted((contact_info or {}).keys()),
        sorted(last_result.keys()),
    )
    # _raw_payload = FE's complete poll response (full contact_info with
    # every email/phone candidate, social profiles, cost meta). Popped at
    # the capture site for the source-chip payload; never sent to the LLM.
    return {"contact_info": contact_info, "_raw_payload": last_result}, cost_usd


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
    # Cap each email lookup at 120s. FE's waterfall "typically finishes in
    # 30-60s" (and p90 here was ~90s), so 120s covers virtually every real
    # match while killing the dead-wait on stuck calls — the old 600s ceiling
    # let ONE hung lookup block 10 minutes, and a cell does ~3 of these
    # serially, which is how a single row stretched into hours. A lookup that
    # legitimately needs >120s now returns empty for that contact (rare tail);
    # successful sub-120s lookups are completely unaffected.
    result, credits = await _fullenrich_bulk_enrich(
        api_key, contact, ["contact.emails"], timeout_s=120,
    )
    if "error" in result:
        return result, credits
    raw_payload = result.pop("_raw_payload", None)
    ci = result.get("contact_info") or {}
    # Prefer the highest-confidence single answer FullEnrich picked,
    # fall back to the first entry in work_emails[].
    mp = ci.get("most_probable_work_email") or {}
    email = mp.get("email")
    status = mp.get("status")
    if not email:
        for e in (ci.get("work_emails") or []):
            if isinstance(e, dict) and e.get("email"):
                email = e.get("email")
                status = e.get("status")
                break
    # Map FullEnrich's verification_status enum to a clear commit signal
    # so the model doesn't have to interpret it. Per FE's official docs
    # (help.fullenrich.com/en/articles/9377499):
    #   DELIVERABLE       — 1%  bounce — safe to send
    #   HIGH_PROBABILITY  — 9%  bounce — usually safe
    #   CATCH_ALL         — 26% bounce — depends
    #   INVALID           —     bounce — do not send
    #
    # Commit everything FE surfaced EXCEPT INVALID. CATCH_ALL gets
    # committed because email_verify_hook fires Scrubby (real-time SMTP
    # probe) on every email-column commit; Scrubby is more rigorous than
    # FE's CATCH_ALL classification and will null any cell that actually
    # bounces, so the high-bounce risk on raw CATCH_ALL is contained.
    #
    # Without this explicit map the model silently dropped
    # HIGH_PROBABILITY results (FE's most common non-DELIVERABLE return)
    # as "not verified enough" and committed null after burning
    # web_searches trying to verify, even when FE returned the right
    # email at 9% bounce.
    INVALID_STATUSES = {"INVALID"}
    should_commit = email is not None and (status or "").upper() not in INVALID_STATUSES
    return {
        "email": email,
        "verification_status": status,
        "commit": should_commit,
        "_raw_payload": raw_payload,
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
    # 300s ceiling: healthy FE phone lookups finish in 1-5 min; ones that
    # haven't by then are stuck and observed to never produce data (user
    # watched runs sit 30+ min and fail). Was 600s — that just doubled the
    # wall-clock of every doomed lookup.
    result, credits = await _fullenrich_bulk_enrich(
        api_key, contact, ["contact.phones"], timeout_s=300,
    )
    if "error" in result:
        return result, credits
    raw_payload = result.pop("_raw_payload", None)
    ci = result.get("contact_info") or {}
    # FE's actual response shape:
    #   most_probable_phone: {"number": "+33...", "region": "FR"}
    #   phones: [{"number": "...", "region": "..."}, ...]
    # The previous wrapper read `most_probable_work_phone` + `work_phones`
    # + `personal_phones` — none of those keys exist in the response.
    # Every phone enrichment was returning null even when FE had the
    # phone. Read the actual field names + key (`number`, not `phone`).
    mp = ci.get("most_probable_phone") or {}
    phone = mp.get("number")
    region = mp.get("region")
    if not phone:
        for p in (ci.get("phones") or []):
            if isinstance(p, dict) and p.get("number"):
                phone = p.get("number")
                region = p.get("region")
                break
    return {"phone": phone, "region": region, "_raw_payload": raw_payload}, credits


async def _fullenrich_enrich_company(args: Dict[str, Any], ctx: ToolContext) -> Tuple[Dict[str, Any], float]:
    """Domain → company info via FullEnrich.

    FE has no dedicated /company/enrich endpoint (the old wrapper hit
    /api/v1/company/enrich which 404s on every call). Use /company/search
    with a domain filter — same result for our purposes, returns the
    LinkedIn-derived company record (description, year founded,
    headcount, HQ address, company LinkedIn URL, industry).
    Cost: 0.25 FE-credit (~$0.013) per returned company.
    """
    import httpx
    api_key = os.getenv("FULLENRICH_API_KEY")
    if not api_key:
        return {"error": "FULLENRICH_API_KEY not configured"}, 0.0
    domain = args.get("domain")
    if not domain:
        return {"error": "domain is required"}, 0.0
    try:
        async with httpx.AsyncClient(timeout=60) as client:
            r = await client.post(
                "https://app.fullenrich.com/api/v2/company/search",
                headers={"Authorization": f"Bearer {api_key}"},
                json={
                    "domains": [{"value": domain, "exact_match": False, "exclude": False}],
                    "limit": 1,
                },
            )
    except (httpx.ReadTimeout, httpx.ConnectTimeout, httpx.PoolTimeout) as e:
        return {"error": f"FE company/search timeout: {e}"}, 0.0
    if r.status_code != 200:
        return {"error": f"FE HTTP {r.status_code}: {r.text[:200]}"}, 0.0
    data = r.json() or {}
    cos = data.get("companies") or []
    if not cos:
        return {"company": None, "found": False}, 0.0
    c = cos[0]
    hq = ((c.get("locations") or {}).get("headquarters") or {})
    linkedin = ((c.get("social_profiles") or {}).get("professional_network") or {}).get("url")
    return {
        "found": True,
        "name": c.get("name"),
        "domain": c.get("domain"),
        "description": (c.get("description") or "")[:800] or None,
        "year_founded": c.get("year_founded"),
        "headcount": c.get("headcount"),
        "headcount_range": c.get("headcount_range"),
        "company_type": c.get("company_type"),
        "industry": (c.get("industry") or {}).get("main_industry"),
        "specialties": c.get("specialties") or None,
        "hq_city": hq.get("city"),
        "hq_region": hq.get("region"),
        "hq_country": hq.get("country"),
        "hq_address": hq.get("line1"),
        "linkedin_url": linkedin,
        "_raw_payload": data,
    }, 0.25 * 0.055


async def _fullenrich_search_people(args: Dict[str, Any], ctx: ToolContext) -> Tuple[Dict[str, Any], float]:
    """Search FullEnrich for people at a company (LinkedIn-derived).

    Per-row variant — returns a compact list of candidates inline (no
    file writing). The cell agent picks the right person from the list
    and commits the value. Used by the `find_person_at_company` skill.

    Cost shape: FE charges 0.25 credit (~$0.013) per RETURNED person,
    NOT per call. So `limit` is the cost knob — at limit=5 that's
    ~$0.065. Keep it small (1-3) when the column wants ONE specific
    person (founder, owner, CEO); bump only when picking from a real
    candidate pool. Hard-capped at 10.
    """
    import httpx
    api_key = os.getenv("FULLENRICH_API_KEY")
    if not api_key:
        return {"error": "FULLENRICH_API_KEY not configured"}, 0.0

    # Build the filter shape FE's /people/search wants. Same param names
    # as the orchestrator-side namespace (company_names, company_domains,
    # titles, person_locations, seniority) so the skill instructions
    # transfer 1:1.
    def _str_filter(values: Any) -> Optional[List[Dict[str, Any]]]:
        if not values:
            return None
        if isinstance(values, str):
            values = [values]
        out = [{"value": str(v), "exact_match": False, "exclude": False}
               for v in values if str(v).strip()]
        return out or None

    filters: Dict[str, Any] = {}
    for arg_key, api_key_name in [
        ("company_names", "current_company_names"),
        ("company_domains", "current_company_domains"),
        ("person_names", "person_names"),
        ("titles", "current_position_titles"),
        ("locations", "person_locations"),
        ("seniority", "current_position_seniority_level"),
    ]:
        f = _str_filter(args.get(arg_key))
        if f:
            filters[api_key_name] = f
    if not filters:
        return {"error": "at least one of company_names/company_domains/titles is required"}, 0.0

    # Default limit=3 — empirically the sweet spot for "find a person at
    # this company" lookups. limit=1 nails clean cases but misses when
    # cross-company noise crowds out the right match (e.g. searching
    # "Sales Hatch" returns 2 wrong-company people before the actual
    # Sales Hatch one). limit=3 catches the right person after the
    # post-filter drops mismatched employers. Each result is 0.25
    # FE-credit ≈ $0.013, so limit=3 = ~$0.04. Hard-capped at 10.
    limit = int(args.get("limit") or 3)
    limit = max(1, min(limit, 10))

    body = {**filters, "limit": limit, "offset": 0}
    try:
        async with httpx.AsyncClient(timeout=60) as client:
            r = await client.post(
                "https://app.fullenrich.com/api/v2/people/search",
                headers={"Authorization": f"Bearer {api_key}"},
                json=body,
            )
    except (httpx.ReadTimeout, httpx.ConnectTimeout, httpx.PoolTimeout) as e:
        return {"error": f"FullEnrich people/search timeout: {e}"}, 0.0
    if r.status_code != 200:
        return {"error": f"FullEnrich HTTP {r.status_code}: {r.text[:200]}"}, 0.0
    data = r.json() or {}
    people = data.get("people") or []
    meta = data.get("metadata") or {}
    # FE's Search API charges 0.25 credit per RETURNED person. When the
    # response includes credits_used, trust it; otherwise fall back to
    # len(people) * 0.25 (not len(people) — the per-result rate is 0.25,
    # not 1; treating it as 1 was 4x too high and would gate the call out
    # of any tight per_row_credit_cap).
    credits_used = float(meta.get("credits_used") if meta.get("credits_used") is not None else len(people) * 0.25)
    cost_usd = credits_used * 0.055

    # Optional post-filter: FE's API has been observed to bleed
    # cross-company results when company_names/domains is the filter.
    # Drop rows whose CURRENT employer doesn't include the requested
    # company string. Same logic as the orchestrator handler.
    wanted_names = [str(v).strip().lower() for v in (args.get("company_names") or []) if str(v).strip()]
    wanted_domains = [str(v).strip().lower() for v in (args.get("company_domains") or []) if str(v).strip()]
    if wanted_names or wanted_domains:
        kept = []
        for p in people:
            cur = (p.get("employment") or {}).get("current") or {}
            co = cur.get("company") or {}
            cname = (co.get("name") or "").lower()
            cdom = (co.get("domain") or "").lower()
            if any(w in cname for w in wanted_names) or any(w in cdom for w in wanted_domains):
                kept.append(p)
        people = kept

    # Compact response — the cell agent doesn't need FE's full payload,
    # just enough to pick a person and pass the chosen one's name/domain
    # into a downstream enrich_contacts call.
    compact = []
    for p in people:
        cur = (p.get("employment") or {}).get("current") or {}
        co = cur.get("company") or {}
        # Person's LinkedIn URL lives under social_profiles, NOT a
        # top-level linkedin_url field. Same shape for the employer's
        # LinkedIn under company.social_profiles.
        person_prof = ((p.get("social_profiles") or {}).get("professional_network") or {})
        company_prof = ((co.get("social_profiles") or {}).get("professional_network") or {})
        # job_functions is FE's STRUCTURED role classification — e.g.
        # {function: "Executive & Leadership", sub_function: "Founder/Owner"}.
        # More reliable than parsing the free-form title when the cell
        # needs to confirm "is this actually a founder vs a senior IC".
        job_functions = cur.get("job_functions") or []
        compact.append({
            "first_name": p.get("first_name"),
            "last_name": p.get("last_name"),
            "full_name": p.get("full_name"),
            "title": cur.get("title"),
            "description": (cur.get("description") or "")[:500] or None,
            "seniority": cur.get("seniority"),
            "job_functions": job_functions if job_functions else None,
            "start_at": cur.get("start_at"),
            "is_current": cur.get("is_current"),
            "linkedin_url": person_prof.get("url"),
            "location": p.get("location"),
            "employer_name": co.get("name"),
            "employer_domain": co.get("domain"),
            "employer_industry": (co.get("industry") or {}).get("main_industry"),
            "employer_headcount": co.get("headcount"),
            "employer_headcount_range": co.get("headcount_range"),
            "employer_year_founded": co.get("year_founded"),
            "employer_linkedin_url": company_prof.get("url"),
        })

    return {
        "people": compact,
        "count": len(compact),
        "total_in_db": meta.get("total"),
        "credits_used": credits_used,
        # Raw response is pre-post-filter — the payload chip shows every
        # person FE billed for, including ones the employer filter dropped.
        "_raw_payload": data,
    }, cost_usd


async def _apollo_org_enrich(args: Dict[str, Any], ctx: ToolContext) -> Tuple[Dict[str, Any], float]:
    import httpx
    api_key = os.getenv("APOLLO_API_KEY")
    if not api_key:
        return {"error": "APOLLO_API_KEY not configured"}, 0.0
    domain = args.get("domain")
    if not domain:
        return {"error": "domain is required"}, 0.0
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            r = await client.get(
                "https://api.apollo.io/api/v1/organizations/enrich",
                params={"domain": domain},
                headers={"X-Api-Key": api_key},
            )
    except (httpx.ReadTimeout, httpx.ConnectTimeout, httpx.PoolTimeout) as e:
        return {"error": f"Apollo organizations/enrich timeout: {e}"}, 0.0
    if r.status_code != 200:
        return {"error": f"Apollo HTTP {r.status_code}"}, 0.0
    org = (r.json() or {}).get("organization") or {}
    # Apollo /organizations/enrich returns ~56 fields. Surface the ones
    # likely to fill enrichment columns directly; skip the noisy
    # internal IDs (industry_tag_id, snippets_loaded, etc.) and
    # high-volume sub-arrays the agent rarely needs (full funding_events
    # list, suborganizations). When the agent needs something not
    # listed below, it should ask for what column it's filling — we can
    # extend this map without changing the contract.
    return {
        "name": org.get("name"),
        "primary_domain": org.get("primary_domain"),
        "website_url": org.get("website_url"),
        "estimated_num_employees": org.get("estimated_num_employees"),
        "industry": org.get("industry"),
        "secondary_industries": org.get("secondary_industries"),
        "keywords": (org.get("keywords") or [])[:30],
        "naics_codes": org.get("naics_codes"),
        "sic_codes": org.get("sic_codes"),
        "annual_revenue": org.get("annual_revenue"),
        "annual_revenue_printed": org.get("annual_revenue_printed"),
        "total_funding": org.get("total_funding"),
        "total_funding_printed": org.get("total_funding_printed"),
        "latest_funding_stage": org.get("latest_funding_stage"),
        "latest_funding_round_date": org.get("latest_funding_round_date"),
        "publicly_traded_exchange": org.get("publicly_traded_exchange"),
        "publicly_traded_symbol": org.get("publicly_traded_symbol"),
        "founded_year": org.get("founded_year"),
        "linkedin_url": org.get("linkedin_url"),
        "twitter_url": org.get("twitter_url"),
        "facebook_url": org.get("facebook_url"),
        "angellist_url": org.get("angellist_url"),
        "crunchbase_url": org.get("crunchbase_url"),
        "logo_url": org.get("logo_url"),
        "phone": org.get("sanitized_phone") or org.get("primary_phone") or org.get("phone"),
        "street_address": org.get("street_address"),
        "city": org.get("city"),
        "state": org.get("state"),
        "postal_code": org.get("postal_code"),
        "country": org.get("country"),
        "raw_address": org.get("raw_address"),
        "short_description": (org.get("short_description") or "")[:500] or None,
        "current_technologies": [t.get("name") for t in (org.get("current_technologies") or [])][:30],
        "technology_names": (org.get("technology_names") or [])[:30],
        "departmental_head_count": org.get("departmental_head_count"),
        "retail_location_count": org.get("retail_location_count"),
        "alexa_ranking": org.get("alexa_ranking"),
    }, 0.0  # organizations/enrich is request-quota-limited, not credit-billed


def _apollo_employer_alignment(args: Dict[str, Any], person: Dict[str, Any]) -> Tuple[Optional[bool], str]:
    """Does Apollo's matched person currently work where the caller said?

    Returns (aligned, evidence): True/False when the match carries employer
    signals to compare against the requested company/domain, None when it
    doesn't (or none was requested). Catches Apollo's classic same-name
    different-company mismatch — asked for Eric Gonzalez @ San Diego Padres,
    got the Cincinnati Reds one (observed live: the wrong man's mobile was
    revealed and committed) — BEFORE any phone credits burn.
    """
    requested_co = str(args.get("company") or args.get("organization_name") or "").strip().lower()
    requested_dom = str(args.get("domain") or "").strip().lower().removeprefix("www.")
    if not (requested_co or requested_dom):
        return None, ""
    org = person.get("organization") or {}
    org_name = (org.get("name") or "").strip()
    org_site = (org.get("website_url") or "").lower()
    headline = (person.get("headline") or "")
    if org_name:
        n = org_name.lower()
        if requested_co and (requested_co in n or n in requested_co):
            return True, org_name
        if requested_dom and requested_dom in org_site:
            return True, org_name
        return False, org_name
    if requested_dom and org_site:
        return (requested_dom in org_site), org_site
    # No org object on the match — fall back to the headline's "<title> at
    # <Company>" convention when present.
    if headline and " at " in headline.lower():
        if requested_co and requested_co in headline.lower():
            return True, headline
        return False, headline
    return None, ""


async def _apollo_enrich_person(args: Dict[str, Any], ctx: ToolContext) -> Tuple[Dict[str, Any], float]:
    """Apollo people/match — person enrichment by name + company/domain.

    Returns the matched person's title, WORK email (+ verification
    status), LinkedIn URL, and location in one synchronous call.
    Deliberately never sets reveal_personal_emails (GDPR/personal data).
    Phone reveals live in the separate apollo_reveal_phone tool
    (webhook-async, ~8 credits per number) so a cheap email match can
    never accidentally trigger a phone charge.

    Billing: ~1 export credit per MATCHED person (env
    APOLLO_MATCH_COST_USD, default $0.01); $0 on a miss.
    """
    import httpx
    api_key = os.getenv("APOLLO_API_KEY")
    if not api_key:
        return {"error": "APOLLO_API_KEY not configured"}, 0.0
    body: Dict[str, Any] = {}
    for k_in, k_out in [
        ("first_name", "first_name"),
        ("last_name", "last_name"),
        ("name", "name"),
        ("company", "organization_name"),
        ("organization_name", "organization_name"),
        ("domain", "domain"),
        ("linkedin_url", "linkedin_url"),
        ("email", "email"),
    ]:
        v = args.get(k_in)
        if v and str(v).strip():
            body[k_out] = str(v).strip()
    if not body:
        return {"error": "pass a name + company/domain, or a linkedin_url, or an email"}, 0.0
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            r = await client.post(
                "https://api.apollo.io/api/v1/people/match",
                json=body,
                headers={"X-Api-Key": api_key},
            )
    except (httpx.ReadTimeout, httpx.ConnectTimeout, httpx.PoolTimeout) as e:
        return {"error": f"Apollo people/match timeout: {e}"}, 0.0
    if r.status_code != 200:
        return {"error": f"Apollo HTTP {r.status_code}: {r.text[:160]}"}, 0.0
    p = (r.json() or {}).get("person") or {}
    if not p:
        return {"matched": False}, 0.0
    email = p.get("email")
    # Plan out of export credits → Apollo returns a placeholder address
    # instead of null. Never surface it as a value.
    if email and "email_not_unlocked" in email:
        email = None
    # No-match comes back as an ECHO of the input (a stub person with the
    # name you sent and every enrichment field null), not as an empty
    # response. No signal beyond the echo = a miss: report it as one and
    # charge nothing.
    if not any([p.get("title"), email, p.get("linkedin_url"), p.get("headline")]):
        return {"matched": False, "note": "Apollo has no data on this person"}, 0.0
    org = p.get("organization") or {}
    out = {
        "matched": True,
        "name": p.get("name"),
        "first_name": p.get("first_name"),
        "last_name": p.get("last_name"),
        "title": p.get("title"),
        "headline": p.get("headline"),
        "email": email,
        "email_status": p.get("email_status"),
        "linkedin_url": p.get("linkedin_url"),
        "city": p.get("city"),
        "state": p.get("state"),
        "country": p.get("country"),
        "organization": {
            "name": org.get("name"),
            "website_url": org.get("website_url"),
            # Self-describing key: this is the company's MAIN line, never a
            # person-level number (agents were committing it as "Work Phone").
            "switchboard_phone": org.get("sanitized_phone") or org.get("phone"),
        },
        "_raw_payload": p,
    }
    aligned, evidence = _apollo_employer_alignment(args, p)
    if aligned is False:
        out["warning"] = (
            f"Apollo's match currently works at '{evidence}', NOT the company "
            f"you asked about — likely a same-name different person (or a job "
            f"change). Do not commit this contact info for the row without "
            f"independent corroboration."
        )
    return out, APOLLO_MATCH_COST_USD


# Poll ceiling for Apollo's webhook-delivered phone reveals. Observed
# delivery: 9-11s after the match call; 90s is a generous ceiling that
# still fails fast enough for the agent to fall back to FullEnrich.
APOLLO_PHONE_POLL_TIMEOUT_S = float(os.getenv("APOLLO_PHONE_POLL_TIMEOUT_S", "90"))


async def _apollo_reveal_phone(args: Dict[str, Any], ctx: ToolContext) -> Tuple[Dict[str, Any], float]:
    """Apollo mobile/direct-dial phone reveal — webhook-async, polled via DB.

    Apollo never returns phone numbers synchronously: people/match with
    reveal_phone_number=true REQUIRES a public webhook_url and delivers
    the numbers there ~10s later (verified live 2026-06-10; a re-match
    does NOT surface the revealed number afterward, so the webhook body
    is the only copy). Instead of running a public receiver, webhook_url
    points at Supabase's REST endpoint — PostgREST inserts the payload
    into apollo_webhook_events (apikey + columns as query params; the
    columns filter makes PostgREST ignore unknown payload keys, so new
    Apollo fields can't break the insert). This handler polls that table
    for the matched person's id — works from any worker sharing the DB,
    local dev included.

    Billing: the webhook reports credits_consumed (observed: 8 per
    mobile, 0 on a failed reveal) → billed at credits_consumed ×
    APOLLO_CREDIT_COST_USD. Timeout/miss = $0.
    """
    import httpx
    api_key = os.getenv("APOLLO_API_KEY")
    if not api_key:
        return {"error": "APOLLO_API_KEY not configured"}, 0.0
    supabase_url = (os.getenv("SUPABASE_URL") or "").rstrip("/")
    anon_key = os.getenv("SUPABASE_ANON_KEY")
    if not (supabase_url and anon_key):
        return {
            "error": (
                "SUPABASE_URL/SUPABASE_ANON_KEY not configured — phone reveal "
                "has no webhook landing table here; use fullenrich_enrich_phone"
            ),
        }, 0.0
    body: Dict[str, Any] = {}
    for k_in, k_out in [
        ("first_name", "first_name"),
        ("last_name", "last_name"),
        ("name", "name"),
        ("company", "organization_name"),
        ("organization_name", "organization_name"),
        ("domain", "domain"),
        ("linkedin_url", "linkedin_url"),
        ("email", "email"),
    ]:
        v = args.get(k_in)
        if v and str(v).strip():
            body[k_out] = str(v).strip()
    if not body:
        return {"error": "pass a name + company/domain, or a linkedin_url, or an email"}, 0.0

    # Phase 1 — VALIDATE the match before any phone credits burn: a plain
    # match (1 export credit ≈ $0.01) tells us WHO Apollo would reveal. A
    # reveal on a same-name wrong-person match costs 8 credits and poisons
    # the row with a real-but-wrong mobile (observed live: asked for Eric
    # Gonzalez @ Padres, Apollo revealed the Cincinnati Reds one).
    async def _match(payload: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                r = await client.post(
                    "https://api.apollo.io/api/v1/people/match",
                    json=payload,
                    headers={"X-Api-Key": api_key},
                )
        except (httpx.ReadTimeout, httpx.ConnectTimeout, httpx.PoolTimeout) as e:
            return None, f"Apollo people/match timeout: {e}"
        if r.status_code != 200:
            return None, f"Apollo HTTP {r.status_code}: {r.text[:160]}"
        return (r.json() or {}).get("person") or {}, None

    p, err = await _match(body)
    if err:
        return {"error": err}, 0.0
    pid = p.get("id")
    if not pid:
        return {"matched": False, "note": "Apollo has no record of this person"}, 0.0
    aligned, evidence = _apollo_employer_alignment(args, p)
    if aligned is False:
        return {
            "found": False,
            "wrong_person": True,
            "matched_person": {
                "name": p.get("name"),
                "title": p.get("title"),
                "employer": evidence,
            },
            "note": (
                f"Apollo's best match for this name currently works at "
                f"'{evidence}', NOT the company you asked about — revealing "
                f"would return a real mobile belonging to the wrong person, "
                f"so nothing was revealed. If this IS the right person (job "
                f"change), re-call with their linkedin_url or email only."
            ),
        }, APOLLO_MATCH_COST_USD

    # Phase 2 — fire the reveal.
    body["reveal_phone_number"] = True
    body["webhook_url"] = (
        f"{supabase_url}/rest/v1/apollo_webhook_events"
        f"?apikey={anon_key}"
        "&columns=status,unique_enriched_records,credits_consumed,people"
    )

    from dsl_api.db import SessionLocal

    # High-water mark BEFORE the reveal: only accept events newer than
    # this, so a re-reveal of the same person never re-bills off a stale
    # event (bigserial beats timestamps — no clock-skew games).
    db = SessionLocal()
    try:
        min_id = db.execute(
            sa_text("SELECT COALESCE(MAX(id), 0) FROM apollo_webhook_events")
        ).scalar() or 0
    except Exception as e:
        log.warning("apollo_webhook_events high-water read failed: %s", e)
        min_id = 0
    finally:
        db.close()

    p2, err = await _match(body)
    if err:
        return {"error": err}, APOLLO_MATCH_COST_USD
    pid = (p2 or {}).get("id") or pid

    event: Optional[Dict[str, Any]] = None
    deadline = time.monotonic() + APOLLO_PHONE_POLL_TIMEOUT_S
    while time.monotonic() < deadline:
        await asyncio.sleep(3)
        db = SessionLocal()
        try:
            row = db.execute(
                sa_text(
                    "SELECT status, credits_consumed, people "
                    "FROM apollo_webhook_events "
                    "WHERE id > :min_id AND people @> CAST(:pid AS jsonb) "
                    "ORDER BY id DESC LIMIT 1"
                ),
                {"min_id": min_id, "pid": json.dumps([{"id": pid}])},
            ).fetchone()
        except Exception as e:
            log.warning("apollo webhook poll failed: %s", e)
            row = None
        finally:
            db.close()
        if row:
            event = {
                "status": row[0],
                "credits_consumed": float(row[1] or 0),
                "people": row[2],
            }
            break
    if event is None:
        return {
            "error": (
                f"Apollo accepted the reveal but no result arrived within "
                f"{int(APOLLO_PHONE_POLL_TIMEOUT_S)}s. Do NOT retry this tool "
                f"for this row; fall back to fullenrich_enrich_phone."
            ),
        }, APOLLO_MATCH_COST_USD

    people = event.get("people") or []
    if isinstance(people, str):
        try:
            people = json.loads(people)
        except json.JSONDecodeError:
            people = []
    entry = next(
        (e for e in people if isinstance(e, dict) and e.get("id") == pid), {},
    )
    # The validation match (phase 1) consumed an export credit too.
    cost_usd = APOLLO_MATCH_COST_USD + event["credits_consumed"] * APOLLO_CREDIT_COST_USD
    phones: List[Dict[str, Any]] = []
    for ph in (entry.get("phone_numbers") or []):
        if not (isinstance(ph, dict) and (ph.get("sanitized_number") or ph.get("raw_number"))):
            continue
        phones.append({
            "number": ph.get("sanitized_number") or ph.get("raw_number"),
            "type": ph.get("type_cd"),
            "confidence": ph.get("confidence_cd"),
            "status": ph.get("status_cd"),
            "dnc_status": ph.get("dnc_status_cd"),
        })
    if not phones:
        return {
            "matched": True,
            "found": False,
            "note": (
                "Apollo matched the person but has no phone number — fall "
                "back to fullenrich_enrich_phone"
            ),
        }, cost_usd
    return {
        "matched": True,
        "found": True,
        "person": {
            "name": p.get("name"),
            "title": p.get("title"),
            "organization": evidence or ((p.get("organization") or {}).get("name")),
        },
        "employer_check": "verified" if aligned else "unverifiable",
        "phones": phones,
        "_raw_payload": event,
    }, cost_usd


async def _google_maps_place_details(args: Dict[str, Any], ctx: ToolContext) -> Tuple[Dict[str, Any], float]:
    import httpx
    api_key = os.getenv("GOOGLE_API_KEY")
    place_id = args.get("place_id")
    if not (api_key and place_id):
        return {"error": "GOOGLE_API_KEY + place_id required"}, 0.0
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            r = await client.get(
                "https://maps.googleapis.com/maps/api/place/details/json",
                params={"key": api_key, "place_id": place_id},
            )
    except (httpx.ReadTimeout, httpx.ConnectTimeout, httpx.PoolTimeout) as e:
        return {"error": f"Google Maps place_details timeout: {e}"}, 0.0
    if r.status_code != 200:
        return {"error": f"Google Maps HTTP {r.status_code}"}, 0.0
    result = (r.json() or {}).get("result") or {}
    return result, 0.03


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
                        from dsl_worker.sources.apify_actor import _apify_run_cost_usd_from_data
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
                    from dsl_worker.sources.apify_actor import _abort_apify_run_and_get_cost
                    partial_usd = await asyncio.shield(
                        _abort_apify_run_and_get_cost(abort_client, api_key, apify_run_id)
                    )
                    if partial_usd > 0:
                        try:
                            ctx.partial_cost_usd = float(
                                getattr(ctx, "partial_cost_usd", 0.0)
                            ) + partial_usd
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
    return {"items": items[:5]}, cost_usd


from dsl_worker.chat.light_tools import (
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
                from dsl_worker.chat import run_state
                from dsl_api.models import ChatRun
                run_obj = ctx.db.query(ChatRun).filter(ChatRun.id == run_id).first()
                if run_obj is not None:
                    run_state.emit_event(ctx.db, run_obj, "tool_heartbeat", {
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


def _render_enrichment_skills_section() -> str:
    """Render the `# Skills` directory section for the research-tier cell agent.

    Lists every enrichment-applicable skill (name + description). Bodies are
    NOT included — loaded on demand via `load_skill`. Returns "" when no
    enrichment skills exist so the prompt stays clean.
    """
    from dsl_worker.skills import list_enrichment_skills
    skills = list_enrichment_skills()
    if not skills:
        return ""
    lines = [
        "# Skills",
        "",
        "A directory of documented playbooks for specific column-fill tasks. Not exhaustive — most cells won't match. If filling this column happens to match a skill below, load it for the optimized approach.",
        "",
        "Available:",
    ]
    for s in skills:
        lines.append(f"- **{s['name']}** — {s.get('description') or ''}")
    lines.append("")
    lines.append("Call `load_skill(name)` to read a playbook. Bodies are not loaded by default.")
    return "\n".join(lines)


async def _load_skill_cell(args: Dict[str, Any], ctx: ToolContext) -> Tuple[Dict[str, Any], float]:
    """Cell-agent variant of load_skill — returns the body of a named skill."""
    name = (args.get("name") or "").strip()
    if not name:
        return {"error": "name is required"}, 0.0
    from dsl_worker.skills import get_skill_body
    body = get_skill_body(name)
    if body is None:
        return {"error": f"unknown skill: {name}"}, 0.0
    return {"name": name, "body": body}, 0.0


# ── fetch_url: open ONE already-known page via Firecrawl, returned paged ──
# Deliberately narrow: a FALLBACK to read a specific URL the agent already has
# (a row field, or a result URL web_search surfaced) when the search snippet
# lacked the detail. NOT a search/discovery tool, NOT a default. ~50x cheaper
# than browser_use. Content is returned in ~9k-token pages so a huge/binary
# page can't dump millions of tokens into context.
_FIRECRAWL_SCRAPE_URL = "https://api.firecrawl.dev/v1/scrape"
FIRECRAWL_SCRAPE_COST_USD = float(os.getenv("FIRECRAWL_SCRAPE_COST_USD", "0.01"))
_FETCH_TIMEOUT = 45.0
_FETCH_PAGE_CHARS = 36_000        # ~9k tokens returned to the LLM per call
_FETCH_MAX_CHARS = 1_500_000      # hard cap on stored markdown (defends huge/binary)
# Cross-call scrape cache (url -> {markdown,title,truncated}): lets the agent
# page through one scrape without re-billing, and dedupes the same URL across
# cells. Bounded LRU so the long-lived worker doesn't leak.
_SCRAPE_CACHE: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
_SCRAPE_CACHE_MAX = 256


async def _fetch_url(args: Dict[str, Any], ctx: ToolContext) -> Tuple[Dict[str, Any], float]:
    """Open ONE already-known URL via Firecrawl and return its text, paged.

    Bills the scrape once (on the real fetch); paging the same URL is free.
    Returns an `error` (cost 0) for guessed/blank URLs, unsupported domains,
    and HTTP/timeout failures so the agent falls back to web_search.
    """
    url = (args.get("url") or "").strip()
    if not url.lower().startswith(("http://", "https://")):
        return {"error": "fetch_url needs a full http(s) URL you already have "
                "(a row field or a prior web_search result) — not a guess."}, 0.0
    try:
        page = max(1, int(args.get("page", 1)))
    except Exception:
        page = 1

    cost = 0.0
    entry = _SCRAPE_CACHE.get(url)
    if entry is None:
        key = os.getenv("FIRECRAWL_API_KEY")
        if not key:
            return {"error": "fetch_url unavailable (FIRECRAWL_API_KEY not set)"}, 0.0
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.post(
                    _FIRECRAWL_SCRAPE_URL,
                    headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                    json={"url": url, "formats": ["markdown"], "onlyMainContent": True,
                          "timeout": int(_FETCH_TIMEOUT * 1000)},
                    timeout=_FETCH_TIMEOUT + 5.0,
                )
        except Exception as e:
            return {"error": f"fetch failed ({type(e).__name__}) — try web_search instead", "url": url}, 0.0
        if resp.status_code == 403:
            return {"error": "this domain can't be scraped (Firecrawl refuses it) — "
                    "fall back to web_search or browser_use", "url": url}, 0.0
        if resp.status_code >= 400:
            return {"error": f"page fetch failed (HTTP {resp.status_code}) — try web_search", "url": url}, 0.0
        try:
            body = resp.json()
        except Exception:
            return {"error": "page returned no parseable content", "url": url}, 0.0
        if not body.get("success"):
            return {"error": (body.get("error") or "fetch_failed")[:160], "url": url}, 0.0
        data = body.get("data") or {}
        md = (data.get("markdown") or "").strip()
        title = ((data.get("metadata") or {}).get("title") or "").strip()
        if not md:
            return {"error": "page has no readable text (image / binary / JS-only) — try web_search",
                    "url": url, "title": title[:200]}, FIRECRAWL_SCRAPE_COST_USD
        truncated = len(md) > _FETCH_MAX_CHARS
        entry = {"markdown": md[:_FETCH_MAX_CHARS], "title": title, "truncated": truncated}
        _SCRAPE_CACHE[url] = entry
        _SCRAPE_CACHE.move_to_end(url)
        while len(_SCRAPE_CACHE) > _SCRAPE_CACHE_MAX:
            _SCRAPE_CACHE.popitem(last=False)
        cost = FIRECRAWL_SCRAPE_COST_USD
    else:
        _SCRAPE_CACHE.move_to_end(url)

    md = entry["markdown"]
    total_pages = max(1, (len(md) + _FETCH_PAGE_CHARS - 1) // _FETCH_PAGE_CHARS)
    page = min(page, total_pages)
    chunk = md[(page - 1) * _FETCH_PAGE_CHARS: page * _FETCH_PAGE_CHARS]
    out: Dict[str, Any] = {
        "url": url,
        "title": entry["title"][:200],
        "page": page,
        "total_pages": total_pages,
        "has_more": page < total_pages,
        "next_page": page + 1 if page < total_pages else None,
        "approx_tokens": len(chunk) // 4,
        "content": chunk,
    }
    if entry.get("truncated"):
        out["note"] = "page was very long; content truncated at the cap"
    return out, cost


# ---------------------------------------------------------------------------
# Field-preview composition — "row agent scales across projects/use-cases"
# ---------------------------------------------------------------------------
# Tool-tier cells (research/deep) get a PREVIEW of each row field (first N
# chars) instead of the full value, plus an `inspect_cell` tool to pull a
# field's full value on demand. This stops a 25k-char job Description from
# riding EVERY turn of a contact lookup that never needs it — the bloat that
# blew the 450k input-tokens/min cap. Lean rows (every field under the cap)
# are untouched: nothing is truncated, the tool isn't even offered.
#
# Default preview length ≈ a couple hundred chars (a few dozen words). Tunable
# via env; not a behavior toggle, just a knob.
def _preview_cap_chars() -> int:
    try:
        return max(40, int(os.getenv("ENRICHMENT_CELL_PREVIEW_CHARS", "280")))
    except (TypeError, ValueError):
        return 280


# Cap on what inspect_cell hands back in one call. The cell loop already
# clamps any tool result to 8k chars when feeding it to the model, so we
# mirror that here and label it, rather than letting the value get silently
# chopped mid-string.
_INSPECT_CELL_MAX_CHARS = 8000


def _compose_preview_fields(
    fields: Dict[str, Any], cap: int, store: Dict[str, Any]
) -> Dict[str, Any]:
    """Return a copy of `fields` with any over-cap value replaced by a preview
    string that names its column + full length; the full value is saved into
    `store` (keyed by field name) for inspect_cell. Never mutates the input.
    Small values (and non-string scalars under cap) pass through unchanged."""
    out: Dict[str, Any] = {}
    for k, v in fields.items():
        s = v if isinstance(v, str) else None
        if s is None:
            try:
                serialized = json.dumps(v, default=str)
            except Exception:
                serialized = str(v)
            # Only preview non-strings if they're genuinely large; keep small
            # scalars/objects typed so the agent can use them directly.
            if len(serialized) <= cap:
                out[k] = v
                continue
            s = serialized
        elif len(s) <= cap:
            out[k] = v
            continue
        # Over cap → store full, emit a marked preview.
        store[k] = v
        out[k] = (
            s[:cap].rstrip()
            + f"… [truncated — {len(s)} chars total; call inspect_cell(column=\"{k}\") for the full value]"
        )
    return out


async def _inspect_cell(args: Dict[str, Any], ctx: ToolContext) -> Tuple[Dict[str, Any], float]:
    """Return the FULL value of a row field that was shown truncated in the
    payload. Free — reads values already in memory (ctx.cell_full_fields)."""
    store = getattr(ctx, "cell_full_fields", None) or {}
    col = args.get("column") or args.get("field") or args.get("name")
    if not col or not isinstance(col, str):
        return {"error": "pass column=<exact field name shown with a [truncated] marker>"}, 0.0
    if col not in store:
        return {
            "error": f"no truncated field named {col!r} in this row",
            "available": sorted(store.keys()),
        }, 0.0
    val = store[col]
    s = val if isinstance(val, str) else json.dumps(val, default=str)
    if isinstance(s, str) and len(s) > _INSPECT_CELL_MAX_CHARS:
        return {
            "column": col,
            "value": s[:_INSPECT_CELL_MAX_CHARS],
            "truncated": True,
            "note": f"showing first {_INSPECT_CELL_MAX_CHARS} of {len(s)} chars",
        }, 0.0
    return {"column": col, "value": val}, 0.0


_INSPECT_CELL_TOOL_DEF: Dict[str, Any] = {
    "type": "function",
    "name": "inspect_cell",
    "description": (
        "Read the FULL value of a row field shown TRUNCATED in "
        "row_visible_to_user / row_hidden_source_fields (look for a "
        "'… [truncated …]' marker). Pass the exact field/column name. Free, "
        "instant. Use ONLY when the preview isn't enough to answer — most "
        "lookups (e.g. finding a contact for a company) never need the full "
        "long text of a description."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "column": {
                "type": "string",
                "description": "Exact field/column name to expand, as shown in the payload.",
            },
        },
        "required": ["column"],
        "additionalProperties": False,
    },
}


CELL_TOOL_HANDLERS: Dict[str, Callable[[Dict[str, Any], ToolContext], Awaitable[Tuple[Dict[str, Any], float]]]] = {
    "inspect_cell": _inspect_cell,
    "fullenrich_search_people": _fullenrich_search_people,
    "fullenrich_enrich_email": _fullenrich_enrich_email,
    "fullenrich_enrich_phone": _fullenrich_enrich_phone,
    "fullenrich_enrich_company": _fullenrich_enrich_company,
    "apollo_org_enrich": _apollo_org_enrich,
    "apollo_enrich_person": _apollo_enrich_person,
    "apollo_reveal_phone": _apollo_reveal_phone,
    "google_maps_place_details": _google_maps_place_details,
    "apify_search_actors": _apify_search_actors,
    "apify_actor_details": _apify_actor_details,
    "apify_call_actor": _apify_call_actor,
    # web_search is the OpenAI hosted tool — the model invokes it
    # server-side as part of its own Responses call. We don't dispatch
    # it ourselves; web_search_call items in the response output are
    # handled in the cell loop directly (billing + tool_calls_log).
    "fetch_url": _fetch_url,
    "browser_use": _browser_use,
    "code_exec": _code_exec,
    "load_skill": _load_skill_cell,
}


# ---------------------------------------------------------------------------
# Cell agent loop
# ---------------------------------------------------------------------------


CELL_SYSTEM_PROMPT = """You are a cell agent: fill specific columns for ONE row of a table.

# CRITICAL: end with `final_result`

Every turn MUST end with a `final_result({values: {...}})` call. No exceptions.

- If you found the value → call `final_result` with it.
- If the value doesn't exist or you give up → call `final_result` with `null` for that column. Null is a valid, expected outcome.
- If you ran out of ideas → call `final_result` with whatever you have (null for the rest).

Stopping without `final_result` means the cell stays blank in the user's table, your tool work + cost is wasted, and the user sees no signal about what happened. This is the single hardest rule in this prompt.

# Inputs (JSON)

- `row_visible_to_user` — the row's already-filled fields as shown in the user's table.
- `row_hidden_source_fields` (optional) — fields the source returned but the orchestrator didn't surface as columns. Use as extra context when reasoning. If the value you need is hiding here, return it via `final_result` for the column in `columns_to_fill`.
- `columns_to_fill` — column names you must produce values for.
- `instruction` — what to find or compute.

Long field values are shown as PREVIEWS that end with `… [truncated — N chars total; call inspect_cell(column="X") for the full value]`. The preview is usually all you need. If (and only if) the full text matters for your task, call `inspect_cell(column="<exact field name>")` to get it — it's free and instant. Don't expand fields you don't need.

# Finishing

End with `final_result({values: {col_name: value, ...}})` where `col_name` is the EXACT name from `columns_to_fill`. Do NOT invent keys like `label`, `value`, `answer`, `result`.

Set a column to `null` when the value genuinely doesn't exist. Null is fine. Don't fabricate.

For any URL / website / link value, return the FULL URL with its scheme — `https://example.com`, not `example.com` or `www.example.com`. A URL without `https://` won't render as a clickable link. Prepend `https://` if you only have a bare domain or a `www.` address.

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

# Picking a source

If you have no tools at all, the answer must come from `row_visible_to_user` + `row_hidden_source_fields` alone. Reason carefully and call `final_result` directly.

You have several sources. Pick the one that BEST FITS what the column wants — don't default to web_search if a structured source covers the type. Multiple sources can succeed; if your first pick comes back empty, fall to another.

## Sources by what the column wants

- **A person at a company** (founder, owner, decision-maker, manager) → `fullenrich_search_people`. LinkedIn-derived people DB — one call (limit=3, with a `titles=[...]` filter for the role you want) returns name + title + LinkedIn URL. See `find_person_at_company` skill for the filter recipe. For a person the row already NAMES, `apollo_enrich_person` (name + company/domain) is the one-call alternative — title + LinkedIn + often a verified email.
- **A verified email** for a known person (you have first_name + last_name + domain) → `apollo_enrich_person` FIRST (~$0.01; commit when `email_status='verified'`), then `fullenrich_enrich_email` when Apollo has no verified email — pass the `linkedin_url` Apollo returned (or the row's) to gate FE's deeper waterfall. If both return null, commit null — never pattern-guess `firstname@domain` as the answer. See `find_emails` skill.
- **A verified phone** → `apollo_reveal_phone` FIRST (~1 cr on success, $0 on miss; returns mobile + work numbers with confidence and do-not-call flags, ~10-30s). Fall back to `fullenrich_enrich_phone` (~6 cr) only when Apollo has no number. Both only when the column explicitly asks for phone. A company switchboard / toll-free main line is NEVER a person's phone — never commit one to a person-level phone column (org numbers belong only in explicitly company-level columns), and never commit a second mobile as someone's "work phone".
- **Company data** (revenue, headcount, funding, tech stack, location) → `apollo_org_enrich` (free on our plan) or `fullenrich_enrich_company`.
- **A local-business detail** (the row has a `place_id` from a prior Google Maps pull) → `google_maps_place_details`.
- **Posts / comments / listings on a specific platform** (Reddit, X, LinkedIn, Instagram, Hacker News, etc.) → `apify_call_actor` with a platform-specific actor. Discover via `apify_search_actors` → `apify_actor_details`. Bounded to `maxItems=5` at cell level.
- **Computation / parsing / regex** on existing row data → `code_exec`. Python sandbox, no network.
- **An arbitrary fact on a web page** not covered above (a one-off detail, news, an "About" page lookup) → `web_search`. Cheap and fast for static / server-rendered content. The catch-all when no structured source fits — not the default. Every call is billed: never issue an empty query, and never re-run a query (or a trivial rewording of one) that already returned nothing — change the angle or move on.
- **The content of a SPECIFIC page whose URL you already have** (a Listing/source URL already in the row, or a result URL `web_search` just surfaced) when the snippet didn't include the detail → `fetch_url`. Opens that one page and returns its text in ~9k-token chunks (pass `page` to read further). A FALLBACK after web_search, only for a URL you already have — never to search/discover, never on a guessed or constructed URL.

If the first-choice source returns empty or wrong, escalate: tighten / loosen the filter on the same source, then fall to `web_search`. If web_search surfaced the right page — or the row already carries a Listing/source URL — but the snippet lacked the detail, `fetch_url` that exact page to read it directly (cheap; the right move for "open this known page and extract X"). `browser_use` is a true last resort ($0.50+ per session) — only after web_search and fetch_url have failed, or for login walls / JS-only / antibot pages with no other access. Never pick `browser_use` predictively, and never call `fetch_url` on a URL you haven't already obtained.
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
                        "Map of column_name → list of source citations. "
                        "Two citation shapes, mix freely:\n"
                        "  (a) Web hit — {\"type\":\"url\",\"value\":\"https://...\"}. "
                        "Use for ANY page/URL you actually used (web_search "
                        "results, browser_use sessions). Cite the real URL, "
                        "never OpenAI's internal annotation pointers.\n"
                        "  (b) Paid/structured service — {\"source\":\"<tool>\"} "
                        "where `<tool>` is fullenrich_enrich_phone, "
                        "apollo_org_enrich, google_maps_place_details, "
                        "apify_call_actor, etc. No value/field needed.\n"
                        "Omit `sources` only when the value came purely from "
                        "reasoning over the row's existing fields."
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
    fr = _final_result_tool_def()
    defs: List[Dict[str, Any]] = [fr]
    if tier_cfg["tools"] != "all":
        return defs
    # Dossier (opt-in): let tool-tier cells record durable non-column data
    # points discovered for this row, replayed to later runs as `known`.
    if _dossier_enabled():
        fr["parameters"]["properties"]["notes"] = _DOSSIER_NOTES_PROP
    # web_search first in the list — reinforces the STEP-1 escalation
    # framing in the system prompt at the tool-picker.
    defs.append({"type": "web_search"})
    defs.extend(_CELL_TOOL_DEFS)
    # Research tier gets load_skill so it can read enrichment-scoped
    # playbooks listed in its system prompt under `# Skills`.
    defs.append({
        "type": "function",
        "name": "load_skill",
        "description": (
            "Load the playbook for a named skill from the directory listed "
            "under '# Skills' in the system prompt. Returns the full body. "
            "Call only when one of the listed skills clearly matches the "
            "current column-fill task."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Skill name as listed in '# Skills'"},
            },
            "required": ["name"],
            "additionalProperties": False,
        },
    })
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
        "name": "fullenrich_search_people",
        "description": (
            "Search FullEnrich (LinkedIn-derived people DB) for people at a "
            "specific company. Use when the column wants a *person* (founder, "
            "owner, decision-maker, manager) and the row only gives you a "
            "company name/domain. **Strictly better than web_search for B2B / "
            "white-collar targets**: structured results with title, seniority, "
            "linkedin URL, employer — no snippet parsing.\n\n"
            "**Cost: 0.25 FE-credit (~$0.013) per RETURNED person, not per call.** "
            "So `limit` IS the cost knob. Default limit=3 is the empirical sweet "
            "spot — captures the right person after the wrapper's employer-name "
            "post-filter drops cross-company noise. Bump to 5 only when picking "
            "from a real candidate pool. Hard-capped at 10.\n\n"
            "**Keep filters MINIMAL.** Start with `company_names` ALONE — adding "
            "titles/seniority/locations on the first call often returns 0. Strip "
            "corporate suffixes (Inc, LLC, Corp, Companies, Group) before passing "
            "company_names. Returns 0 → try `company_domains`. Both 0 → company "
            "not in FE's LinkedIn index (typically local businesses); fall to "
            "web_search.\n\n"
            "**Looking for a NAMED person** (the row already names them)? Pass the "
            "name in `person_names` plus a company filter. NEVER put a name in "
            "`titles` — `titles` means JOB titles only ('CEO', 'Director of "
            "Security'); a name there string-matches against job titles, returns "
            "0, and wastes the call.\n\n"
            "**FE often abbreviates last names** ('Arman Medar' is stored as "
            "'Arman M.', linkedin slug /in/arman-m-986331177), and a FULL-name "
            "`person_names` search does NOT match the abbreviated form (verified "
            "live). Recipe: full name + company first; on 0 results retry with "
            "FIRST NAME ONLY + company, then match the last name or last INITIAL "
            "yourself — same first name + matching initial + consistent "
            "title/location IS your person; use that entry's linkedin_url in "
            "enrich calls. Don't conclude 'not in FE' from a full-name miss."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "company_names": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Company name(s). Strip Inc/LLC/Corp/Companies/Group/Holdings before passing.",
                },
                "company_domains": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional. Domain fallback when company_names returns 0.",
                },
                "person_names": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "The person's name when looking for a specific NAMED person (fuzzy; combine with a company filter). Full name misses people FE stores with abbreviated last names ('Arman M.') — if full name returns 0, retry with FIRST NAME ONLY and match the last name/initial yourself in the results.",
                },
                "titles": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional. JOB titles only ('CEO', 'VP Sales') — NEVER a person's name. ONLY add as a second-call narrowing pass after a no-titles search returned >10.",
                },
                "locations": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional. City/region to narrow national parents to a footprint.",
                },
                "seniority": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional. C-Level, VP, Director, Manager, Senior, Entry. Rarely needed.",
                },
                "limit": {
                    "type": "integer",
                    "description": "Max people to return. Defaults to 3. Each result costs ~$0.05; keep small.",
                },
            },
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
            "**Pass `linkedin_url` whenever the row has it** — FullEnrich's "
            "waterfall is dramatically better with a LinkedIn URL; many "
            "contacts return empty without it but DELIVERABLE with it.\n\n"
            "**Response semantics:** the wrapper returns `{email, "
            "verification_status, commit}`. If `commit: true`, JUST COMMIT THE "
            "EMAIL — don't web_search to 'verify' it, don't second-guess the "
            "status. FullEnrich runs a multi-provider waterfall; if it surfaces "
            "an email with anything other than INVALID, it's usable (DELIVERABLE, "
            "HIGH_PROBABILITY, and CATCH_ALL all count). The model previously "
            "wasted budget on web_search trying to confirm HIGH_PROBABILITY "
            "results that were already the right answer.\n\n"
            "**If `commit: false`** (no email or INVALID status), do NOT pattern-"
            "guess `firstname@domain` and commit it as the answer. Those LOOK "
            "like emails but bounce. Commit null. One targeted web_search to find "
            "a VERIFIED email on a real page is fine before nulling.\n\n"
            "See the `find_emails` skill for fuller recipe."
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
            "Verified phone lookup via FullEnrich. EXPENSIVE — ~6 cr per "
            "successful match; the FALLBACK after apollo_reveal_phone found "
            "nothing. Only use when the column explicitly asks for a phone "
            "number. Same inputs as email; pass linkedin_url when the row "
            "has it for a better hit rate."
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
        "name": "apollo_enrich_person",
        "description": (
            "Apollo person match: name + company/domain in → title, WORK email "
            "(with verification status), LinkedIn URL, location out — one cheap "
            "synchronous call (~$0.01 on match, $0 on miss; ~270M-contact DB).\n\n"
            "**FIRST CHOICE for emails**: commit the email when "
            "`email_status='verified'`. Anything else (guessed / unavailable / "
            "missing) → fall to `fullenrich_enrich_email`, passing the "
            "linkedin_url Apollo returned (a match without a verified email "
            "still usually has the linkedin_url, which gates FE's deeper "
            "waterfall).\n\n"
            "Also a strong NAMED-person finder (alternative to "
            "fullenrich_search_people): one call confirms the person + title + "
            "LinkedIn. Stores full last names (no 'Arman M.' abbreviation "
            "problem). NO phones here — use apollo_reveal_phone for numbers."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "first_name": {"type": "string"},
                "last_name": {"type": "string"},
                "company": {"type": "string", "description": "Company/organization name."},
                "domain": {"type": "string", "description": "Company domain like 'anthropic.com'. Name + domain matches best."},
                "linkedin_url": {"type": "string", "description": "Person's LinkedIn URL — strongest single identifier when the row has it."},
                "email": {"type": "string", "description": "Reverse lookup: enrich a person from a known email."},
            },
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "apollo_reveal_phone",
        "description": (
            "Direct/mobile phone lookup via Apollo — FIRST CHOICE for phone "
            "columns (~1 cr when numbers come back, $0 on miss — vs ~6 cr "
            "for FullEnrich). Same inputs as apollo_enrich_person; also pass "
            "linkedin_url or email when the row has one for the strongest "
            "match. Blocks ~10-30s while Apollo's async reveal lands, then "
            "returns numbers with type (mobile / work_hq...), confidence, "
            "and do-not-call status. Prefer type='mobile' for direct-dial "
            "columns; treat any number with a non-null dnc_status as "
            "uncommittable. Multiple returned numbers are ALL candidates for "
            "the SAME person — never file an extra mobile as the person's "
            "work phone. Validates employer BEFORE revealing: wrong_person=true "
            "means Apollo's match works elsewhere and nothing was revealed — "
            "re-call with linkedin_url/email only if it's genuinely the same "
            "person after a job change. No numbers → fall back to "
            "fullenrich_enrich_phone."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "first_name": {"type": "string"},
                "last_name": {"type": "string"},
                "company": {"type": "string", "description": "Company/organization name."},
                "domain": {"type": "string", "description": "Company domain like 'anthropic.com'. Name + domain matches best."},
                "linkedin_url": {"type": "string", "description": "Person's LinkedIn URL — strongest single identifier when the row has it."},
                "email": {"type": "string", "description": "Known email — exact-match identifier when the row has one."},
            },
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
    {
        "type": "function",
        "name": "fetch_url",
        "description": (
            "Open ONE specific, already-known web page and read its text "
            "(Firecrawl -> markdown). Use ONLY when you already have the exact URL "
            "of a page that should hold the answer - a Listing/source URL in the "
            "row, or a result URL a previous web_search surfaced - and the search "
            "snippet didn't include the detail. FALLBACK after web_search, NOT a "
            "search/discovery tool: never guess or construct a URL, never browse "
            "for pages, never call it on a URL you haven't already obtained. "
            "Returns the page in ~9k-token chunks; if the answer isn't in the chunk "
            "and has_more is true, call again with next_page. Don't page past where "
            "the answer would plausibly be."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "Exact http(s) URL to open - one you already have (a row field or a prior web_search result), never guessed."},
                "page": {"type": "integer", "description": "Which ~9k-token chunk to return (1-indexed, default 1). Use the returned next_page to read further."},
            },
            "required": ["url"],
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


# --- URL formatting: make scheme-less URLs clickable in the FE -------------
# The table FE only linkifies values matching ^https?://… (that's what shows
# the open-in-new-tab icon). An agent that returns "www.example.com" (no
# scheme) renders as plain, non-clickable text. Normalize the agent's output
# so those become real links — WITHOUT risking non-URL cells.
def _normalize_url_value(v: Any) -> Any:
    """Prepend https:// to an unambiguous scheme-less URL. CONSERVATIVE BY
    DESIGN: only touches a value that (case-insensitively) starts with 'www.',
    has a dot after it, no scheme, and no whitespace — nothing but a web
    address looks like that. Everything else is returned UNCHANGED:
    already-schemed URLs, mailto:/tel:, bare words, 'N/A', numbers, multi-word
    text, domains WITHOUT 'www.', and non-strings. So it cannot corrupt a
    non-URL value. Bare domains (example.com) are intentionally left to the
    system-prompt nudge — auto-detecting them risks false positives on things
    like 'v3.2' or 'Node.js'."""
    if not isinstance(v, str):
        return v
    s = v.strip()
    if not s or any(ws in s for ws in (" ", "\t", "\n", "\r")):
        return v
    low = s.lower()
    if "://" in s or low.startswith("mailto:") or low.startswith("tel:"):
        return v
    if low.startswith("www.") and "." in s[4:]:
        return "https://" + s
    return v


def _normalize_url_values(d: Any) -> Any:
    """Map _normalize_url_value over a final_result values dict (no-op for
    non-dicts)."""
    if not isinstance(d, dict):
        return d
    return {k: _normalize_url_value(v) for k, v in d.items()}


def _coerce_value_keys(
    raw: Dict[str, Any],
    columns_to_fill: List[str],
) -> Dict[str, Any]:
    """Remap final_result keys onto columns_to_fill, THEN normalize URL values
    so scheme-less links (www.x.com) render clickable. Thin wrapper over
    _coerce_value_keys_inner so every final_result path gets both."""
    return _normalize_url_values(_coerce_value_keys_inner(raw, columns_to_fill))


def _coerce_value_keys_inner(
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

# Map raw tool names → canonical sources kinds. The FE's sourceDisplay()
# only renders nice labels + favicons for canonical kinds; raw tool names
# fall through to ugly "fullenrich enrich phone" text. Anything not in this
# map passes through unchanged.
_TOOL_TO_SOURCE_KIND: Dict[str, str] = {
    "fullenrich_enrich_phone": "fullenrich_people",
    "fullenrich_enrich_email": "fullenrich_people",
    "fullenrich_enrich_company": "fullenrich_people",
    "apollo_org_enrich": "apollo_companies",
    "apollo_enrich_person": "apollo_people",
    "apollo_reveal_phone": "apollo_people",
    "google_maps_place_details": "google_maps",
    "browser_use": "browser_use",
    "apify_call_actor": "apify_actor",
    "web_search": "web_search",
}


def _normalize_tool_to_kind(tool_name: str) -> str:
    """Map a cell-agent tool name to a canonical sources kind for FE display."""
    return _TOOL_TO_SOURCE_KIND.get(tool_name, tool_name)


# Paid-provider tools whose FULL response payload is worth surfacing to the
# user (they paid for it): captured on the tool_calls_log entry as
# `result_full`, attached to source_record citations as `payload`, and
# rendered by the FE behind the source chip. Handlers attach the provider's
# RAW response under `_raw_payload` (popped at the capture site BEFORE the
# result is serialized for the model) — so the chip shows everything the
# provider returned, not the slimmed dict the agent saw. Bounded responses
# only — apify and browser_use can return megabytes, so they stay
# preview-only.
PROVIDER_PAYLOAD_TOOLS = {
    "apollo_enrich_person",
    "apollo_reveal_phone",
    "apollo_org_enrich",
    "fullenrich_enrich_email",
    "fullenrich_enrich_phone",
    "fullenrich_enrich_company",
    "fullenrich_search_people",
    "google_maps_place_details",
}

# Hard cap on a captured payload's serialized size — protects samples.tags
# from pathological provider responses. Generous because raw provider
# payloads (full Apollo person objects with employment history, FE
# contact_info with all candidates) are the point of the feature; typical
# responses are 2-30KB.
_PAYLOAD_CAP_CHARS = 50_000


def _capture_payload(tool_result: Any) -> Any:
    """Size-capped copy of a provider response for citation attachment."""
    try:
        s = json.dumps(tool_result, default=str)
    except Exception:
        return None
    if len(s) <= _PAYLOAD_CAP_CHARS:
        return tool_result
    return {"_truncated": True, "preview": s[:_PAYLOAD_CAP_CHARS]}


def _last_payload_for(
    tool_calls_log: Optional[List[Dict[str, Any]]],
    src_name: str,
    kind: str,
) -> Any:
    """Most recent captured payload for a cited tool (by raw name or kind)."""
    for entry in reversed(tool_calls_log or []):
        if entry.get("result_full") is None:
            continue
        name = entry.get("name") or ""
        if name == src_name or _normalize_tool_to_kind(name) == kind:
            return entry["result_full"]
    return None


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
        citation: Dict[str, Any] = {"type": "source_record", "source": kind}
        if entry.get("result_full") is not None:
            citation["payload"] = entry["result_full"]
        return citation
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
                citation: Dict[str, Any] = {
                    "type": "source_record",
                    "source": kind,
                }
                # Attach the provider's full response so the FE can show
                # the raw payload behind the chip — the user paid for it.
                payload = _last_payload_for(tool_calls_log, str(src), kind)
                if payload is not None:
                    citation["payload"] = payload
                normalized.append(citation)
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


# ---------------------------------------------------------------------------
# Anthropic (Claude) cell-agent path — used by the research/deep tiers.
# ---------------------------------------------------------------------------
#
# Reuses everything provider-agnostic (CELL_TOOL_HANDLERS, budget gates,
# value/source coercion, cell_traces). Only the model loop differs. Wins over
# the OpenAI hosted web_search the rest of this module used to use:
#   - max_uses caps web searches SERVER-SIDE — no mid-stream abort hack, and
#     the "model reformulates the same query 7x into the cap" failure mode is
#     structurally impossible.
#   - usage.server_tool_use.web_search_requests is the EXACT search count, so
#     we bill the real number ($0.01/search) instead of a calibrated estimate.
#   - Claude returns real citation URLs (better source attribution).
#   - The static system+tools prefix is prompt-cached, so cell 2..N of a job
#     read it from cache (~10x cheaper input) instead of re-billing it.

# Anthropic hosted web_search fee: $10 / 1k searches. The exact count comes
# back in usage, so this is the real rate (no sub-search multiplier guess).
ANTHROPIC_WEB_SEARCH_COST_USD = 0.01
# Pin the SIMPLE hosted web_search tool. translate_tools() emits the newer
# agentic web_search_20260209, which bundles server-side code_execution — it
# spins up a container that must be threaded via container_id across turns, and
# replaying those blocks on a follow-up call otherwise 400s. Factual cell fills
# only need plain search, so we pin the simpler (and cheaper) version.
ANTHROPIC_WEB_SEARCH_TOOL = "web_search_20250305"
# Hard ceiling on web searches per cell, enforced server-side via the tool's
# max_uses. Derived from the per-row cap but clamped to this so one hard-to-
# find value can't burn the whole budget on search reformulations.
ANTHROPIC_MAX_WEB_SEARCHES = 5
# Transient-failure retries for the per-cell Messages call. A concurrent
# enrichment batch (25 research agents) on heavy job-board rows blows the org
# ITPM cap (450k input tok/min) — a SUSTAINED throttle, not a momentary 529 —
# so we retry generously and honor Retry-After to ride the burst out instead
# of giving up (a blank cell is the worst outcome). A retrying cell holds its
# semaphore slot while it backs off, which self-throttles the batch — fewer
# new calls fire, so the limit clears. Retries wrap ONE messages_create call,
# so they don't consume a reasoning turn; the 900s per-cell hard timeout is
# the real ceiling.
ANTHROPIC_CELL_MAX_RETRIES = 8

_cell_anthropic_client = None  # lazily-built TrackedAnthropicClient


def _get_anthropic_client():
    global _cell_anthropic_client
    if _cell_anthropic_client is None:
        from anthropic import AsyncAnthropic
        from dsl_worker.billing.tracked_anthropic_client import TrackedAnthropicClient
        raw = AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        _cell_anthropic_client = TrackedAnthropicClient(raw)
    return _cell_anthropic_client


# Block types that accept a cache_control marker. Server-injected blocks
# (web_search_tool_result, server_tool_use, thinking) don't — skip them and
# mark the nearest earlier cacheable block instead.
_CACHEABLE_BLOCK_TYPES = {"text", "tool_result", "tool_use", "image", "document"}


def _slide_cache_breakpoint(messages: List[Dict[str, Any]]) -> None:
    """Move the conversation cache breakpoint to the newest message.

    Anthropic caches longest-matching PREFIXES up to a cache_control
    marker. The static prefix (tools + system) has its own permanent
    breakpoint; this one slides along the conversation so turn N reads
    turns 1..N-1 from cache and only the newest delta counts toward
    ITPM (cache reads are ITPM-free on the 4.x models and bill at 10%;
    writes bill at 125% once). Without it, every turn re-sends the whole
    accumulated history — payload + every tool result — as fresh input:
    a 25-agent research batch sustains >450k ITPM and 429-kills cells
    (proj 274174f9, 2026-06-10: 13 cells dead after 8 retries each).

    Mutates `messages` in place: strips any previous marker, then marks
    the last cacheable block of the last message. String content is
    converted to a single text block so it can carry the marker.
    """
    for m in messages:
        c = m.get("content")
        if isinstance(c, list):
            for b in c:
                if isinstance(b, dict):
                    b.pop("cache_control", None)
    for m in reversed(messages):
        c = m.get("content")
        if isinstance(c, str):
            if not c:
                continue
            m["content"] = [{"type": "text", "text": c}]
            c = m["content"]
        if not isinstance(c, list):
            continue
        for b in reversed(c):
            if isinstance(b, dict) and b.get("type") in _CACHEABLE_BLOCK_TYPES:
                b["cache_control"] = {"type": "ephemeral"}
                return
    # No cacheable block anywhere — leave unmarked; the request is still valid.


async def _dispatch_cell_tool(
    name: str,
    args: Dict[str, Any],
    ctx: ToolContext,
    tier_cfg: Dict[str, Any],
    total_cost: float,
) -> Tuple[Dict[str, Any], float]:
    """Client-tool dispatch with the same pre-call budget gate the OpenAI path
    uses (FIXED_COST_TOOLS / browser_use / apify). web_search is NOT here — on
    Claude it's a server tool that runs inline, billed via usage."""
    handler = CELL_TOOL_HANDLERS.get(name)
    if not handler:
        return {"error": f"unknown tool {name}"}, 0.0
    remaining = tier_cfg["cap"] - total_cost
    skip_reason: Optional[str] = None
    if name in FIXED_COST_TOOLS:
        est = FIXED_COST_TOOLS[name]
        if remaining < est:
            skip_reason = (
                f"skipped: {name} costs ~${est} but only ${remaining:.3f} "
                f"remaining of per-row cap"
            )
    elif name == "browser_use":
        if remaining < BU_MIN_BUDGET:
            skip_reason = (
                f"skipped: browser_use needs at least ${BU_MIN_BUDGET} "
                f"remaining; only ${remaining:.3f} left"
            )
        else:
            args["__max_cost_usd"] = float(remaining)
    elif name == "apify_call_actor":
        if remaining < APIFY_MIN_BUDGET:
            skip_reason = (
                f"skipped: apify_call_actor needs at least ${APIFY_MIN_BUDGET} "
                f"remaining; only ${remaining:.3f} left"
            )
        else:
            args["__max_cost_usd"] = float(remaining)
    if skip_reason is not None:
        log.info("cell agent pre-tool skip: %s", skip_reason)
        return {"error": "budget", "message": skip_reason}, 0.0
    try:
        return await handler(args, ctx)
    except Exception as e:
        log.exception("cell tool %s raised: %s", name, e)
        return {"error": str(e)[:300]}, 0.0


def _anthropic_web_search_count(response: Any) -> int:
    """Exact number of hosted web searches Claude ran this turn (from usage)."""
    usage = getattr(response, "usage", None)
    stu = getattr(usage, "server_tool_use", None) if usage else None
    if not stu:
        return 0
    return int(getattr(stu, "web_search_requests", 0) or 0)


def _log_anthropic_web_searches(response: Any, tool_calls_log: List[Dict[str, Any]]) -> None:
    """Record each Claude web_search (query + result URLs) into tool_calls_log
    so source inference and the cell_traces row see them — same shape the
    OpenAI path logged."""
    pending_idx: Optional[int] = None
    for b in getattr(response, "content", []) or []:
        bt = getattr(b, "type", None)
        if bt == "server_tool_use" and getattr(b, "name", None) == "web_search":
            inp = getattr(b, "input", None) or {}
            query = inp.get("query", "") if isinstance(inp, dict) else ""
            tool_calls_log.append({
                "name": "web_search",
                "args": {"query": query},
                "result_preview": "anthropic web_search",
                "cost": ANTHROPIC_WEB_SEARCH_COST_USD,
            })
            pending_idx = len(tool_calls_log) - 1
        elif bt == "web_search_tool_result" and pending_idx is not None:
            urls: List[str] = []
            rc = getattr(b, "content", None)
            if isinstance(rc, list):
                for r in rc:
                    u = getattr(r, "url", None)
                    if u:
                        urls.append(u)
            if urls:
                tool_calls_log[pending_idx]["result_preview"] = json.dumps(
                    {"urls": urls[:5]}, default=str
                )[:400]
            pending_idx = None


def _serialize_blocks_for_history(content: List[Any]) -> List[Dict[str, Any]]:
    """Dump Claude response content blocks to dicts so they replay verbatim as
    assistant history (and stay JSON-serializable for caching)."""
    out: List[Dict[str, Any]] = []
    for b in content or []:
        try:
            out.append(b.model_dump(exclude_none=True))
        except Exception:
            if getattr(b, "type", None) == "text":
                out.append({"type": "text", "text": getattr(b, "text", "")})
    return out


# Caching policy: cache ONLY the static tools+system prefix — one breakpoint on
# the system block (see _anthropic_cell_loop). That prefix is identical for every
# cell of a tier, so it's written once and read by all subsequent cells within
# the 5-min TTL. We deliberately do NOT put cache breakpoints on the per-cell
# content (row data, web-search results, conversation): it's unique per cell and
# never reused, so caching it would only pay the 1.25x write premium for nothing.
# (Anthropic's hosted web_search may still auto-cache its own result blocks
# server-side — that's outside our control and unaffected by this.)


async def _anthropic_cell_loop(
    tier_cfg: Dict[str, Any],
    system_prompt: str,
    user_payload: Dict[str, Any],
    columns_to_fill: List[str],
    openai_tool_defs: List[Dict[str, Any]],
    ctx: ToolContext,
    tool_calls_log: List[Dict[str, Any]],
    build_sources_fn: Callable[..., Dict[str, List[Dict[str, Any]]]],
) -> Tuple[Dict[str, Any], Dict[str, List[Dict[str, Any]]], float, str, Optional[str]]:
    """Claude (Messages API) equivalent of the OpenAI cell loop.

    Returns (final_values, final_sources, total_cost_usd, status, error_str).
    status ∈ {"filled","hit_budget","error"} — same contract as the OpenAI path.
    Extended thinking is intentionally off: keeps caching simple (no thinking-
    signature round-trip) and Sonnet 4.6 is strong enough for factual fills.
    """
    from dsl_worker.agents.anthropic_base import translate_tools

    client = _get_anthropic_client()
    cap_usd = tier_cfg["cap"]

    claude_tools, _mcp = translate_tools(openai_tool_defs)
    # Cap hosted web searches server-side. Derived from the per-row budget,
    # clamped to ANTHROPIC_MAX_WEB_SEARCHES.
    max_uses = max(1, min(ANTHROPIC_MAX_WEB_SEARCHES, int(cap_usd / ANTHROPIC_WEB_SEARCH_COST_USD)))
    for t in claude_tools:
        if isinstance(t, dict) and str(t.get("type", "")).startswith("web_search"):
            t["type"] = ANTHROPIC_WEB_SEARCH_TOOL
            t["max_uses"] = max_uses

    # Cache the static prefix (tools + system). A breakpoint on the system
    # block caches everything before it too (Anthropic order: tools → system →
    # messages). Identical across every cell of this tier → cell 2..N read it
    # from cache. Per-cell user data sits AFTER the breakpoint, so it varies
    # freely without busting the cache.
    system_blocks = [{
        "type": "text",
        "text": system_prompt,
        "cache_control": {"type": "ephemeral"},
    }]

    messages: List[Dict[str, Any]] = [
        {"role": "user", "content": json.dumps(user_payload, default=str)},
    ]

    final_values: Dict[str, Any] = {}
    final_sources: Dict[str, List[Dict[str, Any]]] = {}
    total_cost = 0.0
    HARD_TURN_LIMIT = 40

    for _turn in range(HARD_TURN_LIMIT):
        if total_cost >= cap_usd:
            return final_values, final_sources, total_cost, "hit_budget", "budget cap reached"

        # Slide the conversation cache breakpoint onto the newest message
        # so this turn reads the prior history from cache instead of
        # re-paying it as fresh ITPM. Once per turn — retries below resend
        # the identical request, so the marker stays put.
        _slide_cache_breakpoint(messages)

        # Retry transient overload/rate-limit/timeout before giving up. A
        # concurrent contact batch 429'd Anthropic within ~7s and, with no
        # retry here, blanked 145/246 cells on proj 1a3f68bc. Backoff +
        # jitter de-correlates the herd so the batch drains instead of
        # collapsing. Wraps a single call → does NOT consume a turn.
        response = None
        usage_cost = None
        llm_retries = 0
        while True:
            try:
                response, usage_cost = await client.messages_create(
                    model=tier_cfg["model"],
                    system=system_blocks,
                    messages=messages,
                    tools=claude_tools,
                    max_tokens=8000,
                )
                break
            except Exception as e:
                msg = str(e).lower()
                code = getattr(e, "status_code", None) or getattr(
                    getattr(e, "response", None), "status_code", None
                )
                transient = code in (408, 409, 425, 429, 500, 502, 503, 529) or any(
                    t in msg for t in (
                        "overloaded", "rate limit", "rate_limit", "429",
                        "503", "529", "timed out", "timeout", "connection",
                        "temporarily", "service unavailable",
                    )
                )
                if transient and llm_retries < ANTHROPIC_CELL_MAX_RETRIES:
                    llm_retries += 1
                    backoff = min(8.0, 0.75 * (2 ** llm_retries)) + random.uniform(0, 0.75)
                    # The observed failure is a 429 org ITPM cap (450k input
                    # tokens/min on the research model), not a momentary 529.
                    # That's a sustained throttle — honor Anthropic's
                    # Retry-After (seconds) so we wait for the bucket to
                    # refill instead of undershooting with exp-backoff (cap 60s
                    # below; the 900s per-cell hard timeout is the real ceiling).
                    resp = getattr(e, "response", None)
                    hdrs = getattr(resp, "headers", None)
                    if hdrs:
                        try:
                            ra = hdrs.get("retry-after")
                            if ra is not None:
                                # Cap at 60s: ITPM windows are per-minute, so a
                                # full-minute wait is the most that ever helps,
                                # and it stays well under the 900s cell timeout.
                                backoff = min(60.0, max(backoff, float(ra) + random.uniform(0, 1.0)))
                        except (TypeError, ValueError):
                            pass
                    log.info(
                        "anthropic cell call transient err (tier=%s retry %d/%d in %.1fs): %s",
                        tier_cfg["name"], llm_retries, ANTHROPIC_CELL_MAX_RETRIES, backoff, e,
                    )
                    await asyncio.sleep(backoff)
                    continue
                log.warning("anthropic cell agent call failed (tier=%s): %s", tier_cfg["name"], e)
                return final_values, final_sources, total_cost, "error", f"LLM call failed: {e}"[:500]

        # Token cost (incl. cache read/write) from the tracked client; add the
        # hosted web_search fee using the exact count Anthropic reports.
        total_cost += usage_cost.total_cost_usd
        n_ws = _anthropic_web_search_count(response)
        if n_ws:
            total_cost += n_ws * ANTHROPIC_WEB_SEARCH_COST_USD
            _log_anthropic_web_searches(response, tool_calls_log)

        messages.append({
            "role": "assistant",
            "content": _serialize_blocks_for_history(response.content),
        })

        tool_use_blocks = [b for b in response.content if getattr(b, "type", None) == "tool_use"]
        if not tool_use_blocks:
            # Web search can pause a long turn (stop_reason="pause_turn"); resume
            # it with no extra user message so the model can finish, rather than
            # treating the pause as a (premature) final answer.
            if getattr(response, "stop_reason", None) == "pause_turn":
                continue
            # No client tool call — the model answered in text. Mirror the
            # OpenAI fallback: parse a JSON object/{"values":...} out of it.
            text = "".join(
                getattr(b, "text", "") for b in response.content
                if getattr(b, "type", None) == "text"
            ).strip()
            if text:
                try:
                    data = json.loads(text)
                except json.JSONDecodeError:
                    data = None
                if isinstance(data, dict) and "values" in data:
                    final_values = data["values"]
                    final_sources = build_sources_fn(
                        _coerce_sources_keys(data.get("sources"), list(final_values.keys()), tool_calls_log),
                        final_values,
                    )
                    return final_values, final_sources, total_cost, "filled", None
                if isinstance(data, dict):
                    final_values = data
                    final_sources = build_sources_fn({}, final_values)
                    return final_values, final_sources, total_cost, "filled", None
            return final_values, final_sources, total_cost, "error", "no tool_use and no parseable message"

        tool_result_blocks: List[Dict[str, Any]] = []
        for tu in tool_use_blocks:
            name = tu.name
            args = tu.input if isinstance(tu.input, dict) else {}

            if name == "final_result":
                raw_values = args.get("values") if isinstance(args.get("values"), dict) else (args if isinstance(args, dict) else {})
                final_values = _coerce_value_keys(raw_values, columns_to_fill)
                declared = _coerce_sources_keys(args.get("sources"), list(final_values.keys()), tool_calls_log)
                final_sources = build_sources_fn(declared, final_values)
                tool_calls_log.append({
                    "name": "final_result",
                    "args": args,
                    "coerced_values": final_values,
                    "coerced_sources": final_sources,
                    "cost": 0.0,
                })
                return final_values, final_sources, total_cost, "filled", None

            tool_result, tool_cost = await _dispatch_cell_tool(name, args, ctx, tier_cfg, total_cost)
            total_cost += tool_cost
            # Raw provider response rides out of handlers under _raw_payload —
            # pop it BEFORE preview/model serialization so only the payload
            # capture (never the LLM) sees the full blob.
            raw_payload = tool_result.pop("_raw_payload", None) if isinstance(tool_result, dict) else None
            _log_entry: Dict[str, Any] = {
                "name": name,
                "args": args,
                "result_preview": json.dumps(tool_result, default=str)[:400],
                "cost": tool_cost,
            }
            if name in PROVIDER_PAYLOAD_TOOLS and isinstance(tool_result, dict) and "error" not in tool_result:
                _log_entry["result_full"] = _capture_payload(raw_payload if raw_payload is not None else tool_result)
            tool_calls_log.append(_log_entry)
            tool_result_blocks.append({
                "type": "tool_result",
                "tool_use_id": tu.id,
                "content": json.dumps(tool_result, default=str)[:8000],
            })
            if total_cost >= cap_usd:
                return final_values, final_sources, total_cost, "hit_budget", "budget cap reached before final_result"

        messages.append({"role": "user", "content": tool_result_blocks})

    return final_values, final_sources, total_cost, "error", f"hit HARD_TURN_LIMIT={HARD_TURN_LIMIT}"


# ---------------------------------------------------------------------------
# Row dossier — persistent, per-row research memory (verbatim, no summaries).
# ---------------------------------------------------------------------------
#
# Every cell run for a row appends what it LEARNED (facts: filled column values
# + agent-declared intermediate data points, each with provenance) and what it
# TRIED-AND-FAILED (dead-ends: search queries / tool calls that found nothing)
# into samples.tags.dossier. The next run for that row reads it back VERBATIM as
#   `known`         — reuse, don't re-research
#   `already_tried` — don't repeat a losing approach
# Nothing is LLM-summarized: values are stored and replayed exactly as found, so
# there is no path from the dossier to a hallucinated value.
#
# Toggle: ENRICHMENT_ROW_DOSSIER (default OFF → behavior is exactly as before).
# Storage: samples.tags->'dossier' JSONB — no migration (mirrors tags->'sources').

_DOSSIER_MAX_TRIED = 40  # cap dead-end list growth; keep the most recent

_DOSSIER_NOTES_PROP: Dict[str, Any] = {
    "type": "object",
    "description": (
        "Optional. Durable data points you discovered about THIS row that "
        "aren't target columns but are worth remembering for a future fill of "
        "this same row — e.g. {\"company_domain\": \"acme.com\", "
        "\"founder_linkedin\": \"https://linkedin.com/in/...\"}. Stored verbatim "
        "and shown to later runs of this row as `known`, so they skip "
        "re-researching it. Keys are short snake_case labels; values are the "
        "literal data."
    ),
}

_DOSSIER_SYSTEM_SECTION = """# Row memory
This row may include two extra fields:
- `known` — facts already established for THIS row on earlier runs (filled column values AND intermediate data points like a discovered domain), each with its source. Trust and REUSE them: if a value you need is already in `known`, return it directly instead of searching again.
- `already_tried` — searches/tools that already failed for this row. Do NOT repeat them; take a different angle, or conclude the data isn't available.
When you find a durable data point that isn't a target column but could help a future fill of this row (a domain, an HQ city, a LinkedIn URL, an external id), pass it in final_result `notes` as {short_label: value}, verbatim."""


def _dossier_enabled() -> bool:
    # On by default. Optional kill switch: ENRICHMENT_ROW_DOSSIER=off.
    return os.getenv("ENRICHMENT_ROW_DOSSIER", "on").strip().lower() in (
        "1", "true", "yes", "on",
    )


def _load_row_dossier(sample_id: Optional[str]) -> Dict[str, Any]:
    """Read samples.tags->'dossier' for one row on its OWN short-lived session,
    so it never touches the caller's transaction (the inline /run path shares a
    single request session across rows; the jobs path has its own). Best-effort
    → {} on any miss.
    """
    if not sample_id:
        return {}
    from dsl_api.db import SessionLocal
    db = SessionLocal()
    try:
        cur = db.execute(
            sa_text("SELECT tags->'dossier' FROM samples WHERE id=:sid"),
            {"sid": sample_id},
        ).fetchone()
    except Exception as e:
        log.warning("dossier load failed (sample %s): %s", sample_id, e)
        return {}
    finally:
        db.close()
    d = cur[0] if cur else None
    if isinstance(d, str):
        try:
            d = json.loads(d)
        except json.JSONDecodeError:
            d = None
    return d if isinstance(d, dict) else {}


def _dossier_payload_views(
    dossier: Dict[str, Any],
) -> Tuple[Dict[str, Any], List[str]]:
    """Build the verbatim views injected into the cell agent's user payload.

    `known`        — {key: {value, source?}} of facts already established for
                     this row (column values AND intermediate data points).
    `already_tried`— short strings describing approaches that already failed,
                     so the agent doesn't burn budget repeating them.
    """
    facts = dossier.get("facts") if isinstance(dossier.get("facts"), dict) else {}
    known: Dict[str, Any] = {}
    for key, rec in facts.items():
        if not isinstance(rec, dict):
            continue
        val = rec.get("value")
        if val in (None, ""):
            continue
        view: Dict[str, Any] = {"value": val}
        if rec.get("source"):
            view["source"] = rec["source"]
        known[key] = view
    tried_raw = dossier.get("tried") if isinstance(dossier.get("tried"), list) else []
    already_tried: List[str] = []
    for t in tried_raw:
        if isinstance(t, dict) and t.get("q"):
            label = str(t["q"])
            if t.get("outcome"):
                label = f"{label} → {t['outcome']}"
            already_tried.append(label)
        elif isinstance(t, str) and t:
            already_tried.append(t)
    return known, already_tried


def _persist_cell_dossier(
    sample_id: Optional[str],
    columns_to_fill: List[str],
    final_values: Dict[str, Any],
    final_sources: Dict[str, List[Dict[str, Any]]],
    tool_calls_log: List[Dict[str, Any]],
) -> None:
    """Append this run's findings + dead-ends to samples.tags.dossier on its OWN
    short-lived session — never touches the caller's transaction (the inline
    /run path shares one request session across rows). Best-effort, never
    raises. Read-modify-write under the per-sample advisory lock so two runs on
    the same row don't clobber each other. Values stored verbatim.
    """
    if not sample_id:
        return
    from datetime import datetime, timezone
    now_iso = datetime.now(timezone.utc).isoformat()

    # Facts: filled column values (with provenance) + agent-declared notes.
    new_facts: Dict[str, Any] = {}
    if isinstance(final_values, dict):
        for col, val in final_values.items():
            if val in (None, ""):
                continue
            cites = final_sources.get(col) if isinstance(final_sources, dict) else None
            src = cites[0] if isinstance(cites, list) and cites else None
            # Strip the raw provider payload from the dossier copy — facts get
            # replayed verbatim into future agent prompts, and a multi-KB
            # payload there would bloat every later run's input.
            if isinstance(src, dict) and "payload" in src:
                src = {k: v for k, v in src.items() if k != "payload"}
            new_facts[col] = {"value": val, "source": src, "at": now_iso}
    for entry in reversed(tool_calls_log):
        if entry.get("name") == "final_result":
            notes = (entry.get("args") or {}).get("notes")
            if isinstance(notes, dict):
                for k, v in notes.items():
                    if isinstance(k, str) and v not in (None, ""):
                        new_facts.setdefault(k, {"value": v, "source": None, "at": now_iso})
            break

    # Dead-ends: only when the run left target columns unfilled. Record the
    # search queries / clearly-failed tool calls so a retry won't repeat them.
    unfilled = [
        c for c in (columns_to_fill or [])
        if not (isinstance(final_values, dict) and final_values.get(c) not in (None, ""))
    ]
    new_tried: List[Dict[str, Any]] = []
    if unfilled:
        for entry in tool_calls_log:
            name = entry.get("name")
            args = entry.get("args") or {}
            if name == "web_search":
                q = args.get("query")
                if q:
                    new_tried.append({"q": f"web_search: {q}", "at": now_iso})
            elif name in ("final_result", "load_skill", "code_exec", None):
                continue
            else:
                preview = str(entry.get("result_preview") or "")
                failed = (
                    '"error"' in preview
                    or "not_found" in preview
                    or preview.strip() in ("", "[]", "{}", "null")
                )
                if failed:
                    hint = args.get("url") or args.get("actor_id") or args.get("domain") or ""
                    new_tried.append(
                        {"q": f"{name} {hint}".strip(), "outcome": "no_result", "at": now_iso}
                    )

    if not (new_facts or new_tried):
        return

    from dsl_api.db import SessionLocal
    db = SessionLocal()
    try:
        db.execute(
            sa_text("SELECT pg_advisory_xact_lock(hashtextextended(:sid, 0))"),
            {"sid": str(sample_id)},
        )
        cur = db.execute(
            sa_text("SELECT tags->'dossier' FROM samples WHERE id=:sid"),
            {"sid": sample_id},
        ).fetchone()
        existing = cur[0] if cur else None
        if isinstance(existing, str):
            try:
                existing = json.loads(existing or "{}")
            except json.JSONDecodeError:
                existing = {}
        if not isinstance(existing, dict):
            existing = {}

        facts = dict(existing["facts"]) if isinstance(existing.get("facts"), dict) else {}
        facts.update(new_facts)  # latest non-null value wins

        prior_tried = existing.get("tried") if isinstance(existing.get("tried"), list) else []
        combined = [t for t in prior_tried if isinstance(t, dict)] + new_tried
        seen: set = set()
        deduped: List[Dict[str, Any]] = []
        for t in reversed(combined):  # newest first → keep the most-recent dup
            q = t.get("q")
            if not q or q in seen:
                continue
            seen.add(q)
            deduped.append(t)
        deduped.reverse()
        tried_final = deduped[-_DOSSIER_MAX_TRIED:]

        dossier = {"facts": facts, "tried": tried_final, "updated_at": now_iso}
        db.execute(
            sa_text(
                "UPDATE samples "
                "SET tags = jsonb_set(COALESCE(tags, '{}'::jsonb), "
                "'{dossier}', CAST(:d AS jsonb)) WHERE id=:sid"
            ),
            {"d": json.dumps(dossier, default=str), "sid": sample_id},
        )
        db.commit()
    except Exception as e:
        try:
            db.rollback()
        except Exception:
            pass
        log.warning("dossier persist failed (sample %s): %s", sample_id, e)
    finally:
        db.close()


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

    Returns (new_fields_dict, sources_per_column, total_cost_usd, status).
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
    if tier_cfg["tools"] == "all":
        skills_section = _render_enrichment_skills_section()
        if skills_section:
            system_prompt = CELL_SYSTEM_PROMPT + "\n\n" + skills_section
        if _dossier_enabled():
            system_prompt = system_prompt + "\n\n" + _DOSSIER_SYSTEM_SECTION

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

    # Field-preview composition (research/deep only — the looping "row agent").
    # Replace over-cap field values with previews + stash full values for
    # inspect_cell, so heavy free-text (e.g. a 25k-char job Description) is
    # fetched on demand instead of re-sent every turn (that bloat blew the org
    # ITPM cap). classify keeps the FULL row: it's one-shot, has no tools, and
    # needs the text to label. Lean rows truncate nothing → no behavior change.
    cell_full_fields: Dict[str, Any] = {}
    visible_view: Dict[str, Any] = row_data if isinstance(row_data, dict) else {}
    hidden_view: Dict[str, Any] = hidden_fields
    if tier_cfg["tools"] == "all":
        cap = _preview_cap_chars()
        visible_view = _compose_preview_fields(visible_view, cap, cell_full_fields)
        hidden_view = _compose_preview_fields(hidden_fields, cap, cell_full_fields)
        ctx.cell_full_fields = cell_full_fields

    user_payload: Dict[str, Any] = {
        "row_visible_to_user": visible_view,
        "columns_to_fill": columns_to_fill,
        "instruction": prompt,
    }
    if hidden_view:
        user_payload["row_hidden_source_fields"] = hidden_view
        user_payload["note"] = (
            "row_visible_to_user is what's shown in the user's table. "
            "row_hidden_source_fields are extra fields the source returned "
            "that aren't currently mapped to a column — you can read these "
            "as additional context for reasoning, but you can't return them "
            "as values without the orchestrator adding columns."
        )
    if cell_full_fields:
        user_payload["truncated_fields"] = sorted(cell_full_fields.keys())
        user_payload["truncation_note"] = (
            "Long field values are shown as PREVIEWS ending with a "
            "'[truncated …]' marker. Call inspect_cell(column=\"<name>\") to read "
            "the full value — but only when the preview isn't enough; most "
            "lookups never need it."
        )

    # Row dossier (opt-in): replay this row's prior findings + dead-ends so the
    # agent reuses known facts and never repeats a search that already failed.
    if _dossier_enabled() and tier_cfg["tools"] == "all" and sample_id:
        _dossier = _load_row_dossier(sample_id)
        _known, _already_tried = _dossier_payload_views(_dossier)
        if _known:
            user_payload["known"] = _known
        if _already_tried:
            user_payload["already_tried"] = _already_tried

    input_items: List[Dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": json.dumps(user_payload, default=str)},
    ]
    tool_defs = _tool_defs_for_tier(tier_cfg)
    # Offer inspect_cell only when at least one field was actually truncated —
    # no point advertising it on lean rows. Works in both loops: the Anthropic
    # path translates tool_defs via translate_tools; both dispatch by looking
    # the name up in CELL_TOOL_HANDLERS.
    if cell_full_fields:
        tool_defs = tool_defs + [_INSPECT_CELL_TOOL_DEF]
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
    llm_retries = 0  # transient-timeout retries (don't count toward turn limit)
    status = "error"  # default; flipped to "filled" or "hit_budget" on exit
    try:
        # Research/deep tiers run on Claude (Messages API); classify stays on
        # OpenAI nano. The Claude loop reuses tool_calls_log + the sources
        # closure, and reassigns the locals the finally block traces on.
        if tier_cfg.get("provider") == "anthropic":
            final_values, final_sources, total_cost, status, error_str = await _anthropic_cell_loop(
                tier_cfg, system_prompt, user_payload, columns_to_fill,
                tool_defs, ctx, tool_calls_log, _build_sources_with_fallback,
            )
            return final_values, final_sources, total_cost, status
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

            # Stream the Responses API so we can abort mid-flight when a
            # hosted web_search burst would push us over cap. Non-stream
            # mode bills 10+ searches in one round-trip before any cap
            # check fires; with streaming, each web_search_call.done
            # event lets us add cost + check cap + close the stream to
            # prevent the remaining queued searches from running.
            response = None
            streamed_search_ids: set[str] = set()
            aborted_over_cap = False
            cost_before_call = total_cost
            try:
                stream_kwargs = {
                    "model": tier_cfg["model"],
                    "input": input_items,
                    "tools": tool_defs,
                    "reasoning": {"effort": tier_cfg["effort"]},
                    "prompt_cache_key": cache_key,
                }
                raw = client.raw_client
                async with raw.responses.stream(**stream_kwargs) as stream:
                    async for event in stream:
                        etype = getattr(event, "type", None)
                        if etype == "response.output_item.done":
                            done_item = getattr(event, "item", None)
                            if done_item is not None and getattr(done_item, "type", None) == "web_search_call":
                                item_id = getattr(done_item, "id", None) or ""
                                if item_id and item_id in streamed_search_ids:
                                    continue
                                if item_id:
                                    streamed_search_ids.add(item_id)
                                query = ""
                                action = getattr(done_item, "action", None)
                                if action is not None:
                                    query = getattr(action, "query", "") or ""
                                total_cost += WEB_SEARCH_CALL_COST_USD
                                tool_calls_log.append({
                                    "name": "web_search",
                                    "args": {"query": query},
                                    "result_preview": f"native (status={getattr(done_item, 'status', '?')})",
                                    "cost": WEB_SEARCH_CALL_COST_USD,
                                })
                                # Abort the stream the moment a web_search
                                # pushes us over cap — the model's queued
                                # searches won't run, saving the burst spend.
                                if total_cost >= tier_cfg["cap"]:
                                    aborted_over_cap = True
                                    log.info(
                                        "cell agent aborting stream mid-burst "
                                        "(research=%s cost=%.4f cap=%.4f, %d searches)",
                                        tier_cfg["name"], total_cost, tier_cfg["cap"],
                                        len(streamed_search_ids),
                                    )
                                    await stream.close()
                                    break
                    if not aborted_over_cap:
                        response = await stream.get_final_response()
            except Exception as e:
                error_str = f"LLM call failed: {e}"[:500]
                # Transient timeouts spike when many cells run concurrently
                # and the OpenAI connection pool / event loop is saturated.
                # When nothing was billed mid-stream (always true for classify
                # — no tools), retrying is safe and recovers the cell instead
                # of leaving it blank. Up to 2 retries with backoff; retries
                # don't count toward HARD_TURN_LIMIT.
                transient = any(
                    t in str(e).lower()
                    for t in ("timed out", "timeout", "connection", "temporarily", "overloaded")
                )
                if transient and total_cost == cost_before_call and llm_retries < 2:
                    llm_retries += 1
                    iteration -= 1
                    await asyncio.sleep(1.0 * llm_retries)
                    continue
                log.warning("cell agent LLM call failed (research=%s): %s", tier_cfg["name"], e)
                return final_values, final_sources, total_cost, "error"

            # Mid-stream cap abort: don't attempt to read partial output.
            # OpenAI billed us for compute up to the close; we already
            # accounted for the web_search fees as they came in.
            if aborted_over_cap:
                error_str = "budget cap reached during web_search burst"
                return final_values, final_sources, total_cost, "hit_budget"

            # Add LLM token cost from the final response (web_search fees
            # were already accumulated mid-stream).
            try:
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
                    model=tier_cfg["model"],
                    input_tokens=non_cached,
                    output_tokens=out_tok,
                    cached_input_tokens=cached_tok,
                )
                total_cost += cost.total_cost_usd
            except Exception as e:
                log.warning("cell agent failed to compute token cost: %s", e)

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
                    # Already accounted for mid-stream above. Skip the
                    # double-count, just record the item in input_items
                    # for the next-iteration history.
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
                                f"${BU_MIN_BUDGET} remaining; only ${remaining:.3f} left"
                            )
                        else:
                            args["__max_cost_usd"] = float(remaining)
                    elif name == "apify_call_actor":
                        if remaining < APIFY_MIN_BUDGET:
                            skip_reason = (
                                f"skipped: apify_call_actor needs at least "
                                f"${APIFY_MIN_BUDGET} remaining; only ${remaining:.3f} left"
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
                # Same _raw_payload pop as the Anthropic loop: capture the raw
                # provider response, never send it to the model.
                raw_payload = tool_result.pop("_raw_payload", None) if isinstance(tool_result, dict) else None
                _log_entry: Dict[str, Any] = {
                    "name": name,
                    "args": args,
                    "result_preview": json.dumps(tool_result, default=str)[:400],
                    "cost": tool_cost,
                }
                if name in PROVIDER_PAYLOAD_TOOLS and isinstance(tool_result, dict) and "error" not in tool_result:
                    _log_entry["result_full"] = _capture_payload(raw_payload if raw_payload is not None else tool_result)
                tool_calls_log.append(_log_entry)
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
            total_cost * 10.0,
            duration_ms,
        )
        if _dossier_enabled():
            _persist_cell_dossier(
                sample_id,
                columns_to_fill,
                final_values if isinstance(final_values, dict) else {},
                final_sources if isinstance(final_sources, dict) else {},
                tool_calls_log,
            )
