"""Light wrapper tools: apify_search_actors, apify_actor_details, web_search,
code_exec, suggest_replies.

These are thin shells over existing infra. The orchestrator imports HANDLERS
from this module and merges with chat_v2/tools.py.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Tuple

import httpx

from dsl_worker.chat_v2.tools import ToolContext


log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# apify_search_actors
# ---------------------------------------------------------------------------


async def apify_search_actors(args: Dict[str, Any], ctx: ToolContext) -> Tuple[Dict[str, Any], float]:
    query = args.get("query")
    if not query:
        return {"error": "query is required"}, 0.0
    api_key = os.getenv("APIFY_API_KEY")
    if not api_key:
        return {"error": "APIFY_API_KEY not configured"}, 0.0

    async with httpx.AsyncClient(timeout=15) as client:
        r = await client.get(
            "https://api.apify.com/v2/store",
            params={"token": api_key, "search": query, "limit": 8},
        )
        if r.status_code != 200:
            return {"error": f"apify search HTTP {r.status_code}: {r.text[:200]}"}, 0.0
        items = (r.json().get("data") or {}).get("items") or []

    actors = []
    for it in items:
        stats = it.get("stats") or {}
        actors.append({
            "actor_id": f"{(it.get('username') or '')}/{(it.get('name') or '')}",
            "title": it.get("title"),
            "short_description": (it.get("description") or "")[:200],
            "monthly_run_count": stats.get("totalRuns"),
            "rating": (it.get("stats") or {}).get("publicActorStats", {}).get("avgRating"),
            "pricing_summary": [p.get("pricingModel") for p in (it.get("pricingInfos") or [])],
        })
    return {"actors": actors}, 0.0


# ---------------------------------------------------------------------------
# apify_actor_details
# ---------------------------------------------------------------------------


async def apify_actor_details(args: Dict[str, Any], ctx: ToolContext) -> Tuple[Dict[str, Any], float]:
    actor_id = args.get("actor_id")
    if not actor_id:
        return {"error": "actor_id is required"}, 0.0
    api_key = os.getenv("APIFY_API_KEY")
    if not api_key:
        return {"error": "APIFY_API_KEY not configured"}, 0.0

    aid = actor_id.replace("/", "~")
    async with httpx.AsyncClient(timeout=20) as client:
        r = await client.get(
            f"https://api.apify.com/v2/acts/{aid}",
            params={"token": api_key},
        )
        if r.status_code != 200:
            return {"error": f"apify details HTTP {r.status_code}"}, 0.0
        data = r.json().get("data") or {}

        # Pull input schema from latest build.
        builds = data.get("taggedBuilds") or {}
        latest = builds.get("latest") or {}
        bid = latest.get("buildId")
        input_schema = None
        if bid:
            rb = await client.get(
                f"https://api.apify.com/v2/actor-builds/{bid}",
                params={"token": api_key},
            )
            if rb.status_code == 200:
                build = rb.json().get("data") or {}
                schema_str = build.get("inputSchema")
                if isinstance(schema_str, str):
                    import json
                    try:
                        input_schema = json.loads(schema_str)
                    except json.JSONDecodeError:
                        input_schema = None

    return {
        "actor_id": actor_id,
        "title": data.get("title"),
        "description": (data.get("description") or "")[:500],
        "input_schema": input_schema,
        "pricing": data.get("pricingInfos"),
        "stats": {
            "total_runs": (data.get("stats") or {}).get("totalRuns"),
            "total_users": (data.get("stats") or {}).get("totalUsers"),
        },
    }, 0.0


# ---------------------------------------------------------------------------
# web_search
# ---------------------------------------------------------------------------


async def web_search(args: Dict[str, Any], ctx: ToolContext) -> Tuple[Dict[str, Any], float]:
    """Web search via OpenAI's native web_search tool (Responses API).

    We had Brave wired in originally but the subscription key is invalid in
    this environment (HTTP 422 SUBSCRIPTION_TOKEN_INVALID). OpenAI's built-in
    web_search is included in our plan and returns grounded, cite-able
    results — sidecar Responses call returns the JSON shape our agent
    expects so the caller doesn't need to change.
    """
    query = args.get("query")
    if not query:
        return {"error": "query is required"}, 0.0
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return {"error": "OPENAI_API_KEY not configured"}, 0.0

    from openai import AsyncOpenAI
    client = AsyncOpenAI(api_key=api_key)
    model = os.getenv("OPENAI_MODEL_MINI", "gpt-5.4-mini")

    instruction = (
        "Search the web and return up to 10 results for the query below. "
        "Respond with ONLY a JSON array of objects with keys "
        "title (string), url (string), snippet (string up to 300 chars). "
        "No prose, no markdown, no preamble — just the JSON array."
        f"\n\nQuery: {query}"
    )
    try:
        resp = await client.responses.create(
            model=model,
            input=instruction,
            tools=[{"type": "web_search"}],
        )
    except Exception as e:
        return {"error": f"web_search failed: {e}"[:200]}, 0.0

    text = (resp.output_text or "").strip()
    # Strip optional ```json fences if the model still emits them.
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:].strip()
    import json as _json
    import re as _re
    results = None
    try:
        results = _json.loads(text)
    except Exception:
        # The model sometimes appends prose after the array, or wraps it in
        # an object. Pull out the first JSON array via balanced-bracket scan.
        m = _re.search(r"\[\s*\{.*\}\s*\]", text, _re.DOTALL)
        if m:
            try:
                results = _json.loads(m.group(0))
            except Exception:
                results = None
    if not isinstance(results, list):
        log.warning("web_search: model output not parsable as JSON list")
        return {"results": [], "raw": text[:600]}, 0.0

    cleaned = []
    for item in results[:10]:
        if not isinstance(item, dict):
            continue
        cleaned.append({
            "title": str(item.get("title") or ""),
            "url": str(item.get("url") or ""),
            "snippet": str(item.get("snippet") or "")[:300],
        })
    return {"results": cleaned}, 0.0


# ---------------------------------------------------------------------------
# code_exec
# ---------------------------------------------------------------------------


async def code_exec(args: Dict[str, Any], ctx: ToolContext) -> Tuple[Dict[str, Any], float]:
    code = args.get("code")
    files = args.get("files") or []
    if not code:
        return {"error": "code is required"}, 0.0
    try:
        from sandbox_service import SandboxClient
    except ImportError:
        return {"error": "sandbox_service not available"}, 0.0

    sandbox = SandboxClient(os.getenv("SANDBOX_SERVICE_URL", ""))
    try:
        result = await sandbox.exec_python(code, files=files)
        return {
            "ok": result.get("ok", False),
            "stdout": (result.get("stdout") or "")[:8000],
            "stderr": (result.get("stderr") or "")[:2000],
            "duration_ms": result.get("duration_ms"),
        }, 0.0
    except Exception as e:
        log.exception("code_exec failed: %s", e)
        return {"error": str(e)[:300]}, 0.0


# ---------------------------------------------------------------------------
# suggest_replies
# ---------------------------------------------------------------------------


async def suggest_replies(args: Dict[str, Any], ctx: ToolContext) -> Tuple[Dict[str, Any], float]:
    """Emits chip suggestions for the user's next move. UI renders these as
    clickable chips below the assistant's message. Tolerates empty/malformed
    input — agent shouldn't have to retry; just acknowledge."""
    chips = args.get("chips") or []
    if not isinstance(chips, list):
        chips = []
    # Best-effort filter: keep dict-shape chips with at least a label
    cleaned = [
        c for c in chips
        if isinstance(c, dict) and c.get("label") and c.get("message")
    ]
    return {"ok": True, "count": len(cleaned), "chips": cleaned}, 0.0


# ---------------------------------------------------------------------------
# Registry merge point
# ---------------------------------------------------------------------------

HANDLERS = {
    "apify_search_actors": apify_search_actors,
    "apify_actor_details": apify_actor_details,
    "web_search": web_search,
    "code_exec": code_exec,
    "suggest_replies": suggest_replies,
}
