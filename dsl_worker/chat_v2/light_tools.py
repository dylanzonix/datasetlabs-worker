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
    query = args.get("query")
    if not query:
        return {"error": "query is required"}, 0.0
    api_key = os.getenv("BRAVE_API_KEY")
    if not api_key:
        return {"error": "BRAVE_API_KEY not configured"}, 0.0

    async with httpx.AsyncClient(timeout=15) as client:
        r = await client.get(
            "https://api.search.brave.com/res/v1/web/search",
            params={"q": query, "count": 10},
            headers={"X-Subscription-Token": api_key, "Accept": "application/json"},
        )
        if r.status_code != 200:
            return {"error": f"brave HTTP {r.status_code}"}, 0.0
        data = r.json() or {}

    results = []
    for item in (data.get("web") or {}).get("results", [])[:10]:
        results.append({
            "title": item.get("title"),
            "url": item.get("url"),
            "snippet": (item.get("description") or "")[:300],
        })
    return {"results": results}, 0.0


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
    clickable chips below the assistant's message."""
    chips = args.get("chips") or []
    if not isinstance(chips, list) or not chips:
        return {"error": "chips must be a non-empty list of {label, message}"}, 0.0
    # The FE consumes these via the run-event stream. For now we just return
    # them — the streaming loop pulls them out and emits a `suggestions` event.
    return {"ok": True, "count": len(chips), "chips": chips}, 0.0


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
