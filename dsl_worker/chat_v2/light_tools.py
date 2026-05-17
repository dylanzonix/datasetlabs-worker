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

    from dsl_worker.billing.tracked_client import TrackedOpenAIClient
    from openai import AsyncOpenAI
    client = TrackedOpenAIClient(AsyncOpenAI(api_key=api_key))
    model = os.getenv("OPENAI_MODEL_MINI", "gpt-5.4-mini")

    instruction = (
        "Search the web and return up to 10 results for the query below. "
        "Respond with ONLY a JSON array of objects with keys "
        "title (string), url (string), snippet (string up to 300 chars). "
        "No prose, no markdown, no preamble — just the JSON array."
        f"\n\nQuery: {query}"
    )
    llm_cost_usd = 0.0
    try:
        resp, usage_cost = await client.responses_create(
            model=model,
            input=[{"role": "user", "content": instruction}],
            tools=[{"type": "web_search"}],
        )
        llm_cost_usd = float(getattr(usage_cost, "total_cost_usd", 0.0) or 0.0)
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
        return {"results": [], "raw": text[:600]}, llm_cost_usd

    cleaned = []
    for item in results[:10]:
        if not isinstance(item, dict):
            continue
        cleaned.append({
            "title": str(item.get("title") or ""),
            "url": str(item.get("url") or ""),
            "snippet": str(item.get("snippet") or "")[:300],
        })
    # Return the real LLM USD cost. Caller (agent.run_turn) accumulates
    # this directly into total_cost_usd, no × 10 conversion needed since
    # this is already in USD.
    return {"results": cleaned}, llm_cost_usd


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

    url = os.getenv("SANDBOX_SERVICE_URL", "")
    if not url:
        return {"error": "SANDBOX_SERVICE_URL not configured"}, 0.0
    try:
        from dsl_worker.chat_api import candidates
    except Exception:
        candidates = None
    try:
        async with SandboxClient(url, timeout=90) as pool:
            session = await pool.create_session()
            # Pre-existing files (so we can detect newly written ones after exec)
            try:
                pre = {f.name for f in (await session.list_files())}
            except Exception:
                pre = set()
            for fn in files:
                if not candidates or not ctx.project_id:
                    continue
                try:
                    blob_bytes = candidates.read_candidates_bytes(ctx.project_id, fn)
                    await session.upload_content(blob_bytes, fn)
                except Exception as e:
                    log.warning("code_exec file upload %s failed: %s", fn, e)
            result = await session.exec_python(code, timeout=60)

            # Capture newly-written files and stash them in the candidate
            # store so `table_create(source="file", file_id=<name>)` works.
            captured: List[str] = []
            if candidates and ctx.project_id:
                try:
                    post = await session.list_files()
                    for f in post:
                        name = f.name
                        if name in pre or name.startswith("_") or name.endswith(".py"):
                            continue
                        try:
                            data = await session.download_file(name)
                            blob = getattr(data, "content", None) or data
                            if isinstance(blob, bytes):
                                # Upload bytes directly as one JSONL row (it's
                                # not really jsonl but write_candidates wants
                                # an iterable of dicts; bypass by writing raw)
                                from dsl_worker.chat_api.candidates import (
                                    _candidate_blob_path,
                                )
                                from dsl_api.azure.blob import get_blob_client
                                blob_path = _candidate_blob_path(ctx.project_id, name)
                                client = get_blob_client(blob_path)
                                import io
                                client.upload_blob(
                                    io.BytesIO(blob), overwrite=True,
                                    metadata={"tool": "code_exec", "items_count": "0"},
                                )
                                captured.append(name)
                        except Exception as e:
                            log.warning("code_exec capture %s failed: %s", name, e)
                except Exception as e:
                    log.warning("code_exec post-list failed: %s", e)

            return {
                "ok": bool(getattr(result, "success", False)),
                "stdout": (getattr(result, "stdout", "") or "")[:8000],
                "stderr": (getattr(result, "stderr", "") or "")[:2000],
                "exit_code": getattr(result, "exit_code", None),
                "files_captured": captured,
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
    input — agent shouldn't have to retry; just acknowledge.

    Emits a `suggestions` SSE event so the FE renders chips live; end-of-turn
    persistence into applied_changes.suggestions happens in
    chat_v2/runs.py by replaying these events (mirrors how table_cards is
    captured for refresh-survival).
    """
    # Agent has been observed sending several shapes — accept any of them.
    raw = args.get("chips") or args.get("suggestions") or args.get("items") or []
    if not isinstance(raw, list):
        raw = []
    cleaned: list[Dict[str, Any]] = []
    for c in raw:
        if not isinstance(c, dict):
            continue
        label = (c.get("label") or "").strip()
        message = (c.get("message") or "").strip()
        if not label or not message:
            continue
        item: Dict[str, Any] = {"label": label[:80], "message": message[:500]}
        cap = c.get("cap_override_cents")
        if isinstance(cap, (int, float)) and cap > 0:
            item["cap_override_cents"] = int(cap)
        cleaned.append(item)
    if not cleaned:
        return {"ok": False, "count": 0, "note": "no valid chips (need [{label, message}, ...])"}, 0.0

    if ctx.run_id is not None:
        try:
            from dsl_worker.chat_api import runs as legacy_runs
            from dsl_api.models import ChatRun
            run_obj = ctx.db.query(ChatRun).filter(ChatRun.id == ctx.run_id).first()
            if run_obj is not None:
                legacy_runs.emit_event(ctx.db, run_obj, "suggestions", {
                    "items": cleaned,
                })
        except Exception:
            log.exception("suggestions emit failed; continuing")

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
