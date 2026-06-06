"""Light wrapper tools: apify_search_actors, apify_actor_details, web_search,
code_exec, suggest_replies.

These are thin shells over existing infra. The orchestrator imports HANDLERS
from this module and merges with chat/tools.py.
"""

from __future__ import annotations

import logging
import os
import asyncio
from typing import Any, Dict, List, Tuple

import httpx

from dsl_worker.chat.tools import ToolContext


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

    # Pull a wider net than we return (limit=25) so we can re-sort client-side
    # by popularity. Apify's default search ordering is relevance-only, which
    # routinely puts niche / unmaintained actors above battle-tested ones with
    # 100k+ runs. We re-sort by total runs descending and return the top 8 so
    # the agent sees the high-trust options first.
    async with httpx.AsyncClient(timeout=15) as client:
        r = await client.get(
            "https://api.apify.com/v2/store",
            params={"token": api_key, "search": query, "limit": 25},
        )
        if r.status_code != 200:
            return {"error": f"apify search HTTP {r.status_code}: {r.text[:200]}"}, 0.0
        items = (r.json().get("data") or {}).get("items") or []

    def _runs(it: Dict[str, Any]) -> int:
        return int(((it.get("stats") or {}).get("totalRuns") or 0))

    items.sort(key=_runs, reverse=True)

    actors = []
    for it in items[:8]:
        stats = it.get("stats") or {}
        actors.append({
            "actor_id": f"{(it.get('username') or '')}/{(it.get('name') or '')}",
            "title": it.get("title"),
            "short_description": (it.get("description") or "")[:200],
            # All-time, not monthly — old name was misleading. Agent should
            # prefer actors with 10k+ total_runs unless there's a clear
            # reason to pick a niche one.
            "total_runs": stats.get("totalRuns"),
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


def _apply_sandbox_ops(
    table_id: str, ops_bytes: bytes, run_id: str | None = None,
) -> Dict[str, Any]:
    """Process _dsl_ops.jsonl produced by dsl_tools.add_rows() inside the
    sandbox.  Only handles add_rows — other op types are ignored.

    Returns a summary dict {rows_inserted: N} or {error: ...}.
    """
    import json as _json
    from dsl_worker.chat.tools import _commit_rows, resolve_table_id
    from dsl_api.db import SessionLocal

    lines = ops_bytes.decode("utf-8", errors="replace").strip().splitlines()
    all_items: List[Dict[str, Any]] = []
    skipped_ops: List[str] = []
    for line in lines:
        if not line.strip():
            continue
        try:
            op = _json.loads(line)
        except Exception:
            continue
        if op.get("op") == "add_rows":
            items = op.get("items")
            if isinstance(items, list):
                all_items.extend(items)
        elif op.get("op"):
            skipped_ops.append(op["op"])

    result: Dict[str, Any] = {}
    if skipped_ops:
        result["warning"] = (
            f"dsl_tools ops [{', '.join(skipped_ops)}] are not supported "
            "from code_exec — use the direct tool instead (row_delete, etc.)"
        )

    if not all_items:
        result["rows_inserted"] = 0
        return result

    db = SessionLocal()
    try:
        from sqlalchemy import text as sa_text
        tbl_cols = db.execute(
            sa_text("SELECT columns FROM tables WHERE id=:tid"),
            {"tid": table_id},
        ).scalar()

        item_keys = set(all_items[0].keys())

        if tbl_cols and isinstance(tbl_cols, list):
            # Items might use display names ("Company") or source_field
            # names ("company_name"). Detect which and build the map so
            # _commit_rows finds the values.
            display_names = {c["name"] for c in tbl_cols if c.get("name")}
            source_fields = {c.get("source_field", c["name"]) for c in tbl_cols if c.get("name")}

            if item_keys & display_names:
                # Items use display names → identity map
                column_map = [
                    {"name": c["name"], "source_field": c["name"]}
                    for c in tbl_cols if c.get("name")
                ]
            elif item_keys & source_fields:
                # Items use source_field names → normal table map
                column_map = [
                    {"name": c["name"], "source_field": c.get("source_field", c["name"])}
                    for c in tbl_cols if c.get("name")
                ]
            else:
                # No overlap — pass items through as-is
                column_map = [{"name": k, "source_field": k} for k in item_keys]
        else:
            column_map = [{"name": k, "source_field": k} for k in item_keys]

        _commit_rows(db, table_id, all_items, column_map, store_raw=False, run_id=run_id)
        result["rows_inserted"] = len(all_items)
        return result
    except Exception as e:
        log.exception("_apply_sandbox_ops failed: %s", e)
        result["error"] = str(e)[:300]
        return result
    finally:
        db.close()


async def code_exec(args: Dict[str, Any], ctx: ToolContext) -> Tuple[Dict[str, Any], float]:
    code = args.get("code")
    files = args.get("files") or []
    table_id = args.get("table_id")
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
        from dsl_worker.chat import candidates
    except Exception:
        candidates = None

    if table_id and ctx.project_id:
        from dsl_worker.chat.tools import resolve_table_id
        from dsl_api.db import SessionLocal
        _db = SessionLocal()
        try:
            table_id = resolve_table_id(_db, str(ctx.project_id), table_id) or table_id
        finally:
            _db.close()

    try:
        async with SandboxClient(url, timeout=90) as pool:
            session = await pool.create_session()

            # Upload dsl_tools.py so sandbox code can `import dsl_tools`.
            try:
                from dsl_worker.infra.dsl_tools_module import DSL_TOOLS_SOURCE
                await session.upload_content(
                    DSL_TOOLS_SOURCE.encode("utf-8"), "dsl_tools.py",
                )
            except Exception as e:
                log.warning("code_exec dsl_tools upload failed: %s", e)

            # Pre-existing files (so we can detect newly written ones after exec)
            try:
                pre = {f.name for f in (await session.list_files())}
            except Exception:
                pre = set()

            # Upload input files. Try candidates store first, then
            # project_files (user uploads stored in Azure Blob).
            for fn in files:
                if not ctx.project_id:
                    continue
                uploaded = False
                if candidates:
                    try:
                        blob_bytes = candidates.read_candidates_bytes(ctx.project_id, fn)
                        await session.upload_content(blob_bytes, fn)
                        uploaded = True
                    except Exception:
                        pass
                if not uploaded:
                    try:
                        from dsl_worker.sources.file import _read_project_file_bytes
                        pf_bytes, pf_name = _read_project_file_bytes(fn, str(ctx.project_id))
                        if pf_bytes:
                            await session.upload_content(pf_bytes, pf_name or fn)
                            uploaded = True
                    except Exception as e:
                        log.warning("code_exec file upload %s failed: %s", fn, e)

            # If a table_id is in scope, export its rows into the sandbox as
            # table_rows.jsonl so sandbox code can READ existing table data —
            # compute aggregates (max/min/sum/argmax), group-by, dedup, etc.
            # The sandbox has no DB access, so without this the agent can see
            # its own table only by guessing. READ-ONLY: this never mutates the
            # table; writes still go through the normal tools. Each line is the
            # row dict plus "_id" (the row's stable id).
            if table_id and ctx.project_id:
                try:
                    import json
                    from dsl_api.db import SessionLocal as _SL
                    from sqlalchemy import text as _sa_text
                    _rdb = _SL()
                    try:
                        _table_rows = _rdb.execute(
                            _sa_text(
                                "SELECT id::text, row FROM samples "
                                "WHERE table_id=:tid AND deleted_at IS NULL "
                                "ORDER BY created_at LIMIT 50000"
                            ),
                            {"tid": table_id},
                        ).fetchall()
                    finally:
                        _rdb.close()
                    _lines = []
                    for _sid, _row in _table_rows:
                        _rec = dict(_row) if isinstance(_row, dict) else {}
                        _rec["_id"] = _sid
                        _lines.append(json.dumps(_rec, default=str))
                    await session.upload_content(
                        ("\n".join(_lines)).encode("utf-8"), "table_rows.jsonl",
                    )
                except Exception as e:
                    log.warning("code_exec table export failed: %s", e)

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
                                from dsl_worker.chat.candidates import (
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

            # Process dsl_tools.add_rows() ops written by sandbox code.
            ops_result: Dict[str, Any] | None = None
            if table_id:
                try:
                    ops_data = await session.download_file("_dsl_ops.jsonl")
                    ops_blob = getattr(ops_data, "content", None) or ops_data
                    if isinstance(ops_blob, bytes) and ops_blob.strip():
                        ops_result = _apply_sandbox_ops(
                            table_id, ops_blob, run_id=str(ctx.run_id) if ctx.run_id else None,
                        )
                except Exception as e:
                    if "not found" not in str(e).lower() and "404" not in str(e):
                        log.warning("code_exec ops read failed: %s", e)

            raw_stdout = getattr(result, "stdout", "") or ""
            raw_stderr = getattr(result, "stderr", "") or ""
            stdout_truncated = len(raw_stdout) > 8000
            stderr_truncated = len(raw_stderr) > 2000
            # If stderr was truncated, prefer the TAIL (where the actual
            # traceback / fail reason lives) over the HEAD. A truncated
            # head loses the error that caused the failure — the exact
            # thing the agent needs to see to fix the snippet.
            stderr_out = raw_stderr[-2000:] if stderr_truncated else raw_stderr
            envelope: Dict[str, Any] = {
                "ok": bool(getattr(result, "success", False)),
                "stdout": raw_stdout[:8000],
                "stderr": stderr_out,
                "stdout_truncated": stdout_truncated,
                "stderr_truncated": stderr_truncated,
                "stdout_total_chars": len(raw_stdout),
                "stderr_total_chars": len(raw_stderr),
                "exit_code": getattr(result, "exit_code", None),
                "files_captured": captured,
            }
            if ops_result:
                envelope["ops"] = ops_result
            return envelope, 0.0
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
    chat/runs.py by replaying these events (mirrors how table_cards is
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
            from dsl_worker.chat import run_state
            from dsl_api.models import ChatRun
            run_obj = ctx.db.query(ChatRun).filter(ChatRun.id == ctx.run_id).first()
            if run_obj is not None:
                run_state.emit_event(ctx.db, run_obj, "suggestions", {
                    "items": cleaned,
                })
        except Exception:
            log.exception("suggestions emit failed; continuing")

    return {"ok": True, "count": len(cleaned), "chips": cleaned}, 0.0


# ---------------------------------------------------------------------------
# Registry merge point
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# load_skill
# ---------------------------------------------------------------------------


async def load_skill(args: Dict[str, Any], ctx: ToolContext) -> Tuple[Dict[str, Any], float]:
    """Return the body of a named skill from the skills directory.

    The orchestrator (and research-tier cell agent) see each skill's name +
    description in their system prompt under `# Skills`. When one matches the
    current task they call this to read the playbook. Bodies don't enter
    context until called.
    """
    name = (args.get("name") or "").strip()
    if not name:
        return {"error": "name is required"}, 0.0
    from dsl_worker.skills import get_skill_body
    body = get_skill_body(name)
    if body is None:
        return {"error": f"unknown skill: {name}"}, 0.0
    return {"name": name, "body": body}, 0.0


# ---------------------------------------------------------------------------
# plan_options
# ---------------------------------------------------------------------------


async def plan_options(args: Dict[str, Any], ctx: ToolContext) -> Tuple[Dict[str, Any], float]:
    """Pause the agent to ask the user to pick between 2–4 explicit options.

    Use ONLY when picking wrong would meaningfully diverge from the
    user's intent — most table-creation asks have one obvious shape and
    should not call this. The FE shows a small button card; on click,
    `/plan_option_picks/{id}/respond` resolves the Future and this
    handler returns ``{"chosen": "<key>"}`` to the agent so it can
    proceed with that choice.

    args: {
      question: str — the question shown to the user (one short line)
      options:  [{label: str, key: str, description?: str}, ...]
                exactly 2–4 entries. `key` is what comes back in the
                tool result.
    }
    """
    question = (args.get("question") or "").strip()
    raw_options = args.get("options") or []
    if not question:
        return {"error": "question is required"}, 0.0
    if not isinstance(raw_options, list):
        return {"error": "options must be a list"}, 0.0

    cleaned: List[Dict[str, Any]] = []
    seen_keys = set()
    for o in raw_options:
        if not isinstance(o, dict):
            continue
        label = (o.get("label") or "").strip()
        key = (o.get("key") or "").strip()
        if not label or not key or key in seen_keys:
            continue
        seen_keys.add(key)
        item: Dict[str, Any] = {"label": label[:80], "key": key[:60]}
        desc = (o.get("description") or "").strip()
        if desc:
            item["description"] = desc[:240]
        cleaned.append(item)
    if len(cleaned) < 2 or len(cleaned) > 4:
        return {
            "error": "options must contain 2-4 entries with {label, key} each"
        }, 0.0

    if ctx.run_id is None:
        # No chat run = no SSE channel = no way to ask the user. Bail
        # so the agent isn't stuck on a Future nobody can resolve.
        return {"error": "plan_options is only valid inside an active chat run"}, 0.0

    from dsl_worker.chat.option_picks import REGISTRY as OPTION_PICKS

    pending = await OPTION_PICKS.request(
        project_id=ctx.project_id,
        question=question,
        options=cleaned,
    )

    # Emit via run_state.emit_event the way other in-band tool events
    # do — branch_ctxs don't carry an emit_event callback. Persisting
    # the event also means the FE can replay-rehydrate on reconnect.
    try:
        from dsl_worker.chat import run_state
        from dsl_api.models import ChatRun
        run_obj = ctx.db.query(ChatRun).filter(ChatRun.id == ctx.run_id).first()
        if run_obj is not None:
            run_state.emit_event(ctx.db, run_obj, "plan_options_required", {
                "pick_id": pending.id,
                "question": question,
                "options": cleaned,
            })
    except Exception:
        log.exception("plan_options_required emit failed; continuing")

    try:
        chosen = await pending.future
    except asyncio.CancelledError:
        # Make sure the registry entry doesn't dangle if the agent
        # loop gets cancelled mid-await.
        await OPTION_PICKS.resolve(pending.id, "")
        raise

    if not chosen:
        # cleanup_chat_run resolves with empty string when the chat
        # run ends without a user pick — surface that to the agent.
        return {
            "chosen": "",
            "note": "User did not pick an option before the run ended. Don't retry; wait for direction.",
        }, 0.0

    return {"chosen": chosen}, 0.0


# ---------------------------------------------------------------------------
# Registry merge point
# ---------------------------------------------------------------------------

HANDLERS = {
    "apify_search_actors": apify_search_actors,
    "apify_actor_details": apify_actor_details,
    # web_search migrated to the OpenAI hosted tool (added directly to
    # the Responses tools= array in chat/agent.py + cell_agent.py).
    # The sidecar `web_search` function above is kept for any external
    # importer but is no longer dispatched by the chat HANDLERS map.
    "code_exec": code_exec,
    "suggest_replies": suggest_replies,
    "plan_options": plan_options,
    "load_skill": load_skill,
}
