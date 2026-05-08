"""
Apify namespace tools for orchestrator and row generator.

Provides: search_actors, actor_details, call_actor (with auto-download to file)
All with defer_loading for on-demand discovery via tool_search.

Tool design modeled on Apify's official MCP server — search keyword coaching,
input validation with schema feedback, preview + metadata response format.
"""

import json
import logging
import time
from pathlib import Path
from typing import Any, Callable, Awaitable, Dict, List, Optional, Tuple

from dsl_worker.infra.apify_client import ApifyClient
from dsl_worker.infra.apify_dataset import download_apify_results

logger = logging.getLogger(__name__)

ToolHandler = Callable[[Dict[str, Any]], Awaitable[Tuple[str, float]]]

NAMESPACE_DESCRIPTION = (
    "22,000+ pre-built web scrapers for any website. Search for scrapers by "
    "platform or data type, get input schemas and pricing, run scrapers with "
    "results saved to file. Best tool for scraping specific websites — cheaper "
    "and faster than browsing."
)


def register_apify_namespace(
    registry: Any,
    client: ApifyClient,
    api_key: str,
    workspace_dir: Path,
    file_counter: Optional[List[int]] = None,
    on_file_written: Optional[Callable] = None,
    row_count_provider: Optional[Callable[[], int]] = None,
) -> None:
    """Register the apify namespace on a ToolRegistry.

    `row_count_provider` is an optional callable returning the project's
    current row count. Used to enforce the small-first cap: when the
    project has 0 rows, max_items on apify_call_actor is server-clamped
    to 50 so the agent can't fetch a giant noisy first batch and never
    commit anything (the over-perfection failure mode).
    """
    if file_counter is None:
        file_counter = [0]

    SMALL_FIRST_CAP = 50

    def _next_filename(prefix: str) -> Path:
        idx = file_counter[0]
        file_counter[0] += 1
        return workspace_dir / "candidates" / f"{prefix}_{idx}.jsonl"

    # ── search_actors ────────────────────────────────────────────────

    async def search_actors(args: Dict) -> Tuple[str, float]:
        query = args.get("query", "")
        if not query:
            return "Error: query is required.", 0.0

        try:
            results = await client.search_actors(query, limit=args.get("limit", 5))
        except Exception as e:
            return f"Apify search error: {e}", 0.0

        if not results:
            return (
                f"No Apify actors found for: {query}\n\n"
                "Try simpler keywords — use the platform name (e.g. 'Instagram', "
                "'Reddit') and data type (e.g. 'posts', 'profiles'). Avoid generic "
                "terms like 'scraper' or 'data extraction'."
            ), 0.0

        lines = []
        for r in results:
            meta_bits = []
            if r.get("rating") and r.get("review_count"):
                meta_bits.append(f"{r['rating']:.1f}/5 ({r['review_count']})")
            meta_bits.append(f"{r['total_runs']:,} runs")
            meta_bits.append(f"{r['total_users']:,} users")
            if r.get("creator"):
                meta_bits.append(f"by {r['creator']}")
            if r.get("pricing"):
                meta_bits.append(r["pricing"])

            lines.append(
                f"**{r['actor_id']}** — {r['title']}\n"
                f"  {r['description'][:600]}\n"
                f"  {' | '.join(meta_bits)}"
            )

        return (
            f"Found {len(results)} Apify actors for \"{query}\":\n\n"
            + "\n\n".join(lines)
            + "\n\nUse actor_details to get the full input schema before running."
        ), 0.0

    # ── actor_details ────────────────────────────────────────────────

    async def actor_details(args: Dict) -> Tuple[str, float]:
        actor_id = args.get("actor_id", "")
        if not actor_id:
            return "Error: actor_id is required.", 0.0

        try:
            details = await client.get_actor_details(actor_id)
        except Exception as e:
            return f"Error getting actor details: {e}", 0.0

        if not details:
            return (
                f"Actor not found: {actor_id}\n"
                "Verify format is 'username/actor-name'. Use search_actors to find actors."
            ), 0.0

        parts = [f"**{details['title']}**"]
        if details.get("pricing"):
            parts.append(f"Pricing: {details['pricing']}")
        if details.get("stats"):
            parts.append(f"Stats: {details['stats']}")
        parts.append("")
        parts.append(details["description"][:8000])

        if details.get("readme_summary"):
            parts.append("")
            parts.append(details["readme_summary"][:8000])

        # Input schema — required vs optional
        input_schema = details.get("input_schema")
        if input_schema and input_schema.get("properties"):
            required_set = set(input_schema.get("required", []))
            req_props = {k: v for k, v in input_schema["properties"].items()
                         if k in required_set}
            opt_props = {k: v for k, v in input_schema["properties"].items()
                         if k not in required_set}

            parts.append("")
            parts.append("## INPUT SCHEMA")
            parts.append("Pass these as params to call_actor alongside actor_id.")

            if req_props:
                parts.append("")
                parts.append("Required:")
                for name, prop in req_props.items():
                    parts.append(_format_schema_prop(name, prop, True))

            if opt_props:
                parts.append("")
                parts.append("Optional (only pass if you need to override defaults):")
                for name, prop in opt_props.items():
                    parts.append(_format_schema_prop(name, prop, False))

        # Output views (column hints) — often empty, but valuable when present
        output_views = details.get("output_views")
        if output_views:
            view_lines = _format_output_views(output_views)
            if view_lines:
                parts.append("")
                parts.append("## OUTPUT FIELDS")
                parts.append(
                    "Columns the actor produces (publisher-declared; actual "
                    "rows may include additional fields)."
                )
                parts.extend(view_lines)

        return "\n".join(parts), 0.0

    # ── call_actor ───────────────────────────────────────────────────

    async def call_actor(args: Dict) -> Tuple[str, float]:
        actor_id = args.get("actor_id", "")
        if not actor_id:
            return "Error: actor_id is required.", 0.0

        # Everything except known keys is actor input. timeout_secs is a
        # real top-level param now (LLMs were passing it thinking it
        # configured us; previously it leaked into actor input).
        known_keys = {"actor_id", "max_items", "max_cost", "timeout_secs"}
        run_input = {k: v for k, v in args.items() if k not in known_keys}
        max_items = args.get("max_items")
        max_cost = args.get("max_cost")
        timeout_secs = args.get("timeout_secs") or 600

        # Small-first cap: when the project has 0 rows, force a small
        # batch on the first apify_call_actor so the agent can't pull
        # 500 items chasing "scope coverage" and never commit anything.
        # The agent saw run 88328425 / 3fc103bc bail this way: 5 calls,
        # 180+ candidates fetched, 0 rows committed. Cap is independent
        # of what the agent passes in run_input — also strip maxItems
        # from the actor input itself so the actor doesn't return more
        # than we'll download.
        small_first_clamped = False
        if row_count_provider is not None:
            try:
                current_rows = int(row_count_provider() or 0)
            except Exception:
                current_rows = 0
            if current_rows == 0:
                if max_items is None or max_items > SMALL_FIRST_CAP:
                    max_items = SMALL_FIRST_CAP
                    small_first_clamped = True
                # Strip the input-level maxItems too — actors honor their
                # own input field, not our download cap.
                for key in ("maxItems", "max_items", "max_results", "maxResults", "limit"):
                    if key in run_input and (
                        not isinstance(run_input[key], int)
                        or run_input[key] > SMALL_FIRST_CAP
                    ):
                        run_input[key] = SMALL_FIRST_CAP

        logger.info(f"[apify] Running actor: {actor_id}")

        # Start async run
        try:
            import httpx
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            }
            actor_path = actor_id.replace("/", "~")
            run_params: Dict[str, Any] = {"timeout": 0}  # async start
            if max_cost is not None:
                run_params["maxTotalChargeUsd"] = max_cost
            async with httpx.AsyncClient(timeout=30.0) as http:
                resp = await http.post(
                    f"https://api.apify.com/v2/acts/{actor_path}/runs",
                    headers=headers,
                    json=run_input,
                    params=run_params,
                )

            if resp.status_code not in (200, 201):
                # Try to get input schema for error feedback
                error_msg = f"Apify actor {actor_id} failed to start: HTTP {resp.status_code}"
                try:
                    err_body = resp.json()
                    error_msg += f"\n{err_body.get('error', {}).get('message', resp.text[:200])}"
                except Exception:
                    pass

                schema_block = await _schema_feedback_block(client, actor_id)
                if schema_block:
                    error_msg += "\n\n" + schema_block

                return error_msg, 0.0

            run_data = resp.json().get("data", {})
            run_id = run_data.get("id")

            if not run_id:
                return "Error: no run ID returned from Apify.", 0.0

        except Exception as e:
            return f"Apify run error: {e}", 0.0

        # Download results (polls until done, writes to file)
        output_path = _next_filename(f"apify_{actor_id.replace('/', '_')}")

        result = await download_apify_results(
            api_key=api_key,
            output_path=output_path,
            run_id=run_id,
            limit=max_items,
            poll_timeout=timeout_secs,
        )

        if "error" in result:
            return f"Apify actor {actor_id} error: {result['error']}", 0.0

        cost = result.get("cost_usd", 0.0)
        item_count = result.get("item_count", 0)

        if item_count == 0:
            msg = (
                f"Apify actor {actor_id} returned 0 items.\n\n"
                "Possible causes: input was missing required fields, the "
                "search/URL had no matches, or the actor silently rejected "
                "an unknown parameter. If you didn't pass real input, retry "
                "with the required fields below."
            )
            schema_block = await _schema_feedback_block(client, actor_id)
            if schema_block:
                msg += "\n\n" + schema_block
            return msg, cost

        if on_file_written:
            on_file_written(output_path)

        workspace_path = f"/workspace/candidates/{output_path.name}"
        fields_str = ", ".join(result.get("fields", []))
        sample_block = _format_sample_rows(result.get("sample_rows") or [])

        clamp_note = ""
        if small_first_clamped:
            clamp_note = (
                f"\n[Small-first: capped at {SMALL_FIRST_CAP} since project "
                f"has 0 rows. Commit these as rows now (filter inline if "
                f"needed), then call again for more. Pulling a bigger first "
                f"batch and never committing is the over-perfection failure "
                f"mode this cap exists to prevent.]\n"
            )

        return (
            f"Apify {actor_id}: {item_count} items returned.\n"
            f"File: {workspace_path}\n"
            f"Fields: {fields_str}\n"
            f"Cost: ${cost:.4f}\n"
            + clamp_note
            + (f"\nSample rows:\n{sample_block}\n" if sample_block else "")
            + "\nNext: commit candidates as rows (candidates_to_rows or "
              "code_exec with add_rows). Filter inline as you commit if "
              "needed. Don't call apify_call_actor again until you've "
              "landed rows from this batch."
        ), cost

    # ── Register namespace ───────────────────────────────────────────

    tools = [
        {
            "name": "search_actors",
            "description": (
                "Search 22,000+ pre-built scrapers in the Apify Store by platform "
                "or data type.\n\n"
                "Use 1-3 simple keywords: the platform name and data type.\n"
                "Good: 'Instagram posts', 'Reddit', 'Amazon products'\n"
                "Bad: 'Instagram posts profiles comments hashtags' (too many terms)\n"
                "Bad: 'data extraction scraping tools' (too generic)\n\n"
                "Returns actor names, descriptions, usage stats, and user counts."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search keywords (1-3 terms: platform + data type)",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max results (default 5, max 20)",
                    },
                },
                "required": ["query"],
            },
        },
        {
            "name": "actor_details",
            "description": (
                "Get full details for an Apify actor: description, pricing, usage "
                "stats, reviews, and complete input schema with parameter types, "
                "defaults, and examples. Always check this before running an actor."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "actor_id": {
                        "type": "string",
                        "description": "Actor ID in 'username/actor-name' format",
                    },
                },
                "required": ["actor_id"],
            },
        },
        {
            "name": "call_actor",
            "description": (
                "Run an Apify scraper. Pass the actor's input parameters directly "
                "(e.g. startUrls, query, maxItems — whatever the actor schema "
                "requires). The actor runs, results are automatically downloaded "
                "to a JSONL file. Returns file path and summary.\n\n"
                "Use actor_details first to see the input schema."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "actor_id": {
                        "type": "string",
                        "description": "Actor ID from search_actors",
                    },
                    "max_items": {
                        "type": "integer",
                        "description": (
                            "Hard cap on items downloaded. **ALWAYS pass this.** "
                            "First call on a new query: 10–25 (default 25 if "
                            "you somehow forget). Going higher requires the "
                            "user to have given an explicit count ('get me 100 X'). "
                            "Server enforces a cap of 50 on first calls when "
                            "the project has zero rows — fetching more before "
                            "committing anything is the over-perfection failure "
                            "mode. Refine via follow-up turns, not bigger first "
                            "fetches."
                        ),
                    },
                    "max_cost": {
                        "type": "number",
                        "description": "Max USD cost for this run (optional). Apify stops the actor if cost exceeds this.",
                    },
                    "timeout_secs": {
                        "type": "integer",
                        "description": (
                            "How long to wait for the actor to finish "
                            "before giving up (default 600). On timeout "
                            "we abort the Apify run so you don't keep "
                            "getting billed."
                        ),
                    },
                },
                "required": ["actor_id"],
                "additionalProperties": True,
            },
        },
    ]

    handlers = {
        "search_actors": search_actors,
        "actor_details": actor_details,
        "call_actor": call_actor,
    }

    registry.add_namespace(
        name="apify",
        description=NAMESPACE_DESCRIPTION,
        tools=tools,
        handlers=handlers,
    )


async def _schema_feedback_block(
    client: ApifyClient, actor_id: str
) -> Optional[str]:
    """Render the actor's input schema for LLM self-correction.

    Used both on HTTP-error responses and on silent 0-item runs (Apify
    accepts unknown fields and returns a successful empty run, so we
    have to nudge the model with the schema either way).
    """
    try:
        details = await client.get_actor_details(actor_id)
    except Exception:
        return None
    schema = (details or {}).get("input_schema")
    if not schema or not schema.get("properties"):
        return None
    req = set(schema.get("required", []))
    lines = [_format_schema_prop(n, p, n in req)
             for n, p in schema["properties"].items()]
    return "Expected input schema:\n" + "\n".join(lines)


def _format_sample_rows(rows: List[Dict[str, Any]]) -> str:
    """Render up to 3 sample rows with per-value truncation.

    Token safety net: cap total payload at ~4000 chars. Truncate string
    values to 500 chars, list/dict values to 300 chars of repr.
    """
    if not rows:
        return ""
    MAX_TOTAL = 4000
    out_parts: List[str] = []
    used = 0
    for i, row in enumerate(rows):
        compact: Dict[str, Any] = {}
        for k, v in row.items():
            if isinstance(v, str):
                compact[k] = v[:500] + ("…" if len(v) > 500 else "")
            elif isinstance(v, (list, dict)):
                s = json.dumps(v, default=str)
                compact[k] = s[:300] + ("…" if len(s) > 300 else "")
            else:
                compact[k] = v
        rendered = json.dumps(compact, default=str, indent=2)
        block = f"[row {i}]\n{rendered}"
        if used + len(block) > MAX_TOTAL:
            out_parts.append(f"[row {i}] (omitted — sample size cap reached)")
            break
        out_parts.append(block)
        used += len(block)
    return "\n".join(out_parts)


def _format_output_views(views: Dict[str, Any]) -> List[str]:
    """Render Apify dataset view columns as a flat field list.

    `views` is a dict of view_name → {transformation: {fields: [...]},
    display: {properties: {field: {label, format}}}}. We surface the first
    view with content. Field order comes from transformation.fields when
    present (publisher's intended order), else properties dict order.
    """
    for view in views.values():
        if not isinstance(view, dict):
            continue
        display = view.get("display") or {}
        properties = display.get("properties") or {}
        transformation = view.get("transformation") or {}
        ordered_fields = transformation.get("fields") or list(properties.keys())
        if not ordered_fields:
            continue
        out = []
        for field in ordered_fields:
            meta = properties.get(field) if isinstance(properties, dict) else None
            label = (meta or {}).get("label", "")
            fmt = (meta or {}).get("format", "")
            line = f"- **{field}**"
            if fmt:
                line += f" ({fmt})"
            if label and label.lower() != field.lower():
                line += f": {label}"
            out.append(line)
        return out
    return []


def _format_schema_prop(name: str, prop: Dict[str, Any], required: bool) -> str:
    """Format a single JSON Schema property for LLM consumption."""
    ptype = prop.get("type", "any")
    req = "REQUIRED" if required else "optional"
    desc = prop.get("description", "")
    line = f"- **{name}** ({ptype}, {req})"
    if desc:
        line += f": {desc[:1000]}"
    default = prop.get("default")
    prefill = prop.get("prefill")
    enum = prop.get("enum")
    if default is not None:
        line += f"  [default: {json.dumps(default, default=str)[:400]}]"
    if prefill is not None and prefill != default:
        line += f"  [example: {json.dumps(prefill, default=str)[:400]}]"
    if enum:
        vals = ", ".join(str(e) for e in enum[:30])
        line += f"  [values: {vals}]"
    return line
