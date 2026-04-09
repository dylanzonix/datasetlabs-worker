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
) -> None:
    """Register the apify namespace on a ToolRegistry."""
    if file_counter is None:
        file_counter = [0]

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
            lines.append(
                f"**{r['actor_id']}** — {r['title']}\n"
                f"  {r['description'][:300]}\n"
                f"  Runs: {r['total_runs']:,} | Users: {r['total_users']:,}"
            )

        return (
            f"Found {len(results)} Apify actors for \"{query}\":\n\n"
            + "\n\n".join(lines)
            + "\n\nUse actor_details to get the input schema and pricing before running."
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
        parts.append(details["description"][:500])

        if details.get("readme_summary"):
            parts.append("")
            parts.append(details["readme_summary"])

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

        return "\n".join(parts), 0.0

    # ── call_actor ───────────────────────────────────────────────────

    async def call_actor(args: Dict) -> Tuple[str, float]:
        actor_id = args.get("actor_id", "")
        if not actor_id:
            return "Error: actor_id is required.", 0.0

        # Everything except actor_id and max_items is actor input
        known_keys = {"actor_id", "max_items"}
        run_input = {k: v for k, v in args.items() if k not in known_keys}
        max_items = args.get("max_items")

        logger.info(f"[apify] Running actor: {actor_id}")

        # Start async run
        try:
            import httpx
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            }
            actor_path = actor_id.replace("/", "~")
            async with httpx.AsyncClient(timeout=30.0) as http:
                resp = await http.post(
                    f"https://api.apify.com/v2/acts/{actor_path}/runs",
                    headers=headers,
                    json=run_input,
                    params={"timeout": 0},  # async start
                )

            if resp.status_code not in (200, 201):
                # Try to get input schema for error feedback
                error_msg = f"Apify actor {actor_id} failed to start: HTTP {resp.status_code}"
                try:
                    err_body = resp.json()
                    error_msg += f"\n{err_body.get('error', {}).get('message', resp.text[:200])}"
                except Exception:
                    pass

                # Fetch schema to help LLM self-correct
                try:
                    details = await client.get_actor_details(actor_id)
                    schema = (details or {}).get("input_schema")
                    if schema and schema.get("properties"):
                        req = set(schema.get("required", []))
                        lines = [_format_schema_prop(n, p, n in req)
                                 for n, p in schema["properties"].items()]
                        error_msg += "\n\nExpected input schema:\n" + "\n".join(lines)
                except Exception:
                    pass

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
        )

        if "error" in result:
            return f"Apify actor {actor_id} error: {result['error']}", 0.0

        cost = result.get("cost_usd", 0.0)
        item_count = result.get("item_count", 0)

        if item_count == 0:
            return f"Apify actor {actor_id} returned 0 items.", cost

        workspace_path = f"/workspace/candidates/{output_path.name}"
        fields_str = ", ".join(result.get("fields", []))

        return (
            f"Apify {actor_id}: {item_count} items returned.\n"
            f"File: {workspace_path}\n"
            f"Fields: {fields_str}\n"
            f"Cost: ${cost:.4f}\n\n"
            f"Next: submit_candidates with this file, or inspect with code_exec."
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
                        "description": "Limit items downloaded from the dataset (optional)",
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


def _format_schema_prop(name: str, prop: Dict[str, Any], required: bool) -> str:
    """Format a single JSON Schema property for LLM consumption."""
    ptype = prop.get("type", "any")
    req = "REQUIRED" if required else "optional"
    desc = prop.get("description", "")
    line = f"- **{name}** ({ptype}, {req})"
    if desc:
        line += f": {desc[:300]}"
    default = prop.get("default")
    prefill = prop.get("prefill")
    enum = prop.get("enum")
    if default is not None:
        line += f"  [default: {json.dumps(default, default=str)[:150]}]"
    if prefill is not None and prefill != default:
        line += f"  [example: {json.dumps(prefill, default=str)[:150]}]"
    if enum:
        vals = ", ".join(str(e) for e in enum[:15])
        line += f"  [values: {vals}]"
    return line
