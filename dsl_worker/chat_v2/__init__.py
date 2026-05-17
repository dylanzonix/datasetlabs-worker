"""v-next chat module.

Tool registry assembled here. Other modules contribute partial HANDLERS dicts;
this file merges them so the streaming loop has a single map.

  HANDLERS — {tool_name: async (args, ctx) -> (result_dict, cost_usd)}
  TOOL_DEFS — OpenAI function-calling tool definitions (JSON schemas)

Usage:
    from dsl_worker.chat_v2 import HANDLERS, TOOL_DEFS, build_system_prompt, build_project_state
"""

from __future__ import annotations

from typing import Any, Dict, List

from dsl_worker.chat_v2.tools import HANDLERS as _table_handlers, ToolContext
from dsl_worker.chat_v2.light_tools import HANDLERS as _light_handlers
from dsl_worker.chat_v2.enrichment import HANDLERS as _enrichment_handlers
from dsl_worker.chat_v2.comments import HANDLERS as _comment_handlers
from dsl_worker.chat_v2.prompt import build_system_prompt
from dsl_worker.chat_v2.project_state import build_project_state


HANDLERS = {
    **_table_handlers,
    **_light_handlers,
    **_enrichment_handlers,
    **_comment_handlers,
}


# Generic-arg shape so we don't blow up the prompt with full per-tool schemas
# for v1. Source filter cards in the system prompt teach the agent the
# concrete params; the tool schemas accept anything as additionalProperties.
def _generic_schema() -> Dict[str, Any]:
    return {
        "type": "object",
        "properties": {},
        "additionalProperties": True,
    }


def _enrichment_set_schema() -> Dict[str, Any]:
    """Tighter schema for enrichment_set so the agent learns the action shape.

    additionalProperties=True keeps it lenient — unknown fields warn server-side
    but never block the run. Required fields are the bare minimum.
    """
    return {
        "type": "object",
        "properties": {
            "table_id": {"type": "string", "description": "Table to enrich."},
            "enrichment_id": {"type": "string", "description": "Pass when refining an existing enrichment."},
            "name": {"type": "string", "description": "Short name shown in column header."},
            "columns": {
                "type": "array",
                "description": "One or more columns to add and fill. [{name, type}].",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "type": {"type": "string", "enum": ["text", "number", "url", "email", "date", "enum"]},
                    },
                    "required": ["name"],
                    "additionalProperties": True,
                },
            },
            "action": {
                "type": "object",
                "properties": {
                    "research": {
                        "type": "string",
                        "enum": ["classify", "lookup", "search", "investigate"],
                        "description": "classify=nano, no tools. lookup=mini + tools (one known call, e.g. FE email/phone). search=5.5 + tools (standard research). investigate=5.5 + tools + higher effort (multi-step).",
                    },
                    "prompt": {"type": "string", "description": "Natural-language instruction the per-row agent follows."},
                    "columns_to_fill": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional. Defaults to all enrichment column names.",
                    },
                    "per_row_credit_cap": {
                        "type": "number",
                        "description": "Optional. Defaults from research level. Bump for phone (~10) and email (~1.5) enrichments.",
                    },
                },
                "required": ["research", "prompt"],
                "additionalProperties": True,
            },
        },
        "required": ["table_id", "columns", "action"],
        "additionalProperties": True,
    }


def _build_tool_defs() -> List[Dict[str, Any]]:
    """OpenAI function tool definitions for the orchestrator surface."""
    tool_descriptions = {
        # Tables
        "table_create": "Create a table from a source. Atomic: fetches rows, system internally picks human-readable columns from the actual row shape + table intent, commits. If the fetch fails or returns 0 rows, nothing is written — try a different actor/query. Args: source, query_params, name (2-5 words, Title Case), intent (optional one-liner describing what the user wants — helps the column picker).",
        "table_extend": "Pull MORE rows into an EXISTING table with a non-overlapping next slice. Args: table_id, query_params (the new slice — e.g. next batch / page / date window). Reuses the table's existing column map automatically.",
        "table_delete": "Delete a table and all its rows + enrichments. Approval-gated.",
        # Apify discovery
        "apify_search_actors": "Discover Apify actors matching a query. Returns lightweight summaries.",
        "apify_actor_details": "Read an actor's full input_schema, output preview, and pricing.",
        # Columns/enrichments
        "column_map_set": "Edit columns on an existing table — rename a column, add one mapped from another source field, drop one. Args: table_id, columns ([{name, source_field, type}]). table_create already commits with its own columns; only use this to revise after the fact.",
        "enrichment_set": "Define or refine an enrichment. Does NOT auto-run — call enrichment_run after to fill cells. Args: table_id, columns, action: {research, prompt, per_row_credit_cap?}.",
        "enrichment_run": "Run an enrichment over a scope of rows. Approval-gated — user sees an estimated-cost card before it executes.",
        # Filters
        "filter_set": "Apply a non-destructive filter to a column. Returns matched count + sample.",
        "filter_clear": "Remove a filter from a column.",
        "sort_set": "Set the active sort on a table. Args: table_id, column, direction (asc|desc, default desc). Single sort per table.",
        "sort_clear": "Remove the active sort on a table.",
        # Rows
        "row_inspect": "Read-only peek at rows.",
        "row_delete": "Delete rows by id. Approval-gated.",
        # Utility
        "code_exec": "Execute a Python snippet in the sandbox.",
        "web_search": "Quick web search — for scouting, not row-building. Use web_harvest source for table-building from web.",
        "suggest_replies": "Emit chip suggestions for the user's next move. Call at end of turn.",
        # Comments — the description thread visible in the table/column detail panel.
        "comment_on_table": "Append a short agent note to a table's description thread (visible in the table detail panel). Use sparingly — for non-obvious decisions or material changes the user should see ('Switched to Apollo because Google Maps capped at 60 results'). Args: table_id, body (markdown ok).",
        "comment_on_column": "Append a short agent note to a column's description thread. Args: table_id, column (name on the table), body.",
    }
    schema_for = {
        "enrichment_set": _enrichment_set_schema(),
    }
    return [
        {
            "type": "function",
            "function": {
                "name": name,
                "description": desc,
                "parameters": schema_for.get(name, _generic_schema()),
            },
        }
        for name, desc in tool_descriptions.items()
    ]


TOOL_DEFS = _build_tool_defs()


__all__ = [
    "HANDLERS",
    "TOOL_DEFS",
    "ToolContext",
    "build_system_prompt",
    "build_project_state",
]
