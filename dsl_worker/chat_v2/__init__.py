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
                        "enum": ["none", "low", "medium", "high"],
                        "description": "none=nano, no tools (just a label from row text). low=mini + tools (one known call, e.g. FE email/phone). medium=5.5 + tools (standard research). high=5.5 + tools + higher effort (multi-step).",
                    },
                    "prompt": {"type": "string", "description": "Natural-language instruction the per-row agent follows."},
                    "columns_to_fill": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional. Defaults to all enrichment column names.",
                    },
                    "per_row_credit_cap": {
                        "type": "number",
                        "description": "REQUIRED. Cap on credits the cell agent can spend per row. Defaults by research level (classify 0.3, lookup 1.0, search 2.0, investigate 8.0); bump for known-expensive integrations — phone via FE ~10, email via FE ~1.5. The agent is killed mid-row if it tries to exceed this, so set it generously enough that the typical row completes.",
                    },
                },
                "required": ["research", "prompt", "per_row_credit_cap"],
                "additionalProperties": True,
            },
        },
        "required": ["table_id", "columns", "action"],
        "additionalProperties": True,
    }


# Canonical list of valid filter ops. Used in both the tool schema
# (so the model sees a closed set) and the handler validator (so
# unknown ops error out instead of silently no-op-ing).
FILTER_OPS = [
    # Text-ish
    "contains", "not_contains", "starts_with", "ends_with",
    "equals", "not_equals",
    "contains_any", "contains_all", "not_contains_any", "not_contains_all",
    "text_inc_exc",
    "in", "not_in",
    # Numeric / date
    "gt", "gte", "lt", "lte", "between",
    # Symbolic aliases — the handler accepts both; enum here lists the
    # word forms only so the agent has one canonical way.
    # (=, !=, >, >=, <, <= still work server-side as aliases.)
    # Null checks
    "is_null", "is_not_null",
]


def _filter_set_schema() -> Dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "table_id": {"type": "string"},
            "column": {"type": "string", "description": "Column name to filter on."},
            "op": {
                "type": "string",
                "enum": FILTER_OPS,
                "description": "Filter operator. Pick from this enum — anything else is rejected.",
            },
            "value": {
                "description": (
                    "Shape depends on op: scalar for most; list for contains_any/all + in/not_in; "
                    "[min,max] for between; {include:[],exclude:[]} for text_inc_exc; "
                    "null for is_null / is_not_null."
                ),
            },
        },
        "required": ["table_id", "column", "op"],
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
        "filter_set": (
            "Apply a non-destructive filter to a column. Returns matched count + sample.\n\n"
            "Args: {table_id, column, op, value}. `op` must be one of the documented "
            "ops below — anything else is rejected with an error so you can retry "
            "with a valid op. `value` shape depends on op (scalar / list / "
            "{include,exclude} / [min,max]).\n\n"
            "Allowed ops (match to column type — picking the wrong op for the type "
            "will be rejected):\n"
            "  Text / URL / email / enum:\n"
            "    contains            value=string   (case-insensitive substring)\n"
            "    not_contains        value=string\n"
            "    starts_with         value=string\n"
            "    ends_with           value=string\n"
            "    equals              value=string\n"
            "    not_equals          value=string\n"
            "    contains_any        value=[strings]  (OR across terms)\n"
            "    contains_all        value=[strings]  (AND across terms)\n"
            "    not_contains_any    value=[strings]\n"
            "    not_contains_all    value=[strings]\n"
            "    text_inc_exc        value={include:[strings], exclude:[strings]}  (Apollo-style)\n"
            "    in                  value=[strings]  (exact match against a set)\n"
            "    not_in              value=[strings]\n"
            "  Number / date:\n"
            "    >, gt               value=number\n"
            "    >=, gte             value=number\n"
            "    <, lt               value=number\n"
            "    <=, lte             value=number\n"
            "    between             value=[min, max]  (length-2 list)\n"
            "    equals              value=number\n"
            "    not_equals          value=number\n"
            "  Any column:\n"
            "    is_null             value=null\n"
            "    is_not_null         value=null\n\n"
            "Do NOT pass {type, min, max} or other unsupported shapes — use op + "
            "value as documented."
        ),
        "filter_clear": "Remove a filter from a column. Args: {table_id, column}.",
        "sort_set": "Set the active sort on a table. Args: table_id, column, direction (asc|desc, default desc). Single sort per table.",
        "sort_clear": "Remove the active sort on a table.",
        # Rows
        "row_inspect": "Read-only peek at rows.",
        "row_delete": "Delete rows by id. Approval-gated.",
        # Utility
        "code_exec": "Execute a Python snippet in the sandbox.",
        # web_search is the OpenAI hosted tool — added directly to the
        # Responses `tools` array in agent.py as {"type": "web_search"},
        # not registered here as a function. Keeping it out of
        # tool_descriptions so we don't double-list it with an empty
        # function shape.
        "suggest_replies": "Emit chip suggestions for the user's next move. Call at end of turn.",
        # Comments — the description thread visible in the table/column detail panel.
        "comment_on_table": "Append a short agent note to a table's description thread (visible in the table detail panel). Use sparingly — for non-obvious decisions or material changes the user should see ('Switched to Apollo because Google Maps capped at 60 results'). Args: table_id, body (markdown ok).",
        "comment_on_column": "Append a short agent note to a column's description thread. Args: table_id, column (name on the table), body.",
    }
    schema_for = {
        "enrichment_set": _enrichment_set_schema(),
        "filter_set": _filter_set_schema(),
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
