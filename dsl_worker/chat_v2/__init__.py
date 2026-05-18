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


# Canonical column types. Shared across column-passing tools so the
# model sees one closed set everywhere it picks columns.
COLUMN_TYPES = ["text", "number", "url", "email", "date", "enum"]

# Canonical column display formats. Surfaced via tool schema so the agent
# treats it as part of the contract, not a prompt afterthought. Limited
# to formats the FE actually renders — extending here without wiring the
# FE side just produces silent no-ops.
COLUMN_FORMATS = ["percent", "currency", "currency_compact"]


def _column_item_schema(*, include_source_field: bool) -> Dict[str, Any]:
    """Shape of one column entry, shared by table_create / column_map_set /
    enrichment_set. The `format` enum is the one that matters here — without
    surfacing it via schema the agent treats it as optional prompt prose and
    skips it; with it visible, the model picks it up much more reliably.
    """
    props: Dict[str, Any] = {
        "name": {"type": "string", "description": "Display name shown in the column header. Title Case preferred."},
        "type": {"type": "string", "enum": COLUMN_TYPES},
        "format": {
            "type": "string",
            "enum": COLUMN_FORMATS,
            "description": (
                "Optional display format for numbers. Set when raw values would read "
                "as noise: `percent` for decimal ratios (-0.02, 0.67); `currency` for "
                "everyday dollar amounts ($1,234.56); `currency_compact` for "
                "USD revenue/funding/valuation ($1.2M). Leave unset for years, IDs, "
                "counts, scores, etc."
            ),
        },
    }
    required = ["name"]
    if include_source_field:
        props["source_field"] = {
            "type": "string",
            "description": (
                "Path into the raw source row. Plain key (`name`), dotted (`employment.current.title`), "
                "or array fan-out (`founders[].email`)."
            ),
        }
        required.append("source_field")
    return {
        "type": "object",
        "properties": props,
        "required": required,
        "additionalProperties": True,
    }


def _table_create_schema() -> Dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "source": {"type": "string", "description": "Adapter name (apollo_companies, fullenrich_people, google_maps, apify_actor:<id>, web_harvest, browser_use, file)."},
            "query_params": {"type": "object", "description": "Source-specific query. See system prompt source cards for shapes."},
            "name": {"type": "string", "description": "Short Title Case table name (2-5 words)."},
            "intent": {"type": "string", "description": "Optional one-liner on what the user wants from this table."},
            "columns": {
                "type": "array",
                "description": "Optional. If omitted, server raw-passthroughs every top-level key as a text column.",
                "items": _column_item_schema(include_source_field=True),
            },
            "n": {"type": "integer", "description": "Row count target. Defaults to 100."},
        },
        "required": ["source", "query_params"],
        "additionalProperties": True,
    }


def _column_map_set_schema() -> Dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "table_id": {"type": "string"},
            "columns": {
                "type": "array",
                "description": "Replacement column set. Every row gets re-derived from raw_row through the new mapping.",
                "items": _column_item_schema(include_source_field=True),
            },
            "dedup_key_column": {"type": "string", "description": "Optional. Column to use as the dedup key for table_extend overlap."},
        },
        "required": ["table_id", "columns"],
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
                "description": "One or more columns to add and fill. [{name, type, format?}].",
                "items": _column_item_schema(include_source_field=False),
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


# Canonical filter ops — exactly the ops the FE filter panel can render
# and edit. Shrunk from 19 → 7 to enforce a 1:1 contract: every filter
# the AI sets must be expressible in the FE's filter UI, and vice versa.
# Handler normalizes legacy ops (`contains`, `>=`, `in`, etc.) into one
# of these on write so old DB rows + agent slips still work.
FILTER_OPS = [
    "text_inc_exc",   # text / url / email — value: {include: [strings], exclude: [strings]}
    "is_any_of",      # enum (or any column) — value: [strings]
    "between",        # number / date — value: [min, max]
    "gte",            # number / date — value: number-or-iso-date
    "lte",            # number / date — value: number-or-iso-date
    "is_null",        # any column — value: null
    "is_not_null",    # any column — value: null
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
                "description": "Filter operator. See tool description for the value shape per op.",
            },
            "value": {
                "description": (
                    "Shape depends on op: "
                    "[min,max] for between; "
                    "number/iso-date for gte/lte; "
                    "[strings] for is_any_of; "
                    "{include:[],exclude:[]} for text_inc_exc; "
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
        "table_create": "Create a table from a source. Fetches rows and commits them with raw passthrough columns (every top-level source key becomes a snake_case text column). If the fetch fails or returns 0 rows, nothing is written — try a different actor/query. Always follow with column_map_set to clean up names, types, and formats unless the raw columns are already what the user wants. Args: source, query_params, name (2-5 words, Title Case).",
        "table_extend": "Pull MORE rows into an EXISTING table with a non-overlapping next slice. Args: table_id, query_params (the new slice — e.g. next batch / page / date window). Reuses the table's existing column map automatically.",
        "table_delete": "Delete a table and all its rows + enrichments. Approval-gated.",
        # Apify discovery
        "apify_search_actors": "Discover Apify actors matching a query. Returns lightweight summaries.",
        "apify_actor_details": "Read an actor's full input_schema, output preview, and pricing.",
        # Columns/enrichments
        "column_map_set": "Edit columns on an existing table — rename a column, add one mapped from another source field, drop one. Args: table_id, columns ([{name, source_field, type}]). table_create already commits with its own columns; only use this to revise after the fact.",
        "enrichment_set": "Define or refine an enrichment. Does NOT auto-run — call enrichment_run after to fill cells. Args: table_id, columns, action: {research, prompt, per_row_credit_cap?}.",
        "enrichment_run": (
            "Run an enrichment over a scope of rows. Approval-gated — user sees an "
            "estimated-cost card with the row count before it executes. "
            "Args: {enrichment_id, scope, overwrite?}.\n\n"
            "Scope shapes:\n"
            "  {type: 'all_unfilled'} — default. Every row missing at least one of "
            "the enrichment's target columns.\n"
            "  {type: 'first_n', first_n: 10} — first N rows of the table.\n"
            "  {type: 'row_ids', row_ids: ['...']} — explicit row id list.\n"
            "  {type: 'filtered', filters: [{column, op, value}, ...]} — restrict to "
            "rows that pass the explicit filters. Same {column, op, value} shape as "
            "filter_set (canonical 7-op set). Use this when the user has a filter set "
            "and wants the enrichment scoped to the visible rows — copy the filters "
            "from project_state into scope.filters explicitly (do NOT rely on the "
            "table's active filter being read implicitly; scope.filters IS the filter "
            "set that gets applied)."
        ),
        # Filters
        "filter_set": (
            "Apply a non-destructive filter to a column. Returns matched count + "
            "sample_kept + sample_excluded for sanity-check.\n\n"
            "Args: {table_id, column, op, value}. `op` must be one of the 7 ops "
            "below. The set is intentionally small: every op here maps 1:1 to a "
            "filter UI the user can see and edit in the FE filter panel.\n\n"
            "Pick by column type:\n"
            "  text / url / email column:\n"
            "    text_inc_exc   value={include:[strings], exclude:[strings]}\n"
            "                   include terms OR'd together (case-insensitive substring).\n"
            "                   exclude terms also OR'd. include AND exclude AND'd.\n"
            "                   Use single-element lists for one term. Same op covers\n"
            "                   'contains', 'starts_with', 'ends_with', 'not_contains'.\n"
            "  enum column (or any column when filtering to a specific set of values):\n"
            "    is_any_of      value=[strings]   (matches any of the listed values exactly)\n"
            "  number / date column:\n"
            "    between        value=[min, max]  (inclusive on both ends)\n"
            "    gte            value=number_or_iso_date  (one-sided range, inclusive)\n"
            "    lte            value=number_or_iso_date  (one-sided range, inclusive)\n"
            "  any column (empty-cell check):\n"
            "    is_null        value=null   (cell is empty)\n"
            "    is_not_null    value=null   (cell has a value)\n\n"
            "Use `gte`/`lte` (no strict `gt`/`lt` — round the boundary if needed). "
            "For 'value contains X', use `text_inc_exc {include:[\"X\"]}` not a "
            "contains/starts_with shape. The handler will rewrite legacy shapes to "
            "canonical but emit the canonical form to avoid lossy conversions."
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
        "table_create": _table_create_schema(),
        "column_map_set": _column_map_set_schema(),
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
