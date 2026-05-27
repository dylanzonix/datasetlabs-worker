"""v-next chat module.

Tool registry assembled here. Other modules contribute partial HANDLERS dicts;
this file merges them so the streaming loop has a single map.

  HANDLERS — {tool_name: async (args, ctx) -> (result_dict, cost_usd)}
  TOOL_DEFS — OpenAI function-calling tool definitions (JSON schemas)

Usage:
    from dsl_worker.chat import HANDLERS, TOOL_DEFS, build_system_prompt, build_project_state
"""

from __future__ import annotations

# Load .env BEFORE any submodule import — adapters in dsl_worker.sources
# read their API keys via os.getenv at module import (e.g. ApifyActorAdapter
# captures self.api_key in __init__). Without this load happening first, a
# key that exists only in .env (not the shell env) reads as None at import
# and the adapter stays inert for the life of the process. app.py also
# calls load_dotenv but it runs AFTER this package init — too late.
import os as _os
from dotenv import load_dotenv as _load_dotenv
# Path is relative to whatever cwd uvicorn was started from. Worker's
# canonical cwd is the worker repo root which has .env at its top.
_load_dotenv(".env", override=True)

from typing import Any, Dict, List

from dsl_worker.chat.tools import HANDLERS as _table_handlers, ToolContext
from dsl_worker.chat.light_tools import HANDLERS as _light_handlers
from dsl_worker.chat.enrichment import HANDLERS as _enrichment_handlers
from dsl_worker.chat.comments import HANDLERS as _comment_handlers
from dsl_worker.chat.background_tasks import HANDLERS as _bg_handlers
from dsl_worker.chat.prompt import build_system_prompt
from dsl_worker.chat.project_state import build_project_state


HANDLERS = {
    **_table_handlers,
    **_light_handlers,
    **_enrichment_handlers,
    **_comment_handlers,
    **_bg_handlers,
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
        "pinned": {
            "type": "boolean",
            "description": (
                "Optional. When true, the column is frozen to the left so it stays "
                "visible as the user scrolls horizontally. Use sparingly — at most "
                "ONE pinned column per table by default (the row identifier — Name, "
                "Company, Title, Place, etc.). Pinning more than one eats horizontal "
                "space and defeats the point. Default false."
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
            "source": {"type": "string", "description": "Adapter name (apollo_companies, fullenrich_people, google_maps, apify_actor:<id>, web_harvest, browser_use, file, llm)."},
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
            "insert_before": {
                "type": "string",
                "description": (
                    "Optional. Existing enrichment short_id (e.g. 't1e2') the new one should be inserted BEFORE. "
                    "Use when this enrichment should run as a gate before downstream ones — e.g. you want to add "
                    "a qualification check ahead of contact-info lookups already created. Omit to append at the end."
                ),
            },
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
                        "enum": ["classify", "research", "deep"],
                        "description": (
                            "Three tiers. "
                            "`classify` = nano model, NO tools — decides a label from the row's existing text "
                            "(e.g. 'is this a SaaS company yes/no', 'sentiment of bio'). "
                            "`research` = gpt-5.4-mini + all tools (web search, FE, Apollo, browser_use) — "
                            "DEFAULT. Most lookups (email, phone, single-fact, basic synthesis). "
                            "`deep` = gpt-5.5 + all tools — smarter model. Use when the task needs "
                            "multi-step reasoning, ambiguity resolution (sketchy/borderline cases), "
                            "or higher-stakes verification. Don't reach for it by default; pick when "
                            "mini would plausibly miss nuance."
                        ),
                    },
                    "prompt": {"type": "string", "description": "Natural-language instruction the per-row agent follows."},
                    "columns_to_fill": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional. Defaults to all enrichment column names.",
                    },
                    "depends_on": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "Optional. Column names this enrichment needs as inputs. Rows where ANY listed column "
                            "is empty are skipped at run time (no credits spent on guaranteed-fail rows). "
                            "Example: a 'Founder Email' enrichment depends_on ['Founder Name', 'Domain'] — "
                            "rows missing either get skipped until those columns are filled."
                        ),
                    },
                    "per_row_credit_cap": {
                        "type": "number",
                        "description": (
                            "REQUIRED. Cap on credits the cell agent can spend per row. Defaults: "
                            "classify=0.3, research=5.0. Bump for known-expensive integrations — "
                            "phone via FE ~10, browser_use chains ~15. The agent is killed mid-row "
                            "if it tries to exceed this, so set it generously enough that the typical row completes."
                        ),
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
# and edit. Enforces a 1:1 contract: every filter the AI sets must be
# expressible in the FE's filter UI, and vice versa.
# Handler normalizes legacy ops (`contains`, `>=`, `in`, etc.) into one
# of these on write so old DB rows + agent slips still work.
#
# Note: `is_null` was intentionally removed — users almost never want
# "show me only the empty rows" as a visible filter. To target unfilled
# rows for an enrichment run, use scope `{type: "all_unfilled"}` which
# the server resolves against the enrichment's target columns. SQL +
# Python predicate handlers still tolerate `is_null` for any legacy DB
# rows that have it set, but the AI can no longer construct new ones.
FILTER_OPS = [
    "text_inc_exc",   # text / url / email — value: {include: [strings], exclude: [strings]}
    "is_any_of",      # enum (or any column) — value: [strings]
    "between",        # number / date — value: [min, max]
    "gte",            # number / date — value: number-or-iso-date
    "lte",            # number / date — value: number-or-iso-date
    "is_not_null",    # any column — value: null  (hide rows where this cell is empty)
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
                    "null for is_not_null."
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
        "table_create": (
            "Create a table from a source. Fetches rows and commits them with raw passthrough columns "
            "(every top-level source key becomes a snake_case text column). If the fetch fails or returns "
            "0 rows, nothing is written — try a different actor/query. Always follow with column_map_set "
            "to clean up names, types, and formats unless the raw columns are already what the user wants. "
            "Args: source, query_params, name (2-5 words, Title Case). "
            "Optional `wait: false` returns immediately with {status:'running', task_id:'bt<N>'} and "
            "runs the fetch in a background task — use this when you're emitting multiple table_creates "
            "in parallel so subsequent iterations aren't blocked on the slowest source."
        ),
        "table_extend": (
            "Pull MORE rows into an EXISTING table with a non-overlapping next slice. "
            "Args: table_id, query_params (the new slice — e.g. next batch / page / date window). "
            "Reuses the table's existing column map automatically. "
            "Optional `wait: false` backgrounds the fetch — same semantics as table_create."
        ),
        "table_delete": "Delete a table and all its rows + enrichments. Approval-gated.",
        # Apify discovery
        "apify_search_actors": (
            "Discover Apify actors matching a query. Returns up to 8 actors "
            "sorted by total_runs descending — the highest-traffic / most "
            "battle-tested ones first. Prefer actors with >10k total_runs "
            "when they exist; lower-traffic actors are often unmaintained "
            "or have broken pagination (a niche YC actor that doesn't crawl "
            "past the first page, etc.). Inspect input_schema via "
            "apify_actor_details before calling table_create — don't guess."
        ),
        "apify_actor_details": "Read an actor's full input_schema, output preview, and pricing.",
        # Columns/enrichments
        "column_map_set": "Edit columns on an existing table — rename a column, add one mapped from another source field, drop one. Args: table_id, columns ([{name, source_field, type}]). table_create already commits with its own columns; only use this to revise after the fact.",
        "enrichment_set": "Define or refine an enrichment. Does NOT auto-run — call enrichment_run after to fill cells. Args: table_id, columns, action: {research, prompt, per_row_credit_cap?}.",
        "enrichment_run": (
            "Run an enrichment over a scope of rows. Approval-gated — user sees an "
            "estimated-cost card with the row count before it executes. "
            "Args: {enrichment_id, scope, overwrite?}.\n\n"
            "Scope shapes:\n"
            "  {type: 'all_unfilled', first_n?: N} — default. Every row missing at "
            "least one of the enrichment's target columns. Optional first_n CAPS the "
            "result to the first N rows by seq.\n"
            "  {type: 'first_n', first_n: 10} — first N rows of the table (no filter).\n"
            "  {type: 'row_ids', row_ids: ['...']} — explicit row id list.\n"
            "  {type: 'filtered', filters: [{column, op, value}, ...], first_n?: N} — "
            "restrict to rows that pass the explicit filters. Same {column, op, value} "
            "shape as filter_set (canonical 7-op set). Optional first_n CAPS the result "
            "to the first N matching rows by seq. Use this when the user has a filter "
            "set and wants the enrichment scoped to the visible rows — copy the filters "
            "from project_state into scope.filters explicitly.\n\n"
            "IMPORTANT — 'do 10 more' style asks: when the user names a specific batch "
            "size, ALWAYS pass first_n. Without it, 'filtered' runs every matching row "
            "and 'all_unfilled' runs every unfilled row — a 100-row hit will run all 100 "
            "even if the user asked for 10. Shape: "
            "{type: 'all_unfilled', first_n: 10} (server picks the unfilled rows for the "
            "enrichment's target columns automatically).\n\n"
            "Optional `wait: false` returns immediately with {status:'running', task_id:'bt<N>'} once "
            "the user has approved. The cell loop runs in the background — use this when the agent has "
            "more work to do on other tables / enrichments while this run completes. Monitor via "
            "task_status / task_wait. Approval is gated upstream; background mode does NOT bypass the "
            "cost confirmation card."
        ),
        # Background-task monitoring
        "task_status": (
            "Instant peek at one or more background tasks (the ones started via wait=false on "
            "table_create / table_extend / enrichment_run). Args: {task_ids: ['bt1', 'bt2', ...]}. "
            "Returns per-task status (running / complete / error / cancelled), cost so far, "
            "started_at, finished_at, and a result preview. Use this BETWEEN other tool calls when you "
            "just want to check on progress without blocking."
        ),
        "task_wait": (
            "Block until one or all of the listed background tasks finish (or until timeout). "
            "Args: {task_ids: ['bt1', ...], mode: 'all'|'any' (default 'all'), timeout_s: number "
            "(default 300, max 600)}. Returns the same per-task state shape as task_status, plus "
            "`all_done` and `timed_out`. Use this when you need a result before proceeding — e.g. "
            "you backgrounded 3 table_creates and the next move depends on what they returned. "
            "On timeout, re-call to keep waiting, or call task_status to peek without blocking."
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
            "  any column (hide empty cells):\n"
            "    is_not_null    value=null   (cell has a value)\n"
            "                   Use this to hide rows where a column is unfilled —\n"
            "                   the common case after a classification enrichment\n"
            "                   that leaves some cells blank. The user's UI has a\n"
            "                   matching 'Hide empty cells' checkbox.\n\n"
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
        "row_delete": "Delete rows. Args: {table_id, row_ids?: [uuid, ...], filters?: [{column, op, value}, ...]}. Either row_ids or filters required. Filters use the same ops as filter_set. Approval-gated.",
        # Utility
        "code_exec": (
            "Execute a Python snippet in an isolated sandbox. "
            "Args: {code, files?, table_id?}. "
            "files: list of file_id UUIDs (from project uploads) or candidate filenames to inject — "
            "each file is placed in /workspace/ under its original filename. "
            "table_id: when set, `import dsl_tools; dsl_tools.add_rows([...])` in the code "
            "bulk-inserts parsed rows into that table after execution finishes."
        ),
        # web_search is the OpenAI hosted tool — added directly to the
        # Responses `tools` array in agent.py as {"type": "web_search"},
        # not registered here as a function. Keeping it out of
        # tool_descriptions so we don't double-list it with an empty
        # function shape.
        "suggest_replies": "Emit chip suggestions for the user's next move. Call at end of turn.",
        "plan_options": (
            "Pause the turn to ask the user to pick between 2-4 explicit "
            "options before continuing. The call BLOCKS until the user "
            "clicks a button on the FE card; tool returns {chosen: '<key>'} "
            "for the selected option. "
            "Args: {question: str, options: [{label: str, key: str, "
            "description?: str}, ...]} — 2 to 4 options. "
            "USE SPARINGLY. Only when picking wrong would meaningfully "
            "diverge from the user's intent (e.g. 'GSA auctions' could be "
            "Treasury / IRS / US Marshals — different sites, different "
            "column shapes; the user has to pick). Don't ask if the right "
            "answer can be inferred from context, the message itself, or "
            "an existing table — that's friction."
        ),
        "load_skill": (
            "Load the playbook for a named skill from the directory listed under "
            "'# Skills' in the system prompt. Returns the full body of that skill "
            "as a string. Call only when one of the listed skills clearly matches "
            "the current task; most tasks won't need any skill. Args: {name}."
        ),
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
