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
from dsl_worker.chat_v2.prompt import build_system_prompt
from dsl_worker.chat_v2.project_state import build_project_state


HANDLERS = {
    **_table_handlers,
    **_light_handlers,
    **_enrichment_handlers,
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


def _build_tool_defs() -> List[Dict[str, Any]]:
    """OpenAI function tool definitions for the 15-tool orchestrator surface."""
    tool_descriptions = {
        # Tables
        "table_create": "Create a new table backed by one source query and fetch the first batch.",
        "table_extend": "Run another query against the same source, appending rows. Always approval-gated.",
        "table_delete": "Delete a table and all its rows + enrichments. Approval-gated.",
        # Apify discovery
        "apify_search_actors": "Discover Apify actors matching a query. Returns lightweight summaries.",
        "apify_actor_details": "Read an actor's full input_schema, output preview, and pricing.",
        # Columns/enrichments
        "column_map_set": "Commit (or update) the field→column mapping for a table. Required after table_create on unpredictable sources.",
        "enrichment_set": "Define or refine an enrichment. Runs on the first 10 unfilled rows.",
        "enrichment_run": "Extend an enrichment to more rows. Approval-gated.",
        # Filters
        "filter_set": "Apply a non-destructive filter to a column. Returns matched count + sample.",
        "filter_clear": "Remove a filter from a column.",
        # Rows
        "row_inspect": "Read-only peek at rows.",
        "row_delete": "Delete rows by id. Approval-gated.",
        # Utility
        "code_exec": "Execute a Python snippet in the sandbox.",
        "web_search": "Quick web search — for scouting, not row-building. Use web_harvest source for table-building from web.",
        "suggest_replies": "Emit chip suggestions for the user's next move. Call at end of turn.",
    }
    return [
        {
            "type": "function",
            "function": {
                "name": name,
                "description": desc,
                "parameters": _generic_schema(),
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
