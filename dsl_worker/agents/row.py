"""
Row generator agent — generates a single dataset row from a candidate.

V8: Simplified. Row generator receives:
- Raw user conversation (same as orchestrator sees)
- Schema
- Orchestrator instructions (what to do with each candidate)
- The candidate (string or dict from extractor)

set_column() checks identity columns against the submitted rows store
and returns similar existing rows so the LLM can detect duplicates early.
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import jsonschema

from dsl_worker.agents.base import AgentConversation
from dsl_worker.agents.tools import ToolRegistry
from dsl_worker.billing.tracked_client import TrackedOpenAIClient
from dsl_worker.infra.research_tools import ResearchTools, ResearchScope

logger = logging.getLogger(__name__)

MAX_GENERATION_TURNS = 30
READ_FILE_LIMIT = 30_000


@dataclass
class GeneratedRow:
    """Result of row generation."""
    success: bool
    row: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    cost_usd: float = 0.0
    skipped: bool = False
    skip_reason: str = ""


ROW_GENERATOR_SYSTEM_PROMPT = """\
You are generating a single dataset row.

## What the user wants

<conversation>
{conversation}
</conversation>

## Schema

<schema>
{schema_str}
</schema>

## Instructions

{instructions}

## Your candidate

{candidate}

## How to work

1. Read the candidate and instructions.
2. Fill in what you already know from the candidate using set_column().
   - For identity columns (e.g. name, url, email), fill these FIRST. \
If the response shows similar existing rows, check if this is a duplicate \
and call skip_row(reason="duplicate: ...") if so.
3. Research what you still need. Use primary sources. Verify claims that matter.
4. Fill remaining columns with set_column().
5. Call submit_row() when all columns are filled.

## Rules

- Output via tool calls ONLY. Text responses are ignored.
- If the candidate is a dead end (broken URL, entity doesn't exist, doesn't qualify), \
call skip_row(reason="...").
- If you can't find information after 2-3 attempts, note it in the column rather than \
making something up.

## Tools

- set_column(name, value): Set a column value. For identity columns, returns similar \
existing rows — check them for duplicates.
- append_to_column(name, value): Append to a column (json arrays or strings).
- clear_column(name): Clear a column to start over.
- submit_row(): Submit the completed row.
- skip_row(reason): Skip this candidate entirely.
- brave_search(query): Search the web.
- open(url): View a page.
- find(ref_id, pattern): Search within a loaded page.
- click(ref_id, link_id): Follow a link.
- code_exec(script, description): Execute Python.
- interact(url_or_ref_id, task): Browser agent for anti-bot bypass only.
- read_file(path): Read a workspace file.
"""


class RowGeneratorAgent:
    """
    Generates a single dataset row from a candidate.

    Usage:
        agent = RowGeneratorAgent(
            openai_client=tracked_client,
            model="gpt-5.2",
            workspace_dir=Path("/workspace"),
            chat_history=[...],
            identity_columns=["url", "name"],
            submitted_rows=shared_list,
            submitted_rows_lock=shared_lock,
            ...
        )
        result = await agent.generate(
            instructions="You will be given a podcast...",
            candidate={"name": "XYZ Podcast", "url": "https://..."},
            schema=[{"name": "podcast_name", "type": "string"}, ...],
        )
    """

    def __init__(
        self,
        openai_client: TrackedOpenAIClient,
        model: str,
        workspace_dir: Path,
        chat_history: Optional[List[Dict[str, str]]] = None,
        identity_columns: Optional[List[str]] = None,
        submitted_rows: Optional[List[Dict[str, Any]]] = None,
        submitted_rows_lock: Optional[Any] = None,
        brave_api_key: Optional[str] = None,
        sandbox: Optional[Any] = None,
        stop_checker: Optional[Callable[[], bool]] = None,
        stop_event: Optional[asyncio.Event] = None,
        blob_service_client: Optional[Any] = None,
        project_id: Optional[Any] = None,
        uploaded_file_urls: Optional[Dict[str, str]] = None,
        mcp_tools: Optional[List[Dict[str, Any]]] = None,
        on_cost: Optional[Callable] = None,
        on_browser_started: Optional[Callable] = None,
        on_browser_stopped: Optional[Callable] = None,
    ) -> None:
        self.openai_client = openai_client
        self.model = model
        self.workspace_dir = Path(workspace_dir)
        self.chat_history = chat_history or []
        self.identity_columns = set(identity_columns or [])
        self.submitted_rows = submitted_rows if submitted_rows is not None else []
        self.submitted_rows_lock = submitted_rows_lock
        self.stop_checker = stop_checker
        self.stop_event = stop_event
        self.mcp_tools = mcp_tools or []
        self.on_cost = on_cost

        # State — reset per generate() call
        self._current_row: Dict[str, Any] = {}
        self._submitted: bool = False
        self._skipped: bool = False
        self._skip_reason: str = ""
        self._schema: List[Dict] = []

        self._impl = ResearchTools(
            workspace_dir=workspace_dir,
            schema=[],
            brave_api_key=brave_api_key,
            openai_client=openai_client,
            model=model,
            sandbox=sandbox,
            stop_checker=stop_checker,
            blob_service_client=blob_service_client,
            project_id=project_id,
            uploaded_file_urls=uploaded_file_urls,
            on_browser_started=on_browser_started,
            on_browser_stopped=on_browser_stopped,
        )
        self._impl.set_scope(ResearchScope(id="row_gen", description="", quota=0))

        self._registry = ToolRegistry()
        self._register_tools(self._registry)

    def _get_col_def(self, name: str) -> Optional[Dict]:
        for col in self._schema:
            if col.get("name") == name:
                return col
        return None

    def _validate_value(self, col_def: Dict, value: Any) -> Optional[str]:
        col_type = col_def.get("type", "string")
        if col_type == "string":
            if not isinstance(value, str):
                return f"expects string, got {type(value).__name__}"
        elif col_type == "int":
            if not isinstance(value, int) or isinstance(value, bool):
                return f"expects int, got {type(value).__name__}"
        elif col_type == "float":
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                return f"expects float, got {type(value).__name__}"
        elif col_type == "bool":
            if not isinstance(value, bool):
                return f"expects bool, got {type(value).__name__}"
        elif col_type == "enum":
            allowed = col_def.get("enum_values", [])
            if value not in allowed:
                return f"expects one of {allowed}, got {value!r}"
        elif col_type == "json":
            schema = col_def.get("json_schema")
            if schema:
                try:
                    jsonschema.validate(value, schema)
                except jsonschema.ValidationError as e:
                    return f"json_schema validation failed: {e.message}"
        return None

    def _find_similar_rows(self, col_name: str, value: Any) -> List[Dict[str, Any]]:
        """Find submitted rows with a similar value in the given column.

        Case-insensitive exact match on string values.
        Returns up to 5 matching rows.
        """
        if not self.submitted_rows:
            return []

        value_str = str(value).lower().strip()
        matches = []
        for row in self.submitted_rows:
            existing = row.get(col_name)
            if existing is None:
                continue
            if str(existing).lower().strip() == value_str:
                matches.append(row)
                if len(matches) >= 5:
                    break
        return matches

    def _register_tools(self, registry: ToolRegistry) -> None:

        async def set_column(args: Dict) -> tuple[str, float]:
            name = args.get("name", "")
            value = args.get("value")

            col_def = self._get_col_def(name)
            if col_def:
                error = self._validate_value(col_def, value)
                if error:
                    return f"Error: column '{name}' {error}", 0.0

            self._current_row[name] = value

            # Check identity columns for duplicates
            if name in self.identity_columns:
                similar = self._find_similar_rows(name, value)
                if similar:
                    rows_text = json.dumps(similar, indent=2, ensure_ascii=False)
                    return (
                        f"Set {name} = {value!r}\n\n"
                        f"Similar existing rows found ({len(similar)}):\n{rows_text}\n\n"
                        f"If any of these is the same entity, call skip_row(reason=\"duplicate: ...\")."
                    ), 0.0

            return f"Set {name}", 0.0

        registry.add(
            name="set_column",
            description=(
                "Set a column value. For identity columns, returns similar existing rows "
                "so you can detect duplicates early and skip_row() if needed."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Column name"},
                    "value": {"description": "Column value"},
                },
                "required": ["name", "value"],
            },
            handler=set_column,
        )

        async def append_to_column(args: Dict) -> tuple[str, float]:
            name = args.get("name", "")
            value = args.get("value")
            col_def = self._get_col_def(name)
            if not col_def:
                return f"Error: unknown column '{name}'", 0.0
            col_type = col_def.get("type", "string")
            if col_type == "json":
                if name not in self._current_row:
                    self._current_row[name] = []
                if not isinstance(self._current_row[name], list):
                    return f"Error: column '{name}' is not a list", 0.0
                self._current_row[name].append(value)
                return f"Appended to {name} ({len(self._current_row[name])} items)", 0.0
            elif col_type == "string":
                if name not in self._current_row:
                    self._current_row[name] = ""
                if not isinstance(value, str):
                    return "Error: append value must be string for string column", 0.0
                sep = "\n" if self._current_row[name] else ""
                self._current_row[name] += sep + value
                return f"Appended to {name}", 0.0
            else:
                return f"Error: append not supported for type '{col_type}'", 0.0

        registry.add(
            name="append_to_column",
            description="Append to a column. For json columns, appends as list element. For strings, concatenates with newline.",
            parameters={
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "value": {"description": "Value to append"},
                },
                "required": ["name", "value"],
            },
            handler=append_to_column,
        )

        async def clear_column(args: Dict) -> tuple[str, float]:
            name = args.get("name", "")
            self._current_row.pop(name, None)
            return f"Cleared {name}", 0.0

        registry.add(
            name="clear_column",
            description="Clear a column value to start over.",
            parameters={
                "type": "object",
                "properties": {"name": {"type": "string"}},
                "required": ["name"],
            },
            handler=clear_column,
        )

        async def submit_row(args: Dict) -> tuple[str, float]:
            missing = [
                col.get("name") for col in self._schema
                if col.get("name") and col.get("name") not in self._current_row
            ]
            if missing:
                return f"Error: missing columns {missing}. Set them before submitting.", 0.0

            for col in self._schema:
                col_name = col.get("name", "")
                if col_name in self._current_row:
                    error = self._validate_value(col, self._current_row[col_name])
                    if error:
                        return f"Error: column '{col_name}' {error}.", 0.0

            self._submitted = True
            return "Row submitted.", 0.0

        registry.add(
            name="submit_row",
            description="Submit the completed row. Call when all columns are filled.",
            parameters={"type": "object", "properties": {}},
            handler=submit_row,
        )

        async def skip_row(args: Dict) -> tuple[str, float]:
            reason = args.get("reason", "")
            self._skipped = True
            self._skip_reason = reason
            return f"Row skipped: {reason}", 0.0

        registry.add(
            name="skip_row",
            description=(
                "Skip this candidate. Use when it's a duplicate, dead end, "
                "broken URL, or doesn't qualify."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "reason": {"type": "string", "description": "Why this row is being skipped"},
                },
                "required": ["reason"],
            },
            handler=skip_row,
        )

        async def rng(args: Dict) -> tuple[str, float]:
            options = args.get("options", [])
            weights = args.get("weights")
            if not options:
                return "Error: no options provided", 0.0
            if weights and len(weights) == len(options):
                selected = random.choices(options, weights=weights, k=1)[0]
            else:
                selected = random.choice(options)
            return f"Selected: {selected}", 0.0

        registry.add(
            name="rng",
            description="Pick a random option for controlled randomization.",
            parameters={
                "type": "object",
                "properties": {
                    "options": {"type": "array", "items": {"type": "string"}},
                    "weights": {"type": "array", "items": {"type": "number"}},
                },
                "required": ["options"],
            },
            handler=rng,
        )

        async def read_file(args: Dict) -> tuple[str, float]:
            path_str = args.get("path", "")
            try:
                path = Path(path_str)
                if not path.is_absolute():
                    candidate = self.workspace_dir / path
                    if not candidate.exists():
                        candidate = self.workspace_dir / "sources" / path
                    path = candidate
                if not path.exists():
                    return f"File not found: {path_str}", 0.0
                content = path.read_text(encoding="utf-8")
                if len(content) > READ_FILE_LIMIT:
                    content = content[:READ_FILE_LIMIT] + f"\n\n[Truncated at {READ_FILE_LIMIT} chars]"
                return content, 0.0
            except Exception as e:
                return f"Error reading file: {e}", 0.0

        registry.add(
            name="read_file",
            description="Read a file from the workspace.",
            parameters={
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
            handler=read_file,
        )

        self._impl.register_on(registry)

    def _format_conversation(self) -> str:
        if not self.chat_history:
            return "(no conversation history)"
        parts = []
        for msg in self.chat_history:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            parts.append(f"**{role}**: {content}")
        return "\n\n".join(parts)

    def _format_candidate(self, candidate: Any) -> str:
        if candidate is None:
            return "(no candidate)"
        if isinstance(candidate, dict):
            return json.dumps(candidate, indent=2, ensure_ascii=False)
        return str(candidate)

    async def generate(
        self,
        instructions: str,
        candidate: Any,
        schema: Optional[List[Dict]] = None,
    ) -> GeneratedRow:
        """Generate a single row from a candidate."""
        if schema is None:
            schema = []

        self._current_row = {}
        self._submitted = False
        self._skipped = False
        self._skip_reason = ""
        self._schema = schema

        system_prompt = ROW_GENERATOR_SYSTEM_PROMPT.format(
            conversation=self._format_conversation(),
            schema_str=json.dumps(schema, indent=2),
            instructions=instructions or "(no specific instructions)",
            candidate=self._format_candidate(candidate),
        )

        conversation = AgentConversation(
            openai_client=self.openai_client,
            model=self.model,
            system_prompt=system_prompt,
            tools=self._registry,
            stop_checker=self.stop_checker,
            stop_event=self.stop_event,
            max_turns=MAX_GENERATION_TURNS,
            reasoning={"effort": "low", "summary": "auto"},
            label="row_generator",
            on_cost=self.on_cost,
            extra_tools=self.mcp_tools,
        )

        await conversation.send(
            "Process the candidate above and generate a dataset row.",
            exit_condition=lambda: self._submitted or self._skipped,
        )

        cost = conversation.total_cost

        if self._skipped:
            return GeneratedRow(
                success=False, skipped=True, skip_reason=self._skip_reason, cost_usd=cost
            )

        if self._submitted:
            return GeneratedRow(success=True, row=self._current_row, cost_usd=cost)

        return GeneratedRow(
            success=False,
            error=f"Row generation did not complete ({conversation.total_turns} turns)",
            row=self._current_row if self._current_row else None,
            cost_usd=cost,
        )

    @property
    def cost_usd(self) -> float:
        return 0.0

    async def cleanup(self) -> None:
        try:
            await self._impl.cleanup()
        except Exception as e:
            logger.warning(f"RowGeneratorAgent cleanup error: {e}")
