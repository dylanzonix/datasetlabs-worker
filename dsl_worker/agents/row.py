"""
Row generator agent — generates a single dataset row from an assignment.

V4: Takes a dataset brief + row assignment + schema, produces a GeneratedRow.
Each row generator gets one assignment and produces one row. The assignment
is a natural language instruction written by the topic agent. The dataset
brief provides overall context and quality expectations.
"""

from __future__ import annotations

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

# Max chars for read_file results
READ_FILE_LIMIT = 30_000


@dataclass
class GeneratedRow:
    """Result of row generation."""
    success: bool
    row: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    cost_usd: float = 0.0


ROW_GENERATOR_SYSTEM_PROMPT = """\
You are generating a single dataset row.

## Dataset Brief

{brief_section}

## Your Assignment

{assignment_section}

## Schema

<schema>
{schema_str}
</schema>

## Output — TOOLS ONLY

You MUST deliver your output via tool calls. Text responses are discarded by the system.

For each column in the schema, call set_column(name, value). Then call submit_row().

IMPORTANT: Call set_column for ALL columns in a SINGLE response. Do not call set_column one at a time across multiple turns — batch all set_column calls together, then call submit_row in the same response or the next one.

DO NOT write JSON, code blocks, or row content as text. Only tool calls are captured.

## Tools

- set_column(name, value): Set a column value. Values are validated against the schema type.
- append_to_column(name, value): Append to a column. For json columns, appends value as a list element. For string columns, concatenates with a newline.
- clear_column(name): Clear a column value so you can start over.
- submit_row(): Submit the completed row. Call after all columns are set.
- rng(options, weights): Pick a random option for controlled randomization
- read_file(path): Read a file from the workspace
- brave_search(query): Search the web for information
- open(ref_id_or_url, start_line): View a page or file
- find(ref_id, pattern): Search within a loaded page
- click(ref_id, link_id): Follow a link
- code_exec(script, description): Execute Python
- interact(url_or_ref_id, task): Browser agent for complex interactions

## Your Role

You are the execution layer. An orchestrator planned the dataset and a topic agent wrote
your specific assignment. Your job is to produce one high-quality row.

Some assignments are straightforward — maybe the information is right there in the brief
or a source file. Others require you to actually figure things out: search the web, read
sources, run code to verify something, cross-reference claims. Use your judgment on what
the assignment needs.

The key thing: don't just make stuff up. If the assignment requires information you don't
have, go find it. You have tools for this — web search, browsing, code execution. Think
about what would actually give you the answer and use the right tool for that.

## Process

1. Read the assignment and the dataset brief
2. Figure out what you need to know and how to get it
3. Call set_column(name, value) for EACH column in the schema
4. For json array columns, prefer append_to_column to build the list one element at a time
5. Call submit_row() when done
"""


class RowGeneratorAgent:
    """
    Generates a single dataset row from a work item instruction.

    Usage:
        agent = RowGeneratorAgent(
            openai_client=tracked_client,
            model="gpt-5.2",
            workspace_dir=Path("/workspace"),
            brave_api_key="...",
        )

        result = await agent.generate(
            instruction="Write a tip about lean-peeking in DayZ. Consult sources tagged 'combat'.",
            schema=[{"name": "tip", "type": "string"}, ...],
        )

        if result.success:
            print(result.row)

        await agent.cleanup()
    """

    def __init__(
        self,
        openai_client: TrackedOpenAIClient,
        model: str,
        workspace_dir: Path,
        brave_api_key: Optional[str] = None,
        sandbox: Optional[Any] = None,
        stop_checker: Optional[Callable[[], bool]] = None,
        blob_service_client: Optional[Any] = None,
        project_id: Optional[Any] = None,
        uploaded_file_urls: Optional[Dict[str, str]] = None,
        mcp_tools: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        self.openai_client = openai_client
        self.model = model
        self.workspace_dir = Path(workspace_dir)
        self.stop_checker = stop_checker
        self.mcp_tools = mcp_tools or []

        # Tool state — reset per generate() call
        self._current_row: Dict[str, Any] = {}
        self._submitted: bool = False
        self._schema: List[Dict] = []

        # Create ResearchTools for browsing tool implementations
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
        )
        # Set a dummy scope for ResearchTools compatibility
        self._impl.set_scope(ResearchScope(
            id="row_gen",
            description="",
            quota=0,
        ))

        # Build tool registry
        self._registry = ToolRegistry()
        self._register_tools(self._registry)

        # Store config for building conversations per generate() call
        self._brave_api_key = brave_api_key
        self._sandbox = sandbox

    def _get_col_def(self, name: str) -> Optional[Dict]:
        """Look up a column definition by name."""
        for col in self._schema:
            if col.get("name") == name:
                return col
        return None

    def _validate_value(self, col_def: Dict, value: Any) -> Optional[str]:
        """Validate a value against a column type. Returns error string or None."""
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

    def _register_tools(self, registry: ToolRegistry) -> None:
        """Register row generation tools + research browsing tools."""
        impl = self._impl

        # --- Row generation tools ---

        async def set_column(args: Dict) -> tuple[str, float]:
            name = args.get("name", "")
            value = args.get("value")

            col_def = self._get_col_def(name)
            if col_def:
                error = self._validate_value(col_def, value)
                if error:
                    return f"Error: column '{name}' {error}", 0.0

            self._current_row[name] = value
            return f"Set {name}", 0.0

        registry.add(
            name="set_column",
            description="Set a column value for the row. Values are validated against the schema type.",
            parameters={
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Column name"},
                    "value": {"description": "Column value (string, number, list, etc.)"},
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
                    return f"Error: column '{name}' is not a list, cannot append", 0.0
                self._current_row[name].append(value)
                return f"Appended to {name} (now {len(self._current_row[name])} items)", 0.0
            elif col_type == "string":
                if name not in self._current_row:
                    self._current_row[name] = ""
                if not isinstance(value, str):
                    return "Error: append value must be string for string column", 0.0
                if self._current_row[name]:
                    self._current_row[name] += "\n" + value
                else:
                    self._current_row[name] = value
                return f"Appended to {name}", 0.0
            else:
                return f"Error: append not supported for type '{col_type}'", 0.0

        registry.add(
            name="append_to_column",
            description="Append a value to a column. For json columns, appends as a list element. For string columns, concatenates with a newline.",
            parameters={
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Column name"},
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
            description="Clear a column value so you can start over.",
            parameters={
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Column name to clear"},
                },
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
                        return f"Error: column '{col_name}' {error}. Fix it before submitting.", 0.0

            self._submitted = True
            return "Row submitted.", 0.0

        registry.add(
            name="submit_row",
            description="Submit the completed row. Call when all columns are filled.",
            parameters={"type": "object", "properties": {}},
            handler=submit_row,
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
            description="Pick a random option. Use for controlled randomization (e.g., tone, perspective, style).",
            parameters={
                "type": "object",
                "properties": {
                    "options": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Options to choose from",
                    },
                    "weights": {
                        "type": "array",
                        "items": {"type": "number"},
                        "description": "Optional weights (must match length of options)",
                    },
                },
                "required": ["options"],
            },
            handler=rng,
        )

        # --- read_file tool (reads workspace files including sources) ---
        async def read_file(args: Dict) -> tuple[str, float]:
            path_str = args.get("path", "")
            try:
                path = Path(path_str)
                if not path.is_absolute():
                    # Try as relative to workspace first, then sources
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
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "File path (relative to workspace or sources directory)",
                    },
                },
                "required": ["path"],
            },
            handler=read_file,
        )

        # --- Research/browsing tools from ResearchTools ---
        impl.register_on(registry)

    def _build_system_prompt(
        self,
        assignment: str,
        schema: List[Dict],
        dataset_brief: str = "",
    ) -> str:
        schema_str = json.dumps(schema, indent=2)

        brief_section = ""
        if dataset_brief:
            brief_section = f"<dataset_brief>\n{dataset_brief}\n</dataset_brief>"

        assignment_section = f"<assignment>\n{assignment}\n</assignment>"

        return ROW_GENERATOR_SYSTEM_PROMPT.format(
            brief_section=brief_section,
            assignment_section=assignment_section,
            schema_str=schema_str,
        )

    async def generate(
        self,
        assignment: str,
        schema: List[Dict],
        dataset_brief: str = "",
    ) -> GeneratedRow:
        """
        Generate a single row from an assignment.

        Args:
            assignment: The row assignment — a natural language instruction for this row.
            schema: Column definitions for the row.
            dataset_brief: The dataset brief — overall context and quality expectations.

        Creates a fresh AgentConversation per call so each row generation
        starts with a clean message history.
        """
        # Reset state
        self._current_row = {}
        self._submitted = False
        self._schema = schema

        system_prompt = self._build_system_prompt(assignment, schema, dataset_brief)

        conversation = AgentConversation(
            openai_client=self.openai_client,
            model=self.model,
            system_prompt=system_prompt,
            tools=self._registry,
            stop_checker=self.stop_checker,
            max_turns=MAX_GENERATION_TURNS,
            reasoning={"effort": "low", "summary": "auto"},
            label="row_generator",
            extra_tools=self.mcp_tools,
        )

        result = await conversation.send(
            "Generate a dataset row from the assignment above.",
            exit_condition=lambda: self._submitted,
        )

        cost = conversation.total_cost

        if self._submitted:
            return GeneratedRow(
                success=True,
                row=self._current_row,
                cost_usd=cost,
            )

        return GeneratedRow(
            success=False,
            error=f"Row generation did not complete ({conversation.total_turns} turns)",
            row=self._current_row if self._current_row else None,
            cost_usd=cost,
        )

    @property
    def cost_usd(self) -> float:
        return 0.0  # Cost tracked per generate() call, not accumulated

    async def cleanup(self) -> None:
        """Clean up browser and other resources."""
        try:
            await self._impl.cleanup()
        except Exception as e:
            logger.warning(f"RowGeneratorAgent cleanup error: {e}")
