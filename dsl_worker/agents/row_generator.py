"""
Row generator agent — generates a single dataset row from a seed.

Built on AgentConversation with full ResearchTools browsing stack.
Takes a seed + pipeline instructions + schema, produces a GeneratedRow.
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
from dsl_worker.phases.research_tools import ResearchTools, ResearchScope

logger = logging.getLogger(__name__)

MAX_GENERATION_TURNS = 15


@dataclass
class GeneratedRow:
    """Result of row generation."""
    success: bool
    row: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    cost_usd: float = 0.0
    skipped: bool = False
    skip_reason: Optional[str] = None


ROW_GENERATOR_SYSTEM_PROMPT = """\
You are generating a single dataset row from a seed.

{instructions_section}
<seed>
{seed}
</seed>

<schema>
{schema_str}
</schema>

## Output — TOOLS ONLY

You MUST deliver your output via tool calls. Text responses are discarded by the system.

For each column in the schema, call set_column(name, value). Then call submit_row().
If the seed is unusable, call skip_seed(reason).

DO NOT write JSON, code blocks, or row content as text. Only tool calls are captured.

## Tools

- set_column(name, value): Set a column value. Values are validated against the schema type.
- append_to_column(name, value): Append to a column. For json columns, appends value as a list element. For string columns, concatenates with a newline. Use this for building up lists incrementally (e.g., multi-turn conversations) — it's more reliable than constructing a large JSON array in one shot.
- clear_column(name): Clear a column value so you can start over.
- submit_row(): Submit the completed row. Call after all columns are set.
- skip_seed(reason): Skip this seed entirely (irrelevant, duplicate, bad data)
- rng(options, weights): Pick a random option for controlled randomization
- brave_search(query): Search the web for information
- open(ref_id_or_url, start_line): View a page or file
- find(ref_id, pattern): Search within a loaded page
- click(ref_id, link_id): Follow a link
- code_exec(script, description): Execute Python
- interact(url_or_ref_id, task): Browser agent for complex interactions

## Process

1. Read the seed — this is your source material
2. Follow the pipeline instructions to transform the seed into a row
3. If you need additional information, use brave_search/open/code_exec
4. Call set_column(name, value) for EACH column in the schema
5. For json array columns, prefer append_to_column to build the list one element at a time
6. Call submit_row() when done
7. Call skip_seed(reason) if the seed is unusable

Be accurate. Follow the pipeline instructions.
"""


class RowGeneratorAgent:
    """
    Generates a single dataset row from a seed using AgentConversation.

    Usage:
        agent = RowGeneratorAgent(
            openai_client=tracked_client,
            model="gpt-5.2",
            workspace_dir=Path("/workspace"),
            brave_api_key="...",
        )

        result = await agent.generate(
            seed='{"name": "Gandalf", "franchise": "Lord of the Rings"}',
            pipeline_instructions="Generate a roleplay chat log...",
            schema=[{"name": "system_prompt", "type": "string", ...}],
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
    ) -> None:
        self.openai_client = openai_client
        self.model = model
        self.workspace_dir = Path(workspace_dir)
        self.stop_checker = stop_checker

        # Tool state — reset per generate() call
        self._current_row: Dict[str, Any] = {}
        self._submitted: bool = False
        self._skipped: bool = False
        self._skip_reason: str = ""
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
                    return f"Error: append value must be string for string column", 0.0
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
            # Check for missing columns
            missing = [
                col.get("name") for col in self._schema
                if col.get("name") and col.get("name") not in self._current_row
            ]
            if missing:
                return f"Error: missing columns {missing}. Set them before submitting.", 0.0

            # Validate json columns (catches append-built values)
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

        async def skip_seed(args: Dict) -> tuple[str, float]:
            reason = args.get("reason", "No reason given")
            self._skipped = True
            self._skip_reason = reason
            return f"Seed skipped: {reason}", 0.0

        registry.add(
            name="skip_seed",
            description="Skip this seed entirely. Use when the seed is irrelevant, a duplicate, or unusable.",
            parameters={
                "type": "object",
                "properties": {
                    "reason": {"type": "string", "description": "Why this seed is being skipped"},
                },
                "required": ["reason"],
            },
            handler=skip_seed,
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

        # --- Research/browsing tools from ResearchTools ---
        impl.register_on(registry)

    def _build_system_prompt(
        self,
        pipeline_instructions: str,
        seed: str,
        schema: List[Dict],
    ) -> str:
        schema_lines = []
        for col in schema:
            line = f"- {col.get('name')} ({col.get('type', 'string')})"
            if col.get("type") == "enum" and col.get("enum_values"):
                line += f" — values: {col['enum_values']}"
            if col.get("type") == "json" and col.get("json_schema"):
                line += f" — json_schema: {json.dumps(col['json_schema'])}"
            schema_lines.append(line)
        schema_str = "\n".join(schema_lines)

        instructions_section = ""
        if pipeline_instructions:
            instructions_section = (
                f"<pipeline_instructions>\n{pipeline_instructions}\n</pipeline_instructions>"
            )

        return ROW_GENERATOR_SYSTEM_PROMPT.format(
            instructions_section=instructions_section,
            seed=seed,
            schema_str=schema_str,
        )

    async def generate(
        self,
        seed: str,
        pipeline_instructions: str,
        schema: List[Dict],
    ) -> GeneratedRow:
        """
        Generate a single row from a seed.

        Creates a fresh AgentConversation per call so each row generation
        starts with a clean message history.
        """
        # Reset state
        self._current_row = {}
        self._submitted = False
        self._skipped = False
        self._skip_reason = ""
        self._schema = schema

        system_prompt = self._build_system_prompt(pipeline_instructions, seed, schema)

        conversation = AgentConversation(
            openai_client=self.openai_client,
            model=self.model,
            system_prompt=system_prompt,
            tools=self._registry,
            stop_checker=self.stop_checker,
            max_turns=MAX_GENERATION_TURNS,
            reasoning={"effort": "low", "summary": "auto"},
            label="row_generator",
        )

        result = await conversation.send(
            "Generate a dataset row from the seed above.",
            exit_condition=lambda: self._submitted or self._skipped,
        )

        cost = conversation.total_cost

        if self._skipped:
            return GeneratedRow(
                success=False,
                skipped=True,
                skip_reason=self._skip_reason,
                cost_usd=cost,
            )

        if self._submitted:
            return GeneratedRow(
                success=True,
                row=self._current_row,
                cost_usd=cost,
            )

        # Loop ended without submit or skip (max_turns or text-only response)
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
