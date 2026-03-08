"""
Row generator agent — generates a single dataset row from an assignment.

V5: Takes a filled template + seed values + filter findings + schema.
V4 compat: Still accepts assignment + dataset_brief for backward compatibility.

Each row generator gets one work item and produces one row. In V5, the work
item contains a filled template (variables already substituted) plus seed
values and filter findings as reference context.
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

## How to Research

You are the execution layer. An orchestrator planned the dataset and a topic agent wrote
your specific assignment. Your job is to produce one high-quality row — and that means
getting the information right.

### Step 1: Figure out what this row needs

Read your assignment and the dataset brief. Ask yourself:

- **Does this need factual research?** If the assignment involves a real entity, product,
  library, event, person, or any verifiable claim — yes, look it up. Don't rely on what
  you already know. Things change. Your knowledge has a cutoff.
- **Is this synthetic content?** If you're designing a conversation, writing a creative
  prompt, or generating a hypothetical scenario, you probably don't need web research.
  But you might want to ground specific details in reality (e.g., real product names,
  real API methods, real locations).
- **Is this evaluation/judgment?** If you're scoring, classifying, or assessing something,
  focus on analyzing what's in front of you. Your reasoning is the value — don't waste
  time searching unless you're genuinely unsure about evaluation criteria.

### Step 2: Check your assignment for source hints

Your assignment may include context from the topic agent — use it:
- **URLs** — open them directly instead of searching blind
- **File paths** — read them with read_file
- **Key facts or warnings** — these save you from common mistakes
- **Source recommendations** — "use the official docs, not blog posts" means that

### Step 3: Research with purpose

When you do need to look things up:

**Prefer primary sources.** Go to the actual website, repo, platform, or documentation
rather than blog posts, tutorials, or summaries about them. Official docs over Stack
Overflow. The actual PyPI page over a "top 10 libraries" listicle.

**Verify claims that matter.** If a search result says "Library X supports feature Y,"
open the actual docs to confirm. If you're reporting numbers (versions, dates, stats,
damage values), find the primary source. If something seems surprising, double-check it.

**Check freshness.** When dates matter, look at when sources were written. A 2023 blog
post about a library's API may not reflect the 2025 version. Prefer recent sources and
look for changelogs or release notes when versioning matters.

**Use code to verify.** If you can check a claim programmatically — test a code snippet,
check a package version, validate a calculation — use code_exec. Running code is the
strongest form of verification.

**Use the right tool for the job:**
- brave_search: find something when you don't have a URL yet
- open: read a specific URL or search result in detail
- code_exec: verify claims programmatically, test code snippets, process data
- read_file: access workspace files the assignment points you to

### Step 4: Know when you have enough

Stop researching when you can confidently fill every column with verified information.
A few well-chosen sources beat a dozen skimmed ones. Don't over-research simple
assignments, and don't under-research complex ones.

If you can't find information after 2-3 targeted attempts, note that in your output
rather than making something up.

## Process

1. Read the assignment and the dataset brief
2. Decide what needs research vs. what you can produce directly
3. Research with purpose — use source hints, verify claims, prefer primary sources
4. Call set_column(name, value) for EACH column in the schema
5. For json array columns, prefer append_to_column to build the list one element at a time
6. Call submit_row() when done
"""


ROW_GENERATOR_V5_SYSTEM_PROMPT = """\
You are generating a single dataset row.

## Instructions

{template_with_filled_variables}

## Research Context

{research_context_section}

## Additional Context

{filter_findings_section}

## Schema

<schema>
{schema_str}
</schema>

## Output — TOOLS ONLY

You MUST deliver your output via tool calls. Text responses are discarded by the system.

For each column in the schema, call set_column(name, value). Then call submit_row().

IMPORTANT: Call set_column for ALL columns in a SINGLE response. Do not call set_column one at a time across multiple turns — batch all set_column calls together, then call submit_row in the same response or the next one.

DO NOT write JSON, code blocks, or row content as text. Only tool calls are captured.

## Research

Your seed and filter findings may already contain much of what you need.
Check what you have before searching. Only research if the instructions
require verification or additional information beyond what was provided.

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
        on_cost: Optional[Callable] = None,
        langfuse_parent: Optional[Any] = None,
        on_browser_started: Optional[Callable] = None,
    ) -> None:
        self.openai_client = openai_client
        self.model = model
        self.workspace_dir = Path(workspace_dir)
        self.stop_checker = stop_checker
        self.mcp_tools = mcp_tools or []
        self.on_cost = on_cost
        self.langfuse_parent = langfuse_parent

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
            on_browser_started=on_browser_started,
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

    def _build_v5_prompt(
        self,
        template: str,
        seed: Optional[Dict],
        filter_findings: Optional[str],
        schema: List[Dict],
        research_context: Optional[str] = None,
    ) -> str:
        """Build V5 system prompt from filled template + seed + filter findings."""
        schema_str = json.dumps(schema, indent=2)

        filter_section = ""
        if filter_findings:
            filter_section = f"<filter_findings>\n{filter_findings}\n</filter_findings>"
        else:
            filter_section = "(no filter findings)"

        research_section = ""
        if research_context:
            research_section = f"<research_context>\n{research_context}\n</research_context>"
        else:
            research_section = "(no research context)"

        return ROW_GENERATOR_V5_SYSTEM_PROMPT.format(
            template_with_filled_variables=template,
            research_context_section=research_section,
            filter_findings_section=filter_section,
            schema_str=schema_str,
        )

    async def generate(
        self,
        # V5 interface
        template: str = "",
        seed: Optional[Dict] = None,
        filter_findings: Optional[str] = None,
        research_context: Optional[str] = None,
        schema: Optional[List[Dict]] = None,
        # V4 backward compat
        assignment: str = "",
        dataset_brief: str = "",
    ) -> GeneratedRow:
        """
        Generate a single row.

        V5: Pass template (filled), seed, filter_findings, research_context, schema.
        V4: Pass assignment, schema, dataset_brief.

        Creates a fresh AgentConversation per call so each row generation
        starts with a clean message history.
        """
        if schema is None:
            schema = []

        # Reset state
        self._current_row = {}
        self._submitted = False
        self._schema = schema

        if template:
            system_prompt = self._build_v5_prompt(
                template, seed, filter_findings, schema, research_context
            )
        else:
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
            on_cost=self.on_cost,
            extra_tools=self.mcp_tools,
            langfuse_parent=self.langfuse_parent,
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
