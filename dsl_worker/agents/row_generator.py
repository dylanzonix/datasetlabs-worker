"""
Row generator agent — generates a single dataset row from a seed.

Built on AgentConversation with full ResearchTools browsing stack.
Takes a seed + pipeline instructions + schema, produces a GeneratedRow.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

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

## Tools

- set_column(name, value): Set a column value for the row
- submit_row(): Submit the completed row
- skip_seed(reason): Skip this seed entirely (irrelevant, duplicate, bad data)
- rng(options, weights): Pick a random option for controlled randomization
- brave_search(query): Search the web for information
- open(ref_id_or_url, start_line): View a page or file
- find(ref_id, pattern): Search within a loaded page
- click(ref_id, link_id): Follow a link
- code_exec(script, description): Execute Python
- interact(url_or_ref_id, task): Browser agent for complex interactions

## Process

1. Read the seed data — this is your anchor for what this row is about
2. Follow the pipeline instructions to transform the seed into a row
3. Use set_column() to fill each column in the schema
4. Use brave_search/open/code_exec if you need to look up specific information
5. Call submit_row() when all columns are filled
6. Call skip_seed() if the seed is irrelevant, a duplicate, or unusable

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

        # Create ResearchTools for browsing tool implementations
        self._impl = ResearchTools(
            workspace_dir=workspace_dir,
            schema=[],
            brave_api_key=brave_api_key,
            openai_client=openai_client,
            model=model,
            sandbox=sandbox,
            stop_checker=stop_checker,
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

    def _register_tools(self, registry: ToolRegistry) -> None:
        """Register row generation tools + research browsing tools."""
        impl = self._impl

        # --- Row generation tools ---

        async def set_column(args: Dict) -> tuple[str, float]:
            name = args.get("name", "")
            value = args.get("value")
            self._current_row[name] = value
            return f"Set {name}", 0.0

        registry.add(
            name="set_column",
            description="Set a column value for the row.",
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

        async def submit_row(args: Dict) -> tuple[str, float]:
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

        async def brave_search(args: Dict) -> tuple[str, float]:
            return await impl.brave_search(
                query=args.get("query", ""),
                response_length=args.get("response_length", "medium"),
            )

        async def open_page(args: Dict) -> tuple[str, float]:
            return await impl.open(
                ref_id_or_url=args.get("ref_id_or_url", ""),
                start_line=args.get("start_line", 0),
                response_length=args.get("response_length", "medium"),
            )

        async def find(args: Dict) -> tuple[str, float]:
            return await impl.find(
                ref_id=args.get("ref_id", ""),
                pattern=args.get("pattern", ""),
                response_length=args.get("response_length", "medium"),
            )

        async def click(args: Dict) -> tuple[str, float]:
            return await impl.click(
                ref_id=args.get("ref_id", ""),
                link_id=args.get("link_id", 0),
                response_length=args.get("response_length", "medium"),
            )

        async def list_files(args: Dict) -> tuple[str, float]:
            return await impl.list_files(
                directory=args.get("directory", "all"),
            )

        async def code_exec(args: Dict) -> tuple[str, float]:
            return await impl.code_exec(
                script=args.get("script", ""),
                description=args.get("description", ""),
            )

        async def interact(args: Dict) -> tuple[str, float]:
            return await impl.interact(
                url_or_ref_id=args.get("url_or_ref_id", ""),
                task=args.get("task", ""),
            )

        defs = impl.get_tool_definitions(phase="research")
        browsing_handlers = {
            "brave_search": brave_search,
            "open": open_page,
            "find": find,
            "click": click,
            "list_files": list_files,
            "code_exec": code_exec,
            "interact": interact,
        }

        for defn in defs:
            name = defn.get("name")
            if name in browsing_handlers:
                registry.add(
                    name=name,
                    description=defn.get("description", ""),
                    parameters=defn.get("parameters", {}),
                    handler=browsing_handlers[name],
                )

    def _build_system_prompt(
        self,
        pipeline_instructions: str,
        seed: str,
        schema: List[Dict],
    ) -> str:
        schema_str = "\n".join(
            f"- {col.get('name')} ({col.get('type', 'string')}): {col.get('description', '')}"
            for col in schema
        )

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
            # Check for missing columns
            missing = [
                col.get("name") for col in schema
                if col.get("name") and col.get("name") not in self._current_row
            ]
            if missing:
                return GeneratedRow(
                    success=False,
                    error=f"Missing columns: {missing}",
                    row=self._current_row,
                    cost_usd=cost,
                )

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
