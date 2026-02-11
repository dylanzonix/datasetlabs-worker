"""
Generator agent — iterates through a scope and yields seeds into a shared queue.

The orchestrator spawns one or more generators, each with a specific scope
to iterate through. The generator researches its scope and yields structured
seeds as it discovers them.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from dsl_worker.agents.base import AgentConversation, AgentResult
from dsl_worker.agents.tools import ToolRegistry
from dsl_worker.billing.tracked_client import TrackedOpenAIClient

logger = logging.getLogger(__name__)


GENERATOR_SYSTEM_PROMPT = """\
You are a seed extraction agent. Your job is to iterate through a scope and yield
raw seed items as fast as possible. You are a rough extractor — the downstream
pipeline will clean up and enrich your output.

## Your assignment

Scope: {scope}

Seed shape: {seed_description}

Target: ~{target_count} seeds

## Strategy: speed over depth

You are NOT a research agent. Do not write summaries, analyze findings, or deliberate.
Your loop is: search → open source → extract items → yield seeds → next source.

### Efficiency techniques

- **Line references**: When you open a page and see a list/table, note the line numbers.
  Use open() with start_line to jump directly to relevant sections instead of re-reading.
- **Programmatic extraction**: When a source has >20 items in structured format (tables,
  lists, CSV, JSON), use code_exec to extract them all at once and yield in bulk.
- **Bulk yielding**: Call yield_seed() for each item as you find it. Don't batch or wait.
- **Multiple sources**: If one source doesn't cover your scope, search for more. Cast a
  wide net — directories, databases, lists, rankings, registries.

### Seed quality

Each seed should be a JSON object matching the seed shape. Include:
- The core data fields requested
- A source_url when available (where you found it)
- Enough detail that the downstream pipeline can work with it without re-researching

Seeds don't need to be perfect — rough and complete beats polished and sparse.

## Tools

- brave_search(query): Search the web
- open(ref_id_or_url, start_line): View a page (use start_line to jump to known sections)
- find(ref_id, pattern): Search within a loaded page
- click(ref_id, link_id): Follow a link
- list_files(directory): List workspace files
- code_exec(script, description): Execute Python for bulk extraction
- interact(url_or_ref_id, task): Browser agent for JS-heavy pages
- yield_seed(seed): Yield one seed item — call this for every item you discover

## When to stop

- You've reached or exceeded your target count (~{target_count} seeds)
- You've exhausted your scope (no more sources to extract from)
- Prefer to over-produce slightly rather than under-produce
"""


class GeneratorAgent:
    """
    Iterates through a scope and yields seeds into a shared async queue.

    Usage:
        queue = asyncio.Queue()
        gen = GeneratorAgent(
            openai_client=tracked_client,
            model="gpt-5.2",
            scope="Top 100 US universities by enrollment",
            seed_description="name, location, enrollment_count, type (public/private)",
            seed_queue=queue,
            workspace_dir=Path("/workspace"),
            brave_api_key="...",
        )

        # Run in background
        task = asyncio.create_task(gen.run())

        # Consume seeds as they arrive
        while True:
            seed = await queue.get()
            if seed is None:  # Poison pill = done
                break
            process(seed)
    """

    def __init__(
        self,
        openai_client: TrackedOpenAIClient,
        model: str,
        scope: str,
        seed_description: str,
        seed_queue: asyncio.Queue,
        workspace_dir: Path,
        generator_id: str = "generator",
        target_count: int = 100,
        bucket_id: Optional[str] = None,
        brave_api_key: Optional[str] = None,
        sandbox: Optional[Any] = None,
        stop_checker: Optional[Callable[[], bool]] = None,
        max_turns: int = 200,
    ) -> None:
        self.scope = scope
        self.seed_description = seed_description
        self.seed_queue = seed_queue
        self.generator_id = generator_id
        self.target_count = target_count
        self.bucket_id = bucket_id
        self.seeds_yielded: int = 0
        self._stopped = False

        # Build tools — research tools + yield_seed
        registry = ToolRegistry()
        self._register_tools(registry, openai_client, workspace_dir, brave_api_key, sandbox, stop_checker)

        prompt = GENERATOR_SYSTEM_PROMPT.format(
            scope=scope,
            seed_description=seed_description,
            target_count=target_count,
        )

        self._conversation = AgentConversation(
            openai_client=openai_client,
            model=model,
            system_prompt=prompt,
            tools=registry,
            stop_checker=stop_checker,
            max_turns=max_turns,
            reasoning={"effort": "medium", "summary": "detailed"},
            label=f"generator:{generator_id}",
        )

    def _register_tools(
        self,
        registry: ToolRegistry,
        openai_client: TrackedOpenAIClient,
        workspace_dir: Path,
        brave_api_key: Optional[str],
        sandbox: Optional[Any],
        stop_checker: Optional[Callable[[], bool]],
    ) -> None:
        """Register generator tools: research tools + yield_seed."""
        # Import here to avoid circular deps at module level
        from dsl_worker.agents.research import ResearchAgent

        # Create a research agent internally for its tools
        self._research = ResearchAgent(
            openai_client=openai_client,
            model="unused",  # We won't use its conversation, just its tools
            workspace_dir=workspace_dir,
            brave_api_key=brave_api_key,
            sandbox=sandbox,
            stop_checker=stop_checker,
        )

        # Re-register the research tools from the research agent's impl
        impl = self._research._impl
        defs = impl.get_tool_definitions(phase="research")

        handlers = {
            "brave_search": lambda args: impl.brave_search(
                query=args.get("query", ""),
                response_length=args.get("response_length", "medium"),
            ),
            "open": lambda args: impl.open(
                ref_id_or_url=args.get("ref_id_or_url", ""),
                start_line=args.get("start_line", 0),
                response_length=args.get("response_length", "medium"),
            ),
            "find": lambda args: impl.find(
                ref_id=args.get("ref_id", ""),
                pattern=args.get("pattern", ""),
                response_length=args.get("response_length", "medium"),
            ),
            "click": lambda args: impl.click(
                ref_id=args.get("ref_id", ""),
                link_id=args.get("link_id", 0),
                response_length=args.get("response_length", "medium"),
            ),
            "list_files": lambda args: impl.list_files(
                directory=args.get("directory", "all"),
            ),
            "code_exec": lambda args: impl.code_exec(
                script=args.get("script", ""),
                description=args.get("description", ""),
            ),
            "interact": lambda args: impl.interact(
                url_or_ref_id=args.get("url_or_ref_id", ""),
                task=args.get("task", ""),
            ),
        }

        for defn in defs:
            name = defn.get("name")
            if name in handlers:
                registry.add(
                    name=name,
                    description=defn.get("description", ""),
                    parameters=defn.get("parameters", {}),
                    handler=handlers[name],
                )

        # Add yield_seed tool
        async def yield_seed(args: Dict) -> tuple[str, float]:
            seed_data = args.get("seed", args)
            if "seed" in args and isinstance(args["seed"], dict):
                seed_data = args["seed"]

            # Wrap in envelope with bucket and generator metadata
            envelope = {
                "data": seed_data,
                "bucket_id": self.bucket_id,
                "generator_id": self.generator_id,
            }

            await self.seed_queue.put(envelope)
            self.seeds_yielded += 1

            remaining = self.target_count - self.seeds_yielded
            if remaining <= 0:
                return (
                    f"Seed #{self.seeds_yielded} yielded. "
                    f"Target reached ({self.target_count}). You can stop now.",
                    0.0,
                )
            return f"Seed #{self.seeds_yielded} yielded. {remaining} remaining to target.", 0.0

        registry.add(
            name="yield_seed",
            description="Yield a seed item for dataset generation. Call this for each item you discover.",
            parameters={
                "type": "object",
                "properties": {
                    "seed": {
                        "type": "object",
                        "description": "The seed data as a JSON object",
                    },
                },
                "required": ["seed"],
            },
            handler=yield_seed,
        )

    async def run(self) -> AgentResult:
        """
        Run the generator until scope is exhausted or stopped.
        Sends a poison pill (None) to the queue when done.
        """
        try:
            result = await self._conversation.send(
                f"Begin extracting seeds from your scope. "
                f"Target: ~{self.target_count} seeds. Scope: {self.scope}"
            )
            return result
        finally:
            # Signal completion with poison pill
            await self.seed_queue.put(None)

    def stop(self) -> None:
        """Signal the generator to stop."""
        self._stopped = True

    @property
    def cost_usd(self) -> float:
        return self._conversation.total_cost

    async def cleanup(self) -> None:
        """Clean up resources."""
        try:
            await self._research.cleanup()
        except Exception as e:
            logger.warning(f"Generator cleanup error: {e}")
