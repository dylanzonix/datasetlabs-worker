"""
Seed yielder agent — iterates sources and produces seeds for row generation.

V5: Seed yielders are spawned by the pipeline executor after the orchestrator
defines the pipeline. Each yielder:
1. Receives its partitioned subset of sources from the pipeline config
2. Iterates sources (via search, file reading, or synthetic design)
3. Calls yield_seed() for each valid seed found
4. Reports via done() when finished or exits when quota met

Yielders can operate agentically (LLM browses and calls yield_seed) or
programmatically (code_exec script writes seeds to a file).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional

from dsl_worker.agents.base import AgentConversation, AgentResult
from dsl_worker.agents.tools import ToolRegistry
from dsl_worker.billing.tracked_client import TrackedOpenAIClient
from dsl_worker.infra.pipeline import PipelineConfig, Seed, VariableConfig

logger = logging.getLogger(__name__)

READ_FILE_LIMIT = 30_000

# Hard cap: if yielder submits this many times its target without hitting
# accepted quota, force-stop to prevent runaway cost.
HARD_CAP_MULTIPLIER = 3


SEED_YIELDER_SYSTEM_PROMPT = """\
You are a seed yielder for a dataset generation pipeline. Your job is to find and \
submit seeds — specific variable values that will each become one row in the dataset.

<template>
{template}
</template>

<variables>
{variables_description}
</variables>

<sources>
{sources_description}
</sources>

<strategy>
{strategy_description}
</strategy>

<instructions>
{seed_instructions}
</instructions>

<research_context>
{research_context}
</research_context>

## How to Work

1. Start from your assigned sources. Follow the strategy and instructions above.
2. Systematically iterate through your sources, calling yield_seed() for each item you find.
3. Include useful context in yield_seed metadata (URLs, line numbers, source references) \
so the row generator can find the content efficiently.
4. yield_seed() returns pipeline status. Watch for these signals:
   - "pipeline full" → call done() immediately, the dataset has enough seeds.
   - "fair share reached" → call done(), let other yielders contribute diversity.
   - "rejected (dedup)" → try different items, don't re-yield similar seeds.
5. Use find() and open(ref_id, start_line) to navigate through long pages.
   Use click() to follow pagination links.
6. Call done() when your source is exhausted or pipeline signals to stop.

## Browsing

- **open(url)** — Opens a page and returns line-numbered markdown with links table. Fast (~2s).
- **find(ref_id, pattern)** — Search within an already-opened page.
- **click(ref_id, link_id)** — Follow a link from the links table.
- **interact(url_or_ref_id, task)** — ONLY use when open() returns an anti-bot/Cloudflare \
challenge page instead of real content. Give it a small, specific navigation task like \
"bypass the challenge" or "click Accept Cookies". Do NOT ask it to list, extract, or \
summarize content — you see the page directly after it navigates.

## Important

- Focus on QUANTITY and SPEED. Yield items as fast as you can.
- Follow the strategy and instructions — they tell you what to iterate and how.
- Do NOT use interact() to extract or summarize content. Just open() the page and read it.
- If a seed gets rejected, move on. Try a different item.
- Do not yield duplicates of seeds you've already submitted.
- Include source_url, line_range, or other context in yield_seed metadata — \
this helps the row generator find the content faster.

## Tools

- yield_seed(values, metadata): Submit a seed. Include metadata like \
{{"source_url": "...", "source_ref": "p0", "line_range": "45-80"}} to help row generators.
- brave_search(query): Search the web.
- open(ref_id_or_url, start_line): View a page. ALWAYS try this first.
- find(ref_id, pattern): Search within a loaded page.
- click(ref_id, link_id): Follow a link.
- code_exec(script, description): Execute Python for programmatic iteration.
- read_file(path): Read a workspace file.
- interact(url_or_ref_id, task): Browser navigation agent. ONLY for anti-bot bypass or \
page interactions (clicking buttons, scrolling to load). NOT for reading or extracting content.
- done(reason): Signal you're finished yielding.
"""


class SeedYielderAgent:
    """
    Seed yielder — iterates sources, produces seeds for row generation.

    Spawned by the pipeline executor. Runs independently.

    Usage:
        yielder = SeedYielderAgent(
            pipeline_config=config,
            yielder_index=0,
            total_yielders=3,
            openai_client=tracked_client,
            model="gpt-5.2",
            workspace_dir=Path("/workspace"),
            on_yield_seed=seed_processor.submit_seed,
            ...
        )
        result = await yielder.run()
    """

    def __init__(
        self,
        pipeline_config: PipelineConfig,
        yielder_index: int,
        total_yielders: int,
        openai_client: TrackedOpenAIClient,
        model: str,
        workspace_dir: Path,
        on_yield_seed: Callable[[Seed, str], Awaitable[Dict[str, Any]]],
        brave_api_key: Optional[str] = None,
        sandbox: Optional[Any] = None,
        stop_checker: Optional[Callable[[], bool]] = None,
        blob_service_client: Optional[Any] = None,
        project_id: Optional[Any] = None,
        on_tool_call: Optional[Callable[[str, str], None]] = None,
        on_cost: Optional[Callable] = None,
        mcp_tools: Optional[List[Dict[str, Any]]] = None,
        langfuse_parent: Optional[Any] = None,
        on_browser_started: Optional[Callable] = None,
    ) -> None:
        self.pipeline_config = pipeline_config
        self.yielder_index = yielder_index
        self.total_yielders = total_yielders
        self.workspace_dir = Path(workspace_dir)
        self.on_yield_seed = on_yield_seed
        self.stop_checker = stop_checker
        self.on_tool_call = on_tool_call
        self.on_cost = on_cost

        # State
        self._yielded_count = 0   # total submissions (including rejected)
        self._accepted_count = 0  # accepted seeds only
        self._is_done = False
        self._pipeline_full = False
        self.variables = pipeline_config.variables

        # Build research tools
        from dsl_worker.infra.research_tools import ResearchTools, ResearchScope

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
            on_browser_started=on_browser_started,
        )
        self._impl.set_scope(ResearchScope(
            id=f"seed_yielder:{yielder_index}",
            description="",
            quota=0,
        ))

        registry = ToolRegistry()
        self._register_tools(registry)

        # Build system prompt
        system_prompt = self._build_system_prompt()

        self._conversation = AgentConversation(
            openai_client=openai_client,
            model=model,
            system_prompt=system_prompt,
            tools=registry,
            stop_checker=stop_checker,
            max_turns=50,
            reasoning={"effort": "medium", "summary": "detailed"},
            label=f"seed_yielder:{yielder_index}",
            on_tool_call=on_tool_call,
            on_cost=on_cost,
            extra_tools=mcp_tools or [],
            langfuse_parent=langfuse_parent,
        )

    def _partition_sources(self) -> List[VariableConfig]:
        """Create a copy of variables with sources partitioned for this yielder.

        Round-robin distributes seed_sources across yielders so each gets a
        distinct subset. If there are fewer sources than yielders, some
        yielders may share a source.
        """
        if self.total_yielders <= 1:
            return self.variables

        partitioned = []
        for var in self.variables:
            if not var.seed_sources:
                partitioned.append(var)
                continue

            # Round-robin partition
            my_sources = [
                src for i, src in enumerate(var.seed_sources)
                if i % self.total_yielders == self.yielder_index
            ]

            # If this yielder got no sources (fewer sources than yielders),
            # assign at least one to avoid an idle yielder
            if not my_sources and var.seed_sources:
                my_sources = [var.seed_sources[self.yielder_index % len(var.seed_sources)]]

            partitioned.append(VariableConfig(
                name=var.name,
                description=var.description,
                seed_strategy=var.seed_strategy,
                seed_sources=my_sources,
                seed_context=var.seed_context,
                seed_instructions=var.seed_instructions,
            ))
        return partitioned

    def _build_system_prompt(self) -> str:
        """Build system prompt from pipeline config."""
        # Partition sources for this yielder
        my_variables = self._partition_sources()

        # Variables description
        var_parts = []
        for v in my_variables:
            parts = [f"- {v.name}: {v.description}"]
            parts.append(f"  Strategy: {v.seed_strategy}")
            if v.seed_instructions:
                parts.append(f"  Instructions: {v.seed_instructions}")
            var_parts.append("\n".join(parts))
        variables_description = "\n".join(var_parts) if var_parts else "(no variables)"

        # Sources description — only this yielder's partitioned sources
        source_parts = []
        for v in my_variables:
            if v.seed_sources:
                source_parts.append(f"{v.name}:")
                for src in v.seed_sources:
                    source_parts.append(f"  - {src}")
        sources_description = "\n".join(source_parts) if source_parts else "(no specific sources assigned)"

        # Strategy description — aggregate from variables
        strategies = set(v.seed_strategy for v in self.variables)
        if "search" in strategies:
            strategy_description = (
                "Search-based: Use web searches and browsing to find specific items. "
                "Each item becomes a seed with the required variable values."
            )
        elif "iterate" in strategies:
            strategy_description = (
                "Iteration-based: Read through sources (files, paginated results, etc.) "
                "and extract items systematically."
            )
        elif "synthetic" in strategies:
            strategy_description = (
                "Synthetic: You need to DISCOVER what seeds should be. Research the topic "
                "area first — search for information, browse sources, build understanding "
                "of what items/topics/categories exist in this domain. Then yield seeds "
                "for each one you discover. Your research IS the seed discovery process.\n\n"
                "Approach:\n"
                "1. Search broadly for the topic area\n"
                "2. Open 2-3 promising sources to understand the landscape\n"
                "3. Start yielding seeds as you discover items\n"
                "4. Continue searching/browsing for more if you haven't hit quota"
            )
        else:
            strategy_description = "Mixed strategy — see variable-level instructions."

        # Seed instructions
        seed_instructions = self.pipeline_config.seed_yielder_instructions or "(no additional instructions)"

        return SEED_YIELDER_SYSTEM_PROMPT.format(
            template=self.pipeline_config.template,
            variables_description=variables_description,
            sources_description=sources_description,
            strategy_description=strategy_description,
            seed_instructions=seed_instructions,
            research_context=self.pipeline_config.research_context or "(no research context provided)",
        )

    def _register_tools(self, registry: ToolRegistry) -> None:
        """Register seed yielder tools."""

        # --- Research/browsing tools ---
        self._impl.register_on(registry)

        # --- read_file ---
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
            description="Read a file from the workspace (sources, uploads, repo, etc.).",
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

        # --- yield_seed ---
        async def yield_seed(args: Dict) -> tuple[str, float]:
            values = args.get("values", {})
            metadata = args.get("metadata", {})

            # Validate all required variables are present
            missing = [v.name for v in self.variables if v.name not in values]
            if missing:
                return f"Error: missing variables {missing}", 0.0

            seed = Seed(values=values, metadata=metadata)
            status = await self.on_yield_seed(seed, str(self.yielder_index))
            self._yielded_count += 1

            stats = status["stats"]
            accepted_count = stats["accepted"]
            remaining = stats["remaining"]

            if not status["accepted"]:
                return (
                    f"Seed REJECTED ({status['reason']}). "
                    f"Pipeline: {accepted_count} accepted, {remaining} remaining. "
                    f"Try a different item."
                ), 0.0

            self._accepted_count += 1

            # Pipeline full — no more seeds needed
            if remaining <= 0:
                self._pipeline_full = True
                return (
                    f"Seed accepted. Pipeline full ({accepted_count} accepted). "
                    f"Call done()."
                ), 0.0

            # Fair share throttle — let other yielders contribute diversity
            if status.get("over_fair_share"):
                self._is_done = True
                return (
                    f"Seed accepted. You've contributed {self._accepted_count} seeds — "
                    f"fair share reached. Call done() to let other sources contribute. "
                    f"Pipeline: {accepted_count} accepted, {remaining} remaining."
                ), 0.0

            advice = ""
            if stats.get("rejected_dedup", 0) > accepted_count and accepted_count > 0:
                advice = " High dedup rejection — try different sources/approaches."

            return (
                f"Seed accepted ({self._accepted_count} from you). "
                f"Pipeline: {accepted_count} accepted, {remaining} remaining.{advice}"
            ), 0.0

        # Build variable properties for the tool schema
        var_properties = {}
        for v in self.variables:
            var_properties[v.name] = {
                "type": "string",
                "description": v.description,
            }

        registry.add(
            name="yield_seed",
            description=(
                "Submit a seed with resolved variable values. Each seed becomes one row. "
                "All variables must be present in the values object."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "values": {
                        "type": "object",
                        "description": "Variable name → resolved value",
                        "properties": var_properties,
                        "required": [v.name for v in self.variables],
                    },
                    "metadata": {
                        "type": "object",
                        "description": "Optional metadata (source URL, notes, tags, etc.)",
                    },
                },
                "required": ["values"],
            },
            handler=yield_seed,
        )

        # --- done ---
        async def done(args: Dict) -> tuple[str, float]:
            reason = args.get("reason", "complete")
            self._is_done = True
            return (
                f"Seed yielder done: {reason}. "
                f"Yielded {self._yielded_count} (accepted {self._accepted_count})."
            ), 0.0

        registry.add(
            name="done",
            description=(
                "Signal you're finished yielding seeds. Call when you've hit your "
                "target or exhausted sources."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "reason": {
                        "type": "string",
                        "description": "Why you're done (e.g., 'target reached', 'sources exhausted')",
                    },
                },
            },
            handler=done,
        )

    async def run(self) -> AgentResult:
        """Run the seed yielder."""
        result = await self._conversation.send(
            "Begin yielding seeds from your assigned sources.",
            exit_condition=lambda: self._is_done or self._pipeline_full,
        )

        logger.info(
            f"[seed_yielder:{self.yielder_index}] finished: "
            f"{self._accepted_count} accepted, "
            f"{self._yielded_count} total submitted"
        )

        return result

    @property
    def cost_usd(self) -> float:
        return self._conversation.total_cost

    async def cleanup(self) -> None:
        """Clean up browser and other resources."""
        try:
            await self._impl.cleanup()
        except Exception as e:
            logger.warning(f"Seed yielder {self.yielder_index} cleanup error: {e}")
