"""
Orchestrator agent — coordinates dataset generation.

V6.1: Pure strategist. The orchestrator stays in the loop throughout execution,
dispatching typed subagents incrementally, seeing results, and adapting.
No low-level tools (browsing, search, code, files) — all investigation
goes through research() subagents.

Tools:
- research(question, ...) — spawn research subagent
- write_template(instructions, variables) — set row generation template
- parse_seeds(sources, quota, ...) — launch iterative seed generator
- synthesize_seeds(topic, quota, ...) — launch discovery seed generator
- set_dedup(strategy, field, ...) — configure deduplication
- get_status() — check pipeline progress
- done() — signal completion (sources exhausted)
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional

from dsl_worker.agents.base import AgentConversation, AgentResult
from dsl_worker.agents.tools import ToolRegistry
from dsl_worker.billing.tracked_client import TrackedOpenAIClient
from dsl_worker.infra.pipeline import (
    DedupConfig,
    PipelineConfig,
    SeedProcessor,
    VariableConfig,
)

logger = logging.getLogger(__name__)


ORCHESTRATOR_SYSTEM_PROMPT = """\
You are the orchestrator for a dataset generation system. A user described a dataset \
they want. Your job is to figure out how to produce it and coordinate execution.

You stay in the loop throughout — dispatch research, write the row template, launch \
seed producers, see results, and adapt.

## Your Subagents

You coordinate three types of subagents:

**research(question, ...)** — Recon agents that investigate specific questions. \
They can browse the web, search, run code, and read files. They return findings. \
Use these to understand the landscape before writing a template.

**parse_seeds(sources, quota, instructions)** — Source iterators that crawl known \
sources (URLs, search results, paginated pages) and yield seeds. Each seed becomes \
one row. Use when you have concrete sources to iterate.

**synthesize_seeds(topic, quota, instructions)** — Discovery agents that research \
a topic area and yield seeds as they discover items. Use when seeds need to be \
found through research, not just extracted from a known list.

## Common Patterns

**Extraction** — rows from real sources (job listings, products, directories)
  Quick recon to understand the site structure, then parse_seeds on listing URLs.

**Condensation** — rows synthesize info around topics (tips, summaries, guides)
  Research to discover what topics exist, then synthesize_seeds to find and yield them.

**Enrichment** — rows already exist (CSV upload), need additional columns
  Read the file via research(), then parse_seeds to iterate its rows.

**Fan-out** — one source becomes many rows (repo → Q&A, textbook → flashcards)
  Understand the source structure, then parse_seeds to iterate its elements.

**Synthesis** — rows invented from domain knowledge (training data, scenarios)
  Research to build a taxonomy for diversity, then synthesize_seeds to discover items.

## Workflow

1. **Reason** — Which pattern? What unknowns need resolving?

2. **Research** (if needed) — Dispatch research() agents for specific questions.
   - Ask focused questions: "What fields are on an Upwork job listing page?"
   - Call multiple in parallel for different questions.
   - Once findings come back, MOVE ON. Don't over-research.

3. **Write template** — Call write_template() with row generation instructions + \
variable definitions. Variables are the things that change per row (the seed values). \
Include a research approach section if row generators need to look things up.

4. **Produce seeds** — Launch subagents:
   - parse_seeds() for iterating known sources (URLs, files, search results)
   - synthesize_seeds() for discovering seeds via research
   - ONE source per parse_seeds() call. Launch multiple in parallel for different sources.
     Example: 3 listing URLs → 3 parallel parse_seeds() calls, each with one source.
   - For synthesize_seeds(), split by topic partition for diversity.

5. **React** — Check get_status(). If short on seeds, launch more subagents with \
different sources or topics. If enough, call done().

## Principles

- Move fast. Don't over-research. For extraction, one research agent to understand \
the source is usually enough.
- Each subagent gets a FOCUSED scope — specific URLs or queries, not "find everything."
- The row generator has full browsing capabilities. It can visit pages, research, and \
extract. Your job is to give it good seeds and clear instructions.
- Our browsing stack handles anti-bot, CAPTCHAs, and JS-heavy pages automatically.
- After research returns, synthesize findings → write template → produce seeds. No detours.
- For expert or fast-moving topics, prefer real data and sources over your own knowledge. \
Research what actually exists rather than guessing.
- Row generators will skip rows that turn out to be dead ends (broken URLs, unavailable \
content). Overshoot seed count slightly to account for this.

<conversation>
{conversation_summary}
</conversation>

<schema>
{columns_description}
</schema>

<resources>
{resources_section}
</resources>

Target: {num_samples} rows.

## Tools

- research(question, scope, budget, output_format): Spawn a research subagent. \
Returns findings. Call multiple in parallel.
- write_template(instructions, variables): Set row generation instructions with \
{{variable}} placeholders. Must call before launching seed producers.
- parse_seeds(sources, quota, instructions): Launch a seed producer that iterates \
sources. Blocks until done. Call multiple in parallel.
- synthesize_seeds(topic, quota, instructions, research_context): Launch a seed \
producer that discovers seeds via research. Blocks until done.
- set_dedup(strategy, field, threshold): Configure deduplication.
- get_status(): Check pipeline progress.
- done(reason): Signal completion — all seed sources have been dispatched. \
Row generation continues in background.

You do NOT have browsing, search, code execution, or file reading tools. \
All investigation goes through research() subagents.

{feedback_section}
"""


class OrchestratorAgent:
    """
    V6 Orchestrator. Stays in the loop, dispatches typed subagents,
    sees results, adapts. Replaces V5's fire-and-forget pipeline design.

    Usage:
        orchestrator = OrchestratorAgent(
            chat_history=[...],
            columns=[...],
            num_samples=100,
            seed_processor=seed_processor,
            generation_stats=generation_stats,
            openai_client=tracked_client,
            ...
        )
        await orchestrator.run()
    """

    def __init__(
        self,
        chat_history: List[Dict[str, str]],
        columns: List[Dict[str, Any]],
        num_samples: int,
        openai_client: TrackedOpenAIClient,
        model: str,
        workspace_dir: Path,
        seed_processor: SeedProcessor,
        generation_stats: Dict[str, Any],
        uploaded_files: Optional[List[Dict[str, Any]]] = None,
        brave_api_key: Optional[str] = None,
        sandbox: Optional[Any] = None,
        stop_checker: Optional[Callable[[], bool]] = None,
        cost_checker: Optional[Callable[[], tuple[bool, Optional[str]]]] = None,
        blob_service_client: Optional[Any] = None,
        project_id: Optional[Any] = None,
        on_tool_call: Optional[Callable[[str, str], None]] = None,
        on_cost: Optional[Callable] = None,
        uploaded_file_urls: Optional[Dict[str, str]] = None,
        mcp_tools: Optional[List[Dict[str, Any]]] = None,
        feedback_context: Optional[Dict[str, Any]] = None,
        langfuse_parent: Optional[Any] = None,
        on_browser_started: Optional[Callable] = None,
        # V6: model for spawned subagents
        yielder_model: str = "",
    ) -> None:
        self.feedback_context = feedback_context
        self.chat_history = chat_history
        self.columns = columns
        self.num_samples = num_samples
        self.workspace_dir = Path(workspace_dir)
        self.openai_client = openai_client
        self.model = model
        self.brave_api_key = brave_api_key
        self.sandbox = sandbox
        self.stop_checker = stop_checker
        self.cost_checker = cost_checker
        self.blob_service_client = blob_service_client
        self.project_id = project_id
        self.on_tool_call = on_tool_call
        self.on_cost = on_cost
        self.uploaded_file_urls = uploaded_file_urls
        self.mcp_tools = mcp_tools or []
        self.on_browser_started = on_browser_started
        self.langfuse_parent = langfuse_parent
        self.yielder_model = yielder_model or model

        # V6 state
        self._is_done = False
        self._research_counter = 0
        self._subagent_counter = 0
        self._seed_processor = seed_processor
        self._generation_stats = generation_stats

        # Pipeline config built incrementally
        self._template: Optional[str] = None
        self._variables: List[VariableConfig] = []
        self._dedup: DedupConfig = DedupConfig()
        self._research_context: str = ""

        # Build tools
        registry = ToolRegistry()
        self._register_tools(registry)

        # Build system prompt
        columns_desc = self._format_columns()
        convo_summary = self._format_conversation()
        resources_section = self._format_resources(uploaded_files)
        feedback_section = self._format_feedback()

        system_prompt = ORCHESTRATOR_SYSTEM_PROMPT.format(
            num_samples=num_samples,
            columns_description=columns_desc,
            conversation_summary=convo_summary,
            resources_section=resources_section,
            feedback_section=feedback_section,
        )

        # V6: Higher turn limits since orchestrator stays in the loop
        # and waits for blocking subagent calls.
        from dsl_worker.config import settings
        max_turns = getattr(settings, 'orchestrator_max_turns', 40)
        soft_limit = getattr(settings, 'orchestrator_soft_limit', 25)

        self._conversation = AgentConversation(
            openai_client=openai_client,
            model=model,
            system_prompt=system_prompt,
            tools=registry,
            stop_checker=stop_checker,
            max_turns=max_turns,
            soft_turn_limit=soft_limit,
            reasoning={"effort": "high", "summary": "detailed"},
            label="orchestrator",
            continue_on_text=True,
            on_tool_call=on_tool_call,
            on_cost=on_cost,
            extra_tools=self.mcp_tools,
            langfuse_parent=langfuse_parent,
        )

    def _format_columns(self) -> str:
        if not self.columns:
            return "(no columns defined)"
        return json.dumps(self.columns, indent=2)

    def _format_conversation(self) -> str:
        if not self.chat_history:
            return "(no conversation history)"

        parts = []
        for msg in self.chat_history:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            parts.append(f"**{role}**: {content}")
        return "\n\n".join(parts)

    def _format_resources(self, uploaded_files: Optional[List[Dict[str, Any]]]) -> str:
        lines = []
        if uploaded_files:
            lines.append("Uploaded files:")
            for f in uploaded_files:
                name = f.get("filename", "unknown")
                size = f.get("size_bytes", 0)
                ctype = f.get("content_type", "")
                if size > 1_000_000:
                    size_str = f"{size / 1_000_000:.1f} MB"
                elif size > 1_000:
                    size_str = f"{size / 1_000:.0f} KB"
                else:
                    size_str = f"{size} bytes"
                lines.append(f"  - {name} ({ctype}, {size_str})")
        else:
            lines.append("No uploaded files.")
        return "\n".join(lines)

    def _format_feedback(self) -> str:
        if not self.feedback_context:
            return ""

        prev = self.feedback_context.get("previous_config", {})
        feedback = self.feedback_context.get("user_feedback", "")

        return (
            f"**Previous pipeline config:**\n"
            f"```json\n{json.dumps(prev, indent=2)}\n```\n\n"
            f"**User feedback:** \"{feedback}\"\n\n"
            f"The previous results were discarded. Design a new pipeline "
            f"based on this feedback."
        )

    def _register_tools(self, registry: ToolRegistry) -> None:
        """Register V6.1 orchestrator tools — pure coordination, no low-level tools."""

        # --- research ---
        async def research(args: Dict) -> tuple[str, float]:
            """Spawn a research subagent to explore a question."""
            from dsl_worker.agents.research import ResearchAgent

            question = args.get("question", "")
            scope = args.get("scope", "")
            budget = args.get("budget", 10)
            output_format = args.get("output_format", "")

            if not question:
                return "Error: question is required", 0.0

            # Get the current langfuse span for nesting
            langfuse_span = getattr(self._conversation, "_current_langfuse_span", None)

            agent = ResearchAgent(
                openai_client=self.openai_client,
                model=self.model,
                workspace_dir=self.workspace_dir,
                brave_api_key=self.brave_api_key,
                sandbox=self.sandbox,
                stop_checker=self.stop_checker,
                max_turns=budget,
                blob_service_client=self.blob_service_client,
                project_id=self.project_id,
                on_browser_started=self.on_browser_started,
            )
            # Override langfuse parent so subagent traces nest under orchestrator
            if langfuse_span:
                agent._conversation.langfuse_parent = langfuse_span

            full_question = question
            if scope:
                full_question += f"\n\nScope: {scope}"
            if output_format:
                full_question += f"\n\nExpected output format: {output_format}"

            try:
                result = await agent.ask_full(full_question)
            finally:
                await agent.cleanup()

            # Track subagent cost
            if self.on_cost and result.cost_usd > 0:
                await self.on_cost(result.cost_usd, "research_subagent")

            # Save research results to workspace file
            n = self._research_counter
            self._research_counter += 1
            research_dir = self.workspace_dir / "research"
            research_dir.mkdir(exist_ok=True)
            filepath = research_dir / f"finding_{n}.md"
            try:
                filepath.write_text(
                    f"# Research: {question}\n\n{result.text}",
                    encoding="utf-8",
                )
            except Exception as e:
                logger.warning(f"Failed to save research finding: {e}")

            return (
                f"[Saved to research/finding_{n}.md]\n\n{result.text}"
            ), result.cost_usd

        registry.add(
            name="research",
            description=(
                "Spawn a research subagent to explore a question. Returns findings "
                "(also saved to research/finding_N.md). Call multiple in one response "
                "for parallel research."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "Specific research question to investigate",
                    },
                    "scope": {
                        "type": "string",
                        "description": (
                            "Focus area for the research "
                            "(e.g., 'focus on official documentation')"
                        ),
                    },
                    "budget": {
                        "type": "integer",
                        "description": (
                            "Max tool calls for the subagent (default 10). "
                            "Use 5 for quick lookups, 10-15 for moderate, 20 for deep."
                        ),
                    },
                    "output_format": {
                        "type": "string",
                        "description": (
                            "Expected format for the answer (e.g., 'Return a bulleted "
                            "list of URL parameters with names and descriptions', "
                            "'Return a JSON mapping of field names to CSS selectors')"
                        ),
                    },
                },
                "required": ["question"],
            },
            handler=research,
        )

        # --- write_template (V6 — replaces define_pipeline) ---
        async def write_template(args: Dict) -> tuple[str, float]:
            """Set the row generation template and declare variables."""
            template = args.get("instructions", "")
            variables_raw = args.get("variables", [])

            if not template:
                return "Error: instructions is required", 0.0

            try:
                variables = [
                    VariableConfig(
                        name=v["name"],
                        description=v.get("description", ""),
                        seed_strategy="iterate",  # default; overridden by parse_seeds/synthesize_seeds
                    )
                    for v in variables_raw
                ]
            except (TypeError, KeyError) as e:
                return f"Error parsing variables: {e}. Each variable needs at least a 'name'.", 0.0

            self._template = template
            self._variables = variables

            # Update SeedProcessor
            self._seed_processor.set_template(template)
            self._seed_processor.set_variables(variables)
            if self._research_context:
                self._seed_processor.set_research_context(self._research_context)

            var_names = [v.name for v in variables]
            return (
                f"Template set with {len(variables)} variable(s): "
                f"{', '.join(var_names) if var_names else '(none)'}. "
                f"You can now parse_seeds or synthesize_seeds."
            ), 0.0

        registry.add(
            name="write_template",
            description=(
                "Set the row generation template. This becomes each row generator's "
                "instructions. Use {variable_name} placeholders for values that change "
                "per row. Must be called before submitting seeds or spawning yielders."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "instructions": {
                        "type": "string",
                        "description": (
                            "Row generation instructions with {variable} placeholders. "
                            "Include a 'Research approach' section if row generators "
                            "need to do additional per-row research."
                        ),
                    },
                    "variables": {
                        "type": "array",
                        "description": "Variables that change per row (used in template as {name})",
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": {
                                    "type": "string",
                                    "description": "Variable name (used in template as {name})",
                                },
                                "description": {
                                    "type": "string",
                                    "description": "What this variable represents",
                                },
                            },
                            "required": ["name", "description"],
                        },
                    },
                },
                "required": ["instructions"],
            },
            handler=write_template,
        )

        # --- parse_seeds (iterates known sources) ---
        async def parse_seeds(args: Dict) -> tuple[str, float]:
            """Launch an iterative seed producer. Blocks until it finishes."""
            from dsl_worker.agents.seed_yielder import SeedYielderAgent

            sources = args.get("sources", [])
            quota = args.get("quota", 20)
            instructions = args.get("instructions", "")

            if not self._template:
                return "Error: call write_template first", 0.0

            # Build a PipelineConfig for this yielder
            variables = []
            for v in self._variables:
                variables.append(VariableConfig(
                    name=v.name,
                    description=v.description,
                    seed_strategy="iterate",
                    seed_sources=sources,
                    seed_instructions=instructions,
                ))

            config = PipelineConfig(
                template=self._template,
                variables=variables,
                dedup=self._dedup,
                target_rows=quota,
                research_context=self._research_context,
            )

            idx = self._subagent_counter
            self._subagent_counter += 1

            langfuse_span = getattr(self._conversation, "_current_langfuse_span", None)

            yielder = SeedYielderAgent(
                pipeline_config=config,
                yielder_index=idx,
                total_yielders=1,
                openai_client=self.openai_client,
                model=self.yielder_model,
                workspace_dir=self.workspace_dir,
                on_yield_seed=self._seed_processor.submit_seed,
                brave_api_key=self.brave_api_key,
                sandbox=self.sandbox,
                stop_checker=self.stop_checker,
                blob_service_client=self.blob_service_client,
                project_id=self.project_id,
                on_tool_call=self.on_tool_call,
                on_cost=self.on_cost,
                mcp_tools=self.mcp_tools,
                langfuse_parent=langfuse_span,
                on_browser_started=self.on_browser_started,
            )

            try:
                result = await yielder.run()
            finally:
                await yielder.cleanup()

            if self.on_cost and yielder.cost_usd > 0:
                await self.on_cost(yielder.cost_usd, f"parse_seeds:{idx}")

            stats = self._seed_processor.stats
            return (
                f"Parser {idx} finished: {result.turns_taken} turns, "
                f"cost=${yielder.cost_usd:.3f}. "
                f"Pipeline: {stats['accepted']} accepted, "
                f"{stats['remaining']} remaining."
            ), yielder.cost_usd

        registry.add(
            name="parse_seeds",
            description=(
                "Launch a seed producer that iterates known sources (URLs, pages, "
                "search results) and yields seeds. Blocks until it finishes. "
                "Call multiple in parallel for different source partitions."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "sources": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "URLs, file paths, or search queries to iterate. "
                            "The agent will browse these and yield seeds."
                        ),
                    },
                    "quota": {
                        "type": "integer",
                        "description": (
                            "How many accepted seeds to aim for (default 20). "
                            "Set based on how many items you expect the sources to contain."
                        ),
                    },
                    "instructions": {
                        "type": "string",
                        "description": (
                            "Specific instructions for this agent "
                            "(e.g., 'Extract job title and URL from each listing')"
                        ),
                    },
                },
                "required": ["sources"],
            },
            handler=parse_seeds,
        )

        # --- synthesize_seeds (discovers seeds via research) ---
        async def synthesize_seeds(args: Dict) -> tuple[str, float]:
            """Launch a discovery seed producer. Blocks until it finishes."""
            from dsl_worker.agents.seed_yielder import SeedYielderAgent

            topic = args.get("topic", "")
            quota = args.get("quota", 20)
            instructions = args.get("instructions", "")
            research_context = args.get("research_context", "")

            if not self._template:
                return "Error: call write_template first", 0.0

            # Build a PipelineConfig for this synthesizer
            variables = []
            for v in self._variables:
                variables.append(VariableConfig(
                    name=v.name,
                    description=v.description,
                    seed_strategy="synthetic",
                    seed_context=topic,
                    seed_instructions=instructions,
                ))

            effective_context = research_context or self._research_context
            config = PipelineConfig(
                template=self._template,
                variables=variables,
                dedup=self._dedup,
                target_rows=quota,
                research_context=effective_context,
            )

            idx = self._subagent_counter
            self._subagent_counter += 1

            langfuse_span = getattr(self._conversation, "_current_langfuse_span", None)

            yielder = SeedYielderAgent(
                pipeline_config=config,
                yielder_index=idx,
                total_yielders=1,
                openai_client=self.openai_client,
                model=self.yielder_model,
                workspace_dir=self.workspace_dir,
                on_yield_seed=self._seed_processor.submit_seed,
                brave_api_key=self.brave_api_key,
                sandbox=self.sandbox,
                stop_checker=self.stop_checker,
                blob_service_client=self.blob_service_client,
                project_id=self.project_id,
                on_tool_call=self.on_tool_call,
                on_cost=self.on_cost,
                mcp_tools=self.mcp_tools,
                langfuse_parent=langfuse_span,
                on_browser_started=self.on_browser_started,
            )

            try:
                result = await yielder.run()
            finally:
                await yielder.cleanup()

            if self.on_cost and yielder.cost_usd > 0:
                await self.on_cost(yielder.cost_usd, f"synthesize_seeds:{idx}")

            stats = self._seed_processor.stats
            return (
                f"Synthesizer {idx} finished: {result.turns_taken} turns, "
                f"cost=${yielder.cost_usd:.3f}. "
                f"Pipeline: {stats['accepted']} accepted, "
                f"{stats['remaining']} remaining."
            ), yielder.cost_usd

        registry.add(
            name="synthesize_seeds",
            description=(
                "Launch a discovery agent that researches a topic area and yields "
                "seeds as it discovers items. Blocks until it finishes. Use when seeds "
                "need to be found through research, not just iterated from a known source."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "topic": {
                        "type": "string",
                        "description": (
                            "Topic area to research and discover seeds from "
                            "(e.g., 'DayZ survival mechanics', "
                            "'AI companies in Minnesota')"
                        ),
                    },
                    "quota": {
                        "type": "integer",
                        "description": "How many accepted seeds to aim for (default 20)",
                    },
                    "instructions": {
                        "type": "string",
                        "description": "Specific instructions for this agent",
                    },
                    "research_context": {
                        "type": "string",
                        "description": (
                            "Research findings to seed the agent with. "
                            "If not provided, uses accumulated research context."
                        ),
                    },
                },
                "required": ["topic"],
            },
            handler=synthesize_seeds,
        )

        # --- set_dedup ---
        async def set_dedup(args: Dict) -> tuple[str, float]:
            """Configure seed deduplication."""
            strategy = args.get("strategy", "exact")
            dedup_field = args.get("field", "")
            threshold = args.get("threshold", 0.85)

            self._dedup = DedupConfig(
                strategy=strategy,
                field=dedup_field,
                threshold=threshold,
            )
            self._seed_processor.set_dedup(self._dedup)
            return f"Dedup set: {strategy} on '{dedup_field}'", 0.0

        registry.add(
            name="set_dedup",
            description=(
                "Configure seed deduplication. Usually 'exact' on the main variable. "
                "Call before launching seed producers."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "strategy": {
                        "type": "string",
                        "enum": ["none", "exact", "embedding_similarity"],
                        "description": "Dedup strategy",
                    },
                    "field": {
                        "type": "string",
                        "description": "Variable name to dedup on (empty = all values)",
                    },
                    "threshold": {
                        "type": "number",
                        "description": "Similarity threshold for embedding strategy (default 0.85)",
                    },
                },
            },
            handler=set_dedup,
        )

        # --- get_status ---
        async def get_status(args: Dict) -> tuple[str, float]:
            """Check current pipeline progress."""
            stats = self._seed_processor.stats
            gen = self._generation_stats
            return (
                f"Seeds: {stats['accepted']} accepted, {stats['remaining']} remaining, "
                f"{stats['rejected_dedup']} dedup rejected, "
                f"{stats['submitted_total']} total submitted.\n"
                f"Rows: {gen.get('rows_generated', 0)} generated, "
                f"{gen.get('skipped', 0)} skipped, "
                f"{gen.get('errors', 0)} errors."
            ), 0.0

        registry.add(
            name="get_status",
            description="Check current pipeline progress (seeds accepted, rows generated, etc.).",
            parameters={"type": "object", "properties": {}},
            handler=get_status,
        )

        # --- done ---
        async def done(args: Dict) -> tuple[str, float]:
            reason = args.get("reason", "complete")
            self._is_done = True

            # Save final config as recipe for checkpoint
            self._save_recipe()

            return f"Orchestrator done: {reason}", 0.0

        registry.add(
            name="done",
            description=(
                "Signal orchestration is complete — all seed sources have been "
                "dispatched or exhausted. Row generation continues in background."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "reason": {
                        "type": "string",
                        "description": "Why orchestration is done (e.g., 'all sources dispatched')",
                    },
                },
            },
            handler=done,
        )

    def _save_recipe(self) -> None:
        """Save current pipeline config as a recipe file for checkpoint/resume."""
        try:
            recipe = {
                "template": self._template,
                "variables": [
                    {"name": v.name, "description": v.description,
                     "seed_strategy": v.seed_strategy}
                    for v in self._variables
                ],
                "dedup": {
                    "strategy": self._dedup.strategy,
                    "field": self._dedup.field,
                    "threshold": self._dedup.threshold,
                },
                "research_context": self._research_context,
                "target_rows": self.num_samples,
            }
            recipe_path = self.workspace_dir / "pipeline_recipe.json"
            recipe_path.write_text(json.dumps(recipe, indent=2), encoding="utf-8")
        except Exception as e:
            logger.warning(f"Failed to save recipe: {e}")

    async def run(self) -> AgentResult:
        """Run the orchestrator."""
        if self.feedback_context:
            message = (
                "Begin. The user reviewed previous results and gave feedback "
                "(shown in system prompt). Research as needed and design a new pipeline."
            )
        else:
            message = (
                "Begin. Read the conversation history and resources, reason about "
                "strategy, research the landscape, then write a template and produce seeds."
            )

        result = await self._conversation.send(
            message,
            exit_condition=lambda: self._is_done,
        )
        return result

    @property
    def cost_usd(self) -> float:
        """Total cost accumulated by the orchestrator."""
        return self._conversation.total_cost

    async def cleanup(self) -> None:
        """Clean up resources."""
        # Orchestrator has no browser/sandbox resources of its own —
        # subagents (research, yielders) clean up after themselves.
        pass
