"""
Orchestrator agent — coordinates dataset generation.

V6: The orchestrator stays in the loop throughout execution. Instead of
designing a pipeline and exiting (V5), it dispatches typed subagents
incrementally, sees results, and adapts.

Tools:
- research(question, ...) — spawn research subagent
- write_template(instructions, variables) — set row generation template
- submit_seeds(seeds) — submit preset seeds directly
- spawn_yielder(sources, quota, ...) — launch iterative seed generator
- spawn_synthesizer(topic, quota, ...) — launch synthetic seed generator
- set_filter(name, description, ...) — add seed validation filter
- set_dedup(strategy, field, ...) — configure deduplication
- get_status() — check pipeline progress
- brave_search, read_file, code_exec — utility tools
- done() — signal completion
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
    FilterConfig,
    PipelineConfig,
    Seed,
    SeedProcessor,
    VariableConfig,
)

logger = logging.getLogger(__name__)

READ_FILE_LIMIT = 30_000


ORCHESTRATOR_SYSTEM_PROMPT = """\
You are the orchestrator for a dataset generation system. A user described a dataset
they want. Your job is to figure out the best strategy and coordinate execution.

You stay in the loop throughout — dispatch research, write the row template, produce
seeds, and adapt based on results.

## Dataset Archetypes

Identify which pattern fits — it determines your approach:

**Extraction** — rows scraped from real sources (job listings, products, directories)
  Research: light — figure out access patterns (URLs, pagination, anti-bot).
  Seeds: URLs or identifiers via spawn_yielder(). Row generator visits each and extracts.

**Condensation** — rows synthesize info around topics (wiki tips, guide summaries)
  Research: moderate — discover what topics exist and where sources live.
  Seeds: topics via submit_seeds() (from your research) or spawn_synthesizer().

**Enrichment** — rows already exist (CSV, database), need additional columns
  Research: minimal — data is provided. Seeds from input data via submit_seeds().

**Fan-out** — one source becomes many rows (code repo → Q&A, textbook → flashcards)
  Research: moderate — understand the source structure.
  Seeds: source elements via spawn_yielder() on the source.

**Synthesis** — rows invented from domain knowledge (training data, scenarios)
  Research: heavy — build a taxonomy so seeds have structural diversity.
  Seeds: taxonomy nodes via spawn_synthesizer() or submit_seeds().

## Your Workflow

1. REASON — What archetype? What unknowns need resolving?

2. RESEARCH — Dispatch research agents to explore the landscape.
   - Call research() with specific questions, output_format, appropriate budget
   - Call multiple in parallel for different aspects
   - Read results, then do targeted follow-ups if gaps remain
   - CRITICAL: Do NOT investigate yourself after research returns. If findings are
     insufficient, dispatch another research() agent.

3. WRITE TEMPLATE — Call write_template() with:
   - Row generation instructions with {{variable}} placeholders
   - Variable definitions (name, description)
   - Include a "Research approach" section if row generators need to look things up

4. CONFIGURE — Optionally call:
   - set_dedup() — usually "exact" on the main variable
   - set_filter() — only if seeds need validation beyond dedup

5. PRODUCE SEEDS — Use one or more:

   submit_seeds(seeds) — For items you already know from research.
     Best for: specific topics, URLs, or items discovered during research.
     Example: you researched DayZ tips → submit 50 tip topics directly.

   spawn_yielder(sources, quota, instructions) — For iterating known sources.
     Best for: paginated pages, directories, search results to crawl.
     Each yielder gets specific sources and a quota. Call multiple in parallel
     for different source partitions.

   spawn_synthesizer(topic, quota, instructions) — For discovering seeds via research.
     Best for: domains where seeds need to be discovered, not just listed.
     The synthesizer does its own research to find items, then yields seeds.

6. REACT — After subagents return, check results via get_status():
   - If enough seeds: call done()
   - If shortfall: dispatch more yielders/synthesizers with different sources
   - If high filter rejection: adjust approach or submit different seeds
   - If sources exhausted: try different sources or submit synthetic seeds

7. DONE — Call done() when satisfied. Row generation continues in the background.

## Key Principles

- Start simple. Try preset seeds or 1-2 yielders first. Scale up if needed.
- Each yielder/synthesizer gets a FOCUSED scope. "Search these 3 pages" not "find everything."
- React to results. If a yielder found 10/50, try different sources or approaches.
- After research returns: synthesize → write_template → produce seeds. No detours.
- Diversity comes from sources, not artificial imposition. Don't invent categories
  or distributions the user didn't ask for.

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

- research(question, scope, budget, output_format): Spawn a research subagent.
  Returns findings (also saved to workspace files). Call multiple in one response
  for parallel research. Budget controls tool calls (5=quick, 10-15=moderate, 20=deep).
  Subagents have full browsing capabilities (open, interact, code_exec, etc.).
- write_template(instructions, variables): Set the row generation template. Must be
  called before spawning yielders or submitting seeds.
- submit_seeds(seeds): Submit preset seed values directly. Each entry is a dict of
  variable_name → value.
- spawn_yielder(sources, quota, instructions): Launch an iterative seed generator.
  Blocks until it finishes. Call multiple in parallel for different source partitions.
- spawn_synthesizer(topic, quota, instructions, research_context): Launch a synthetic
  seed generator that researches and discovers seeds. Blocks until it finishes.
- set_filter(name, description, complexity): Add a seed validation filter.
- set_dedup(strategy, field, threshold): Configure seed deduplication.
- get_status(): Check current pipeline progress (seeds accepted, rows generated, etc.).
- brave_search(query): Quick web search for simple fact-checks ONLY. For any real
  investigation, use research().
- code_exec(script, description): Execute Python for data exploration.
- read_file(path): Read a workspace file (uploads, research findings, etc.).
- done(reason): Signal that orchestration is complete.

You do NOT have browsing tools (open, find, click, interact). All browsing goes
through research() subagents or spawned yielders/synthesizers.

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
        # V6: filter callback for SeedProcessor
        on_filter: Optional[Callable] = None,
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
        self.on_filter = on_filter
        self.yielder_model = yielder_model or model

        # V6 state
        self._is_done = False
        self._research_counter = 0
        self._yielder_counter = 0
        self._seed_processor = seed_processor
        self._generation_stats = generation_stats

        # Pipeline config built incrementally
        self._template: Optional[str] = None
        self._variables: List[VariableConfig] = []
        self._filters: List[FilterConfig] = []
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
        """Register V6 orchestrator tools."""

        # --- Research tools (brave_search, code_exec + web_search_preview) ---
        from dsl_worker.infra.research_tools import ResearchTools, ResearchScope

        self._impl = ResearchTools(
            workspace_dir=self.workspace_dir,
            schema=[],
            brave_api_key=self.brave_api_key,
            openai_client=self.openai_client,
            model=self.model,
            sandbox=self.sandbox,
            stop_checker=self.stop_checker,
            blob_service_client=self.blob_service_client,
            project_id=self.project_id,
            uploaded_file_urls=self.uploaded_file_urls,
            on_browser_started=self.on_browser_started,
        )
        self._impl.set_scope(ResearchScope(
            id="orchestrator",
            description="",
            quota=0,
        ))
        # Only give orchestrator quick-check tools. Deep browsing (open, find,
        # click, interact) should go through research() subagents — otherwise
        # the orchestrator falls into rabbit holes post-research.
        self._impl.register_on(registry, exclude=[
            "open", "find", "click", "interact", "list_files", "shell_exec",
        ])

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
            description="Read a file from the workspace (uploads, downloads, etc.).",
            parameters={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "File path (relative to workspace)",
                    },
                },
                "required": ["path"],
            },
            handler=read_file,
        )

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
                variables = [VariableConfig(**v) for v in variables_raw]
            except (TypeError, KeyError) as e:
                return f"Error parsing variables: {e}", 0.0

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
                f"You can now submit_seeds, spawn_yielder, or spawn_synthesizer."
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

        # --- submit_seeds (V6) ---
        async def submit_seeds(args: Dict) -> tuple[str, float]:
            """Submit preset seed values directly."""
            seeds = args.get("seeds", [])
            if not self._template:
                return "Error: call write_template first", 0.0
            if not seeds:
                return "Error: seeds array is empty", 0.0

            accepted = 0
            rejected_reasons = []
            for seed_values in seeds:
                seed = Seed(values=seed_values, metadata={"source": "orchestrator_preset"})
                status = await self._seed_processor.submit_seed(seed)
                if status["accepted"]:
                    accepted += 1
                else:
                    rejected_reasons.append(status.get("reason", "unknown"))

            stats = self._seed_processor.stats
            result = (
                f"{accepted}/{len(seeds)} seeds accepted. "
                f"Pipeline: {stats['accepted']} total accepted, "
                f"{stats['remaining']} remaining."
            )
            if rejected_reasons:
                from collections import Counter
                counts = Counter(rejected_reasons)
                result += f" Rejections: {dict(counts)}"
            return result, 0.0

        registry.add(
            name="submit_seeds",
            description=(
                "Submit preset seed values directly. Each entry is a dict of "
                "variable_name → value. Requires write_template() to be called first."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "seeds": {
                        "type": "array",
                        "description": (
                            "Array of seed value dicts. Each dict maps variable names "
                            "to their values for one row."
                        ),
                        "items": {"type": "object"},
                    },
                },
                "required": ["seeds"],
            },
            handler=submit_seeds,
        )

        # --- spawn_yielder (V6) ---
        async def spawn_yielder(args: Dict) -> tuple[str, float]:
            """Launch an iterative seed yielder. Blocks until it finishes."""
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
                filters=self._filters,
                dedup=self._dedup,
                target_rows=quota,
                research_context=self._research_context,
            )

            idx = self._yielder_counter
            self._yielder_counter += 1

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
                await self.on_cost(yielder.cost_usd, f"yielder:{idx}")

            stats = self._seed_processor.stats
            return (
                f"Yielder {idx} finished: {result.turns_taken} turns, "
                f"cost=${yielder.cost_usd:.3f}. "
                f"Pipeline: {stats['accepted']} accepted, "
                f"{stats['remaining']} remaining."
            ), yielder.cost_usd

        registry.add(
            name="spawn_yielder",
            description=(
                "Launch an iterative seed yielder that browses/iterates sources and "
                "yields seeds. Blocks until it finishes. Call multiple in parallel for "
                "different source partitions."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "sources": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "URLs, file paths, or search queries to iterate. "
                            "The yielder will browse these and yield seeds."
                        ),
                    },
                    "quota": {
                        "type": "integer",
                        "description": (
                            "How many accepted seeds this yielder should aim for "
                            "(default 20). Set based on how many items you expect "
                            "the sources to contain."
                        ),
                    },
                    "instructions": {
                        "type": "string",
                        "description": (
                            "Specific instructions for this yielder "
                            "(e.g., 'Extract job title and URL from each listing')"
                        ),
                    },
                },
                "required": ["sources"],
            },
            handler=spawn_yielder,
        )

        # --- spawn_synthesizer (V6) ---
        async def spawn_synthesizer(args: Dict) -> tuple[str, float]:
            """Launch a synthetic seed generator. Blocks until it finishes."""
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
                filters=self._filters,
                dedup=self._dedup,
                target_rows=quota,
                research_context=effective_context,
            )

            idx = self._yielder_counter
            self._yielder_counter += 1

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
                await self.on_cost(yielder.cost_usd, f"synthesizer:{idx}")

            stats = self._seed_processor.stats
            return (
                f"Synthesizer {idx} finished: {result.turns_taken} turns, "
                f"cost=${yielder.cost_usd:.3f}. "
                f"Pipeline: {stats['accepted']} accepted, "
                f"{stats['remaining']} remaining."
            ), yielder.cost_usd

        registry.add(
            name="spawn_synthesizer",
            description=(
                "Launch a synthetic seed generator that researches and discovers seeds. "
                "Blocks until it finishes. Use when seeds need to be discovered through "
                "research, not just iterated from a known source."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "topic": {
                        "type": "string",
                        "description": (
                            "Topic area for the synthesizer to research and generate "
                            "seeds from (e.g., 'DayZ survival mechanics', "
                            "'AI companies in Minnesota')"
                        ),
                    },
                    "quota": {
                        "type": "integer",
                        "description": "How many accepted seeds to aim for (default 20)",
                    },
                    "instructions": {
                        "type": "string",
                        "description": "Specific instructions for this synthesizer",
                    },
                    "research_context": {
                        "type": "string",
                        "description": (
                            "Research findings to seed the synthesizer with. "
                            "If not provided, uses accumulated research context."
                        ),
                    },
                },
                "required": ["topic"],
            },
            handler=spawn_synthesizer,
        )

        # --- set_filter (V6) ---
        async def set_filter(args: Dict) -> tuple[str, float]:
            """Add a seed validation filter."""
            name = args.get("name", "")
            description = args.get("description", "")
            complexity = args.get("complexity", "simple")

            if not name or not description:
                return "Error: name and description are required", 0.0

            filter_config = FilterConfig(
                name=name,
                description=description,
                complexity=complexity,
            )
            self._filters.append(filter_config)
            self._seed_processor.set_filters(self._filters)
            return f"Filter '{name}' added ({complexity}). {len(self._filters)} total.", 0.0

        registry.add(
            name="set_filter",
            description=(
                "Add a seed validation filter. Seeds will be checked against this "
                "criteria before being accepted. Use 'simple' for quick classification, "
                "'judgment' for multi-turn research-based validation."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Filter name"},
                    "description": {
                        "type": "string",
                        "description": "What to check (e.g., 'Job posting must be from the last 30 days')",
                    },
                    "complexity": {
                        "type": "string",
                        "enum": ["simple", "judgment"],
                        "description": "'simple' for single-turn, 'judgment' for multi-turn with research",
                    },
                },
                "required": ["name", "description"],
            },
            handler=set_filter,
        )

        # --- set_dedup (V6) ---
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
                "Call before submitting seeds or spawning yielders."
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

        # --- get_status (V6) ---
        async def get_status(args: Dict) -> tuple[str, float]:
            """Check current pipeline progress."""
            stats = self._seed_processor.stats
            gen = self._generation_stats
            return (
                f"Seeds: {stats['accepted']} accepted, {stats['remaining']} remaining, "
                f"{stats['rejected_dedup']} dedup rejected, "
                f"{stats['rejected_filter']} filter rejected, "
                f"{stats['submitted_total']} total submitted.\n"
                f"Rows: {gen.get('rows_generated', 0)} generated, "
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
            description="Signal that orchestration is complete. Row generation continues in the background.",
            parameters={
                "type": "object",
                "properties": {
                    "reason": {
                        "type": "string",
                        "description": "Why orchestration is done",
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
                "filters": [
                    {"name": f.name, "description": f.description,
                     "complexity": f.complexity}
                    for f in self._filters
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
        """Clean up browser, sandbox, and other resources."""
        try:
            await self._impl.cleanup()
        except Exception as e:
            logger.warning(f"Orchestrator cleanup error: {e}")
