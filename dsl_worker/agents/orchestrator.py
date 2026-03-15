"""
Orchestrator agent — coordinates dataset generation.

V8: Simplified pipeline. No template variables, no seed-level dedup.
The orchestrator does recon, sets row-generator instructions, declares
identity columns for row-level dedup, then harvests in rounds.

Tools:
- research(question, ...) — spawn research subagent
- set_instructions(instructions, candidate_description) — set row generator instructions
- set_identity_columns(columns) — declare which columns identify an entity (for dedup)
- harvest(source, instructions, quota) — crawl + extract candidates (returns status)
- done() — signal completion
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from dsl_worker.agents.base import AgentConversation, AgentResult
from dsl_worker.agents.tools import ToolRegistry
from dsl_worker.billing.tracked_client import TrackedOpenAIClient
from dsl_worker.infra.pipeline import SeedProcessor

logger = logging.getLogger(__name__)


ORCHESTRATOR_SYSTEM_PROMPT = """\
You are the orchestrator for a dataset generation system. A user described a dataset \
they want. Your job is to figure out how to produce it and coordinate execution.

You stay in the loop throughout — dispatch research, write row instructions, launch \
harvesters in rounds, and adapt based on results.

## Your Subagents

**research(question, ...)** — Recon agents that investigate specific questions. \
They can browse the web, search, run code, and read files. They return findings. \
Use these to understand the landscape before writing instructions.

**harvest(source, instructions, quota)** — Crawl a source and extract candidates. \
A smart crawler navigates pages and dumps them; a cheap extractor pulls out \
candidate items. Each candidate becomes one row. Use for both known sources \
(URLs, files) and discovery (search queries, topic exploration).

## Workflow

1. **Reason** — Which pattern? What unknowns need resolving?

2. **Research** (if needed) — Dispatch research() agents for specific questions.
   - Ask focused questions: "What fields are on an Upwork job listing page?"
   - Call multiple in parallel for different questions.
   - Once findings come back, MOVE ON. Don't over-research.

3. **Set instructions** — Call set_instructions() with row generation instructions \
and a candidate_description. Instructions tell row generators how to process each \
candidate. candidate_description tells the extractor what items to look for.

4. **Set identity columns** — Call set_identity_columns() to declare which output \
columns uniquely identify an entity. Row generators will check these for duplicates \
automatically. E.g., ["url"] for job listings, ["name", "email"] for contacts.

5. **Harvest (in rounds)** — Launch harvest() calls:
   - Each harvest() = one crawler = one slice of the problem.
   - A slice is a single URL, search query, file, or topic.
   - Launch up to 10 crawlers total (hard limit). Prefer more slices over fewer \
     for better coverage — don't hesitate to launch 8–10 for broad searches.
   - Each harvest() returns candidate and row counts.
   - After a round completes, assess: enough rows? Any gaps? \
     Launch another round targeting underrepresented areas if needed.

6. **Done** — Call done() when enough seeds have been dispatched.

## Common Patterns

**Extraction** — rows from real sources (job listings, products, directories)
  Quick recon to understand the site structure, then harvest() on listing URLs.

**Enrichment** — rows already exist (CSV upload), need additional columns
  Read the file via research(), then harvest() to iterate its rows.

**Discovery** — find entities matching criteria (podcasts, companies, churches)
  Research to find sources and subcategories, then parallel harvest() calls \
  across platforms and topics. Check results and add more slices if thin.

**Qualification** — given a list, filter then enrich
  harvest() the list file; row generators visit each entity and skip_row() \
  if it doesn't qualify.

## Principles

- ONE harvest() = ONE slice. Each spawns its own crawler with its own browser.
- harvest() blocks until done, then returns status. Launch multiple in parallel.
- After each round: assess row counts, then either launch more or done().
- Row generators see the full user conversation — your instructions are additive. \
  Keep them focused: what to look for, how to find it, what to skip.
- Our browsing stack handles anti-bot, CAPTCHAs, and JS-heavy pages automatically.
- Row generators will skip_row() for dead ends. Expect some rejection — that's normal.
- harvest() is fast and cheap — don't hesitate to launch many slices (up to 10 total).
- harvest() instructions tell the crawler HOW to navigate: which links to follow, \
  when to stop, how to recognize the end of a source. Express stopping criteria as \
  observable content signals (e.g. "stop when listings are older than 7 days") not \
  UI assumptions (never assume specific filter buttons or controls exist).

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
- set_instructions(instructions, candidate_description): Set row generator instructions. \
  instructions: plain text — what to do with each candidate, what columns to fill, \
  how to research. No {{variable}} placeholders needed. \
  candidate_description: what a candidate looks like (tells the extractor what to find).
- set_identity_columns(columns): Declare which output columns identify an entity. \
  When a row generator sets one of these columns, it gets back similar existing rows \
  and can skip_row() if it's a duplicate. E.g., ["url"] or ["name", "email"].
- harvest(source, instructions): Crawl one slice, extract candidates. \
  Returns rows generated so far, rows still needed, and cost.
- done(reason): Signal completion — all seed sources dispatched.

You do NOT have browsing, search, code execution, or file reading tools. \
All investigation goes through research() subagents.

{feedback_section}
"""


class OrchestratorAgent:
    """
    V8 Orchestrator. Simplified: set_instructions replaces write_template,
    set_identity_columns replaces set_dedup, no get_status (status comes
    from harvest() responses).
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
        stop_event: Optional[asyncio.Event] = None,
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
        on_browser_stopped: Optional[Callable] = None,
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
        self.stop_event = stop_event
        self.cost_checker = cost_checker
        self.blob_service_client = blob_service_client
        self.project_id = project_id
        self.on_tool_call = on_tool_call
        self.on_cost = on_cost
        self.uploaded_file_urls = uploaded_file_urls
        self.mcp_tools = mcp_tools or []
        self.on_browser_started = on_browser_started
        self.on_browser_stopped = on_browser_stopped
        self.langfuse_parent = langfuse_parent
        self.yielder_model = yielder_model or model

        self._is_done = False
        self._research_counter = 0
        self._subagent_counter = 0
        self._seed_processor = seed_processor
        self._generation_stats = generation_stats

        registry = ToolRegistry()
        self._register_tools(registry)

        columns_desc = json.dumps(columns, indent=2) if columns else "(no columns defined)"
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

        from dsl_worker.config import settings
        max_turns = getattr(settings, 'orchestrator_max_turns', 40)
        soft_limit = getattr(settings, 'orchestrator_soft_limit', 25)

        self._conversation = AgentConversation(
            openai_client=openai_client,
            model=model,
            system_prompt=system_prompt,
            tools=registry,
            stop_checker=stop_checker,
            stop_event=stop_event,
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
        if not uploaded_files:
            return "No uploaded files."
        lines = ["Uploaded files:"]
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

        # --- research ---
        async def research(args: Dict) -> tuple[str, float]:
            from dsl_worker.agents.research import ResearchAgent

            question = args.get("question", "")
            scope = args.get("scope", "")
            budget = args.get("budget", 10)
            output_format = args.get("output_format", "")

            if not question:
                return "Error: question is required", 0.0

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
                on_browser_stopped=self.on_browser_stopped,
            )
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

            if self.on_cost and result.cost_usd > 0:
                await self.on_cost(result.cost_usd, "research_subagent")

            n = self._research_counter
            self._research_counter += 1
            research_dir = self.workspace_dir / "research"
            research_dir.mkdir(exist_ok=True)
            try:
                (research_dir / f"finding_{n}.md").write_text(
                    f"# Research: {question}\n\n{result.text}", encoding="utf-8"
                )
            except Exception as e:
                logger.warning(f"Failed to save research finding: {e}")

            return f"[Saved to research/finding_{n}.md]\n\n{result.text}", result.cost_usd

        registry.add(
            name="research",
            description=(
                "Spawn a research subagent to explore a question. Returns findings. "
                "Call multiple in one response for parallel research."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "question": {"type": "string", "description": "Specific research question"},
                    "scope": {"type": "string", "description": "Focus area for research"},
                    "budget": {
                        "type": "integer",
                        "description": "Max tool calls (default 10). 5 for quick, 15-20 for deep.",
                    },
                    "output_format": {
                        "type": "string",
                        "description": "Expected format for the answer",
                    },
                },
                "required": ["question"],
            },
            handler=research,
        )

        # --- set_instructions ---
        async def set_instructions(args: Dict) -> tuple[str, float]:
            instructions = args.get("instructions", "")
            candidate_description = args.get("candidate_description", "")

            if not instructions:
                return "Error: instructions is required", 0.0

            self._seed_processor.set_instructions(instructions, candidate_description)

            return (
                f"Instructions set. Candidate description: \"{candidate_description}\". "
                f"You can now harvest()."
            ), 0.0

        registry.add(
            name="set_instructions",
            description=(
                "Set row generator instructions. Plain text — what to do with each candidate, "
                "what columns to fill, how to research. Also sets candidate_description which "
                "tells the extractor what items to look for on crawled pages."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "instructions": {
                        "type": "string",
                        "description": (
                            "Instructions for row generators. Plain text, no {variable} placeholders. "
                            "E.g.: 'You will be given a podcast. Find the host name, description, "
                            "and contact email. Visit the podcast website if available.'"
                        ),
                    },
                    "candidate_description": {
                        "type": "string",
                        "description": (
                            "What a candidate looks like — tells the extractor what to find on pages. "
                            "E.g.: 'podcast entries with name and URL', 'church listings with name and address'"
                        ),
                    },
                },
                "required": ["instructions"],
            },
            handler=set_instructions,
        )

        # --- set_identity_columns ---
        async def set_identity_columns(args: Dict) -> tuple[str, float]:
            columns = args.get("columns", [])
            if not isinstance(columns, list):
                return "Error: columns must be a list of column names", 0.0

            self._seed_processor.set_identity_columns(columns)
            return (
                f"Identity columns set: {columns}. Row generators will check these "
                f"for duplicates when filling them."
            ), 0.0

        registry.add(
            name="set_identity_columns",
            description=(
                "Declare which output columns uniquely identify an entity. When a row "
                "generator sets one of these columns, it receives similar existing rows "
                "and can skip_row() if it's a duplicate. E.g., ['url'] for job listings, "
                "['name', 'email'] for contacts, ['podcast_name'] for podcasts."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "columns": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Column names that identify an entity",
                    },
                },
                "required": ["columns"],
            },
            handler=set_identity_columns,
        )

        MAX_CONCURRENT_CRAWLERS = 10
        crawler_semaphore = asyncio.Semaphore(MAX_CONCURRENT_CRAWLERS)

        # --- harvest ---
        async def harvest(args: Dict) -> tuple[str, float]:
            from dsl_worker.agents.crawler import CrawlerAgent
            from dsl_worker.agents.extractor import CandidateExtractor
            from dsl_worker.config import settings as worker_settings

            source = args.get("source", "")
            instructions = args.get("instructions", "")

            if not source:
                return "Error: source is required", 0.0

            if not self._seed_processor._instructions:
                return "Error: call set_instructions first", 0.0

            idx = self._subagent_counter
            self._subagent_counter += 1

            async with crawler_semaphore:
                langfuse_span = getattr(self._conversation, "_current_langfuse_span", None)

                candidate_description = self._seed_processor._candidate_description

                extractor = CandidateExtractor(
                    openai_client=self.openai_client,
                    model=worker_settings.extractor_model,
                    candidate_description=candidate_description,
                    on_submit=self._seed_processor.submit_seed,
                    on_cost=self.on_cost,
                )
                extract_semaphore = asyncio.Semaphore(20)
                extraction_tasks: List[asyncio.Task] = []
                pages_dumped = 0

                async def on_dump_page(page: Dict):
                    nonlocal pages_dumped
                    pages_dumped += 1
                    task = asyncio.create_task(
                        extractor.extract_page(page, extract_semaphore)
                    )
                    extraction_tasks.append(task)

                crawler = CrawlerAgent(
                    sources=[source],
                    instructions=instructions,
                    candidate_description=candidate_description,
                    openai_client=self.openai_client,
                    model=self.yielder_model,
                    workspace_dir=self.workspace_dir,
                    on_dump_page=on_dump_page,
                    crawler_index=idx,
                    research_context=self._seed_processor._research_context,
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
                    on_browser_stopped=self.on_browser_stopped,
                )

                t0 = time.time()
                try:
                    await crawler.run()
                finally:
                    await crawler.cleanup()

                crawl_time = time.time() - t0
                crawl_cost = crawler.cost_usd

                if not extraction_tasks:
                    stats = self._seed_processor.stats
                    gen = self._generation_stats
                    return (
                        f"Harvester {idx}: no pages found in {crawl_time:.0f}s. "
                        f"Cost: ${crawl_cost:.3f}. "
                        f"Pipeline: {stats['accepted']} candidates accepted, "
                        f"{gen.get('rows_generated', 0)} rows generated."
                    ), crawl_cost

                results = await asyncio.gather(*extraction_tasks, return_exceptions=True)

                candidates_found = 0
                extract_cost = 0.0
                for r in results:
                    if isinstance(r, BaseException):
                        continue
                    count, cost = r
                    candidates_found += count
                    extract_cost += cost

                total_cost = crawl_cost + extract_cost
                stats = self._seed_processor.stats
                gen = self._generation_stats

                return (
                    f"Harvester {idx} done: {pages_dumped} pages in {crawl_time:.0f}s, "
                    f"{candidates_found} candidates extracted. "
                    f"Pipeline total: {stats['accepted']} accepted, "
                    f"{stats['remaining']} rows still needed, "
                    f"{gen.get('rows_generated', 0)} rows generated, "
                    f"{gen.get('skipped', 0)} skipped. "
                    f"Cost: ${total_cost:.3f} (crawl=${crawl_cost:.3f}, extract=${extract_cost:.3f})."
                ), total_cost

        registry.add(
            name="harvest",
            description=(
                "Crawl one source slice and extract candidates. Blocks until done, "
                "then returns row counts and cost. Call multiple in parallel for different slices. "
                f"Maximum {MAX_CONCURRENT_CRAWLERS} concurrent harvest() calls."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "source": {
                        "type": "string",
                        "description": (
                            "One source to crawl: a URL, file path, search query, or topic. "
                            "For multiple sources, call harvest() multiple times in parallel."
                        ),
                    },
                    "instructions": {
                        "type": "string",
                        "description": (
                            "Navigation instructions for the crawler: which links to follow, "
                            "how to paginate, when to stop. Use observable content signals for "
                            "stopping (e.g. 'stop when you see listings older than 7 days'). "
                            "Do not assume specific UI controls exist."
                        ),
                    },
                },
                "required": ["source"],
            },
            handler=harvest,
        )

        # --- done ---
        async def done(args: Dict) -> tuple[str, float]:
            reason = args.get("reason", "complete")
            self._is_done = True
            self._save_recipe()
            return f"Orchestrator done: {reason}", 0.0

        registry.add(
            name="done",
            description=(
                "Signal orchestration is complete — all seed sources dispatched. "
                "Row generation continues in background."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "reason": {"type": "string", "description": "Why orchestration is done"},
                },
            },
            handler=done,
        )

    def _save_recipe(self) -> None:
        try:
            recipe = {
                "instructions": self._seed_processor._instructions,
                "candidate_description": self._seed_processor._candidate_description,
                "identity_columns": self._seed_processor._identity_columns,
                "research_context": self._seed_processor._research_context,
                "target_rows": self.num_samples,
            }
            (self.workspace_dir / "pipeline_recipe.json").write_text(
                json.dumps(recipe, indent=2), encoding="utf-8"
            )
        except Exception as e:
            logger.warning(f"Failed to save recipe: {e}")

    async def run(self) -> AgentResult:
        if self.feedback_context:
            message = (
                "Begin. The user reviewed previous results and gave feedback "
                "(shown in system prompt). Research as needed and design a new pipeline."
            )
        else:
            message = (
                "Begin. Read the conversation history and resources, reason about "
                "strategy, then set instructions and harvest candidates."
            )

        return await self._conversation.send(
            message,
            exit_condition=lambda: self._is_done,
        )

    @property
    def cost_usd(self) -> float:
        return self._conversation.total_cost

    async def cleanup(self) -> None:
        pass
