"""
Orchestrator agent — coordinates dataset generation.

V11: Orchestrator-driven batch pipeline. The orchestrator is the sole
decision-maker — no background consumers, no Thompson Sampling, no events.

- start_harvest() creates a harvester and runs its first batch
- process() processes candidates into rows (auto-batches if buffer empty)
- close_harvest() stops a harvester and frees its browser session
- The orchestrator sees real cost breakdowns and makes all scaling decisions
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional

from dsl_worker.agents.base import AgentConversation, AgentResult
from dsl_worker.agents.tools import ToolRegistry
from dsl_worker.billing.tracked_client import TrackedOpenAIClient
from dsl_worker.infra.candidate_pool import Candidate

logger = logging.getLogger(__name__)


# ── HarvesterState ────────────────────────────────────────────────────

@dataclass
class HarvesterState:
    """Tracks a harvester and its candidates."""
    id: str                           # "harvest:0"
    source: str
    description: str
    agent: Any                        # HarvesterAgent (forward ref)
    candidates: List[Candidate] = field(default_factory=list)
    total_harvested: int = 0
    total_processed: int = 0
    rows_produced: int = 0
    duplicates: int = 0
    skipped: int = 0
    errors: int = 0
    harvest_cost: float = 0.0
    process_cost: float = 0.0
    batches: int = 0
    exhausted: bool = False
    last_report: str = ""


# ── System Prompt ─────────────────────────────────────────────────────

ORCHESTRATOR_SYSTEM_PROMPT = """\
# Dataset Generation Orchestrator

You are the strategist in a dataset generation pipeline. You coordinate \
harvesters (which collect candidates from sources) and row generators \
(which turn candidates into verified dataset rows). You make all the decisions \
— what sources to tap, how fast to scale, when to pivot.

## How the Pipeline Works

1. You create harvesters pointed at sources (URLs, files, search queries).
2. You call process() which harvests a batch of candidates and kicks off \
row generators to turn them into dataset rows.
3. Row generators run in the background. You'll see results as background \
updates on your next tool call — you don't have to wait.
4. You keep going — creating more harvesters, processing more batches — \
until the target is reached.

## Your Tools

**explore_agent(task)** — Inspect uploaded files, data, schemas via code execution. \
No web access.

**web_search(task)** — Send a task to a browser agent that can search the web, \
navigate pages, and extract information. Write as a clear instruction, not keywords. \
The browser may come pre-authenticated to some sites via cookies.

**create_harvester(source, candidate_description)** — Set up a harvester for a source. \
The candidate_description is critical — it tells the harvester exactly what a \
good candidate looks like, what data to extract, and what to skip at a glance. \
Write it as a clear, complete brief. Returns a harvester_id.

**process(harvester_id, max_count)** — Harvest a batch of candidates and start \
processing them into rows. Returns the harvest report immediately. Row generation \
results appear as background updates on subsequent tool calls.

**close_harvest(harvester_id)** — Close a harvester and free its browser session.

## Writing Good Candidate Descriptions

The candidate_description on create_harvester is the most important thing you write. \
It's the spec that harvesters use to decide what to grab and what to skip. Include:
- What a good candidate IS (the entity, the context)
- What data to extract per candidate (all visible fields you'd want)
- Any dealbreakers that are obvious from the surface (date range, status, type)

Example: "An open Upwork job posting from the last 7 days where the client wants \
a tabular dataset delivered (CSV, spreadsheet, structured list). Include: job title, \
URL, full description, budget/hourly rate, posted date, client location, and skills. \
Skip any posting that is clearly not about data/dataset work based on the title."

The harvester will grab anything that looks like a match and skip only what is \
obviously wrong. When in doubt, it produces the candidate — validation happens \
downstream in row generators.

## Strategy

- If you know the source, go straight to create_harvester + process. \
Don't research first unless you genuinely don't know where the data lives.
- web_search is for when you're stuck — you don't know what sources exist, or \
harvesters keep failing and you need to figure out why.
- Ramp up gradually: start with process(max_count=3-5) to validate, then \
increase as confidence grows. But stay aware of how many rows you still need — \
don't massively overshoot.
- Dedup is cheap — overlap between sources is fine.
- Close sources that aren't producing.
- You can call multiple tools in parallel (e.g., create multiple harvesters, or \
process from one while creating another).
- The browser is highly capable — it handles JavaScript, anti-bot, CAPTCHAs, \
and dynamic pages. JS is almost never the issue if something isn't loading.

Today's date: {current_date}

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

{feedback_section}
"""


class OrchestratorAgent:
    """
    V11 Orchestrator. Drives the full pipeline directly — no background
    consumers, no Thompson Sampling, no event system.
    """

    def __init__(
        self,
        chat_history: List[Dict[str, str]],
        columns: List[Dict[str, Any]],
        num_samples: int,
        openai_client: TrackedOpenAIClient,
        model: str,
        workspace_dir: Path,
        generation_stats: Dict[str, Any],
        dedup_store: Any,  # DedupStore from row.py
        save_row: Callable[..., Awaitable[Optional[str]]],
        uploaded_files: Optional[List[Dict[str, Any]]] = None,
        bu_client: Optional[Any] = None,
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
        harvester_model: str = "",
        generation_model: str = "",
    ) -> None:
        self.feedback_context = feedback_context
        self.chat_history = chat_history
        self.columns = columns
        self.num_samples = num_samples
        self.workspace_dir = Path(workspace_dir)
        self.openai_client = openai_client
        self.model = model
        self.bu_client = bu_client
        self.sandbox = sandbox
        self.stop_checker = stop_checker
        self.stop_event = stop_event
        self.cost_checker = cost_checker
        self.blob_service_client = blob_service_client
        self.project_id = project_id
        self.on_tool_call = on_tool_call
        self.on_cost = on_cost
        self.uploaded_file_urls = uploaded_file_urls
        self.uploaded_files = uploaded_files
        self.mcp_tools = mcp_tools or []
        self.harvester_model = harvester_model or model
        self.generation_model = generation_model or "gpt-5-mini"

        self._generation_stats = generation_stats
        self._save_row = save_row
        self._dedup_store = dedup_store
        self._save_lock = asyncio.Lock()

        self._research_counter = 0
        self._harvester_counter = 0
        self._harvesters: Dict[str, HarvesterState] = {}
        self._background_tasks: List[asyncio.Task] = []
        self._pending_events: List[str] = []

        from dsl_worker.config import settings
        self._generation_semaphore = asyncio.Semaphore(
            settings.generation_parallel_samples
        )

        registry = ToolRegistry()
        self._register_tools(registry)

        columns_desc = json.dumps(columns, indent=2) if columns else "(no columns defined)"
        convo_summary = self._format_conversation()
        resources_section = self._format_resources(uploaded_files)
        feedback_section = self._format_feedback()

        from datetime import date
        system_prompt = ORCHESTRATOR_SYSTEM_PROMPT.format(
            num_samples=num_samples,
            columns_description=columns_desc,
            conversation_summary=convo_summary,
            resources_section=resources_section,
            feedback_section=feedback_section,
            current_date=date.today().isoformat(),
        )

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
            drain_events=self.drain_pending_events,
        )

    # ── Formatting helpers ────────────────────────────────────────────

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
        lines = ["Uploaded files (accessible in harvester via code_exec):"]
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
            lines.append(f"  - /workspace/uploads/{name} ({ctype}, {size_str})")
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

    def _format_harvester_status(self) -> str:
        """Format status of all harvesters for tool output."""
        rows_done = self._generation_stats.get("rows_generated", 0)
        lines = [f"\nOverall: {rows_done}/{self.num_samples} rows generated."]

        if not self._harvesters:
            return "\n".join(lines)

        lines.append("\nHarvesters:")
        for hid, state in self._harvesters.items():
            fertility = (
                f"{state.rows_produced / state.total_processed:.0%}"
                if state.total_processed > 0 else "N/A"
            )
            total_cost = state.harvest_cost + state.process_cost
            cpr = (
                f"${total_cost / state.rows_produced:.3f}"
                if state.rows_produced > 0 else "N/A"
            )
            status = "exhausted" if state.exhausted else f"{len(state.candidates)} buffered"
            lines.append(
                f"  {hid} ({state.source[:60]}): "
                f"{fertility} fertility, {cpr}/row, "
                f"{state.rows_produced} rows from {state.total_processed} processed, "
                f"harvest ${state.harvest_cost:.3f}, {status}"
            )

        return "\n".join(lines)

    # ── Tool registration ─────────────────────────────────────────────

    def _register_tools(self, registry: ToolRegistry) -> None:

        # --- web_search (direct BU call, no LLM wrapper) ---
        async def web_search(args: Dict) -> tuple[str, float]:
            task = args.get("task", "")
            if not task:
                return "Error: task is required", 0.0

            if not self.bu_client:
                return "Error: web search not available (no BU client)", 0.0

            fast_task = (
                f"{task}\n\n"
                "Be fast and direct. Take the shortest path to the answer. "
                "Do not explore or enumerate page elements — just do what's needed and return."
            )
            text, cost, _sid = await self.bu_client.research(fast_task)

            if self.on_cost and cost > 0:
                await self.on_cost(cost, "web_search")

            if len(text) > 6000:
                text = text[:6000] + "\n\n[Truncated]"

            return text or "(no results)", cost

        registry.add(
            name="web_search",
            description=(
                "Send a task to a browser agent that can search the web, navigate "
                "pages, and extract information. Write the task as a full natural "
                "language instruction — NOT a search query or keywords."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "task": {
                        "type": "string",
                        "description": (
                            "A clear instruction for the browser agent. "
                            "Example: 'Go to upwork.com/nx/search/jobs and figure out "
                            "the URL parameters for filtering by recency and date posted. "
                            "Return the full URL with the correct filters applied.'"
                        ),
                    },
                },
                "required": ["task"],
            },
            handler=web_search,
        )

        # --- explore_agent (unchanged) ---
        async def explore_agent(args: Dict) -> tuple[str, float]:
            from dsl_worker.agents.code_exec import CodeExecAgent
            from dsl_worker.config import settings as worker_settings

            task = args.get("task", "")
            budget = args.get("budget", 6)

            if not task:
                return "Error: task is required", 0.0

            agent = CodeExecAgent(
                openai_client=self.openai_client,
                model=worker_settings.research_subagent_model,
                workspace_dir=self.workspace_dir,
                sandbox=self.sandbox,
                stop_checker=self.stop_checker,
                max_turns=budget,
                tool_budget=budget,
                blob_service_client=self.blob_service_client,
                project_id=self.project_id,
                uploaded_file_urls=self.uploaded_file_urls,
            )

            try:
                result = await agent.ask_full(task)
            finally:
                await agent.cleanup()

            if self.on_cost and result.cost_usd > 0:
                await self.on_cost(result.cost_usd, "explore_agent")

            n = self._research_counter
            self._research_counter += 1
            research_dir = self.workspace_dir / "research"
            research_dir.mkdir(exist_ok=True)
            try:
                (research_dir / f"finding_{n}.md").write_text(
                    f"# Explore: {task}\n\n{result.text}", encoding="utf-8"
                )
            except Exception as e:
                logger.warning(f"Failed to save finding: {e}")

            return f"[Saved to research/finding_{n}.md]\n\n{result.text}", 0.0

        registry.add(
            name="explore_agent",
            description=(
                "Inspect files, data, and connected resources via code execution. "
                "No web access. Use this to understand uploaded files, schemas, etc."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "task": {"type": "string", "description": "What to inspect or analyze"},
                    "budget": {
                        "type": "integer",
                        "description": "Max tool calls (default 6). 3 for quick, 10 for deep.",
                    },
                },
                "required": ["task"],
            },
            handler=explore_agent,
        )

        # --- create_harvester ---
        async def create_harvester(args: Dict) -> tuple[str, float]:
            source = args.get("source", "")
            candidate_description = args.get("candidate_description", "")

            if not source:
                return "Error: source is required", 0.0
            if not candidate_description:
                return "Error: candidate_description is required — the harvester needs to know what to look for.", 0.0

            from dsl_worker.agents.harvester import HarvesterAgent

            idx = self._harvester_counter
            self._harvester_counter += 1
            harvester_id = f"harvest:{idx}"

            harvester = HarvesterAgent(
                source=source,
                description=candidate_description,
                source_id=harvester_id,
                openai_client=self.openai_client,
                model=self.harvester_model,
                workspace_dir=self.workspace_dir,
                bu_client=self.bu_client,
                harvester_index=idx,
                sandbox=self.sandbox,
                stop_checker=self.stop_checker,
                stop_event=self.stop_event,
                blob_service_client=self.blob_service_client,
                project_id=self.project_id,
                on_tool_call=self.on_tool_call,
                on_cost=self.on_cost,
                uploaded_file_urls=self.uploaded_file_urls,
                uploaded_files=self.uploaded_files,
                mcp_tools=self.mcp_tools,
            )

            state = HarvesterState(
                id=harvester_id,
                source=source,
                description=candidate_description,
                agent=harvester,
            )
            self._harvesters[harvester_id] = state

            return (
                f"Harvester {harvester_id} created for: {source}\n"
                f"Call process(harvester_id='{harvester_id}', max_count=N) to start."
            ), 0.0

        registry.add(
            name="create_harvester",
            description=(
                "Set up a harvester for a source. Returns a harvester_id. "
                "No harvesting happens yet — call process() to start. "
                "The candidate_description is the brief for the harvester — "
                "what a good candidate looks like, what to extract, what to skip."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "source": {
                        "type": "string",
                        "description": "Source to harvest: URL, file path, search query, or topic.",
                    },
                    "candidate_description": {
                        "type": "string",
                        "description": (
                            "What candidates to look for. Describe the entity, what data to extract, "
                            "and any obvious surface-level dealbreakers. See system prompt for examples."
                        ),
                    },
                },
                "required": ["source", "candidate_description"],
            },
            handler=create_harvester,
        )

        # --- process ---
        async def process(args: Dict) -> tuple[str, float]:
            harvester_id = args.get("harvester_id", "")
            max_count = args.get("max_count", 10)

            state = self._harvesters.get(harvester_id)
            if not state:
                available = list(self._harvesters.keys())
                return f"Error: unknown harvester '{harvester_id}'. Available: {available}", 0.0

            batch_report = None
            harvest_cost = 0.0

            # Auto-fetch if buffer empty and not exhausted
            if not state.candidates and not state.exhausted:
                is_first = state.batches == 0
                message = (
                    "Begin harvesting candidates from your assigned source."
                    if is_first else
                    "Get the next batch of candidates."
                )
                candidates, report = await state.agent.run_batch(message)
                cost_delta = state.agent.batch_cost_delta
                state.candidates.extend(candidates)
                state.total_harvested += len(candidates)
                state.harvest_cost += cost_delta
                harvest_cost = cost_delta
                state.batches += 1
                state.exhausted = state.agent.exhausted
                state.last_report = report
                batch_report = report

            if not state.candidates:
                lines = [
                    f"Harvester {harvester_id} has no candidates.",
                    f"Exhausted: {state.exhausted}",
                    self._format_harvester_status(),
                ]
                return "\n".join(lines), harvest_cost

            # Dequeue up to max_count
            to_process = state.candidates[:max_count]
            state.candidates = state.candidates[max_count:]

            # Spawn row generators in the background — don't block
            task = asyncio.create_task(
                self._process_batch_background(harvester_id, state, to_process)
            )
            self._background_tasks.append(task)

            lines = [
                f"Harvested from {harvester_id}: {len(to_process)} candidates queued for processing.",
                f"  Buffer remaining: {len(state.candidates)}",
                f"  Exhausted: {state.exhausted}",
            ]
            if batch_report:
                lines.append(f"\nBatch report: {batch_report}")
            lines.append(f"\nRow generation running in background. Results will appear on your next tool call.")
            lines.append(self._format_harvester_status())

            return "\n".join(lines), harvest_cost

        registry.add(
            name="process",
            description=(
                "Harvest a batch of candidates and start processing them into rows. "
                "Returns immediately with the harvest report. Row generation runs "
                "in the background — results appear as updates on subsequent tool calls."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "harvester_id": {
                        "type": "string",
                        "description": "Which harvester to process from (e.g. 'harvest:0')",
                    },
                    "max_count": {
                        "type": "integer",
                        "description": "Maximum candidates to process (default 10)",
                    },
                },
                "required": ["harvester_id"],
            },
            handler=process,
        )

        # --- close_harvest ---
        async def close_harvest(args: Dict) -> tuple[str, float]:
            harvester_id = args.get("harvester_id", "")

            state = self._harvesters.get(harvester_id)
            if not state:
                return f"Error: unknown harvester '{harvester_id}'", 0.0

            await state.agent.close()

            total_cost = state.harvest_cost + state.process_cost
            cpr = (
                f"${total_cost / state.rows_produced:.3f}"
                if state.rows_produced > 0 else "N/A"
            )
            summary = (
                f"Harvester {harvester_id} closed.\n"
                f"  Source: {state.source}\n"
                f"  {state.total_harvested} harvested, {state.total_processed} processed\n"
                f"  {state.rows_produced} rows, {state.duplicates} dupes, "
                f"{state.skipped} skipped, {state.errors} errors\n"
                f"  Harvest cost: ${state.harvest_cost:.4f}, "
                f"Process cost: ${state.process_cost:.4f}\n"
                f"  All-in cost/row: {cpr}"
            )

            del self._harvesters[harvester_id]

            return f"{summary}\n{self._format_harvester_status()}", 0.0

        registry.add(
            name="close_harvest",
            description=(
                "Close a harvester and free its browser session. Use when a source "
                "is exhausted or not worth continuing."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "harvester_id": {
                        "type": "string",
                        "description": "Which harvester to close (e.g. 'harvest:0')",
                    },
                },
                "required": ["harvester_id"],
            },
            handler=close_harvest,
        )

    # ── Background processing ──────────────────────────────────────────

    async def _process_batch_background(
        self,
        harvester_id: str,
        state: HarvesterState,
        candidates: List[Candidate],
    ) -> None:
        """Process a batch of candidates in the background. Pushes events when done."""
        try:
            results = await asyncio.gather(*[
                self._generate_row(candidate, state)
                for candidate in candidates
            ], return_exceptions=True)

            # Aggregate
            rows = 0
            skipped = 0
            dupes = 0
            errors = 0
            process_cost = 0.0

            for r in results:
                if isinstance(r, Exception):
                    errors += 1
                    logger.error(f"Row generation error: {r}")
                    continue
                gen_row, cost, saved = r
                process_cost += cost
                if gen_row.success:
                    rows += 1
                elif gen_row.skipped:
                    if "duplicate" in (gen_row.skip_reason or "").lower():
                        dupes += 1
                    else:
                        skipped += 1
                else:
                    errors += 1

            # Update state
            state.total_processed += len(candidates)
            state.rows_produced += rows
            state.duplicates += dupes
            state.skipped += skipped
            state.errors += errors
            state.process_cost += process_cost

            self._generation_stats["rows_generated"] = (
                self._generation_stats.get("rows_generated", 0) + rows
            )
            self._generation_stats["skipped"] = (
                self._generation_stats.get("skipped", 0) + skipped + dupes
            )
            self._generation_stats["errors"] = (
                self._generation_stats.get("errors", 0) + errors
            )

            # Push batch_done event
            total_rows = self._generation_stats.get("rows_generated", 0)
            event = (
                f"[batch_done] {harvester_id}: {len(candidates)} processed → "
                f"{rows} rows, {skipped} skipped, {dupes} dupes, {errors} errors. "
                f"Cost: ${process_cost:.4f}. "
                f"Progress: {total_rows}/{self.num_samples} rows."
            )
            self._pending_events.append(event)

            # Check if source exhausted (no buffer left + harvester done)
            if state.exhausted and not state.candidates:
                self._pending_events.append(
                    f"[source_exhausted] {harvester_id}: source fully drained. "
                    f"Total: {state.rows_produced} rows from {state.total_harvested} candidates."
                )

            logger.info(f"[orchestrator] {event}")

        except Exception as e:
            logger.error(f"[orchestrator] Background processing error for {harvester_id}: {e}")
            self._pending_events.append(
                f"[error] {harvester_id}: background processing failed: {e}"
            )

    def drain_pending_events(self) -> str:
        """Collect and clear all pending background events. Called by base.py after tool execution."""
        if not self._pending_events:
            return ""
        events = "\n".join(self._pending_events)
        self._pending_events.clear()
        # Clean up completed background tasks
        self._background_tasks = [t for t in self._background_tasks if not t.done()]
        return f"\n\n---\nBackground updates:\n{events}"

    # ── Row generation ────────────────────────────────────────────────

    async def _generate_row(
        self,
        candidate: Candidate,
        state: HarvesterState,
    ) -> tuple:
        """Generate one row from a candidate. Semaphore-limited for concurrency."""
        from dsl_worker.agents.row import RowGeneratorAgent, GeneratedRow

        async with self._generation_semaphore:
            agent = RowGeneratorAgent(
                openai_client=self.openai_client,
                model=self.generation_model,
                workspace_dir=self.workspace_dir,
                chat_history=self.chat_history,
                dedup_store=self._dedup_store,
                bu_client=self.bu_client,
                sandbox=self.sandbox,
                stop_checker=self.stop_checker,
                stop_event=self.stop_event,
                blob_service_client=self.blob_service_client,
                project_id=self.project_id,
                uploaded_file_urls=self.uploaded_file_urls,
                mcp_tools=self.mcp_tools,
                on_cost=self.on_cost,
            )
            try:
                result = await agent.generate(
                    candidate=candidate.values,
                    schema=self.columns,
                    source_context=candidate.source_context or state.description,
                )

                saved = False
                if result.success and result.row:
                    async with self._save_lock:
                        row_id = await self._save_row(result.row)
                        saved = row_id is not None

                return result, result.cost_usd, saved

            except Exception as e:
                logger.error(f"Row generation error: {e}", exc_info=True)
                return GeneratedRow(success=False, error=str(e)), 0.0, False

            finally:
                try:
                    await agent.cleanup()
                except Exception:
                    pass

    # ── Run ───────────────────────────────────────────────────────────

    async def run(self) -> AgentResult:
        if self.feedback_context:
            message = (
                "Begin. The user reviewed previous results and gave feedback "
                "(shown in system prompt). Research as needed and design a new pipeline."
            )
        else:
            message = (
                "Begin. Read the conversation history and resources, reason about "
                "strategy, then start harvesting candidate sources."
            )

        def _should_exit() -> bool:
            rows_done = self._generation_stats.get("rows_generated", 0)
            if rows_done >= self.num_samples:
                logger.info(
                    f"[orchestrator] Target reached: {rows_done}/{self.num_samples}"
                )
                return True
            return False

        return await self._conversation.send(
            message,
            exit_condition=_should_exit,
        )

    @property
    def cost_usd(self) -> float:
        return self._conversation.total_cost

    async def cleanup(self) -> None:
        """Wait for background tasks and close all harvesters."""
        # Wait for any in-flight row generation
        if self._background_tasks:
            pending = [t for t in self._background_tasks if not t.done()]
            if pending:
                logger.info(f"[orchestrator] Waiting for {len(pending)} background tasks...")
                await asyncio.gather(*pending, return_exceptions=True)
            self._background_tasks.clear()

        for hid, state in list(self._harvesters.items()):
            try:
                await state.agent.close()
            except Exception as e:
                logger.warning(f"Harvester {hid} cleanup error: {e}")
        self._harvesters.clear()
