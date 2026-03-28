"""
Orchestrator agent — coordinates dataset generation.

V11: Orchestrator-driven batch pipeline. The orchestrator is the sole
decision-maker — no background consumers, no Thompson Sampling, no events.

- create_harvester() creates a harvester for web/file sources
- apollo_search() / apollo_search_companies() query Apollo directly (no harvester)
- process() processes candidates into rows (auto-batches if buffer empty for harvesters)
- close_source() stops a source and frees resources
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
from dsl_worker.infra.apollo_client import ApolloClient

logger = logging.getLogger(__name__)


# ── SourceState ──────────────────────────────────────────────────────

@dataclass
class SourceState:
    """Tracks a candidate source (harvester or Apollo query) and its candidates."""
    id: str                           # "harvest:0" or "apollo:0"
    source: str
    description: str
    agent: Any = None                 # HarvesterAgent or None (Apollo sources)
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
candidate sources and row generators (which turn candidates into verified \
dataset rows). You make all the decisions — what sources to tap, how fast \
to scale, when to pivot.

## How the Pipeline Works

1. You find candidates — via Apollo searches, web harvesters, or file uploads.
2. You call process() which runs row generators and returns the results \
immediately: how many rows submitted, skipped (with reasons), duplicates, errors.
3. Based on those results you decide: process more from the same source, \
try a different source, adjust filters, or pivot strategy.
4. You keep iterating until the target is reached.

## Your Tools

**explore_agent(task)** — Inspect uploaded files, data, schemas via code execution. \
No web access.

**Web search** (built-in) is available for general research — quick factual lookups, \
finding URLs, verifying information. It uses pre-indexed, rendered web content.

**browse(task)** — Launch a full cloud browser for tasks that need live interaction: \
navigating JS-heavy sites, bypassing anti-bot/captcha, scrolling through paginated \
content, filling forms, or accessing content that wouldn't be indexed. Write the task \
as a clear instruction, not keywords. The browser has anti-bot stealth, residential \
proxy, and automatic captcha solving.

{apollo_tools_section}\
**create_harvester(source, candidate_description)** — Set up a web/file harvester \
for a source. The candidate_description tells the harvester what to look for. \
Returns a source_id. Use for websites, search results, uploaded files — anything \
that needs browser navigation or code execution.

**process(source_id, batch_size)** — Start continuously processing candidates from \
a source. Keeps the pipeline fed automatically — you call it once per source, not \
per batch. Returns after the first batch completes with initial results. The loop \
keeps running in the background, reporting results as batches finish. Stops when \
the buffer is empty or target is reached.

**close_source(source_id)** — Close a source and free its resources.

## Writing Good Candidate Descriptions (for create_harvester)

The candidate_description tells the harvester what to look for ON LIST PAGES. \
Keep it simple — the harvester just grabs names/identifiers, row generators do \
all the deep research and enrichment.

Include:
- What entity to look for (company, person, job posting, etc.)
- Any obvious surface-level dealbreakers (wrong category, wrong location)

Do NOT ask the harvester to find emails, phone numbers, LinkedIn profiles, or \
detailed info per candidate. That's the row generator's job.

Example: "Apartment property management companies in the Seattle/Bellevue/Tacoma WA \
area. Skip anything that's clearly not a property management company (staffing firms, \
real estate brokerages, industry associations)."

## Strategy

- If you know or can guess the source, go straight to action. \
You likely already know how major sites work (URL patterns, search params, etc.) — \
use that knowledge directly instead of researching first.
- **Read the results.** process() tells you exactly what happened — how many rows, \
how many skipped and WHY, duplicates, errors. Use this to decide your next move. \
If a source has low fertility or bad skip reasons, stop using it.
- process() starts a continuous processing loop on a source — call it once, not \
per batch. It keeps feeding candidates into row generators automatically. You \
see results as batches complete. Multiple process() calls on different sources \
run concurrently.
- Start small to validate a source, then scale up as confidence grows. \
Stay aware of how many rows you still need (shown in the status).
- Close sources that aren't producing.
- The browser (browse tool) handles JavaScript, anti-bot, CAPTCHAs, and dynamic \
pages. JS is almost never the issue if something isn't loading.

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
        apollo_client: Optional[ApolloClient] = None,
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
        self.apollo_client = apollo_client
        self.harvester_model = harvester_model or model
        self.generation_model = generation_model or "gpt-5-mini"

        self._generation_stats = generation_stats
        self._save_row = save_row
        self._dedup_store = dedup_store
        self._save_lock = asyncio.Lock()

        self._research_counter = 0
        self._harvester_counter = 0
        self._apollo_counter = 0
        self._sources: Dict[str, SourceState] = {}
        self._pending_batches: List[asyncio.Task] = []
        self._completed_results: List[str] = []

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
        apollo_tools_section = self._format_apollo_tools_section()

        from datetime import date
        system_prompt = ORCHESTRATOR_SYSTEM_PROMPT.format(
            num_samples=num_samples,
            columns_description=columns_desc,
            conversation_summary=convo_summary,
            resources_section=resources_section,
            feedback_section=feedback_section,
            apollo_tools_section=apollo_tools_section,
            current_date=date.today().isoformat(),
        )

        max_turns = getattr(settings, 'orchestrator_max_turns', 40)
        soft_limit = getattr(settings, 'orchestrator_soft_limit', 25)

        # Built-in web search available to all agents
        web_search_tool = {"type": "web_search"}
        all_extra_tools = [web_search_tool] + self.mcp_tools

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
            extra_tools=all_extra_tools,
            drain_events=self._collect_completed,
        )

    # ── Formatting helpers ────────────────────────────────────────────

    def _format_apollo_tools_section(self) -> str:
        if not self.apollo_client:
            return ""
        return (
            "**apollo_search(...)** — Search Apollo.io's 210M+ contact database. FREE, \\\n"
            "no credits. Returns people with title, company, LinkedIn. Each result is \\\n"
            "auto-buffered as a candidate. Use structured filters: titles, seniorities, \\\n"
            "locations, industries, company size, revenue, tech stack, email status. \\\n"
            "Paginate with page param (100/page, up to 500 pages). Call process() on \\\n"
            "the returned source_id to start row generation.\n\n"
            "**apollo_search_companies(...)** — Search Apollo.io's 30M+ company database. \\\n"
            "Filter by industry, location, size, revenue, funding stage, founding year, \\\n"
            "tech stack. Each result auto-buffered as a candidate.\n\n"
            "Row generators have **apollo_enrich** to get emails, phones, and full details \\\n"
            "(1 credit/person). You don't need to enrich — row generators do it automatically \\\n"
            "when they see an apollo_id in the candidate.\n\n"
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

    def _format_source_status(self) -> str:
        """Format full status of all sources for tool output."""
        rows_done = self._generation_stats.get("rows_generated", 0)
        total_skipped = self._generation_stats.get("skipped", 0)
        total_errors = self._generation_stats.get("errors", 0)
        remaining = self.num_samples - rows_done

        lines = [
            f"\n--- Pipeline Status ---",
            f"Progress: {rows_done}/{self.num_samples} rows ({remaining} remaining)",
            f"Total skipped: {total_skipped}, Total errors: {total_errors}",
        ]

        if not self._sources:
            return "\n".join(lines)

        lines.append(f"\nSources ({len(self._sources)}):")
        for sid, state in self._sources.items():
            fertility = (
                f"{state.rows_produced / state.total_processed:.0%}"
                if state.total_processed > 0 else "N/A"
            )
            total_cost = state.harvest_cost + state.process_cost
            cpr = (
                f"${total_cost / state.rows_produced:.3f}"
                if state.rows_produced > 0 else "N/A"
            )
            status = "EXHAUSTED" if state.exhausted else f"{len(state.candidates)} buffered"
            lines.append(
                f"  {sid} ({state.source[:50]}):"
            )
            lines.append(
                f"    {state.rows_produced} rows, {state.skipped} skipped, "
                f"{state.duplicates} dupes, {state.errors} errors "
                f"(from {state.total_processed}/{state.total_harvested} processed/harvested)"
            )
            lines.append(
                f"    Fertility: {fertility}, Cost/row: {cpr}, "
                f"Harvest: ${state.harvest_cost:.3f}, Process: ${state.process_cost:.3f}, "
                f"Buffer: {status}"
            )

        return "\n".join(lines)

    # ── Tool registration ─────────────────────────────────────────────

    def _register_tools(self, registry: ToolRegistry) -> None:

        # --- browse (full cloud browser via BU) ---
        async def browse(args: Dict) -> tuple[str, float]:
            task = args.get("task", "")
            if not task:
                return "Error: task is required", 0.0

            if not self.bu_client:
                return "Error: browser not available (no BU client)", 0.0

            fast_task = (
                f"{task}\n\n"
                "Be fast and direct. Take the shortest path to the answer. "
                "Do not explore or enumerate page elements — just do what's needed and return."
            )
            text, cost, _sid = await self.bu_client.research(fast_task)

            if self.on_cost and cost > 0:
                await self.on_cost(cost, "browse")

            if len(text) > 6000:
                text = text[:6000] + "\n\n[Truncated]"

            return text or "(no results)", cost

        registry.add(
            name="browse",
            description=(
                "Launch a full cloud browser to interact with web pages. Use for "
                "tasks that need live navigation, anti-bot bypass, JS rendering, "
                "captcha solving, form filling, or scrolling. Write as a clear "
                "instruction. Prefer the built-in web search for simple lookups."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "task": {
                        "type": "string",
                        "description": (
                            "A clear instruction for the browser. "
                            "Example: 'Go to upwork.com/nx/search/jobs and figure out "
                            "the URL parameters for filtering by recency and date posted. "
                            "Return the full URL with the correct filters applied.'"
                        ),
                    },
                },
                "required": ["task"],
            },
            handler=browse,
        )

        # --- explore_agent ---
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

        # --- Apollo tools (direct on orchestrator) ---
        if self.apollo_client:
            self._register_apollo_tools(registry)

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
            source_id = f"harvest:{idx}"

            harvester = HarvesterAgent(
                source=source,
                description=candidate_description,
                source_id=source_id,
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

            state = SourceState(
                id=source_id,
                source=source,
                description=candidate_description,
                agent=harvester,
            )
            self._sources[source_id] = state

            return (
                f"Harvester {source_id} created for: {source}\n"
                f"Call process(source_id='{source_id}', max_count=N) to start."
            ), 0.0

        registry.add(
            name="create_harvester",
            description=(
                "Set up a web/file harvester for a source. Returns a source_id. "
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
            source_id = args.get("source_id", "")
            batch_size = args.get("batch_size", 10)

            state = self._sources.get(source_id)
            if not state:
                available = list(self._sources.keys())
                return f"Error: unknown source '{source_id}'. Available: {available}", 0.0

            rows_done = self._generation_stats.get("rows_generated", 0)
            if rows_done >= self.num_samples:
                return f"Target already reached ({rows_done}/{self.num_samples}).\n" + self._format_source_status(), 0.0

            harvest_cost = 0.0

            # Auto-fetch if buffer empty, not exhausted, AND has a harvester agent
            if not state.candidates and not state.exhausted and state.agent is not None:
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

            if not state.candidates:
                hint = (
                    " Search for more with apollo_search/apollo_search_companies."
                    if state.agent is None else ""
                )
                return (
                    f"Source {source_id} has no candidates.{hint}\n"
                    f"Exhausted: {state.exhausted}\n"
                    + self._format_source_status()
                ), harvest_cost

            # Launch continuous processing loop for this source
            task = asyncio.create_task(
                self._process_source_continuous(source_id, state, batch_size)
            )
            self._pending_batches.append(task)

            # Wait for the first batch to complete so we have initial results
            pending = [t for t in self._pending_batches if not t.done()]
            if pending:
                await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
                await asyncio.sleep(0.01)

            completed_text = self._collect_completed()
            lines = []
            if completed_text:
                lines.append(completed_text)
            pending_count = len([t for t in self._pending_batches if not t.done()])
            if pending_count:
                lines.append(f"\n{pending_count} source(s) still processing.")
            lines.append(self._format_source_status())

            return "\n".join(lines), harvest_cost

        registry.add(
            name="process",
            description=(
                "Start continuously processing candidates from a source into rows. "
                "Keeps the pipeline fed automatically — you don't need to call it "
                "repeatedly. Waits for the first batch to complete and returns results. "
                "Subsequent batch results appear on your next action via status updates. "
                "The loop stops when the source buffer is empty or target is reached."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "source_id": {
                        "type": "string",
                        "description": "Which source to process from (e.g. 'harvest:0' or 'apollo:0')",
                    },
                    "batch_size": {
                        "type": "integer",
                        "description": "Row generators per batch (default 10). Controls concurrency.",
                    },
                },
                "required": ["source_id"],
            },
            handler=process,
        )

        # --- close_source ---
        async def close_source(args: Dict) -> tuple[str, float]:
            source_id = args.get("source_id", "")

            state = self._sources.get(source_id)
            if not state:
                return f"Error: unknown source '{source_id}'", 0.0

            if state.agent is not None:
                await state.agent.close()

            total_cost = state.harvest_cost + state.process_cost
            cpr = (
                f"${total_cost / state.rows_produced:.3f}"
                if state.rows_produced > 0 else "N/A"
            )
            summary = (
                f"Source {source_id} closed.\n"
                f"  Source: {state.source}\n"
                f"  {state.total_harvested} harvested, {state.total_processed} processed\n"
                f"  {state.rows_produced} rows, {state.duplicates} dupes, "
                f"{state.skipped} skipped, {state.errors} errors\n"
                f"  Harvest cost: ${state.harvest_cost:.4f}, "
                f"Process cost: ${state.process_cost:.4f}\n"
                f"  All-in cost/row: {cpr}"
            )

            del self._sources[source_id]

            return f"{summary}\n{self._format_source_status()}", 0.0

        registry.add(
            name="close_source",
            description=(
                "Close a source and free its resources. Use when a source "
                "is exhausted or not worth continuing."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "source_id": {
                        "type": "string",
                        "description": "Which source to close (e.g. 'harvest:0' or 'apollo:0')",
                    },
                },
                "required": ["source_id"],
            },
            handler=close_source,
        )

    # ── Apollo tools (direct on orchestrator) ─────────────────────────

    def _register_apollo_tools(self, registry: ToolRegistry) -> None:

        async def apollo_search(args: Dict) -> tuple[str, float]:
            source_id = args.get("source_id", "")
            page = args.get("page", 1)

            try:
                people, total = await self.apollo_client.search_people(
                    person_titles=args.get("person_titles") or None,
                    person_seniorities=args.get("person_seniorities") or None,
                    person_locations=args.get("person_locations") or None,
                    person_names=args.get("person_names") or None,
                    contact_email_status=args.get("contact_email_status") or None,
                    department_ids=args.get("department_ids") or None,
                    include_similar_titles=args.get("include_similar_titles"),
                    organization_keywords=args.get("organization_keywords") or None,
                    organization_name=args.get("organization_name") or None,
                    organization_locations=args.get("organization_locations") or None,
                    organization_not_locations=args.get("organization_not_locations") or None,
                    organization_num_employees_ranges=args.get("employee_ranges") or None,
                    organization_ids=args.get("organization_ids") or None,
                    organization_domains=args.get("organization_domains") or None,
                    organization_revenue_ranges=args.get("revenue_ranges") or None,
                    industry_tag_ids=args.get("industry_tag_ids") or None,
                    technology_uids=args.get("technology_uids") or None,
                    q_keywords=args.get("q_keywords") or None,
                    per_page=100,
                    page=page,
                )
            except Exception as e:
                return f"Apollo search error: {e}", 0.0

            # Get or create source state
            if source_id and source_id in self._sources:
                state = self._sources[source_id]
            else:
                idx = self._apollo_counter
                self._apollo_counter += 1
                source_id = f"apollo:{idx}"
                state = SourceState(
                    id=source_id,
                    source="Apollo.io People Search",
                    description="Apollo contact search",
                    agent=None,
                )
                self._sources[source_id] = state

            # Buffer each person as a candidate
            for person in people:
                org = person.get("organization") or {}
                candidate_data = json.dumps({
                    "apollo_id": person.get("id"),
                    "name": person.get("name"),
                    "first_name": person.get("first_name"),
                    "last_name": person.get("last_name"),
                    "title": person.get("title"),
                    "headline": person.get("headline"),
                    "linkedin_url": person.get("linkedin_url"),
                    "city": person.get("city"),
                    "state": person.get("state"),
                    "country": person.get("country"),
                    "seniority": person.get("seniority"),
                    "departments": person.get("departments"),
                    "organization_name": org.get("name"),
                    "organization_id": person.get("organization_id"),
                }, ensure_ascii=False)
                state.candidates.append(Candidate(
                    values=candidate_data,
                    source_id=source_id,
                    source_context="Apollo contact search",
                    metadata={"origin": "apollo"},
                ))
            state.total_harvested += len(people)

            # Pagination info
            if total > 0:
                total_pages = min((total + 99) // 100, 500)
                pagination_info = f"page {page}/{total_pages} ({total:,} total matches)"
            else:
                pagination_info = (
                    f"page {page} ({len(people)} returned). Try next page for more."
                    if len(people) >= 100 else
                    f"page {page} ({len(people)} returned)"
                )

            sample = people[0] if people else {}
            redacted_note = ""
            if people and not sample.get("name"):
                redacted_note = (
                    "\nNote: Names redacted in search — row generators will enrich "
                    "via apollo_id to get full contact info."
                )

            return (
                f"Apollo search: {len(people)} people, {pagination_info}.\n"
                f"Source: {source_id} ({len(state.candidates)} buffered, "
                f"{state.total_harvested} total).\n"
                f"Call process(source_id='{source_id}', max_count=N) to generate rows.\n"
                f"For more results, call apollo_search(source_id='{source_id}', page={page + 1})."
                f"{redacted_note}"
            ), 0.0

        registry.add(
            name="apollo_search",
            description=(
                "Search Apollo.io's 210M+ contact database. FREE — no credits. "
                "Returns people with title, company, LinkedIn. Each result is "
                "auto-buffered as a candidate. Call process() on the returned source_id "
                "to start row generation."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "person_titles": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Job titles (e.g. ['CEO', 'VP Marketing', 'Director of Sales'])",
                    },
                    "person_seniorities": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Seniority levels: c_suite, founder, owner, vp, director, manager, senior, head, entry, intern",
                    },
                    "person_locations": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Person locations (e.g. ['California, US', 'United States'])",
                    },
                    "person_names": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Search by individual names",
                    },
                    "contact_email_status": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Filter by email availability: 'verified', 'guessed', 'unavailable'",
                    },
                    "department_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Department classification IDs",
                    },
                    "include_similar_titles": {
                        "type": "boolean",
                        "description": "Expand search to include similar/related job titles",
                    },
                    "organization_keywords": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Industry/keyword tags (e.g. ['healthcare', 'fintech'])",
                    },
                    "organization_name": {
                        "type": "string",
                        "description": "Company name search (partial match)",
                    },
                    "organization_locations": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Company HQ locations",
                    },
                    "organization_not_locations": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Exclude companies HQ'd in these locations",
                    },
                    "employee_ranges": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Employee count: '1-10', '11-50', '51-200', '201-500', '501-1000', '1001-5000', '5001-10000', '10001+'",
                    },
                    "organization_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Apollo organization IDs (from previous searches)",
                    },
                    "organization_domains": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Company domains to filter by (e.g. ['apollo.io', 'google.com'])",
                    },
                    "revenue_ranges": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Company annual revenue ranges (Apollo revenue bracket codes)",
                    },
                    "industry_tag_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Industry category IDs",
                    },
                    "technology_uids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Technology stack UIDs — filter by tools/tech companies use",
                    },
                    "q_keywords": {
                        "type": "string",
                        "description": "Free text keyword search",
                    },
                    "source_id": {
                        "type": "string",
                        "description": "Append to existing Apollo source (for pagination). Omit to create new.",
                    },
                    "page": {
                        "type": "integer",
                        "description": "Page number (1-500). 100 results per page.",
                    },
                },
            },
            handler=apollo_search,
        )

        async def apollo_search_companies(args: Dict) -> tuple[str, float]:
            source_id = args.get("source_id", "")
            page = args.get("page", 1)

            try:
                orgs, total = await self.apollo_client.search_companies(
                    organization_keywords=args.get("keywords") or None,
                    organization_name=args.get("name") or None,
                    organization_locations=args.get("locations") or None,
                    organization_not_locations=args.get("not_locations") or None,
                    organization_num_employees_ranges=args.get("employee_ranges") or None,
                    organization_revenue_ranges=args.get("revenue_ranges") or None,
                    organization_latest_funding_stage_cd=args.get("funding_stages") or None,
                    technology_uids=args.get("technology_uids") or None,
                    website_urls=args.get("website_urls") or None,
                    industry_tag_ids=args.get("industry_tag_ids") or None,
                    founded_year_min=args.get("founded_year_min"),
                    founded_year_max=args.get("founded_year_max"),
                    publicly_traded=args.get("publicly_traded"),
                    per_page=100,
                    page=page,
                )
            except Exception as e:
                return f"Apollo company search error: {e}", 0.0

            if source_id and source_id in self._sources:
                state = self._sources[source_id]
            else:
                idx = self._apollo_counter
                self._apollo_counter += 1
                source_id = f"apollo:{idx}"
                state = SourceState(
                    id=source_id,
                    source="Apollo.io Company Search",
                    description="Apollo company search",
                    agent=None,
                )
                self._sources[source_id] = state

            for org in orgs:
                candidate_data = json.dumps({
                    "apollo_org_id": org.get("id"),
                    "company_name": org.get("name"),
                    "website": org.get("website_url"),
                    "industry": org.get("industry"),
                    "keywords": org.get("keywords"),
                    "estimated_employees": org.get("estimated_num_employees"),
                    "city": org.get("city"),
                    "state": org.get("state"),
                    "country": org.get("country"),
                    "linkedin_url": org.get("linkedin_url"),
                    "short_description": org.get("short_description"),
                    "founded_year": org.get("founded_year"),
                    "annual_revenue": org.get("annual_revenue"),
                    "total_funding": org.get("total_funding"),
                    "latest_funding_stage": org.get("latest_funding_stage"),
                }, ensure_ascii=False)
                state.candidates.append(Candidate(
                    values=candidate_data,
                    source_id=source_id,
                    source_context="Apollo company search",
                    metadata={"origin": "apollo"},
                ))
            state.total_harvested += len(orgs)

            if total > 0:
                total_pages = min((total + 99) // 100, 500)
                pagination_info = f"page {page}/{total_pages} ({total:,} total matches)"
            else:
                pagination_info = (
                    f"page {page} ({len(orgs)} returned). Try next page for more."
                    if len(orgs) >= 100 else
                    f"page {page} ({len(orgs)} returned)"
                )

            return (
                f"Apollo company search: {len(orgs)} companies, {pagination_info}.\n"
                f"Source: {source_id} ({len(state.candidates)} buffered, "
                f"{state.total_harvested} total).\n"
                f"Call process(source_id='{source_id}', max_count=N) to generate rows.\n"
                f"For more results, call apollo_search_companies(source_id='{source_id}', page={page + 1})."
            ), 0.0

        registry.add(
            name="apollo_search_companies",
            description=(
                "Search Apollo.io's 30M+ company database. Each result is "
                "auto-buffered as a candidate. Call process() on the returned source_id."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "keywords": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Industry/keyword tags (e.g. ['healthcare', 'design agency'])",
                    },
                    "name": {
                        "type": "string",
                        "description": "Company name search",
                    },
                    "locations": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Company HQ locations",
                    },
                    "not_locations": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Exclude companies in these locations",
                    },
                    "employee_ranges": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Employee count: '1-10', '11-50', '51-200', '201-500', '501-1000', '1001-5000', etc.",
                    },
                    "revenue_ranges": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Annual revenue ranges (Apollo revenue bracket codes)",
                    },
                    "funding_stages": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Latest funding stage codes (e.g. 'seed', 'series_a', 'series_b', 'ipo')",
                    },
                    "technology_uids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Technology stack UIDs — filter by tools/tech companies use",
                    },
                    "website_urls": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Filter by specific website URLs",
                    },
                    "industry_tag_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Industry category IDs",
                    },
                    "founded_year_min": {
                        "type": "integer",
                        "description": "Earliest founding year",
                    },
                    "founded_year_max": {
                        "type": "integer",
                        "description": "Latest founding year",
                    },
                    "publicly_traded": {
                        "type": "boolean",
                        "description": "Filter for publicly traded companies only",
                    },
                    "source_id": {
                        "type": "string",
                        "description": "Append to existing Apollo source (for pagination). Omit to create new.",
                    },
                    "page": {
                        "type": "integer",
                        "description": "Page number (1-500). 100 results per page.",
                    },
                },
            },
            handler=apollo_search_companies,
        )

    # ── Batch processing ─────────────────────────────────────────────

    async def _process_source_continuous(
        self,
        source_id: str,
        state: SourceState,
        batch_size: int,
    ) -> None:
        """Continuously feed candidates from a source into row generators.

        Keeps the semaphore full by launching new row generators as slots
        free up. Runs until buffer is empty or target is reached.
        Reports results per batch via _completed_results.
        """
        while state.candidates:
            # Check target
            rows_done = self._generation_stats.get("rows_generated", 0)
            if rows_done >= self.num_samples:
                logger.info(f"[orchestrator] {source_id}: target reached, stopping")
                break

            # Check stop
            if self.stop_checker and self.stop_checker():
                break

            # Dequeue a batch
            to_process = state.candidates[:batch_size]
            state.candidates = state.candidates[batch_size:]

            # Run this batch (row gens acquire semaphore internally)
            await self._run_batch_background(source_id, state, to_process)

            # If buffer empty, try to fetch more from harvester
            if not state.candidates and not state.exhausted and state.agent is not None:
                try:
                    candidates, report = await state.agent.run_batch(
                        "Get the next batch of candidates."
                    )
                    cost_delta = state.agent.batch_cost_delta
                    state.candidates.extend(candidates)
                    state.total_harvested += len(candidates)
                    state.harvest_cost += cost_delta
                    state.batches += 1
                    state.exhausted = state.agent.exhausted
                except Exception as e:
                    logger.error(f"[orchestrator] {source_id} fetch error: {e}")
                    break

        logger.info(f"[orchestrator] {source_id}: continuous processing ended")

    def _collect_completed(self) -> str:
        """Collect results from completed background batches."""
        self._pending_batches = [t for t in self._pending_batches if not t.done()]
        if not self._completed_results:
            return ""
        results = "\n".join(self._completed_results)
        self._completed_results.clear()
        return f"\n--- Completed Batches ---\n{results}"

    async def _run_batch_background(
        self,
        source_id: str,
        state: SourceState,
        candidates: List[Candidate],
    ) -> None:
        """Run row generators for a batch. Pushes result to _completed_results."""
        try:
            results = await asyncio.gather(*[
                self._generate_row(candidate, state)
                for candidate in candidates
            ], return_exceptions=True)

            rows = 0
            skipped = 0
            dupes = 0
            errors = 0
            process_cost = 0.0
            skip_reasons = []

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
                    if gen_row.is_duplicate:
                        dupes += 1
                    else:
                        skipped += 1
                        if gen_row.skip_reason:
                            skip_reasons.append(gen_row.skip_reason)
                else:
                    errors += 1

            state.total_processed += len(candidates)
            state.rows_produced += rows
            state.duplicates += dupes
            state.skipped += skipped
            state.errors += errors
            state.process_cost += process_cost

            self._generation_stats["skipped"] = (
                self._generation_stats.get("skipped", 0) + skipped + dupes
            )
            self._generation_stats["errors"] = (
                self._generation_stats.get("errors", 0) + errors
            )

            total_rows = self._generation_stats.get("rows_generated", 0)

            lines = [
                f"[batch_complete] {source_id}: {len(candidates)} processed → "
                f"{rows} rows, {skipped} skipped, {dupes} dupes, {errors} errors. "
                f"Cost: ${process_cost:.4f}. Progress: {total_rows}/{self.num_samples}.",
            ]
            if skip_reasons:
                lines.append("  Skip reasons:")
                for reason in skip_reasons[:5]:
                    lines.append(f"    - {reason[:120]}")
            lines.append(self._format_source_status())

            self._completed_results.append("\n".join(lines))

            logger.info(
                f"[orchestrator] batch done {source_id}: {len(candidates)} → "
                f"{rows} rows, {skipped} skipped, {dupes} dupes, {errors} errors. "
                f"Progress: {total_rows}/{self.num_samples}."
            )

        except Exception as e:
            logger.error(f"[orchestrator] Batch error {source_id}: {e}")
            self._completed_results.append(f"[batch_error] {source_id}: {e}")

    # ── Row generation ────────────────────────────────────────────────

    async def _generate_row(
        self,
        candidate: Candidate,
        state: SourceState,
    ) -> tuple:
        """Generate one row from a candidate. Semaphore-limited for concurrency."""
        from dsl_worker.agents.row import RowGeneratorAgent, GeneratedRow

        # Skip if target already reached (avoids wasting LLM calls)
        rows_done = self._generation_stats.get("rows_generated", 0)
        if rows_done >= self.num_samples:
            return GeneratedRow(success=False, skipped=True, skip_reason="target reached"), 0.0, False

        async with self._generation_semaphore:
            # Re-check after acquiring semaphore (may have been queued)
            rows_done = self._generation_stats.get("rows_generated", 0)
            if rows_done >= self.num_samples:
                return GeneratedRow(success=False, skipped=True, skip_reason="target reached"), 0.0, False

            # Row gen stop checker: also stops when target is reached
            def _row_stop():
                if self.stop_checker and self.stop_checker():
                    return True
                return self._generation_stats.get("rows_generated", 0) >= self.num_samples

            agent = RowGeneratorAgent(
                openai_client=self.openai_client,
                model=self.generation_model,
                workspace_dir=self.workspace_dir,
                chat_history=self.chat_history,
                dedup_store=self._dedup_store,
                bu_client=self.bu_client,
                sandbox=self.sandbox,
                stop_checker=_row_stop,
                stop_event=self.stop_event,
                blob_service_client=self.blob_service_client,
                project_id=self.project_id,
                uploaded_file_urls=self.uploaded_file_urls,
                apollo_client=self.apollo_client,
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
                        if saved:
                            # Update immediately so other row gens see it
                            self._generation_stats["rows_generated"] = (
                                self._generation_stats.get("rows_generated", 0) + 1
                            )

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
        """Cancel pending batches and close all sources."""
        pending = [t for t in self._pending_batches if not t.done()]
        if pending:
            for t in pending:
                t.cancel()
            await asyncio.gather(*pending, return_exceptions=True)
        self._pending_batches.clear()

        for sid, state in list(self._sources.items()):
            if state.agent is not None:
                try:
                    await state.agent.close()
                except Exception as e:
                    logger.warning(f"Source {sid} cleanup error: {e}")
        self._sources.clear()
