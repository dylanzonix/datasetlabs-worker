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

**Web search** (built-in) — Use for general research: finding information, \
discovering sources, verifying facts. Fast, cheap, pre-indexed rendered web content.

**code_exec(script)** — Run Python in a sandbox. Files are at /workspace/uploads/. \
Use for quick data inspection if needed, but file metadata is already shown in \
Resources below.

{apollo_tools_section}\
## Tools

**create_harvester(source, candidate_description)** — Set up a harvester for a \
specific source. One harvester = one specific search query, page, or data slice. \
If you have 3 different search terms on the same site, create 3 harvesters. \
For uploaded files, point the harvester at the file path — it can run Python \
scripts to parse CSVs, JSON, etc. and programmatically submit hundreds of \
candidates in one shot.

**process(source_id, batch_size)** — Start continuously processing candidates from \
a source. Keeps the pipeline fed automatically — call it once per source. Returns \
after the first batch completes with results. The loop keeps running, reporting \
results as batches finish. Stops when the buffer empties or target is reached.

**close_source(source_id)** — Close a source and free its resources.

## Candidate Descriptions (for create_harvester)

The candidate_description tells the harvester what to look for. Keep it focused \
on the WHAT, not the HOW. The harvester discovers and yields candidates — row \
generators handle all enrichment (emails, phones, LinkedIn, etc.).

For list-based sources (directories, search results): just describe what entity \
to grab and obvious surface-level dealbreakers.

For open-ended research (niche leads with no clean list): the harvester may need \
to do quick validation per candidate (e.g. "is this actually a taco shell \
manufacturer?") but should NOT enrich candidates with contact details.

## Strategy

- **Apollo first for B2B.** Apollo is a structured business database (like \
LinkedIn data via API). Search is free, enrichment is cheap. For any project \
involving businesses, professionals, or companies, start with Apollo. It won't \
help for non-business entities (rappers, restaurants, etc.).
- **Read the results.** process() tells you exactly what happened — rows, skips \
(with reasons), duplicates, errors. React to this. If a source has low fertility \
or bad skip reasons, stop using it.
- **Don't create backup sources preemptively.** Start Apollo or one harvester, \
see the results, THEN decide if you need more sources. Don't hedge by creating \
harvesters before seeing how the first source performs.
- process() starts a continuous loop — call it once per source. Multiple sources \
run concurrently.
- Stay aware of how many rows you still need (shown in the status).

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
        on_checkpoint: Optional[Callable] = None,
        uploaded_file_urls: Optional[Dict[str, str]] = None,
        mcp_tools: Optional[List[Dict[str, Any]]] = None,
        apollo_client: Optional[ApolloClient] = None,
        feedback_context: Optional[Dict[str, Any]] = None,
        resume_context: Optional[Dict[str, Any]] = None,
        harvester_model: str = "",
        generation_model: str = "",
    ) -> None:
        self.feedback_context = feedback_context
        self.resume_context = resume_context
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
        self._on_checkpoint = on_checkpoint
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
        self._last_checkpoint_time: float = 0.0

        from dsl_worker.config import settings
        self._generation_semaphore = asyncio.Semaphore(
            settings.generation_parallel_samples
        )

        registry = ToolRegistry()
        self._register_tools(registry)

        # Format columns as readable lines (supports both old type-based and new format-based)
        if columns:
            col_lines = []
            for col in columns:
                name = col.get("name", "?")
                fmt = col.get("format", "")
                col_type = col.get("type", "")
                if fmt:
                    col_lines.append(f"- {name} — {fmt}")
                elif col_type:
                    col_lines.append(f"- {name} ({col_type})")
                else:
                    col_lines.append(f"- {name}")
            columns_desc = "\n".join(col_lines)
        else:
            columns_desc = "(no columns defined)"
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
            "## Apollo.io\n\n"
            "Apollo is a B2B database — 210M+ professionals and 30M+ companies. Think of \\\n"
            "it like LinkedIn data via API. Great for: business contacts, company lists, \\\n"
            "lead generation, professional searches. Not useful for: non-business entities \\\n"
            "(artists, restaurants, events, etc.).\n\n"
            "**apollo_search(...)** — Search people. FREE, no credits. Filter by title, \\\n"
            "seniority, location, industry, company size, etc. 100 results per page. \\\n"
            "Results are auto-buffered as candidates.\n\n"
            "**apollo_search_companies(...)** — Search companies. Filter by industry, \\\n"
            "location, size, revenue, funding stage, tech stack, founding year.\n\n"
            "Apollo search is free and fast — much cheaper than web harvesting. Prefer \\\n"
            "it over harvesters when the project involves businesses or professionals. \\\n"
            "Row generators automatically enrich Apollo candidates with emails, phones, \\\n"
            "and full details — you don't need to do that.\n\n"
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
        lines = ["Uploaded files (create a harvester pointed at the file path — it can parse programmatically):"]
        for idx, f in enumerate(uploaded_files, 1):
            name = f.get("filename", "unknown")
            size = f.get("size_bytes", 0)
            ctype = f.get("content_type", "")
            if size > 1_000_000:
                size_str = f"{size / 1_000_000:.1f} MB"
            elif size > 1_000:
                size_str = f"{size / 1_000:.0f} KB"
            else:
                size_str = f"{size} bytes"
            lines.append(f"  [{idx}] /workspace/uploads/{name} ({ctype}, {size_str})")

            # Rich metadata from file inspection
            inspection = f.get("inspection")
            if inspection:
                ftype = inspection.get("type", "")
                row_count = inspection.get("row_count") or inspection.get("item_count") or inspection.get("line_count")
                cols = inspection.get("columns") or inspection.get("keys") or []
                if row_count is not None:
                    lines.append(f"    {row_count} rows, {len(cols)} columns")
                if cols:
                    cols_str = ", ".join(cols[:20])
                    if len(cols) > 20:
                        cols_str += f" ... ({len(cols)} total)"
                    lines.append(f"    Columns: {cols_str}")
                preview = inspection.get("preview")
                if preview:
                    lines.append(f"    Sample row: {json.dumps(preview[0], default=str)[:300]}")
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

    def _maybe_checkpoint(self, force: bool = False) -> None:
        """Fire checkpoint callback if enough time has passed (15s throttle)."""
        if not self._on_checkpoint:
            return
        now = time.time()
        if not force and now - self._last_checkpoint_time < 15.0:
            return
        self._last_checkpoint_time = now
        try:
            self._on_checkpoint(self)
        except Exception as e:
            logger.warning(f"[orchestrator] checkpoint callback error: {e}")

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

        # --- code_exec ---
        self._sandbox_impl = None
        if self.sandbox:
            from dsl_worker.infra.research_tools import ResearchTools, ResearchScope
            self._sandbox_impl = ResearchTools(
                workspace_dir=self.workspace_dir,
                schema=[],
                brave_api_key=None,
                openai_client=self.openai_client,
                model=self.model,
                sandbox=self.sandbox,
                stop_checker=self.stop_checker,
                blob_service_client=self.blob_service_client,
                project_id=self.project_id,
                uploaded_file_urls=self.uploaded_file_urls,
            )
            self._sandbox_impl.set_scope(ResearchScope(
                id="orchestrator",
                description="",
                quota=0,
            ))
            self._sandbox_impl.register_on(
                registry,
                exclude=[
                    "brave_search", "open", "find", "click",
                    "interact", "shell_exec", "list_files",
                ],
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
            self._maybe_checkpoint(force=True)

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
                self._process_source_continuous(source_id, state, report_every=batch_size)
            )
            self._pending_batches.append(task)

            # Wait until we have initial results to report
            while not self._completed_results:
                pending = [t for t in self._pending_batches if not t.done()]
                if not pending:
                    break  # All done (fast source or error)
                await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
                await asyncio.sleep(0.01)

            completed_text = self._collect_completed()
            lines = []
            if completed_text:
                lines.append(completed_text)
            pending_count = len([t for t in self._pending_batches if not t.done()])
            if pending_count:
                lines.append(f"\n{pending_count} source(s) still processing in background.")
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
        report_every: int,
    ) -> None:
        """Continuously feed candidates into the semaphore one at a time.

        No batching — as each row generator finishes, its semaphore slot is
        immediately filled by the next candidate. Reports results to the
        orchestrator every `report_every` completions via _completed_results.
        """
        active: set = set()
        completed_count = 0
        batch_rows = 0
        batch_skipped = 0
        batch_dupes = 0
        batch_errors = 0
        batch_cost = 0.0
        batch_skip_reasons: List[str] = []

        def _should_stop_source() -> bool:
            if self._generation_stats.get("rows_generated", 0) >= self.num_samples:
                return True
            if self.stop_checker and self.stop_checker():
                return True
            return False

        def _report_batch() -> None:
            nonlocal completed_count, batch_rows, batch_skipped, batch_dupes
            nonlocal batch_errors, batch_cost, batch_skip_reasons
            if completed_count == 0:
                return
            total_rows = self._generation_stats.get("rows_generated", 0)
            lines = [
                f"[batch_complete] {source_id}: {completed_count} processed → "
                f"{batch_rows} rows, {batch_skipped} skipped, "
                f"{batch_dupes} dupes, {batch_errors} errors. "
                f"Cost: ${batch_cost:.4f}. Progress: {total_rows}/{self.num_samples}.",
            ]
            if batch_skip_reasons:
                lines.append("  Skip reasons:")
                for reason in batch_skip_reasons[:5]:
                    lines.append(f"    - {reason[:120]}")
            lines.append(self._format_source_status())
            self._completed_results.append("\n".join(lines))
            logger.info(
                f"[orchestrator] {source_id}: {completed_count} processed → "
                f"{batch_rows} rows. Progress: {total_rows}/{self.num_samples}."
            )
            self._maybe_checkpoint()
            completed_count = 0
            batch_rows = 0
            batch_skipped = 0
            batch_dupes = 0
            batch_errors = 0
            batch_cost = 0.0
            batch_skip_reasons = []

        def _handle_completed(task: asyncio.Task) -> None:
            nonlocal completed_count, batch_rows, batch_skipped, batch_dupes
            nonlocal batch_errors, batch_cost, batch_skip_reasons
            try:
                gen_row, cost, saved = task.result()
                batch_cost += cost
                if gen_row.success:
                    batch_rows += 1
                elif gen_row.skipped:
                    if gen_row.is_duplicate:
                        batch_dupes += 1
                    else:
                        batch_skipped += 1
                        if gen_row.skip_reason:
                            batch_skip_reasons.append(gen_row.skip_reason)
                else:
                    batch_errors += 1
            except Exception as e:
                batch_errors += 1
                logger.error(f"Row generation error: {e}")
            completed_count += 1

        # Feed candidates one at a time. Candidates stay in the buffer until
        # we actually acquire a semaphore slot. On pause, unstarted candidates
        # are still in the buffer and get checkpointed.

        while True:
            if _should_stop_source():
                break

            # Drain completed tasks (non-blocking)
            if active:
                done_now = {t for t in active if t.done()}
                if done_now:
                    active -= done_now
                    for task in done_now:
                        _handle_completed(task)

            # Report if threshold reached
            if completed_count >= report_every:
                state.rows_produced += batch_rows
                state.skipped += batch_skipped
                state.duplicates += batch_dupes
                state.errors += batch_errors
                state.process_cost += batch_cost
                self._generation_stats["skipped"] = (
                    self._generation_stats.get("skipped", 0) + batch_skipped + batch_dupes
                )
                self._generation_stats["errors"] = (
                    self._generation_stats.get("errors", 0) + batch_errors
                )
                _report_batch()

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

            # Nothing to do — wait for active tasks or exit
            if not state.candidates and not active:
                break
            if not state.candidates:
                # No more candidates but tasks still running — wait for one
                done, active = await asyncio.wait(
                    active, return_when=asyncio.FIRST_COMPLETED,
                )
                for task in done:
                    _handle_completed(task)
                continue

            # Acquire semaphore BEFORE popping candidate. This is the key:
            # candidate stays in buffer (checkpointable) until we actually
            # have a slot to process it.
            try:
                await self._generation_semaphore.acquire()
            except asyncio.CancelledError:
                break

            if _should_stop_source():
                self._generation_semaphore.release()
                break

            # NOW pop the candidate — we have a slot
            candidate = state.candidates.pop(0)
            state.total_processed += 1

            # Launch task (semaphore already acquired, task releases it)
            task = asyncio.create_task(
                self._generate_row_with_slot(candidate, state)
            )
            active.add(task)

        # Cancel remaining active tasks on stop
        if active:
            for t in active:
                t.cancel()
            await asyncio.gather(*active, return_exceptions=True)
            # Collect results from any that finished before cancel took effect
            for t in active:
                if t.done() and not t.cancelled():
                    try:
                        _handle_completed(t)
                    except Exception:
                        pass

        # Final report for any unreported completions
        if completed_count > 0:
            state.rows_produced += batch_rows
            state.skipped += batch_skipped
            state.duplicates += batch_dupes
            state.errors += batch_errors
            state.process_cost += batch_cost
            self._generation_stats["skipped"] = (
                self._generation_stats.get("skipped", 0) + batch_skipped + batch_dupes
            )
            self._generation_stats["errors"] = (
                self._generation_stats.get("errors", 0) + batch_errors
            )
            _report_batch()

        logger.info(f"[orchestrator] {source_id}: continuous processing ended")

    def _collect_completed(self) -> str:
        """Collect results from completed background batches."""
        self._pending_batches = [t for t in self._pending_batches if not t.done()]
        if not self._completed_results:
            return ""
        results = "\n".join(self._completed_results)
        self._completed_results.clear()
        return f"\n--- Completed Batches ---\n{results}"


    # ── Row generation ────────────────────────────────────────────────

    async def _generate_row_with_slot(
        self,
        candidate: Candidate,
        state: SourceState,
    ) -> tuple:
        """Generate one row. Semaphore already acquired — released on exit."""
        from dsl_worker.agents.row import RowGeneratorAgent, GeneratedRow

        try:
            rows_done = self._generation_stats.get("rows_generated", 0)
            if rows_done >= self.num_samples:
                return GeneratedRow(success=False, skipped=True, skip_reason="target reached"), 0.0, False

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
                uploaded_files=self.uploaded_files,
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
                        row_id = await self._save_row(
                            result.row,
                            tags={"sources": result.sources} if result.sources else {},
                        )
                        saved = row_id is not None
                        if saved:
                            self._generation_stats["rows_generated"] = (
                                self._generation_stats.get("rows_generated", 0) + 1
                            )

                return result, result.cost_usd, saved

            except asyncio.CancelledError:
                # Stopped mid-generation — not an error, just interrupted
                return GeneratedRow(success=False, skipped=True, skip_reason="stopped"), 0.0, False

            except Exception as e:
                logger.error(f"Row generation error: {e}", exc_info=True)
                return GeneratedRow(success=False, error=str(e)), 0.0, False

            finally:
                try:
                    await agent.cleanup()
                except Exception:
                    pass
        finally:
            # Always release the semaphore slot
            self._generation_semaphore.release()

    # ── Run ───────────────────────────────────────────────────────────

    async def run(self) -> AgentResult:
        def _should_exit() -> bool:
            rows_done = self._generation_stats.get("rows_generated", 0)
            if rows_done >= self.num_samples:
                logger.info(
                    f"[orchestrator] Target reached: {rows_done}/{self.num_samples}"
                )
                return True
            return False

        # If conversation was restored from checkpoint, just continue the loop.
        # No injected message — the LLM sees its full prior conversation and
        # picks up exactly where it left off (same tool calls, same results).
        if self._conversation.messages:
            logger.info(
                f"[orchestrator] Resuming from checkpoint: "
                f"{len(self._conversation.messages)} messages, "
                f"{self._generation_stats.get('rows_generated', 0)}/{self.num_samples} rows"
            )
            return await self._conversation._run_loop(exit_condition=_should_exit)

        # Fresh start
        if self.feedback_context:
            message = (
                "Begin. The user reviewed previous results and gave feedback "
                "(shown in system prompt). Research as needed and design a new pipeline."
            )
        elif self.resume_context:
            # Resume without V11 checkpoint state (legacy checkpoint or first V11 resume)
            rc = self.resume_context
            message = (
                f"RESUME. This job was paused and is now continuing. "
                f"{rc['rows_generated']}/{rc['target']} rows already generated "
                f"({rc['remaining']} remaining). Prior cost: ${rc['prior_cost_usd']:.4f}.\n\n"
                f"Pick up where the previous run left off. The conversation history "
                f"has all the context. Focus on generating the remaining "
                f"{rc['remaining']} rows. Existing rows are in the dedup store."
            )
        else:
            message = (
                "Begin. Read the conversation history and resources, reason about "
                "strategy, then start harvesting candidate sources."
            )

        return await self._conversation.send(
            message,
            exit_condition=_should_exit,
        )

    @property
    def cost_usd(self) -> float:
        return self._conversation.total_cost

    def export_state(self) -> Dict[str, Any]:
        """Export full pipeline state for checkpointing.

        Takes a snapshot of all mutable state. Safe to call from on_tool_call
        since orchestrator tools run sequentially (no concurrent mutation).
        """
        sources = []
        for sid, state in list(self._sources.items()):
            source_data = {
                "id": state.id,
                "source": state.source,
                "description": state.description,
                "total_harvested": state.total_harvested,
                "total_processed": state.total_processed,
                "rows_produced": state.rows_produced,
                "duplicates": state.duplicates,
                "skipped": state.skipped,
                "errors": state.errors,
                "exhausted": state.exhausted,
                "harvest_cost": state.harvest_cost,
                "process_cost": state.process_cost,
                "batches": state.batches,
                "last_report": state.last_report,
                # Snapshot candidate buffer (copy to avoid mutation during save)
                "candidates": [
                    {
                        "values": c.values,
                        "source_id": c.source_id,
                        "source_context": c.source_context,
                        "metadata": c.metadata,
                    }
                    for c in list(state.candidates)
                ],
                # Serialize harvester conversation if alive
                "harvester_conversation": None,
                "harvester_candidates_total": 0,
                "harvester_exhausted": False,
                "harvester_bu_session_id": None,
            }
            if state.agent is not None:
                agent = state.agent
                source_data["harvester_conversation"] = {
                    "messages": agent._conversation.messages,
                    "total_cost": agent._conversation.total_cost,
                    "total_turns": agent._conversation.total_turns,
                }
                source_data["harvester_candidates_total"] = agent._candidates_total
                source_data["harvester_exhausted"] = agent._exhausted
                source_data["harvester_bu_session_id"] = agent._bu_session_id
            sources.append(source_data)

        return {
            "orchestrator_conversation": {
                "messages": list(self._conversation.messages),
                "total_cost": self._conversation.total_cost,
                "total_turns": self._conversation.total_turns,
            },
            "sources": sources,
            "generation_stats": dict(self._generation_stats),
            "harvester_counter": self._harvester_counter,
            "apollo_counter": self._apollo_counter,
            "research_counter": self._research_counter,
        }

    def restore_state(self, state: Dict[str, Any]) -> None:
        """Restore pipeline state from checkpoint. Call BEFORE run()."""
        # Restore orchestrator conversation
        conv = state.get("orchestrator_conversation")
        if conv:
            self._conversation.messages = conv["messages"]
            self._conversation.total_cost = conv.get("total_cost", 0.0)
            self._conversation.total_turns = conv.get("total_turns", 0)
            logger.info(
                f"[orchestrator] Restored conversation: "
                f"{len(conv['messages'])} messages, "
                f"{conv.get('total_turns', 0)} turns, "
                f"${conv.get('total_cost', 0):.4f}"
            )

        # Restore counters
        self._harvester_counter = state.get("harvester_counter", 0)
        self._apollo_counter = state.get("apollo_counter", 0)
        self._research_counter = state.get("research_counter", 0)

        # Restore generation stats (merge — DB-seeded values take priority)
        saved_stats = state.get("generation_stats", {})
        for key in ("skipped", "errors", "total_cost"):
            if key in saved_stats:
                self._generation_stats[key] = saved_stats[key]

        # Restore source states — recreate harvester agents from saved conversations
        for src in state.get("sources", []):
            candidates = [
                Candidate(
                    values=c["values"],
                    source_id=c["source_id"],
                    source_context=c.get("source_context", ""),
                    metadata=c.get("metadata", {}),
                )
                for c in src.get("candidates", [])
            ]

            # Recreate harvester agent if conversation was saved and source not exhausted
            agent = None
            harvester_conv = src.get("harvester_conversation")
            if harvester_conv and not src.get("exhausted", False):
                try:
                    from dsl_worker.agents.harvester import HarvesterAgent
                    # Extract index from source ID (e.g. "harvest:2" → 2)
                    idx = int(src["id"].split(":")[1]) if ":" in src["id"] else 0

                    agent = HarvesterAgent(
                        source=src["source"],
                        description=src["description"],
                        source_id=src["id"],
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
                    # Restore conversation state
                    agent._conversation.messages = harvester_conv["messages"]
                    agent._conversation.total_cost = harvester_conv.get("total_cost", 0.0)
                    agent._conversation.total_turns = harvester_conv.get("total_turns", 0)
                    agent._candidates_total = src.get("harvester_candidates_total", 0)
                    agent._exhausted = src.get("harvester_exhausted", False)
                    # BU session is dead (server-side timeout), new one created on demand
                    agent._bu_session_id = None

                    logger.info(
                        f"[orchestrator] Restored harvester for {src['id']}: "
                        f"{len(harvester_conv['messages'])} messages, "
                        f"{harvester_conv.get('total_turns', 0)} turns"
                    )
                except Exception as e:
                    logger.warning(f"[orchestrator] Failed to restore harvester for {src['id']}: {e}")
                    agent = None

            source_state = SourceState(
                id=src["id"],
                source=src["source"],
                description=src["description"],
                agent=agent,
                candidates=candidates,
                total_harvested=src.get("total_harvested", 0),
                total_processed=src.get("total_processed", 0),
                rows_produced=src.get("rows_produced", 0),
                duplicates=src.get("duplicates", 0),
                skipped=src.get("skipped", 0),
                errors=src.get("errors", 0),
                harvest_cost=src.get("harvest_cost", 0.0),
                process_cost=src.get("process_cost", 0.0),
                batches=src.get("batches", 0),
                exhausted=src.get("exhausted", False),
                last_report=src.get("last_report", ""),
            )
            self._sources[src["id"]] = source_state
            logger.info(
                f"[orchestrator] Restored source {src['id']}: "
                f"{src.get('total_harvested', 0)} harvested, "
                f"{src.get('rows_produced', 0)} rows, "
                f"{'exhausted' if src.get('exhausted') else 'active'}, "
                f"{len(candidates)} buffered candidates, "
                f"{'agent restored' if agent else 'no agent'}"
            )

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

        if self._sandbox_impl:
            try:
                await self._sandbox_impl.cleanup()
            except Exception:
                pass
