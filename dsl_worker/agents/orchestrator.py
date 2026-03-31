"""
Orchestrator agent — coordinates dataset generation.

V12: Async check-in orchestrator. The orchestrator sleeps between check-ins,
waking on time/cost intervals. Harvesters and row generators run in the
background. A round-robin dispatcher feeds candidates to row generators
automatically.

- create_harvester() creates AND starts a harvester immediately
- stop_harvester() kills a harvester
- apollo_search/apollo_search_companies query Apollo and push to dispatcher
- inspect() drills down into source/candidate/step detail
- finish() ends the job early
- Dashboard auto-injected at every check-in
- No process() — processing is automatic
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
from dsl_worker.infra.dispatcher import CandidateDispatcher, BatchToken

logger = logging.getLogger(__name__)


# ── SourceState ──────────────────────────────────────────────────────

@dataclass
class SourceState:
    """Tracks a candidate source (harvester or Apollo query)."""
    id: str                           # "harvest:0" or "apollo:0"
    source: str
    description: str
    agent: Any = None                 # HarvesterAgent or None (Apollo sources)
    task: Optional[asyncio.Task] = None  # background harvester loop task
    total_harvested: int = 0
    harvest_cost: float = 0.0
    batches: int = 0
    exhausted: bool = False
    harvester_reports: List[str] = field(default_factory=list)
    error: str = ""  # if harvester died with an error


# ── System Prompt ─────────────────────────────────────────────────────

ORCHESTRATOR_SYSTEM_PROMPT = """\
# Dataset Generation Orchestrator

You are the strategist in a dataset generation pipeline. Your job: produce \
{num_samples} rows at the lowest possible cost. You control which sources to \
tap and when to start/stop them. Everything else is automatic.

## How It Works

1. You create harvesters (or run Apollo searches) — they start immediately.
2. Candidates flow from harvesters to row generators automatically (round-robin, \
10 concurrent). You do NOT call process() — it's automatic.
3. At regular intervals, you see a DASHBOARD with per-source stats: cost, \
rows produced, skip rate, harvester reports.
4. Based on the dashboard, you decide: do nothing (things are fine), start a \
new source, kill a bad source, or finish.

## Your Tools

**Web search** (built-in) — Research to figure out where candidates live. \
Fast, cheap, pre-indexed.

**code_exec(script)** — Run Python in a sandbox. Files at /workspace/uploads/. \
Use to inspect uploaded files before creating harvesters.

{apollo_tools_section}\

**create_harvester(source, candidate_description)** — Create AND start a harvester \
immediately. One harvester = one slice (specific search query, page, or file). \
Multiple slices of the same site = multiple harvesters. The harvester runs in \
the background, producing candidates in batches. Candidates flow to row generators \
automatically.

**stop_harvester(source_id, reason)** — Kill a harvester. Remaining buffered \
candidates still get processed.

**inspect(source_id, ...)** — Drill down into a source for more context. \
Shows harvester tool calls, candidate outcomes, row generator details. Use when \
the dashboard shows something wrong and you need to understand why.

**finish(reason)** — End the job early. Use when: target is reached, budget \
concerns, or enough quality rows.

## Strategy

**Budget hint:** Aim for ~$2 to produce the first row (exploration budget). \
Once you know cost-per-row from real results, optimize. A source is expensive \
or cheap RELATIVE to other sources, not in absolute terms.

- **Apollo first for B2B.** Free search, cheap enrichment. Always try before web.
- **Start small, observe, scale.** Create 1-2 harvesters, see results, then decide.
- **React to the dashboard.** High skip rate? Kill it, try different keywords. \
High cost but good rows? Acceptable if it's the best source. Zero rows after \
a batch? Definitely kill it.
- **Sources are slices.** "upwork: dataset" and "upwork: lead list" are different \
harvesters. Don't make multiple harvesters for depth (page 1, page 2) — each \
harvester handles its own pagination.
- **Uploaded files are the candidates** when present. Don't spawn web harvesters \
if the user uploaded a CSV — create a harvester pointed at the file.

## Check-in Protocol

After each decision, set when you want the next check-in:
- Include "NEXT_CHECKIN: Xs" or "NEXT_CHECKIN: $X" in your response (time or cost)
- Be conservative — shorter intervals when uncertain, longer when things are stable
- Default: 60s if you don't specify

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
    V12 Orchestrator. Async check-in loop with automatic processing.
    Harvesters run in background, dispatcher feeds candidates to row
    generators round-robin, orchestrator checks in periodically.
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
        dedup_store: Any,
        save_row: Callable[..., Awaitable[Optional[str]]],
        dispatcher: CandidateDispatcher,
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
        self._dispatcher = dispatcher

        self._research_counter = 0
        self._harvester_counter = 0
        self._apollo_counter = 0
        self._sources: Dict[str, SourceState] = {}
        self._last_checkpoint_time: float = 0.0
        self._start_time: float = time.time()
        self._finish_requested: bool = False

        # Check-in state
        self._checkin_interval: float = 60.0  # seconds
        self._checkin_cost_trigger: float = 0.50  # dollars
        self._cost_at_last_checkin: float = 0.0
        self._force_checkin = asyncio.Event()

        registry = ToolRegistry()
        self._register_tools(registry)

        # Format system prompt
        columns_desc = self._format_columns(columns)
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

        from dsl_worker.config import settings
        max_turns = getattr(settings, 'orchestrator_max_turns', 40)

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
            reasoning={"effort": "high", "summary": "detailed"},
            label="orchestrator",
            continue_on_text=False,  # V12: we control the loop, not AgentConversation
            on_tool_call=on_tool_call,
            on_cost=on_cost,
            extra_tools=all_extra_tools,
        )

    # ── Formatting helpers ────────────────────────────────────────────

    def _format_columns(self, columns: List[Dict[str, Any]]) -> str:
        if not columns:
            return "(no columns defined)"
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
        return "\n".join(col_lines)

    def _format_apollo_tools_section(self) -> str:
        if not self.apollo_client:
            return ""
        return (
            "## Apollo.io\n\n"
            "Apollo is a B2B database — 210M+ professionals and 30M+ companies. "
            "Great for: business contacts, company lists, lead generation. "
            "Not useful for: non-business entities (artists, restaurants, etc.).\n\n"
            "**apollo_search(...)** — Search people. FREE, no credits. Results are "
            "auto-pushed to the processing pipeline.\n\n"
            "**apollo_search_companies(...)** — Search companies. Results auto-pushed.\n\n"
            "Apollo is free and fast — much cheaper than web harvesting. Prefer "
            "it for B2B projects.\n\n"
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
        lines = ["Uploaded files (create a harvester pointed at the file path):"]
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
            inspection = f.get("inspection")
            if inspection:
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

    # ── Dashboard ─────────────────────────────────────────────────────

    def _build_dashboard(self) -> str:
        """Build the dashboard string injected at each check-in."""
        rows_done = self._generation_stats.get("rows_generated", 0)
        total_skipped = self._generation_stats.get("skipped", 0)
        total_errors = self._generation_stats.get("errors", 0)
        elapsed = time.time() - self._start_time

        # Compute total cost across sources
        total_cost = 0.0
        for sid, state in self._sources.items():
            results = self._dispatcher.get_source_results(sid)
            total_cost += state.harvest_cost + results.process_cost

        avg_cpr = f"${total_cost / rows_done:.3f}" if rows_done > 0 else "N/A"

        lines = [
            "--- DASHBOARD ---",
            f"PROGRESS: {rows_done}/{self.num_samples} rows | "
            f"Cost: ${total_cost:.2f} | Avg: {avg_cpr}/row | "
            f"Elapsed: {elapsed:.0f}s",
            "",
        ]

        # Per-source table
        if self._sources:
            lines.append(
                f"{'SOURCE':<45} {'cand':>5} {'proc':>5} {'pend':>5} "
                f"{'rows':>5} {'skip':>5} {'dupe':>5} {'err':>4} {'$/row':>8} {'status'}"
            )
            for sid, state in self._sources.items():
                results = self._dispatcher.get_source_results(sid)
                total_src = state.harvest_cost + results.process_cost
                cpr = f"${total_src / results.rows:.3f}" if results.rows > 0 else "—"
                status = "exhausted" if state.exhausted else (
                    f"error: {state.error[:30]}" if state.error else "active"
                )
                lines.append(
                    f"  {sid} {state.source[:35]:<35} "
                    f"{state.total_harvested:>5} {results.processed:>5} {results.pending:>5} "
                    f"{results.rows:>5} {results.skipped:>5} {results.duplicates:>5} "
                    f"{results.errors:>4} {cpr:>8} {status}"
                )

            # Cost breakdown
            lines.append("")
            lines.append("COST BREAKDOWN:")
            for sid, state in self._sources.items():
                results = self._dispatcher.get_source_results(sid)
                total_src = state.harvest_cost + results.process_cost
                cpr = f"${total_src / results.rows:.3f}" if results.rows > 0 else "—"
                lines.append(
                    f"  {sid}: harvest=${state.harvest_cost:.3f} "
                    f"process=${results.process_cost:.3f} "
                    f"total=${total_src:.3f} → {cpr}/row"
                )

        # Outcomes delta (since last check-in)
        delta = self._dispatcher.get_results_delta()
        if delta:
            lines.append("")
            lines.append("OUTCOMES (since last check-in):")
            for sid, d in delta.items():
                parts = []
                if d.get("rows"): parts.append(f"+{d['rows']} rows")
                if d.get("skipped"): parts.append(f"+{d['skipped']} skipped")
                if d.get("duplicates"): parts.append(f"+{d['duplicates']} dupes")
                if d.get("errors"): parts.append(f"+{d['errors']} errors")
                if parts:
                    lines.append(f"  {sid}: {', '.join(parts)}")
                for reason in d.get("skip_reasons", [])[:3]:
                    lines.append(f"    skip: {reason[:120]}")

        # Pipeline
        lines.append("")
        lines.append(
            f"PIPELINE: {self._dispatcher.total_pending} pending | "
            f"{self._dispatcher.active_task_count}/10 generators active"
        )

        # Harvester reports
        has_reports = any(s.harvester_reports for s in self._sources.values())
        if has_reports:
            lines.append("")
            lines.append("HARVESTER REPORTS (latest):")
            for sid, state in self._sources.items():
                if state.harvester_reports:
                    report = state.harvester_reports[-1][:200]
                    lines.append(f"  {sid}: {report}")

        lines.append("--- END DASHBOARD ---")
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
                    "shell_exec", "list_files",
                ],
            )

        # --- Apollo tools ---
        if self.apollo_client:
            self._register_apollo_tools(registry)

        # --- create_harvester ---
        async def create_harvester(args: Dict) -> tuple[str, float]:
            source = args.get("source", "")
            candidate_description = args.get("candidate_description", "")

            if not source:
                return "Error: source is required", 0.0
            if not candidate_description:
                return "Error: candidate_description is required", 0.0

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
            self._dispatcher.add_source(source_id)

            # Start harvesting immediately in background
            task = asyncio.create_task(
                self._run_harvester_loop(source_id)
            )
            state.task = task
            self._maybe_checkpoint(force=True)

            return (
                f"Harvester {source_id} started for: {source}\n"
                f"Candidates will flow to row generators automatically."
            ), 0.0

        registry.add(
            name="create_harvester",
            description=(
                "Create AND start a harvester immediately. Candidates flow to "
                "row generators automatically. One harvester = one source slice."
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
                            "What candidates to look for. The entity, what data to extract, "
                            "any obvious surface-level dealbreakers."
                        ),
                    },
                },
                "required": ["source", "candidate_description"],
            },
            handler=create_harvester,
        )

        # --- stop_harvester ---
        async def stop_harvester(args: Dict) -> tuple[str, float]:
            source_id = args.get("source_id", "")
            reason = args.get("reason", "")

            state = self._sources.get(source_id)
            if not state:
                available = list(self._sources.keys())
                return f"Error: unknown source '{source_id}'. Available: {available}", 0.0

            # Cancel the harvester loop task
            if state.task and not state.task.done():
                state.task.cancel()
                try:
                    await asyncio.wait_for(asyncio.shield(state.task), timeout=5.0)
                except (asyncio.CancelledError, asyncio.TimeoutError, Exception):
                    pass

            # Close harvester resources (BU session, sandbox)
            if state.agent is not None:
                try:
                    await state.agent.close()
                except Exception as e:
                    logger.warning(f"[orchestrator] Error closing {source_id}: {e}")

            state.exhausted = True
            # Don't remove from dispatcher — let remaining buffer drain
            self._dispatcher.remove_source(source_id)

            results = self._dispatcher.get_source_results(source_id)
            total_cost = state.harvest_cost + results.process_cost
            cpr = f"${total_cost / results.rows:.3f}" if results.rows > 0 else "N/A"

            return (
                f"Stopped {source_id}: {reason}\n"
                f"  {results.rows} rows, {results.skipped} skipped, "
                f"{results.duplicates} dupes from {state.total_harvested} candidates\n"
                f"  Cost: ${total_cost:.3f} ({cpr}/row)"
            ), 0.0

        registry.add(
            name="stop_harvester",
            description=(
                "Stop a harvester. Remaining buffered candidates still get processed."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "source_id": {
                        "type": "string",
                        "description": "Which source to stop (e.g. 'harvest:0')",
                    },
                    "reason": {
                        "type": "string",
                        "description": "Why you're stopping this source",
                    },
                },
                "required": ["source_id"],
            },
            handler=stop_harvester,
        )

        # --- inspect ---
        async def inspect(args: Dict) -> tuple[str, float]:
            source_id = args.get("source_id", "")
            candidate_idx = args.get("candidate")
            step_idx = args.get("step")

            state = self._sources.get(source_id)
            if not state:
                return f"Error: unknown source '{source_id}'", 0.0

            lines = []

            if candidate_idx is not None:
                # TODO: implement candidate/row-gen drill-down
                # For now, show what we have
                lines.append(f"Candidate {candidate_idx} detail not yet implemented.")
                lines.append("Use the dashboard for per-source outcomes.")
            elif step_idx is not None and state.agent:
                # Show a specific harvester step
                msgs = state.agent._conversation.messages
                if 0 <= step_idx < len(msgs):
                    msg = msgs[step_idx]
                    lines.append(f"Step {step_idx}:")
                    lines.append(json.dumps(msg, indent=2, default=str)[:2000])
                else:
                    lines.append(f"Step {step_idx} not found ({len(msgs)} total)")
            elif state.agent:
                # Show harvester overview: tool calls with cost/time
                msgs = state.agent._conversation.messages
                lines.append(f"Harvester {source_id}: {len(msgs)} messages, "
                             f"{state.batches} batches, ${state.harvest_cost:.3f}")
                lines.append("")
                for i, msg in enumerate(msgs):
                    if isinstance(msg, dict):
                        role = msg.get("role", "?")
                        # Summarize tool calls
                        if role == "assistant":
                            content = msg.get("content", "")
                            if isinstance(content, str):
                                lines.append(f"  [{i}] {role}: {content[:150]}")
                        elif "tool_call_id" in msg or role == "tool":
                            content = msg.get("content", "")
                            if isinstance(content, str):
                                lines.append(f"  [{i}] tool: {content[:150]}")
                    if len(lines) > 50:
                        lines.append(f"  ... ({len(msgs) - i - 1} more messages)")
                        break
            else:
                # Apollo or no-agent source
                results = self._dispatcher.get_source_results(source_id)
                lines.append(f"Source {source_id}: {state.source}")
                lines.append(f"  {results.rows} rows, {results.skipped} skipped, "
                             f"{results.duplicates} dupes")
                if results.skip_reasons:
                    lines.append("  Recent skip reasons:")
                    for r in results.skip_reasons[-5:]:
                        lines.append(f"    - {r[:120]}")

            return "\n".join(lines), 0.0

        registry.add(
            name="inspect",
            description=(
                "Drill down into a source for more context. Shows harvester "
                "tool calls, candidate outcomes, row generator details."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "source_id": {
                        "type": "string",
                        "description": "Which source to inspect",
                    },
                    "candidate": {
                        "type": "integer",
                        "description": "Specific candidate index to inspect (optional)",
                    },
                    "step": {
                        "type": "integer",
                        "description": "Specific harvester step index to inspect (optional)",
                    },
                },
                "required": ["source_id"],
            },
            handler=inspect,
        )

        # --- finish ---
        async def finish(args: Dict) -> tuple[str, float]:
            reason = args.get("reason", "")
            self._finish_requested = True
            return f"Finishing: {reason}", 0.0

        registry.add(
            name="finish",
            description="End the job early. Use for budget concerns or enough quality rows.",
            parameters={
                "type": "object",
                "properties": {
                    "reason": {
                        "type": "string",
                        "description": "Why you're ending early",
                    },
                },
            },
            handler=finish,
        )

    # ── Apollo tools ──────────────────────────────────────────────────

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

            # Build candidates and push to dispatcher
            candidates = []
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
                candidates.append(Candidate(
                    values=candidate_data,
                    source_id=source_id,
                    source_context="Apollo contact search",
                    metadata={"origin": "apollo"},
                ))

            state.total_harvested += len(candidates)

            # Push to dispatcher (no backpressure for Apollo — instant results)
            self._dispatcher.submit_batch(source_id, candidates)

            if total > 0:
                total_pages = min((total + 99) // 100, 500)
                pagination_info = f"page {page}/{total_pages} ({total:,} total matches)"
            else:
                pagination_info = (
                    f"page {page} ({len(people)} returned). Try next page for more."
                    if len(people) >= 100 else
                    f"page {page} ({len(people)} returned)"
                )

            return (
                f"Apollo search: {len(people)} people, {pagination_info}.\n"
                f"Source: {source_id} — candidates auto-pushed to processing.\n"
                f"For more: apollo_search(source_id='{source_id}', page={page + 1})."
            ), 0.0

        registry.add(
            name="apollo_search",
            description=(
                "Search Apollo.io's 210M+ contact database. FREE — no credits. "
                "Results auto-pushed to processing pipeline."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "person_titles": {
                        "type": "array", "items": {"type": "string"},
                        "description": "Job titles (e.g. ['CEO', 'VP Marketing'])",
                    },
                    "person_seniorities": {
                        "type": "array", "items": {"type": "string"},
                        "description": "Seniority: c_suite, founder, vp, director, manager, senior, entry",
                    },
                    "person_locations": {
                        "type": "array", "items": {"type": "string"},
                        "description": "Person locations (e.g. ['California, US'])",
                    },
                    "person_names": {
                        "type": "array", "items": {"type": "string"},
                        "description": "Search by individual names",
                    },
                    "contact_email_status": {
                        "type": "array", "items": {"type": "string"},
                        "description": "Email availability: 'verified', 'guessed', 'unavailable'",
                    },
                    "department_ids": {
                        "type": "array", "items": {"type": "string"},
                        "description": "Department classification IDs",
                    },
                    "include_similar_titles": {
                        "type": "boolean",
                        "description": "Include similar/related job titles",
                    },
                    "organization_keywords": {
                        "type": "array", "items": {"type": "string"},
                        "description": "Industry/keyword tags (e.g. ['healthcare', 'fintech'])",
                    },
                    "organization_name": {
                        "type": "string",
                        "description": "Company name search (partial match)",
                    },
                    "organization_locations": {
                        "type": "array", "items": {"type": "string"},
                        "description": "Company HQ locations",
                    },
                    "organization_not_locations": {
                        "type": "array", "items": {"type": "string"},
                        "description": "Exclude companies in these locations",
                    },
                    "employee_ranges": {
                        "type": "array", "items": {"type": "string"},
                        "description": "Employee count: '1-10', '11-50', '51-200', '201-500', '501-1000', etc.",
                    },
                    "organization_ids": {
                        "type": "array", "items": {"type": "string"},
                        "description": "Apollo organization IDs",
                    },
                    "organization_domains": {
                        "type": "array", "items": {"type": "string"},
                        "description": "Company domains (e.g. ['apollo.io'])",
                    },
                    "revenue_ranges": {
                        "type": "array", "items": {"type": "string"},
                        "description": "Annual revenue ranges",
                    },
                    "industry_tag_ids": {
                        "type": "array", "items": {"type": "string"},
                        "description": "Industry category IDs",
                    },
                    "technology_uids": {
                        "type": "array", "items": {"type": "string"},
                        "description": "Technology stack UIDs",
                    },
                    "q_keywords": {
                        "type": "string",
                        "description": "Free text keyword search",
                    },
                    "source_id": {
                        "type": "string",
                        "description": "Append to existing Apollo source (for pagination)",
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

            candidates = []
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
                candidates.append(Candidate(
                    values=candidate_data,
                    source_id=source_id,
                    source_context="Apollo company search",
                    metadata={"origin": "apollo"},
                ))

            state.total_harvested += len(candidates)
            self._dispatcher.submit_batch(source_id, candidates)

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
                f"Source: {source_id} — candidates auto-pushed to processing.\n"
                f"For more: apollo_search_companies(source_id='{source_id}', page={page + 1})."
            ), 0.0

        registry.add(
            name="apollo_search_companies",
            description=(
                "Search Apollo.io's 30M+ company database. Results auto-pushed "
                "to processing pipeline."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "keywords": {"type": "array", "items": {"type": "string"}, "description": "Industry/keyword tags"},
                    "name": {"type": "string", "description": "Company name search"},
                    "locations": {"type": "array", "items": {"type": "string"}, "description": "Company HQ locations"},
                    "not_locations": {"type": "array", "items": {"type": "string"}, "description": "Exclude locations"},
                    "employee_ranges": {"type": "array", "items": {"type": "string"}, "description": "Employee count ranges"},
                    "revenue_ranges": {"type": "array", "items": {"type": "string"}, "description": "Revenue ranges"},
                    "funding_stages": {"type": "array", "items": {"type": "string"}, "description": "Funding stage codes"},
                    "technology_uids": {"type": "array", "items": {"type": "string"}, "description": "Tech stack UIDs"},
                    "website_urls": {"type": "array", "items": {"type": "string"}, "description": "Filter by website URLs"},
                    "industry_tag_ids": {"type": "array", "items": {"type": "string"}, "description": "Industry IDs"},
                    "founded_year_min": {"type": "integer", "description": "Earliest founding year"},
                    "founded_year_max": {"type": "integer", "description": "Latest founding year"},
                    "publicly_traded": {"type": "boolean", "description": "Publicly traded only"},
                    "source_id": {"type": "string", "description": "Append to existing source (pagination)"},
                    "page": {"type": "integer", "description": "Page number (1-500)"},
                },
            },
            handler=apollo_search_companies,
        )

    # ── Harvester background loop ─────────────────────────────────────

    async def _run_harvester_loop(self, source_id: str) -> None:
        """Background task: run harvester in batch loop with backpressure."""
        state = self._sources.get(source_id)
        if not state or not state.agent:
            return

        harvester = state.agent
        is_first = True

        try:
            while not harvester.exhausted:
                # Check stop
                if self.stop_checker and self.stop_checker():
                    break
                rows_done = self._generation_stats.get("rows_generated", 0)
                if rows_done >= self.num_samples:
                    break

                # Run one batch
                msg = (
                    "Begin harvesting candidates from your assigned source."
                    if is_first else
                    "Get the next batch of candidates."
                )
                is_first = False

                try:
                    candidates, report = await harvester.run_batch(msg)
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    state.error = str(e)[:200]
                    logger.error(f"[orchestrator] Harvester {source_id} error: {e}")
                    break

                cost_delta = harvester.batch_cost_delta
                state.total_harvested += len(candidates)
                state.harvest_cost += cost_delta
                state.batches += 1
                state.exhausted = harvester.exhausted
                if report:
                    state.harvester_reports.append(report[:500])
                    # Cap report history
                    if len(state.harvester_reports) > 10:
                        state.harvester_reports = state.harvester_reports[-5:]

                self._maybe_checkpoint()

                if not candidates:
                    if state.exhausted:
                        break
                    continue

                # Push to dispatcher and wait for backpressure
                token = self._dispatcher.submit_batch(source_id, candidates)
                try:
                    await token.event.wait()
                except asyncio.CancelledError:
                    break

        except asyncio.CancelledError:
            pass
        except Exception as e:
            state.error = str(e)[:200]
            logger.error(f"[orchestrator] Harvester loop {source_id} died: {e}")

        state.exhausted = True
        logger.info(
            f"[orchestrator] Harvester {source_id} done: "
            f"{state.total_harvested} candidates, {state.batches} batches, "
            f"${state.harvest_cost:.3f}"
        )

        # Force check-in if all sources are now dead
        all_dead = all(s.exhausted for s in self._sources.values())
        if all_dead and self._dispatcher.total_pending == 0:
            self._force_checkin.set()

    # ── Check-in timing ───────────────────────────────────────────────

    async def _wait_for_checkin(self) -> None:
        """Sleep until next check-in trigger."""
        rows_done = self._generation_stats.get("rows_generated", 0)

        if rows_done == 0:
            # Pre-first-row: tight intervals
            interval = 60.0
            cost_trigger = 0.50
        else:
            # Post-first-row: orchestrator-set intervals
            interval = self._checkin_interval
            cost_trigger = self._checkin_cost_trigger

        start_cost = self._get_total_cost()
        deadline = asyncio.get_event_loop().time() + interval

        while asyncio.get_event_loop().time() < deadline:
            # Structural events force immediate check-in
            if self._force_checkin.is_set():
                self._force_checkin.clear()
                return
            if self._should_stop():
                return
            # Target reached
            if self._generation_stats.get("rows_generated", 0) >= self.num_samples:
                return
            # Cost trigger (pre-first-row only, or if set)
            if cost_trigger and (self._get_total_cost() - start_cost) >= cost_trigger:
                return
            await asyncio.sleep(2.0)

    def _get_total_cost(self) -> float:
        """Current total cost across all sources."""
        total = 0.0
        for sid, state in self._sources.items():
            results = self._dispatcher.get_source_results(sid)
            total += state.harvest_cost + results.process_cost
        return total

    def _should_stop(self) -> bool:
        if self.stop_checker and self.stop_checker():
            return True
        return False

    def _should_exit(self) -> bool:
        if self._finish_requested:
            return True
        rows_done = self._generation_stats.get("rows_generated", 0)
        if rows_done >= self.num_samples:
            return True
        if self._should_stop():
            return True
        # All sources dead AND nothing in pipeline
        all_dead = all(s.exhausted for s in self._sources.values()) if self._sources else False
        if all_dead and self._dispatcher.total_pending == 0 and self._dispatcher.active_task_count == 0:
            return True
        return False

    def _parse_checkin_interval(self, text: str) -> None:
        """Parse NEXT_CHECKIN from orchestrator's response."""
        import re
        match = re.search(r'NEXT_CHECKIN:\s*(\d+)\s*s', text, re.IGNORECASE)
        if match:
            self._checkin_interval = max(15.0, min(300.0, float(match.group(1))))
            return
        match = re.search(r'NEXT_CHECKIN:\s*\$?([\d.]+)', text, re.IGNORECASE)
        if match:
            self._checkin_cost_trigger = max(0.10, float(match.group(1)))

    # ── Run ───────────────────────────────────────────────────────────

    async def run(self) -> AgentResult:
        """V12 async check-in loop."""

        # Start dispatcher
        dispatcher_task = asyncio.create_task(self._dispatcher.run())

        try:
            # Determine initial message
            if self._conversation.messages:
                # Resuming from checkpoint
                logger.info(
                    f"[orchestrator] Resuming: {len(self._conversation.messages)} messages, "
                    f"{self._generation_stats.get('rows_generated', 0)}/{self.num_samples} rows"
                )
                # Restart harvester loops for non-exhausted sources
                for sid, state in self._sources.items():
                    if state.agent and not state.exhausted and (not state.task or state.task.done()):
                        state.task = asyncio.create_task(
                            self._run_harvester_loop(sid)
                        )
                dashboard = self._build_dashboard()
                initial_msg = f"RESUMED. Here is the current state:\n\n{dashboard}"
            elif self.feedback_context:
                initial_msg = (
                    "Begin. The user reviewed previous results and gave feedback "
                    "(shown in system prompt). Research as needed and design a new pipeline."
                )
            elif self.resume_context:
                rc = self.resume_context
                initial_msg = (
                    f"RESUME. {rc['rows_generated']}/{rc['target']} rows done "
                    f"({rc['remaining']} remaining). Prior cost: ${rc['prior_cost_usd']:.4f}.\n"
                    f"Pick up where the previous run left off."
                )
            else:
                initial_msg = (
                    "Begin. Read the conversation history and resources, reason about "
                    "strategy, then start creating harvesters."
                )

            # First check-in
            result = await self._conversation.send(initial_msg)
            if result and result.text:
                self._parse_checkin_interval(result.text)

            # Check-in loop
            while not self._should_exit():
                await self._wait_for_checkin()

                if self._should_exit():
                    break

                dashboard = self._build_dashboard()
                result = await self._conversation.send(dashboard)

                if result and result.text:
                    self._parse_checkin_interval(result.text)

                self._maybe_checkpoint()

            logger.info(
                f"[orchestrator] Exiting: "
                f"{self._generation_stats.get('rows_generated', 0)}/{self.num_samples} rows"
            )
            return result or AgentResult(text="Orchestrator finished.")

        finally:
            # Stop dispatcher and all harvesters
            self._dispatcher.stop()
            try:
                await asyncio.wait_for(dispatcher_task, timeout=30.0)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                dispatcher_task.cancel()

    @property
    def cost_usd(self) -> float:
        return self._conversation.total_cost

    def export_state(self) -> Dict[str, Any]:
        """Export full pipeline state for checkpointing."""
        sources = []
        for sid, state in list(self._sources.items()):
            source_data = {
                "id": state.id,
                "source": state.source,
                "description": state.description,
                "total_harvested": state.total_harvested,
                "harvest_cost": state.harvest_cost,
                "batches": state.batches,
                "exhausted": state.exhausted,
                "harvester_reports": state.harvester_reports[-5:],
                "error": state.error,
                "harvester_conversation": None,
                "harvester_candidates_total": 0,
                "harvester_exhausted": False,
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
            "dispatcher_state": self._dispatcher.export_state(),
        }

    def restore_state(self, state: Dict[str, Any]) -> None:
        """Restore pipeline state from checkpoint. Call BEFORE run()."""
        conv = state.get("orchestrator_conversation")
        if conv:
            self._conversation.messages = conv["messages"]
            self._conversation.total_cost = conv.get("total_cost", 0.0)
            self._conversation.total_turns = conv.get("total_turns", 0)
            logger.info(
                f"[orchestrator] Restored conversation: "
                f"{len(conv['messages'])} messages"
            )

        self._harvester_counter = state.get("harvester_counter", 0)
        self._apollo_counter = state.get("apollo_counter", 0)
        self._research_counter = state.get("research_counter", 0)

        saved_stats = state.get("generation_stats", {})
        for key in ("skipped", "errors", "total_cost"):
            if key in saved_stats:
                self._generation_stats[key] = saved_stats[key]

        # Restore dispatcher results
        dispatcher_state = state.get("dispatcher_state")
        if dispatcher_state:
            self._dispatcher.restore_results(dispatcher_state)

        # Restore sources and recreate harvesters
        for src in state.get("sources", []):
            agent = None
            harvester_conv = src.get("harvester_conversation")
            if harvester_conv and not src.get("exhausted", False):
                try:
                    from dsl_worker.agents.harvester import HarvesterAgent
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
                    agent._conversation.messages = harvester_conv["messages"]
                    agent._conversation.total_cost = harvester_conv.get("total_cost", 0.0)
                    agent._conversation.total_turns = harvester_conv.get("total_turns", 0)
                    agent._candidates_total = src.get("harvester_candidates_total", 0)
                    agent._exhausted = src.get("harvester_exhausted", False)
                    agent._bu_session_id = None  # BU sessions die on pause

                    logger.info(f"[orchestrator] Restored harvester for {src['id']}")
                except Exception as e:
                    logger.warning(f"[orchestrator] Failed to restore harvester {src['id']}: {e}")
                    agent = None

            source_state = SourceState(
                id=src["id"],
                source=src["source"],
                description=src["description"],
                agent=agent,
                total_harvested=src.get("total_harvested", 0),
                harvest_cost=src.get("harvest_cost", 0.0),
                batches=src.get("batches", 0),
                exhausted=src.get("exhausted", False),
                harvester_reports=src.get("harvester_reports", []),
                error=src.get("error", ""),
            )
            self._sources[src["id"]] = source_state

            # Register source with dispatcher
            if not source_state.exhausted:
                self._dispatcher.add_source(src["id"])

            logger.info(
                f"[orchestrator] Restored source {src['id']}: "
                f"{src.get('total_harvested', 0)} harvested, "
                f"{'exhausted' if src.get('exhausted') else 'active'}"
            )

    async def cleanup(self) -> None:
        """Cancel all harvester tasks and close resources."""
        # Cancel harvester loop tasks
        for sid, state in list(self._sources.items()):
            if state.task and not state.task.done():
                state.task.cancel()
        # Wait for cancellation
        tasks = [s.task for s in self._sources.values() if s.task and not s.task.done()]
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

        # Close harvester resources
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
