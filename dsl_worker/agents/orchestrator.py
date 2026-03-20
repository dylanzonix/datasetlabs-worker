"""
Orchestrator agent — coordinates dataset generation.

V10: Event-driven, multi-armed bandit. The orchestrator is a reactive strategist:
- harvest() is non-blocking — spawns harvesters in background
- No set_instructions — row generators use conversation context directly
- No done() — system auto-exits when target reached or all sources exhausted
- Events (source_exhausted, milestone, stall, etc.) auto-inject as messages
  via the on_idle callback in AgentConversation

Tools:
- research(question, ...) — spawn research subagent
- harvest(source, description) — spawn a harvester (non-blocking)
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
from dsl_worker.infra.candidate_pool import CandidatePool, StrategyMonitor

logger = logging.getLogger(__name__)

HARVESTER_WALL_CLOCK_TIMEOUT = 600  # seconds — max time for a single harvester


ORCHESTRATOR_SYSTEM_PROMPT = """\
# Dataset Generation Orchestrator

A user described a dataset they want. You coordinate its production.

## Your Role

You are a strategist. You dispatch harvesters to find candidates from sources, \
and the system automatically processes them into dataset rows. You do NOT manage \
quotas, throttling, or operational details — the system handles that via \
cost-optimized source sampling (Thompson Sampling).

## Your Subagents

**explore_agent(task)** — Recon agent for files, data, and connected resources. \
Uses code execution to inspect uploaded files, parse schemas, analyze data. \
No web access. Use this to understand what you already have.

**web_search_agent(task)** — Web research agent. Browses websites, searches \
the web. Use this to understand the landscape of potential sources online.

**harvester_agent(source, description)** — Start a harvester for one source slice. \
Non-blocking — returns immediately. The harvester navigates the source \
and submits ALL items it finds as candidates. Do NOT put filtering criteria \
in the description — filtering happens downstream in row generators.

## Candidate Scope

Pay attention to the user's intended candidate scope — where candidates \
live varies by project.

Examples:
- **Enrichment**: User uploads a spreadsheet of hospitals and wants director \
contacts added. The spreadsheet rows ARE the candidates — one harvester on the \
file, done. Don't spawn web harvesters to find more hospitals.
- **Single-source extraction**: User wants product listings from Amazon. \
Candidates are on Amazon — harvest search pages on that site. Don't also \
harvest eBay or Walmart unless asked.
- **Open discovery**: User wants a list of craft breweries in the midwest. \
No source given — cast a wide net across directories, search engines, \
industry databases. This is where many parallel harvesters shine.

## Workflow

1. **Recon** (if needed) — Use explore_agent for files/data, web_search_agent \
for web. Not every project needs both. A file enrichment job may only need \
explore_agent to inspect the schema. A web extraction job may not need recon \
at all if the source is obvious.
   - Recon gives you a plan. Harvesting gives you data.
   - Don't wait for perfect information — launch harvesters early and learn \
from their results. The Thompson Sampling system tells you which sources work.

2. **Harvest** — Call harvester_agent() for each source slice.

3. **React to Events** — After dispatching, you'll receive status updates:
   - **source_exhausted**: A harvester finished. Consider new sources/search terms.
   - **fertility_shift**: A source's success rate changed significantly.
   - **milestone**: Progress checkpoint (25%/50%/75%/100%).
   - **stall**: No successful rows recently. Diagnose and adjust strategy.
   - **pool_empty**: All candidates consumed but more rows needed.
   - **target_reached**: Done! System exits automatically.

   When you have nothing to do, just say so and wait for the next event.

## Strategy

- Deduplication is cheap and reliable — overlap between sources is fine.
- Broad search terms > narrow ones.
- The system automatically favors sources with better success rates. \
  You just need to FIND good sources.
- You'll see metrics: fertility rate (% candidates → rows) and cost/row per source.
- If a source is underperforming, try different search terms or a new source entirely.

## Principles

- ONE harvester_agent() = ONE source slice. Each spawns its own navigator.
- harvester_agent() is non-blocking — returns immediately. Don't wait for results.
- Row generators see the full user conversation — they understand the task.
- The browsing stack handles anti-bot, CAPTCHAs, and JS-heavy pages automatically.
- Row generators will skip_row() for dead ends — some rejection is normal.

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

- explore_agent(task, budget): Inspect files, data, integrations. No web access.
- web_search_agent(task, budget): Research the web. Browse sites, search engines.
- harvester_agent(source, description): Start a harvester. Non-blocking. \
  source: URL, file path, search query, or topic. \
  description: what kind of candidates to find.

{feedback_section}
"""


class OrchestratorAgent:
    """
    V10 Orchestrator. Event-driven, multi-armed bandit source allocation.

    - harvest() is non-blocking (spawns background task)
    - Events auto-inject via on_idle callback
    - No set_instructions, no done, no quotas
    - Auto-exits when target reached or all sources exhausted
    """

    def __init__(
        self,
        chat_history: List[Dict[str, str]],
        columns: List[Dict[str, Any]],
        num_samples: int,
        openai_client: TrackedOpenAIClient,
        model: str,
        workspace_dir: Path,
        candidate_pool: CandidatePool,
        strategy_monitor: StrategyMonitor,
        generation_stats: Dict[str, Any],
        uploaded_files: Optional[List[Dict[str, Any]]] = None,
        bu_client: Optional[Any] = None,
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
        harvester_model: str = "",
    ) -> None:
        self.feedback_context = feedback_context
        self.chat_history = chat_history
        self.columns = columns
        self.num_samples = num_samples
        self.workspace_dir = Path(workspace_dir)
        self.openai_client = openai_client
        self.model = model
        self.bu_client = bu_client
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
        self.uploaded_files = uploaded_files
        self.mcp_tools = mcp_tools or []
        self.on_browser_started = on_browser_started
        self.on_browser_stopped = on_browser_stopped
        self.langfuse_parent = langfuse_parent
        self.harvester_model = harvester_model or model

        self._pool = candidate_pool
        self._monitor = strategy_monitor
        self._generation_stats = generation_stats

        self._research_counter = 0
        self._harvester_counter = 0
        self._active_harvesters = 0
        self._harvester_tasks: List[asyncio.Task] = []

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
            on_tool_call=on_tool_call,
            on_cost=on_cost,
            extra_tools=self.mcp_tools,
            langfuse_parent=langfuse_parent,
            on_idle=self._on_idle,
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

    # ── on_idle: event injection ─────────────────────────────────────

    async def _on_idle(self) -> Optional[str]:
        """
        Called when the orchestrator outputs text with no tool calls.
        Blocks until a strategic event arrives, then returns it as a string.
        Returns None to signal the conversation loop to exit.
        """
        # Check stop/pause immediately
        if self.stop_checker and self.stop_checker():
            return None

        # Check if target already reached
        rows_done = self._generation_stats.get("rows_generated", 0)
        if rows_done >= self.num_samples:
            logger.info(f"[orchestrator] Target reached: {rows_done}/{self.num_samples}")
            return None

        # Check if nothing to wait for
        if self._active_harvesters == 0 and self._pool.total_pending == 0:
            if self._pool._is_fully_drained():
                logger.info("[orchestrator] All sources exhausted, pool drained")
                return None

        # Poll for events in short intervals so pause/stop is detected quickly
        elapsed = 0.0
        max_wait = 120.0
        poll_interval = 2.0

        while elapsed < max_wait:
            if self.stop_checker and self.stop_checker():
                return None

            event = await self._monitor.wait_for_event(timeout=poll_interval)
            if event is not None:
                return event.message

            elapsed += poll_interval

            # Re-check exit conditions each poll
            rows_done = self._generation_stats.get("rows_generated", 0)
            if rows_done >= self.num_samples:
                return None
            if self._active_harvesters == 0 and self._pool._is_fully_drained():
                return None

        # Full timeout — inject status update
        status = self._pool.format_status()
        return f"Status update (no events in 2 minutes):\n{status}"

    # ── Tool registration ────────────────────────────────────────────

    def _register_tools(self, registry: ToolRegistry) -> None:

        # --- web_search_agent ---
        async def web_search_agent(args: Dict) -> tuple[str, float]:
            from dsl_worker.agents.research import ResearchAgent
            from dsl_worker.config import settings as worker_settings

            task = args.get("task", "")
            budget = args.get("budget", 8)

            if not task:
                return "Error: task is required", 0.0

            langfuse_span = getattr(self._conversation, "_current_langfuse_span", None)

            agent = ResearchAgent(
                openai_client=self.openai_client,
                model=worker_settings.research_subagent_model,
                workspace_dir=self.workspace_dir,
                bu_client=self.bu_client,
                sandbox=self.sandbox,
                stop_checker=self.stop_checker,
                max_turns=budget,
                tool_budget=budget,
                blob_service_client=self.blob_service_client,
                project_id=self.project_id,
                uploaded_file_urls=self.uploaded_file_urls,
            )
            if langfuse_span:
                agent._conversation.langfuse_parent = langfuse_span

            try:
                result = await agent.ask_full(task)
            finally:
                await agent.cleanup()

            if self.on_cost and result.cost_usd > 0:
                await self.on_cost(result.cost_usd, "web_search_agent")

            n = self._research_counter
            self._research_counter += 1
            research_dir = self.workspace_dir / "research"
            research_dir.mkdir(exist_ok=True)
            try:
                (research_dir / f"finding_{n}.md").write_text(
                    f"# Web Search: {task}\n\n{result.text}", encoding="utf-8"
                )
            except Exception as e:
                logger.warning(f"Failed to save research finding: {e}")

            return f"[Saved to research/finding_{n}.md]\n\n{result.text}", 0.0

        registry.add(
            name="web_search_agent",
            description=(
                "Web research agent. Browses websites and searches the web. "
                "Returns findings. Call multiple in one response for parallel research."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "task": {"type": "string", "description": "What to research on the web"},
                    "budget": {
                        "type": "integer",
                        "description": "Max tool calls (default 8). 4 for quick, 12 for deep.",
                    },
                },
                "required": ["task"],
            },
            handler=web_search_agent,
        )

        # --- explore_agent ---
        async def explore_agent(args: Dict) -> tuple[str, float]:
            from dsl_worker.agents.code_exec import CodeExecAgent
            from dsl_worker.config import settings as worker_settings

            task = args.get("task", "")
            budget = args.get("budget", 6)

            if not task:
                return "Error: task is required", 0.0

            langfuse_span = getattr(self._conversation, "_current_langfuse_span", None)

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
            if langfuse_span:
                agent._conversation.langfuse_parent = langfuse_span

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

        # --- harvester_agent (non-blocking) ---
        async def harvester_agent(args: Dict) -> tuple[str, float]:
            source = args.get("source", "")
            description = args.get("description", "")

            if not source:
                return "Error: source is required", 0.0

            idx = self._harvester_counter
            self._harvester_counter += 1
            source_id = f"harvest:{idx}"

            # Register source arm in the pool
            label = source[:80]
            self._pool.register_source(source_id, label=label)

            # Spawn harvester in background
            self._active_harvesters += 1
            task = asyncio.create_task(
                self._run_harvester(source, description, source_id, idx)
            )
            self._harvester_tasks.append(task)

            return (
                f"Harvester {idx} started for: {source}\n"
                f"Description: {description or '(auto)'}\n"
                f"Candidates will flow into the pool as they're found."
            ), 0.0

        registry.add(
            name="harvester_agent",
            description=(
                "Start a harvester for one source slice. Non-blocking — returns "
                "immediately. Call multiple in parallel for different sources."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "source": {
                        "type": "string",
                        "description": (
                            "One source to harvest: URL, file path, search query, or topic."
                        ),
                    },
                    "description": {
                        "type": "string",
                        "description": (
                            "What kind of items are on this source (e.g. 'job listings', "
                            "'company profiles'). Do NOT include filtering criteria — "
                            "filtering happens downstream automatically."
                        ),
                    },
                },
                "required": ["source"],
            },
            handler=harvester_agent,
        )

    # ── Harvester lifecycle ──────────────────────────────────────────

    async def _run_harvester(
        self,
        source: str,
        description: str,
        source_id: str,
        idx: int,
    ) -> None:
        """
        Run a harvester in the background. When done, fires source_exhausted event.
        """
        from dsl_worker.agents.harvester import HarvesterAgent

        langfuse_span = getattr(self._conversation, "_current_langfuse_span", None)

        harvester = HarvesterAgent(
            source=source,
            description=description,
            source_id=source_id,
            openai_client=self.openai_client,
            model=self.harvester_model,
            workspace_dir=self.workspace_dir,
            pool=self._pool,
            harvester_index=idx,
            bu_client=self.bu_client,
            sandbox=self.sandbox,
            stop_checker=self.stop_checker,
            blob_service_client=self.blob_service_client,
            project_id=self.project_id,
            on_tool_call=self.on_tool_call,
            on_cost=self.on_cost,
            uploaded_file_urls=self.uploaded_file_urls,
            uploaded_files=self.uploaded_files,
            mcp_tools=self.mcp_tools,
            langfuse_parent=langfuse_span,
        )

        t0 = time.time()
        try:
            await asyncio.wait_for(
                harvester.run(),
                timeout=HARVESTER_WALL_CLOCK_TIMEOUT,
            )
        except asyncio.TimeoutError:
            logger.warning(
                f"[orchestrator] Harvester {idx} wall-clock timeout "
                f"({HARVESTER_WALL_CLOCK_TIMEOUT}s)"
            )
        except Exception as e:
            logger.error(f"[orchestrator] Harvester {idx} error: {e}", exc_info=True)
        finally:
            await harvester.cleanup()

        cost = harvester.cost_usd
        self._pool.add_production_cost(source_id, cost)

        elapsed = time.time() - t0
        logger.info(
            f"[orchestrator] Harvester {idx} done: "
            f"{harvester.candidates_submitted} candidates, "
            f"{elapsed:.0f}s, ${cost:.3f}"
        )

        self._active_harvesters -= 1
        await self._monitor.on_source_done(source_id)

    # ── Run ──────────────────────────────────────────────────────────

    async def run(self) -> AgentResult:
        if self.feedback_context:
            message = (
                "Begin. The user reviewed previous results and gave feedback "
                "(shown in system prompt). Research as needed and design a new pipeline."
            )
        else:
            message = (
                "Begin. Read the conversation history and resources, reason about "
                "strategy, then dispatch harvesters for candidate sources."
            )

        def _should_exit() -> bool:
            rows_done = self._generation_stats.get("rows_generated", 0)
            if rows_done >= self.num_samples:
                logger.info(
                    f"[orchestrator] Auto-done: {rows_done}/{self.num_samples} rows generated"
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
        # Cancel any remaining harvester tasks
        for task in self._harvester_tasks:
            if not task.done():
                task.cancel()
                try:
                    await task
                except (asyncio.CancelledError, Exception):
                    pass
