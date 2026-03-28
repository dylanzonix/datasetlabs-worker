"""
Harvester agent — produces candidates from sources in batches.

V11: Batch-oriented iterator controlled by the orchestrator.
- Each run_batch() call runs the agent loop until it produces a text response
  (natural batch boundary) or calls done() (source exhausted).
- BU sessions are reused between batches via keep_alive + session_id.
- Candidates are buffered internally, not pushed to a shared pool.
- The orchestrator decides when to harvest more, process, or close.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from dsl_worker.agents.base import AgentConversation, AgentResult
from dsl_worker.agents.tools import ToolRegistry
from dsl_worker.billing.tracked_client import TrackedOpenAIClient
from dsl_worker.infra.bu_client import BUClient
from dsl_worker.infra.candidate_pool import Candidate

logger = logging.getLogger(__name__)


HARVESTER_SYSTEM_PROMPT = """\
# Source Harvester

You are a harvester in a dataset generation pipeline. Your job is to collect \
candidates from the assigned source. Downstream, row generators will validate \
each candidate and turn it into a dataset row — that's not your job.

## Your Source

<source>
{source}
</source>

## What to Look For

<candidate_description>
{description}
</candidate_description>

## Your Role in the Pipeline

You are the collector, not the researcher. Your job is to find and submit \
candidates as fast as possible. A candidate can be as simple as a company name, \
a person's name, a URL — whatever identifies the entity. Row generators \
downstream will do all the deep research, enrichment, and validation.

**Yield generously.** If something looks like it could be a valid candidate, \
submit it. It's worse to miss a good candidate than to submit a borderline \
one. Only skip things that are obviously wrong at a glance.

**Don't research individual candidates.** Don't look up emails, phone numbers, \
LinkedIn profiles, or details for individual items. Don't verify if a company \
qualifies. Just grab the name/identifier and move on. All of that happens \
downstream in row generators which have enrichment tools (Apollo, web search).

## How to Work

1. Navigate to the source and extract candidate lists.
2. Use **web search** (built-in) for finding the right URLs, discovering \
list pages, or checking what a source offers.
3. Use **browse(url, task)** for navigating actual web pages — extracting \
lists from JS-heavy sites, paginating, scrolling, bypassing anti-bot. \
browse() launches a full cloud browser with stealth, proxy, and captcha solving.
4. Submit each candidate with whatever data is visible on the list page. \
Even just a company name is enough — row generators handle the rest.
5. After extracting what's available, respond with a short report.
6. Call done() only when the source is fully exhausted.

## Rules

- Submit candidates quickly. A name + any visible context is sufficient.
- Do NOT research individual candidates (no email lookups, no phone lookups, \
no LinkedIn searches per candidate). That's the row generator's job.
- Keep browse tasks simple: "Extract all listings on this page."
- For file sources, use code_exec to parse and submit candidates.
- After extracting a batch, STOP and report. Don't browse endlessly.

Today's date: {current_date}

{files_section}
"""


class HarvesterAgent:
    """
    Harvester — navigates sources and produces candidates in batches.

    Uses BU V3 SDK for web extraction (bu-mini, server-side).
    Code_exec for file sources (sandbox).
    BU sessions are reused between batches for efficiency.
    """

    def __init__(
        self,
        source: str,
        description: str,
        source_id: str,
        openai_client: TrackedOpenAIClient,
        model: str,
        workspace_dir: Path,
        bu_client: BUClient,
        harvester_index: int = 0,
        research_context: str = "",
        sandbox: Optional[Any] = None,
        stop_checker: Optional[Callable[[], bool]] = None,
        stop_event: Optional[asyncio.Event] = None,
        blob_service_client: Optional[Any] = None,
        project_id: Optional[Any] = None,
        on_tool_call: Optional[Callable[[str, str], None]] = None,
        on_cost: Optional[Callable] = None,
        uploaded_file_urls: Optional[Dict[str, str]] = None,
        uploaded_files: Optional[List[Dict[str, Any]]] = None,
        mcp_tools: Optional[List[Dict[str, Any]]] = None,
        langfuse_parent: Optional[Any] = None,
    ) -> None:
        self.source = source
        self.description = description
        self.source_id = source_id
        self.workspace_dir = Path(workspace_dir)
        self.bu_client = bu_client
        self.harvester_index = harvester_index
        self.stop_checker = stop_checker
        self.on_tool_call = on_tool_call
        self.on_cost = on_cost

        # Batch state
        self._buffer: List[Candidate] = []
        self._bu_session_id: Optional[str] = None
        self._bu_lock = asyncio.Lock()  # serialize BU calls (one session at a time)
        self._exhausted: bool = False
        self._candidates_total: int = 0
        self._prev_cost: float = 0.0

        # Sandbox for code_exec (file sources)
        self._sandbox_impl: Optional[Any] = None
        if sandbox:
            from dsl_worker.infra.research_tools import ResearchTools, ResearchScope
            self._sandbox_impl = ResearchTools(
                workspace_dir=workspace_dir,
                schema=[],
                brave_api_key=None,
                openai_client=openai_client,
                model=model,
                sandbox=sandbox,
                stop_checker=stop_checker,
                blob_service_client=blob_service_client,
                project_id=project_id,
                uploaded_file_urls=uploaded_file_urls,
            )
            self._sandbox_impl.set_scope(ResearchScope(
                id=f"harvester:{harvester_index}",
                description="",
                quota=0,
            ))
            self._sandbox_impl.on_seed_from_code = self._handle_code_seed

        registry = ToolRegistry()
        self._register_tools(registry)

        from datetime import date
        files_section = self._format_files_section(uploaded_files)
        system_prompt = HARVESTER_SYSTEM_PROMPT.format(
            source=source,
            description=description or "(navigate and find candidate items)",
            research_context=research_context or "(none)",
            files_section=files_section,
            current_date=date.today().isoformat(),
        )

        # Built-in web search available to all agents
        web_search_tool = {"type": "web_search"}
        all_extra_tools = [web_search_tool] + (mcp_tools or [])

        self._conversation = AgentConversation(
            openai_client=openai_client,
            model=model,
            system_prompt=system_prompt,
            tools=registry,
            stop_checker=stop_checker,
            stop_event=stop_event,
            max_turns=30,
            reasoning={"effort": "medium", "summary": "detailed"},
            label=f"harvester:{harvester_index}",
            on_tool_call=on_tool_call,
            on_cost=on_cost,
            extra_tools=all_extra_tools,
            langfuse_parent=langfuse_parent,
        )

    def _format_files_section(self, uploaded_files: Optional[List[Dict[str, Any]]]) -> str:
        if not uploaded_files:
            return ""
        lines = ["\n## Uploaded Files\nAccessible via code_exec at the paths below:"]
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

    # ── Buffer management ─────────────────────────────────────────────

    def _add_to_buffer(self, content: Any, origin: str = "browse") -> None:
        """Add a candidate to the internal buffer."""
        candidate = Candidate(
            values=content,
            source_id=self.source_id,
            source_context=self.description,
            metadata={"origin": origin},
        )
        self._buffer.append(candidate)
        self._candidates_total += 1

    async def _handle_code_seed(self, seed_data: Dict) -> None:
        """Bridge code_exec submit_seed() calls to internal buffer."""
        self._add_to_buffer(
            content=seed_data.get("content", ""),
            origin="code_exec",
        )

    # ── BU V3 SDK extraction ─────────────────────────────────────────

    async def _run_bu_extract(self, url: str, task: str) -> Tuple[str, float]:
        """Extract candidates from a page via BU V3 SDK with session reuse."""
        async with self._bu_lock:
            return await self._run_bu_extract_inner(url, task)

    async def _run_bu_extract_inner(self, url: str, task: str) -> Tuple[str, float]:
        scope_id = f"harvester:{self.harvester_index}"

        bu_task = (
            f"Navigate to: {url}\n\n{task}\n\n"
            "Extract ALL items from the page using JavaScript/evaluate where possible. "
            "Include all visible fields (title, URL, description, price, date, etc.). "
            "Be fast — extract and return, don't over-explore."
        ) if url else task

        logger.info(f"[{scope_id}] BU extract START: {url[:80] if url else 'no url'}")
        t0 = time.time()

        try:
            items, bu_cost, session_id = await self.bu_client.extract(
                bu_task,
                session_id=self._bu_session_id,
                keep_alive=True,
            )
            elapsed = time.time() - t0

            # Store session for reuse
            if session_id:
                self._bu_session_id = session_id

            for item in items:
                self._add_to_buffer(json.dumps(item), origin="bu_extract")

            logger.info(
                f"[{scope_id}] BU extract DONE: {len(items)} items, "
                f"{elapsed:.1f}s, ${bu_cost:.4f}"
            )

            return (
                f"Extracted {len(items)} candidates from the page.\n"
                f"Total candidates this session: {self._candidates_total}"
            ), bu_cost

        except Exception as e:
            elapsed = time.time() - t0
            logger.error(f"[{scope_id}] BU extract error ({elapsed:.1f}s): {e}")
            # Session may be dead — reset for next attempt
            if self._bu_session_id:
                logger.info(f"[{scope_id}] Resetting BU session after error")
                self._bu_session_id = None
            return f"Extraction error: {e}", 0.0

    # ── Tool registration ─────────────────────────────────────────────

    def _register_tools(self, registry: ToolRegistry) -> None:
        """Register harvester tools."""

        # --- browse: BU V3 SDK extraction ---
        async def browse(args: Dict) -> tuple[str, float]:
            task = args.get("task", "")
            if not task:
                return "Error: task is required", 0.0
            return await self._run_bu_extract("", task)

        registry.add(
            name="browse",
            description=(
                "Launch a full cloud browser to navigate pages and extract candidates. "
                "The browser can search, navigate, scroll, bypass anti-bot, and extract "
                "structured data. Each extracted item is buffered as a candidate."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "task": {
                        "type": "string",
                        "description": (
                            "What to do. E.g.: 'Go to apartments.com/pmc/seattle-wa/ "
                            "and extract all property management company listings.'"
                        ),
                    },
                },
                "required": ["task"],
            },
            handler=browse,
        )

        # --- code_exec + list_files from sandbox ---
        if self._sandbox_impl:
            self._sandbox_impl.register_on(
                registry,
                exclude=[
                    "brave_search", "open", "find", "click",
                    "interact", "shell_exec",
                ],
            )

        # --- submit_candidate: coordinator submits directly ---
        async def submit_candidate(args: Dict) -> tuple[str, float]:
            content = args.get("content", "")
            if not content:
                return "Error: content is required", 0.0
            self._add_to_buffer(content, origin="coordinator")
            return (
                f"Candidate #{self._candidates_total} buffered. "
                f"Total: {self._candidates_total}"
            ), 0.0

        registry.add(
            name="submit_candidate",
            description="Submit a candidate you found directly.",
            parameters={
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": "Candidate data as JSON string",
                    },
                },
                "required": ["content"],
            },
            handler=submit_candidate,
        )

        # --- done ---
        async def done(args: Dict) -> tuple[str, float]:
            reason = args.get("reason", "complete")
            self._exhausted = True
            return (
                f"Harvester done: {reason}. "
                f"{self._candidates_total} candidates produced."
            ), 0.0

        registry.add(
            name="done",
            description="Signal the source is fully exhausted. Only call when there is genuinely nothing left.",
            parameters={
                "type": "object",
                "properties": {
                    "reason": {"type": "string", "description": "Why you're done"},
                },
            },
            handler=done,
        )

    # ── Batch interface ───────────────────────────────────────────────

    async def run_batch(
        self, message: str = "Begin harvesting candidates from your assigned source.",
    ) -> Tuple[List[Candidate], str]:
        """
        Run one batch of harvesting.

        The agent loop runs until the harvester outputs text without tool calls
        (natural batch boundary) or calls done() (source exhausted).

        Returns (new_candidates, report_text).
        Calling run_batch again continues the SAME conversation.
        """
        batch_start = len(self._buffer)

        result = await self._conversation.send(
            message,
            exit_condition=lambda: self._exhausted,
        )

        new_candidates = self._buffer[batch_start:]
        report = result.text or "(no report)"

        logger.info(
            f"[harvester:{self.harvester_index}] batch done: "
            f"{len(new_candidates)} new candidates, "
            f"exhausted={self._exhausted}"
        )

        return new_candidates, report

    @property
    def exhausted(self) -> bool:
        return self._exhausted

    @property
    def batch_cost_delta(self) -> float:
        """Cost since last check."""
        delta = self._conversation.total_cost - self._prev_cost
        self._prev_cost = self._conversation.total_cost
        return delta

    @property
    def cost_usd(self) -> float:
        return self._conversation.total_cost

    @property
    def candidates_total(self) -> int:
        return self._candidates_total

    @property
    def buffer_size(self) -> int:
        return len(self._buffer)

    async def close(self) -> None:
        """Stop BU session and clean up sandbox resources."""
        if self._bu_session_id and self.bu_client:
            try:
                await self.bu_client.stop_session(self._bu_session_id)
            except Exception as e:
                logger.warning(f"Harvester {self.harvester_index} session stop error: {e}")
            self._bu_session_id = None
        if self._sandbox_impl:
            try:
                await self._sandbox_impl.cleanup()
            except Exception as e:
                logger.warning(f"Harvester {self.harvester_index} cleanup error: {e}")
