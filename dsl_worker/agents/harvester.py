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
# Source Harvester — Dataset Generation Pipeline

You are harvesting candidates from the assigned source for a dataset.

<source>
{source}
</source>

<description>
{description}
</description>

<research_context>
{research_context}
</research_context>

## How to Work

1. Navigate to the source. Use browse(url, task) to extract candidates.
2. browse() sends a browser agent to the page — it extracts ALL items \
and returns them. Each item is buffered as a candidate.
3. If you spot candidates yourself, call submit_candidate() directly.
4. After extracting what's available, respond with a short report:
   - How many candidates you found
   - What lies ahead (more pages? running out? source nearly tapped?)
5. Call done() only when the source is fully exhausted.

## Important

- Do NOT filter candidates. Submit everything. Filtering happens downstream.
- Extract ALL visible data for each candidate (title, URL, date, price, location, \
description — whatever is shown on the page). The more complete each candidate, \
the less downstream research is needed.
- Do NOT click into individual items to get more data — just grab what's on the list page.
- Keep browse tasks simple and focused: "Extract all listings on this page."
- Do NOT include quality/date/topic filters in browse tasks.
- For file sources, use code_exec to parse and submit candidates programmatically.
- Cast a wide net — deduplication and filtering are cheap downstream.
- After extracting a batch, STOP and report. Don't keep browsing endlessly.

Today's date: {current_date}

{files_section}

## Tools

- browse(url, task): Browser agent extracts candidates from a page. Returns summary.
- submit_candidate(content): Submit a candidate you found directly.
- code_exec(script, description): Execute Python in sandbox. Use submit_seed() \
to yield candidates from code.
- done(reason): Signal the source is fully exhausted.
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
            extra_tools=mcp_tools or [],
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
            url = args.get("url", "")
            task = args.get("task", "")
            if not url and not task:
                return "Error: provide url and/or task", 0.0
            bu_task = task or "Extract all items on this page."
            return await self._run_bu_extract(url, bu_task)

        registry.add(
            name="browse",
            description=(
                "Navigate a page and extract candidates. A browser agent will "
                "find all items and buffer them as candidates. "
                "Returns a summary with candidate count."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "URL to navigate to",
                    },
                    "task": {
                        "type": "string",
                        "description": (
                            "Extraction task — keep it simple. E.g.: "
                            "'Extract all job listings on this page'."
                        ),
                    },
                },
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
