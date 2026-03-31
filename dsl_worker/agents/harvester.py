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

You are the candidate collector. Row generators downstream handle all enrichment \
(emails, phones, LinkedIn, verification). Your job is to discover and yield \
candidate entities as efficiently as possible.

**Yield generously.** If something looks like it could be a valid candidate, \
submit it. Only skip things that are obviously wrong at a glance.

**By default, don't research individual candidates.** When you're iterating a \
list (directory, search results, file), just grab what's visible and move on. \
Don't look up emails, phone numbers, or details per candidate.

**Exception for open-ended research:** When there's no clean list and candidates \
must be discovered through research, quick validation per candidate is OK \
(e.g., confirming a company actually does what's needed). But still don't \
enrich — no email/phone/contact lookups. That's the row generator's job.

## Web Research

**Web search** (built-in) — Use for general research: finding sources, \
discovering list pages, looking up directories, verifying what's available. \
Fast and cheap. 10x faster and cheaper than browse. Always try web search first.

**browse(task)** — Full cloud browser. EXPENSIVE and SLOW (1-5 minutes per call). \
Use ONLY when you have a specific URL AND need a real browser to extract from it \
(JS-heavy pages, infinite scroll, anti-bot, forms). Never use browse for general \
searching or research — use web search for that.

### How to use browse effectively

Browse works in **page-scoped batches**. Each call should target ONE page:
- Give it a specific URL + what to extract. E.g.: "Go to apartments.com/pmc/seattle-wa/ \
and extract all property management company listings."
- Browse extracts what's visible on that page and returns items + a report.
- YOU (the harvester) decide what to do next based on the report: send browse \
to page 2? A different URL? Enough candidates?
- Do NOT ask browse to explore a site, follow links, or find more pages. \
That's YOUR job using web search.
- If browse reports "blocked" (anti-bot), don't retry the same URL. Try a \
different source or search approach.

## How to Work

1. Use **web search** to find the right sources, URLs, and list pages.
2. Use **browse** to extract from specific pages that need a real browser.
3. Read the browse report — if it says pagination exists, decide whether \
to send browse to the next page or if you have enough candidates.
4. Submit each candidate with whatever data is visible. Even just a name is \
enough — row generators handle enrichment.
5. After each browse batch, respond with a short report of your own.
6. Call done() only when the source is fully exhausted.

## File Sources

For file-based sources (CSV, JSON, etc.), use **code_exec** to parse the file \
and call **submit_seed()** for each candidate. This is the fastest way to yield \
candidates from structured data — one script can submit hundreds of candidates \
programmatically.

Example: read a CSV and submit each row as a candidate using code_exec:
  import csv; [submit_seed(json.dumps(row), source="file.csv") for row in csv.DictReader(open("/workspace/uploads/file.csv"))]

submit_seed(content, source) is a built-in function available in code_exec. \
Each call adds one candidate to the buffer.

## Rules

- One harvester = one specific search/query/page. Don't try to cover multiple \
search terms or slices in one harvester.
- Submit candidates quickly. A name + any visible context is sufficient.
- Only include URLs you actually saw in search results or on a page. \
Never construct or guess URLs — many sites use IDs or slugs that can't be inferred from titles.
- Do NOT enrich individual candidates (no email/phone/LinkedIn lookups).
- After extracting a batch, STOP and report.

Today's date: {current_date}

{files_section}
"""


class HarvesterAgent:
    """
    Harvester — navigates sources and produces candidates in batches.

    Uses BU V3 SDK for web extraction (bu-max, server-side).
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
        on_candidate: Optional[Callable[[Candidate], None]] = None,
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
        self._on_candidate = on_candidate

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
            reasoning={"effort": "high", "summary": "detailed"},
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
        """Add a candidate and stream it to the dispatcher immediately."""
        candidate = Candidate(
            values=content,
            source_id=self.source_id,
            source_context=self.description,
            metadata={"origin": origin},
        )
        self._buffer.append(candidate)
        self._candidates_total += 1
        if self._on_candidate:
            self._on_candidate(candidate)

    async def _handle_code_seed(self, seed_data: Dict) -> None:
        """Bridge code_exec submit_seed() calls to internal buffer."""
        self._add_to_buffer(
            content=seed_data.get("content", ""),
            origin="code_exec",
        )

    # ── BU V3 SDK extraction ─────────────────────────────────────────

    async def _run_bu_extract(self, task: str) -> Tuple[str, float]:
        """Extract candidates from a page via BU V3 SDK with session reuse."""
        async with self._bu_lock:
            return await self._run_bu_extract_inner(task)

    async def _run_bu_extract_inner(self, task: str) -> Tuple[str, float]:
        scope_id = f"harvester:{self.harvester_index}"

        bu_task = (
            f"{task}\n\n"
            "INSTRUCTIONS:\n"
            "- Extract items visible on THIS page only. Use JavaScript/evaluate to "
            "parse the DOM programmatically where possible.\n"
            "- Include all visible fields per item (title, URL, description, price, "
            "date, etc.).\n"
            "- Do NOT navigate away, follow links, or explore the rest of the site. "
            "Stay on this page.\n"
            "- If anti-bot or CAPTCHA blocks you after 2 attempts, STOP and report "
            "'blocked' — do not keep retrying.\n"
            "- After extracting, end with a brief REPORT: how many items you found, "
            "whether pagination or 'load more' exists (and total pages/items if "
            "visible), and any other navigation you noticed (tabs, filters, "
            "categories) that could yield more candidates."
        )

        logger.info(f"[{scope_id}] BU extract START: {task[:100]}")
        t0 = time.time()

        try:
            items, bu_cost, session_id, bu_summary = await self.bu_client.extract(
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

            # Include BU's summary so the harvester can act on pagination/nav info
            report = (
                f"Extracted {len(items)} candidates from the page.\n"
                f"Total candidates this session: {self._candidates_total}"
            )
            if bu_summary:
                report += f"\nBrowser report: {bu_summary[:500]}"

            return report, bu_cost

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
            return await self._run_bu_extract(task)

        registry.add(
            name="browse",
            description=(
                "Launch a full cloud browser to extract candidates from a specific page. "
                "EXPENSIVE (~$0.10-0.50 per call, 1-5 min). Use only when web search "
                "can't get the data (JS-heavy pages, anti-bot, forms, infinite scroll). "
                "Give it ONE specific URL + extraction task per call. It returns extracted "
                "items plus a report on pagination/navigation it saw. "
                "Cannot process video or audio — don't use on YouTube, TikTok, etc."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "task": {
                        "type": "string",
                        "description": (
                            "A specific URL + what to extract from THAT page. "
                            "E.g.: 'Go to apartments.com/pmc/seattle-wa/ and "
                            "extract all property management company listings.' "
                            "Keep it to one page — don't ask it to explore the site."
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
                    "shell_exec",
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
