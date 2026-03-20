"""
Harvester agent — produces candidates from sources.

V10.2: Uses BU V3 SDK for web extraction. Each browse() call is a single
API call to BU Cloud — no local browser management.

- Web: BU V3 SDK extract() → structured items → pool
- Files: code_exec with submit_seed in sandbox
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
from dsl_worker.infra.candidate_pool import Candidate, CandidatePool

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
and returns them. Each item is automatically submitted as a candidate.
3. If you spot candidates yourself (e.g. from thinking about the source), \
you can call submit_candidate() directly.
4. Continue: try different pages, search terms, or sub-sections of the source.
5. Call done() when the source is exhausted.

## Important

- Do NOT filter candidates. Submit everything. Filtering happens downstream.
- Keep browse tasks simple and focused: "Extract all listings on this page."
- Do NOT include quality/date/topic filters in browse tasks.
- For file sources, use code_exec to parse and submit candidates programmatically.
- Cast a wide net — deduplication and filtering are cheap downstream.

{files_section}

## Tools

- browse(url, task): Browser agent extracts candidates from a page. Returns summary.
- submit_candidate(content): Submit a candidate you found directly.
- code_exec(script, description): Execute Python in sandbox. Use submit_seed() \
to yield candidates from code.
- done(reason): Signal you're finished.
"""


class HarvesterAgent:
    """
    Harvester — navigates sources and produces candidates.

    Uses BU V3 SDK for web extraction (bu-mini, server-side).
    Code_exec for file sources (sandbox).
    """

    def __init__(
        self,
        source: str,
        description: str,
        source_id: str,
        openai_client: TrackedOpenAIClient,
        model: str,
        workspace_dir: Path,
        pool: CandidatePool,
        bu_client: BUClient,
        harvester_index: int = 0,
        research_context: str = "",
        sandbox: Optional[Any] = None,
        stop_checker: Optional[Callable[[], bool]] = None,
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
        self.pool = pool
        self.bu_client = bu_client
        self.harvester_index = harvester_index
        self.stop_checker = stop_checker
        self.on_tool_call = on_tool_call
        self.on_cost = on_cost

        # State
        self._candidates_submitted = 0
        self._is_done = False

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

        files_section = self._format_files_section(uploaded_files)
        system_prompt = HARVESTER_SYSTEM_PROMPT.format(
            source=source,
            description=description or "(navigate and find candidate items)",
            research_context=research_context or "(none)",
            files_section=files_section,
        )

        self._conversation = AgentConversation(
            openai_client=openai_client,
            model=model,
            system_prompt=system_prompt,
            tools=registry,
            stop_checker=stop_checker,
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

    async def _submit_to_pool(self, content: Any, origin: str = "browse") -> None:
        """Submit a candidate to the pool."""
        candidate = Candidate(
            values=content,
            source_id=self.source_id,
            source_context=self.description,
            metadata={"origin": origin},
        )
        await self.pool.submit(candidate)
        self._candidates_submitted += 1

    async def _handle_code_seed(self, seed_data: Dict) -> None:
        """Bridge code_exec submit_seed() calls to the CandidatePool."""
        await self._submit_to_pool(
            content=seed_data.get("content", ""),
            origin="code_exec",
        )

    # ── BU V3 SDK extraction ─────────────────────────────────────────

    async def _run_bu_extract(self, url: str, task: str) -> Tuple[str, float]:
        """
        Extract candidates from a page via BU V3 SDK.

        Single API call — BU handles browser, navigation, extraction server-side.
        Returns structured items which are submitted to the pool.
        """
        scope_id = f"harvester:{self.harvester_index}"

        bu_task = (
            f"Navigate to: {url}\n\n{task}\n\n"
            "Extract ALL items from the page. For each item, include all visible "
            "fields (title, URL, description, price, date, etc.) in the data."
        ) if url else task

        logger.info(f"[{scope_id}] BU extract START: {url[:80] if url else 'no url'}")
        t0 = time.time()

        try:
            items, bu_cost = await self.bu_client.extract(bu_task)
            elapsed = time.time() - t0

            for item in items:
                await self._submit_to_pool(json.dumps(item), origin="bu_extract")

            logger.info(
                f"[{scope_id}] BU extract DONE: {len(items)} items, "
                f"{elapsed:.1f}s, ${bu_cost:.4f}"
            )

            return (
                f"Extracted {len(items)} candidates from the page.\n"
                f"Total candidates submitted: {self._candidates_submitted}"
            ), bu_cost

        except Exception as e:
            elapsed = time.time() - t0
            logger.error(f"[{scope_id}] BU extract error ({elapsed:.1f}s): {e}")
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
                "find all items and submit them automatically. "
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
            await self._submit_to_pool(content, origin="coordinator")
            return (
                f"Candidate #{self._candidates_submitted} submitted. "
                f"Total: {self._candidates_submitted}"
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
            self._is_done = True
            return (
                f"Harvester done: {reason}. "
                f"{self._candidates_submitted} candidates submitted."
            ), 0.0

        registry.add(
            name="done",
            description="Signal you're finished harvesting.",
            parameters={
                "type": "object",
                "properties": {
                    "reason": {"type": "string", "description": "Why you're done"},
                },
            },
            handler=done,
        )

    async def run(self) -> AgentResult:
        """Run the harvester."""
        result = await self._conversation.send(
            "Begin harvesting candidates from your assigned source.",
            exit_condition=lambda: self._is_done,
        )

        logger.info(
            f"[harvester:{self.harvester_index}] finished: "
            f"{self._candidates_submitted} candidates submitted"
        )
        return result

    @property
    def cost_usd(self) -> float:
        return self._conversation.total_cost

    @property
    def candidates_submitted(self) -> int:
        return self._candidates_submitted

    async def cleanup(self) -> None:
        """Clean up sandbox resources."""
        if self._sandbox_impl:
            try:
                await self._sandbox_impl.cleanup()
            except Exception as e:
                logger.warning(f"Harvester {self.harvester_index} cleanup error: {e}")
