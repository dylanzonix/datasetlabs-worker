"""
Crawler agent — navigates sources and dumps pages for candidate extraction.

V7: Replaces the seed yielder's dual role (navigate + extract). The crawler
focuses purely on navigation: finding pages that contain candidate items,
dumping their full content, and moving on fast. Extraction happens later
via the Extractor (cheap batch LLM calls on dumped pages).

Key differences from SeedYielderAgent:
- dump_page() replaces yield_seed() — saves full page content to a shared buffer
- Context trimming: old tool outputs (>3 turns back) get truncated to prevent
  context window explosion that plagued seed yielders
- Simpler prompt focused on coverage and speed
- Fewer max turns (30 vs 50)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional

from dsl_worker.agents.base import AgentConversation, AgentResult
from dsl_worker.agents.tools import ToolRegistry
from dsl_worker.billing.tracked_client import TrackedOpenAIClient

logger = logging.getLogger(__name__)

READ_FILE_LIMIT = 30_000

# Truncate tool outputs older than this many turns to save context
TRIM_AGE_TURNS = 3
TRIM_PREVIEW_CHARS = 200
TRIM_MIN_LENGTH = 500  # Don't bother truncating short outputs


CRAWLER_SYSTEM_PROMPT = """\
You are a source navigator for a dataset generation pipeline. Your job is to \
browse assigned sources and save pages that contain candidate items for the dataset.

<what_to_find>
{candidate_description}
</what_to_find>

<sources>
{sources}
</sources>

<instructions>
{instructions}
</instructions>

<research_context>
{research_context}
</research_context>

## How to Work

1. Open your assigned sources. Use brave_search if no direct URLs are given.
2. When you find a page with candidate items, call save_page(ref_id, description).
3. Follow pagination links and listing links to cover as much ground as possible.
4. Focus on COVERAGE and SPEED. Save pages liberally — extraction happens later.
5. Don't analyze individual items. Just find pages that contain them.
6. Call done() when sources are exhausted or you've saved enough pages.

## Tips

- save_page() is cheap. When in doubt, save it.
- Follow "Next page", "Page 2", "Load more" links for pagination.
- If a listing page links to detail pages, save both the listing AND detail pages.
- If a search returns many results pages, paginate through all of them.
- Use brave_search to discover additional sources if your assigned ones are thin.

## Tools

- open(url): View a page. Returns line-numbered markdown with links table.
- find(ref_id, pattern): Search within a loaded page.
- click(ref_id, link_id): Follow a link from the links table.
- brave_search(query): Search the web.
- interact(url_or_ref_id, task): Browser agent for anti-bot bypass ONLY.
- save_page(ref_id, description): Save this page for candidate extraction. \
Description should note what candidates are on the page.
- read_file(path): Read a workspace file.
- done(reason): Signal you're finished.
"""


class CrawlerAgent:
    """
    Crawler — navigates sources, dumps pages for later extraction.

    Usage:
        crawler = CrawlerAgent(
            sources=["https://example.com/listings"],
            instructions="Find all product pages",
            candidate_description="products with name and price",
            openai_client=tracked_client,
            model="gpt-5.2",
            workspace_dir=Path("/workspace"),
            on_dump_page=buffer_callback,
            ...
        )
        result = await crawler.run()
    """

    def __init__(
        self,
        sources: List[str],
        instructions: str,
        candidate_description: str,
        openai_client: TrackedOpenAIClient,
        model: str,
        workspace_dir: Path,
        on_dump_page: Callable[[Dict[str, Any]], Awaitable[None]],
        crawler_index: int = 0,
        research_context: str = "",
        brave_api_key: Optional[str] = None,
        sandbox: Optional[Any] = None,
        stop_checker: Optional[Callable[[], bool]] = None,
        blob_service_client: Optional[Any] = None,
        project_id: Optional[Any] = None,
        on_tool_call: Optional[Callable[[str, str], None]] = None,
        on_cost: Optional[Callable] = None,
        mcp_tools: Optional[List[Dict[str, Any]]] = None,
        langfuse_parent: Optional[Any] = None,
        on_browser_started: Optional[Callable] = None,
        on_browser_stopped: Optional[Callable] = None,
    ) -> None:
        self.sources = sources
        self.instructions = instructions
        self.candidate_description = candidate_description
        self.workspace_dir = Path(workspace_dir)
        self.on_dump_page = on_dump_page
        self.crawler_index = crawler_index
        self.stop_checker = stop_checker
        self.on_tool_call = on_tool_call
        self.on_cost = on_cost

        # State
        self._dumped_count = 0
        self._is_done = False

        # Build research tools (browsing infra)
        from dsl_worker.infra.research_tools import ResearchTools, ResearchScope

        self._impl = ResearchTools(
            workspace_dir=workspace_dir,
            schema=[],
            brave_api_key=brave_api_key,
            openai_client=openai_client,
            model=model,
            sandbox=sandbox,
            stop_checker=stop_checker,
            blob_service_client=blob_service_client,
            project_id=project_id,
            on_browser_started=on_browser_started,
            on_browser_stopped=on_browser_stopped,
        )
        self._impl.set_scope(ResearchScope(
            id=f"crawler:{crawler_index}",
            description="",
            quota=0,
        ))

        registry = ToolRegistry()
        self._register_tools(registry)

        # Build system prompt
        system_prompt = CRAWLER_SYSTEM_PROMPT.format(
            candidate_description=candidate_description,
            sources="\n".join(f"- {s}" for s in sources) if sources else "(none — use brave_search)",
            instructions=instructions or "(no specific instructions)",
            research_context=research_context or "(none)",
        )

        self._conversation = AgentConversation(
            openai_client=openai_client,
            model=model,
            system_prompt=system_prompt,
            tools=registry,
            stop_checker=stop_checker,
            max_turns=30,
            reasoning={"effort": "medium", "summary": "detailed"},
            label=f"crawler:{crawler_index}",
            on_tool_call=on_tool_call,
            on_cost=on_cost,
            extra_tools=mcp_tools or [],
            langfuse_parent=langfuse_parent,
        )

    def _register_tools(self, registry: ToolRegistry) -> None:
        """Register crawler tools."""

        # --- Research/browsing tools (open, find, click, brave_search, interact, etc.) ---
        self._impl.register_on(registry, exclude=["code_exec", "list_files"])

        # --- read_file ---
        async def read_file(args: Dict) -> tuple[str, float]:
            path_str = args.get("path", "")
            try:
                path = Path(path_str)
                if not path.is_absolute():
                    candidate = self.workspace_dir / path
                    if not candidate.exists():
                        candidate = self.workspace_dir / "sources" / path
                    path = candidate

                if not path.exists():
                    return f"File not found: {path_str}", 0.0

                content = path.read_text(encoding="utf-8")
                if len(content) > READ_FILE_LIMIT:
                    content = content[:READ_FILE_LIMIT] + f"\n\n[Truncated at {READ_FILE_LIMIT} chars]"
                return content, 0.0
            except Exception as e:
                return f"Error reading file: {e}", 0.0

        registry.add(
            name="read_file",
            description="Read a file from the workspace.",
            parameters={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "File path (relative to workspace or sources directory)",
                    },
                },
                "required": ["path"],
            },
            handler=read_file,
        )

        # --- save_page ---
        async def save_page(args: Dict) -> tuple[str, float]:
            ref_id = args.get("ref_id", "")
            description = args.get("description", "")

            page = self._impl.artifacts.get_page(ref_id)
            if not page:
                return f"Error: no page loaded with ref_id '{ref_id}'", 0.0

            # Build raw markdown from lines
            raw_content = "\n".join(page.lines)

            # Save to shared page buffer via callback
            await self.on_dump_page({
                "url": page.url,
                "content": raw_content,
                "ref_id": ref_id,
                "description": description,
                "crawler_id": self.crawler_index,
            })
            self._dumped_count += 1

            # Trim old page content from conversation to prevent context explosion
            self._trim_old_outputs()

            return (
                f"Page saved ({page.url[:80]}, {len(raw_content)} chars). "
                f"{self._dumped_count} pages saved total."
            ), 0.0

        registry.add(
            name="save_page",
            description=(
                "Save the current page for candidate extraction. Call this when "
                "you find a page that contains candidate items. The page content "
                "will be extracted by a separate process."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "ref_id": {
                        "type": "string",
                        "description": "Page reference ID (e.g. 'p0', 'p1') from open() or click()",
                    },
                    "description": {
                        "type": "string",
                        "description": (
                            "What candidates are on this page "
                            "(e.g., 'listing page with ~50 churches and addresses')"
                        ),
                    },
                },
                "required": ["ref_id"],
            },
            handler=save_page,
        )

        # --- done ---
        async def done(args: Dict) -> tuple[str, float]:
            reason = args.get("reason", "complete")
            self._is_done = True
            return (
                f"Navigator done: {reason}. "
                f"Saved {self._dumped_count} pages."
            ), 0.0

        registry.add(
            name="done",
            description="Signal you're finished navigating. Call when sources are exhausted.",
            parameters={
                "type": "object",
                "properties": {
                    "reason": {
                        "type": "string",
                        "description": "Why you're done (e.g., 'all pages crawled', 'sources exhausted')",
                    },
                },
            },
            handler=done,
        )

    def _trim_old_outputs(self) -> None:
        """Truncate old function_call_output items to prevent context explosion.

        After each dump, scan conversation messages and truncate any
        function_call_output that is:
        - More than TRIM_AGE_TURNS turns back (counted by API calls, not messages)
        - Longer than TRIM_MIN_LENGTH chars

        Replaces with: first TRIM_PREVIEW_CHARS + truncation notice.
        This keeps the crawler smart for recent navigation decisions but prevents
        the exponential context cost that plagued seed yielders.
        """
        messages = self._conversation.messages
        if not messages:
            return

        # Count turns from the end (each function_call_output is roughly one turn)
        output_indices = [
            i for i, msg in enumerate(messages)
            if isinstance(msg, dict) and msg.get("type") == "function_call_output"
        ]

        if len(output_indices) <= TRIM_AGE_TURNS:
            return

        # Trim all but the last TRIM_AGE_TURNS outputs
        old_indices = set(output_indices[:-TRIM_AGE_TURNS])

        for i in old_indices:
            msg = messages[i]
            output = msg.get("output", "")
            if len(output) > TRIM_MIN_LENGTH:
                msg["output"] = (
                    output[:TRIM_PREVIEW_CHARS]
                    + "\n[...truncated, page was saved...]"
                )

    async def run(self) -> AgentResult:
        """Run the crawler."""
        result = await self._conversation.send(
            "Begin browsing your assigned sources. Save pages that contain candidate items.",
            exit_condition=lambda: self._is_done,
        )

        logger.info(
            f"[crawler:{self.crawler_index}] finished: "
            f"{self._dumped_count} pages dumped"
        )

        return result

    @property
    def cost_usd(self) -> float:
        return self._conversation.total_cost

    async def cleanup(self) -> None:
        """Clean up browser and other resources."""
        try:
            await self._impl.cleanup()
        except Exception as e:
            logger.warning(f"Crawler {self.crawler_index} cleanup error: {e}")
