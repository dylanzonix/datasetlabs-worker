"""
Harvester agent — produces candidates from sources.

V10: Handles both web and file sources with Browser Use autonomy:
- Web: spawns BU agents with submit_seed custom tool, BU navigates AND extracts
- Files: uses code_exec with submit_seed in sandbox

The harvester coordinator (our LLM) decides strategy — which pages to visit,
when to continue, when to stop. BU handles the actual navigation + extraction.
The coordinator can also submit candidates directly via submit_candidate().
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple

from dsl_worker.agents.base import AgentConversation, AgentResult
from dsl_worker.agents.tools import ToolRegistry
from dsl_worker.billing.tracked_client import TrackedOpenAIClient
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

1. Navigate to the source. Use brave_search if no direct URL is given.
2. Use browse(url, task) to have the browser agent navigate pages and extract \
candidates. It extracts ALL items and returns structured data with real URLs. \
You'll get a summary back with how many were found and what's available next.
3. If you spot candidates yourself (e.g. from search results or a page you opened), \
you can also call submit_candidate() directly.
4. Continue: follow pagination, try different search terms, explore related pages.
5. Call done() when the source is exhausted.

## Important

- Do NOT filter candidates. Submit everything. Filtering happens downstream.
- The browse() agent extracts every item on the page. Keep browse tasks simple: \
"Extract all listings on this page, then go to page 2."
- Do NOT include quality/date/topic filters in browse tasks — the search URL \
already scopes the results, and row generators handle filtering later.
- For file sources, use code_exec to parse and submit candidates programmatically.
- Cast a wide net — deduplication and filtering are cheap downstream.

{files_section}

## Tools

- browse(url, task): Browser agent navigates + extracts candidates. Returns summary.
- submit_candidate(content): Submit a candidate you found directly.
- brave_search(query): Search the web for sources.
- open(url): View a page as markdown.
- find(ref_id, pattern): Search within a loaded page.
- click(ref_id, link_id): Follow a link.
- code_exec(script, description): Execute Python in sandbox. Use submit_seed() \
to yield candidates from code.
- done(reason): Signal you're finished.
"""


class HarvesterAgent:
    """
    Harvester — navigates sources and produces candidates.

    Two modes:
    - Web: spawns Browser Use agents with submit_seed custom tool
    - File: uses code_exec with submit_seed in sandbox

    The coordinator LLM decides strategy; BU handles navigation + extraction.
    The coordinator can also submit candidates directly via submit_candidate().
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
        harvester_index: int = 0,
        research_context: str = "",
        brave_api_key: Optional[str] = None,
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
        on_browser_started: Optional[Callable] = None,
        on_browser_stopped: Optional[Callable] = None,
    ) -> None:
        self.source = source
        self.description = description
        self.source_id = source_id
        self.workspace_dir = Path(workspace_dir)
        self.pool = pool
        self.harvester_index = harvester_index
        self.stop_checker = stop_checker
        self.on_tool_call = on_tool_call
        self.on_cost = on_cost

        # State
        self._candidates_submitted = 0
        self._is_done = False

        # Build research tools (browsing infra, code_exec, etc.)
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
            uploaded_file_urls=uploaded_file_urls,
            on_browser_started=on_browser_started,
            on_browser_stopped=on_browser_stopped,
        )
        self._impl.set_scope(ResearchScope(
            id=f"harvester:{harvester_index}",
            description="",
            quota=0,
        ))

        # Set up code_exec seed bridge → pool
        self._impl.on_seed_from_code = self._handle_code_seed

        registry = ToolRegistry()
        self._register_tools(registry)

        # Build system prompt
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

    # ── BU with custom submit_seed tool ──────────────────────────────

    async def _run_bu_browse(
        self, url: str, task: str,
    ) -> Tuple[str, float]:
        """
        Run a BU agent to extract candidates from a page.

        Uses BU's output_model_schema for structured extraction — BU adds
        a done() action typed to our Pydantic model and returns validated data.
        This avoids URL hallucination from custom tools (BU extracts hrefs
        from the DOM natively).

        CDP lifecycle: disconnect Playwright → create BU → run → cleanup → reconnect.
        """
        try:
            from browser_use import Agent
            from browser_use.browser.profile import BrowserProfile
            from browser_use.llm.browser_use import ChatBrowserUse
            from pydantic import BaseModel, Field
        except ImportError:
            return "browser-use not installed", 0.0

        scope_id = f"harvester:{self.harvester_index}"

        # Define structured output model for BU extraction
        class ExtractedItem(BaseModel):
            url: str = Field(
                description="The item's own URL — the EXACT href from the link "
                "element on the page. Do NOT construct from the title."
            )
            title: str = Field(description="The item's title or name")
            description: str = Field(
                default="", description="Description or summary text"
            )
            extra: str = Field(
                default="",
                description="Any other visible fields as JSON "
                "(price, date, location, category, etc.)",
            )

        class ExtractionResult(BaseModel):
            items: List[ExtractedItem] = Field(
                description="All items extracted from the page(s)"
            )

        try:
            # Ensure cloud browser exists
            await self._impl._get_browser()
            if not self._impl._cdp_url:
                return "No cloud browser session available", 0.0

            # CRITICAL: disconnect Playwright before BU
            await self._impl._disconnect_playwright()

            profile = BrowserProfile(
                cdp_url=self._impl._cdp_url,
                keep_alive=True,
            )

            async def should_stop():
                return bool(self.stop_checker and self.stop_checker())

            llm = ChatBrowserUse(model='bu-2-0')

            bu_task = (
                f"Navigate to: {url}\n\n{task}\n\n"
                "Extract ALL items from the page. For each item, get the EXACT "
                "href URL from the link element (do NOT construct URLs from titles). "
                "Follow pagination if available. Submit everything — do NOT filter."
            ) if url else task

            agent_kwargs = dict(
                task=bu_task,
                llm=llm,
                browser_profile=profile,
                output_model_schema=ExtractionResult,
                extend_system_message=(
                    "You are a data extraction agent. Extract ALL items from pages.\n"
                    "CRITICAL: For each item's URL, copy the EXACT href from the "
                    "link element. Do NOT construct or guess URLs.\n"
                    "Do NOT filter or skip items — extract everything visible."
                ),
                register_should_stop_callback=should_stop,
                max_actions_per_step=3,
                use_judge=False,
                directly_open_url=True,
            )

            candidates_before = self._candidates_submitted
            logger.info(f"[{scope_id}] BU browse START: url={url[:80] if url else 'none'} task={task[:100]}")
            t0 = time.time()

            agent = Agent(**agent_kwargs)
            bu_steps = 0

            try:
                supervisor = self._impl._make_bu_supervisor()
                result = await agent.run(max_steps=100, on_step_end=supervisor)
                bu_steps = agent.state.n_steps

                # Get structured output — typed ExtractionResult
                structured = result.structured_output
                if structured and hasattr(structured, 'items'):
                    for item in structured.items:
                        candidate_data = {
                            "url": item.url,
                            "title": item.title,
                            "description": item.description,
                        }
                        if item.extra:
                            try:
                                extra = json.loads(item.extra)
                                if isinstance(extra, dict):
                                    candidate_data.update(extra)
                            except json.JSONDecodeError:
                                candidate_data["extra"] = item.extra
                        await self._submit_to_pool(
                            json.dumps(candidate_data), origin="bu_extract"
                        )
                else:
                    # Fallback: parse final_result text
                    raw = result.final_result() if result.final_result() else ""
                    if raw:
                        await self._parse_and_submit_bu_output(raw, scope_id)

            except Exception as run_err:
                logger.warning(f"[{scope_id}] BU run error: {run_err}")
                bu_steps = getattr(agent.state, 'n_steps', 0)
            finally:
                await self._impl._stop_bu_cdp_client(agent)
                try:
                    await agent.close()
                except Exception:
                    pass
                # Kill any EventBus that restarted during cleanup
                await self._kill_event_bus(agent)
                await self._impl._reconnect_playwright()

            candidates_found = self._candidates_submitted - candidates_before
            elapsed = time.time() - t0

            logger.info(
                f"[{scope_id}] BU browse DONE: {bu_steps} steps, "
                f"{candidates_found} candidates, {elapsed:.1f}s"
            )

            return (
                f"Browser agent completed: {bu_steps} steps\n"
                f"Candidates submitted this browse: {candidates_found}\n"
                f"Total candidates submitted: {self._candidates_submitted}"
            ), 0.0

        except Exception as e:
            logger.error(f"[{scope_id}] BU browse error: {e}", exc_info=True)
            try:
                await self._impl._reconnect_playwright()
            except Exception:
                pass
            return f"Browser agent error: {e}", 0.0

    @staticmethod
    async def _kill_event_bus(agent: Any) -> None:
        """Ensure BU agent's EventBus is fully dead after cleanup.

        agent.close() with keep_alive=True can leave a zombie EventBus if
        late events (CDP disconnect, watchdog timers) restart it after the
        queue was nulled out. This catches any such zombies.
        """
        try:
            session = getattr(agent, 'browser_session', None)
            if not session:
                return
            bus = getattr(session, 'event_bus', None)
            if not bus:
                return
            if getattr(bus, '_is_running', False):
                await bus.stop(clear=True, timeout=2)
            task = getattr(bus, '_runloop_task', None)
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except (asyncio.CancelledError, Exception):
                    pass
        except Exception:
            pass

    async def _parse_and_submit_bu_output(self, raw: str, scope_id: str) -> None:
        """Fallback: parse BU's text output and submit each item to the pool."""
        text = raw.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines).strip()

        items = []
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                items = parsed
            elif isinstance(parsed, dict):
                items = [parsed]
        except json.JSONDecodeError:
            match = re.search(r'\[[\s\S]*\]', text)
            if match:
                try:
                    items = json.loads(match.group())
                except json.JSONDecodeError:
                    pass

        if not items:
            logger.warning(f"[{scope_id}] Could not parse BU output: {text[:200]}")
            if text and len(text) > 20:
                await self._submit_to_pool(text, origin="bu_raw")
            return

        logger.info(f"[{scope_id}] Parsed {len(items)} items from BU text output")
        for item in items:
            if isinstance(item, dict):
                await self._submit_to_pool(json.dumps(item), origin="bu_extract")
            elif isinstance(item, str) and len(item) > 10:
                await self._submit_to_pool(item, origin="bu_extract")

    # ── Tool registration ────────────────────────────────────────────

    def _register_tools(self, registry: ToolRegistry) -> None:
        """Register harvester tools."""

        # --- Research/browsing tools (brave_search, open, find, click, code_exec) ---
        # Exclude interact — we replace it with browse() which has custom BU tools
        self._impl.register_on(registry, exclude=["interact"])

        # --- browse: BU native extraction (no custom tools) ---
        async def browse(args: Dict) -> tuple[str, float]:
            url = args.get("url", "")
            task = args.get("task", "")

            if not url and not task:
                return "Error: provide url and/or task", 0.0

            # BU extracts data natively and returns JSON — no custom tools.
            # This avoids URL hallucination from BU's LLM constructing URLs.
            bu_task = task or "Extract all items on this page."

            return await self._run_bu_browse(url, bu_task)

        registry.add(
            name="browse",
            description=(
                "Navigate a page and extract candidates. The browser agent will "
                "extract all items and return structured data with real URLs. "
                "Returns a summary with candidate count and what's available next. "
                "The browser agent extracts everything — filtering happens "
                "downstream, so don't include filtering criteria in the task."
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
                            "Navigation task — keep it simple. E.g.: "
                            "'Submit all job listings on this page, then go to page 2'. "
                            "Do NOT include filtering criteria — submit everything."
                        ),
                    },
                },
            },
            handler=browse,
        )

        # --- submit_candidate: coordinator can submit directly ---
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
            description=(
                "Submit a candidate you found directly. Use when you spot "
                "a candidate from search results or a page you opened."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": (
                            "Candidate data — name, URL, description, or structured "
                            "info as JSON string"
                        ),
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
            description="Signal you're finished harvesting. Call when source is exhausted.",
            parameters={
                "type": "object",
                "properties": {
                    "reason": {
                        "type": "string",
                        "description": "Why you're done",
                    },
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
        """Clean up browser and other resources."""
        try:
            await self._impl.cleanup()
        except Exception as e:
            logger.warning(f"Harvester {self.harvester_index} cleanup error: {e}")
