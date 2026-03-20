"""
Research agent — conversational agent with web browsing and code execution tools.

V10.2: Uses BU V3 SDK for all web interaction. browse() is a single API call
to BU Cloud (bu-mini). No local browser management.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable, Dict, Optional, List, Tuple

from dsl_worker.agents.base import AgentConversation, AgentResult
from dsl_worker.agents.tools import ToolRegistry
from dsl_worker.billing.tracked_client import TrackedOpenAIClient
from dsl_worker.infra.bu_client import BUClient

logger = logging.getLogger(__name__)

RESEARCH_SYSTEM_PROMPT = """\
# Research Agent

An orchestrator sends specific questions; investigate and give a focused answer.

## Tools

- browse(url, task): Browse the web — visit a URL or search. Describe what \
you need. Returns extracted text.
- code_exec(script, description): Execute Python (pandas, json, csv available).
- list_files(directory): List uploaded/downloaded files.
- respond(content): Submit your final answer — you MUST call this when done.

## How to work

1. Use browse() to search the web or visit specific pages.
   - browse(task="Search for X") to search
   - browse(url="https://...", task="Find Y on this page") to visit a page
2. Use browse() on 1-2 promising results to get details.
3. Call respond() as soon as you can answer the question.

Be efficient. A focused answer from 1-2 good sources beats an exhaustive survey.

## Response format

Call respond() with well-structured content:
- Use headers and bullet points for scannability
- Cite source URLs for key claims
"""

class ResearchAgent:
    """
    Conversational research agent. Uses BU V3 SDK for web access.
    """

    def __init__(
        self,
        openai_client: TrackedOpenAIClient,
        model: str,
        workspace_dir: Path,
        bu_client: Optional[BUClient] = None,
        sandbox: Optional[Any] = None,
        stop_checker: Optional[Callable[[], bool]] = None,
        max_turns: int = 20,
        tool_budget: int = 0,
        system_prompt: Optional[str] = None,
        blob_service_client: Optional[Any] = None,
        project_id: Optional[Any] = None,
        on_tool_call: Optional[Callable[[str, str], None]] = None,
        uploaded_file_urls: Optional[Dict[str, str]] = None,
        mcp_tools: Optional[List[Dict[str, Any]]] = None,
        # Legacy kwargs (ignored)
        brave_api_key: Optional[str] = None,
        on_browser_started: Optional[Callable] = None,
        on_browser_stopped: Optional[Callable] = None,
    ) -> None:
        self.workspace_dir = Path(workspace_dir)
        self.bu_client = bu_client
        self.stop_checker = stop_checker

        # respond() tool state
        self._responded: bool = False
        self._response_text: str = ""

        # Sandbox for code_exec (optional)
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
                id="research", description="", quota=0,
            ))

        # Build tool registry
        registry = ToolRegistry(tool_budget=tool_budget)
        self._register_tools(registry)

        effective_prompt = system_prompt or RESEARCH_SYSTEM_PROMPT

        soft_limit = max(max_turns, 5)
        hard_cap = soft_limit + 5
        self._conversation = AgentConversation(
            openai_client=openai_client,
            model=model,
            system_prompt=effective_prompt,
            tools=registry,
            stop_checker=stop_checker,
            max_turns=hard_cap,
            soft_turn_limit=soft_limit,
            reasoning={"effort": "medium", "summary": "detailed"},
            label="research",
            on_tool_call=on_tool_call,
            extra_tools=mcp_tools or [],
        )

    # ── BU V3 SDK web access ──────────────────────────────────────────

    async def _browse(self, url: str, task: str) -> Tuple[str, float]:
        """Browse the web via BU V3 SDK."""
        if not self.bu_client:
            return "Error: BU client not configured", 0.0

        bu_task = f"Navigate to: {url}\n\n{task}" if url else task

        try:
            text, bu_cost = await self.bu_client.research(bu_task)
            if len(text) > 4000:
                text = text[:4000] + "\n\n[Truncated to 4K chars]"
            return text, bu_cost
        except Exception as e:
            logger.warning(f"[research] browse error: {e}")
            return f"Browse error: {e}", 0.0

    # ── Tool registration ─────────────────────────────────────────────

    def _register_tools(self, registry: ToolRegistry) -> None:
        """Register research tools."""

        # --- browse: BU V3 SDK ---
        async def browse(args: Dict) -> tuple[str, float]:
            url = args.get("url", "")
            task = args.get("task", "")
            if not url and not task:
                return "Error: url or task is required", 0.0
            if url and not task:
                task = "Extract relevant information from this page."
            return await self._browse(url, task)

        registry.add(
            name="browse",
            description=(
                "Browse the web — visit a URL or search. "
                "Provide a URL to visit, or just a task to search the web."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "URL to visit (optional if task is a search query)",
                    },
                    "task": {
                        "type": "string",
                        "description": "What to find or extract",
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

        # --- respond ---
        async def respond(args: Dict) -> tuple[str, float]:
            content = args.get("content", "")
            self._responded = True
            self._response_text = content
            logger.info(f"[{self._conversation.label}] respond() called ({len(content)} chars)")
            return "Response recorded.", 0.0

        registry.add(
            name="respond",
            description=(
                "Submit your final response. The content parameter IS "
                "your answer — make it structured and complete."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": "Your research findings and answer",
                    },
                },
                "required": ["content"],
            },
            handler=respond,
        )

    async def ask(self, message: str) -> str:
        result = await self.ask_full(message)
        return result.text

    async def ask_full(self, message: str) -> AgentResult:
        self._responded = False
        self._response_text = ""

        result = await self._conversation.send(
            message,
            exit_condition=lambda: self._responded,
        )

        if self._response_text:
            result.text = self._response_text

        if not result.text:
            logger.warning(
                f"[{self._conversation.label}] finished without respond()"
            )
            result.text = self._extract_fallback_text()

        return result

    def _extract_fallback_text(self) -> str:
        for msg in reversed(self._conversation.messages):
            if not isinstance(msg, dict):
                continue
            if msg.get("type") == "message":
                content = msg.get("content", [])
                if isinstance(content, list):
                    texts = [
                        c.get("text", "") for c in content
                        if isinstance(c, dict) and c.get("type") == "output_text"
                    ]
                    if any(texts):
                        return "".join(texts)

        summaries = []
        for msg in self._conversation.messages:
            if isinstance(msg, dict) and msg.get("type") == "reasoning":
                for s in (msg.get("summary") or []):
                    if isinstance(s, dict) and s.get("text"):
                        summaries.append(s["text"])
        if summaries:
            return "\n\n".join(summaries[-3:])

        return ""

    @property
    def cost_usd(self) -> float:
        return self._conversation.total_cost

    async def cleanup(self) -> None:
        if self._sandbox_impl:
            try:
                await self._sandbox_impl.cleanup()
            except Exception as e:
                logger.warning(f"Research agent cleanup error: {e}")
