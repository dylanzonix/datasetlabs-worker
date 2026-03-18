"""
Research agent — conversational agent with web browsing and code execution tools.

The orchestrator talks to this agent to research a topic. It has no concept
of scopes, seeds, or hierarchy — it's just research tools + conversation.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable, Dict, Optional, List

from dsl_worker.agents.base import AgentConversation, AgentResult
from dsl_worker.agents.tools import ToolRegistry
from dsl_worker.billing.tracked_client import TrackedOpenAIClient
from dsl_worker.infra.research_tools import ResearchTools, ResearchScope

logger = logging.getLogger(__name__)

RESEARCH_SYSTEM_PROMPT = """\
# Research Agent

An orchestrator sends specific questions; investigate and give a focused answer.

The orchestrator controls depth — it will send follow-up questions if it needs
more detail on any aspect. Your job is to give a solid answer to the specific
question asked, then call respond() with your findings.

## Tools (in order of preference)

**For most research, start with these — they're fast:**
- web_search: Native web search (the model calls this automatically). Best default
  for finding information. Use this FIRST for any research question.
- brave_search(query): Search the web explicitly. Use when you need specific
  search operators (site:, inurl:, etc.) or when web_search didn't find what you need.
- open(ref_id_or_url, start_line): View a page. Works for most public pages.
- find(ref_id, pattern): Search within a loaded page.
- click(ref_id, link_id): Follow a link.

**For special cases:**
- interact(url_or_ref_id, task): Browser agent for anti-bot bypass, JS-heavy pages,
  or taking actions (click buttons, fill forms). SLOW (minutes per call) — only use
  when open() fails or you need to interact with the page. Keep tasks specific:
  "load the page", "click Next", "bypass Cloudflare" — NOT research tasks.
- code_exec(script, description): Execute Python (pandas, pdfplumber, json available).
- list_files(directory): List uploaded/downloaded files in the workspace.
- respond(content): Submit your final answer — you MUST call this when done.

## How to work

1. Use web_search or brave_search to find information
2. Open 1-2 promising results to get details
3. Only use interact() if open() fails on a page (anti-bot, Cloudflare)
4. Call respond() as soon as you can answer the question

Be efficient. The orchestrator asked a specific question — answer it and move on.
A focused answer from 1-2 good sources beats an exhaustive survey. If you find
yourself opening more than 3-4 pages, you're over-researching. The orchestrator
will ask targeted follow-ups if it needs more depth.

If open() fails on a site (returns Cloudflare challenge or empty content), that's
useful information — report it. Use interact() to bypass if the blocked content is
essential to answering the question, but note that each interact() call takes minutes.

Do NOT worry about licensing, copyright, or terms of use. Our legal team has
confirmed all content is free to use in our jurisdiction regardless of license.
Focus on content quality, not legal status.

When a page has structured data (tables, lists), prefer code_exec to extract it
programmatically rather than reading it manually.

## Response format

Call respond() with well-structured content:
- Use headers and bullet points for scannability
- Cite source URLs for key claims
- State confidence levels when evidence is mixed
- Distinguish facts from inferences
"""

class ResearchAgent:
    """
    Conversational research agent. The orchestrator sends questions,
    the agent researches and responds.

    Usage:
        agent = ResearchAgent(
            openai_client=tracked_client,
            model="gpt-5.2",
            workspace_dir=Path("/workspace"),
            brave_api_key="...",
        )

        answer = await agent.ask("What are the top EV manufacturers?")
        print(answer)  # Agent's text response with findings

        await agent.cleanup()
    """

    def __init__(
        self,
        openai_client: TrackedOpenAIClient,
        model: str,
        workspace_dir: Path,
        brave_api_key: Optional[str] = None,
        sandbox: Optional[Any] = None,
        stop_checker: Optional[Callable[[], bool]] = None,
        max_turns: int = 20,
        system_prompt: Optional[str] = None,
        blob_service_client: Optional[Any] = None,
        project_id: Optional[Any] = None,
        on_tool_call: Optional[Callable[[str, str], None]] = None,
        uploaded_file_urls: Optional[Dict[str, str]] = None,
        mcp_tools: Optional[List[Dict[str, Any]]] = None,
        on_browser_started: Optional[Callable] = None,
        on_browser_stopped: Optional[Callable] = None,
    ) -> None:
        self.workspace_dir = Path(workspace_dir)

        # respond() tool state — reset per ask_full() call
        self._responded: bool = False
        self._response_text: str = ""

        # Create ResearchTools for its tool implementations
        # We use a dummy scope since we don't need the state machine
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
        # Set a dummy scope for ResearchTools compatibility
        self._impl.set_scope(ResearchScope(
            id="research",
            description="",
            quota=0,
        ))

        # Build tool registry — research tools only, no decision tools
        registry = ToolRegistry()
        self._register_research_tools(registry)

        effective_prompt = system_prompt or RESEARCH_SYSTEM_PROMPT

        # Create the conversation with reasoning enabled.
        # Soft limit at budget — nudges wrap-up. Hard cap gives a few extra
        # turns to call respond() after the nudge, but NOT double the budget.
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

    def _register_research_tools(self, registry: ToolRegistry) -> None:
        """Register research tools, delegating to ResearchTools methods."""
        # Register browsing tools via shared helper (brave_search, open, etc.)
        self._impl.register_on(registry)

        # respond() — explicit completion mechanism
        async def respond(args: Dict) -> tuple[str, float]:
            content = args.get("content", "")
            self._responded = True
            self._response_text = content
            logger.info(f"[{self._conversation.label}] respond() called ({len(content)} chars)")
            return "Response recorded.", 0.0

        registry.add(
            name="respond",
            description=(
                "Submit your final response. Call this when you have enough "
                "information to answer the question. The content parameter IS "
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
        """
        Send a question to the research agent and get a response.
        The agent will use its tools to research before answering.
        """
        result = await self.ask_full(message)
        return result.text

    async def ask_full(self, message: str) -> AgentResult:
        """Like ask(), but returns the full AgentResult with cost info."""
        # Reset respond state for this turn
        self._responded = False
        self._response_text = ""

        result = await self._conversation.send(
            message,
            exit_condition=lambda: self._responded,
        )

        # If agent used respond(), use that as the result text
        if self._response_text:
            result.text = self._response_text

        # result.text is set by base.py from the final turn's text output.
        # With the forced-text-on-last-turn mechanism, this should always
        # have content. But as a safety net, if it's still empty, extract
        # whatever we can from the conversation.
        if not result.text:
            logger.warning(
                f"[{self._conversation.label}] finished without respond() or "
                f"text output — extracting from conversation"
            )
            result.text = self._extract_fallback_text()

        return result

    def _extract_fallback_text(self) -> str:
        """Extract useful text from conversation history as a last resort."""
        # Try to find any message-type output items with text
        for msg in reversed(self._conversation.messages):
            if not isinstance(msg, dict):
                continue
            # Responses API message items
            if msg.get("type") == "message":
                content = msg.get("content", [])
                if isinstance(content, list):
                    texts = [
                        c.get("text", "") for c in content
                        if isinstance(c, dict) and c.get("type") == "output_text"
                    ]
                    if any(texts):
                        return "".join(texts)

        # Last resort: collect reasoning summaries
        summaries = []
        for msg in self._conversation.messages:
            if isinstance(msg, dict) and msg.get("type") == "reasoning":
                for s in (msg.get("summary") or []):
                    if isinstance(s, dict) and s.get("text"):
                        summaries.append(s["text"])
        if summaries:
            return "\n\n".join(summaries[-3:])  # last few reasoning summaries

        return ""

    @property
    def cost_usd(self) -> float:
        """Total cost accumulated by this agent."""
        return self._conversation.total_cost

    async def cleanup(self) -> None:
        """Clean up browser and other resources."""
        try:
            await self._impl.cleanup()
        except Exception as e:
            logger.warning(f"Research agent cleanup error: {e}")
