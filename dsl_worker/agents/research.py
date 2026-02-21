"""
Research agent — conversational agent with web browsing and code execution tools.

The orchestrator talks to this agent to research a topic. It has no concept
of scopes, seeds, or hierarchy — it's just research tools + conversation.

V3: When given a SourceManager, the agent can also save research material
as persistent sources for later use by row generators.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable, Dict, Optional, List, TYPE_CHECKING

from dsl_worker.agents.base import AgentConversation, AgentResult
from dsl_worker.agents.tools import ToolRegistry
from dsl_worker.billing.tracked_client import TrackedOpenAIClient
from dsl_worker.phases.research_tools import ResearchTools, ResearchScope

if TYPE_CHECKING:
    from dsl_worker.phases.source_manager import SourceManager

logger = logging.getLogger(__name__)

RESEARCH_SYSTEM_PROMPT = """\
You are a research agent in a multi-agent system. An orchestrator sends you
specific questions; your job is to investigate and give a focused answer.

The orchestrator controls depth — it will send follow-up questions if it needs
more detail on any aspect. Your job is to give a solid answer to the specific
question asked, then call respond() with your findings.

## Tools

- brave_search(query, response_length): Search the web
- open(ref_id_or_url, start_line, response_length): View a page or file
- find(ref_id, pattern, response_length): Search within a loaded page
- click(ref_id, link_id, response_length): Follow a link
- list_files(directory): List uploaded/downloaded files in the workspace
- code_exec(script, description): Execute Python (pandas, pdfplumber, json available)
- interact(url_or_ref_id, task): Browser agent for complex interactions (forms, JS-heavy pages)
- respond(content): Submit your final answer — you MUST call this when done

## How to work

1. Search for information relevant to the question
2. Open 1-2 promising results to get what you need
3. Call respond() as soon as you can answer the question

Be efficient. The orchestrator asked a specific question — answer it and move on.
A focused answer from 1-2 good sources beats an exhaustive survey. If you find
yourself opening more than 3-4 pages, you're over-researching. The orchestrator
will ask targeted follow-ups if it needs more depth.

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

SAVE_SOURCE_PROMPT_ADDITION = """

## Saving Sources

Use save_source() to persist any valuable research material you find. This saves
the content as a file that row generators can later read when producing dataset rows.
Save broadly — it's better to have extra sources than to miss useful material.

Choose appropriate tags and authority scores:
- Authority: wiki/docs=0.9, expert blog=0.7, forum/community=0.5, random=0.3
- Tags: use short, descriptive labels relevant to the dataset topic
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
        mcp_tools: Optional[List[Dict[str, Any]]] = None,
        source_manager: Optional["SourceManager"] = None,
    ) -> None:
        self.workspace_dir = Path(workspace_dir)
        self._source_manager = source_manager

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

        # Build system prompt — add save_source instructions if source_manager provided
        effective_prompt = system_prompt or RESEARCH_SYSTEM_PROMPT
        if source_manager and not system_prompt:
            effective_prompt = RESEARCH_SYSTEM_PROMPT + SAVE_SOURCE_PROMPT_ADDITION

        # Create the conversation with reasoning enabled
        self._conversation = AgentConversation(
            openai_client=openai_client,
            model=model,
            system_prompt=effective_prompt,
            tools=registry,
            stop_checker=stop_checker,
            max_turns=max_turns,
            reasoning={"effort": "medium", "summary": "detailed"},
            label="research",
            on_tool_call=on_tool_call,
            extra_tools=mcp_tools or [],
        )

    def _register_research_tools(self, registry: ToolRegistry) -> None:
        """Register research tools, delegating to ResearchTools methods."""
        # Register browsing tools via shared helper (brave_search, open, etc.)
        self._impl.register_on(registry)

        # save_source — only if source_manager is provided
        if self._source_manager:
            source_mgr = self._source_manager

            async def save_source(args: Dict) -> tuple[str, float]:
                try:
                    result = await source_mgr.save_source(
                        content=args.get("content", ""),
                        path=args.get("path", ""),
                        description=args.get("description", ""),
                        tags=args.get("tags", []),
                        authority=args.get("authority", 0.5),
                        source_type=args.get("source_type", "article"),
                        url=args.get("url"),
                    )
                    return f"Source saved: {result}", 0.0
                except Exception as e:
                    return f"Error saving source: {e}", 0.0

            registry.add(
                name="save_source",
                description=(
                    "Save research material as a persistent source file. "
                    "Row generators will later read these sources when producing rows. "
                    "Save any valuable content you find during research."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "content": {
                            "type": "string",
                            "description": "The content to save (text, markdown, data, etc.)",
                        },
                        "path": {
                            "type": "string",
                            "description": "Relative path within sources/ (e.g., 'combat/wiki_peek.md')",
                        },
                        "description": {
                            "type": "string",
                            "description": "Brief description of what this source contains",
                        },
                        "tags": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Tags for filtering (e.g., ['combat', 'mechanics'])",
                        },
                        "authority": {
                            "type": "number",
                            "description": "Authority score 0-1 (wiki=0.9, expert=0.7, forum=0.5, random=0.3)",
                        },
                        "source_type": {
                            "type": "string",
                            "description": "Source category: wiki, forum, article, code, data, upload, etc.",
                        },
                        "url": {
                            "type": "string",
                            "description": "Original URL if this is web content (optional)",
                        },
                    },
                    "required": ["content", "path", "description", "tags", "authority", "source_type"],
                },
                handler=save_source,
            )

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

        return result

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
