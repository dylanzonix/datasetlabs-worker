"""
Research agent — conversational agent with web browsing and code execution tools.

The orchestrator talks to this agent to research a topic. It has no concept
of scopes, seeds, or hierarchy — it's just research tools + conversation.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from dsl_worker.agents.base import AgentConversation, AgentResult
from dsl_worker.agents.tools import ToolRegistry
from dsl_worker.billing.tracked_client import TrackedOpenAIClient
from dsl_worker.phases.research_tools import ResearchTools, ResearchScope

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
2. Open promising results and cross-reference key claims across 2-3 sources
3. When you have enough to give a useful answer, call respond() with structured findings

Do not over-research. A focused answer from 2-4 good sources is better than an
exhaustive survey. If something is unclear or you can't find reliable information,
say so — the orchestrator can ask targeted follow-ups.

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
        max_turns: int = 50,
        system_prompt: Optional[str] = None,
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

        # Create the conversation with reasoning enabled
        self._conversation = AgentConversation(
            openai_client=openai_client,
            model=model,
            system_prompt=system_prompt or RESEARCH_SYSTEM_PROMPT,
            tools=registry,
            stop_checker=stop_checker,
            max_turns=max_turns,
            reasoning={"effort": "medium", "summary": "detailed"},
            label="research",
        )

    def _register_research_tools(self, registry: ToolRegistry) -> None:
        """Register research tools, delegating to ResearchTools methods."""
        impl = self._impl

        async def brave_search(args: Dict) -> tuple[str, float]:
            return await impl.brave_search(
                query=args.get("query", ""),
                response_length=args.get("response_length", "medium"),
            )

        async def open_page(args: Dict) -> tuple[str, float]:
            return await impl.open(
                ref_id_or_url=args.get("ref_id_or_url", ""),
                start_line=args.get("start_line", 0),
                response_length=args.get("response_length", "medium"),
            )

        async def find(args: Dict) -> tuple[str, float]:
            return await impl.find(
                ref_id=args.get("ref_id", ""),
                pattern=args.get("pattern", ""),
                response_length=args.get("response_length", "medium"),
            )

        async def click(args: Dict) -> tuple[str, float]:
            return await impl.click(
                ref_id=args.get("ref_id", ""),
                link_id=args.get("link_id", 0),
                response_length=args.get("response_length", "medium"),
            )

        async def list_files(args: Dict) -> tuple[str, float]:
            return await impl.list_files(
                directory=args.get("directory", "all"),
            )

        async def code_exec(args: Dict) -> tuple[str, float]:
            return await impl.code_exec(
                script=args.get("script", ""),
                description=args.get("description", ""),
            )

        async def interact(args: Dict) -> tuple[str, float]:
            return await impl.interact(
                url_or_ref_id=args.get("url_or_ref_id", ""),
                task=args.get("task", ""),
            )

        # Get tool definitions from ResearchTools (research phase only)
        defs = impl.get_tool_definitions(phase="research")

        # Map tool name -> (handler, definition)
        handlers = {
            "brave_search": brave_search,
            "open": open_page,
            "find": find,
            "click": click,
            "list_files": list_files,
            "code_exec": code_exec,
            "interact": interact,
        }

        for defn in defs:
            name = defn.get("name")
            if name in handlers:
                registry.add(
                    name=name,
                    description=defn.get("description", ""),
                    parameters=defn.get("parameters", {}),
                    handler=handlers[name],
                )
            # Skip conclude_research, note — not used in new model

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
