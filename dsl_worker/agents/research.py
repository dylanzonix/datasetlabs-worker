"""
Research agent — conversational agent with web browsing and code execution tools.

The orchestrator talks to this agent to research a topic. It has no concept
of scopes, seeds, or hierarchy — it's just research tools + conversation.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from dsl_worker.agents.base import AgentConversation, AgentResult
from dsl_worker.agents.tools import ToolRegistry
from dsl_worker.billing.tracked_client import TrackedOpenAIClient
from dsl_worker.phases.research_tools import ResearchTools, ResearchScope

logger = logging.getLogger(__name__)

RESEARCH_SYSTEM_PROMPT = """\
You are a research agent. Your job is to thoroughly research topics using your tools.

Available tools:
- brave_search: Search the web
- open: Open a URL or view a page you've already loaded
- find: Search within a loaded page
- click: Follow a link from a loaded page
- interact: Use browser agent for complex interactions (forms, JS-heavy pages)
- list_files: List available files in the workspace
- code_exec: Execute Python code (pandas, pdfplumber available)
- note: Record observations and findings

Guidelines:
- Research thoroughly before drawing conclusions
- Cross-reference multiple sources when possible
- Use note() freely to record important facts
- Open and read full pages when snippets aren't enough
- Use code_exec to process data files (CSV, Excel, PDF extraction)
- When done, provide a clear summary of your findings
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
        self.notes: List[str] = []

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
        # Set a dummy scope so note() works
        self._impl.set_scope(ResearchScope(
            id="research",
            description="",
            quota=0,
        ))

        # Build tool registry — research tools only, no decision tools
        registry = ToolRegistry()
        self._register_research_tools(registry)

        # Create the conversation
        self._conversation = AgentConversation(
            openai_client=openai_client,
            model=model,
            system_prompt=system_prompt or RESEARCH_SYSTEM_PROMPT,
            tools=registry,
            stop_checker=stop_checker,
            max_turns=max_turns,
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

        async def note(args: Dict) -> tuple[str, float]:
            content = args.get("content", "")
            self.notes.append(content)
            # Also delegate to impl so scope.notes stays in sync
            return impl.note(content=content)

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
            "note": note,
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
            # Skip conclude_research — not used in new model

    async def ask(self, message: str) -> str:
        """
        Send a question to the research agent and get a response.
        The agent will use its tools to research before answering.
        """
        result = await self._conversation.send(message)
        return result.text

    async def ask_full(self, message: str) -> AgentResult:
        """Like ask(), but returns the full AgentResult with cost info."""
        return await self._conversation.send(message)

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
