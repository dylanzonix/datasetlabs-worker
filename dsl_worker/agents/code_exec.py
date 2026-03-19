"""
Code Execution agent — lightweight subagent for file/data investigation.

V10: Used by orchestrator for:
- Inspecting uploaded files (schemas, row counts, column types)
- Exploring data structures
- Running analysis scripts

No browser, no search — just code execution and file access.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from dsl_worker.agents.base import AgentConversation, AgentResult
from dsl_worker.agents.tools import ToolRegistry
from dsl_worker.billing.tracked_client import TrackedOpenAIClient

logger = logging.getLogger(__name__)


CODE_EXEC_SYSTEM_PROMPT = """\
# Code Execution Agent

You investigate files and data by running Python code.

## Tools

- code_exec(script, description): Execute Python. pandas, json, csv, openpyxl, \
pdfplumber are available. Files are at /workspace/uploads/.
- list_files(directory): List files in a directory.
- read_file(path): Read a file's contents.
- respond(content): Submit your findings — call this when done.

## How to work

1. Understand the question.
2. Use code_exec to inspect files, parse data, or compute answers.
3. Call respond() with structured findings.

Be efficient. Answer the specific question asked.
"""


class CodeExecAgent:
    """
    Lightweight agent for file/data investigation via code execution.

    Usage:
        agent = CodeExecAgent(
            openai_client=tracked_client,
            model="gpt-5.2",
            workspace_dir=Path("/workspace"),
            sandbox=sandbox_client,
        )
        answer = await agent.ask("What columns are in /workspace/uploads/data.csv?")
        await agent.cleanup()
    """

    def __init__(
        self,
        openai_client: TrackedOpenAIClient,
        model: str,
        workspace_dir: Path,
        sandbox: Optional[Any] = None,
        stop_checker: Optional[Callable[[], bool]] = None,
        max_turns: int = 10,
        blob_service_client: Optional[Any] = None,
        project_id: Optional[Any] = None,
        uploaded_file_urls: Optional[Dict[str, str]] = None,
        on_tool_call: Optional[Callable[[str, str], None]] = None,
    ) -> None:
        self.workspace_dir = Path(workspace_dir)
        self._responded = False
        self._response_text = ""

        from dsl_worker.infra.research_tools import ResearchTools, ResearchScope

        self._impl = ResearchTools(
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
        self._impl.set_scope(ResearchScope(
            id="code_exec",
            description="",
            quota=0,
        ))

        registry = ToolRegistry()
        self._register_tools(registry)

        soft_limit = max(max_turns, 5)
        hard_cap = soft_limit + 3
        self._conversation = AgentConversation(
            openai_client=openai_client,
            model=model,
            system_prompt=CODE_EXEC_SYSTEM_PROMPT,
            tools=registry,
            stop_checker=stop_checker,
            max_turns=hard_cap,
            soft_turn_limit=soft_limit,
            reasoning={"effort": "low", "summary": "auto"},
            label="code_exec",
            on_tool_call=on_tool_call,
        )

    def _register_tools(self, registry: ToolRegistry) -> None:
        # Only register code_exec and list_files from ResearchTools
        self._impl.register_on(
            registry,
            exclude=[
                "brave_search", "open", "find", "click",
                "interact", "shell_exec",
            ],
            include_builtins=False,
        )

        # respond() — explicit completion
        async def respond(args: Dict) -> tuple[str, float]:
            content = args.get("content", "")
            self._responded = True
            self._response_text = content
            return "Response recorded.", 0.0

        registry.add(
            name="respond",
            description="Submit your findings. Call when you have the answer.",
            parameters={
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": "Your findings",
                    },
                },
                "required": ["content"],
            },
            handler=respond,
        )

    async def ask(self, question: str) -> str:
        result = await self.ask_full(question)
        return result.text

    async def ask_full(self, question: str) -> AgentResult:
        self._responded = False
        self._response_text = ""

        result = await self._conversation.send(
            question,
            exit_condition=lambda: self._responded,
        )

        if self._response_text:
            result.text = self._response_text

        return result

    @property
    def cost_usd(self) -> float:
        return self._conversation.total_cost

    async def cleanup(self) -> None:
        try:
            await self._impl.cleanup()
        except Exception as e:
            logger.warning(f"CodeExecAgent cleanup error: {e}")
