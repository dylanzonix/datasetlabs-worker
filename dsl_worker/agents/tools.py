"""
Composable tool registry for agent conversations.

Each tool has a JSON schema definition and an async handler function.
Handlers return (result_text, cost_usd).
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Awaitable, Dict, List, Tuple

logger = logging.getLogger(__name__)

# Tool handler signature: async (args: dict) -> (result_text: str, cost_usd: float)
ToolHandler = Callable[[Dict[str, Any]], Awaitable[Tuple[str, float]]]


class ToolRegistry:
    """
    Registry of tools available to an agent.

    Usage:
        registry = ToolRegistry()

        @registry.register(
            name="brave_search",
            description="Search the web",
            parameters={
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        )
        async def brave_search(args):
            result = await do_search(args["query"])
            return result, 0.0

        # Or register without decorator:
        registry.add("note", note_schema, note_handler)

        # Get definitions for OpenAI:
        tools = registry.get_definitions()

        # Execute:
        result, cost = await registry.execute("brave_search", {"query": "test"})
    """

    def __init__(self, tool_budget: int = 0) -> None:
        self._definitions: Dict[str, Dict[str, Any]] = {}
        self._handlers: Dict[str, ToolHandler] = {}
        self._tool_budget = tool_budget  # 0 = unlimited
        self._tool_calls_used = 0

    def add(
        self,
        name: str,
        description: str,
        parameters: Dict[str, Any],
        handler: ToolHandler,
    ) -> None:
        """Register a tool with its schema and handler."""
        self._definitions[name] = {
            "type": "function",
            "name": name,
            "description": description,
            "parameters": parameters,
        }
        self._handlers[name] = handler

    def register(
        self,
        name: str,
        description: str,
        parameters: Dict[str, Any],
    ) -> Callable[[ToolHandler], ToolHandler]:
        """Decorator to register a tool handler."""

        def decorator(fn: ToolHandler) -> ToolHandler:
            self.add(name, description, parameters, fn)
            return fn

        return decorator

    def add_builtin(self, definition: Dict[str, Any]) -> None:
        """Add a non-function tool (e.g. web_search) that OpenAI handles natively."""
        # These don't have handlers — OpenAI processes them internally
        tool_type = definition.get("type", "")
        self._definitions[f"__builtin_{tool_type}"] = definition

    def get_definitions(self) -> List[Dict[str, Any]]:
        """Get all tool definitions in OpenAI format."""
        return list(self._definitions.values())

    def has_tool(self, name: str) -> bool:
        return name in self._handlers

    async def execute(self, name: str, args: Dict[str, Any]) -> Tuple[str, float]:
        """
        Execute a tool by name.

        Returns:
            (result_text, cost_usd)

        Raises:
            KeyError if tool not found.
        """
        handler = self._handlers.get(name)
        if handler is None:
            logger.warning(f"Unknown tool called: {name}")
            return f"Unknown tool: {name}", 0.0

        try:
            result_text, cost = await handler(args)
        except Exception as e:
            logger.error(f"Tool {name} failed: {e}", exc_info=True)
            result_text, cost = f"Tool error: {e}", 0.0

        # Tool budget countdown (skip for respond/done — those are completion signals)
        if self._tool_budget > 0 and name not in ("respond", "done"):
            self._tool_calls_used += 1
            remaining = max(0, self._tool_budget - self._tool_calls_used)
            result_text += f"\n\n[{remaining} tool calls remaining]"

        return result_text, cost

    def merge(self, other: ToolRegistry) -> None:
        """Merge another registry's tools into this one."""
        self._definitions.update(other._definitions)
        self._handlers.update(other._handlers)

    def __len__(self) -> int:
        return len(self._definitions)

    def __contains__(self, name: str) -> bool:
        return name in self._handlers
