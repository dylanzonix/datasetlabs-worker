"""
Composable tool registry for agent conversations.

Supports flat function tools and namespaced tool groups with defer_loading.

Each tool has a JSON schema definition and an async handler function.
Handlers return (result_text, cost_usd).
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Awaitable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Tool handler signature: async (args: dict) -> (result_text: str, cost_usd: float)
ToolHandler = Callable[[Dict[str, Any]], Awaitable[Tuple[str, float]]]


class ToolRegistry:
    """
    Registry of tools available to an agent.

    Supports:
    - Flat function tools (always loaded)
    - Namespaced tool groups with defer_loading (loaded on demand via tool_search)
    - Built-in tools (web_search, tool_search — processed by OpenAI)

    Usage:
        registry = ToolRegistry()

        # Flat function tool
        registry.add("submit_row", "Submit the row", {...}, handler)

        # Namespaced tools (deferred)
        registry.add_namespace(
            name="fullenrich",
            description="Search people/companies, enrich contacts with verified emails and phones.",
            tools=[
                {"name": "search_people", "description": "...", "parameters": {...}},
                {"name": "search_companies", ...},
            ],
            handlers={"search_people": handler_fn, "search_companies": handler_fn},
        )

        # Get definitions for OpenAI:
        tools = registry.get_definitions()

        # Execute (handles both flat and namespaced):
        result, cost = await registry.execute("search_people", {"query": "..."})
    """

    def __init__(self, tool_budget: int = 0) -> None:
        self._definitions: Dict[str, Dict[str, Any]] = {}
        self._handlers: Dict[str, ToolHandler] = {}
        self._namespaces: Dict[str, Dict[str, Any]] = {}  # namespace_name -> definition
        self._tool_budget = tool_budget  # 0 = unlimited
        self._tool_calls_used = 0

    def add(
        self,
        name: str,
        description: str,
        parameters: Dict[str, Any],
        handler: ToolHandler,
    ) -> None:
        """Register a flat function tool (always loaded)."""
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

    def add_namespace(
        self,
        name: str,
        description: str,
        tools: List[Dict[str, Any]],
        handlers: Dict[str, ToolHandler],
    ) -> None:
        """Register a namespace of deferred tools.

        Tools within the namespace are marked defer_loading=true. The model
        sees only the namespace name + description until it uses tool_search
        to load specific tools.

        Args:
            name: Namespace name (e.g. "fullenrich", "apify")
            description: What this namespace provides (shown to model)
            tools: List of tool defs (name, description, parameters)
            handlers: Map of tool_name -> handler function
        """
        ns_tools = []
        for tool in tools:
            tool_def = {
                "type": "function",
                "name": tool["name"],
                "description": tool["description"],
                "parameters": tool.get("parameters", {"type": "object", "properties": {}}),
                "defer_loading": True,
            }
            if tool.get("additionalProperties"):
                tool_def["parameters"]["additionalProperties"] = True
            ns_tools.append(tool_def)
            # Register handler (keyed by tool name, not namespace-qualified)
            if tool["name"] in handlers:
                self._handlers[tool["name"]] = handlers[tool["name"]]

        self._namespaces[name] = {
            "type": "namespace",
            "name": name,
            "description": description,
            "tools": ns_tools,
        }

    def add_builtin(self, definition: Dict[str, Any]) -> None:
        """Add a non-function tool (e.g. web_search, tool_search) that OpenAI handles natively."""
        tool_type = definition.get("type", "")
        self._definitions[f"__builtin_{tool_type}"] = definition

    def get_definitions(self) -> List[Dict[str, Any]]:
        """Get all tool definitions in OpenAI format.

        Returns flat function tools + namespace definitions + built-in tools.
        """
        result = list(self._definitions.values())
        result.extend(self._namespaces.values())
        return result

    def has_tool(self, name: str) -> bool:
        return name in self._handlers

    async def execute(self, name: str, args: Dict[str, Any]) -> Tuple[str, float]:
        """Execute a tool by name.

        Handles both flat tools and namespaced tools (looked up by tool name).
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

        # Tool budget countdown
        if self._tool_budget > 0 and name not in ("respond", "done"):
            self._tool_calls_used += 1
            remaining = max(0, self._tool_budget - self._tool_calls_used)
            result_text += f"\n\n[{remaining} tool calls remaining]"

        return result_text, cost

    def merge(self, other: ToolRegistry) -> None:
        """Merge another registry's tools into this one."""
        self._definitions.update(other._definitions)
        self._handlers.update(other._handlers)
        self._namespaces.update(other._namespaces)

    def __len__(self) -> int:
        return len(self._definitions) + sum(
            len(ns.get("tools", [])) for ns in self._namespaces.values()
        )

    def __contains__(self, name: str) -> bool:
        return name in self._handlers
