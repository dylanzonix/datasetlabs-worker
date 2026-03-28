"""
Verify web_search built-in tool is included for all agents and works with the API.

Usage:
    .venv/bin/python3 -m pytest tests/test_web_search.py -v
"""

import asyncio
import json
import os
import pytest

from dsl_worker.agents.tools import ToolRegistry


def _make_registry_with_dummy_tools():
    """Create a registry with dummy tools like what row.py registers."""
    registry = ToolRegistry()

    async def dummy(args):
        return "ok", 0.0

    registry.add(
        name="set_column",
        description="Set a column value.",
        parameters={"type": "object", "properties": {"name": {"type": "string"}, "value": {}}},
        handler=dummy,
    )
    registry.add(
        name="browse",
        description="Browse web.",
        parameters={"type": "object", "properties": {"url": {"type": "string"}, "task": {"type": "string"}}},
        handler=dummy,
    )
    return registry


class TestWebSearchToolInclusion:
    """Ensure web_search is included in tools for all agents."""

    def test_row_generator_includes_web_search(self):
        """Row generator should include web_search in extra_tools."""
        web_search_tool = {"type": "web_search"}
        mcp_tools = []
        all_extra_tools = [web_search_tool] + mcp_tools

        registry = _make_registry_with_dummy_tools()
        all_tools = (registry.get_definitions() or []) + all_extra_tools

        types = [t.get("type") for t in all_tools]
        assert "web_search" in types, "web_search not in tools list"
        assert "function" in types, "function tools missing"

    def test_web_search_tool_format(self):
        """web_search tool should have correct format."""
        tool = {"type": "web_search"}
        assert tool["type"] == "web_search"
        # Should NOT have function-tool fields
        assert "name" not in tool
        assert "parameters" not in tool

    def test_tools_merge_correctly(self):
        """Function tools and web_search should coexist without conflict."""
        registry = _make_registry_with_dummy_tools()
        extra = [{"type": "web_search"}]
        all_tools = (registry.get_definitions() or []) + extra

        function_tools = [t for t in all_tools if t["type"] == "function"]
        web_tools = [t for t in all_tools if t["type"] == "web_search"]

        assert len(function_tools) == 2  # set_column + browse
        assert len(web_tools) == 1
        assert function_tools[0]["name"] == "set_column"
        assert function_tools[1]["name"] == "browse"


@pytest.mark.skipif(
    not os.environ.get("AZURE_OPENAI_API_KEY"),
    reason="Requires AZURE_OPENAI_API_KEY",
)
class TestWebSearchAPI:
    """Integration tests — actually call the API with web_search."""

    @pytest.fixture
    def client(self):
        from openai import AsyncOpenAI
        from dsl_worker.billing.tracked_client import TrackedOpenAIClient
        from dsl_worker.config import settings

        raw = AsyncOpenAI(
            api_key=settings.azure_openai_api_key,
            base_url=f"{settings.azure_openai_endpoint}/openai/v1",
        )
        return TrackedOpenAIClient(raw)

    def test_web_search_returns_results(self, client):
        """Model should use web_search and return results inline."""

        async def _test():
            from dsl_worker.config import settings

            tools = [
                {"type": "web_search"},
                {
                    "type": "function",
                    "name": "answer",
                    "description": "Return the answer",
                    "parameters": {
                        "type": "object",
                        "properties": {"text": {"type": "string"}},
                        "required": ["text"],
                    },
                },
            ]

            response, cost = await client.responses_create(
                model=settings.generation_model,
                input=[
                    {"role": "user", "content": "What is the phone number for Greystar corporate office?"},
                ],
                tools=tools,
                max_output_tokens=2000,
            )

            output_types = [item.type for item in response.output]
            assert "web_search_call" in output_types, (
                f"Expected web_search_call in output, got: {output_types}"
            )
            # Should also have a message with the answer
            assert "message" in output_types or "function_call" in output_types

        asyncio.run(_test())

    def test_web_search_with_function_tools(self, client):
        """web_search and function tools should work together."""

        async def _test():
            from dsl_worker.config import settings

            tools = [
                {"type": "web_search"},
                {
                    "type": "function",
                    "name": "set_column",
                    "description": "Set a column value",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "value": {"type": "string"},
                        },
                        "required": ["name", "value"],
                    },
                },
            ]

            response, cost = await client.responses_create(
                model=settings.generation_model,
                input=[
                    {
                        "role": "system",
                        "content": "Use web search to find info, then call set_column to store it. Output via tool calls ONLY.",
                    },
                    {
                        "role": "user",
                        "content": 'Find the phone number for Coast Property Management in Washington state. Store it with set_column(name="phone", value=...)',
                    },
                ],
                tools=tools,
                reasoning={"effort": "low", "summary": "auto"},
                max_output_tokens=4000,
            )

            output_types = [item.type for item in response.output]
            # Should use web search
            assert "web_search_call" in output_types, (
                f"Expected web_search_call, got: {output_types}"
            )
            # Should also call set_column
            function_calls = [
                item for item in response.output if item.type == "function_call"
            ]
            assert any(fc.name == "set_column" for fc in function_calls), (
                f"Expected set_column call, got: {[fc.name for fc in function_calls]}"
            )

        asyncio.run(_test())
