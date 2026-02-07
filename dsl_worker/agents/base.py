"""
Base agent conversation class.

Wraps the OpenAI Responses API with a tool-use loop, cost tracking,
and stop checking. All agent types (research, generator, orchestrator)
build on this.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from dsl_worker.agents.tools import ToolRegistry
from dsl_worker.billing.tracked_client import TrackedOpenAIClient

logger = logging.getLogger(__name__)

# Max characters to include from a tool result
TOOL_OUTPUT_LIMIT = 15_000


@dataclass
class AgentResult:
    """Result from an agent conversation turn or full run."""

    text: str = ""
    cost_usd: float = 0.0
    turns_taken: int = 0
    stopped: bool = False


class AgentConversation:
    """
    Manages a multi-turn conversation with an LLM agent that can use tools.

    Core loop:
    1. Send messages to OpenAI Responses API
    2. Parse response for text and function calls
    3. Execute function calls via ToolRegistry
    4. Append results to message history
    5. Repeat until no more tool calls (or exit condition met)

    Usage:
        tools = ToolRegistry()
        # ... register tools ...

        agent = AgentConversation(
            openai_client=tracked_client,
            model="gpt-5.2",
            system_prompt="You are a research agent.",
            tools=tools,
        )

        result = await agent.send("Research X topic")
        print(result.text)  # Agent's text response
        print(result.cost_usd)  # Total cost
    """

    def __init__(
        self,
        openai_client: TrackedOpenAIClient,
        model: str,
        system_prompt: str,
        tools: ToolRegistry,
        stop_checker: Optional[Callable[[], bool]] = None,
        max_turns: int = 100,
        max_output_tokens: int = 16_000,
    ) -> None:
        self.openai_client = openai_client
        self.model = model
        self.system_prompt = system_prompt
        self.tools = tools
        self.stop_checker = stop_checker
        self.max_turns = max_turns
        self.max_output_tokens = max_output_tokens

        self.messages: List[Dict[str, Any]] = []
        self.total_cost: float = 0.0
        self.total_turns: int = 0

    def _should_stop(self) -> bool:
        return self.stop_checker is not None and self.stop_checker()

    async def send(
        self,
        message: str,
        exit_condition: Optional[Callable[[], bool]] = None,
    ) -> AgentResult:
        """
        Send a message and run the agent loop until the agent responds
        with text only (no more tool calls), or exit_condition returns True.

        Args:
            message: User message to send.
            exit_condition: Optional callable that returns True to break
                           the loop early (checked after each tool execution).

        Returns:
            AgentResult with the agent's text response, cost, and turn count.
        """
        self.messages.append({"role": "user", "content": message})
        return await self._run_loop(exit_condition)

    async def step(
        self,
        exit_condition: Optional[Callable[[], bool]] = None,
    ) -> AgentResult:
        """
        Run one iteration of the agent loop without adding a new user message.
        Useful for continuing after injecting messages directly.
        """
        return await self._run_loop(exit_condition)

    def inject_message(self, role: str, content: str) -> None:
        """Add a message to the history without triggering a loop."""
        self.messages.append({"role": role, "content": content})

    async def _run_loop(
        self,
        exit_condition: Optional[Callable[[], bool]] = None,
    ) -> AgentResult:
        """Core agent loop. Calls API, handles tools, repeats."""
        result = AgentResult()

        for turn in range(self.max_turns):
            if self._should_stop():
                result.stopped = True
                break

            if exit_condition and exit_condition():
                break

            # Build input: system prompt + message history
            input_items = (
                [{"role": "system", "content": self.system_prompt}]
                + self.messages
            )

            try:
                response, cost = await self.openai_client.responses_create(
                    model=self.model,
                    input=input_items,
                    tools=self.tools.get_definitions() or None,
                    max_output_tokens=self.max_output_tokens,
                )
            except Exception as e:
                logger.error(f"API call failed: {e}", exc_info=True)
                # Append error so agent can try to recover next turn
                self.messages.append({
                    "role": "user",
                    "content": f"API error occurred: {e}. Please continue.",
                })
                continue

            self.total_cost += cost.total_cost_usd
            result.cost_usd = self.total_cost
            self.total_turns += 1
            result.turns_taken = self.total_turns

            # Parse response
            text_parts: list[str] = []
            tool_calls: list = []

            for item in response.output:
                if item.type == "message":
                    for content_block in item.content:
                        if hasattr(content_block, "text"):
                            text_parts.append(content_block.text)
                elif item.type == "function_call":
                    tool_calls.append(item)

            output_text = "".join(text_parts)

            if not tool_calls:
                # No tools called — agent is done for this turn
                if output_text:
                    self.messages.append({"role": "assistant", "content": output_text})
                result.text = output_text
                break

            # Agent made tool calls — execute them and continue the loop
            if output_text:
                self.messages.append({"role": "assistant", "content": output_text})

            for tc in tool_calls:
                # Record the function call in message history
                self.messages.append({
                    "type": "function_call",
                    "call_id": tc.call_id,
                    "name": tc.name,
                    "arguments": tc.arguments,
                })

                # Parse args and execute
                try:
                    args = json.loads(tc.arguments)
                except json.JSONDecodeError:
                    logger.warning(f"Bad tool args for {tc.name}: {tc.arguments}")
                    args = {}

                tool_result, tool_cost = await self.tools.execute(tc.name, args)
                self.total_cost += tool_cost
                result.cost_usd = self.total_cost

                # Record the tool output
                self.messages.append({
                    "type": "function_call_output",
                    "call_id": tc.call_id,
                    "output": tool_result[:TOOL_OUTPUT_LIMIT],
                })

                # Check exit/stop after each tool
                if self._should_stop():
                    result.stopped = True
                    break
                if exit_condition and exit_condition():
                    break

            # Capture any text the agent produced alongside tool calls
            result.text = output_text
        else:
            # Loop exhausted max_turns
            logger.warning(
                f"Agent loop hit max turns ({self.max_turns})"
            )

        return result
