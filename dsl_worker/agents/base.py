"""
Base agent conversation class.

Wraps the OpenAI Responses API with a tool-use loop, cost tracking,
and stop checking. All agent types (research, generator, orchestrator)
build on this.

Manages context manually — all messages (including reasoning items) are
stored in self.messages and replayed as input each turn. This gives full
debuggability and control over the conversation state.
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple

from dsl_worker.agents.tools import ToolRegistry
from dsl_worker.billing.tracked_client import TrackedOpenAIClient
from dsl_worker.utils import count_tokens

logger = logging.getLogger(__name__)

# Langfuse is optional — tracing is a no-op if not configured
try:
    from langfuse import get_client as _get_langfuse_client

    def _get_langfuse():
        try:
            return _get_langfuse_client()
        except Exception:
            return None
except ImportError:
    def _get_langfuse():
        return None

# Max characters to include from a tool result
TOOL_OUTPUT_LIMIT = 15_000


def _serialize_response_output(output_items) -> list:
    """Convert OpenAI response output items to plain dicts for Langfuse logging."""
    result = []
    for item in output_items:
        if item.type == "reasoning":
            summaries = [
                {"text": s.text}
                for s in (item.summary or [])
                if hasattr(s, "text")
            ]
            result.append({"type": "reasoning", "summary": summaries})
        elif item.type == "function_call":
            args = item.arguments
            if len(args) > 500:
                args = args[:500] + "…"
            result.append({"type": "function_call", "name": item.name, "arguments": args})
        elif item.type == "message":
            content = []
            for cb in item.content:
                if hasattr(cb, "text"):
                    text = cb.text
                    if len(text) > 2000:
                        text = text[:2000] + "…"
                    content.append({"type": "text", "text": text})
            result.append({"type": "message", "content": content})
    return result


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

    Manages context manually — all output items (reasoning, messages, tool calls)
    are captured into self.messages and replayed as input each turn. This preserves
    reasoning context across turns while keeping the full conversation inspectable.

    Core loop:
    1. Build input from system prompt + self.messages
    2. Send to OpenAI Responses API
    3. Capture all output items (reasoning, text, tool calls) into self.messages
    4. Execute function calls via ToolRegistry (parallel if multiple)
    5. Append tool outputs to self.messages
    6. Repeat until no more tool calls (or exit condition met)

    Usage:
        tools = ToolRegistry()
        # ... register tools ...

        agent = AgentConversation(
            openai_client=tracked_client,
            model="gpt-5.2",
            system_prompt="You are a research agent.",
            tools=tools,
            reasoning={"effort": "medium", "summary": "auto"},
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
        stop_event: Optional[asyncio.Event] = None,
        max_turns: int = 100,
        soft_turn_limit: int = 50,
        max_output_tokens: int = 16_000,
        reasoning: Optional[Dict[str, Any]] = None,
        label: str = "agent",
        continue_on_text: bool = False,
        context_window: int = 400_000,
        on_tool_call: Optional[Callable[[str, str], None]] = None,
        on_cost: Optional[Callable[[float, str], Awaitable[None]]] = None,
        extra_tools: Optional[List[Dict[str, Any]]] = None,
        langfuse_parent: Optional[Any] = None,
        on_idle: Optional[Callable[[], Awaitable[Optional[str]]]] = None,
    ) -> None:
        self.openai_client = openai_client
        self.model = model
        self.system_prompt = system_prompt
        self.tools = tools
        self.stop_checker = stop_checker
        self.stop_event = stop_event
        self.max_turns = max_turns
        self.soft_turn_limit = soft_turn_limit
        self.max_output_tokens = max_output_tokens
        self.reasoning = reasoning if reasoning is not None else {"effort": "medium", "summary": "detailed"}
        self.label = label
        self.continue_on_text = continue_on_text
        self.context_window = context_window
        self.on_tool_call = on_tool_call
        self.on_cost = on_cost
        # Extra tool definitions (e.g. MCP connectors) passed directly to API
        self.extra_tools = extra_tools or []
        # Explicit Langfuse parent span — avoids context-var inference issues
        # across asyncio.create_task() boundaries.
        self.langfuse_parent = langfuse_parent
        # on_idle: called when agent outputs text with no tool calls.
        # If set, blocks until it returns a string (injected as user message)
        # or None (signals the loop to exit). Takes priority over continue_on_text.
        self.on_idle = on_idle

        # Conversation state — this IS the context sent to the API each turn.
        # Contains user messages, reasoning items, assistant messages,
        # function_call items, and function_call_output items.
        self.messages: List[Dict[str, Any]] = []
        self.total_cost: float = 0.0
        self.total_turns: int = 0
        self._warned_soft_limit: bool = False
        # Active Langfuse span for this agent — set in _run_loop_traced,
        # used to create child generation/tool spans inside _run_loop_inner.
        self._current_langfuse_span: Any = None

    def _should_stop(self) -> bool:
        return self.stop_checker is not None and self.stop_checker()

    async def _api_call_with_stop_check(self, **kwargs):
        """Wrap an API call so we can detect stop/pause quickly.

        If a stop_event is provided, races the API call against it — any agent
        that detects a pause sets the event, and all other agents cancel
        immediately without waiting for their own poll cycle.

        Falls back to polling stop_checker every 2s if no event is provided.
        """
        api_task = asyncio.create_task(
            self.openai_client.responses_create(**kwargs)
        )

        if not self.stop_checker:
            return await api_task

        def _cancel_and_return_none():
            api_task.cancel()
            return None

        # If we have a stop_event, race the API call against it for instant wakeup.
        if self.stop_event is not None:
            stop_wait = asyncio.ensure_future(self.stop_event.wait())
            try:
                done, _ = await asyncio.wait(
                    {api_task, stop_wait},
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if stop_wait in done:
                    return _cancel_and_return_none()
                stop_wait.cancel()
                return await api_task
            except Exception:
                stop_wait.cancel()
                raise

        # Fallback: poll stop_checker every 2s while API call is in flight
        while not api_task.done():
            try:
                await asyncio.wait_for(asyncio.shield(api_task), timeout=2.0)
            except asyncio.TimeoutError:
                if self._should_stop():
                    return _cancel_and_return_none()
            except Exception:
                break

        return await api_task

    def _trim_context(self) -> None:
        """Drop oldest messages if total context would exceed the window limit."""
        # Reserve space for system prompt + output
        system_tokens = count_tokens(self.system_prompt)
        budget = self.context_window - system_tokens - self.max_output_tokens

        if budget <= 0 or not self.messages:
            return

        # Count tokens per message (json serialization as proxy)
        msg_tokens = []
        total = 0
        for msg in self.messages:
            t = count_tokens(json.dumps(msg, ensure_ascii=False))
            msg_tokens.append(t)
            total += t

        if total <= budget:
            return

        # Drop from the front until we fit
        dropped = 0
        while total > budget and dropped < len(self.messages):
            total -= msg_tokens[dropped]
            dropped += 1

        if dropped > 0:
            self.messages = self.messages[dropped:]
            logger.warning(
                f"[{self.label}] trimmed {dropped} oldest messages "
                f"to fit context window ({total} tokens remaining, "
                f"budget {budget})"
            )

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
        msg = {"role": "user", "content": message}
        self.messages.append(msg)
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
        msg = {"role": role, "content": content}
        self.messages.append(msg)

    async def _run_loop(
        self,
        exit_condition: Optional[Callable[[], bool]] = None,
    ) -> AgentResult:
        """Core agent loop. Calls API, handles tools, repeats."""
        # Use explicit parent if provided, otherwise fall back to context-var
        # inference via the global Langfuse client.
        parent = self.langfuse_parent or _get_langfuse()
        if parent:
            return await self._run_loop_traced(parent, exit_condition)
        return await self._run_loop_inner(exit_condition)

    async def _run_loop_traced(
        self,
        langfuse_parent,
        exit_condition: Optional[Callable[[], bool]] = None,
    ) -> AgentResult:
        """Wrapper that creates a Langfuse span around the agent loop.

        Calls start_as_current_observation on the parent — this creates
        an explicit child span AND sets it as the current observation in
        contextvars, so the OpenAI auto-wrapper and tool spans nest
        correctly underneath.
        """
        with langfuse_parent.start_as_current_observation(
            as_type="span",
            name=self.label,
            metadata={"model": self.model, "max_turns": self.max_turns},
        ) as span:
            self._current_langfuse_span = span
            result = await self._run_loop_inner(exit_condition)
            try:
                span.update(
                    output=result.text[:2000] if result.text else None,
                    metadata={
                        "total_cost_usd": round(result.cost_usd, 6),
                        "total_turns": result.turns_taken,
                        "stopped": result.stopped,
                    },
                )
            except Exception:
                pass
            self._current_langfuse_span = None
            return result

    async def _call_api_with_trace(self, turn: int, input_items: list, **kwargs):
        """Make one LLM API call, logging a Langfuse generation span if active.

        Returns same as _api_call_with_stop_check: (response, cost) or None.

        Uses _get_langfuse() (not span.start_as_current_observation) so that
        the SDK's context var — set when _run_loop_traced entered the agent span
        — automatically parents new observations correctly.
        Langfuse failures are completely non-fatal: agent continues untraced.
        """
        if self._current_langfuse_span is None:
            return await self._api_call_with_stop_check(input=input_items, **kwargs)

        lf = _get_langfuse()
        if lf is None:
            return await self._api_call_with_stop_check(input=input_items, **kwargs)

        try:
            obs_ctx = lf.start_as_current_observation(
                as_type="generation",
                name=f"{self.label}:llm",
                model=self.model,
                input=input_items,
            )
        except Exception:
            return await self._api_call_with_stop_check(input=input_items, **kwargs)

        with obs_ctx as gen_obs:
            result = await self._api_call_with_stop_check(input=input_items, **kwargs)
            if result is not None:
                try:
                    response, cost = result
                    usage_details = None
                    if hasattr(response, "usage") and response.usage:
                        usage_details = {
                            "input": getattr(response.usage, "input_tokens", 0),
                            "output": getattr(response.usage, "output_tokens", 0),
                        }
                    gen_obs.update(
                        output=_serialize_response_output(response.output),
                        usage_details=usage_details,
                        metadata={"cost_usd": round(cost.total_cost_usd, 6)},
                    )
                except Exception:
                    pass
            return result

    async def _run_loop_inner(
        self,
        exit_condition: Optional[Callable[[], bool]] = None,
    ) -> AgentResult:
        """Core agent loop. Calls API, handles tools, repeats."""
        result = AgentResult()

        for turn in range(self.max_turns):
            if self._should_stop():
                logger.info(f"[{self.label}] stopped by stop_checker at turn {turn}")
                result.stopped = True
                break

            if exit_condition and exit_condition():
                logger.info(f"[{self.label}] exit_condition met at turn {turn}")
                break

            # Trim oldest messages if approaching context window limit
            self._trim_context()

            # Soft limit warning — nudge the agent to wrap up, but don't
            # strip tools or force anything. It can keep going if needed.
            if turn == self.soft_turn_limit and not self._warned_soft_limit:
                self._warned_soft_limit = True
                logger.info(f"[{self.label}] soft turn limit ({self.soft_turn_limit}) — injecting wrap-up nudge")
                self.messages.append({
                    "role": "user",
                    "content": (
                        "WRAP UP NOW. You are at your turn budget. Call respond() "
                        "immediately with your findings so far. Do not make more "
                        "tool calls — submit what you have."
                    ),
                })

            # Always build full input — system prompt + all messages
            input_items = (
                [{"role": "system", "content": self.system_prompt}]
                + self.messages
            )
            logger.info(f"[{self.label}] turn {turn} — {len(input_items)} input items, ${self.total_cost:.4f} spent")

            # Build kwargs
            create_kwargs: Dict[str, Any] = {}
            if self.reasoning is not None:
                create_kwargs["reasoning"] = self.reasoning

            # Merge function tools with extra tools (MCP connectors, etc.)
            all_tools = (self.tools.get_definitions() or []) + self.extra_tools

            try:
                api_result = await self._call_api_with_trace(
                    turn=turn,
                    input_items=input_items,
                    model=self.model,
                    tools=all_tools or None,
                    max_output_tokens=self.max_output_tokens,
                    **create_kwargs,
                )
                if api_result is None:
                    # Cancelled by stop_checker
                    logger.info(f"[{self.label}] API call cancelled by stop_checker at turn {turn}")
                    result.stopped = True
                    break
                response, cost = api_result
            except Exception as e:
                logger.error(f"API call failed: {e}", exc_info=True)
                self.messages.append({
                    "role": "user",
                    "content": f"API error occurred: {e}. Please continue.",
                })
                continue

            self.total_cost += cost.total_cost_usd
            result.cost_usd = self.total_cost
            if self.on_cost and cost.total_cost_usd > 0:
                await self.on_cost(cost.total_cost_usd, self.label)
            self.total_turns += 1
            result.turns_taken = self.total_turns

            # Log cached token counts for prompt caching verification
            if hasattr(response, "usage") and response.usage and logger.isEnabledFor(logging.DEBUG):
                usage = response.usage
                input_tokens = getattr(usage, "input_tokens", 0)
                cached = getattr(usage, "input_tokens_details", None)
                cached_tokens = getattr(cached, "cached_tokens", 0) if cached else 0
                if input_tokens > 0:
                    logger.debug(
                        f"[{self.label}] turn {turn} cache stats: "
                        f"{cached_tokens}/{input_tokens} input tokens cached "
                        f"({cached_tokens * 100 // input_tokens}%)"
                    )

            # Parse response — capture ALL output items into self.messages
            # using model_dump() to preserve every field the API needs.
            # Order is critical: reasoning MUST be followed by its associated
            # message or function_call (linked by internal IDs).
            text_parts: list[str] = []
            tool_calls: list = []

            for item in response.output:
                # Store the complete item for replay.
                # Reasoning items need special handling: the SDK's
                # construct() + field_get_default() can turn the required
                # `summary` field into None when the API returns null,
                # and model_dump(exclude_none=True) then strips it entirely.
                if item.type == "reasoning":
                    summary = []
                    if item.summary:
                        summary = [
                            {"type": s.type, "text": s.text}
                            for s in item.summary
                        ]
                    dumped: dict = {
                        "type": "reasoning",
                        "id": item.id,
                        "summary": summary,
                    }
                    if logger.isEnabledFor(logging.DEBUG):
                        logger.debug(
                            f"Reasoning item: summary has {len(summary)} entries"
                        )
                else:
                    dumped = item.model_dump(exclude_none=True)
                self.messages.append(dumped)

                # Also extract what we need locally
                if item.type == "message":
                    for content_block in item.content:
                        if hasattr(content_block, "text"):
                            text_parts.append(content_block.text)
                elif item.type == "function_call":
                    tool_calls.append(item)

            output_text = "".join(text_parts)

            if not tool_calls:
                preview = output_text[:120].replace("\n", " ")
                logger.info(f"[{self.label}] turn {turn} — text response ({len(output_text)} chars): {preview}")
                result.text = output_text

                if self.on_idle is not None:
                    # Event-driven mode: block until an event arrives or exit.
                    # Takes priority over continue_on_text.
                    event_msg = await self.on_idle()
                    if event_msg is None:
                        break  # on_idle signals exit
                    self.messages.append({
                        "role": "user",
                        "content": event_msg,
                    })
                    continue
                elif self.continue_on_text:
                    # Don't break — inject a continuation prompt so the agent
                    # keeps working (e.g. orchestrator thinking before acting).
                    self.messages.append({
                        "role": "user",
                        "content": "Continue.",
                    })
                    continue

                break

            # Execute tools and append function_call_output items
            # (function_call items are already in self.messages from parsing above)
            tool_names = [tc.name for tc in tool_calls]
            logger.info(f"[{self.label}] turn {turn} — {len(tool_calls)} tool call(s): {', '.join(tool_names)}")

            if len(tool_calls) > 1:
                await self._execute_tools_parallel(tool_calls, result)
            else:
                tc = tool_calls[0]
                result_text, tool_cost = await self._execute_tool(tc)
                self.total_cost += tool_cost
                result.cost_usd = self.total_cost
                if self.on_cost and tool_cost > 0:
                    await self.on_cost(tool_cost, self.label)

                self.messages.append({
                    "type": "function_call_output",
                    "call_id": tc.call_id,
                    "output": result_text[:TOOL_OUTPUT_LIMIT],
                })

            # Check exit/stop after tool execution
            if self._should_stop():
                logger.info(f"[{self.label}] stopped by stop_checker after tools at turn {turn}")
                result.stopped = True
                break
            if exit_condition and exit_condition():
                logger.info(f"[{self.label}] exit_condition met after tools at turn {turn}")
                break

            # Capture any text the agent produced alongside tool calls
            result.text = output_text
        else:
            # Loop exhausted max_turns
            logger.warning(
                f"[{self.label}] hit max turns ({self.max_turns})"
            )

        logger.info(
            f"[{self.label}] loop done — {self.total_turns} turns, "
            f"${self.total_cost:.4f}, {len(self.messages)} messages"
        )

        return result

    async def _execute_tool(self, tc) -> Tuple[str, float]:
        """Execute a single tool call. Returns (result_text, cost)."""
        if self.on_tool_call:
            self.on_tool_call(self.label, tc.name)

        try:
            args = json.loads(tc.arguments)
        except json.JSONDecodeError:
            logger.warning(f"Bad tool args for {tc.name}: {tc.arguments[:500]}")
            return (
                f"Error: invalid JSON in tool arguments for {tc.name}. "
                f"Your arguments could not be parsed. Please retry with valid JSON.",
                0.0,
            )

        if self._current_langfuse_span is None:
            return await self.tools.execute(tc.name, args)

        lf = _get_langfuse()
        if lf is None:
            return await self.tools.execute(tc.name, args)

        try:
            obs_ctx = lf.start_as_current_observation(
                as_type="span",
                name=f"tool:{tc.name}",
                input=args,
            )
        except Exception:
            return await self.tools.execute(tc.name, args)

        with obs_ctx as tool_obs:
            result_text, cost = await self.tools.execute(tc.name, args)
            try:
                preview = result_text if len(result_text) <= 1000 else result_text[:1000] + "…"
                tool_obs.update(
                    output=preview,
                    metadata={"cost_usd": round(cost, 6), "output_len": len(result_text)},
                )
            except Exception:
                pass
            return result_text, cost

    async def _execute_tools_parallel(
        self, tool_calls: list, result: AgentResult,
    ) -> None:
        """Execute multiple tool calls concurrently.

        Note: function_call items are already in self.messages from parsing.
        This method only appends the function_call_output items.
        """

        async def run_one(tc):
            result_text, tool_cost = await self._execute_tool(tc)
            return tc, result_text, tool_cost

        results = await asyncio.gather(
            *[run_one(tc) for tc in tool_calls],
            return_exceptions=True,
        )

        for i, r in enumerate(results):
            tc = tool_calls[i]

            if isinstance(r, BaseException):
                logger.error(f"Parallel tool error for {tc.name}: {r}")
                output_text = f"Error executing tool: {r}"
            else:
                _, output_text, tool_cost = r
                self.total_cost += tool_cost
                result.cost_usd = self.total_cost
                if self.on_cost and tool_cost > 0:
                    await self.on_cost(tool_cost, self.label)
                output_text = output_text[:TOOL_OUTPUT_LIMIT]

            self.messages.append({
                "type": "function_call_output",
                "call_id": tc.call_id,
                "output": output_text,
            })
