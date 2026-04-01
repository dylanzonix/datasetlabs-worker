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
        elif item.type == "web_search_call":
            action = getattr(item, "action", None)
            action_type = getattr(action, "type", "?") if action else "?"
            entry = {"type": "web_search_call", "action_type": action_type}
            if action_type == "search":
                entry["query"] = getattr(action, "query", "") or ""
            elif action_type == "open_page":
                entry["url"] = getattr(action, "url", "") or ""
            elif action_type == "find_in_page":
                entry["query"] = getattr(action, "query", "") or ""
            result.append(entry)
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
            reasoning={"effort": "medium", "summary": "detailed"},
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
        drain_events: Optional[Callable[[], str]] = None,
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
        self._consecutive_text_turns = 0  # for capping continue_on_text retries
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
        # drain_events: called after tool execution to collect background updates.
        # Returns a string to append to the tool output, or "" if nothing pending.
        self.drain_events = drain_events

        # Conversation state — this IS the context sent to the API each turn.
        # Contains user messages, reasoning items, assistant messages,
        # function_call items, and function_call_output items.
        self.messages: List[Dict[str, Any]] = []
        self.total_cost: float = 0.0
        self.total_turns: int = 0
        self._warned_soft_limit: bool = False
        self._deferred_tasks: List[tuple] = []  # (task, tc) from parallel tools
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
            # Never leave an orphaned reasoning item at the new start.
            # A reasoning item must be followed by its associated output
            # (message or function_call). If we'd start on a reasoning
            # item, skip it too.
            while (
                dropped < len(self.messages)
                and isinstance(self.messages[dropped], dict)
                and self.messages[dropped].get("type") == "reasoning"
            ):
                total -= msg_tokens[dropped]
                dropped += 1

            self.messages = self.messages[dropped:]

            # Remove orphaned function_call_output items whose matching
            # function_call was dropped. The API requires every
            # function_call_output to be preceded by its function_call
            # (matched by call_id).
            surviving_call_ids = set()
            for msg in self.messages:
                if isinstance(msg, dict) and msg.get("type") == "function_call":
                    cid = msg.get("call_id")
                    if cid:
                        surviving_call_ids.add(cid)

            cleaned = []
            for msg in self.messages:
                if (
                    isinstance(msg, dict)
                    and msg.get("type") == "function_call_output"
                    and msg.get("call_id") not in surviving_call_ids
                ):
                    logger.warning(
                        f"[{self.label}] removing orphaned function_call_output "
                        f"(call_id={msg.get('call_id')}) after context trim"
                    )
                    continue
                cleaned.append(msg)
            self.messages = cleaned

            logger.warning(
                f"[{self.label}] trimmed {dropped} oldest messages "
                f"to fit context window ({total} tokens remaining, "
                f"budget {budget})"
            )

    def _scrub_orphaned_reasoning(self) -> None:
        """Remove reasoning items whose required following item is missing.

        The Responses API requires every reasoning item to be immediately
        followed by a message or function_call from the same response.
        If context trimming or an error leaves a reasoning item at the end
        of the history (or followed by a non-output item like a user
        message), remove it.
        """
        cleaned = []
        for i, msg in enumerate(self.messages):
            if (
                isinstance(msg, dict)
                and msg.get("type") == "reasoning"
            ):
                # Check if next item is a valid following item
                next_msg = self.messages[i + 1] if i + 1 < len(self.messages) else None
                if next_msg is None or (
                    isinstance(next_msg, dict)
                    and next_msg.get("type") not in ("message", "function_call", "web_search_call", "reasoning")
                ):
                    logger.warning(
                        f"[{self.label}] removing orphaned reasoning item {msg.get('id', '?')}"
                    )
                    continue
            cleaned.append(msg)
        self.messages = cleaned

    def _scrub_orphaned_tool_outputs(self) -> None:
        """Remove function_call_output items whose function_call is missing.

        The Responses API requires every function_call_output to be preceded
        by a function_call with the same call_id. Context trimming or errors
        can leave orphaned outputs.
        """
        call_ids = {
            msg.get("call_id")
            for msg in self.messages
            if isinstance(msg, dict) and msg.get("type") == "function_call"
        }
        cleaned = []
        for msg in self.messages:
            if (
                isinstance(msg, dict)
                and msg.get("type") == "function_call_output"
                and msg.get("call_id") not in call_ids
            ):
                logger.warning(
                    f"[{self.label}] removing orphaned function_call_output "
                    f"(call_id={msg.get('call_id')})"
                )
                continue
            cleaned.append(msg)
        self.messages = cleaned

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
        # v4: Langfuse uses OTel contextvars — start_as_current_observation
        # automatically nests under the active span. No explicit parent needed.
        lf = _get_langfuse()
        if lf:
            return await self._run_loop_traced(lf, exit_condition)
        return await self._run_loop_inner(exit_condition)

    async def _run_loop_traced(
        self,
        lf,
        exit_condition: Optional[Callable[[], bool]] = None,
    ) -> AgentResult:
        """Wrapper that creates a Langfuse span around the agent loop.

        Uses v4 OTel context — start_as_current_observation automatically
        nests under the active span via contextvars.
        """
        with lf.start_as_current_observation(
            as_type="span",
            name=self.label,
            metadata={"model": str(self.model), "max_turns": str(self.max_turns)},
        ) as span:
            self._current_langfuse_span = span
            result = await self._run_loop_inner(exit_condition)
            try:
                span.update(
                    output=result.text[:2000] if result.text else None,
                    metadata={
                        "total_cost_usd": str(round(result.cost_usd, 6)),
                        "total_turns": str(result.turns_taken),
                        "stopped": str(result.stopped),
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
                        usage = response.usage
                        output_tokens = getattr(usage, "output_tokens", 0) or 0
                        # Extract reasoning tokens from output_tokens_details
                        otd = getattr(usage, "output_tokens_details", None)
                        reasoning_tokens = getattr(otd, "reasoning_tokens", 0) if otd else 0
                        usage_details = {
                            "input": getattr(usage, "input_tokens", 0) or 0,
                            "output": output_tokens,
                            "reasoning": reasoning_tokens,
                        }
                    gen_obs.update(
                        output=_serialize_response_output(response.output),
                        usage_details=usage_details,
                        metadata={"cost_usd": str(round(cost.total_cost_usd, 6))},
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

            # Deliver results from deferred parallel tool tasks
            if self._deferred_tasks:
                completed_msgs = []
                still_pending = []
                for task, tc in self._deferred_tasks:
                    if task.done():
                        try:
                            text, cost = task.result()
                            self.total_cost += cost
                            result.cost_usd = self.total_cost
                            if self.on_cost and cost > 0:
                                asyncio.ensure_future(self.on_cost(cost, self.label))
                            completed_msgs.append(f"[Completed] {tc.name}:\n{text[:TOOL_OUTPUT_LIMIT]}")
                        except Exception as e:
                            completed_msgs.append(f"[Completed] {tc.name}: Error: {e}")
                    else:
                        still_pending.append((task, tc))
                self._deferred_tasks = still_pending
                if completed_msgs:
                    self.messages.append({
                        "role": "user",
                        "content": "\n\n".join(completed_msgs),
                    })

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

            # Merge function tools with extra tools (MCP connectors, built-in web_search, etc.)
            all_tools = (self.tools.get_definitions() or []) + self.extra_tools
            if turn == 0 and logger.isEnabledFor(logging.DEBUG):
                tool_summary = [t.get("type", "?") + (":" + t.get("name", "") if t.get("name") else "") for t in all_tools]
                logger.debug(f"[{self.label}] tools: {tool_summary}")

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
                    # Cancelled by stop_checker — remove the dangling user message
                    # so the conversation doesn't have an unanswered turn on resume.
                    if self.messages and self.messages[-1].get("role") == "user":
                        self.messages.pop()
                    logger.info(f"[{self.label}] API call cancelled by stop_checker at turn {turn}")
                    result.stopped = True
                    break
                response, cost = api_result
            except Exception as e:
                logger.error(f"API call failed: {e}", exc_info=True)
                err_str = str(e)
                # If the error is about orphaned reasoning items, scrub
                # them from the history so the next turn can succeed.
                if "reasoning" in err_str and "required following item" in err_str:
                    self._scrub_orphaned_reasoning()
                # If a function_call_output lost its matching function_call
                # (e.g. after context trim), remove the orphan.
                if "No tool call found" in err_str or "function call output" in err_str.lower():
                    self._scrub_orphaned_tool_outputs()
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
                if item.type == "web_search_call":
                    # Server-side tool — already executed by the API, but
                    # must be kept in history to preserve reasoning item
                    # pairing (reasoning → web_search_call linkage).
                    action = getattr(item, "action", None)
                    action_type = getattr(action, "type", "?") if action else "?"
                    if action_type == "search":
                        query = getattr(action, "query", None) or "?"
                        logger.info(f"[{self.label}] web_search: {query[:120]}")
                    elif action_type == "open_page":
                        url = getattr(action, "url", None) or "?"
                        logger.info(f"[{self.label}] web_open_page: {url[:120]}")
                    elif action_type == "find_in_page":
                        query = getattr(action, "query", None) or "?"
                        logger.info(f"[{self.label}] web_find_in_page: {query[:120]}")
                    else:
                        logger.info(f"[{self.label}] web_search_call: type={action_type}")
                    dumped = item.model_dump(exclude_none=True)
                elif item.type == "reasoning":
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
                self._consecutive_text_turns += 1

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
                    self._consecutive_text_turns = 0
                    continue
                elif self.continue_on_text:
                    if self._consecutive_text_turns >= 2:
                        logger.warning(
                            f"[{self.label}] {self._consecutive_text_turns} consecutive "
                            f"text responses — giving up (likely repeated refusal)"
                        )
                        break
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
            self._consecutive_text_turns = 0
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

                # Append any pending background events
                if self.drain_events:
                    events = self.drain_events()
                    if events:
                        result_text += events

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

        # Collect any deferred tasks before exiting — don't orphan background work.
        # Give them a short window to finish, then record whatever we have.
        if self._deferred_tasks:
            pending_tasks = [(t, tc) for t, tc in self._deferred_tasks if not t.done()]
            done_tasks = [(t, tc) for t, tc in self._deferred_tasks if t.done()]

            # Wait up to 5s for pending tasks to finish
            if pending_tasks:
                tasks_only = [t for t, _ in pending_tasks]
                finished, still_pending = await asyncio.wait(
                    tasks_only, timeout=5.0,
                )
                # Cancel anything still running
                for t in still_pending:
                    t.cancel()
                if still_pending:
                    await asyncio.gather(*still_pending, return_exceptions=True)
                # Add finished to done list
                task_to_tc = {t: tc for t, tc in pending_tasks}
                for t in finished:
                    done_tasks.append((t, task_to_tc[t]))

            # Collect costs but do NOT append duplicate function_call_output
            # items — the "Still running" output was already appended during
            # parallel execution, and the actual results were delivered as
            # user messages during the loop. Adding another function_call_output
            # with the same call_id would confuse the API on replay.
            for task, tc in done_tasks:
                try:
                    _result_text, cost = task.result()
                    self.total_cost += cost
                except Exception:
                    pass

            self._deferred_tasks.clear()

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
                    metadata={"cost_usd": str(round(cost, 6)), "output_len": str(len(result_text))},
                )
            except Exception:
                pass
            return result_text, cost

    async def _execute_tools_parallel(
        self, tool_calls: list, result: AgentResult,
    ) -> None:
        """Execute multiple tool calls concurrently.

        Uses FIRST_COMPLETED: when any tool finishes, gives remaining tools
        a brief grace period (2s) for other fast tools to complete, then
        returns. Still-running tools get "Pending" output and continue in
        the background — their results are delivered via drain_events.
        """
        # Launch all
        tasks: Dict[asyncio.Task, Any] = {}
        for tc in tool_calls:
            task = asyncio.create_task(self._execute_tool(tc))
            tasks[task] = tc

        outputs: Dict[str, str] = {}
        costs: Dict[str, float] = {}
        pending = set(tasks.keys())

        # Wait for first completion
        done, pending = await asyncio.wait(
            pending, return_when=asyncio.FIRST_COMPLETED,
        )
        self._collect_done_tasks(done, tasks, outputs, costs, result)

        # Grace period — let other fast tools finish (apollo, create_harvester)
        if pending:
            done2, pending = await asyncio.wait(pending, timeout=2.0)
            self._collect_done_tasks(done2, tasks, outputs, costs, result)

        # Still-running tools get "Pending" and continue in background.
        # Include current status via drain_events so orchestrator has context.
        pending_status = ""
        if pending and self.drain_events:
            pending_status = self.drain_events()

        for task in pending:
            tc = tasks[task]
            outputs[tc.call_id] = (
                f"Still running — results will appear on your next action."
                f"{pending_status}"
            )
            pending_status = ""  # only attach status to first pending
            self._deferred_tasks.append((task, tc))

        # Append drain_events to last output
        last_id = tool_calls[-1].call_id
        if self.drain_events:
            events = self.drain_events()
            if events and last_id in outputs:
                outputs[last_id] += events

        # Append all outputs in original order
        for tc in tool_calls:
            self.messages.append({
                "type": "function_call_output",
                "call_id": tc.call_id,
                "output": (outputs.get(tc.call_id, "Error: no result"))[:TOOL_OUTPUT_LIMIT],
            })

    def _collect_done_tasks(
        self,
        done: set,
        tasks: Dict,
        outputs: Dict[str, str],
        costs: Dict[str, float],
        result: Any,
    ) -> None:
        """Process completed tasks from asyncio.wait."""
        for task in done:
            tc = tasks[task]
            try:
                text, tool_cost = task.result()
                self.total_cost += tool_cost
                result.cost_usd = self.total_cost
                if self.on_cost and tool_cost > 0:
                    asyncio.ensure_future(self.on_cost(tool_cost, self.label))
                outputs[tc.call_id] = text
                costs[tc.call_id] = tool_cost
            except Exception as e:
                logger.error(f"Parallel tool error for {tc.name}: {e}")
                outputs[tc.call_id] = f"Error executing tool: {e}"
