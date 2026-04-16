"""
Anthropic (Claude) native agent conversation.

Claude-native equivalent of AgentConversation in base.py. Same public surface
(send, step, inject_message, messages, total_cost, total_turns) so it can be
swapped in via the factory in agents/factory.py.

self.messages is a list of Claude-format messages:
  {"role": "user" | "assistant", "content": str | list[content_block]}

Content blocks are stored as dicts (via model_dump) so the full history is
JSON-serializable for Langfuse and so thinking-block signatures round-trip
verbatim across turns — Claude rejects tampered signatures.
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple

from dsl_worker.agents.tools import ToolRegistry
from dsl_worker.billing.tracked_anthropic_client import TrackedAnthropicClient
from dsl_worker.utils import count_tokens

logger = logging.getLogger(__name__)

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

TOOL_OUTPUT_LIMIT = 15_000


@dataclass
class AgentResult:
    text: str = ""
    cost_usd: float = 0.0
    turns_taken: int = 0
    stopped: bool = False


# -----------------------------------------------------------------------------
# Tool translation: OpenAI-format tool definitions → Claude format.
# The worker's ToolRegistry emits OpenAI-shape dicts; we translate here so the
# registry doesn't need to know which provider it's feeding.
# -----------------------------------------------------------------------------

def translate_tools(
    openai_tools: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Convert OpenAI-shape tool definitions to (claude_tools, claude_mcp_servers).

    OpenAI shapes handled:
      {"type": "function", "name", "description", "parameters"}
        → {"name", "description", "input_schema": parameters}
      {"type": "namespace", "tools": [...]}
        → flattened to individual functions (Claude has no defer_loading)
      {"type": "web_search"}
        → {"type": "web_search_20260209", "name": "web_search"}
      {"type": "mcp", ...}
        → extracted to claude_mcp_servers (best-effort; header/connector_id
          OpenAI-specific fields are dropped with a warning)
    """
    claude_tools: List[Dict[str, Any]] = []
    mcp_servers: List[Dict[str, Any]] = []
    seen_names: set = set()

    def _add_tool(tool: Dict[str, Any]) -> None:
        """Add a tool, deduping by name. Claude requires unique tool names."""
        name = tool.get("name")
        if not name:
            return
        if name in seen_names:
            logger.warning(
                f"Duplicate tool name '{name}' — skipping (first registration wins). "
                f"This usually means two namespaces expose the same tool."
            )
            return
        seen_names.add(name)
        claude_tools.append(tool)

    for t in openai_tools or []:
        t_type = t.get("type")

        if t_type == "function":
            _add_tool(_function_to_claude(t))

        elif t_type == "namespace":
            # Flatten — Claude doesn't have namespace/defer_loading.
            for sub in t.get("tools", []):
                _add_tool(_function_to_claude(sub))

        elif t_type == "web_search":
            _add_tool({
                "type": "web_search_20260209",
                "name": "web_search",
            })

        elif t_type == "mcp":
            server = _mcp_to_claude(t)
            if server is not None:
                mcp_servers.append(server)

        else:
            logger.warning(f"Skipping unsupported tool type for Claude: {t_type}")

    return claude_tools, mcp_servers


def _function_to_claude(t: Dict[str, Any]) -> Dict[str, Any]:
    params = t.get("parameters") or {"type": "object", "properties": {}}
    out = {
        "name": t["name"],
        "description": t.get("description", ""),
        "input_schema": params,
    }
    return out


def _mcp_to_claude(t: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Best-effort OpenAI MCP → Claude MCP server spec.

    OpenAI-only fields (connector_id, headers) are dropped with warnings —
    Claude's MCP requires a URL.
    """
    label = t.get("server_label") or t.get("name") or "mcp"
    url = t.get("server_url")
    if not url:
        if t.get("connector_id"):
            logger.warning(
                f"MCP connector '{label}' uses OpenAI connector_id — "
                f"not supported on Claude; skipping"
            )
        else:
            logger.warning(f"MCP connector '{label}' has no server_url; skipping")
        return None

    server: Dict[str, Any] = {"type": "url", "name": label, "url": url}
    auth = t.get("authorization")
    if auth:
        server["authorization_token"] = auth
    if t.get("allowed_tools") is not None:
        server["tool_configuration"] = {"allowed_tools": t["allowed_tools"]}
    if t.get("headers"):
        logger.warning(
            f"MCP connector '{label}' has custom headers — "
            f"not supported on Claude; dropped"
        )
    return server


# -----------------------------------------------------------------------------
# Effort translation. OpenAI reasoning.effort values: minimal|low|medium|high.
# Claude effort values: low|medium|high|max (Opus also supports "xhigh").
# -----------------------------------------------------------------------------

def _translate_effort(openai_effort: Optional[str]) -> Optional[str]:
    if not openai_effort:
        return None
    mapping = {
        "minimal": "low",
        "low": "low",
        "medium": "medium",
        "high": "high",
        "max": "max",
        "xhigh": "xhigh",
    }
    return mapping.get(openai_effort, "high")


# -----------------------------------------------------------------------------
# Langfuse helpers for logging Claude-native traces.
# -----------------------------------------------------------------------------

def _serialize_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Trim oversize fields in messages for Langfuse input logging."""
    result = []
    for m in messages[-20:]:  # only log last 20 for trace size
        content = m.get("content")
        if isinstance(content, str):
            if len(content) > 2000:
                content = content[:2000] + "…"
            result.append({"role": m.get("role"), "content": content})
        elif isinstance(content, list):
            summarized = []
            for block in content:
                if not isinstance(block, dict):
                    continue
                btype = block.get("type", "?")
                if btype == "text":
                    text = block.get("text", "")
                    summarized.append({
                        "type": "text",
                        "text": text[:500] + "…" if len(text) > 500 else text,
                    })
                elif btype == "thinking":
                    summarized.append({"type": "thinking", "len": len(block.get("thinking", ""))})
                elif btype == "tool_use":
                    inp = block.get("input") or {}
                    inp_str = json.dumps(inp, default=str)
                    if len(inp_str) > 500:
                        inp_str = inp_str[:500] + "…"
                    summarized.append({
                        "type": "tool_use",
                        "name": block.get("name"),
                        "input": inp_str,
                    })
                elif btype == "tool_result":
                    c = block.get("content", "")
                    if isinstance(c, list):
                        c = " ".join(x.get("text", "") for x in c if isinstance(x, dict))
                    summarized.append({
                        "type": "tool_result",
                        "preview": (c[:500] + "…") if len(c) > 500 else c,
                    })
                else:
                    summarized.append({"type": btype})
            result.append({"role": m.get("role"), "content": summarized})
    return result


def _serialize_response_content(content: List[Any]) -> List[Dict[str, Any]]:
    """Summarize Claude response content blocks for Langfuse output logging."""
    result = []
    for block in content:
        btype = getattr(block, "type", None) or (block.get("type") if isinstance(block, dict) else None)
        if btype == "text":
            text = getattr(block, "text", None) or (block.get("text", "") if isinstance(block, dict) else "")
            result.append({
                "type": "text",
                "text": text[:2000] + "…" if len(text) > 2000 else text,
            })
        elif btype == "thinking":
            thinking = getattr(block, "thinking", None) or (block.get("thinking", "") if isinstance(block, dict) else "")
            result.append({"type": "thinking", "len": len(thinking)})
        elif btype == "tool_use":
            name = getattr(block, "name", None) or (block.get("name") if isinstance(block, dict) else None)
            inp = getattr(block, "input", None) or (block.get("input") if isinstance(block, dict) else None) or {}
            inp_str = json.dumps(inp, default=str)
            if len(inp_str) > 500:
                inp_str = inp_str[:500] + "…"
            result.append({"type": "tool_use", "name": name, "input": inp_str})
        elif btype in ("server_tool_use", "web_search_tool_result"):
            result.append({"type": btype})
        else:
            result.append({"type": btype or "?"})
    return result


# -----------------------------------------------------------------------------
# Main class.
# -----------------------------------------------------------------------------

class AnthropicAgentConversation:
    """Claude-native multi-turn conversation with tool use and cost tracking.

    Public surface matches AgentConversation (see agents/base.py):
      - send(message, exit_condition=None) → AgentResult
      - step(exit_condition=None) → AgentResult
      - inject_message(role, content)
      - self.messages  (Claude format)
      - self.total_cost, self.total_turns
    """

    def __init__(
        self,
        openai_client: TrackedAnthropicClient,
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
        prompt_cache_key: Optional[str] = None,
    ) -> None:
        # Parameter is called `openai_client` for drop-in compatibility
        # with AgentConversation constructor — holds a TrackedAnthropicClient here.
        self.client = openai_client
        # Also keep the original attribute name so any external code poking
        # at .openai_client keeps working transparently.
        self.openai_client = openai_client
        self.model = model
        self.system_prompt = system_prompt
        self.tools = tools
        self.stop_checker = stop_checker
        self.stop_event = stop_event
        self.max_turns = max_turns
        self.soft_turn_limit = soft_turn_limit
        self.max_output_tokens = max_output_tokens
        self.reasoning = reasoning or {"effort": "medium"}
        self.label = label
        self.continue_on_text = continue_on_text
        self._consecutive_text_turns = 0
        self.context_window = context_window
        self.on_tool_call = on_tool_call
        self.on_cost = on_cost
        self.extra_tools = extra_tools or []
        self.langfuse_parent = langfuse_parent
        self.on_idle = on_idle
        self.drain_events = drain_events
        self.after_turn: Optional[Callable[[], Optional[str]]] = None
        # prompt_cache_key is OpenAI-specific routing hint — not applicable
        # on Anthropic; accepted for interface parity and ignored.
        self.prompt_cache_key = prompt_cache_key

        # Conversation state. Claude format:
        #   user: {"role": "user", "content": str | [content blocks]}
        #   assistant: {"role": "assistant", "content": [content blocks]}
        self.messages: List[Dict[str, Any]] = []
        self.total_cost: float = 0.0
        self.total_turns: int = 0
        self._warned_soft_limit: bool = False
        # Deferred tool tasks, shape: (task, tool_use_id, tool_name)
        self._deferred_tasks: List[Tuple[asyncio.Task, str, str]] = []
        self._current_langfuse_span: Any = None

    # ------------------------------------------------------------------
    # Public API — matches AgentConversation
    # ------------------------------------------------------------------

    async def send(
        self,
        message: str,
        exit_condition: Optional[Callable[[], bool]] = None,
    ) -> AgentResult:
        self.messages.append({"role": "user", "content": message})
        return await self._run_loop(exit_condition)

    async def step(
        self,
        exit_condition: Optional[Callable[[], bool]] = None,
    ) -> AgentResult:
        return await self._run_loop(exit_condition)

    def inject_message(self, role: str, content: str) -> None:
        self.messages.append({"role": role, "content": content})

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _should_stop(self) -> bool:
        return self.stop_checker is not None and self.stop_checker()

    async def _api_call_with_stop_check(self, **kwargs):
        """Race the API call against stop_event for instant wakeup on pause."""
        api_task = asyncio.create_task(self.client.messages_create(**kwargs))

        if not self.stop_checker:
            return await api_task

        def _cancel_and_return_none():
            api_task.cancel()
            return None

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
        """Drop oldest messages if total context exceeds the window limit.

        Preserves the tool_use ↔ tool_result pairing: if dropping would leave
        an orphaned tool_result as the new first message, drop it too.
        """
        system_tokens = count_tokens(self.system_prompt)
        budget = self.context_window - system_tokens - self.max_output_tokens
        if budget <= 0 or not self.messages:
            return

        msg_tokens = [count_tokens(_rough_text(m)) for m in self.messages]
        total = sum(msg_tokens)
        if total <= budget:
            return

        dropped = 0
        while total > budget and dropped < len(self.messages):
            total -= msg_tokens[dropped]
            dropped += 1

        if dropped > 0:
            self.messages = self.messages[dropped:]
            # Never start on an orphaned tool_result (has no preceding tool_use).
            while self.messages and _starts_with_tool_result(self.messages[0]):
                self.messages.pop(0)
            logger.warning(
                f"[{self.label}] trimmed {dropped} oldest messages "
                f"to fit context window (~{total} tokens, budget {budget})"
            )

    async def _run_loop(
        self,
        exit_condition: Optional[Callable[[], bool]] = None,
    ) -> AgentResult:
        lf = _get_langfuse()
        if lf:
            return await self._run_loop_traced(lf, exit_condition)
        return await self._run_loop_inner(exit_condition)

    async def _run_loop_traced(
        self,
        lf,
        exit_condition: Optional[Callable[[], bool]] = None,
    ) -> AgentResult:
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

    async def _call_api_with_trace(self, **kwargs):
        """One API call, logged to Langfuse if a span is active."""
        if self._current_langfuse_span is None:
            return await self._api_call_with_stop_check(**kwargs)

        lf = _get_langfuse()
        if lf is None:
            return await self._api_call_with_stop_check(**kwargs)

        try:
            obs_ctx = lf.start_as_current_observation(
                as_type="generation",
                name=f"{self.label}:llm",
                model=self.model,
                input=_serialize_messages(kwargs.get("messages", [])),
            )
        except Exception:
            return await self._api_call_with_stop_check(**kwargs)

        with obs_ctx as gen_obs:
            api_result = await self._api_call_with_stop_check(**kwargs)
            if api_result is not None:
                try:
                    response, cost = api_result
                    usage = getattr(response, "usage", None)
                    usage_details = None
                    cost_details = None
                    if usage is not None:
                        input_tokens = getattr(usage, "input_tokens", 0) or 0
                        output_tokens = getattr(usage, "output_tokens", 0) or 0
                        cache_read = getattr(usage, "cache_read_input_tokens", 0) or 0
                        usage_details = {
                            "input": input_tokens,
                            "output": output_tokens,
                            "total": input_tokens + output_tokens + cache_read,
                            "input_cached_tokens": cache_read,
                        }
                        cost_details = {
                            "input": cost.input_cost_usd,
                            "output": cost.output_cost_usd,
                            "total": cost.total_cost_usd,
                            "input_cached_tokens": cost.cached_input_cost_usd,
                        }
                    gen_obs.update(
                        output=_serialize_response_content(response.content),
                        usage_details=usage_details,
                        cost_details=cost_details,
                        metadata={"cost_usd": str(round(cost.total_cost_usd, 6))},
                    )
                except Exception:
                    pass
            return api_result

    async def _run_loop_inner(
        self,
        exit_condition: Optional[Callable[[], bool]] = None,
    ) -> AgentResult:
        result = AgentResult()

        # Pre-translate tools once. extra_tools may contain MCP defs which
        # split into a separate mcp_servers param.
        fn_tools, mcp_servers = translate_tools(
            (self.tools.get_definitions() or []) + self.extra_tools
        )

        # Cache the tools + system prefix. Render order is tools → system → messages,
        # so cache_control on the last system text block covers both. This prefix is
        # stable across all turns of this conversation AND across parallel agents
        # sharing the same system prompt (e.g. 100 row generators) → big savings.
        system_blocks = [{
            "type": "text",
            "text": self.system_prompt,
            "cache_control": {"type": "ephemeral"},
        }]

        # Thinking — use adaptive on Claude 4.6/4.7 (budget_tokens is deprecated
        # on 4.6 and removed on 4.7). Enable summarized display so any
        # logging of thinking text shows real content rather than empty strings.
        thinking = {"type": "adaptive", "display": "summarized"}

        effort = _translate_effort(self.reasoning.get("effort") if isinstance(self.reasoning, dict) else None)
        output_config = {"effort": effort} if effort else None

        for turn in range(self.max_turns):
            if self._should_stop():
                logger.info(f"[{self.label}] stopped by stop_checker at turn {turn}")
                result.stopped = True
                break

            if exit_condition and exit_condition():
                logger.info(f"[{self.label}] exit_condition met at turn {turn}")
                break

            # Deliver results from deferred parallel tool tasks as a user message.
            # These are tools whose real results arrived after their synthesized
            # "still running" tool_result was already sent back.
            if self._deferred_tasks:
                completed_msgs: List[str] = []
                still_pending: List[Tuple[asyncio.Task, str, str]] = []
                for task, tu_id, tu_name in self._deferred_tasks:
                    if task.done():
                        try:
                            text, cost = task.result()
                            self.total_cost += cost
                            result.cost_usd = self.total_cost
                            if self.on_cost and cost > 0:
                                asyncio.ensure_future(self.on_cost(cost, self.label))
                            completed_msgs.append(
                                f"[Completed] {tu_name}:\n{text[:TOOL_OUTPUT_LIMIT]}"
                            )
                        except Exception as e:
                            completed_msgs.append(f"[Completed] {tu_name}: Error: {e}")
                    else:
                        still_pending.append((task, tu_id, tu_name))
                self._deferred_tasks = still_pending
                if completed_msgs:
                    self.messages.append({
                        "role": "user",
                        "content": "\n\n".join(completed_msgs),
                    })

            self._trim_context()

            if turn == self.soft_turn_limit and not self._warned_soft_limit:
                self._warned_soft_limit = True
                logger.info(
                    f"[{self.label}] soft turn limit ({self.soft_turn_limit}) — wrap-up nudge"
                )
                self.messages.append({
                    "role": "user",
                    "content": (
                        "WRAP UP NOW. You are at your turn budget. Call respond() "
                        "immediately with your findings so far. Do not make more "
                        "tool calls — submit what you have."
                    ),
                })

            logger.info(
                f"[{self.label}] turn {turn} — {len(self.messages)} messages, "
                f"${self.total_cost:.4f} spent"
            )
            if turn == 0:
                tool_summary = [t.get("name") for t in fn_tools]
                if mcp_servers:
                    tool_summary.extend(f"mcp:{s['name']}" for s in mcp_servers)
                logger.info(
                    f"[{self.label}] {len(tool_summary)} tools: {tool_summary}"
                )

            # Build call. Top-level cache_control auto-places a second
            # breakpoint on the last cacheable message block, so the growing
            # conversation prefix gets cached turn-over-turn.
            call_kwargs: Dict[str, Any] = {
                "model": self.model,
                "system": system_blocks,
                "messages": self.messages,
                "max_tokens": self.max_output_tokens,
                "thinking": thinking,
                "cache_control": {"type": "ephemeral"},
            }
            if fn_tools:
                call_kwargs["tools"] = fn_tools
            if output_config:
                call_kwargs["output_config"] = output_config
            if mcp_servers:
                call_kwargs["mcp_servers"] = mcp_servers

            try:
                api_result = await self._call_api_with_trace(**call_kwargs)
                if api_result is None:
                    # Cancelled by stop_checker — drop dangling user message
                    if self.messages and self.messages[-1].get("role") == "user":
                        self.messages.pop()
                    logger.info(f"[{self.label}] API call cancelled at turn {turn}")
                    result.stopped = True
                    break
                response, cost = api_result
            except Exception as e:
                logger.error(f"[{self.label}] API call failed: {e}", exc_info=True)
                # Circuit breaker: 4xx client errors (malformed request, auth, etc.)
                # never recover — fail fast instead of looping. 429 rate limits
                # are retryable, so let those fall through to the retry path.
                status_code = getattr(e, "status_code", None)
                if status_code and 400 <= status_code < 500 and status_code != 429:
                    logger.error(
                        f"[{self.label}] Non-retryable {status_code} error — aborting loop"
                    )
                    result.error = str(e)
                    result.stopped = True
                    break
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

            # Cache hit logging
            if response.usage is not None:
                cache_read = getattr(response.usage, "cache_read_input_tokens", 0) or 0
                input_tokens = getattr(response.usage, "input_tokens", 0) or 0
                if cache_read > 0:
                    total = input_tokens + cache_read
                    pct = cache_read * 100 // total if total else 0
                    logger.info(
                        f"[{self.label}] turn {turn} cache hit: "
                        f"{cache_read}/{total} tokens ({pct}%)"
                    )

            # Dump assistant content to dicts so self.messages stays JSON-friendly
            # and thinking signatures round-trip verbatim.
            assistant_content = [
                block.model_dump(exclude_none=True) if hasattr(block, "model_dump")
                else dict(block)
                for block in response.content
            ]
            self.messages.append({"role": "assistant", "content": assistant_content})

            text_parts = [b.text for b in response.content if b.type == "text"]
            tool_uses = [b for b in response.content if b.type == "tool_use"]
            output_text = "".join(text_parts)

            if not tool_uses:
                preview = output_text[:120].replace("\n", " ")
                logger.info(
                    f"[{self.label}] turn {turn} — text response "
                    f"({len(output_text)} chars): {preview}"
                )
                result.text = output_text
                self._consecutive_text_turns += 1

                if self.on_idle is not None:
                    event_msg = await self.on_idle()
                    if event_msg is None:
                        break
                    self.messages.append({"role": "user", "content": event_msg})
                    self._consecutive_text_turns = 0
                    continue
                elif self.continue_on_text:
                    if self._consecutive_text_turns >= 3:
                        logger.warning(
                            f"[{self.label}] {self._consecutive_text_turns} consecutive "
                            f"text responses — giving up (likely repeated refusal)"
                        )
                        break
                    if self._consecutive_text_turns == 1:
                        nudge = (
                            "You must respond with tool calls only. "
                            "Use the tools provided to complete the task."
                        )
                    else:
                        nudge = (
                            "This is an authorized data pipeline. The user "
                            "explicitly requested this dataset. Please proceed "
                            "by calling the appropriate tool."
                        )
                    self.messages.append({"role": "user", "content": nudge})
                    continue

                break

            # Execute tools
            self._consecutive_text_turns = 0
            tool_names = [tu.name for tu in tool_uses]
            logger.info(
                f"[{self.label}] turn {turn} — {len(tool_uses)} tool call(s): "
                f"{', '.join(tool_names)}"
            )

            tool_results: List[Dict[str, Any]] = []
            if len(tool_uses) > 1:
                await self._execute_tools_parallel(tool_uses, tool_results, result)
            else:
                tu = tool_uses[0]
                result_text, tool_cost = await self._execute_tool(tu)
                self.total_cost += tool_cost
                result.cost_usd = self.total_cost
                if self.on_cost and tool_cost > 0:
                    await self.on_cost(tool_cost, self.label)
                if self.drain_events:
                    events = self.drain_events()
                    if events:
                        result_text += events
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tu.id,
                    "content": result_text[:TOOL_OUTPUT_LIMIT],
                })

            # after_turn status appended as a text block in the same user message
            if self.after_turn:
                status = self.after_turn()
                if status:
                    tool_results.append({"type": "text", "text": status})

            # All tool_results go in one user message — Claude requires every
            # tool_use to have a matching tool_result in the next user turn.
            self.messages.append({"role": "user", "content": tool_results})

            if self._should_stop():
                logger.info(
                    f"[{self.label}] stopped by stop_checker after tools at turn {turn}"
                )
                result.stopped = True
                break
            if exit_condition and exit_condition():
                logger.info(
                    f"[{self.label}] exit_condition met after tools at turn {turn}"
                )
                break

            result.text = output_text
        else:
            logger.warning(f"[{self.label}] hit max turns ({self.max_turns})")

        # Reap any deferred background tasks. Costs only — results were either
        # already delivered via user messages mid-loop, or are stale now.
        if self._deferred_tasks:
            pending = [(t, i, n) for t, i, n in self._deferred_tasks if not t.done()]
            done = [(t, i, n) for t, i, n in self._deferred_tasks if t.done()]
            if pending:
                tasks_only = [t for t, _, _ in pending]
                finished, still = await asyncio.wait(tasks_only, timeout=5.0)
                for t in still:
                    t.cancel()
                if still:
                    await asyncio.gather(*still, return_exceptions=True)
                triple_by_task = {t: (i, n) for t, i, n in pending}
                for t in finished:
                    i, n = triple_by_task[t]
                    done.append((t, i, n))
            for task, _, _ in done:
                try:
                    _text, cost = task.result()
                    self.total_cost += cost
                except Exception:
                    pass
            self._deferred_tasks.clear()

        logger.info(
            f"[{self.label}] loop done — {self.total_turns} turns, "
            f"${self.total_cost:.4f}, {len(self.messages)} messages"
        )
        return result

    async def _execute_tool(self, tu) -> Tuple[str, float]:
        """Execute a single tool_use block. Returns (result_text, cost)."""
        if self.on_tool_call:
            self.on_tool_call(self.label, tu.name)

        # Claude parses tool input to dict already
        args = getattr(tu, "input", None) or {}
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                return (
                    f"Error: invalid JSON in tool arguments for {tu.name}.",
                    0.0,
                )

        if self._current_langfuse_span is None:
            return await self.tools.execute(tu.name, args)

        lf = _get_langfuse()
        if lf is None:
            return await self.tools.execute(tu.name, args)

        try:
            obs_ctx = lf.start_as_current_observation(
                as_type="span",
                name=f"tool:{tu.name}",
                input=args,
            )
        except Exception:
            return await self.tools.execute(tu.name, args)

        with obs_ctx as tool_obs:
            result_text, cost = await self.tools.execute(tu.name, args)
            try:
                preview = result_text if len(result_text) <= 1000 else result_text[:1000] + "…"
                tool_obs.update(
                    output=preview,
                    metadata={
                        "cost_usd": str(round(cost, 6)),
                        "output_len": str(len(result_text)),
                    },
                )
            except Exception:
                pass
            return result_text, cost

    async def _execute_tools_parallel(
        self,
        tool_uses: List[Any],
        tool_results: List[Dict[str, Any]],
        result: AgentResult,
    ) -> None:
        """Execute multiple tool_use blocks concurrently.

        Every tool_use must have a matching tool_result in the next user
        message — we synthesize "Still running" placeholders for tools that
        don't finish within the grace window, and deliver real results later
        via drain-on-next-turn.
        """
        tasks: Dict[asyncio.Task, Any] = {}
        for tu in tool_uses:
            task = asyncio.create_task(self._execute_tool(tu))
            tasks[task] = tu

        outputs: Dict[str, str] = {}
        pending = set(tasks.keys())

        done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
        self._collect_done_tasks(done, tasks, outputs, result)

        if pending:
            done2, pending = await asyncio.wait(pending, timeout=2.0)
            self._collect_done_tasks(done2, tasks, outputs, result)

        pending_status = ""
        if pending and self.drain_events:
            pending_status = self.drain_events()

        for task in pending:
            tu = tasks[task]
            outputs[tu.id] = (
                "Still running — results will appear on your next action."
                f"{pending_status}"
            )
            pending_status = ""
            self._deferred_tasks.append((task, tu.id, tu.name))

        # Attach drain_events to last output
        last_id = tool_uses[-1].id
        if self.drain_events:
            events = self.drain_events()
            if events and last_id in outputs:
                outputs[last_id] += events

        for tu in tool_uses:
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": tu.id,
                "content": (outputs.get(tu.id, "Error: no result"))[:TOOL_OUTPUT_LIMIT],
            })

    def _collect_done_tasks(
        self,
        done: set,
        tasks: Dict,
        outputs: Dict[str, str],
        result: AgentResult,
    ) -> None:
        for task in done:
            tu = tasks[task]
            try:
                text, tool_cost = task.result()
                self.total_cost += tool_cost
                result.cost_usd = self.total_cost
                if self.on_cost and tool_cost > 0:
                    asyncio.ensure_future(self.on_cost(tool_cost, self.label))
                outputs[tu.id] = text
            except Exception as e:
                logger.error(f"Parallel tool error for {tu.name}: {e}")
                outputs[tu.id] = f"Error executing tool: {e}"


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _rough_text(msg: Dict[str, Any]) -> str:
    """Rough text extraction for token counting during trim."""
    content = msg.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if not isinstance(block, dict):
                continue
            btype = block.get("type")
            if btype == "text":
                parts.append(block.get("text", ""))
            elif btype == "thinking":
                parts.append(block.get("thinking", ""))
            elif btype == "tool_use":
                parts.append(json.dumps(block.get("input", {}), default=str))
            elif btype == "tool_result":
                c = block.get("content", "")
                if isinstance(c, list):
                    for sub in c:
                        if isinstance(sub, dict):
                            parts.append(sub.get("text", ""))
                else:
                    parts.append(str(c))
        return "\n".join(parts)
    return str(content)


def _starts_with_tool_result(msg: Dict[str, Any]) -> bool:
    if msg.get("role") != "user":
        return False
    content = msg.get("content")
    if isinstance(content, list) and content:
        first = content[0]
        return isinstance(first, dict) and first.get("type") == "tool_result"
    return False
