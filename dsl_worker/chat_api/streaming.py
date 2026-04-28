"""SSE streaming for chat-mode send-message.

Wire-compatible with the API's previous /v1/projects/{id}/chat/stream
endpoint. Event types: thinking, token, tool_start, tool_call,
tool_result, change, questions, project_name, done, error.

Stop semantics: between turns and between tool calls, we check
`request.is_disconnected()`. On disconnect we flush the DB and exit
gracefully — the assistant message is persisted with whatever ran so
far so the user sees partial progress on reload.
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from typing import Any, AsyncGenerator, Dict, List, Optional
from uuid import UUID

from fastapi import Request
from openai import AsyncOpenAI
from sqlalchemy.orm import Session

from dsl_api.config import settings
from dsl_api.credits import consume_credits
from dsl_api.db import SessionLocal
from dsl_api.models import Account, ChatMessage, Project

from dsl_worker.chat_api import agent, sources as _sources, tracing

log = logging.getLogger(__name__)


# ---- OpenAI client (singleton) -------------------------------------------
_client: Optional[AsyncOpenAI] = None


def get_openai_client() -> AsyncOpenAI:
    global _client
    if _client is None:
        _client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
    return _client


# ---- Cost / billing -------------------------------------------------------
_INPUT_COST = 0.0000025          # $2.50 / 1M tokens
_CACHED_INPUT_COST = 0.00000025  # $0.25 / 1M tokens
_OUTPUT_COST = 0.000015          # $15.00 / 1M tokens

MAX_TOOL_ROUNDS = 12


def _response_cost_usd(response) -> float:
    usage = getattr(response, "usage", None)
    if not usage:
        return 0.0
    input_tokens = usage.input_tokens or 0
    output_tokens = usage.output_tokens or 0
    cached_tokens = 0
    details = getattr(usage, "input_tokens_details", None)
    if details:
        cached_tokens = getattr(details, "cached_tokens", 0) or 0
    non_cached = max(0, input_tokens - cached_tokens)
    return (
        non_cached * _INPUT_COST
        + cached_tokens * _CACHED_INPUT_COST
        + output_tokens * _OUTPUT_COST
    )


def _charge_credits(db: Session, user_id, cost_usd: float, project_id=None) -> None:
    if cost_usd <= 0:
        return
    credits = cost_usd / settings.COMPUTE_COST_PER_CREDIT
    if credits < 0.01:
        return
    account = db.query(Account).filter(Account.user_id == str(user_id)).first()
    if not account:
        log.warning(f"No account found for user {user_id}, skipping credit charge")
        return
    consume_credits(db, account, credits, project_id=project_id)


# ---- Citation stripping (web_search marker cleanup) ----------------------
# Two formats observed from OpenAI's web_search built-in:
#   1. Bracketed:  【4:0†source title】
#   2. Bare token: citeturn1search14turn8view2
# OpenAI wraps the bare form in Private Use Area Unicode chars (U+E000-
# U+F8FF) which render as garbage glyphs ("chinese symbols") in most fonts
# AND prevent the bare regex from matching, since `cite` and `turn\d+...`
# are separated by ``. We strip all PUA chars first so the bare
# regex collapses to the visible `citeturnXsearchY` form, then strip both
# forms. PUA has no legitimate use in chat output, so global strip is safe.
_PUA_RE = re.compile(r'[-]')
# Bracketed: 【4:0†source title】
_BRACKETED_RE = re.compile(r'【[^】]*?†[^】]*?】')
# Bare: citeturn0search24turn3view7... — anchor on the unmistakable
# `citeturn\d` prefix so we catch unknown kind words too, but never match
# `citeturning` or other false positives. Requires at least one digit-ended
# segment to avoid swallowing trailing prose.
_BARE_CITE_RE = re.compile(r'citeturn\d+(?:[a-z]+\d+)*')


def _clean_citations(text: str) -> str:
    text = _PUA_RE.sub('', text)
    return _BARE_CITE_RE.sub('', _BRACKETED_RE.sub('', text))


class _CitationStripper:
    """Streaming-safe citation stripper.

    Buffers up to ~100 chars when a partial marker is in flight (bracketed
    form needs to wait for the closing 】, bare form needs to wait until
    we know whether trailing 'cite...' is followed by 'turn\\d+kind\\d+').
    """

    # Max chars we'll buffer waiting to see if a partial marker completes.
    # Bare cite tokens can chain many segments; cap generously.
    _MAX_BUFFER = 200

    def __init__(self):
        self._buf = ""

    def feed(self, token: str) -> str:
        # Strip OpenAI's PUA control chars on entry so the bare regex can
        # match `citeturn\d...`. Without this, ``cite``turn0search8`` ends up
        # at the "treat as ordinary text" fallthrough below and `cite` leaks
        # through followed by the rendered-glyph PUA chars.
        token = _PUA_RE.sub('', token)
        self._buf += token
        out = ""
        while True:
            # Bracketed form takes priority — it's unambiguous once both
            # 【 and 】 are present.
            br_start = self._buf.find('【')
            if br_start != -1:
                out += self._buf[:br_start]
                self._buf = self._buf[br_start:]
                br_end = self._buf.find('】')
                if br_end == -1:
                    # Partial bracket — stall unless it's clearly garbage
                    if len(self._buf) > self._MAX_BUFFER:
                        out += self._buf
                        self._buf = ""
                    return out
                candidate = self._buf[:br_end + 1]
                if '†' in candidate:
                    self._buf = self._buf[br_end + 1:]
                else:
                    out += candidate
                    self._buf = self._buf[br_end + 1:]
                continue

            # Bare form — find "cite" and check what follows
            bc_start = self._buf.find('cite')
            if bc_start == -1:
                # Could a "cite" be split across token boundaries? Hold the
                # last few chars in case they prefix a future "cite".
                if len(self._buf) >= 4:
                    out += self._buf[:-3]
                    self._buf = self._buf[-3:]
                return out

            out += self._buf[:bc_start]
            self._buf = self._buf[bc_start:]

            m = _BARE_CITE_RE.match(self._buf)
            if m:
                # Marker matched. The marker only definitively ends when we
                # see a non-alphanumeric tail char — until then more letters
                # or digits could extend the trailing segment OR start a new
                # one. Stall in that case.
                tail = self._buf[m.end():]
                could_extend = tail == "" or tail[0].isalnum()
                if could_extend and len(self._buf) < self._MAX_BUFFER:
                    return out
                self._buf = self._buf[m.end():]
                continue

            # No match yet at this position. Two possibilities:
            #   - "cite" is the start of a marker but the trailing
            #     "turnNkindM..." hasn't arrived yet → keep buffering
            #   - "cite" is just regular text (e.g. "citation") → emit it
            # Heuristic: if the buffer length is small AND the chars right
            # after "cite" still look like they could be building "turn",
            # stall. Otherwise emit "cite" and move on.
            tail = self._buf[4:]
            looks_like_marker = (
                tail == ""
                or "turn".startswith(tail[:4])
                or tail.startswith("turn")
            )
            if looks_like_marker and len(self._buf) < self._MAX_BUFFER:
                return out
            # Treat "cite" as ordinary text
            out += self._buf[:4]
            self._buf = self._buf[4:]

    def flush(self) -> str:
        out = self._buf
        self._buf = ""
        # One final cleanup pass on anything left in the buffer
        return _clean_citations(out)


# ---- Inline summaries for the per-tool log -------------------------------
def _summarize_args(name: str, args: Dict[str, Any]) -> str:
    if not isinstance(args, dict):
        return ""
    bits: List[str] = []
    interesting = (
        "query", "name", "columns", "where", "limit", "items", "titles",
        "industries", "keywords", "actor_id", "task", "code", "place_id",
        "domain", "linkedin_url", "email", "first_name", "last_name",
    )
    for k in interesting:
        if k not in args:
            continue
        v = args[k]
        if isinstance(v, list):
            preview = ", ".join(str(x) for x in v[:3])
            if len(v) > 3:
                preview += f" +{len(v) - 3}"
            bits.append(f"{k}=[{preview}]")
        elif isinstance(v, dict):
            keys = list(v.keys())[:3]
            bits.append(f"{k}={{{', '.join(keys)}{'...' if len(v) > 3 else ''}}}")
        elif isinstance(v, str):
            preview = v if len(v) <= 60 else v[:57] + "..."
            bits.append(f'{k}="{preview}"')
        else:
            bits.append(f"{k}={v}")
        if len(", ".join(bits)) > 120:
            break
    return ", ".join(bits)[:160]


def _summarize_result(name: str, result_text: str) -> str:
    if not result_text:
        return ""
    try:
        d = json.loads(result_text) if result_text.lstrip().startswith(("{", "[")) else None
    except (ValueError, TypeError):
        d = None
    if not isinstance(d, dict):
        return result_text[:120]
    if "error" in d:
        return f"error: {str(d['error'])[:120]}"
    bits: List[str] = []
    for key in ("returned", "count", "inserted", "merged", "deleted", "affected", "matched_rows", "cells_filled", "total_in_db", "total"):
        if key in d:
            bits.append(f"{key}={d[key]}")
    if "rows" in d and isinstance(d["rows"], list):
        bits.append(f"rows={len(d['rows'])}")
    if "results" in d and isinstance(d["results"], list):
        bits.append(f"results={len(d['results'])}")
    if "candidates" in d and isinstance(d["candidates"], list):
        bits.append(f"candidates={len(d['candidates'])}")
    if "by_status" in d and isinstance(d["by_status"], dict):
        bits.append(", ".join(f"{k}={v}" for k, v in d["by_status"].items()))
    if "ok" in d and not bits:
        bits.append("ok")
    return ", ".join(bits)[:160] if bits else "done"


# ---- Auto-name on first message ------------------------------------------
async def _auto_name_project(
    client: AsyncOpenAI, project: Project, user_message: str
) -> float:
    if project.name != "New Dataset":
        return 0.0
    try:
        resp = await client.responses.create(
            model=settings.OPENAI_MODEL,
            instructions=(
                "Generate a short project name (2-5 words) for this dataset request. "
                "Never include the word 'dataset'. Be concise. "
                "Return ONLY the name, nothing else. No quotes."
            ),
            input=[{"role": "user", "content": user_message}],
            max_output_tokens=30,
        )
        name = ""
        for item in resp.output:
            if item.type == "message":
                for block in item.content:
                    if block.type == "output_text":
                        name += block.text
        name = name.strip().strip('"').strip("'")
        if name:
            project.name = name[:100]
        return _response_cost_usd(resp)
    except Exception as e:
        log.warning(f"Auto-name failed: {e}")
        return 0.0


def _get_chat_history(db: Session, project_id: UUID, limit: int = 50) -> List[Dict[str, str]]:
    messages = (
        db.query(ChatMessage)
        .filter(ChatMessage.project_id == project_id)
        .order_by(ChatMessage.created_at.desc())
        .limit(limit)
        .all()
    )
    messages = list(reversed(messages))
    return [{"role": m.role, "content": m.content} for m in messages]


def _sse(payload: Dict[str, Any]) -> str:
    return f"data: {json.dumps(payload)}\n\n"


# ---- Reasoning effort escalation -----------------------------------------
# Default to "low" for snappy turns. Escalate when the user message signals
# they want deeper thinking — explicit asks ("think carefully", "deep
# research") OR dissatisfaction with the previous answer ("you're wrong",
# "no, that's not right"). Tier "high" reserved for explicit "really think
# hard" / "extensive research" framings.

_HIGH_EFFORT_SIGNALS = (
    "think very", "think really", "think extra hard",
    "extensive research", "very thorough", "exhaustive",
    "deep research", "really dig", "dig very deep",
)

_MEDIUM_EFFORT_SIGNALS = (
    "carefully", "thoroughly", "in depth", "in-depth", "deeply",
    "deep dive", "dig deeper", "think hard", "think harder",
    "do research", "research this", "research it", "analyze",
    "examine", "double check", "double-check", "be precise",
    # Dissatisfaction / correction signals — model probably needs more
    # reasoning to figure out what went wrong last turn.
    "you're wrong", "that's wrong", "this is wrong",
    "actually, no", "no, ", "not what i", "not what's",
    "fix this", "this is broken", "doesn't work", "didn't work",
    "missed", "you missed", "wrong answer",
)


def _resolve_reasoning_effort(user_content: str) -> str:
    return "medium"


# ---- Main streaming generator --------------------------------------------
async def stream_chat_response(
    project_id: UUID,
    user_id: UUID,
    user_content: str,
    request: Request,
) -> AsyncGenerator[str, None]:
    """Run one send-message turn and stream SSE events.

    Cooperative cancel: between rounds and between tool calls we check
    `request.is_disconnected()` (frontend AbortController.abort closes the
    stream). On disconnect we commit whatever ran and exit — the partial
    assistant message is preserved.
    """
    db = SessionLocal()
    try:
      with tracing.start_trace(
          "chat_send_message",
          user_id=str(user_id),
          project_id=str(project_id),
          input_text=user_content,
          metadata={"endpoint": "chat_api.stream"},
      ) as _trace_span:
        project = (
            db.query(Project)
            .filter(
                Project.id == project_id,
                Project.user_id == user_id,
                Project.deleted_at.is_(None),
            )
            .first()
        )
        if not project:
            yield _sse({"type": "error", "message": "Project not found"})
            return
        if project.mode != "chat":
            yield _sse({
                "type": "error",
                "message": "Project is not in chat mode (use the API's classic chat endpoint)",
            })
            return

        # Immediate signal so the UI doesn't sit silent during prompt
        # assembly + the OpenAI Responses request's first-byte latency.
        yield _sse({"type": "status", "content": "Thinking…"})

        history = _get_chat_history(db, project_id)
        is_first_message = len(history) == 0

        user_msg = ChatMessage(
            project_id=project_id, role="user", content=user_content
        )
        db.add(user_msg)
        db.commit()

        input_items: List[Dict[str, str]] = [
            {"role": "system", "content": agent.build_context_message(db, project)}
        ]
        input_items.extend(history)
        input_items.append({"role": "user", "content": user_content})

        client = get_openai_client()

        full_content = ""
        applied: Dict[str, Any] = {}
        thinking_total = 0.0
        total_cost = 0.0
        stopped = False
        tool_log: List[Dict[str, Any]] = []
        # Map call_id -> index in tool_log so we can update on tool_result
        tool_log_index: Dict[str, int] = {}
        # Web search citations the agent used. Stripped from the displayed
        # text and rendered separately as a sources list. Deduped by URL.
        sources: List[Dict[str, Any]] = []
        sources_by_url: Dict[str, int] = {}

        def _record_source(url: Optional[str], title: Optional[str]) -> Optional[Dict[str, Any]]:
            if not url:
                return None
            if url in sources_by_url:
                return None
            entry = {"n": len(sources) + 1, "url": url, "title": title or url}
            sources.append(entry)
            sources_by_url[url] = entry["n"]
            return entry

        try:
            # Manual conversation state — replayed in full each round.
            # Mirrors dsl_worker/agents/base.py so we don't depend on
            # `previous_response_id` (which fails with HTTP 400
            # "No tool call found for function call output" when the prior
            # response state is dropped/incomplete after long round 0 runs).
            running_input: List[Dict[str, Any]] = list(input_items)
            # Reasoning effort scales with the user's message — default low
            # for snap turns, escalates to medium/high when the user asks
            # for deeper thinking or signals dissatisfaction.
            _effort = _resolve_reasoning_effort(user_content)
            stripper = _CitationStripper()
            # Track web_search_call items we've already surfaced as synthetic
            # tool calls, so we don't double-emit on retry/replay.
            seen_web_search_ids: set = set()

            for round_num in range(MAX_TOOL_ROUNDS + 1):
                if await request.is_disconnected():
                    stopped = True
                    break

                stream_kwargs: Dict[str, Any] = {
                    "model": settings.OPENAI_MODEL,
                    "instructions": agent.SYSTEM_PROMPT,
                    "input": running_input,
                    "tools": agent.CHAT_TOOLS,
                    "reasoning": {"effort": _effort, "summary": "auto"},
                    "max_output_tokens": 8000,
                }

                # OpenAI sometimes ends a long-reasoning response without
                # sending response.completed (server-side stream cut on
                # heavy round-2 reasoning). Retry the inner stream once
                # when this happens AND we haven't streamed any visible
                # text yet — otherwise a retry would produce duplicate
                # tokens for the user. Manual re-runs always succeed; an
                # automatic retry turns the freeze into a slower-but-OK turn.
                MAX_STREAM_RETRIES = 1
                final_response = None
                round_thinking_start = time.time()
                got_output_this_round = False
                round_text_collected = ""

                for attempt in range(MAX_STREAM_RETRIES + 1):
                    # Reset per-attempt timing/output. full_content,
                    # tool_log, seen_web_search_ids etc. stay (cumulative).
                    round_thinking_start = time.time()
                    got_output_this_round = False
                    round_text_collected = ""
                    final_response = None

                    trace_label = (
                        f"openai.responses.round_{round_num}"
                        if attempt == 0
                        else f"openai.responses.round_{round_num}_retry_{attempt}"
                    )
                    with tracing.start_generation(
                        trace_label,
                        model=settings.OPENAI_MODEL,
                        input_payload=stream_kwargs.get("input"),
                        metadata={"round": round_num, "attempt": attempt},
                    ) as gen:
                        async with client.responses.stream(**stream_kwargs) as stream:
                            async for event in stream:
                                event_type = getattr(event, "type", None)

                                if event_type == "response.reasoning_summary_text.delta":
                                    delta = getattr(event, "delta", "") or ""
                                    if delta:
                                        yield _sse({"type": "thinking", "content": delta})
                                        await asyncio.sleep(0)

                                elif event_type == "response.output_text.annotation.added":
                                    ann = getattr(event, "annotation", None)
                                    if ann is not None:
                                        a_type = getattr(ann, "type", None) or (
                                            ann.get("type") if isinstance(ann, dict) else None
                                        )
                                        if a_type == "url_citation":
                                            url = getattr(ann, "url", None) or (
                                                ann.get("url") if isinstance(ann, dict) else None
                                            )
                                            title = getattr(ann, "title", None) or (
                                                ann.get("title") if isinstance(ann, dict) else None
                                            )
                                            added = _record_source(url, title)
                                            if added is not None:
                                                yield _sse({"type": "source_added", **added})
                                                await asyncio.sleep(0)

                                elif event_type == "response.output_item.added":
                                    # Surface OpenAI's built-in web_search calls
                                    # as synthetic tool_call entries so the UI
                                    # doesn't go silent during searches (which
                                    # can each take 30-60s and chain).
                                    added_item = getattr(event, "item", None)
                                    if (
                                        added_item is not None
                                        and getattr(added_item, "type", None) == "web_search_call"
                                    ):
                                        item_id = getattr(added_item, "id", None) or ""
                                        if item_id and item_id not in seen_web_search_ids:
                                            seen_web_search_ids.add(item_id)
                                            action = getattr(added_item, "action", None)
                                            a_type = getattr(action, "type", None) if action else None
                                            if a_type == "search":
                                                preview = (getattr(action, "query", None) or "")[:120]
                                                args_preview = f'query="{preview}"'
                                            elif a_type == "open_page":
                                                preview = (getattr(action, "url", None) or "")[:120]
                                                args_preview = f'url="{preview}"'
                                            elif a_type == "find_in_page":
                                                preview = (getattr(action, "query", None) or "")[:120]
                                                args_preview = f'find="{preview}"'
                                            else:
                                                args_preview = ""
                                            tool_log_index[item_id] = len(tool_log)
                                            tool_log.append({
                                                "id": item_id,
                                                "name": "web_search",
                                                "args_preview": args_preview,
                                            })
                                            yield _sse({
                                                "type": "tool_call",
                                                "id": item_id,
                                                "name": "web_search",
                                                "args_preview": args_preview,
                                            })
                                            await asyncio.sleep(0)

                                elif event_type == "response.output_item.done":
                                    done_item = getattr(event, "item", None)
                                    if (
                                        done_item is not None
                                        and getattr(done_item, "type", None) == "web_search_call"
                                    ):
                                        item_id = getattr(done_item, "id", None) or ""
                                        if item_id:
                                            status = getattr(done_item, "status", None) or "completed"
                                            idx = tool_log_index.get(item_id)
                                            summary = "done" if status == "completed" else status
                                            if idx is not None:
                                                tool_log[idx]["summary"] = summary
                                            yield _sse({
                                                "type": "tool_result",
                                                "id": item_id,
                                                "name": "web_search",
                                                "summary": summary,
                                                "cost": 0,
                                            })
                                            await asyncio.sleep(0)

                                elif event_type == "response.output_text.delta":
                                    if not got_output_this_round:
                                        got_output_this_round = True
                                        thinking_total += time.time() - round_thinking_start
                                        if (
                                            round_num > 0
                                            and full_content
                                            and not full_content.endswith("\n\n")
                                        ):
                                            sep = "\n\n"
                                            full_content += sep
                                            yield _sse({"type": "token", "content": sep})

                                    token = event.delta or ""
                                    if token:
                                        round_text_collected += token
                                        clean = stripper.feed(token)
                                        if clean:
                                            full_content += clean
                                            yield _sse({"type": "token", "content": clean})
                                            await asyncio.sleep(0)

                            # OpenAI's stream sometimes ends without sending
                            # a `response.completed` event. The SDK raises
                            # RuntimeError in that case. Caught and decided
                            # below: retry, or treat as incomplete.
                            try:
                                final_response = await stream.get_final_response()
                            except RuntimeError as _rt:
                                if "response.completed" in str(_rt):
                                    final_response = None
                                else:
                                    raise

                    # End of attempt's tracing context. Now decide:
                    # success → break; text-already-emitted or out-of-retries
                    # → break (fall through to incomplete handler);
                    # otherwise → retry.
                    if final_response is not None:
                        break
                    if round_text_collected:
                        log.warning(
                            f"round {round_num}: stream ended without completion "
                            f"event after {len(round_text_collected)} chars "
                            f"emitted — treating as incomplete"
                        )
                        break
                    if attempt >= MAX_STREAM_RETRIES:
                        log.warning(
                            f"round {round_num}: stream ended without completion "
                            f"event after {attempt + 1} attempt(s) — giving up"
                        )
                        break
                    log.warning(
                        f"round {round_num} attempt {attempt + 1}: stream ended "
                        f"without completion event (no text emitted yet), retrying"
                    )

                if final_response is None:
                    # Incomplete round: skip post-stream processing and
                    # exit the loop. full_content has whatever text we
                    # streamed; the assistant message will persist with
                    # that + a `stopped` flag.
                    stopped = True
                    remaining = stripper.flush()
                    if remaining:
                        full_content += remaining
                        yield _sse({"type": "token", "content": remaining})
                    if not got_output_this_round:
                        thinking_total += time.time() - round_thinking_start
                    break

                    # Fallback: walk the final response for any url_citation
                    # annotations the streaming event missed (depends on SDK
                    # / model version). De-duped by URL via _record_source.
                    for out_item in getattr(final_response, "output", []) or []:
                        content = getattr(out_item, "content", None) or []
                        for block in content:
                            anns = getattr(block, "annotations", None) or []
                            for ann in anns:
                                a_type = getattr(ann, "type", None) or (
                                    ann.get("type") if isinstance(ann, dict) else None
                                )
                                if a_type != "url_citation":
                                    continue
                                url = getattr(ann, "url", None) or (
                                    ann.get("url") if isinstance(ann, dict) else None
                                )
                                title = getattr(ann, "title", None) or (
                                    ann.get("title") if isinstance(ann, dict) else None
                                )
                                added = _record_source(url, title)
                                if added is not None:
                                    yield _sse({"type": "source_added", **added})
                                    await asyncio.sleep(0)

                    usage = getattr(final_response, "usage", None)
                    usage_dict: Dict[str, Any] = {}
                    if usage:
                        usage_dict = {
                            "input": usage.input_tokens or 0,
                            "output": usage.output_tokens or 0,
                        }
                        details = getattr(usage, "input_tokens_details", None)
                        if details:
                            usage_dict["cache_read_input_tokens"] = (
                                getattr(details, "cached_tokens", 0) or 0
                            )
                    tracing.update_generation(
                        gen,
                        output=round_text_collected,
                        usage=usage_dict,
                        cost_usd=_response_cost_usd(final_response),
                    )

                remaining = stripper.flush()
                if remaining:
                    full_content += remaining
                    yield _sse({"type": "token", "content": remaining})

                total_cost += _response_cost_usd(final_response)

                # Bill built-in web_search calls separately — they're not
                # included in usage.input/output_tokens, only as
                # web_search_call items in response.output.
                web_search_count = sum(
                    1 for item in final_response.output
                    if item.type == "web_search_call"
                )
                if web_search_count:
                    # Main agent uses search_context_size="low" (see agent.py CHAT_TOOLS).
                    total_cost += web_search_count * _sources.WEB_SEARCH_USD_BY_TIER["low"]

                if not got_output_this_round:
                    thinking_total += time.time() - round_thinking_start

                # Capture every output item into running_input so the next
                # round sees the full conversation. Reasoning items get a
                # manual dump because model_dump(exclude_none=True) drops
                # the required `summary` field when the API returns null.
                for out_item in final_response.output or []:
                    if out_item.type == "reasoning":
                        summary = []
                        if out_item.summary:
                            summary = [
                                {"type": s.type, "text": s.text}
                                for s in out_item.summary
                            ]
                        running_input.append({
                            "type": "reasoning",
                            "id": out_item.id,
                            "summary": summary,
                        })
                    else:
                        running_input.append(out_item.model_dump(exclude_none=True))

                tool_calls = [
                    item for item in final_response.output
                    if item.type == "function_call"
                ]
                if not tool_calls:
                    break

                yield _sse({
                    "type": "tool_start",
                    "tools": [item.name for item in tool_calls],
                })
                await asyncio.sleep(0)

                tool_result_items: List[Dict[str, Any]] = []
                round_applied: Dict[str, Any] = {}

                for item in tool_calls:
                    if await request.is_disconnected():
                        stopped = True
                        break

                    try:
                        args = json.loads(item.arguments)
                    except json.JSONDecodeError:
                        tool_result_items.append({
                            "type": "function_call_output",
                            "call_id": item.call_id,
                            "output": "Error: Invalid arguments",
                        })
                        continue

                    args_preview = _summarize_args(item.name, args)
                    tool_log_index[item.call_id] = len(tool_log)
                    tool_log.append({
                        "id": item.call_id,
                        "name": item.name,
                        "args_preview": args_preview,
                    })
                    yield _sse({
                        "type": "tool_call",
                        "id": item.call_id,
                        "name": item.name,
                        "args_preview": args_preview,
                    })
                    await asyncio.sleep(0)

                    # For long tools (notably rows_fill), pipe progress events
                    # back through SSE so the UI isn't a black box.
                    progress_q: asyncio.Queue = asyncio.Queue()

                    async def _on_progress(ev: Dict[str, Any]) -> None:
                        ev = dict(ev)
                        ev.setdefault("tool_call_id", item.call_id)
                        await progress_q.put(ev)
                        # Hand control back to the event loop so the SSE
                        # drain coroutine actually runs between emits.
                        # Without this, an unbounded asyncio.Queue.put never
                        # yields, and tight emit loops (e.g. rows_add over
                        # N items) starve the streaming generator — all
                        # events arrive in a single burst at tool end.
                        await asyncio.sleep(0)

                    with tracing.start_span(
                        f"tool.{item.name}",
                        input_payload=args,
                        metadata={"call_id": item.call_id, "round": round_num},
                    ) as tool_span:
                        tool_task = asyncio.create_task(
                            agent.execute_tool(
                                db, project, item.name, args,
                                progress_cb=_on_progress,
                            )
                        )

                        while True:
                            # Race: queue.get vs task done. Whichever fires first.
                            getter = asyncio.create_task(progress_q.get())
                            done_set, _pending = await asyncio.wait(
                                {getter, tool_task},
                                return_when=asyncio.FIRST_COMPLETED,
                            )
                            if getter in done_set:
                                ev = getter.result()
                                yield _sse(ev)
                                await asyncio.sleep(0)
                            else:
                                getter.cancel()
                            if tool_task in done_set:
                                # Drain anything remaining in the queue
                                while not progress_q.empty():
                                    ev = progress_q.get_nowait()
                                    yield _sse(ev)
                                break

                        item_applied, result, tool_cost = await tool_task
                        result_text = agent.format_tool_result(item.name, result)
                        # Append a one-line table-state hint to every tool
                        # result so the agent can't drift off thinking it
                        # added rows when the table is still empty.
                        result_text += agent.project_state_hint(db, project)
                        total_cost += tool_cost
                        round_applied.update(item_applied)
                        applied.update(item_applied)
                        tracing.update_span(
                            tool_span,
                            output=result,
                            cost_usd=tool_cost,
                        )

                    summary = _summarize_result(item.name, result_text)
                    cost_rounded = round(tool_cost, 4) if tool_cost > 0 else 0
                    idx = tool_log_index.get(item.call_id)
                    if idx is not None:
                        tool_log[idx]["summary"] = summary
                        tool_log[idx]["cost"] = cost_rounded

                    yield _sse({
                        "type": "tool_result",
                        "id": item.call_id,
                        "name": item.name,
                        "summary": summary,
                        "cost": cost_rounded,
                    })
                    await asyncio.sleep(0)

                    tool_result_items.append({
                        "type": "function_call_output",
                        "call_id": item.call_id,
                        "output": result_text,
                    })

                # Carry tool outputs into the conversation state so the
                # next round's input includes them (replaces the old
                # previous_response_id-based chaining).
                running_input.extend(tool_result_items)

                if stopped:
                    break

                # Surface suggest_replies as a dedicated SSE event
                # (separate from the generic "change" feed which is for
                # table mutations). Frontend renders these as clickable
                # text replies under the assistant's message.
                if isinstance(round_applied.get("suggestions"), dict):
                    sg = round_applied["suggestions"]
                    yield _sse({
                        "type": "suggestions",
                        "items": sg.get("items") or [],
                    })

                for change in agent.describe_applied(round_applied):
                    if change.field == "questions":
                        yield _sse({"type": "questions", "questions": change.value})
                    else:
                        event_data: dict = {
                            "type": "change",
                            "field": change.field,
                            "description": change.description,
                        }
                        if change.value is not None:
                            event_data["value"] = change.value
                        yield _sse(event_data)

                # End the conversation after these tools — they're terminal:
                # ask_questions waits for a structured answer;
                # suggest_replies hands control to the user via clickable
                # text suggestions.
                if any(
                    item.name in (
                        "ask_questions",
                        "suggest_replies",
                    )
                    for item in tool_calls
                ):
                    break

        except Exception as e:
            log.exception("OpenAI streaming error")
            yield _sse({"type": "error", "message": f"AI service error: {str(e)}"})
            # Best-effort: persist whatever we have so the user sees a trace
            # on history reload instead of a silent disappearance. Without
            # this, a mid-stream error leaves only the user message in DB
            # and the UI shows nothing on reload.
            try:
                err_ac: Dict[str, Any] = {"error": str(e)[:500]}
                if applied:
                    err_ac["changes"] = applied
                if tool_log:
                    err_ac["tool_log"] = tool_log
                if total_cost > 0:
                    err_ac["total_cost_usd"] = round(total_cost, 4)
                if sources:
                    err_ac["sources"] = sources
                if thinking_total >= 0.5:
                    err_ac["thinking_duration"] = round(thinking_total, 1)
                partial = ChatMessage(
                    project_id=project_id,
                    role="assistant",
                    content=_clean_citations(full_content),
                    applied_changes=err_ac,
                )
                db.add(partial)
                _charge_credits(db, user_id, total_cost, project_id=project_id)
                db.commit()
            except Exception:
                log.exception("Failed to persist error stub")
                try:
                    db.rollback()
                except Exception:
                    pass
            return

        # Forced text wrap-up if we exited the tool loop with no text. Hits
        # when the agent burned all rounds on tools and never produced a
        # final reply (e.g. round cap reached). Without this we save an
        # empty assistant message — visually "nothing happened" to the user
        # even though tools ran. One non-streaming call replays the full
        # input and forces a short summary.
        # NOTE: we don't use previous_response_id here — Azure's saved
        # state can drop after long rounds, causing HTTP 400 "No tool
        # call found for function call output". Pass running_input
        # (the replayed conversation) directly instead.
        if (
            not stopped
            and len(tool_log) > 0
            and not full_content.strip()
        ):
            try:
                wrap_input = list(running_input) + [{
                    "role": "user",
                    "content": (
                        "(System note — not from the user.) You ran tools "
                        "but produced no text reply, and the tool-round "
                        "cap stopped the loop. Look at the project state "
                        "in the prior context and write ONE short reply "
                        "summarizing what you did this turn AND, if rows "
                        "did not actually land in the table, finish the "
                        "job: add columns + commit candidates_to_rows "
                        "before replying. The user is staring at the "
                        "table waiting."
                    ),
                }]
                wrap = await client.responses.create(
                    model=settings.OPENAI_MODEL,
                    instructions=agent.SYSTEM_PROMPT,
                    input=wrap_input,
                    max_output_tokens=600,
                )
                total_cost += _response_cost_usd(wrap)
                for item in wrap.output:
                    if item.type == "message":
                        for block in item.content:
                            text = getattr(block, "text", None)
                            if text:
                                full_content += text
                                yield _sse({"type": "token", "content": text})
            except Exception:
                log.exception("forced-text wrap-up failed")

        # Auto-name on first message
        if is_first_message and not stopped:
            name_cost = await _auto_name_project(client, project, user_content)
            total_cost += name_cost
            yield _sse({"type": "project_name", "name": project.name})

        full_content = _clean_citations(full_content)

        ac_data: Dict[str, Any] = {}
        if applied:
            ac_data["changes"] = applied
        if thinking_total >= 0.5:
            ac_data["thinking_duration"] = round(thinking_total, 1)
        if stopped:
            ac_data["stopped"] = True
        if tool_log:
            ac_data["tool_log"] = tool_log
        if total_cost > 0:
            ac_data["total_cost_usd"] = round(total_cost, 4)
        if sources:
            ac_data["sources"] = sources

        assistant_msg = ChatMessage(
            project_id=project_id,
            role="assistant",
            content=full_content,
            applied_changes=ac_data if ac_data else None,
        )
        db.add(assistant_msg)

        _charge_credits(db, user_id, total_cost, project_id=project_id)
        db.commit()
        db.refresh(assistant_msg)

        tracing.update_span(
            _trace_span,
            output=full_content,
            cost_usd=total_cost,
            metadata={
                "tool_count": len(tool_log),
                "stopped": stopped,
                "message_id": str(assistant_msg.id),
            },
        )

        done_event: Dict[str, Any] = {
            "type": "done",
            "message_id": str(assistant_msg.id),
            "total_cost_usd": round(total_cost, 4),
        }
        if thinking_total >= 0.5:
            done_event["thinking_duration"] = round(thinking_total, 1)
        if stopped:
            done_event["stopped"] = True
        if sources:
            done_event["sources"] = sources
        yield _sse(done_event)

    except Exception as e:
        log.exception("Streaming error")
        yield _sse({"type": "error", "message": str(e)})
    finally:
        db.close()
        tracing.flush()
