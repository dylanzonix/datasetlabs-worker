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
from dsl_api.models import Account, ChatMessage, ChatRun, Project

from dsl_worker.chat_api import agent, runs, sources as _sources, tracing

log = logging.getLogger(__name__)


# Shown to the user when an OpenAI stream ends without `response.completed`
# (e.g. `response.incomplete` due to max_output_tokens, content_filter, or a
# network drop). We replace whatever partial text the model emitted — it's
# usually a half-thought that would read like gibberish — with this clean
# warning. The partial reasoning + completed tool calls ARE preserved in
# the assistant message's `applied_changes.resume_input` so the next user
# message will resume from where the model left off.
_INCOMPLETE_WARNING = (
    "Something went wrong on my side mid-thought. Send any message "
    "(e.g. \"continue\") and I'll pick up where I left off."
)


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

# No round cap — the loop terminates naturally when the agent stops
# calling tools (or hits ask_questions / suggest_replies, which are
# turn-ending). Pause/cancel still bound runaway behavior via run.status
# polling between rounds. With the "don't perfect candidates" prompt
# guidance the agent self-limits; an explicit cap was just a way to
# truncate output and confuse the user with "I ran out of rounds".


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


def _commit_with_deadlock_retry(db: Session, max_attempts: int = 3) -> None:
    """Commit, retrying on transient Postgres deadlocks.

    A turn's final commit batches the assistant message, the auto-name
    rename, and the credit charge — that touches projects + accounts +
    chat_messages + balance_ledger in one go. If a concurrent request
    holds locks in opposite order Postgres picks a victim, and we'd
    otherwise lose the entire turn (the user already saw the streamed
    reply). Postgres's deadlock detector is fast and the loser is
    safe to re-run; one or two retries clears it in practice.
    """
    for attempt in range(max_attempts):
        try:
            db.commit()
            return
        except Exception as e:
            cause = e.__cause__ or e.__context__
            cause_name = type(cause).__name__ if cause is not None else ""
            is_deadlock = (
                "DeadlockDetected" in cause_name
                or "deadlock detected" in str(e).lower()
            )
            if is_deadlock and attempt < max_attempts - 1:
                log.warning(
                    f"deadlock on commit (attempt {attempt + 1}/{max_attempts}), "
                    f"rolling back + retrying"
                )
                try:
                    db.rollback()
                except Exception:
                    log.exception("rollback after deadlock failed")
                continue
            raise


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


class _BillingMeter:
    """Incremental credit-charging for a single chat run. Accumulates
    unbilled cost; flushes whenever the running balance crosses the
    consume_credits floor (0.01 credits ≈ $0.0025 at 25¢/credit).

    Without this, a 5-minute turn shows $0 spent until the very end
    when the legacy single-call _charge_credits fires — confusing for
    the user staring at a stable balance while a deep agent run burns
    real money. With it, the FE's credit display ticks down live.

    Also enforces balance: `out_of_credits` is set when consume_credits
    returns False (account has no remaining balance). The agent loop
    polls this between tool calls and ends the run cleanly when set —
    same pattern as runs.check_should_stop. Without this, an account
    at $0 keeps firing OpenAI calls with nothing to charge against.
    """

    def __init__(self, db: Session, user_id, project_id) -> None:
        self._db = db
        self._user_id = user_id
        self._project_id = project_id
        self._unbilled_cost = 0.0
        self._out_of_credits = False

    @property
    def out_of_credits(self) -> bool:
        return self._out_of_credits

    def add(self, delta_usd: float) -> None:
        if delta_usd <= 0:
            return
        self._unbilled_cost += delta_usd
        self._try_flush()

    def flush(self) -> None:
        # Final pass at end-of-turn / pause / error. Residuals below
        # the floor are intentionally dropped (rounding loss < $0.0025
        # per turn — well under measurement noise).
        self._try_flush()

    def force_flush_with_fresh_session(self) -> bool:
        """Last-resort billing flush when the main session is poisoned.

        Opens a brand-new SessionLocal so a PendingRollbackError or
        deadlock on `self._db` can't strand unbilled cost. Same idea as
        _force_persist_assistant_message: belt-and-suspenders so a
        failure mid-turn never silently drops accrued compute charges.

        Returns True if anything was actually flushed (ledger row
        written or below-floor residual swallowed), False on failure.

        Bug seen on ee0db354 (Apr 28): run died ~T+469s after last
        successful billing flush, then ran ~159s more before crashing
        terminally. That tail of work never made it to the ledger
        because billing.flush() on the poisoned main session raised
        and the inner except just logged.
        """
        if self._unbilled_cost <= 0:
            return True
        credits = self._unbilled_cost / settings.COMPUTE_COST_PER_CREDIT
        if credits < 0.01:
            self._unbilled_cost = 0.0  # below floor — accept rounding loss
            return True
        fresh = SessionLocal()
        try:
            account = (
                fresh.query(Account)
                .filter(Account.user_id == str(self._user_id))
                .first()
            )
            if account is None:
                log.warning(
                    "force_flush: no account for user %s; %s credits unbilled",
                    self._user_id, credits,
                )
                return False
            ok = consume_credits(
                fresh, account, credits, project_id=self._project_id
            )
            fresh.commit()
            self._unbilled_cost = 0.0
            if not ok:
                self._out_of_credits = True
            log.warning(
                "force_flush: charged %.4f credits via fresh session for user %s",
                credits, self._user_id,
            )
            return True
        except Exception:
            log.exception(
                "force_flush: fresh-session billing failed for user %s "
                "(%.4f credits unbilled)", self._user_id, credits,
            )
            try:
                fresh.rollback()
            except Exception:
                pass
            return False
        finally:
            fresh.close()

    def _try_flush(self) -> None:
        if self._unbilled_cost <= 0:
            return
        credits = self._unbilled_cost / settings.COMPUTE_COST_PER_CREDIT
        if credits < 0.01:
            return
        account = (
            self._db.query(Account)
            .filter(Account.user_id == str(self._user_id))
            .first()
        )
        if account is None:
            log.warning("No account for user %s; skipping incremental charge", self._user_id)
            return
        ok = consume_credits(
            self._db, account, credits, project_id=self._project_id
        )
        self._unbilled_cost = 0.0
        if not ok:
            self._out_of_credits = True
            log.warning(
                "user %s ran out of credits mid-turn (project %s); "
                "agent loop will end cleanly at next checkpoint",
                self._user_id, self._project_id,
            )


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
# After the bracketed-pair regex strips well-formed citations, any
# remaining lone 【 or 】 is an artifact of an incomplete citation
# (e.g. the model emitted only the closing 】 at end-of-stream). No
# legitimate user-facing prose contains these glyphs, so global strip
# is safe.
_LONELY_BRACKET_RE = re.compile(r'[【】]')


def _clean_citations(text: str) -> str:
    text = _PUA_RE.sub('', text)
    text = _BRACKETED_RE.sub('', text)
    text = _BARE_CITE_RE.sub('', text)
    text = _LONELY_BRACKET_RE.sub('', text)
    return text


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

        def _emit(chunk: str) -> str:
            # Anything reaching emit-stage is no-longer-a-citation by
            # construction (bracketed pairs are matched + dropped above).
            # Lone 【 or 】 surviving here is a stray artifact — strip
            # before flushing to the SSE stream.
            return _LONELY_BRACKET_RE.sub('', chunk)

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
                    return _emit(out)
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
                return _emit(out)

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
                    return _emit(out)
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
                return _emit(out)
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


def _attr_or_key(obj: Any, *names: str) -> Any:
    """Pull an attribute from a pydantic-ish model OR a key from a dict.
    OpenAI's stream events sometimes hand us typed objects, sometimes
    dicts depending on the event type — accept both."""
    for n in names:
        if obj is None:
            return None
        if hasattr(obj, n):
            v = getattr(obj, n, None)
            if v is not None:
                return v
        if isinstance(obj, dict) and n in obj and obj[n] is not None:
            return obj[n]
    return None


def _web_search_args_preview(item: Any) -> str:
    """Format a web_search_call's action as `kind="value"` for the UI.

    OpenAI's web_search built-in fires `output_item.added`,
    `response.web_search_call.*`, and `output_item.done` events with
    an `action` sub-object. The action type is one of `search`
    (action.query), `open_page` (action.url), or `find_in_page`
    (action.query). The query may be empty on the initial added event
    and populated later — callers re-run this on subsequent events to
    backfill.
    """
    action = _attr_or_key(item, "action")
    if action is None:
        return ""
    a_type = _attr_or_key(action, "type")
    if a_type == "search":
        raw = _attr_or_key(action, "query")
        v = (str(raw) if raw else "")[:120]
        return f'query="{v}"' if v else ""
    if a_type == "open_page":
        raw = _attr_or_key(action, "url")
        v = (str(raw) if raw else "")[:160]
        return f'url="{v}"' if v else ""
    if a_type == "find_in_page":
        raw = _attr_or_key(action, "query", "pattern")
        v = (str(raw) if raw else "")[:120]
        return f'find="{v}"' if v else ""
    return ""


# Frontend-tier (auto/fast/balanced/highest) → OpenAI reasoning_effort.
# `auto` falls back to the dynamic per-message resolver above so the
# default behavior is unchanged when the user hasn't picked a tier.
_EFFORT_TO_REASONING = {
    "fast": "low",
    "balanced": "medium",
    "highest": "high",
}

# Per-tier hint appended to the system context message so the agent
# scales research breadth alongside the reasoning_effort. The numbers
# are advisory — the agent already knows how to scope; this just sets
# the dial. `auto` gets no hint (agent uses its own judgment).
_EFFORT_HINT = {
    "fast": (
        "Effort: fast. Minimize research and tool depth: cap web_search "
        "at 1-2 calls, prefer the cheapest source that fits, return a "
        "small starter batch quickly. Skip optional verification."
    ),
    "balanced": (
        "Effort: balanced. Normal research depth: ~3 web_searches max "
        "before committing rows; use direct API sources when they fit; "
        "reasonable batch size."
    ),
    "highest": (
        "Effort: highest. The user wants thoroughness. Cast a wider net "
        "(more searches/sources OK), pull a larger batch, double-check "
        "ambiguous fields, but still commit rows promptly — don't "
        "research forever."
    ),
}


# ---- Run-aware agent loop ------------------------------------------------
async def run_agent_loop(
    run_id: UUID,
    user_id: UUID,
    user_content: str,
    effort: Optional[str] = None,
) -> None:
    """Execute one chat turn against a ChatRun row.

    Decoupled from any HTTP request. Events are persisted to
    ChatRunEvent and fanned out via the in-process bus; subscribers
    (HTTP handlers tailing the events) consume them.

    Stop signals: `runs.check_should_stop()` is polled between rounds
    and between tool calls. On `pause` we save partial progress and
    mark the run paused; on `cancel` we save and mark cancelled.
    """
    db = SessionLocal()
    try:
        run = db.query(ChatRun).filter(ChatRun.id == run_id).first()
        if run is None:
            log.warning("run_agent_loop: run %s not found", run_id)
            return

        project = (
            db.query(Project)
            .filter(
                Project.id == run.project_id,
                Project.user_id == user_id,
                Project.deleted_at.is_(None),
            )
            .first()
        )
        if project is None:
            runs.emit_event(db, run, "error", {"message": "Project not found"})
            runs.mark_run_cancelled(db, run, {"reason": "project-not-found"})
            return
        if project.mode != "chat":
            runs.emit_event(db, run, "error", {"message": "Project is not in chat mode"})
            runs.mark_run_cancelled(db, run, {"reason": "wrong-mode"})
            return

        # The triggering user message was created by start_run(). Look
        # it up so we can replay history *excluding* it (we'll inject
        # the user content explicitly into input_items).
        user_msg = (
            db.query(ChatMessage)
            .filter(ChatMessage.id == run.triggering_message_id)
            .first()
        )

        # Initial signal so live subscribers see something fast.
        runs.update_run_phase(db, run, "thinking")
        runs.emit_event(db, run, "status", {"content": "Thinking…"})

        # Replay-friendly chat history for the model: exclude the
        # current user message (already injected at the end below).
        history_msgs = (
            db.query(ChatMessage)
            .filter(
                ChatMessage.project_id == project.id,
                ChatMessage.id != (user_msg.id if user_msg else None),
            )
            .order_by(ChatMessage.created_at.desc())
            .limit(50)
            .all()
        )
        # If the most recent assistant message left a `resume_input`
        # behind (because that turn's OpenAI stream ended with
        # `response.incomplete`), replay the full prior input —
        # including reasoning + tool calls + tool results — instead of
        # rebuilding from `ChatMessage.content` only. This is what lets
        # the user say "continue" after a silent stream death and have
        # the model pick up where it left off without redoing research.
        resume_input: Optional[List[Dict[str, Any]]] = None
        for m in history_msgs:  # ordered desc, most recent first
            if m.role == "assistant":
                ac = m.applied_changes or {}
                candidate = ac.get("resume_input")
                if isinstance(candidate, list) and candidate:
                    resume_input = candidate
                break

        if resume_input:
            history = []  # resume_input already contains prior history
        else:
            history = [
                {"role": m.role, "content": m.content} for m in reversed(history_msgs)
            ]
        is_first_message = (not resume_input) and len(history) == 0

        # Fork a new ProjectVersion for this turn — same semantics as
        # the legacy path. user_msg.version_id is set inside the helper.
        new_version = agent.start_user_turn_version(db, project, user_msg)
        run.version_id = new_version.id
        db.commit()

        runs.emit_event(db, run, "version", {
            "version_id": str(new_version.id),
            "version_number": new_version.version_number,
            "label": new_version.label,
            "message_id": str(user_msg.id) if user_msg else None,
        })

        # Server-side gate for "highest" effort tier on free plans.
        effective_effort = effort
        if effective_effort == "highest":
            from dsl_api.models.subscription import Subscription
            sub = (
                db.query(Subscription)
                .filter(Subscription.user_id == user_id)
                .order_by(Subscription.created_at.desc())
                .first()
            )
            is_paid = (
                sub is not None
                and sub.plan != "free"
                and sub.status in ("active", "past_due")
            )
            if not is_paid:
                effective_effort = "balanced"

        context_msg = agent.build_context_message(db, project)
        effort_hint = _EFFORT_HINT.get(effective_effort or "")
        if effort_hint:
            context_msg = f"{context_msg}\n\n{effort_hint}"

        if resume_input:
            # Replace the (stale) system context_msg from the prior run
            # with a fresh one — column/row counts may have changed —
            # then keep everything else and append the new user message.
            body = resume_input
            if (
                body
                and isinstance(body[0], dict)
                and body[0].get("role") == "system"
            ):
                body = body[1:]
            input_items: List[Dict[str, Any]] = [
                {"role": "system", "content": context_msg},
                *body,
                {"role": "user", "content": user_content},
            ]
        else:
            input_items = [
                {"role": "system", "content": context_msg},
                *history,
                {"role": "user", "content": user_content},
            ]

        client = get_openai_client()

        full_content = ""
        applied: Dict[str, Any] = {}
        thinking_total = 0.0
        total_cost = 0.0
        stopped = False
        stop_reason: Optional[str] = None  # "pause" | "cancel" | None
        # Set if this run dies because the OpenAI stream ended without
        # `response.completed` (incomplete / network drop). Snapshot of
        # the full `running_input` at the moment of failure goes here so
        # the next user message can resume from the same model state.
        resume_input_snapshot: Optional[List[Dict[str, Any]]] = None
        resume_reason: Optional[str] = None
        # Charges credits live as cost accumulates instead of one big
        # debit at end-of-turn. See _BillingMeter docstring.
        billing = _BillingMeter(db, user_id, project.id)
        tool_log: List[Dict[str, Any]] = []
        tool_log_index: Dict[str, int] = {}
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
            running_input: List[Dict[str, Any]] = list(input_items)
            _effort = _EFFORT_TO_REASONING.get(effective_effort or "") \
                or _resolve_reasoning_effort(user_content)
            stripper = _CitationStripper()
            seen_web_search_ids: set = set()

            round_num = 0
            while True:
                round_num += 1
                signal = runs.check_should_stop(db, run)
                if signal:
                    stopped = True
                    stop_reason = signal
                    break
                if billing.out_of_credits:
                    stopped = True
                    stop_reason = "out_of_credits"
                    break

                runs.update_run_phase(db, run, f"reasoning (round {round_num})")

                stream_kwargs: Dict[str, Any] = {
                    "model": settings.OPENAI_MODEL,
                    "instructions": agent.SYSTEM_PROMPT,
                    "input": running_input,
                    "tools": agent.CHAT_TOOLS,
                    "reasoning": {"effort": _effort, "summary": "auto"},
                    "max_output_tokens": 200000,
                }

                MAX_STREAM_RETRIES = 1
                final_response = None
                round_thinking_start = time.time()
                got_output_this_round = False
                round_text_collected = ""

                # Per-attempt accumulator of fully-streamed output items.
                # Used as the source-of-truth for "what the model managed
                # to emit before the cap hit" when `final_response` ends
                # up None (stream ended on `response.incomplete` instead
                # of `response.completed`). Reset on every attempt so a
                # retry doesn't double-count items from the prior try.
                collected_round_items: List[Any] = []
                incomplete_reason: Optional[str] = None

                for attempt in range(MAX_STREAM_RETRIES + 1):
                    round_thinking_start = time.time()
                    got_output_this_round = False
                    round_text_collected = ""
                    final_response = None
                    collected_round_items = []
                    incomplete_reason = None
                    # Set when a tool fires mid-stream after text was emitted;
                    # the next text delta prepends `\n\n` so narration around
                    # built-in tools (web_search) doesn't mush together.
                    text_resume_needs_separator = False

                    async with client.responses.stream(**stream_kwargs) as stream:
                        async for event in stream:
                            event_type = getattr(event, "type", None)

                            if event_type == "response.reasoning_summary_text.delta":
                                delta = getattr(event, "delta", "") or ""
                                if delta:
                                    # Live-only: bus fanout, no DB write.
                                    runs.publish_thinking_delta(run.id, delta)
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
                                            runs.emit_event(db, run, "source_added", added)
                                            await asyncio.sleep(0)

                            elif event_type == "response.output_item.added":
                                added_item = getattr(event, "item", None)
                                if (
                                    added_item is not None
                                    and getattr(added_item, "type", None) == "web_search_call"
                                ):
                                    item_id = getattr(added_item, "id", None) or ""
                                    if item_id and item_id not in seen_web_search_ids:
                                        seen_web_search_ids.add(item_id)
                                        args_preview = _web_search_args_preview(added_item)
                                        tool_log_index[item_id] = len(tool_log)
                                        tool_log.append({
                                            "id": item_id,
                                            "name": "web_search",
                                            "args_preview": args_preview,
                                        })
                                        runs.emit_event(db, run, "tool_call", {
                                            "id": item_id,
                                            "name": "web_search",
                                            "args_preview": args_preview,
                                        })
                                        await asyncio.sleep(0)

                            elif event_type == "response.output_item.done":
                                done_item = getattr(event, "item", None)
                                # Capture every fully-streamed item so we
                                # can replay them into running_input if
                                # the stream dies before completion.
                                if done_item is not None:
                                    collected_round_items.append(done_item)
                                if (
                                    done_item is not None
                                    and getattr(done_item, "type", None) == "web_search_call"
                                ):
                                    item_id = getattr(done_item, "id", None) or ""
                                    if item_id:
                                        status = getattr(done_item, "status", None) or "completed"
                                        idx = tool_log_index.get(item_id)
                                        summary = "done" if status == "completed" else status
                                        final_args = _web_search_args_preview(done_item)
                                        if idx is not None:
                                            tool_log[idx]["summary"] = summary
                                            if final_args:
                                                tool_log[idx]["args_preview"] = final_args
                                        runs.emit_event(db, run, "tool_result", {
                                            "id": item_id,
                                            "name": "web_search",
                                            "summary": summary,
                                            "cost": 0,
                                            "args_preview": final_args or None,
                                        })
                                        # Web_search is a built-in tool that
                                        # fires INSIDE a single OpenAI response,
                                        # mid-text-stream. If the model already
                                        # produced text in this round, the next
                                        # text delta needs a paragraph break or
                                        # we get "...accounts.I found..." mush.
                                        if got_output_this_round:
                                            text_resume_needs_separator = True
                                        await asyncio.sleep(0)

                            elif event_type == "response.incomplete":
                                # OpenAI signaled the response will end
                                # without `response.completed`. Capture
                                # the reason (max_output_tokens,
                                # content_filter, …) for diagnostics.
                                # The actual "treat as stopped + persist
                                # resume" flow lives in the post-stream
                                # `if final_response is None:` block —
                                # this handler just records the cause.
                                resp = getattr(event, "response", None)
                                details = (
                                    getattr(resp, "incomplete_details", None)
                                    if resp is not None
                                    else None
                                )
                                reason = None
                                if details is not None:
                                    reason = (
                                        getattr(details, "reason", None)
                                        or (
                                            details.get("reason")
                                            if isinstance(details, dict)
                                            else None
                                        )
                                    )
                                incomplete_reason = reason or "unknown"
                                log.warning(
                                    "run %s round %s: response.incomplete (%s)",
                                    run_id,
                                    round_num,
                                    incomplete_reason,
                                )

                            elif event_type and event_type.startswith("response.web_search_call."):
                                progress_item = getattr(event, "item", None)
                                progress_id = (
                                    getattr(event, "item_id", None)
                                    or (getattr(progress_item, "id", None) if progress_item else None)
                                    or ""
                                )
                                if progress_id and progress_id in tool_log_index:
                                    idx = tool_log_index[progress_id]
                                    existing_args = tool_log[idx].get("args_preview", "") or ""
                                    new_args = (
                                        _web_search_args_preview(progress_item)
                                        if progress_item is not None
                                        else ""
                                    )
                                    if new_args and new_args != existing_args:
                                        tool_log[idx]["args_preview"] = new_args
                                        runs.emit_event(db, run, "tool_call_update", {
                                            "id": progress_id,
                                            "args_preview": new_args,
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
                                        runs.publish_token_delta(run.id, sep)
                                elif text_resume_needs_separator and not full_content.endswith("\n\n"):
                                    # Mid-round text resuming after a built-in
                                    # tool (web_search) fired. Insert a break
                                    # so "...accounts.I found..." renders as
                                    # two paragraphs.
                                    sep = "\n\n"
                                    full_content += sep
                                    runs.publish_token_delta(run.id, sep)
                                    text_resume_needs_separator = False

                                token = event.delta or ""
                                if token:
                                    round_text_collected += token
                                    clean = stripper.feed(token)
                                    if clean:
                                        full_content += clean
                                        runs.publish_token_delta(run.id, clean)
                                        await asyncio.sleep(0)
                                # Once we've emitted a real text token, clear
                                # the resume-separator flag — already handled
                                # above, but keep this idempotent.
                                if token:
                                    text_resume_needs_separator = False

                        try:
                            final_response = await stream.get_final_response()
                        except RuntimeError as _rt:
                            if "response.completed" in str(_rt):
                                final_response = None
                            else:
                                raise

                    if final_response is not None:
                        break
                    if round_text_collected:
                        log.warning(
                            f"run {run_id} round {round_num}: stream ended without "
                            f"completion event after {len(round_text_collected)} "
                            f"chars emitted — treating as incomplete"
                        )
                        break
                    if attempt >= MAX_STREAM_RETRIES:
                        log.warning(
                            f"run {run_id} round {round_num}: stream ended without "
                            f"completion event after {attempt + 1} attempt(s) — giving up"
                        )
                        break
                    log.warning(
                        f"run {run_id} round {round_num} attempt {attempt + 1}: "
                        f"stream ended without completion event "
                        f"(no text emitted yet), retrying"
                    )

                if final_response is None:
                    stopped = True
                    if not got_output_this_round:
                        thinking_total += time.time() - round_thinking_start

                    # Replay every fully-streamed item from the dead
                    # round into running_input so the next user message
                    # can resume from this exact state. Any function_call
                    # that completed but never executed gets a stub
                    # output — OpenAI's responses API rejects input
                    # where a function_call has no matching
                    # function_call_output, so without the stub the
                    # resume request would 400.
                    pending_call_ids: List[str] = []
                    for out_item in collected_round_items:
                        item_type = getattr(out_item, "type", None)
                        if item_type == "reasoning":
                            r_summary = []
                            if getattr(out_item, "summary", None):
                                r_summary = [
                                    {"type": s.type, "text": s.text}
                                    for s in out_item.summary
                                ]
                            running_input.append({
                                "type": "reasoning",
                                "id": out_item.id,
                                "summary": r_summary,
                            })
                        else:
                            running_input.append(out_item.model_dump(exclude_none=True))
                            if item_type == "function_call":
                                cid = getattr(out_item, "call_id", None)
                                if cid:
                                    pending_call_ids.append(cid)
                    for cid in pending_call_ids:
                        running_input.append({
                            "type": "function_call_output",
                            "call_id": cid,
                            "output": "Error: round terminated before tool execution",
                        })

                    # Snapshot for resume on next user message.
                    resume_input_snapshot = list(running_input)
                    resume_reason = incomplete_reason or "stream_ended_without_completion"

                    # Flush the round's reasoning buffer to DB before we
                    # exit — otherwise the thinking text accumulated
                    # during the dead round is lost from diagnose_run's
                    # transcript. (Mirrors the success-path call below.)
                    runs.emit_thinking_checkpoint(db, run, round_num=round_num)

                    # Replace any partial half-thought streamed before
                    # the cap hit with a clean warning. The user sees a
                    # friendly message; the actual reasoning + searches
                    # they ran are preserved in resume_input_snapshot.
                    # No "error" event — that's reserved for FAILED runs
                    # (the FE shows it as a toast/banner). This run isn't
                    # failed; it has a valid assistant message and the
                    # next user message will resume cleanly.
                    stripper.flush()  # drain stripper, discard tail
                    full_content = _INCOMPLETE_WARNING
                    runs.replace_text_content(db, run, full_content)
                    break

                remaining = stripper.flush()
                if remaining:
                    full_content += remaining
                    runs.publish_token_delta(run.id, remaining)

                # Round complete — persist durable checkpoints. Token
                # deltas were live-only; reasoning summary deltas were
                # too. Both get persisted here so reconnects + offline
                # diagnosis can read the full transcript.
                runs.emit_text_checkpoint(db, run)
                runs.emit_thinking_checkpoint(db, run, round_num=round_num)

                round_cost = _response_cost_usd(final_response)
                total_cost += round_cost

                web_search_count = sum(
                    1 for item in final_response.output
                    if item.type == "web_search_call"
                )
                if web_search_count:
                    web_cost = web_search_count * _sources.WEB_SEARCH_USD_BY_TIER["low"]
                    total_cost += web_cost
                    round_cost += web_cost

                # Bill the round incrementally so the FE credit display
                # ticks down live instead of jumping at end-of-turn.
                billing.add(round_cost)

                if not got_output_this_round:
                    thinking_total += time.time() - round_thinking_start

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

                runs.emit_event(db, run, "tool_start", {
                    "tools": [item.name for item in tool_calls],
                })
                await asyncio.sleep(0)

                tool_result_items: List[Dict[str, Any]] = []
                round_applied: Dict[str, Any] = {}

                for item in tool_calls:
                    signal = runs.check_should_stop(db, run)
                    if signal:
                        stopped = True
                        stop_reason = signal
                        break
                    if billing.out_of_credits:
                        stopped = True
                        stop_reason = "out_of_credits"
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
                    runs.emit_event(db, run, "tool_call", {
                        "id": item.call_id,
                        "name": item.name,
                        "args_preview": args_preview,
                        # Full input the model passed to the tool — used
                        # for offline diagnosis. Not rendered by the FE
                        # (which uses args_preview).
                        "args_full": args,
                    })
                    await asyncio.sleep(0)
                    runs.update_run_phase(db, run, f"tool: {item.name}")

                    progress_q: asyncio.Queue = asyncio.Queue()
                    # Track cell costs we've billed incrementally via
                    # `cell_done` progress events. When the tool itself
                    # finally returns, its tool_cost includes these
                    # cells — subtract what we already billed so we
                    # don't double-charge. Critical because if the tool
                    # CRASHES mid-flight (e.g. rows_fill on row 91 of
                    # 206), the orchestrator never gets tool_cost back
                    # but the cells that DID complete still cost us
                    # real money. Without this incremental path those
                    # cells go unbilled — confirmed gap on ee0db354
                    # which under-billed by ~$9.
                    cell_cost_billed = 0.0

                    async def _on_progress(ev: Dict[str, Any]) -> None:
                        ev = dict(ev)
                        ev.setdefault("tool_call_id", item.call_id)
                        await progress_q.put(ev)
                        await asyncio.sleep(0)

                    tool_task = asyncio.create_task(
                        agent.execute_tool(
                            db, project, item.name, args,
                            progress_cb=_on_progress,
                            effort=effective_effort,
                        )
                    )

                    while True:
                        getter = asyncio.create_task(progress_q.get())
                        done_set, _pending = await asyncio.wait(
                            {getter, tool_task},
                            return_when=asyncio.FIRST_COMPLETED,
                        )
                        if getter in done_set:
                            ev = getter.result()
                            ev_type = ev.pop("type", "progress")
                            # Bill cell agents incrementally as they
                            # complete — see cell_cost_billed comment
                            # above. The dedupe at tool-end strips this
                            # from the tool_cost so no double-charge.
                            if ev_type == "cell_done":
                                c = ev.get("cost") or 0
                                if isinstance(c, (int, float)) and c > 0:
                                    billing.add(float(c))
                                    cell_cost_billed += float(c)
                            runs.emit_event(db, run, ev_type, ev)
                            await asyncio.sleep(0)
                        else:
                            getter.cancel()
                        if tool_task in done_set:
                            while not progress_q.empty():
                                ev = progress_q.get_nowait()
                                ev_type = ev.pop("type", "progress")
                                if ev_type == "cell_done":
                                    c = ev.get("cost") or 0
                                    if isinstance(c, (int, float)) and c > 0:
                                        billing.add(float(c))
                                        cell_cost_billed += float(c)
                                runs.emit_event(db, run, ev_type, ev)
                            break

                    item_applied, result, tool_cost = await tool_task
                    result_text = agent.format_tool_result(item.name, result)
                    result_text += agent.project_state_hint(db, project)
                    total_cost += tool_cost
                    # Bill the residual: tool_cost includes the cells
                    # we already billed incrementally (cell_cost_billed),
                    # plus any orchestrator overhead inside the tool
                    # (e.g. rows_fill's bookkeeping). Charge only the
                    # difference. Negative deltas (rounding) clamp to 0.
                    residual = max(0.0, tool_cost - cell_cost_billed)
                    billing.add(residual)
                    round_applied.update(item_applied)
                    applied.update(item_applied)

                    # Emit live row count so the FE pagination total
                    # tracks rows_delete / rows_add as they happen.
                    # Without this the FE only reconciles at end-of-turn
                    # and a user mid-stream can land on an invalid page
                    # when rows_delete shrinks the table under them.
                    try:
                        runs.emit_event(db, run, "row_count", {
                            "count": agent.project_row_count(db, project),
                        })
                    except Exception:
                        log.exception("run %s: row_count emit failed", run_id)

                    summary = _summarize_result(item.name, result_text)
                    cost_rounded = round(tool_cost, 4) if tool_cost > 0 else 0
                    idx = tool_log_index.get(item.call_id)
                    if idx is not None:
                        tool_log[idx]["summary"] = summary
                        tool_log[idx]["cost"] = cost_rounded

                    runs.emit_event(db, run, "tool_result", {
                        "id": item.call_id,
                        "name": item.name,
                        "summary": summary,
                        "cost": cost_rounded,
                        # Full result text — exactly what the model saw
                        # back from the tool. Diagnosis fuel; not used
                        # for FE rendering.
                        "result_text": result_text,
                    })
                    await asyncio.sleep(0)

                    tool_result_items.append({
                        "type": "function_call_output",
                        "call_id": item.call_id,
                        "output": result_text,
                    })

                running_input.extend(tool_result_items)

                if stopped:
                    break

                if isinstance(round_applied.get("suggestions"), dict):
                    sg = round_applied["suggestions"]
                    runs.emit_event(db, run, "suggestions", {
                        "items": sg.get("items") or [],
                    })
                if isinstance(round_applied.get("version_label"), dict):
                    vl = round_applied["version_label"]
                    runs.emit_event(db, run, "version", {
                        "version_id": vl.get("version_id"),
                        "version_number": vl.get("version_number"),
                        "label": vl.get("label"),
                    })

                for change in agent.describe_applied(round_applied):
                    if change.field == "questions":
                        runs.emit_event(db, run, "questions", {"questions": change.value})
                    else:
                        event_data: dict = {
                            "field": change.field,
                            "description": change.description,
                        }
                        if change.value is not None:
                            event_data["value"] = change.value
                        runs.emit_event(db, run, "change", event_data)

                if any(
                    item.name in ("ask_questions", "suggest_replies")
                    for item in tool_calls
                ):
                    break

        except Exception as e:
            log.exception("run %s agent loop crashed", run_id)
            err_ac: Dict[str, Any] = {"error": str(e)[:500], "interrupted": True}
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
            # Snapshot running_input as resume_input so the next user
            # message can pick up exactly where we left off — same
            # mechanism cofounder added for response.incomplete. We
            # capture completed function_calls; the in-flight one (if
            # any) is dropped, the user will re-trigger it.
            if running_input:
                err_ac["resume_input"] = list(running_input)
            cleaned_content = _clean_citations(full_content)
            persisted_via_main_session = False
            try:
                _persist_assistant_message(
                    db, run, project, new_version,
                    full_content=cleaned_content,
                    applied_changes=err_ac,
                )
                # Flush whatever incremental charges hadn't crossed the
                # consume_credits floor yet. Most cost was already
                # billed live; this catches the residual.
                billing.flush()
                _commit_with_deadlock_retry(db)
                persisted_via_main_session = True
            except Exception:
                log.exception("run %s: failed to persist error stub", run_id)
                try:
                    db.rollback()
                except Exception:
                    pass
            # Fresh-session fallback: if the main session was poisoned
            # by the originating exception (PendingRollbackError,
            # deadlock, etc.), open a clean session and write a stub
            # message so the user always sees what happened, never a
            # silent void. Bug seen on ee0db354 was exactly this case.
            if not persisted_via_main_session:
                _force_persist_assistant_message(
                    run_id=run.id,
                    project_id=project.id,
                    full_content=cleaned_content,
                    applied_changes=err_ac,
                )
                # Same belt-and-suspenders for billing: if main session
                # was poisoned, billing.flush() above raised and was
                # caught — residual unbilled compute hasn't hit the
                # ledger yet. Force-flush via a fresh session so we
                # don't silently swallow real costs.
                billing.force_flush_with_fresh_session()
            # Never leak raw exception text to the FE — it dumps SQL,
            # frame names, and DB internals (e.g. DeadlockDetected
            # includes the offending UPDATE verbatim). Full detail goes
            # to the worker log + the run.error column for diagnosis.
            user_safe_msg = "Something went wrong on our end. Please try again."
            runs.emit_event(db, run, "error", {"message": user_safe_msg})
            from sqlalchemy.sql import func
            run.status = runs.RUN_STATUS_FAILED
            run.error = str(e)[:500]
            run.completed_at = func.now()
            runs.emit_event(db, run, "done", {"stopped": True, "error": user_safe_msg})
            return

        # Forced text wrap-up if we exited the loop with no text. Run
        # this even when stopped=True (incomplete OpenAI stream / pause
        # / cancel) — otherwise we ship an empty assistant message
        # after a real turn's work, which is the worst possible UX.
        # Cause that bit us 2026-04-29 on project 7941c11b: 87 events,
        # $0.92 spent, 1/60 rows committed, empty body because the
        # final round's stream cut off without response.completed and
        # we skipped the wrap-up.
        if (
            len(tool_log) > 0
            and not full_content.strip()
        ):
            try:
                wrap_input = list(running_input) + [{
                    "role": "user",
                    "content": (
                        "(System note — not from the user.) You ran tools "
                        "but produced no text reply. Look at the project "
                        "state in the prior context and write ONE short "
                        "reply summarizing what you did this turn AND, if "
                        "rows did not actually land in the table, finish "
                        "the job: add columns + commit candidates_to_rows "
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
                wrap_cost = _response_cost_usd(wrap)
                total_cost += wrap_cost
                billing.add(wrap_cost)
                for item in wrap.output:
                    if item.type == "message":
                        for block in item.content:
                            text = getattr(block, "text", None)
                            if text:
                                full_content += text
                                runs.publish_token_delta(run.id, text)
                # Persist the wrap-up text into the durable checkpoint.
                runs.emit_text_checkpoint(db, run)
            except Exception:
                log.exception("run %s: forced-text wrap-up failed", run_id)

        if is_first_message and not stopped:
            name_cost = await _auto_name_project(client, project, user_content)
            total_cost += name_cost
            billing.add(name_cost)
            runs.emit_event(db, run, "project_name", {"name": project.name})

        full_content = _clean_citations(full_content)

        ac_data: Dict[str, Any] = {}
        if applied:
            ac_data["changes"] = applied
        if thinking_total >= 0.5:
            ac_data["thinking_duration"] = round(thinking_total, 1)
        if stopped:
            ac_data["stopped"] = True
            if stop_reason:
                ac_data["stop_reason"] = stop_reason
        if tool_log:
            ac_data["tool_log"] = tool_log
        if total_cost > 0:
            ac_data["total_cost_usd"] = round(total_cost, 4)
        if sources:
            ac_data["sources"] = sources
        if resume_input_snapshot is not None:
            # Replayed verbatim into the next user message's
            # `running_input` so the model resumes with full context
            # (reasoning chain, tool calls, search results) instead of
            # restarting research from scratch.
            ac_data["resume_input"] = resume_input_snapshot
            if resume_reason:
                ac_data["resume_reason"] = resume_reason

        assistant_msg = _persist_assistant_message(
            db, run, project, new_version,
            full_content=full_content,
            applied_changes=ac_data if ac_data else None,
        )
        # Final flush — most cost was billed incrementally during the
        # run; this catches any residual under the per-call floor.
        billing.flush()
        _commit_with_deadlock_retry(db)
        db.refresh(assistant_msg)

        done_payload: Dict[str, Any] = {
            "message_id": str(assistant_msg.id),
            "total_cost_usd": round(total_cost, 4),
        }
        if thinking_total >= 0.5:
            done_payload["thinking_duration"] = round(thinking_total, 1)
        if stopped:
            done_payload["stopped"] = True
            if stop_reason:
                done_payload["stop_reason"] = stop_reason
        if resume_input_snapshot is not None:
            # Surface "stream died, resume available on next message"
            # so the FE can show a subtle hint if it wants. Distinct
            # from `stop_reason` (pause/cancel) — those are user-driven.
            done_payload["incomplete"] = True
            if resume_reason:
                done_payload["incomplete_reason"] = resume_reason
        if sources:
            done_payload["sources"] = sources

        if stopped and stop_reason == "pause":
            runs.mark_run_paused(db, run, done_payload)
        elif stopped and stop_reason == "cancel":
            runs.mark_run_cancelled(db, run, done_payload)
        else:
            runs.mark_run_completed(db, run, done_payload)

    finally:
        db.close()


def _persist_assistant_message(
    db: Session,
    run: ChatRun,
    project: Project,
    new_version,
    *,
    full_content: str,
    applied_changes: Optional[Dict[str, Any]],
) -> ChatMessage:
    """Create-or-update the assistant ChatMessage attached to this
    run. Re-entrant: if the run already has an assistant_message_id,
    we update that row in place (used by the error path)."""
    if run.assistant_message_id is not None:
        msg = (
            db.query(ChatMessage)
            .filter(ChatMessage.id == run.assistant_message_id)
            .first()
        )
        if msg is not None:
            msg.content = full_content
            msg.applied_changes = applied_changes
            return msg
    msg = ChatMessage(
        project_id=project.id,
        role="assistant",
        content=full_content,
        applied_changes=applied_changes,
        version_id=new_version.id,
        run_id=run.id,
    )
    db.add(msg)
    db.flush()
    run.assistant_message_id = msg.id
    return msg


def _force_persist_assistant_message(
    run_id: UUID,
    project_id: UUID,
    *,
    full_content: str,
    applied_changes: Optional[Dict[str, Any]],
) -> bool:
    """Belt-and-suspenders persist using a FRESH SessionLocal.

    Called only from the exception path when the main session is
    poisoned (PendingRollbackError, deadlock, etc.) and the normal
    `_persist_assistant_message` failed. Opens a clean session,
    creates a minimal ChatMessage row, and links it to the run via
    `assistant_message_id`. This is the last line of defense against
    a run going terminal with NO visible message in chat history —
    the failure mode that bit project ee0db354.

    Returns True if persisted, False if even the fresh session
    couldn't write (logged but swallowed — caller continues).
    """
    fresh_db = SessionLocal()
    try:
        run = fresh_db.query(ChatRun).filter(ChatRun.id == run_id).first()
        if run is None:
            log.warning("force_persist: run %s vanished", run_id)
            return False
        # Don't double-write if the main session somehow already did.
        if run.assistant_message_id is not None:
            return True
        version_id = run.version_id  # already committed at fork time
        msg = ChatMessage(
            project_id=project_id,
            role="assistant",
            content=full_content,
            applied_changes=applied_changes,
            version_id=version_id,
            run_id=run.id,
        )
        fresh_db.add(msg)
        fresh_db.flush()
        run.assistant_message_id = msg.id
        fresh_db.commit()
        log.warning(
            "force_persist: wrote stub assistant message %s for run %s "
            "(main session was poisoned)", msg.id, run_id,
        )
        return True
    except Exception:
        log.exception("force_persist: even fresh session failed for run %s", run_id)
        try:
            fresh_db.rollback()
        except Exception:
            pass
        return False
    finally:
        fresh_db.close()


# ---- Legacy /chat/stream wrapper -----------------------------------------
async def stream_chat_response(
    project_id: UUID,
    user_id: UUID,
    user_content: str,
    request: Request,
    effort: Optional[str] = None,
) -> AsyncGenerator[str, None]:
    """Wire-compatible legacy wrapper around the run-based pipeline.

    Creates a ChatRun, tails its events, yields SSE. If the HTTP client
    disconnects (AbortController.abort), we request a pause on the run
    so the prior FE behavior is preserved during the rollover. The new
    explicit endpoints (POST /chat/runs + GET /chat/runs/{id}/events)
    do NOT pause-on-disconnect; they let the run keep running in the
    background and rely on /pause to stop it.
    """
    try:
        run = await runs.start_run(
            project_id=project_id,
            user_id=user_id,
            user_content=user_content,
            effort=effort,
        )
    except ValueError as e:
        yield _sse({"type": "error", "message": str(e)})
        return
    except Exception as e:
        log.exception("legacy stream: start_run failed")
        yield _sse({"type": "error", "message": f"AI service error: {e}"})
        return

    run_id = run.id
    disconnected = False
    try:
        async for event in runs.tail_events(
            run_id,
            cursor=0,
            is_disconnected=request.is_disconnected,
        ):
            yield _sse(event)
    finally:
        # If the client dropped, fall back to legacy behavior: pause.
        if await request.is_disconnected():
            disconnected = True
            try:
                runs.request_pause(run_id)
            except Exception:
                log.exception("legacy stream: request_pause failed for %s", run_id)
        if disconnected:
            log.info("legacy stream %s: client disconnected, paused run", run_id)


