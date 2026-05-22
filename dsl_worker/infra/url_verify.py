"""Per-URL verification via firecrawl + LLM tool-use.

Verifies whether each scraped URL points to a page that matches its
row's expected entity. Mirrors email_verify hard-bounce semantics —
when a URL fails verification the cell is cleared and the URL is
kept in ``tags.failed_urls[col]`` so the FE can show it in cell
details (same shape as ``failed_emails`` for Scrubby Invalid).

Pipeline per URL:
  1. Firecrawl scrape (markdown + title). BROKEN / HTTP-4xx pages
     bypass the LLM entirely and go straight to INVALID — a 4xx
     page is a bad URL regardless of content.
  2. LLM tool-use loop: the model calls ``find(keyword)`` to probe
     the scraped markdown for distinguishing terms drawn from the
     row's entity context. If any find() hits the model calls
     ``decide("VALID")``; if it can't find anything relevant it
     calls ``decide("INVALID")``. Empty / login-walled pages →
     ``decide("UNVERIFIED")``.
  3. Persist: stamp ``tags.url_verification[col]`` for the FE
     badge. On INVALID, also clear ``sample.row[col]``, append the
     URL to ``tags.failed_urls[col]``, and write a ``fill_status``
     entry.

Why tool-use instead of feed-the-markdown: firecrawl markdown is
noisy (nav, footer, ad text). An LLM-as-judge over the whole page
makes unreliable yes/no calls because it gets distracted by junk.
``find()`` reduces the model's job to picking 2–4 distinguishing
keywords — a much cleaner binary signal than a holistic content
judgement, with much more predictable token cost.

Best-effort: every failure path swallows and marks UNVERIFIED so
the FE shows no badge. The verification task NEVER raises.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from collections import defaultdict
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple

import httpx
from openai import AsyncOpenAI

from dsl_api.db import SessionLocal
from dsl_api.models.sample import Sample


log = logging.getLogger(__name__)

ProgressCallback = Callable[[Dict[str, Any]], Awaitable[None]]


# Firecrawl REST. We hit /v1/scrape directly with httpx — the
# firecrawl-py SDK adds a dep for one HTTP call, not worth it.
_FIRECRAWL_URL = "https://api.firecrawl.dev/v1/scrape"
# 60s per attempt, one retry on timeout / transient error. Total
# wall-clock cap per URL is ~2 min; if both attempts time out we
# bail to UNVERIFIED ("couldn't verify") rather than retrying
# forever. Firecrawl's own renderer ceiling is ~30s; the extra
# headroom is for queue + slow JS.
_FIRECRAWL_TIMEOUT = 60.0
_FIRECRAWL_MAX_ATTEMPTS = 2
_FIRECRAWL_CONCURRENCY = 5

# Per-URL LLM loop budget: enough find()s to give the model room to
# try a few keywords plus one decide(). Beyond this we bail UNVERIFIED.
_JUDGE_MAX_TURNS = 5

# Cap the markdown we hold in memory for find(). firecrawl already
# strips boilerplate via onlyMainContent so 12K chars is plenty.
_MARKDOWN_CAP = 12_000

_VERIFY_MODEL = os.environ.get("URL_VERIFY_MODEL", "gpt-5.4-mini")

_OPENAI: Optional[AsyncOpenAI] = None


def _get_openai() -> Optional[AsyncOpenAI]:
    """Direct OpenAI client for the judge call.

    Bypasses TrackedOpenAIClient — verification is background work
    that shouldn't appear in the run's usage breakdown.
    """
    global _OPENAI
    if _OPENAI is not None:
        return _OPENAI
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        return None
    _OPENAI = AsyncOpenAI(api_key=key)
    return _OPENAI


def _firecrawl_key() -> Optional[str]:
    return os.environ.get("FIRECRAWL_API_KEY") or None


# ---------------------------------------------------------------------------
# Firecrawl scrape
# ---------------------------------------------------------------------------


async def _scrape_once(
    client: httpx.AsyncClient,
    url: str,
    api_key: str,
) -> Dict[str, Any]:
    """ONE firecrawl attempt, no retry. Caller wraps with the retry loop."""
    try:
        resp = await client.post(
            _FIRECRAWL_URL,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "url": url,
                "formats": ["markdown"],
                "onlyMainContent": True,
                "timeout": int(_FIRECRAWL_TIMEOUT * 1000),
            },
            timeout=_FIRECRAWL_TIMEOUT + 5.0,
        )
    except httpx.TimeoutException:
        # Transient — retryable.
        return {"url": url, "status": "TIMEOUT", "error": "fetch_timeout", "code": 0}
    except httpx.RequestError as e:
        # Transient network error — retryable.
        return {"url": url, "status": "TIMEOUT", "error": f"fetch_{type(e).__name__}", "code": 0}

    if resp.status_code >= 500:
        # Firecrawl side error — retryable.
        return {"url": url, "status": "TIMEOUT", "error": f"firecrawl_http_{resp.status_code}", "code": resp.status_code}
    if resp.status_code >= 400:
        # 4xx — terminal, this URL is bad regardless of retry.
        return {"url": url, "status": "BROKEN", "error": f"firecrawl_http_{resp.status_code}", "code": resp.status_code}

    try:
        body = resp.json()
    except Exception:
        return {"url": url, "status": "BROKEN", "error": "firecrawl_bad_json", "code": resp.status_code}

    if not body.get("success"):
        err = (body.get("error") or "").strip()[:200] or "firecrawl_failed"
        md = (body.get("data") or {}).get("metadata") or {}
        code = int(md.get("statusCode") or 0)
        status = "BROKEN" if code >= 400 else "EMPTY"
        return {"url": url, "status": status, "error": err, "code": code}

    data = body.get("data") or {}
    metadata = data.get("metadata") or {}
    code = int(metadata.get("statusCode") or 200)
    if code >= 400:
        return {"url": url, "status": "BROKEN", "error": f"page_http_{code}", "code": code}

    markdown = (data.get("markdown") or "").strip()
    title = (metadata.get("title") or "").strip()
    if not markdown:
        return {"url": url, "status": "EMPTY", "error": "no_markdown", "code": code, "title": title[:300]}
    return {
        "url": url,
        "status": None,
        "markdown": markdown[:_MARKDOWN_CAP],
        "title": title[:300],
        "code": code,
    }


async def _scrape_one(
    client: httpx.AsyncClient,
    url: str,
    api_key: str,
    sem: asyncio.Semaphore,
) -> Dict[str, Any]:
    """Return ``{url, status, markdown, title, code, error}``.

    ``status`` is one of:
      • ``"BROKEN"``  — page returned HTTP 4xx or firecrawl rejected it.
      • ``"EMPTY"``   — scrape succeeded but no readable content.
      • ``"TIMEOUT"`` — both attempts timed out / transient errors —
        verifier maps this to ``UNVERIFIED`` ("couldn't verify").
      • ``None``      — scrape succeeded; LLM will decide.

    Retries once on timeout / 5xx / network error before bailing.
    Never raises — failure modes are encoded in the returned dict.
    """
    async with sem:
        last: Dict[str, Any] = {}
        for attempt in range(_FIRECRAWL_MAX_ATTEMPTS):
            last = await _scrape_once(client, url, api_key)
            if last.get("status") != "TIMEOUT":
                return last
            if attempt + 1 < _FIRECRAWL_MAX_ATTEMPTS:
                log.info(
                    "url_verify: firecrawl retry %d/%d for %s (err=%s)",
                    attempt + 2, _FIRECRAWL_MAX_ATTEMPTS, url, last.get("error"),
                )
        return last


# ---------------------------------------------------------------------------
# LLM judge — find() / decide() tool-use loop
# ---------------------------------------------------------------------------


def _entity_context(row: Dict[str, Any], url_columns: List[str]) -> str:
    """Compact "what is this row about" hint for the LLM judge.

    Skips URL columns themselves and underscore-prefixed metadata.
    """
    if not isinstance(row, dict):
        return ""
    skip = set(url_columns)
    parts: List[str] = []
    for k, v in row.items():
        if not isinstance(k, str) or k.startswith("_"):
            continue
        if k in skip:
            continue
        if v in (None, ""):
            continue
        if isinstance(v, (str, int, float)):
            parts.append(f"{k}: {v}")
        if len(parts) >= 12:
            break
    return "; ".join(parts)[:800]


# Caps on excerpt window. Hard-bounds the per-call token cost: 3
# excerpts × 300 chars × 2 sides ≈ 1.8K chars max per find_context.
_MAX_EXCERPTS = 3
_MAX_CHARS_AROUND = 300


_TOOLS_FOR_JUDGE: List[Dict[str, Any]] = [
    {
        "type": "function",
        "name": "find",
        "description": (
            "Case-insensitive substring search of the scraped page text. "
            "Returns {found: bool, count: int}. Cheap probe — use this "
            "first. Try distinguishing keywords from the expected entity "
            "(entity name, domain, founder, industry terms). Plain text "
            "only, no regex."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "keyword": {
                    "type": "string",
                    "description": "Word or short phrase to search for.",
                },
            },
            "required": ["keyword"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "find_context",
        "description": (
            "Like find(), but returns up to 3 excerpts showing the keyword "
            "with surrounding text. Use this AFTER a find() hit to check "
            "whether the match is actually about the expected entity or "
            "just one of many items on a directory/listing page. If the "
            "excerpts show the keyword among a list of unrelated items "
            "(\"Acme Corp · BetaCo · GammaInc · ...\"), the page is "
            "probably a directory — not the entity's detail page. If the "
            "excerpts read like a profile/about/product page focused on "
            "the entity, it's likely the right page. Returns {found, "
            "count, excerpts: [string, ...]}."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "keyword": {
                    "type": "string",
                    "description": "Word or short phrase to search for.",
                },
                "chars_around": {
                    "type": "integer",
                    "description": (
                        "Characters of context on each side of the match. "
                        "Defaults to 150; capped at 300."
                    ),
                },
            },
            "required": ["keyword"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "decide",
        "description": (
            "Set the final verdict. Call exactly once when you've made "
            "your decision. After this call the loop ends."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "status": {
                    "type": "string",
                    "enum": ["VALID", "INVALID", "UNVERIFIED"],
                    "description": (
                        "VALID = the page is clearly about the expected "
                        "entity. INVALID = page was readable but is "
                        "about something else OR is a directory/listing "
                        "rather than the specific detail page. "
                        "UNVERIFIED = page was empty / login-walled / "
                        "unreadable so a real check wasn't possible."
                    ),
                },
                "reason": {
                    "type": "string",
                    "description": "Short justification, under 100 chars.",
                },
            },
            "required": ["status", "reason"],
            "additionalProperties": False,
        },
    },
]


_JUDGE_SYSTEM = (
    "You verify that a scraped webpage matches the entity described in "
    "a table row. Workflow:\n"
    "  1. find(keyword) — cheap boolean probe. Try the entity's "
    "distinguishing name first.\n"
    "  2. find_context(keyword) — once find() hits, always confirm "
    "with this. Read the excerpts: do they describe THIS entity "
    "(profile, about page, product detail) or do they show the entity "
    "as one item in a list of many similar items (\"X · Y · Z · ...\", "
    "\"Browse companies: X, Y, Z\", \"Showing 1-20 of 500\")? The "
    "second case means the URL points to a DIRECTORY or CATEGORY page, "
    "not the entity's detail page → INVALID.\n"
    "  3. decide(status, reason) — exactly one call, ends the loop. "
    "VALID = the page is clearly the entity's own page. INVALID = "
    "readable but wrong entity OR a directory/category page rather "
    "than a detail page. UNVERIFIED = page was unreadable (empty / "
    "login wall / blocked).\n"
    "Heuristic on count: a SPECIFIC entity name appearing only 1-2 "
    "times on a long page is often just a listing entry. Appearing "
    "3+ times in focused excerpts (title, headers, body copy about "
    "the entity) is usually the real detail page."
)


def _find_excerpts(
    markdown: str,
    md_lower: str,
    keyword: str,
    chars_around: int,
) -> List[str]:
    """Return up to ``_MAX_EXCERPTS`` snippets showing ``keyword`` + its
    surroundings. Used by ``find_context`` so the LLM can disambiguate
    "the page IS about this entity" from "this entity is one of many
    items on a directory page." Whitespace is collapsed inside each
    excerpt so the model isn't billed for raw markdown indentation.
    """
    kw_lower = keyword.lower()
    if not kw_lower:
        return []
    out: List[str] = []
    start = 0
    while len(out) < _MAX_EXCERPTS:
        idx = md_lower.find(kw_lower, start)
        if idx < 0:
            break
        lo = max(0, idx - chars_around)
        hi = min(len(markdown), idx + len(keyword) + chars_around)
        snippet = markdown[lo:hi]
        # Collapse runs of whitespace so we don't burn tokens on
        # markdown indentation. Ellipses mark trimmed edges so the
        # model knows the excerpt isn't sentence-start / sentence-end.
        snippet = " ".join(snippet.split())
        if lo > 0:
            snippet = "…" + snippet
        if hi < len(markdown):
            snippet = snippet + "…"
        out.append(snippet)
        start = idx + len(keyword)
    return out


async def _judge_one(
    client: AsyncOpenAI,
    url: str,
    title: str,
    markdown: str,
    expected_entity: str,
    column: str,
) -> Tuple[str, str]:
    """Run the find()/decide() loop for ONE URL. Returns (status, reason).

    The full markdown stays out of the model's context — it only
    sees a short preview (so it can spot login walls) and probes
    the rest via find() calls. Keeps cost predictable and stops the
    model from getting distracted by nav/footer noise.
    """
    md_lower = markdown.lower()
    preview = markdown[:300].replace("\n", " ")

    user_msg = (
        f"URL: {url}\n"
        f"Column: {column}\n"
        f"Expected entity: {expected_entity or '(none)'}\n"
        f"Page title: {title or '(none)'}\n"
        f"Content length: {len(markdown)} chars\n"
        f"Content preview: {preview}\n\n"
        "Verify with find() then call decide()."
    )

    input_items: List[Dict[str, Any]] = [
        {"role": "system", "content": _JUDGE_SYSTEM},
        {"role": "user", "content": user_msg},
    ]

    for _turn in range(_JUDGE_MAX_TURNS):
        try:
            resp = await client.responses.create(
                model=_VERIFY_MODEL,
                input=input_items,
                tools=_TOOLS_FOR_JUDGE,
            )
        except Exception:
            log.exception("url_verify: %s call raised for %s", _VERIFY_MODEL, url)
            return "UNVERIFIED", "llm_error"

        tool_calls: List[Any] = []
        for item in resp.output:
            input_items.append(item.model_dump(exclude_none=True))
            if getattr(item, "type", None) == "function_call":
                tool_calls.append(item)

        if not tool_calls:
            return "UNVERIFIED", "no_tool_call"

        for tc in tool_calls:
            try:
                args = json.loads(tc.arguments or "{}")
            except Exception:
                args = {}

            if tc.name == "find":
                keyword = (args.get("keyword") or "").strip()
                if not keyword:
                    input_items.append({
                        "type": "function_call_output",
                        "call_id": tc.call_id,
                        "output": json.dumps({"found": False, "count": 0}),
                    })
                    continue
                kw_lower = keyword.lower()
                count = md_lower.count(kw_lower)
                input_items.append({
                    "type": "function_call_output",
                    "call_id": tc.call_id,
                    "output": json.dumps({"found": count > 0, "count": count}),
                })
            elif tc.name == "find_context":
                keyword = (args.get("keyword") or "").strip()
                try:
                    chars_around = int(args.get("chars_around", 150))
                except (TypeError, ValueError):
                    chars_around = 150
                chars_around = max(20, min(chars_around, _MAX_CHARS_AROUND))
                if not keyword:
                    input_items.append({
                        "type": "function_call_output",
                        "call_id": tc.call_id,
                        "output": json.dumps({"found": False, "count": 0, "excerpts": []}),
                    })
                    continue
                excerpts = _find_excerpts(markdown, md_lower, keyword, chars_around)
                count = md_lower.count(keyword.lower())
                input_items.append({
                    "type": "function_call_output",
                    "call_id": tc.call_id,
                    "output": json.dumps({
                        "found": count > 0,
                        "count": count,
                        "excerpts": excerpts,
                    }),
                })
            elif tc.name == "decide":
                status = args.get("status")
                reason = (args.get("reason") or "")[:200]
                if status in {"VALID", "INVALID", "UNVERIFIED"}:
                    return status, reason
                return "UNVERIFIED", "bad_decide"
            else:
                input_items.append({
                    "type": "function_call_output",
                    "call_id": tc.call_id,
                    "output": json.dumps({"error": f"unknown tool: {tc.name}"}),
                })

    return "UNVERIFIED", "turn_limit"


# ---------------------------------------------------------------------------
# Per-URL verify + per-row persist
# ---------------------------------------------------------------------------


async def _verify_one_url(
    *,
    client: httpx.AsyncClient,
    openai_client: AsyncOpenAI,
    api_key: str,
    sem: asyncio.Semaphore,
    column: str,
    url: str,
    row: Dict[str, Any],
    url_columns: List[str],
) -> Tuple[str, str]:
    """Scrape + judge one URL. Returns (status, reason).

    BROKEN scrapes go straight to INVALID (a 4xx page is a bad URL
    whether or not we could read its content). EMPTY / TIMEOUT
    scrapes go to UNVERIFIED — "couldn't verify" — neither penalizing
    the URL nor pretending we checked it. Everything else hands off
    to find()/decide().
    """
    scraped = await _scrape_one(client, url, api_key, sem)
    s = scraped.get("status")
    if s == "BROKEN":
        return "INVALID", scraped.get("error") or "broken"
    if s == "EMPTY":
        return "UNVERIFIED", scraped.get("error") or "empty"
    if s == "TIMEOUT":
        return "UNVERIFIED", scraped.get("error") or "timeout"

    expected = _entity_context(row, url_columns)
    return await _judge_one(
        openai_client,
        url=url,
        title=scraped.get("title", ""),
        markdown=scraped.get("markdown", ""),
        expected_entity=expected,
        column=column,
    )


def _persist_row_verdicts(
    sample_id: str,
    verdicts: List[Tuple[str, str, str]],  # (column, url, status)
) -> Optional[Dict[str, Any]]:
    """Persist all of a row's URL verdicts in ONE tag write.

    Within-row serialization avoids the read-modify-write race on
    sample.tags that per-column writes would otherwise lose. INVALID
    verdicts ALSO clear the cell value and append to ``failed_urls``
    — same hard-bounce shape email_verify uses for Scrubby Invalid.
    """
    if not verdicts:
        return None
    write_db = SessionLocal()
    try:
        sample = write_db.query(Sample).filter(Sample.id == sample_id).first()
        if sample is None:
            return None
        tags = dict(sample.tags or {})
        verifications = dict(tags.get("url_verification") or {})
        failed = dict(tags.get("failed_urls") or {})
        fill_status = dict(tags.get("fill_status") or {})
        row = dict(sample.row or {})
        row_changed = False

        for column, url, status in verdicts:
            verifications[column] = {
                "value": url,
                "status": status,
                "source": "firecrawl",
            }
            if status == "INVALID":
                if row.get(column) == url:
                    row[column] = None
                    row_changed = True
                bucket = list(failed.get(column) or [])
                if url not in bucket:
                    bucket.append(url)
                failed[column] = bucket
                fill_status[column] = {
                    "status": "null_legitimate",
                    "reason": "URL failed verification — page does not match the expected entity.",
                    "cost": 0.0,
                    "strategy": "firecrawl_verify",
                }

        tags["url_verification"] = verifications
        if failed:
            tags["failed_urls"] = failed
        if fill_status:
            tags["fill_status"] = fill_status
        sample.tags = tags
        if row_changed:
            sample.row = row
        write_db.commit()
        write_db.refresh(sample)
        return {
            "_id": str(sample.id),
            "_seq": sample.seq,
            "_tags": sample.tags or {},
            **(sample.row or {}),
        }
    except Exception:
        log.exception("url_verify: persist failed for row %s", sample_id)
        try:
            write_db.rollback()
        except Exception:
            pass
        return None
    finally:
        write_db.close()


# ---------------------------------------------------------------------------
# Batch entry — group by row, verify rows in parallel
# ---------------------------------------------------------------------------


async def verify_batch(
    *,
    rows_by_column: Dict[str, List[Tuple[str, str, Dict[str, Any]]]],
    url_columns: List[str],
    progress_cb: Optional[ProgressCallback],
) -> None:
    """Public entry — verify every URL in the batch.

    Per-URL: each URL gets its own firecrawl + LLM tool-use judge.
    Per-row: all URLs in a single row are verified concurrently,
    then persisted in ONE tag write so concurrent column writes
    don't race on ``sample.tags``. Rows themselves run in parallel.

    Cancellation: this is fire-and-forget background work. If the
    surrounding asyncio loop is cancelled (uvicorn reload, server
    shutdown), we propagate cancellation through ``asyncio.gather``
    so the worker doesn't sit in "Waiting for background tasks to
    complete" for tens of minutes. Partial verdicts already
    persisted stay; in-flight rows go unverified, which is
    indistinguishable from "didn't try."
    """
    try:
        return await _verify_batch_impl(
            rows_by_column=rows_by_column,
            url_columns=url_columns,
            progress_cb=progress_cb,
        )
    except asyncio.CancelledError:
        log.info("url_verify: batch cancelled — partial verdicts retained")
        raise
    except Exception:
        log.exception("url_verify: batch crashed — partial verdicts retained")
        return


async def _verify_batch_impl(
    *,
    rows_by_column: Dict[str, List[Tuple[str, str, Dict[str, Any]]]],
    url_columns: List[str],
    progress_cb: Optional[ProgressCallback],
) -> None:
    if not rows_by_column:
        return
    api_key = _firecrawl_key()
    if not api_key:
        log.info("url_verify: FIRECRAWL_API_KEY unset — skipping batch")
        return
    openai_client = _get_openai()
    if openai_client is None:
        log.info("url_verify: OPENAI_API_KEY unset — skipping batch")
        return

    by_row: Dict[str, List[Tuple[str, str, Dict[str, Any]]]] = defaultdict(list)
    for col, entries in rows_by_column.items():
        for sid, url, row in entries:
            by_row[sid].append((col, url, row))
    if not by_row:
        return

    total_urls = sum(len(items) for items in by_row.values())
    log.info(
        "url_verify: verifying %d URL(s) across %d row(s)",
        total_urls, len(by_row),
    )

    # Up-front url_verifying so all spinners light up before the first
    # firecrawl returns. Mirrors email_verify_bulk's emit-then-wait pattern.
    if progress_cb is not None:
        for sid, items in by_row.items():
            for col, url, _row in items:
                try:
                    await progress_cb({
                        "type": "url_verifying",
                        "row_id": sid,
                        "column": col,
                        "value": url,
                    })
                except Exception:
                    log.exception("url_verify: progress_cb url_verifying raised; suppressed")

    sem = asyncio.Semaphore(_FIRECRAWL_CONCURRENCY)

    async with httpx.AsyncClient() as client:

        async def _do_row(sample_id: str, items: List[Tuple[str, str, Dict[str, Any]]]) -> None:
            try:
                results = await asyncio.gather(
                    *[
                        _verify_one_url(
                            client=client,
                            openai_client=openai_client,
                            api_key=api_key,
                            sem=sem,
                            column=col,
                            url=url,
                            row=row,
                            url_columns=url_columns,
                        )
                        for col, url, row in items
                    ],
                    return_exceptions=True,
                )
            except Exception:
                log.exception("url_verify: gather failed for row %s", sample_id)
                results = [Exception("gather_failed")] * len(items)

            verdicts: List[Tuple[str, str, str]] = []
            for (col, url, _row), res in zip(items, results):
                if isinstance(res, Exception):
                    log.exception(
                        "url_verify: _verify_one_url raised for %s/%s",
                        sample_id, col, exc_info=res,
                    )
                    verdicts.append((col, url, "UNVERIFIED"))
                else:
                    status, _reason = res
                    verdicts.append((col, url, status))

            snapshot = await asyncio.to_thread(_persist_row_verdicts, sample_id, verdicts)

            if progress_cb is None:
                return
            try:
                for col, url, status in verdicts:
                    await progress_cb({
                        "type": "url_verified",
                        "row_id": sample_id,
                        "column": col,
                        "status": status,
                        "value": url if status != "INVALID" else None,
                    })
                if snapshot is not None:
                    await progress_cb({"type": "row_merged", "row": snapshot})
            except Exception:
                log.exception("url_verify: progress_cb raised; suppressed")

        await asyncio.gather(
            *[_do_row(sid, items) for sid, items in by_row.items()],
            return_exceptions=True,
        )

    log.info(
        "url_verify: batch complete (%d URLs across %d rows)",
        total_urls, len(by_row),
    )
