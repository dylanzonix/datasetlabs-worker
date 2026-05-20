"""Sample-based URL verification — mirrors email_verify.py shape.

When a fetch lands hundreds of rows at once we don't try to verify
every URL. The failure mode for URL-bearing sources is bimodal:
either <10% of URLs are broken (rare outliers) or ~100% are broken
(the source pattern is wrong / the scraper returned junk). The
middle ground is rare. Sampling ~5 URLs per (column, batch) tells
us which side of that bimodal we're on with high confidence, at
roughly 1/20th the cost of exhaustive verification.

Pipeline per (run, column):
  1. Collect every (sample_id, url) in the batch for that column.
  2. Random-sample up to N_SAMPLE rows. Firecrawl-scrape each one
     in parallel (markdown + page metadata).
  3. ONE Haiku 4.5 call judges all N sampled URLs against their
     row context (other column values for that row act as the
     "expected entity" hint).
  4. Aggregate: if >= INVALID_THRESHOLD fraction came back WRONG /
     BROKEN → mark every row in the batch as INVALID for that
     column. Otherwise mark every row VALID.
  5. ONE persist pass (single UPDATE per row touching tags JSON)
     and ONE SSE event per column carrying the aggregate verdict
     and the sample stats. Never per-URL events — that's what
     starved the event loop in a9bd552.

Best-effort: every failure path swallows and marks UNCHECKED so the
FE shows no badge instead of breaking the run. The verification
task NEVER raises.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import random
import re
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple

import httpx
from openai import AsyncOpenAI

from dsl_api.db import SessionLocal
from dsl_api.models.sample import Sample


log = logging.getLogger(__name__)

ProgressCallback = Callable[[Dict[str, Any]], Awaitable[None]]

# How many URLs we scrape per (column, batch). The bimodal failure
# mode means 5 is enough to be ~99% confident in the aggregate call;
# raising it just burns firecrawl credits.
N_SAMPLE = 5

# Fraction of the sample that must come back BROKEN/WRONG for the
# whole batch to flip to INVALID. 0.6 is the sweet spot — below that
# we'd flip on a single bad outlier in a 5-sample (1/5 = 0.2), above
# that we'd miss a column that's mostly-but-not-entirely broken.
INVALID_THRESHOLD = 0.6

# Firecrawl REST. We hit /v1/scrape directly with httpx — the
# firecrawl-py SDK adds a dep for one HTTP call, not worth it.
_FIRECRAWL_URL = "https://api.firecrawl.dev/v1/scrape"
_FIRECRAWL_TIMEOUT = 25.0  # per-URL ceiling; firecrawl JS-render can be slow
_FIRECRAWL_CONCURRENCY = 5  # max parallel scrapes per batch

_URL_RE = re.compile(r"^https?://", re.IGNORECASE)

_VERIFY_MODEL = os.environ.get("URL_VERIFY_MODEL", "gpt-5.4-mini")

_OPENAI: Optional[AsyncOpenAI] = None


def _get_openai() -> Optional[AsyncOpenAI]:
    """Direct OpenAI client for the judge call.

    Bypasses TrackedOpenAIClient — verification is background work
    that shouldn't appear in the run's usage breakdown. Per-call cost
    on gpt-5.4-mini at ~2K input tokens is sub-cent so the lost
    tracking isn't material.
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


async def _scrape_one(
    client: httpx.AsyncClient,
    url: str,
    api_key: str,
    sem: asyncio.Semaphore,
) -> Dict[str, Any]:
    """Return a dict with {url, status, markdown, title, code, error}.

    Never raises — all failure modes are encoded in the returned dict.
    """
    async with sem:
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
        except (httpx.TimeoutException, httpx.RequestError) as e:
            return {"url": url, "status": "BROKEN", "error": f"fetch_{type(e).__name__}", "code": 0}

        if resp.status_code >= 400:
            # Firecrawl returns 4xx for unreachable / blocked / DNS-fail URLs.
            # That's strong "URL is bad" signal; mark BROKEN immediately.
            return {"url": url, "status": "BROKEN", "error": f"firecrawl_http_{resp.status_code}", "code": resp.status_code}

        try:
            body = resp.json()
        except Exception:
            return {"url": url, "status": "UNCHECKED", "error": "firecrawl_bad_json", "code": resp.status_code}

        if not body.get("success"):
            err = (body.get("error") or "").strip()[:200] or "firecrawl_failed"
            # If the page returned a real HTTP error, firecrawl puts the
            # downstream status in metadata.statusCode. Surface it so the
            # LLM doesn't get asked to judge a 404 page.
            md = (body.get("data") or {}).get("metadata") or {}
            code = int(md.get("statusCode") or 0)
            status = "BROKEN" if code >= 400 else "UNCHECKED"
            return {"url": url, "status": status, "error": err, "code": code}

        data = body.get("data") or {}
        metadata = data.get("metadata") or {}
        code = int(metadata.get("statusCode") or 200)
        if code >= 400:
            return {"url": url, "status": "BROKEN", "error": f"page_http_{code}", "code": code}

        markdown = (data.get("markdown") or "").strip()
        title = (metadata.get("title") or "").strip()
        return {
            "url": url,
            "status": None,  # decided by LLM
            "markdown": markdown[:2000],  # cap to keep judge cheap
            "title": title[:300],
            "code": code,
        }


# ---------------------------------------------------------------------------
# LLM judge
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
        if len(parts) >= 10:
            break
    return "; ".join(parts)[:600]


_JUDGE_SYSTEM = (
    "You judge whether each scraped URL points to a page that matches "
    "the expected entity for its row.\n\n"
    "Return ONE verdict per item:\n"
    "  VALID     — page content clearly matches the expected entity\n"
    "  WRONG     — page loads but is about a different entity / generic / parked\n"
    "  UNCHECKED — content is too sparse, behind login, or otherwise undecidable\n\n"
    "Output STRICT JSON, no prose:\n"
    '{"results":[{"url":"...","status":"VALID|WRONG|UNCHECKED","reason":"short"}]}'
)


async def _llm_judge(items: List[Dict[str, Any]]) -> Dict[str, Dict[str, str]]:
    """Judge a batch of items in ONE gpt-5.4-mini call via Responses API.

    `items` shape per element: {url, markdown, title, expected_entity, column}.
    Returns {url: {status, reason}}.
    """
    if not items:
        return {}
    client = _get_openai()
    if client is None:
        log.info("url_verify: OPENAI_API_KEY unset — sample marked UNCHECKED")
        return {it["url"]: {"status": "UNCHECKED", "reason": "no_llm"} for it in items}

    payload = {
        "items": [
            {
                "url": it["url"],
                "column": it["column"],
                "expected": (it.get("expected_entity") or "")[:400],
                "page_title": it.get("title") or "",
                "page_content": (it.get("markdown") or "")[:1500],
            }
            for it in items
        ]
    }
    prompt = _JUDGE_SYSTEM + "\n\nInput:\n" + json.dumps(payload, ensure_ascii=False)

    try:
        resp = await client.responses.create(
            model=_VERIFY_MODEL,
            input=prompt,
        )
    except Exception:
        log.exception("url_verify: %s call raised — sample UNCHECKED", _VERIFY_MODEL)
        return {it["url"]: {"status": "UNCHECKED", "reason": "llm_error"} for it in items}

    text_out = (getattr(resp, "output_text", "") or "").strip()
    if text_out.startswith("```"):
        text_out = re.sub(r"^```(?:json)?\s*", "", text_out).rstrip("`").strip()

    try:
        parsed = json.loads(text_out)
    except Exception:
        log.warning("url_verify: LLM JSON parse failed: %s", text_out[:300])
        return {it["url"]: {"status": "UNCHECKED", "reason": "parse_error"} for it in items}

    out: Dict[str, Dict[str, str]] = {}
    for r in parsed.get("results") or []:
        if not isinstance(r, dict):
            continue
        url = r.get("url")
        status = r.get("status")
        reason = (r.get("reason") or "")[:200]
        if isinstance(url, str) and status in {"VALID", "WRONG", "UNCHECKED"}:
            out[url] = {"status": status, "reason": reason}
    for it in items:
        out.setdefault(it["url"], {"status": "UNCHECKED", "reason": "no_verdict"})

    try:
        usage = getattr(resp, "usage", None)
        if usage is not None:
            log.info(
                "url_verify: judged %d url(s) — input=%s output=%s",
                len(items),
                getattr(usage, "input_tokens", 0),
                getattr(usage, "output_tokens", 0),
            )
    except Exception:
        pass
    return out


# ---------------------------------------------------------------------------
# Persist + emit
# ---------------------------------------------------------------------------


def _persist_batch_verdicts(
    verdicts_by_column: Dict[str, Dict[str, Any]],
    rows_by_column: Dict[str, List[Tuple[str, str, Dict[str, Any]]]],
) -> List[Dict[str, Any]]:
    """Merge all column verdicts into each row's tags in ONE pass.

    `verdicts_by_column[col]` = {status, sample_size, sample_invalid}
    `rows_by_column[col]`     = list of (sample_id, url, _row_dict)

    Single SessionLocal, single commit. Touches each affected Sample
    exactly once even if a row appears in multiple columns — avoids
    the concurrent read-modify-write race that would otherwise drop
    one column's verdict.
    """
    # Build per-row column → url map so a single touch of the row can
    # write every column's verdict.
    row_to_cols: Dict[str, Dict[str, str]] = {}
    for col, rows in rows_by_column.items():
        if col not in verdicts_by_column:
            continue
        for sid, url, _row in rows:
            row_to_cols.setdefault(sid, {})[col] = url
    if not row_to_cols:
        return []

    snapshots: List[Dict[str, Any]] = []
    write_db = SessionLocal()
    try:
        samples = (
            write_db.query(Sample)
            .filter(Sample.id.in_(list(row_to_cols.keys())))
            .all()
        )
        for sample in samples:
            sid = str(sample.id)
            cols_for_row = row_to_cols.get(sid) or {}
            if not cols_for_row:
                continue
            tags = dict(sample.tags or {})
            verifications = dict(tags.get("url_verification") or {})
            for col, url in cols_for_row.items():
                v = verdicts_by_column[col]
                verifications[col] = {
                    "value": url,
                    "status": v["status"],
                    "source": "firecrawl_sample",
                    "sample_size": v["sample_size"],
                    "sample_invalid": v["sample_invalid"],
                }
            tags["url_verification"] = verifications
            sample.tags = tags
        write_db.commit()
        for sample in samples:
            write_db.refresh(sample)
            snapshots.append({
                "_id": str(sample.id),
                "_seq": sample.seq,
                "_tags": sample.tags or {},
                **(sample.row or {}),
            })
    except Exception:
        log.exception("url_verify: batch persist failed")
        try:
            write_db.rollback()
        except Exception:
            pass
    finally:
        write_db.close()
    return snapshots


# ---------------------------------------------------------------------------
# Per-column verification — one task per (run, column) in the batch
# ---------------------------------------------------------------------------


async def _decide_column(
    *,
    column: str,
    rows: List[Tuple[str, str, Dict[str, Any]]],  # (sample_id, url, row_dict)
    url_columns: List[str],
    client: httpx.AsyncClient,
    api_key: str,
    progress_cb: Optional[ProgressCallback],
) -> Optional[Dict[str, Any]]:
    """Sample N rows, scrape + judge, return aggregate verdict for the column.

    Returns {status, sample_size, sample_invalid} or None if the column
    can't be decided (empty rows, etc.). Does NOT persist — the caller
    aggregates all column verdicts and persists in a single pass to
    avoid the read-modify-write race on `tags`.
    """
    if not rows:
        return None
    sample_n = min(N_SAMPLE, len(rows))
    sampled = random.sample(rows, sample_n)

    if progress_cb is not None:
        try:
            await progress_cb({
                "type": "url_batch_verifying",
                "column": column,
                "sample_size": sample_n,
                "total_rows": len(rows),
            })
        except Exception:
            log.exception("url_verify: progress_cb url_batch_verifying raised; suppressed")

    sem = asyncio.Semaphore(_FIRECRAWL_CONCURRENCY)
    scrape_tasks = [_scrape_one(client, url, api_key, sem) for _sid, url, _row in sampled]
    scraped = await asyncio.gather(*scrape_tasks, return_exceptions=False)

    llm_items: List[Dict[str, Any]] = []
    per_url_status: Dict[str, str] = {}
    for (sid, url, row), s in zip(sampled, scraped):
        if s["status"] in {"BROKEN", "UNCHECKED"}:
            per_url_status[url] = s["status"]
            continue
        llm_items.append({
            "url": url,
            "column": column,
            "markdown": s.get("markdown", ""),
            "title": s.get("title", ""),
            "expected_entity": _entity_context(row, url_columns),
        })

    if llm_items:
        verdicts = await _llm_judge(llm_items)
        for it in llm_items:
            per_url_status[it["url"]] = verdicts.get(it["url"], {}).get("status", "UNCHECKED")

    # Aggregate: BROKEN + WRONG count as bad; UNCHECKED doesn't push
    # either way (we don't penalize sites we can't read).
    bad = sum(1 for st in per_url_status.values() if st in {"BROKEN", "WRONG"})
    unchecked = sum(1 for st in per_url_status.values() if st == "UNCHECKED")
    valid = sample_n - bad - unchecked
    decideable = bad + valid
    if decideable == 0:
        verdict = "UNCHECKED"
    elif bad / decideable >= INVALID_THRESHOLD:
        verdict = "INVALID"
    else:
        verdict = "VALID"

    log.info(
        "url_verify: column=%s rows=%d sampled=%d bad=%d valid=%d unchecked=%d → %s",
        column, len(rows), sample_n, bad, valid, unchecked, verdict,
    )
    return {"status": verdict, "sample_size": sample_n, "sample_invalid": bad}


async def verify_batch(
    *,
    rows_by_column: Dict[str, List[Tuple[str, str, Dict[str, Any]]]],
    url_columns: List[str],
    progress_cb: Optional[ProgressCallback],
) -> None:
    """Public entry — verify a batch of rows across one or more URL columns.

    `rows_by_column[col]` = list of (sample_id, url, row_dict) for that
    column. Columns are decided in parallel (separate firecrawl + LLM
    calls); verdicts are then persisted in a SINGLE pass so each row
    gets one tags-write covering every column's verdict at once.
    """
    if not rows_by_column:
        return
    api_key = _firecrawl_key()
    if not api_key:
        log.info("url_verify: FIRECRAWL_API_KEY unset — skipping batch")
        return

    cols = list(rows_by_column.keys())
    async with httpx.AsyncClient() as client:
        results = await asyncio.gather(
            *[
                _decide_column(
                    column=col,
                    rows=rows_by_column[col],
                    url_columns=url_columns,
                    client=client,
                    api_key=api_key,
                    progress_cb=progress_cb,
                )
                for col in cols
            ],
            return_exceptions=True,
        )

    verdicts_by_column: Dict[str, Dict[str, Any]] = {}
    for col, res in zip(cols, results):
        if isinstance(res, Exception):
            log.exception("url_verify: _decide_column raised for %s", col, exc_info=res)
            continue
        if res is None:
            continue
        verdicts_by_column[col] = res
    if not verdicts_by_column:
        return

    snapshots = await asyncio.to_thread(
        _persist_batch_verdicts, verdicts_by_column, rows_by_column,
    )

    if progress_cb is None:
        return
    try:
        for col, v in verdicts_by_column.items():
            await progress_cb({
                "type": "url_batch_verified",
                "column": col,
                "status": v["status"],
                "sample_size": v["sample_size"],
                "sample_invalid": v["sample_invalid"],
                "row_ids": [
                    sid for sid, _url, _row in rows_by_column.get(col, [])
                ],
            })
        # One row_merged per row covers all columns' verdicts at once.
        for snap in snapshots:
            await progress_cb({"type": "row_merged", "row": snap})
    except Exception:
        log.exception("url_verify: progress_cb raised; suppressed")
