"""LLM-judged URL verification — mirrors email_verify.py.

After cells commit URL values, we fetch each URL's preview signal
(title, OG card, meta description, h1) and then ask Haiku 4.5 in ONE
batched call per row "are these URLs about the expected entity?"

Three classification paths, in order of preference:
  1. HTTP status / fetch error → BROKEN (no LLM)
  2. Known-platform slug heuristic match (LinkedIn `/in/<slug>`, GitHub
     `/<slug>`, etc.) → VALID (no LLM)
  3. Bot-walled domain (LinkedIn anonymous, etc.) → UNCHECKED honestly
     instead of bluffing
  4. Everything else → bulk Haiku call, one per row across all the
     row's URL columns

Results land in `sample.tags["url_verification"][col]` next to the
existing `email_verification` dict, so the FE renders both with the
same badge component pattern.

Best-effort: every failure path swallows the error and either marks
UNCHECKED or skips entirely. The verification task NEVER raises.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from typing import Any, Awaitable, Callable, Dict, List, Optional, Set, Tuple
from urllib.parse import urlparse

from anthropic import AsyncAnthropic

from dsl_api.db import SessionLocal
from dsl_api.models.sample import Sample

from dsl_worker.infra.url_preview import UrlPreview, fetch_preview


log = logging.getLogger(__name__)

ProgressCallback = Callable[[Dict[str, Any]], Awaitable[None]]


# Status the FE renders. Mirrors the email_verify status alphabet so
# DataTable can reuse the same badge pattern.
#   VALID     — preview matches the row's expected entity
#   BROKEN    — HTTP error, DNS fail, timeout, or 4xx/5xx
#   WRONG     — page loads but content is about a different entity
#   UNCHECKED — bot-walled, empty preview, or LLM unavailable — we
#               tried and can't tell. Distinct from "didn't try."
UrlStatus = str

# Anonymous fetches behind these domains return a generic logged-out
# shell with identical preview metadata for every URL. Marking these
# UNCHECKED is honest; pretending they passed is worse than a missing
# badge.
_BOT_WALLED_DOMAINS = {
    "linkedin.com", "www.linkedin.com",
    "facebook.com", "www.facebook.com", "m.facebook.com",
    "instagram.com", "www.instagram.com",
    "x.com", "twitter.com", "www.twitter.com", "mobile.twitter.com",
    "tiktok.com", "www.tiktok.com",
}

# For known platforms, the URL slug IS the identity claim. Fast-path
# without an LLM call: if the slug fuzzy-matches the row's entity name
# we accept; otherwise fall through to the LLM (which on bot-walled
# domains will then return UNCHECKED — that path is OK too).
_SLUG_PATTERNS: Dict[str, re.Pattern[str]] = {
    "linkedin.com": re.compile(r"/(?:in|company|school)/([^/?#]+)"),
    "github.com": re.compile(r"/([^/?#]+)/?"),
    "twitter.com": re.compile(r"/([^/?#]+)/?"),
    "x.com": re.compile(r"/([^/?#]+)/?"),
}

_URL_RE = re.compile(r"^https?://", re.IGNORECASE)


def is_url_column(col_def: Optional[Dict[str, Any]], col_name: str) -> bool:
    """Columns whose values get URL verification.

    Explicit type wins; otherwise a name-match on the usual URL suspects.
    """
    if isinstance(col_def, dict):
        t = (col_def.get("type") or "").lower()
        if t in {"url", "link", "website"}:
            return True
        fmt = (col_def.get("format") or "").lower()
        if fmt in {"url", "uri"}:
            return True
        if (col_def.get("contact_type") or "").lower() == "url":
            return True
        v2t = (col_def.get("v2_type") or "").lower()
        if v2t == "url":
            return True
    return bool(re.search(
        r"(?:^|[_-])(url|link|website|homepage|profile)(?:$|[_-])",
        col_name or "", re.IGNORECASE,
    ))


def _domain(url: str) -> str:
    try:
        return (urlparse(url).hostname or "").lower()
    except Exception:
        return ""


def _normalize_slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", (s or "").lower())


def _slug_for(domain: str) -> Optional[re.Pattern[str]]:
    # Strip leading "www." so both forms match the same pattern.
    bare = domain[4:] if domain.startswith("www.") else domain
    return _SLUG_PATTERNS.get(bare)


def _slug_heuristic(preview: UrlPreview, expected_entity: str) -> Optional[UrlStatus]:
    """Decide without an LLM where the slug carries identity. Else None."""
    if not expected_entity:
        return None
    pattern = _slug_for(_domain(preview.final_url or preview.url))
    if pattern is None:
        return None
    try:
        path = urlparse(preview.final_url or preview.url).path or ""
    except Exception:
        return None
    m = pattern.search(path)
    if not m:
        return None
    slug = _normalize_slug(m.group(1))
    entity = _normalize_slug(expected_entity)
    if not slug or not entity:
        return None
    if entity in slug or slug in entity:
        return "VALID"
    return None


def _is_bot_walled(preview: UrlPreview) -> bool:
    d = _domain(preview.final_url or preview.url)
    bare = d[4:] if d.startswith("www.") else d
    return bare in {b[4:] if b.startswith("www.") else b for b in _BOT_WALLED_DOMAINS}


# Module-level Anthropic client. None when ANTHROPIC_API_KEY is unset —
# url_verify then marks every URL UNCHECKED rather than guessing.
_VERIFY_MODEL = os.environ.get("URL_VERIFY_MODEL", "claude-haiku-4-5")
_ANTHROPIC: Optional[AsyncAnthropic] = None


def _get_anthropic() -> Optional[AsyncAnthropic]:
    global _ANTHROPIC
    if _ANTHROPIC is not None:
        return _ANTHROPIC
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        return None
    _ANTHROPIC = AsyncAnthropic(api_key=key)
    return _ANTHROPIC


def _entity_context(row: Dict[str, Any], exclude_columns: Set[str]) -> str:
    """Compact "what is this row about" hint for the LLM judge.

    Skips the URL columns themselves (we don't want the LLM to "verify"
    a URL by looking at itself) and any underscore-prefixed metadata.
    Caps total length so the input stays cheap.
    """
    if not isinstance(row, dict):
        return ""
    parts: List[str] = []
    for k, v in row.items():
        if not isinstance(k, str) or k.startswith("_"):
            continue
        if k in exclude_columns:
            continue
        if v in (None, ""):
            continue
        if isinstance(v, (str, int, float)):
            parts.append(f"{k}: {v}")
        if len(parts) >= 12:
            break
    return "; ".join(parts)[:1000]


def _build_payload_item(item: Dict[str, Any]) -> Dict[str, Any]:
    p: UrlPreview = item["preview"]
    snippet_parts: List[str] = []
    if p.title:
        snippet_parts.append(f"title: {p.title}")
    if p.og_title and p.og_title != p.title:
        snippet_parts.append(f"og_title: {p.og_title}")
    if p.description:
        snippet_parts.append(f"description: {p.description}")
    if p.og_site_name:
        snippet_parts.append(f"site: {p.og_site_name}")
    if p.h1 and p.h1 != p.title:
        snippet_parts.append(f"h1: {p.h1}")
    return {
        "url": item["url"],
        "column": item["column"],
        "expected_entity": (item["expected_entity"] or "")[:500],
        "preview": " | ".join(snippet_parts)[:800],
    }


_JUDGE_SYSTEM = (
    "You verify whether each URL points to a page about the expected "
    "entity for that row.\n\n"
    "For each item return ONE of:\n"
    "  VALID     — preview clearly matches the expected entity\n"
    "  WRONG     — preview is for a different entity / generic / parked / 404 shell\n"
    "  UNCHECKED — preview is too sparse or generic to decide "
    '(e.g. JS loading shell, "Sign in to view", anti-bot challenge)\n\n'
    "Output STRICT JSON, no prose, with shape:\n"
    '{"results":[{"url":"...","status":"VALID|WRONG|UNCHECKED","reason":"short"}]}'
)


async def _llm_batch_judge(
    items: List[Dict[str, Any]],
) -> Dict[str, Dict[str, str]]:
    """One Haiku call for all items in this batch."""
    if not items:
        return {}
    client = _get_anthropic()
    if client is None:
        log.info("url_verify: ANTHROPIC_API_KEY unset — marking %d URL(s) UNCHECKED", len(items))
        return {it["url"]: {"status": "UNCHECKED", "reason": "no_llm_client"} for it in items}

    user_payload = json.dumps(
        {"items": [_build_payload_item(it) for it in items]},
        ensure_ascii=False,
    )

    try:
        resp = await client.messages.create(
            model=_VERIFY_MODEL,
            system=_JUDGE_SYSTEM,
            messages=[{"role": "user", "content": user_payload}],
            max_tokens=2048,
        )
    except Exception:
        log.exception("url_verify: Haiku call raised — marking batch UNCHECKED")
        return {it["url"]: {"status": "UNCHECKED", "reason": "llm_error"} for it in items}

    text_out = ""
    try:
        for block in getattr(resp, "content", []) or []:
            if getattr(block, "type", None) == "text":
                text_out += getattr(block, "text", "") or ""
    except Exception:
        pass
    text_out = (text_out or "").strip()
    # Strip ```json fences if Haiku decided to be helpful anyway.
    if text_out.startswith("```"):
        text_out = re.sub(r"^```(?:json)?\s*", "", text_out)
        text_out = text_out.rstrip("`").strip()

    try:
        parsed = json.loads(text_out)
    except Exception:
        log.warning("url_verify: failed to parse LLM JSON: %s", text_out[:300])
        return {it["url"]: {"status": "UNCHECKED", "reason": "llm_parse_error"} for it in items}

    out: Dict[str, Dict[str, str]] = {}
    for r in (parsed.get("results") or []):
        url = r.get("url") if isinstance(r, dict) else None
        status = r.get("status") if isinstance(r, dict) else None
        reason = ((r.get("reason") if isinstance(r, dict) else "") or "").strip()[:300]
        if isinstance(url, str) and status in {"VALID", "WRONG", "UNCHECKED"}:
            out[url] = {"status": status, "reason": reason}
    # Backfill any missing items so callers always see a verdict per input.
    for it in items:
        out.setdefault(it["url"], {"status": "UNCHECKED", "reason": "no_response"})

    try:
        usage = getattr(resp, "usage", None)
        if usage is not None:
            log.info(
                "url_verify: judged %d URL(s) — input=%s output=%s",
                len(items),
                getattr(usage, "input_tokens", 0),
                getattr(usage, "output_tokens", 0),
            )
    except Exception:
        pass
    return out


# ---------------------------------------------------------------------------
# Global LLM batch queue
#
# Many cells often write a URL each — bulk source fetches commit ~100
# rows at once. Per-row Haiku calls (which is what we'd do without this
# queue) burn one HTTP round-trip per URL. The queue accumulates LLM-
# bound items across rows and fires ONE Haiku call per ~25 URLs, or
# every _BATCH_MAX_WAIT seconds — whichever comes first.
# ---------------------------------------------------------------------------

_BATCH_MAX_SIZE = 25
_BATCH_MAX_WAIT = 1.5  # seconds

_batch_queue: List[Tuple[Dict[str, Any], "asyncio.Future"]] = []
_batch_lock: Optional[asyncio.Lock] = None
_batch_flush_task: Optional[asyncio.Task] = None


def _get_batch_lock() -> asyncio.Lock:
    global _batch_lock
    if _batch_lock is None:
        _batch_lock = asyncio.Lock()
    return _batch_lock


async def _enqueue_for_llm(item: Dict[str, Any]) -> Dict[str, str]:
    """Add an item to the global LLM batch and await its verdict.

    The submitter awaits a per-item Future; the flush coroutine
    resolves every Future in the batch with its respective verdict.
    """
    global _batch_flush_task
    loop = asyncio.get_event_loop()
    fut: asyncio.Future = loop.create_future()
    lock = _get_batch_lock()
    async with lock:
        _batch_queue.append((item, fut))
        if len(_batch_queue) >= _BATCH_MAX_SIZE:
            if _batch_flush_task is not None and not _batch_flush_task.done():
                _batch_flush_task.cancel()
            _batch_flush_task = asyncio.create_task(_flush_batch())
        elif _batch_flush_task is None or _batch_flush_task.done():
            _batch_flush_task = asyncio.create_task(_flush_after_delay())
    return await fut


async def _flush_after_delay() -> None:
    try:
        await asyncio.sleep(_BATCH_MAX_WAIT)
    except asyncio.CancelledError:
        return
    await _flush_batch()


async def _flush_batch() -> None:
    lock = _get_batch_lock()
    async with lock:
        if not _batch_queue:
            return
        batch = list(_batch_queue)
        _batch_queue.clear()

    items = [item for item, _f in batch]
    try:
        results = await _llm_batch_judge(items)
    except Exception:
        log.exception("url_verify: batch judge raised — marking batch UNCHECKED")
        results = {it["url"]: {"status": "UNCHECKED", "reason": "llm_error"} for it in items}

    for item, fut in batch:
        if fut.done():
            continue
        verdict = results.get(item["url"]) or {"status": "UNCHECKED", "reason": "no_verdict"}
        fut.set_result(verdict)


# ---------------------------------------------------------------------------
# Per-row classification — non-blocking against other rows
# ---------------------------------------------------------------------------


def _persist_and_emit(
    *,
    sample_id: str,
    classified: Dict[str, Dict[str, Any]],
    progress_cb: Optional[ProgressCallback],
) -> Optional[Dict[str, Any]]:
    """Merge `classified` into sample.tags['url_verification'] and emit
    url_verified + row_merged for each column. Synchronous DB write."""
    if not classified:
        return None
    row_snapshot_out: Optional[Dict[str, Any]] = None
    write_db = SessionLocal()
    try:
        sample = write_db.query(Sample).filter(Sample.id == sample_id).first()
        if sample is None:
            return None
        tags = dict(sample.tags or {})
        existing = dict(tags.get("url_verification") or {})
        existing.update(classified)
        tags["url_verification"] = existing
        sample.tags = tags
        write_db.commit()
        write_db.refresh(sample)
        row_snapshot_out = {
            "_id": str(sample.id),
            "_seq": sample.seq,
            "_tags": sample.tags or {},
            **(sample.row or {}),
        }
    except Exception:
        log.exception("url_verify persist failed for sample=%s", sample_id)
        try:
            write_db.rollback()
        except Exception:
            pass
        return None
    finally:
        write_db.close()
    return row_snapshot_out


async def _emit_results(
    sample_id: str,
    classified: Dict[str, Dict[str, Any]],
    row_snapshot_out: Optional[Dict[str, Any]],
    progress_cb: Optional[ProgressCallback],
) -> None:
    if progress_cb is None:
        return
    try:
        for col, verdict in classified.items():
            await progress_cb({
                "type": "url_verified",
                "row_id": str(sample_id),
                "column": col,
                "status": verdict["status"],
                "reason": verdict.get("reason") or "",
            })
        if row_snapshot_out is not None:
            await progress_cb({"type": "row_merged", "row": row_snapshot_out})
    except Exception:
        log.exception("progress_cb raised in url_verify (suppressed)")


async def _verify_row(
    *,
    sample_id: str,
    targets: List[Tuple[str, str]],
    row_snapshot: Dict[str, Any],
    progress_cb: Optional[ProgressCallback],
) -> None:
    """Fetch previews + classify + persist for one row's URLs.

    Two-stage persist:
      1. Tier 0/1/2 results (HTTP status, slug heuristic, bot-wall,
         empty preview) → write + emit IMMEDIATELY. The user sees
         BROKEN/VALID badges within a second.
      2. Tier 3 (LLM-bound) results → submit each URL to the global
         batch queue. Each row's submitter awaits its Future
         independently, so a slow LLM batch doesn't hold up tier 0/1/2
         results for other rows.
    """
    # Fetch previews in parallel. The semaphore inside fetch_preview
    # caps cross-row concurrency.
    preview_tasks = [fetch_preview(url) for _col, url in targets]
    previews = await asyncio.gather(*preview_tasks, return_exceptions=True)

    immediate: Dict[str, Dict[str, Any]] = {}
    llm_items: List[Dict[str, Any]] = []
    target_cols = {c for c, _ in targets}
    expected_entity = _entity_context(row_snapshot, target_cols)

    for (col, url), preview in zip(targets, previews):
        if isinstance(preview, Exception):
            immediate[col] = {
                "value": url, "status": "UNCHECKED",
                "reason": "fetch_exception",
                "source": "preview", "code": 0, "final_url": url,
            }
            continue
        p: UrlPreview = preview
        if p.fetch_error or p.status_code == 0:
            immediate[col] = {
                "value": url, "status": "BROKEN",
                "reason": f"fetch_failed_{p.fetch_error or 'no_response'}",
                "source": "preview", "code": p.status_code, "final_url": p.final_url,
            }
            continue
        if p.status_code >= 400:
            immediate[col] = {
                "value": url, "status": "BROKEN",
                "reason": f"http_{p.status_code}",
                "source": "preview", "code": p.status_code, "final_url": p.final_url,
            }
            continue
        slug_verdict = _slug_heuristic(p, expected_entity)
        if slug_verdict:
            immediate[col] = {
                "value": url, "status": slug_verdict,
                "reason": "slug_match",
                "source": "slug", "code": p.status_code, "final_url": p.final_url,
            }
            continue
        if _is_bot_walled(p):
            immediate[col] = {
                "value": url, "status": "UNCHECKED",
                "reason": "bot_walled",
                "source": "preview", "code": p.status_code, "final_url": p.final_url,
            }
            continue
        if not any([p.title, p.description, p.og_title, p.h1]):
            immediate[col] = {
                "value": url, "status": "UNCHECKED",
                "reason": "empty_preview",
                "source": "preview", "code": p.status_code, "final_url": p.final_url,
            }
            continue
        llm_items.append({
            "url": url, "column": col, "preview": p,
            "expected_entity": expected_entity,
        })

    # Stage 1: flush immediates now so the FE shows badges fast.
    if immediate:
        row_snapshot_out = _persist_and_emit(
            sample_id=sample_id, classified=immediate, progress_cb=progress_cb,
        )
        await _emit_results(sample_id, immediate, row_snapshot_out, progress_cb)

    # Stage 2: submit LLM items to the global batcher; each item's
    # future resolves when ITS batch flushes. Awaiting in parallel so a
    # slow batch on column A doesn't delay column B.
    if not llm_items:
        return
    verdicts = await asyncio.gather(
        *[_enqueue_for_llm(it) for it in llm_items],
        return_exceptions=False,
    )
    classified: Dict[str, Dict[str, Any]] = {}
    for it, verdict in zip(llm_items, verdicts):
        classified[it["column"]] = {
            "value": it["url"],
            "status": verdict.get("status", "UNCHECKED"),
            "reason": verdict.get("reason") or "",
            "source": "llm",
            "code": it["preview"].status_code,
            "final_url": it["preview"].final_url,
        }
    row_snapshot_out = _persist_and_emit(
        sample_id=sample_id, classified=classified, progress_cb=progress_cb,
    )
    await _emit_results(sample_id, classified, row_snapshot_out, progress_cb)


def schedule_verifications(
    *,
    sample_id: str,
    written_values: Dict[str, Any],
    url_columns: Set[str],
    row_snapshot: Dict[str, Any],
    progress_cb: Optional[ProgressCallback],
) -> List[asyncio.Task[None]]:
    """Fire URL verification for a row's qualifying URL values.

    Returns ONE task per row — within a row, all URLs are previewed in
    parallel and judged in a single Haiku call. Concurrent rows still
    run in parallel; per-URL caching dedupes repeat fetches.
    """
    if not url_columns:
        return []
    targets: List[Tuple[str, str]] = []
    for col, val in written_values.items():
        if col not in url_columns:
            continue
        if not isinstance(val, str) or not _URL_RE.match(val):
            continue
        targets.append((col, val))
    if not targets:
        return []
    log.info(
        "url_verify: scheduling verifies for sample %s — %d url(s) across cols %s",
        sample_id, len(targets), [c for c, _ in targets],
    )

    async def _run() -> None:
        if progress_cb is not None:
            for col, val in targets:
                try:
                    await progress_cb({
                        "type": "url_verifying",
                        "row_id": str(sample_id),
                        "column": col,
                        "value": val,
                    })
                except Exception:
                    log.exception("url_verify url_verifying progress_cb raised; suppressed")
        await _verify_row(
            sample_id=sample_id,
            targets=targets,
            row_snapshot=row_snapshot,
            progress_cb=progress_cb,
        )

    return [asyncio.create_task(_run())]
