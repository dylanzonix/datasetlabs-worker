"""Glue between chat row-write sites and the URL verification module.

Mirrors email_verify_hook: same _BACKGROUND_TASKS pinning pattern,
same fresh-SessionLocal-per-event emitter. The verify module itself
is path-agnostic and takes an async progress_cb; this module bridges
it to chat's SSE layer.

Source filtering: URLs from upstream-validated providers (apify
actors, apollo, fullenrich, google_maps, user-uploaded files) are
skipped entirely — those sources already return curated URLs, and in
apify's case the scraped sites frequently don't render under
firecrawl so a verify pass would mark a lot of real URLs INVALID. We
only verify sources where an LLM/agent picks the URL (browser_use,
web_harvest, llm).

Non-HTML filter: image/document/archive URLs (.jpg, .pdf, .zip, ...)
are skipped per-URL — firecrawl can't extract text from binary
assets so there's nothing for the judge to read.

Detection is value-based (`^https?://`) because most user-defined
columns store URLs under generic `text` type.

Tasks are pinned in `_BACKGROUND_TASKS` so fire-and-forget callers
can drop the returned reference without risk of GC cancelling
mid-await.
"""

from __future__ import annotations

import asyncio
import logging
import re
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple
from urllib.parse import urlparse

from dsl_api.db import SessionLocal
from dsl_api.models import ChatRun

from dsl_worker.chat import run_state
from dsl_worker.infra import url_verify


log = logging.getLogger(__name__)


_BACKGROUND_TASKS: set = set()


def _register_background(task: asyncio.Task) -> None:
    _BACKGROUND_TASKS.add(task)
    task.add_done_callback(_BACKGROUND_TASKS.discard)


_VALUE_URL_RE = re.compile(r"^https?://", re.IGNORECASE)


# Per-batch ceiling. A single browser_use scrape can return 1000+ rows,
# and verifying each at ~30s (firecrawl + LLM judge, concurrency 5) is
# ~100 minutes of background work that also blocks uvicorn reload.
# Beyond this cap we just skip the verify entirely — same UX as if
# FIRECRAWL_API_KEY were unset (no badge, no penalty). Override via
# ``URL_VERIFY_MAX_PER_BATCH`` if you really want longer runs.
import os as _os  # noqa: E402

try:
    _MAX_PER_BATCH = max(0, int(_os.environ.get("URL_VERIFY_MAX_PER_BATCH", "60")))
except ValueError:
    _MAX_PER_BATCH = 60


# Sources whose URLs are pre-validated by the provider/scraper. Apify
# actors return live-scraped pages so the URL by definition resolved
# at scrape time; apollo / fullenrich / google_maps return curated
# company URLs from their own data; file uploads come from the user.
# Skipping these saves firecrawl credits and avoids false-INVALIDs on
# sites firecrawl can't render.
_TRUSTED_SOURCES_EXACT = {
    "apollo_companies",
    "fullenrich_people",
    "google_maps",
    "file",
}
_TRUSTED_SOURCE_PREFIXES = ("apify_actor:",)


def _is_trusted_source(source: Optional[str]) -> bool:
    if not source:
        return False
    if source in _TRUSTED_SOURCES_EXACT:
        return True
    return any(source.startswith(p) for p in _TRUSTED_SOURCE_PREFIXES)


# Binary/asset extensions firecrawl can't extract meaningful text from.
# `.html` / `.htm` / `.php` / `.aspx` etc are intentionally absent — those
# are HTML pages worth scraping.
_NON_HTML_EXT_RE = re.compile(
    r"\.(jpg|jpeg|png|gif|webp|svg|bmp|ico|tiff|avif|heic|"
    r"pdf|mp4|mp3|m4a|wav|mov|avi|mkv|webm|flv|"
    r"zip|tar|gz|bz2|7z|rar|"
    r"csv|xlsx|xls|doc|docx|ppt|pptx|odt|ods|"
    r"json|xml|rss|atom|"
    r"exe|dmg|pkg|msi|deb|rpm)$",
    re.IGNORECASE,
)


def _is_verifiable_url(url: str) -> bool:
    """Skip URLs that point to binary assets — firecrawl returns no text."""
    try:
        parsed = urlparse(url)
    except Exception:
        return False
    path = (parsed.path or "").rstrip("/")
    return not bool(_NON_HTML_EXT_RE.search(path))


def _url_columns_from_values(rows: List[Tuple[str, Dict[str, Any]]]) -> List[str]:
    """URL columns are detected by VALUE across the batch.

    Apify / web_harvest / many user-defined columns default to
    `type: "text"` even when they store URLs, so the column-type
    detector was missing real URL fields in the old code. A column is
    a URL column if ANY row in the batch has a URL-shaped value in it.
    Stable-ordered for deterministic batches in logs/tests.
    """
    seen: Dict[str, None] = {}  # preserve first-seen order
    for _sid, written in rows:
        if not isinstance(written, dict):
            continue
        for k, v in written.items():
            if not isinstance(k, str) or k in seen:
                continue
            if isinstance(v, str) and _VALUE_URL_RE.match(v):
                seen[k] = None
    return list(seen.keys())


def _make_event_emitter(
    run_id: Optional[Any],
) -> Optional[Callable[[Dict[str, Any]], Awaitable[None]]]:
    if not run_id:
        return None

    async def emit(event: Dict[str, Any]) -> None:
        try:
            event_type = event.get("type")
            if not isinstance(event_type, str) or not event_type:
                return
            payload = {k: v for k, v in event.items() if k != "type"}
            db = SessionLocal()
            try:
                run_obj = db.query(ChatRun).filter(ChatRun.id == run_id).first()
                if run_obj is None:
                    return
                run_state.emit_event(db, run_obj, event_type, payload)
                db.commit()
            finally:
                db.close()
        except Exception:
            log.debug("url_verify_hook emit failed (suppressed)", exc_info=True)

    return emit


def schedule_bulk_for_rows(
    *,
    run_id: Optional[Any],
    rows: List[Tuple[str, Dict[str, Any]]],
    source: Optional[str] = None,
    column_descriptions: Optional[Dict[str, str]] = None,
) -> Optional[asyncio.Task]:
    """Fire one URL-verify task for the batch.

    `rows`: list of (sample_id, written_values) tuples — same shape
    email_verify_hook.schedule_bulk_for_rows accepts.

    `source`: the table's source identifier (e.g. ``apify_actor:abc``,
    ``browser_use``, ``web_harvest``). Trusted sources short-circuit
    here — the URLs they return are already validated upstream.

    Returns the spawned task (pinned in `_BACKGROUND_TASKS`), or None
    if firecrawl isn't configured, the source is trusted, or there
    are no verifiable URL cells in the batch.
    """
    if not rows:
        return None
    if _is_trusted_source(source):
        log.info(
            "url_verify: skipping batch — source %r is upstream-trusted",
            source,
        )
        return None
    url_cols = _url_columns_from_values(rows)
    if not url_cols:
        return None

    # Group: rows_by_column[col] = [(sample_id, url, row_dict), ...].
    # Image/PDF/archive URLs are filtered here — firecrawl can't read
    # binary assets so verifying them is wasted credits, and the FE
    # treats absence-of-verification as "no badge" (same as if the
    # feature were disabled), which is the right outcome for assets.
    #
    # Truncate to the first _MAX_PER_BATCH URL cells in row order.
    # Above the cap, the OLD behavior skipped the entire batch (so an
    # 800-row table got zero verification signal even though the user
    # would happily pay to validate the first chunk). NEW behavior:
    # verify the first N URLs as a sample — same cost ceiling, but the
    # user sees verdicts on the leading rows. Iteration is row-major
    # (sample-by-sample, then column-within-sample) so the verified
    # set is deterministic + reproducible across runs.
    rows_by_column: Dict[str, List[Tuple[str, str, Dict[str, Any]]]] = {}
    total_added = 0
    total_seen = 0
    cap_hit = False
    for sample_id, written in rows:
        if not isinstance(written, dict):
            continue
        for col in url_cols:
            val = written.get(col)
            if not isinstance(val, str) or not _VALUE_URL_RE.match(val):
                continue
            if not _is_verifiable_url(val):
                continue
            total_seen += 1
            if _MAX_PER_BATCH > 0 and total_added >= _MAX_PER_BATCH:
                cap_hit = True
                continue
            rows_by_column.setdefault(col, []).append(
                (str(sample_id), val, written)
            )
            total_added += 1
        if _MAX_PER_BATCH > 0 and total_added >= _MAX_PER_BATCH and cap_hit:
            # Finish the row we started, then break out of the outer
            # row loop. Avoids the asymmetric "row's first 2 URL cols
            # got verified, last 3 didn't" case.
            break
    if not rows_by_column:
        return None
    if cap_hit:
        log.info(
            "url_verify: truncating batch — verifying first %d of %d URLs (row-major sample)",
            total_added, total_seen,
        )

    progress_cb = _make_event_emitter(run_id)
    log.info(
        "url_verify: scheduling bulk verify — %d URL(s) across %d column(s) / %d row(s)",
        total_added, len(rows_by_column), len(rows),
    )
    task = asyncio.create_task(
        url_verify.verify_batch(
            rows_by_column=rows_by_column,
            url_columns=url_cols,
            progress_cb=progress_cb,
            column_descriptions=column_descriptions,
        )
    )
    _register_background(task)
    return task
