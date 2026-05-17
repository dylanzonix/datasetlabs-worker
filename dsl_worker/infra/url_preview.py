"""Cheap URL preview fetcher for relevance verification.

We don't want to render pages or pull megabytes of HTML — we just want
the same signal a Slack/Twitter/Facebook preview crawler reads:
`<title>`, `<meta name="description">`, OG card tags, the first `<h1>`,
and the canonical link. Sites that care about being shared elsewhere
emit these in the document head, so they appear in the opening few
kilobytes of the response.

Caches results in-process for an hour so re-fills / multi-row checks
of the same URL don't re-fetch.

Never raises — fetch errors come back on the returned dataclass so the
caller can decide (typically: classify as BROKEN, skip the LLM call).
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Optional
from urllib.parse import urlparse

import httpx

try:
    from selectolax.parser import HTMLParser  # type: ignore
    _SELECTOLAX_AVAILABLE = True
except ImportError:
    HTMLParser = None  # type: ignore
    _SELECTOLAX_AVAILABLE = False

logger = logging.getLogger(__name__)

if not _SELECTOLAX_AVAILABLE:
    logger.warning(
        "url_preview: selectolax is not installed — URL verification will "
        "still classify by HTTP status but cannot extract preview metadata. "
        "Run `pip install selectolax` (already in requirements.txt) to enable."
    )


@dataclass(frozen=True)
class UrlPreview:
    url: str
    status_code: int        # 0 means fetch failed entirely
    title: str
    description: str
    og_title: str
    og_site_name: str
    h1: str
    canonical: str
    final_url: str          # after redirects
    fetch_error: str        # "" if no error


# Cap concurrent outbound fetches per worker. Two layers:
#   * Global: how many fetches can be in-flight at all (across every
#     domain). Bounds the worker's overall socket usage.
#   * Per-host: how many fetches can be in-flight against ONE domain.
#     Critical for bulk verifies on a table where every URL is on the
#     same site (e.g. 100 gsaauctions.gov rows): the global cap alone
#     would happily blast 20 simultaneous requests at one origin,
#     which trips anti-bot rules and is rude regardless.
_MAX_CONCURRENT_FETCHES = 20
_MAX_CONCURRENT_PER_HOST = 4

_FETCH_SEMAPHORE: Optional[asyncio.Semaphore] = None
_HOST_SEMAPHORES: dict[str, asyncio.Semaphore] = {}


def _get_semaphore() -> asyncio.Semaphore:
    global _FETCH_SEMAPHORE
    if _FETCH_SEMAPHORE is None:
        _FETCH_SEMAPHORE = asyncio.Semaphore(_MAX_CONCURRENT_FETCHES)
    return _FETCH_SEMAPHORE


def _get_host_semaphore(host: str) -> asyncio.Semaphore:
    sem = _HOST_SEMAPHORES.get(host)
    if sem is None:
        sem = asyncio.Semaphore(_MAX_CONCURRENT_PER_HOST)
        _HOST_SEMAPHORES[host] = sem
    return sem


_FETCH_BYTES = 32 * 1024
_FETCH_TIMEOUT = 8.0

# A real-browser User-Agent. Government, .edu, and many enterprise
# sites (gsaauctions.gov, anything behind Akamai/Cloudflare default
# rules) drop non-browser UAs at the edge, which used to make every
# URL in those tables come back UNCHECKED. We're not pretending to be
# a different identity — link-preview crawlers (Slack, Twitter, LI)
# all ship Mozilla UAs; we're matching that convention.
_USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36"
)

_PREVIEW_CACHE: dict[str, tuple[UrlPreview, float]] = {}
_PREVIEW_CACHE_TTL = 3600.0


def _empty_preview(url: str, error: str, status: int = 0, final: str = "") -> UrlPreview:
    return UrlPreview(
        url=url, status_code=status, title="", description="", og_title="",
        og_site_name="", h1="", canonical="", final_url=final or url,
        fetch_error=error,
    )


async def fetch_preview(url: str) -> UrlPreview:
    """Return a UrlPreview for `url`. Always returns; never raises."""
    if not isinstance(url, str) or not url:
        return _empty_preview(url or "", "empty_url")
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        return _empty_preview(url, "unsupported_scheme")

    now = time.monotonic()
    cached = _PREVIEW_CACHE.get(url)
    if cached and cached[1] > now:
        return cached[0]

    host = (parsed.hostname or "").lower()
    global_sem = _get_semaphore()
    host_sem = _get_host_semaphore(host) if host else None
    if host_sem is not None:
        # Hold the host-level slot OUTSIDE the global one so the
        # global cap stays "true concurrent connections" rather than
        # "queue waiters." If a host is saturated, queue here instead
        # of consuming a global slot just to block on a host slot.
        async with host_sem:
            async with global_sem:
                preview = await _fetch_once(url)
    else:
        async with global_sem:
            preview = await _fetch_once(url)

    _PREVIEW_CACHE[url] = (preview, now + _PREVIEW_CACHE_TTL)
    return preview


async def _fetch_once(url: str) -> UrlPreview:
    headers = {
        "User-Agent": _USER_AGENT,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
    }
    body: bytes
    status: int
    final_url: str
    try:
        async with httpx.AsyncClient(
            timeout=_FETCH_TIMEOUT,
            follow_redirects=True,
            headers=headers,
        ) as client:
            async with client.stream("GET", url) as resp:
                status = resp.status_code
                final_url = str(resp.url)
                buf = bytearray()
                async for chunk in resp.aiter_bytes(chunk_size=4096):
                    buf.extend(chunk)
                    if len(buf) >= _FETCH_BYTES:
                        break
                body = bytes(buf)
    except (httpx.TimeoutException, httpx.HTTPError, asyncio.TimeoutError) as e:
        # Demoted from info → debug. With bulk verifies firing on every
        # row insert, gov / enterprise sites that 403 or RST our request
        # flood the log with one line per URL otherwise.
        logger.debug("url_preview: %s for %s: %s", type(e).__name__, url, e)
        return _empty_preview(url, type(e).__name__)
    except Exception as e:
        logger.warning("url_preview: unexpected error for %s: %s", url, e, exc_info=True)
        return _empty_preview(url, type(e).__name__)

    if status >= 400:
        # Don't bother parsing — status alone marks it broken. (Some
        # sites serve preview meta even on 404s; we ignore that to keep
        # classification consistent.)
        return UrlPreview(
            url=url, status_code=status, title="", description="", og_title="",
            og_site_name="", h1="", canonical="", final_url=final_url,
            fetch_error="",
        )

    return _parse_html(url, status, final_url, body)


def _parse_html(url: str, status: int, final_url: str, body: bytes) -> UrlPreview:
    try:
        text = body.decode("utf-8", errors="replace")
    except Exception:
        text = ""

    title = description = og_title = og_site_name = h1 = canonical = ""
    if text and _SELECTOLAX_AVAILABLE:
        try:
            tree = HTMLParser(text)
            t_node = tree.css_first("title")
            if t_node is not None:
                title = (t_node.text() or "").strip()[:300]
            h1_node = tree.css_first("h1")
            if h1_node is not None:
                h1 = (h1_node.text() or "").strip()[:300]
            canon_node = tree.css_first('link[rel="canonical"]')
            if canon_node is not None:
                canonical = (canon_node.attributes.get("href") or "").strip()[:500]
            for meta in tree.css("meta"):
                attrs = meta.attributes
                name = (attrs.get("name") or "").lower()
                prop = (attrs.get("property") or "").lower()
                content = (attrs.get("content") or "").strip()
                if not content:
                    continue
                if name == "description" and not description:
                    description = content[:500]
                elif prop == "og:description" and not description:
                    description = content[:500]
                elif prop == "og:title" and not og_title:
                    og_title = content[:300]
                elif prop == "og:site_name" and not og_site_name:
                    og_site_name = content[:200]
        except Exception:
            logger.warning("url_preview: parse failed for %s", url, exc_info=True)

    return UrlPreview(
        url=url, status_code=status, title=title, description=description,
        og_title=og_title, og_site_name=og_site_name, h1=h1, canonical=canonical,
        final_url=final_url, fetch_error="",
    )
