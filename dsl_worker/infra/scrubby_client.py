"""
Scrubby email-validation API client.

Single-email validation only — Scrubby's bulk endpoint is overkill for our
drip-fed cell fills. Calls return None on any error so callers can degrade
gracefully (no crashes, no UI alerts).

API: https://api.scrubby.io/validate_email
Auth: x-api-key header
Cost: 1 credit per validation regardless of result.
"""

import asyncio
import logging
import os
import time
from dataclasses import dataclass
from typing import Literal, Optional

import httpx

logger = logging.getLogger(__name__)

BASE_URL = "https://api.scrubby.io"

# Internal status — what we persist on enrichment_data.results.email.status.
# Maps Scrubby's `result` field, which is the same coarse bucket their site uses.
ScrubbyStatus = Literal["DELIVERABLE", "RISKY", "INVALID", "UNVERIFIED"]


@dataclass(frozen=True)
class ScrubbyResult:
    email: str
    status: ScrubbyStatus
    raw_status: str           # Scrubby's underlying detail (HARD_BOUNCE, RISKY, OK, etc.)
    credits_used: int
    remaining_credits: int


class ScrubbyClient:
    def __init__(self, api_key: str, timeout: float = 60.0) -> None:
        self._api_key = api_key
        self._timeout = timeout
        self._headers = {
            "x-api-key": api_key,
            "Content-Type": "application/json",
        }
        # Scrubby rate-limits at 1 request per second per API key. Pace
        # submissions with a lock so concurrent verifications across cells
        # don't get 429'd. Calls themselves take 15-60s on Scrubby's side
        # but are independent — this lock only serializes the *submit*.
        self._submit_lock = asyncio.Lock()
        self._last_submit_at: float = 0.0

    async def validate_email(self, email: str) -> Optional[ScrubbyResult]:
        """Validate one email. Returns None on any error — do not raise.

        Retries ONCE on transient failure (network error, 5xx, 429). When
        many cells fill in one orchestrator turn we get 30+ concurrent
        verifications and Scrubby occasionally drops one to a timeout or
        rate-limit response; a single retry catches those without burning
        credits on real Invalids (their cache returns identical results
        for free on the second hit).
        """
        for attempt in range(2):
            result = await self._validate_once(email)
            if result is not None:
                return result
            if attempt == 0:
                await asyncio.sleep(3.0)
        return None

    async def _validate_once(self, email: str) -> Optional[ScrubbyResult]:
        async with self._submit_lock:
            elapsed = time.monotonic() - self._last_submit_at
            if elapsed < 1.05:
                await asyncio.sleep(1.05 - elapsed)
            self._last_submit_at = time.monotonic()
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.post(
                    f"{BASE_URL}/validate_email",
                    headers=self._headers,
                    json={"email": email},
                )
        except (httpx.TimeoutException, httpx.HTTPError, asyncio.TimeoutError) as e:
            logger.info("scrubby: network error for %s: %s", email, e)
            return None
        except Exception as e:
            logger.warning("scrubby: unexpected error for %s: %s", email, e, exc_info=True)
            return None

        if resp.status_code != 200:
            logger.info("scrubby: HTTP %s for %s: %s", resp.status_code, email, resp.text[:200])
            return None

        try:
            data = resp.json()
        except Exception:
            logger.info("scrubby: non-json response for %s: %s", email, resp.text[:200])
            return None

        result_raw = (data.get("result") or "").strip()
        status_map = {
            "Valid": "DELIVERABLE",
            "Invalid": "INVALID",
            "Risky": "RISKY",
            "Unknown": "UNVERIFIED",
        }
        mapped: Optional[ScrubbyStatus] = status_map.get(result_raw)
        if mapped is None:
            logger.info("scrubby: unknown result '%s' for %s", result_raw, email)
            return None

        return ScrubbyResult(
            email=email,
            status=mapped,
            raw_status=str(data.get("status") or ""),
            credits_used=int(data.get("credits_used") or 0),
            remaining_credits=int(data.get("remaining_credits") or 0),
        )


_singleton: Optional[ScrubbyClient] = None
_logged_status: bool = False


def get_scrubby_client() -> Optional[ScrubbyClient]:
    """Return a process-wide ScrubbyClient, or None if SCRUBBY_API_KEY is unset.

    Callers must handle None — the feature is intentionally optional so the
    worker runs fine without a key configured.
    """
    global _singleton, _logged_status
    if _singleton is not None:
        return _singleton
    key = os.environ.get("SCRUBBY_API_KEY")
    if not key:
        if not _logged_status:
            # Log loudly once so it's obvious in worker logs why no
            # emails are getting badges — the silent skip path was
            # the #1 source of "is email verify broken?" confusion.
            logger.warning(
                "scrubby: SCRUBBY_API_KEY is unset — email verification disabled. "
                "Cells will commit without DELIVERABLE/RISKY/INVALID badges."
            )
            _logged_status = True
        return None
    _singleton = ScrubbyClient(key)
    if not _logged_status:
        logger.info("scrubby: enabled (key length=%d)", len(key))
        _logged_status = True
    return _singleton
