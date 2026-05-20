"""
Scrubby email-validation API client.

Two paths:
  • validate_email — single email, returns immediately (15s server timeout).
    Right call for cell-agent enrichment (one email at a time).
  • validate_emails_bulk — submit a batch, poll until done (30–60s typical).
    Right call for connector imports that drop N emails in one go.

Calls return None / empty on any error so callers can degrade gracefully
(no crashes, no UI alerts).

API: https://api.scrubby.io/
Auth: x-api-key header
Cost: 1 credit per email, billed at submission.
Rate limit: 1 request per second per API key.
"""

import asyncio
import logging
import os
import time
from dataclasses import dataclass
from typing import Dict, List, Literal, Optional

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

    async def validate_emails_bulk(
        self,
        emails: List[str],
        poll_interval: float = 30.0,
        max_wait: float = 600.0,
    ) -> Dict[str, ScrubbyResult]:
        """Submit a batch and poll until done. Returns {email: ScrubbyResult}.

        Emails missing from the returned map either failed to validate or
        the batch timed out — caller treats absence as UNVERIFIED. Errors
        never raise; the worst case is an empty dict (no badges; same as
        the feature being disabled).

        Cost: 1 credit per email, billed at submit time. Scrubby caches
        results — re-submitting the same email is free on the second hit.
        """
        if not emails:
            return {}
        # Dedup before submit. Scrubby would charge per duplicate even
        # though it returns identical results.
        unique = list({e.strip().lower() for e in emails if isinstance(e, str) and "@" in e})
        if not unique:
            return {}

        # Submit step — respects the 1-RPS lock since /validate_bulk_emails
        # is rate-limited the same as the single endpoint.
        async with self._submit_lock:
            elapsed = time.monotonic() - self._last_submit_at
            if elapsed < 1.05:
                await asyncio.sleep(1.05 - elapsed)
            self._last_submit_at = time.monotonic()

        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.post(
                    f"{BASE_URL}/validate_bulk_emails",
                    headers=self._headers,
                    json={"email": unique},
                )
        except Exception as e:
            logger.warning("scrubby: bulk submit failed (%d emails): %s", len(unique), e)
            return {}

        if resp.status_code not in (200, 202):
            logger.info("scrubby: bulk submit HTTP %s: %s", resp.status_code, resp.text[:300])
            return {}

        try:
            sub = resp.json()
        except Exception:
            logger.info("scrubby: bulk submit non-json: %s", resp.text[:200])
            return {}

        identifier = sub.get("identifier")
        if not identifier:
            logger.info("scrubby: bulk submit missing identifier: %s", sub)
            return {}
        retry_after = float(sub.get("retry_after_seconds") or poll_interval)
        logger.info(
            "scrubby: bulk submitted — %d emails, batch=%s, first poll in %.0fs",
            len(unique), identifier, retry_after,
        )

        # Poll step — wait retry_after first, then every poll_interval.
        await asyncio.sleep(retry_after)
        started = time.monotonic()
        while True:
            async with self._submit_lock:
                elapsed = time.monotonic() - self._last_submit_at
                if elapsed < 1.05:
                    await asyncio.sleep(1.05 - elapsed)
                self._last_submit_at = time.monotonic()
            try:
                async with httpx.AsyncClient(timeout=self._timeout) as client:
                    poll = await client.post(
                        f"{BASE_URL}/fetch_bulk_results",
                        headers=self._headers,
                        json={"identifier": identifier},
                    )
            except Exception as e:
                logger.info("scrubby: bulk poll failed (batch=%s): %s", identifier, e)
                if time.monotonic() - started > max_wait:
                    return {}
                await asyncio.sleep(poll_interval)
                continue

            if poll.status_code != 200:
                logger.info("scrubby: bulk poll HTTP %s: %s", poll.status_code, poll.text[:200])
                if time.monotonic() - started > max_wait:
                    return {}
                await asyncio.sleep(poll_interval)
                continue

            try:
                pdata = poll.json()
            except Exception:
                logger.info("scrubby: bulk poll non-json: %s", poll.text[:200])
                await asyncio.sleep(poll_interval)
                continue

            status = pdata.get("status")
            if status == "completed":
                return self._parse_bulk_results(pdata.get("results") or {})
            if time.monotonic() - started > max_wait:
                logger.warning(
                    "scrubby: bulk batch=%s timed out after %.0fs in %s",
                    identifier, max_wait, status,
                )
                return self._parse_bulk_results(pdata.get("results") or {})
            await asyncio.sleep(poll_interval)

    @staticmethod
    def _parse_bulk_results(results: Dict[str, Dict[str, str]]) -> Dict[str, "ScrubbyResult"]:
        status_map = {
            "Valid": "DELIVERABLE",
            "Invalid": "INVALID",
            "Risky": "RISKY",
            "Unknown": "UNVERIFIED",
        }
        out: Dict[str, ScrubbyResult] = {}
        for email, payload in results.items():
            if not isinstance(payload, dict):
                continue
            result_raw = (payload.get("result") or "").strip()
            if result_raw in ("pending", ""):
                continue
            mapped: Optional[ScrubbyStatus] = status_map.get(result_raw)
            if mapped is None:
                continue
            out[email.lower()] = ScrubbyResult(
                email=email,
                status=mapped,
                raw_status=str(payload.get("status") or ""),
                credits_used=1,
                remaining_credits=0,
            )
        return out


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
