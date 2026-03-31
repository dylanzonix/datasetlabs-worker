"""Thin HTTP client for the credential pool service."""

import logging
from typing import List

import httpx

logger = logging.getLogger(__name__)


async def load_pool_cookies(credential_pool_url: str) -> List[dict]:
    """Fetch LRU-distributed cookies from the credential pool service."""
    if not credential_pool_url:
        return []
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(f"{credential_pool_url.rstrip('/')}/session/cookies")
            resp.raise_for_status()
        data = resp.json()
        cookies = data.get("cookies", [])
        if cookies:
            domains = {c.get("domain", "").lstrip(".") for c in cookies}
            logger.info(f"[CredentialPool] Loaded {len(cookies)} cookies for {domains}")
        return cookies
    except Exception as e:
        logger.warning(f"[CredentialPool] Failed to fetch cookies: {e}")
        return []
