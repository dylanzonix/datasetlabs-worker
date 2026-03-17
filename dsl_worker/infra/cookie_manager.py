"""
Cookie manager — load browser cookies from Azure Blob Storage.

Global cookies: Pre-authenticated cookies for common sites (LinkedIn, X, Google, etc.)
Shared across ALL projects. Stored at a configurable blob path (default:
``browser/global_cookies.json``). Workers read-only; managed out-of-band.

Cookie format is Playwright's native ``storage_state`` dict::

    {
        "cookies": [
            {
                "name": "session_id",
                "value": "abc123",
                "domain": ".example.com",
                "path": "/",
                "expires": 1735689600,
                "httpOnly": true,
                "secure": true,
                "sameSite": "Lax"
            },
            ...
        ],
        "origins": [
            {
                "origin": "https://example.com",
                "localStorage": [
                    {"name": "key", "value": "val"}
                ]
            },
            ...
        ]
    }

To upload global cookies:
    1. Export cookies from a local Playwright session (``context.storage_state()``)
    2. Upload the JSON file to Azure Blob:
       ``az storage blob upload -c <container> -n browser/global_cookies.json -f cookies.json``
    3. Or use Azure Storage Explorer to upload to the same path.
"""

import json
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def _download_blob_json(
    blob_service_client: Any,
    container: str,
    blob_path: str,
) -> Optional[Dict]:
    """Download a JSON blob, returning parsed dict or None if not found."""
    try:
        blob_client = blob_service_client.get_blob_client(
            container=container, blob=blob_path,
        )
        data = blob_client.download_blob().readall()
        return json.loads(data)
    except Exception as e:
        logger.debug(f"[CookieManager] Could not load {blob_path}: {e}")
        return None


def load_cookies(
    blob_service_client: Any,
    container: str,
    global_cookies_blob_path: str = "browser/global_cookies.json",
) -> Optional[Dict]:
    """Load global cookies from Azure Blob.

    Returns a Playwright-compatible storage_state dict, or None if no cookies found.
    """
    state = _download_blob_json(
        blob_service_client, container, global_cookies_blob_path,
    )
    if state:
        cookie_count = len(state.get("cookies", []))
        origin_count = len(state.get("origins", []))
        logger.info(
            f"[CookieManager] Loaded {cookie_count} cookies, "
            f"{origin_count} origins"
        )
    return state
