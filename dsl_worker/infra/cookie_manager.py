"""
Cookie manager — load/save browser cookies from Azure Blob Storage.

Two tiers of cookies:
- **Global**: Pre-authenticated cookies for common sites (LinkedIn, X, Google, etc.)
  Shared across ALL projects. Stored at a configurable blob path (default:
  ``browser/global_cookies.json``). Workers read-only; managed out-of-band.
- **Per-project**: Accumulated during a project's browsing sessions. Stored at
  ``projects/{project_id}/browser_cookies.json``. Workers read on start, write on cleanup.

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
        # ResourceNotFoundError or any other issue — not fatal
        logger.debug(f"[CookieManager] Could not load {blob_path}: {e}")
        return None


def _upload_blob_json(
    blob_service_client: Any,
    container: str,
    blob_path: str,
    data: Dict,
) -> None:
    """Upload a dict as JSON to a blob, overwriting if exists."""
    blob_client = blob_service_client.get_blob_client(
        container=container, blob=blob_path,
    )
    blob_client.upload_blob(
        json.dumps(data, indent=2).encode("utf-8"),
        overwrite=True,
    )
    logger.info(f"[CookieManager] Saved cookies to {blob_path}")


def _merge_storage_states(
    base: Optional[Dict],
    overlay: Optional[Dict],
) -> Optional[Dict]:
    """Merge two Playwright storage_state dicts.

    Overlay cookies override base cookies when domain+name+path match.
    Overlay origins override base origins when origin matches.
    """
    if not base and not overlay:
        return None
    if not base:
        return overlay
    if not overlay:
        return base

    # Merge cookies: overlay wins on (domain, name, path) key
    base_cookies: List[Dict] = base.get("cookies", [])
    overlay_cookies: List[Dict] = overlay.get("cookies", [])

    cookie_key = lambda c: (c.get("domain", ""), c.get("name", ""), c.get("path", "/"))
    merged_cookies = {cookie_key(c): c for c in base_cookies}
    for c in overlay_cookies:
        merged_cookies[cookie_key(c)] = c

    # Merge origins: overlay wins on origin key
    base_origins: List[Dict] = base.get("origins", [])
    overlay_origins: List[Dict] = overlay.get("origins", [])

    merged_origins = {o.get("origin", ""): o for o in base_origins}
    for o in overlay_origins:
        merged_origins[o.get("origin", "")] = o

    return {
        "cookies": list(merged_cookies.values()),
        "origins": list(merged_origins.values()),
    }


def load_cookies(
    blob_service_client: Any,
    container: str,
    project_id: str,
    global_cookies_blob_path: str = "browser/global_cookies.json",
) -> Optional[Dict]:
    """Load and merge global + per-project cookies from Azure Blob.

    Returns a Playwright-compatible storage_state dict, or None if no cookies found.
    """
    global_state = _download_blob_json(
        blob_service_client, container, global_cookies_blob_path,
    )
    project_state = _download_blob_json(
        blob_service_client, container,
        f"projects/{project_id}/browser_cookies.json",
    )

    merged = _merge_storage_states(global_state, project_state)
    if merged:
        cookie_count = len(merged.get("cookies", []))
        origin_count = len(merged.get("origins", []))
        logger.info(
            f"[CookieManager] Loaded {cookie_count} cookies, "
            f"{origin_count} origins for project {project_id}"
        )
    return merged


def save_project_cookies(
    browser_session: Any,
    blob_service_client: Any,
    container: str,
    project_id: str,
) -> None:
    """Save current browser cookies to per-project blob path.

    Accesses Playwright's storage_state through browser-use's BrowserSession.
    Does NOT overwrite global cookies — those are managed out-of-band.
    """
    # browser-use's BrowserSession wraps Playwright. Access the underlying
    # BrowserContext to get storage_state synchronously.
    try:
        context = getattr(browser_session, '_context', None)
        if context is None:
            # Try alternate access paths
            context = getattr(browser_session, 'context', None)
        if context is None:
            logger.warning("[CookieManager] Could not access browser context for cookie save")
            return

        # Playwright's storage_state() is sync when called without a path
        state = context.storage_state()
    except Exception as e:
        logger.warning(f"[CookieManager] Could not extract storage state: {e}")
        return

    if not state or (not state.get("cookies") and not state.get("origins")):
        logger.debug("[CookieManager] No cookies to save")
        return

    blob_path = f"projects/{project_id}/browser_cookies.json"
    _upload_blob_json(blob_service_client, container, blob_path, state)
