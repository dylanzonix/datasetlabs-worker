"""browser_use — bounded single-session scrape via Browser Use Cloud.

One session per call. Bounded by the `task` prompt scope, NOT item count
(the user feedback: 'don't cap by items; cap by what you ask BU to do').

Unpredictable source: row shape depends on the extraction task. Agent inspects
preview + calls column_map_set.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from urllib.parse import urlparse

from dsl_worker.sources.base import FetchResult, SourceAdapter, SourceDescription, register


log = logging.getLogger(__name__)


class BrowserUseAdapter(SourceAdapter):
    name = "browser_use"
    label = "Browser Session"
    favicon_url = None  # derived from the target URL at describe time
    predictable = False

    def describe(
        self,
        query_params: Dict[str, Any],
        source: Optional[str] = None,
    ) -> SourceDescription:
        qp = query_params or {}
        url = str(qp.get("url") or "")
        task = str(qp.get("task") or "")
        cand = str(qp.get("candidate_description") or "")
        domain = ""
        favicon = None
        if url:
            try:
                domain = urlparse(url).hostname or ""
                if domain:
                    favicon = f"https://www.google.com/s2/favicons?domain={domain}&sz=32"
            except Exception:
                pass
        headline = f"Extract from {domain}" if domain else "Browser-session extract"
        details_parts = [f"**URL:** {url}"] if url else []
        if task:
            details_parts.append(f"**Task:**\n\n{task}")
        if cand:
            details_parts.append(f"**Row shape:** {cand}")
        return SourceDescription(
            kind=self.name,
            label=self.label,
            query_text=headline,
            details="\n\n".join(details_parts),
            favicon_url=favicon,
        )

    def validate_query_params(self, query_params: Dict[str, Any]) -> Optional[str]:
        if "url" not in query_params:
            return "browser_use requires `url`"
        if "task" not in query_params:
            return "browser_use requires `task` (verbose, vertical, one site)"
        return None

    async def fetch(
        self,
        query_params: Dict[str, Any],
        n: int,
        prior_cursor: Optional[Dict[str, Any]] = None,
    ) -> FetchResult:
        try:
            from dsl_worker.infra.bu_client import bu_extract_rows
        except ImportError:
            log.warning("browser_use infra not available")
            return FetchResult(rows=[], schema=[], cost_credits=0.0, exhausted=True)

        try:
            rows, cost_usd = await bu_extract_rows(
                url=query_params["url"],
                task=query_params["task"],
                candidate_description=query_params.get("candidate_description", ""),
            )
        except Exception as e:
            log.exception("browser_use session failed: %s", e)
            return FetchResult(rows=[], schema=[], cost_credits=0.0, exhausted=True)

        schema_keys = sorted({k for r in rows for k in r.keys()}) if rows else []
        # bu_extract_rows returns the BU SDK's real USD session cost.
        # Convert to credits at 1 credit = $0.10 of compute.
        return FetchResult(
            rows=rows,
            schema=schema_keys,
            cost_credits=cost_usd * 10.0,
            exhausted=True,  # one-shot per session
            cursor=None,
            dedup_key_column_hint="url" if "url" in schema_keys else None,
        )


register(BrowserUseAdapter())
