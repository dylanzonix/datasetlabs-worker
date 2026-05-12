"""browser_use — bounded single-session scrape via Browser Use Cloud.

One session per call. Bounded by the `task` prompt scope, NOT item count
(the user feedback: 'don't cap by items; cap by what you ask BU to do').

Unpredictable source: row shape depends on the extraction task. Agent inspects
preview + calls column_map_set.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from dsl_worker.sources_v2.base import FetchResult, SourceAdapter, register


log = logging.getLogger(__name__)


class BrowserUseAdapter(SourceAdapter):
    name = "browser_use"
    predictable = False

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
            rows, cost = await bu_extract_rows(
                url=query_params["url"],
                task=query_params["task"],
                candidate_description=query_params.get("candidate_description", ""),
            )
        except Exception as e:
            log.exception("browser_use session failed: %s", e)
            return FetchResult(rows=[], schema=[], cost_credits=0.0, exhausted=True)

        schema_keys = sorted({k for r in rows for k in r.keys()}) if rows else []
        return FetchResult(
            rows=rows,
            schema=schema_keys,
            cost_credits=cost,
            exhausted=True,  # one-shot per session
            cursor=None,
            dedup_key_column_hint="url" if "url" in schema_keys else None,
        )


register(BrowserUseAdapter())
