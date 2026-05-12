"""web_harvest — bounded research subagent on a topic.

One subagent invocation per call. Wraps the existing web_harvest implementation
in dsl_worker.infra.research_tools. Returns candidates as rows.

Unpredictable source: the row schema depends on `candidate_description`. Agent
inspects the preview, calls column_map_set.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from dsl_worker.sources_v2.base import FetchResult, SourceAdapter, register


log = logging.getLogger(__name__)


class WebHarvestAdapter(SourceAdapter):
    name = "web_harvest"
    predictable = False

    def validate_query_params(self, query_params: Dict[str, Any]) -> Optional[str]:
        required = {"query", "candidate_description"}
        missing = required - set(query_params)
        if missing:
            return f"web_harvest requires {sorted(required)}; missing: {sorted(missing)}"
        return None

    async def fetch(
        self,
        query_params: Dict[str, Any],
        n: int,
        prior_cursor: Optional[Dict[str, Any]] = None,
    ) -> FetchResult:
        # Lean on existing infra. The legacy web_harvest module lives in
        # dsl_worker.infra and is used by the V13/chat-v1 paths today; we
        # call its async entrypoint with the new param shape.
        try:
            from dsl_worker.infra.research_tools import web_harvest_run
        except ImportError:
            log.warning("web_harvest infra not available")
            return FetchResult(rows=[], schema=[], cost_credits=0.0, exhausted=True)

        query = query_params["query"]
        candidate_description = query_params["candidate_description"]
        max_candidates = int(query_params.get("max_candidates", min(n, 30)))
        max_turns = int(query_params.get("max_turns", 6))

        # Continuation: agent supplies a `continuation_hint` in query_params on
        # subsequent calls. The subagent uses it to bias toward a non-overlapping
        # angle. Stored on prior_cursor for visibility but not used by adapter
        # — agent passes via query_params.
        try:
            rows, cost = await web_harvest_run(
                query=query,
                candidate_description=candidate_description,
                max_candidates=max_candidates,
                max_turns=max_turns,
            )
        except Exception as e:
            log.exception("web_harvest run failed: %s", e)
            return FetchResult(rows=[], schema=[], cost_credits=0.0, exhausted=True)

        schema_keys = sorted({k for r in rows for k in r.keys()}) if rows else []
        return FetchResult(
            rows=rows,
            schema=schema_keys,
            cost_credits=cost,
            exhausted=True,  # one-shot; agent passes a fresh continuation_hint for more
            cursor=None,
            dedup_key_column_hint="url" if "url" in schema_keys else None,
        )


register(WebHarvestAdapter())
