"""apify_actor — generic actor source.

Source name format: "apify_actor:<actor_id>" where actor_id is e.g.
"clearpath/reddit-search-scraper". The adapter resolves the colon-suffix,
fetches the actor's input_schema (auto-fills plumbing fields like
proxy), runs the actor, and returns rows.

Two run modes:

  fetch(...)        — blocking, single call, returns all rows when actor
                      completes. Used by table_extend / one-shot reads.

  fetch_stream(...) — async-generator: yields batches of rows as they
                      arrive in the actor's dataset. Used by table_create
                      so the UI sees rows as soon as the actor produces
                      them, not minutes later when it finishes.
"""

from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path
from typing import Any, AsyncIterator, Dict, List, Optional

import httpx

from dsl_worker.sources_v2.base import FetchResult, SourceAdapter, register


log = logging.getLogger(__name__)

APIFY_BASE = "https://api.apify.com/v2"
APIFY_PROXY_KEYS = ("proxy", "proxyConfiguration")


class ApifyActorAdapter(SourceAdapter):
    name = "apify_actor"
    predictable = False  # output schema is actor-specific

    def __init__(self) -> None:
        self.api_key = os.getenv("APIFY_API_KEY")
        if not self.api_key:
            log.warning("APIFY_API_KEY not set — apify_actor adapter inert")

    async def _get_input_schema(self, client: httpx.AsyncClient, actor_id: str) -> Optional[Dict[str, Any]]:
        """Fetch latest build's input schema. Cheap, used for plumbing autofill."""
        aid = actor_id.replace("/", "~")
        try:
            r = await client.get(
                f"{APIFY_BASE}/acts/{aid}",
                params={"token": self.api_key},
                timeout=15,
            )
            if r.status_code != 200:
                return None
            data = r.json().get("data") or {}
            builds = data.get("taggedBuilds") or {}
            latest = builds.get("latest") or {}
            bid = latest.get("buildId")
            if not bid:
                return None
            rb = await client.get(f"{APIFY_BASE}/actor-builds/{bid}", params={"token": self.api_key}, timeout=15)
            if rb.status_code != 200:
                return None
            build = rb.json().get("data") or {}
            schema_str = build.get("inputSchema")
            if not schema_str:
                return None
            import json
            return json.loads(schema_str) if isinstance(schema_str, str) else schema_str
        except Exception as e:
            log.warning("apify_actor schema fetch failed: %s", e)
            return None

    @staticmethod
    def _autofill_plumbing(actor_input: Dict[str, Any], input_schema: Dict[str, Any]) -> None:
        """Server-side autofill of proxy / proxyConfiguration so the agent
        doesn't have to think about it. Mirrors the v1 fix shipped earlier."""
        props = (input_schema or {}).get("properties") or {}
        for key in APIFY_PROXY_KEYS:
            prop = props.get(key) or {}
            if isinstance(prop, dict) and prop.get("type") == "object" and key not in actor_input:
                actor_input[key] = {"useApifyProxy": True}
        # required fields with declared defaults
        required = (input_schema or {}).get("required") or []
        for field in required:
            if field in actor_input:
                continue
            prop = props.get(field) or {}
            if isinstance(prop, dict) and "default" in prop:
                actor_input[field] = prop["default"]

    async def fetch(
        self,
        query_params: Dict[str, Any],
        n: int,
        prior_cursor: Optional[Dict[str, Any]] = None,
        source_full: str = "",
    ) -> FetchResult:
        """source_full is the colon-suffix form 'apify_actor:<actor_id>'."""
        if not self.api_key:
            return FetchResult(rows=[], schema=[], cost_credits=0.0, exhausted=True)

        actor_id = source_full.split(":", 1)[1] if ":" in source_full else query_params.get("actor_id", "")
        actor_input = dict(query_params.get("input") or {})

        if not actor_id:
            return FetchResult(
                rows=[], schema=[], cost_credits=0.0, exhausted=True,
            )

        # Honor n as a cap on the actor's output
        max_items = int(query_params.get("maxItems") or n)
        actor_input.setdefault("maxItems", max_items)

        async with httpx.AsyncClient(timeout=180.0) as client:
            schema = await self._get_input_schema(client, actor_id)
            if schema:
                self._autofill_plumbing(actor_input, schema)

            aid = actor_id.replace("/", "~")
            resp = await client.post(
                f"{APIFY_BASE}/acts/{aid}/run-sync-get-dataset-items",
                params={"token": self.api_key, "format": "json"},
                json=actor_input,
                timeout=300.0,
            )
            if resp.status_code >= 400:
                log.warning("apify_actor HTTP %s for %s: %s", resp.status_code, actor_id, resp.text[:200])
                return FetchResult(rows=[], schema=[], cost_credits=0.0, exhausted=True)

            try:
                rows = resp.json()
            except Exception:
                rows = []
            if not isinstance(rows, list):
                rows = []

        # Cost: best estimate from actor metadata; not always available at fetch
        # time. Caller updates tables.last_fetch_cost_credits from balance ledger
        # if we hook up cost accounting. For preview-card empirics, 0 here is OK
        # — the next-fetch estimate uses the actual recorded value.
        cost = 0.0

        # Schema: union of keys across returned rows.
        schema_keys = sorted({k for r in rows for k in r.keys()}) if rows else []

        return FetchResult(
            rows=rows[:max_items],
            schema=schema_keys[:60],
            cost_credits=cost,
            exhausted=len(rows) < max_items,
            cursor=None,  # date-window pagination is agent-driven via query_params
            dedup_key_column_hint="id" if "id" in schema_keys else ("url" if "url" in schema_keys else None),
        )


    async def fetch_stream(
        self,
        query_params: Dict[str, Any],
        n: int,
        source_full: str = "",
        poll_interval: float = 2.5,
        first_batch_min: int = 3,
        first_batch_timeout: float = 25.0,
    ) -> AsyncIterator[Dict[str, Any]]:
        """Stream actor output in batches as they appear in the dataset.

        Yields dicts of shape:
          {"rows": [...], "exhausted": bool, "cost_credits": float}

        The first yield happens as soon as `first_batch_min` rows are
        available or `first_batch_timeout` elapses, whichever is first —
        so the caller can show the user something within seconds. Later
        yields fire each poll interval with whatever new rows showed up.
        Final yield has exhausted=True.
        """
        if not self.api_key:
            yield {"rows": [], "exhausted": True, "cost_credits": 0.0}
            return

        actor_id = source_full.split(":", 1)[1] if ":" in source_full else query_params.get("actor_id", "")
        if not actor_id:
            yield {"rows": [], "exhausted": True, "cost_credits": 0.0}
            return

        actor_input = dict(query_params.get("input") or {})
        max_items = int(query_params.get("maxItems") or n)
        actor_input.setdefault("maxItems", max_items)

        aid = actor_id.replace("/", "~")
        async with httpx.AsyncClient(timeout=60.0) as client:
            schema = await self._get_input_schema(client, actor_id)
            if schema:
                self._autofill_plumbing(actor_input, schema)

            # Start the run async.
            start = await client.post(
                f"{APIFY_BASE}/acts/{aid}/runs",
                params={"token": self.api_key},
                json=actor_input,
            )
            if start.status_code >= 400:
                log.warning("apify_actor start HTTP %s for %s: %s", start.status_code, actor_id, start.text[:200])
                yield {"rows": [], "exhausted": True, "cost_credits": 0.0}
                return
            data = (start.json() or {}).get("data") or {}
            run_id = data.get("id")
            dataset_id = data.get("defaultDatasetId")
            if not run_id or not dataset_id:
                yield {"rows": [], "exhausted": True, "cost_credits": 0.0}
                return

            yielded = 0
            first_yield_done = False
            t0 = asyncio.get_event_loop().time()
            terminal = {"SUCCEEDED", "FAILED", "ABORTED", "TIMED-OUT"}

            while True:
                # Pull whatever rows are available beyond what we've yielded.
                items_resp = await client.get(
                    f"{APIFY_BASE}/datasets/{dataset_id}/items",
                    params={
                        "token": self.api_key,
                        "format": "json",
                        "offset": yielded,
                        "limit": max_items - yielded,
                    },
                )
                new_rows: List[Dict[str, Any]] = []
                if items_resp.status_code == 200:
                    try:
                        body = items_resp.json()
                        if isinstance(body, list):
                            new_rows = [r for r in body if isinstance(r, dict)]
                    except Exception:
                        pass

                # Check run status.
                run_resp = await client.get(
                    f"{APIFY_BASE}/actor-runs/{run_id}",
                    params={"token": self.api_key},
                )
                run_status = "RUNNING"
                if run_resp.status_code == 200:
                    run_status = (run_resp.json().get("data") or {}).get("status") or "RUNNING"
                is_terminal = run_status in terminal

                # Time gate for the first yield: don't sit silent forever
                # if the actor is slow to produce its first item.
                elapsed = asyncio.get_event_loop().time() - t0
                should_first_yield = (
                    not first_yield_done
                    and (len(new_rows) >= first_batch_min or is_terminal or elapsed >= first_batch_timeout)
                )

                if new_rows and (first_yield_done or should_first_yield):
                    yielded += len(new_rows)
                    first_yield_done = True
                    yield {
                        "rows": new_rows,
                        "exhausted": is_terminal and yielded >= (max_items if max_items else yielded),
                        "cost_credits": 0.0,
                    }
                elif should_first_yield and not new_rows:
                    # Time-out first yield with empty rows so caller isn't
                    # blocked forever on a slow / no-result actor.
                    first_yield_done = True
                    yield {"rows": [], "exhausted": is_terminal, "cost_credits": 0.0}

                if is_terminal:
                    # Drain any final rows that landed between status checks.
                    drain_resp = await client.get(
                        f"{APIFY_BASE}/datasets/{dataset_id}/items",
                        params={
                            "token": self.api_key,
                            "format": "json",
                            "offset": yielded,
                            "limit": max(0, max_items - yielded),
                        },
                    )
                    final_rows: List[Dict[str, Any]] = []
                    if drain_resp.status_code == 200:
                        try:
                            body = drain_resp.json()
                            if isinstance(body, list):
                                final_rows = [r for r in body if isinstance(r, dict)]
                        except Exception:
                            pass
                    yield {"rows": final_rows, "exhausted": True, "cost_credits": 0.0}
                    return

                if yielded >= max_items:
                    yield {"rows": [], "exhausted": True, "cost_credits": 0.0}
                    return

                await asyncio.sleep(poll_interval)


register(ApifyActorAdapter())
