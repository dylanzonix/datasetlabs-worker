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
from typing import Any, AsyncIterator, Callable, Dict, List, Optional

import httpx

from dsl_worker.sources.base import FetchResult, SourceAdapter, SourceDescription, register


log = logging.getLogger(__name__)

APIFY_BASE = "https://api.apify.com/v2"
APIFY_PROXY_KEYS = ("proxy", "proxyConfiguration")


def _apify_run_cost_usd_from_data(data: Dict[str, Any]) -> float:
    """Compute real USD cost of a finished apify run.

    Apify exposes `data.usageTotalUsd` directly but it's 0 for free /
    community actors. `data.stats.computeUnits` is always populated and
    is the actual resource we pay apify for. Convert at the platform's
    CU rate (env-configurable, default $0.40 / CU which matches apify's
    starter plan).

    Note: this misses proxy and dataset-write surcharges for some actors.
    Good enough for v1; can be sharpened to sum `usage.*USD` keys when
    apify populates them.
    """
    usd = data.get("usageTotalUsd")
    if usd is not None and float(usd) > 0:
        return float(usd)
    stats = data.get("stats") or {}
    cu = float(stats.get("computeUnits") or 0.0)
    rate = float(os.getenv("APIFY_USD_PER_CU", "0.40"))
    return cu * rate


async def _abort_apify_run_and_get_cost(
    client: httpx.AsyncClient,
    api_key: str,
    run_id: str,
) -> float:
    """Abort a running apify actor + fetch its final usage cost.

    Called from the CancelledError path of fetch/fetch_stream so the
    actor stops consuming compute units (we pay per-CU). Apify bills
    for whatever was used BEFORE the abort lands — we capture that and
    report it back via the on_cost callback so the chat turn's ledger
    entry includes the spend. Without this, a user cancel during a
    minute-long Apify run would abandon the cost (the agent's
    CancelledError catch only sees what was already returned).

    Returns the USD billed by apify; 0.0 on any failure (abort already
    fired, run already terminal, network blip during the GET).
    """
    try:
        # Best-effort abort. 4xx is fine — typically means already
        # terminal. The follow-up GET still returns the final cost.
        await client.post(
            f"{APIFY_BASE}/actor-runs/{run_id}/abort",
            params={"token": api_key},
            timeout=10.0,
        )
    except Exception:
        log.debug("apify abort POST failed for run %s; continuing to fetch cost", run_id, exc_info=True)
    try:
        final = await client.get(
            f"{APIFY_BASE}/actor-runs/{run_id}",
            params={"token": api_key},
            timeout=10.0,
        )
        if final.status_code == 200:
            rd = (final.json() or {}).get("data") or {}
            return _apify_run_cost_usd_from_data(rd)
    except Exception:
        log.debug("apify cost GET failed for run %s after abort", run_id, exc_info=True)
    return 0.0


class ApifyActorAdapter(SourceAdapter):
    name = "apify_actor"
    label = "Apify"
    favicon_url = "https://www.google.com/s2/favicons?domain=apify.com&sz=32"
    predictable = False  # output schema is actor-specific

    def __init__(self) -> None:
        self.api_key = os.getenv("APIFY_API_KEY")
        if not self.api_key:
            log.warning("APIFY_API_KEY not set — apify_actor adapter inert")

    def describe(
        self,
        query_params: Dict[str, Any],
        source: Optional[str] = None,
    ) -> SourceDescription:
        actor_id = ""
        if source and ":" in source:
            actor_id = source.split(":", 1)[1]
        actor_id = actor_id or (query_params or {}).get("actor_id", "")
        label = f"Apify — {actor_id}" if actor_id else "Apify actor"
        # Headline = a one-liner pulled from common input keys (search query,
        # URLs, subreddit, etc). Falls back to the actor id.
        actor_input = (query_params or {}).get("input") or {}
        headline_keys = ("query", "search", "searches", "searchQuery", "queries", "startUrls", "subreddits", "subreddit", "domain", "url", "username", "usernames")
        headline_val: Optional[str] = None
        for k in headline_keys:
            v = actor_input.get(k)
            if v:
                if isinstance(v, list):
                    headline_val = ", ".join(str(x) for x in v[:3])
                else:
                    headline_val = str(v)
                break
        headline = f"{actor_id}: {headline_val}" if headline_val else (actor_id or "Apify actor run")

        # Details: full actor input (truncated values for readability).
        detail_lines: List[str] = []
        for k, v in (actor_input or {}).items():
            if isinstance(v, (list, dict)):
                import json
                s = json.dumps(v, separators=(", ", ": "))
            else:
                s = str(v)
            if len(s) > 200:
                s = s[:200] + "…"
            detail_lines.append(f"- **{k}:** {s}")
        if "maxItems" in (query_params or {}) and "maxItems" not in (actor_input or {}):
            detail_lines.append(f"- **maxItems:** {query_params['maxItems']}")
        return SourceDescription(
            kind=self.name,
            label=label,
            query_text=headline,
            details="\n".join(detail_lines),
            favicon_url=self.favicon_url,
        )

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
        on_cost: Optional[Callable[[float], None]] = None,
    ) -> FetchResult:
        """source_full is the colon-suffix form 'apify_actor:<actor_id>'.

        on_cost: optional callback invoked with the actor run's billed
        cost in USD whenever it's captured (including the cancel path).
        Lets the agent loop's CancelledError handler attribute the
        spend to the turn's balance ledger even if this function never
        returns its FetchResult."""
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
            # POST /runs starts the actor async, returning the run object
            # immediately with id + defaultDatasetId. We poll the run until
            # terminal, then fetch items + read usageTotalUsd. Two-step
            # pattern (mirror of fetch_stream) — the only way to get cost.
            start = await client.post(
                f"{APIFY_BASE}/acts/{aid}/runs",
                params={"token": self.api_key},
                json=actor_input,
            )
            if start.status_code >= 400:
                log.warning("apify_actor start HTTP %s for %s: %s", start.status_code, actor_id, start.text[:200])
                return FetchResult(rows=[], schema=[], cost_credits=0.0, exhausted=True)
            run_data = (start.json() or {}).get("data") or {}
            run_id = run_data.get("id")
            dataset_id = run_data.get("defaultDatasetId")
            if not (run_id and dataset_id):
                log.warning("apify_actor: no run id/dataset id from start for %s", actor_id)
                return FetchResult(rows=[], schema=[], cost_credits=0.0, exhausted=True)

            terminal = {"SUCCEEDED", "FAILED", "ABORTED", "TIMED-OUT"}
            t0 = asyncio.get_event_loop().time()
            cost_usd = 0.0
            try:
                while True:
                    if asyncio.get_event_loop().time() - t0 > 280:
                        log.warning("apify_actor poll timeout for run %s", run_id)
                        break
                    run_resp = await client.get(
                        f"{APIFY_BASE}/actor-runs/{run_id}",
                        params={"token": self.api_key},
                    )
                    if run_resp.status_code != 200:
                        await asyncio.sleep(2)
                        continue
                    rd = (run_resp.json() or {}).get("data") or {}
                    if rd.get("status") in terminal:
                        # Apify populates usageTotalUsd a beat AFTER
                        # status flips. Retry briefly until it lands.
                        cost_usd = _apify_run_cost_usd_from_data(rd)
                        for cost_attempt in range(5):
                            if cost_usd > 0:
                                break
                            await asyncio.sleep(0.5 * (cost_attempt + 1))
                            retry_resp = await client.get(
                                f"{APIFY_BASE}/actor-runs/{run_id}",
                                params={"token": self.api_key},
                            )
                            if retry_resp.status_code == 200:
                                rd2 = (retry_resp.json() or {}).get("data") or {}
                                cost_usd = _apify_run_cost_usd_from_data(rd2)
                        break
                    await asyncio.sleep(2.5)
            except asyncio.CancelledError:
                # User cancelled mid-poll. Abort the actor (stops
                # further compute-unit spend) + read whatever was
                # already billed; surface that cost up through on_cost
                # so the run's ledger entry is correct. asyncio.shield
                # ensures the abort completes even though we're being
                # cancelled — without it, the await inside abort would
                # immediately re-raise CancelledError and the actor
                # would keep running on Apify's side.
                try:
                    aborted_usd = await asyncio.shield(
                        _abort_apify_run_and_get_cost(client, self.api_key, run_id)
                    )
                    if on_cost and aborted_usd > 0:
                        on_cost(aborted_usd * 10.0)
                except Exception:
                    log.exception("apify abort-on-cancel cleanup failed for run %s", run_id)
                raise

            rows: List[Dict[str, Any]] = []
            items_resp = await client.get(
                f"{APIFY_BASE}/datasets/{dataset_id}/items",
                params={"token": self.api_key, "format": "json", "limit": max_items},
                timeout=60.0,
            )
            if items_resp.status_code == 200:
                try:
                    body = items_resp.json()
                    if isinstance(body, list):
                        rows = [r for r in body if isinstance(r, dict)]
                except Exception:
                    pass

        # Convention: 1 credit covers $0.10 of compute. Convert real USD → credits.
        cost = cost_usd * 10.0
        if on_cost and cost > 0:
            on_cost(cost)

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
        first_batch_timeout: float = 60.0,
        on_cost: Optional[Callable[[float], None]] = None,
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
            # Surface this loudly — silently returning 0 rows makes it look
            # like every Apify actor is broken when really we just don't
            # have credentials. The startup warning at __init__ is easy to
            # miss; this fires per-call so a bad env config doesn't hide.
            log.error("apify_actor: APIFY_API_KEY not set — call to %r returning 0 rows", source_full)
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

            try:
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
                    # Fetch real run cost from apify's API and bill it.
                    # Apify populates `usageTotalUsd` asynchronously AFTER
                    # status flips to SUCCEEDED — querying immediately
                    # often returns 0 / None. Retry with short backoff
                    # until it lands. Without this every PAY_PER_EVENT
                    # actor (harvestapi, etc.) billed $0 to the user
                    # because we asked too early.
                    cost_usd = 0.0
                    for attempt in range(6):
                        final_run_resp = await client.get(
                            f"{APIFY_BASE}/actor-runs/{run_id}",
                            params={"token": self.api_key},
                        )
                        if final_run_resp.status_code == 200:
                            rd = (final_run_resp.json() or {}).get("data") or {}
                            cost_usd = _apify_run_cost_usd_from_data(rd)
                            if cost_usd > 0:
                                break
                        if attempt < 5:
                            await asyncio.sleep(0.5 * (attempt + 1))
                    yield {
                        "rows": final_rows,
                        "exhausted": True,
                        "cost_credits": cost_usd * 10.0,
                    }
                    return

                if yielded >= max_items:
                    yield {"rows": [], "exhausted": True, "cost_credits": 0.0}
                    return

                await asyncio.sleep(poll_interval)
            except asyncio.CancelledError:
                # User cancelled mid-poll. Mirror the abort + cost
                # capture from fetch(): stop the actor on Apify's side
                # (no more billable CU) and report whatever it already
                # cost via on_cost so the run's ledger entry stays
                # accurate. asyncio.shield protects the cleanup from
                # the cancel that's already propagating.
                try:
                    aborted_usd = await asyncio.shield(
                        _abort_apify_run_and_get_cost(client, self.api_key, run_id)
                    )
                    if on_cost and aborted_usd > 0:
                        on_cost(aborted_usd * 10.0)
                except Exception:
                    log.exception("apify abort-on-cancel cleanup failed for run %s", run_id)
                raise


register(ApifyActorAdapter())
