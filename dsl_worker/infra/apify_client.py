"""
Apify API client.

Three operations:
- search_actors(query) → find relevant scrapers in the Apify store
- run_actor(actor_id, input, timeout) → run a scraper and get results
- get_run_results(dataset_id) → get results from a completed run

Cost varies per actor. Most are pay-per-result or pay-per-compute-unit.
"""

import asyncio
import logging
from typing import Any, Awaitable, Callable, Dict, List, Optional

import httpx

logger = logging.getLogger(__name__)

BASE_URL = "https://api.apify.com/v2"


class ApifyClient:
    """Apify platform API client."""

    def __init__(self, api_key: str) -> None:
        self._api_key = api_key
        self._headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

    async def get_actor_details(self, actor_id: str) -> Optional[Dict[str, Any]]:
        """Get full details for an actor including description, readme, and input schema.

        Returns dict with: title, description, readme_summary, example_input,
        input_schema, url.

        The input_schema is fetched from the actor's latest build definition —
        this is the full JSON Schema with properties, types, descriptions,
        enums, defaults, and required flags.  Without it, LLMs cannot figure
        out what input to pass to apify_run.
        """
        actor_path = actor_id.replace("/", "~")
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.get(
                f"{BASE_URL}/acts/{actor_path}",
                headers=self._headers,
            )
        if resp.status_code != 200:
            return None

        d = resp.json().get("data", {})

        # Fetch the input schema from the actor's latest build.
        # The build definition contains actorDefinition.input — the full
        # JSON Schema that the Apify MCP server also uses.
        input_schema = await self._fetch_input_schema(d)
        pricing = self._extract_pricing(d.get("pricingInfos", []))
        stats = self._extract_stats(d.get("stats", {}))

        # NOTE: we deliberately omit `exampleRunInput` from the response.
        # Many actor publishers set it to garbage placeholders like
        # `{"helloWorld": 123}` which mislead the LLM into thinking the
        # actor "doesn't take real input" and bailing to web_search.
        # Real diagnosis 2026-04-29 on project 664dbc64: model's reasoning
        # explicitly said "the Apify call can't take input" after seeing
        # exactly that placeholder. The `input_schema` below is the
        # authoritative source of truth — it has property names, types,
        # descriptions, enums, and defaults. The agent should construct
        # input from that.
        return {
            "actor_id": actor_id,
            "title": d.get("title", ""),
            "description": d.get("description", ""),
            "readme_summary": d.get("readmeSummary", ""),
            "input_schema": input_schema,
            "pricing": pricing,
            "stats": stats,
            "url": f"https://apify.com/{actor_id}",
            "_input_hint": (
                "Construct the input object from `input_schema.properties`. "
                "Each property has a type, description, and often a default "
                "or enum of valid values. Required fields are listed in "
                "`input_schema.required`. Pass it as the `input` arg to "
                "`apify_call_actor`."
            ),
        }

    @staticmethod
    def _extract_stats(stats: Dict[str, Any]) -> Optional[str]:
        """Extract human-readable stats summary."""
        if not stats:
            return None
        parts = []
        total_runs = stats.get("totalRuns", 0)
        total_users = stats.get("totalUsers", 0)
        rating = stats.get("actorReviewRating", 0)
        reviews = stats.get("actorReviewCount", 0)
        if total_runs:
            parts.append(f"{total_runs:,} runs")
        if total_users:
            parts.append(f"{total_users:,} users")
        if rating and reviews:
            parts.append(f"{rating:.1f}/5 ({reviews} reviews)")
        run_stats = stats.get("publicActorRunStats30Days", {})
        succeeded = run_stats.get("SUCCEEDED", 0)
        total_30d = run_stats.get("TOTAL", 0)
        if total_30d > 0:
            parts.append(f"{succeeded}/{total_30d} succeeded last 30d")
        return ", ".join(parts) if parts else None

    @staticmethod
    def _extract_pricing(pricing_infos: List[Dict[str, Any]]) -> Optional[str]:
        """Extract human-readable pricing from the latest pricing entry."""
        if not pricing_infos:
            return None
        latest = pricing_infos[-1]
        model = latest.get("pricingModel", "")
        if model == "PAY_PER_EVENT":
            events = (
                latest.get("pricingPerEvent", {})
                .get("actorChargeEvents", {})
            )
            parts = []
            for event_info in events.values():
                title = event_info.get("eventTitle", "")
                price = event_info.get("eventPriceUsd", 0)
                if price:
                    parts.append(f"${price}/{ title}")
            return ", ".join(parts) if parts else None
        elif model == "FLAT_PRICE_PER_MONTH":
            price = latest.get("pricePerUnitUsd", 0)
            return f"${price}/month subscription"
        elif model == "PRICE_PER_DATASET_ITEM":
            price = latest.get("pricePerUnitUsd", 0)
            return f"${price}/result"
        return None

    async def _fetch_input_schema(
        self, actor_data: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Fetch the input JSON Schema from the actor's latest build.

        Tries taggedBuilds.latest first, then falls back to the most recent
        version's build.
        """
        # Strategy 1: taggedBuilds.latest → buildId → build details
        build_id = (
            actor_data.get("taggedBuilds", {})
            .get("latest", {})
            .get("buildId")
        )
        if build_id:
            schema = await self._get_build_input_schema(build_id)
            if schema:
                return schema

        # Strategy 2: walk versions to find one with a buildId
        for version in actor_data.get("versions", {}).get("items", []):
            vid = version.get("buildId")
            if vid:
                schema = await self._get_build_input_schema(vid)
                if schema:
                    return schema

        return None

    async def _get_build_input_schema(
        self, build_id: str
    ) -> Optional[Dict[str, Any]]:
        """Fetch a build by ID and extract its actorDefinition.input schema."""
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                resp = await client.get(
                    f"{BASE_URL}/actor-builds/{build_id}",
                    headers=self._headers,
                )
            if resp.status_code != 200:
                return None
            build_data = resp.json().get("data", {})
            return build_data.get("actorDefinition", {}).get("input")
        except Exception:
            logger.debug(f"[Apify] Failed to fetch build {build_id}")
            return None

    async def search_actors(
        self,
        query: str,
        limit: int = 5,
    ) -> List[Dict[str, Any]]:
        """Search the Apify store for actors matching a query.

        Returns list of actor summaries with: id, name, title, description,
        username, stats (runs, users), pricing info.
        """
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.get(
                f"{BASE_URL}/store",
                params={"search": query, "limit": limit},
                headers=self._headers,
            )
            data = resp.json()

        items = data.get("data", {}).get("items", [])
        results = []
        for item in items:
            stats = item.get("stats", {})
            results.append({
                "actor_id": f"{item.get('username')}/{item.get('name')}",
                "title": item.get("title", ""),
                "description": (item.get("description") or "")[:2000],
                "total_runs": stats.get("totalRuns", 0),
                "total_users": stats.get("totalUsers", 0),
                "url": f"https://apify.com/{item.get('username')}/{item.get('name')}",
            })

        return results

    async def run_actor(
        self,
        actor_id: str,
        run_input: Dict[str, Any],
        timeout: int = 300,
        max_items: Optional[int] = None,
        on_item: Optional[Callable[[Dict[str, Any]], Awaitable[None]]] = None,
    ) -> Dict[str, Any]:
        """Run an Apify actor and wait for completion.

        Args:
            actor_id: Actor ID like "neatrat/upwork-job-scraper"
            run_input: Input parameters for the actor
            timeout: Max seconds to wait for completion
            max_items: Max items to return from dataset (None = all)
            on_item: Optional async callback fired per item as the
                actor's dataset grows. When provided, the loop short-
                polls (3s) the run status AND fetches new dataset items
                by offset on each tick — streaming items to the caller
                as they're written. When None, uses the long-poll path
                (single GET hangs up to 60s on terminal status) which
                is faster for non-streaming callers.

        Returns dict with: status, items (list of result dicts),
        dataset_id, run_id, cost_usd. When on_item is set, items are
        also yielded incrementally to the callback during the run; the
        final list returned matches what was streamed.
        """
        # Normalize actor_id: slash → tilde for API
        actor_path = actor_id.replace("/", "~")

        # Start the run
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.post(
                f"{BASE_URL}/acts/{actor_path}/runs",
                headers=self._headers,
                json=run_input,
            )

        if resp.status_code not in (200, 201):
            error = resp.json().get("error", {}).get("message", resp.text[:200])
            return {"status": "FAILED", "items": [], "error": error}

        run_data = resp.json().get("data", {})
        run_id = run_data.get("id")
        dataset_id = run_data.get("defaultDatasetId")

        if not run_id:
            return {"status": "FAILED", "items": [], "error": "No run ID returned"}

        TERMINAL = ("SUCCEEDED", "FAILED", "ABORTED", "TIMED-OUT")
        start = asyncio.get_event_loop().time()
        status = "RUNNING"
        run_info: Dict[str, Any] = {}

        if on_item is None:
            # Non-streaming: long-poll for terminal status (fastest).
            WAIT_FOR_FINISH_SECS = 60
            while (asyncio.get_event_loop().time() - start) < timeout:
                wait_secs = min(
                    WAIT_FOR_FINISH_SECS,
                    max(1, int(timeout - (asyncio.get_event_loop().time() - start))),
                )
                try:
                    async with httpx.AsyncClient(timeout=wait_secs + 10.0) as client:
                        resp = await client.get(
                            f"{BASE_URL}/actor-runs/{run_id}",
                            headers=self._headers,
                            params={"waitForFinish": wait_secs},
                        )
                except httpx.RequestError:
                    await asyncio.sleep(1)
                    continue
                if resp.status_code != 200:
                    await asyncio.sleep(1)
                    continue
                run_info = resp.json().get("data", {})
                status = run_info.get("status", "RUNNING")
                if status in TERMINAL:
                    break
        else:
            # Streaming: short-poll (3s) status + drain new dataset
            # items by offset on each tick. Verified end-to-end against
            # apify in scripts/test_apify_streaming.py: items appear in
            # the dataset within seconds of the actor producing them,
            # so this is real streaming, not best-effort.
            POLL_INTERVAL = 3.0
            seen_count = 0
            streamed_items: List[Dict[str, Any]] = []
            cap = max_items if max_items is not None else None
            while (asyncio.get_event_loop().time() - start) < timeout:
                # Status check
                try:
                    async with httpx.AsyncClient(timeout=10.0) as client:
                        s_resp = await client.get(
                            f"{BASE_URL}/actor-runs/{run_id}",
                            headers=self._headers,
                        )
                    if s_resp.status_code == 200:
                        run_info = s_resp.json().get("data", {})
                        status = run_info.get("status", "RUNNING")
                except httpx.RequestError:
                    await asyncio.sleep(POLL_INTERVAL)
                    continue
                # New items beyond what we've drained
                try:
                    async with httpx.AsyncClient(timeout=15.0) as client:
                        i_resp = await client.get(
                            f"{BASE_URL}/datasets/{dataset_id}/items",
                            headers=self._headers,
                            params={
                                "offset": seen_count,
                                "limit": 1000,
                                "clean": 1,
                            },
                        )
                    if i_resp.status_code == 200:
                        new_items = i_resp.json() or []
                    else:
                        new_items = []
                except httpx.RequestError:
                    new_items = []
                for item in new_items:
                    if cap is not None and len(streamed_items) >= cap:
                        break
                    streamed_items.append(item)
                    seen_count += 1
                    try:
                        await on_item(item)
                    except Exception:
                        logger.exception("on_item callback raised")
                if status in TERMINAL:
                    break
                if cap is not None and len(streamed_items) >= cap:
                    # Hit max_items — abort the actor server-side so
                    # it doesn't keep running (and billing) after we
                    # have what we asked for. Caller-perspective success.
                    try:
                        async with httpx.AsyncClient(timeout=10.0) as client:
                            await client.post(
                                f"{BASE_URL}/actor-runs/{run_id}/abort",
                                headers=self._headers,
                            )
                    except Exception:
                        logger.exception("abort after max_items raised")
                    # Re-fetch run_info to get final cost after abort
                    try:
                        async with httpx.AsyncClient(timeout=10.0) as client:
                            s_resp = await client.get(
                                f"{BASE_URL}/actor-runs/{run_id}",
                                headers=self._headers,
                            )
                        if s_resp.status_code == 200:
                            run_info = s_resp.json().get("data", run_info)
                    except Exception:
                        pass
                    status = "SUCCEEDED"  # caller got their N items
                    break
                await asyncio.sleep(POLL_INTERVAL)

            cost_usd = float(run_info.get("usageTotalUsd") or 0.0)
            if status == "SUCCEEDED":
                return {
                    "status": "SUCCEEDED",
                    "items": streamed_items,
                    "run_id": run_id,
                    "dataset_id": dataset_id,
                    "cost_usd": cost_usd,
                }
            # Terminal-but-failed (FAILED/ABORTED-by-server/TIMED-OUT)
            return {
                "status": status,
                "items": streamed_items,
                "run_id": run_id,
                "dataset_id": dataset_id,
                "cost_usd": cost_usd,
                "error": f"Actor run ended with status: {status}",
            }

        if status != "SUCCEEDED":
            return {
                "status": status,
                "items": [],
                "run_id": run_id,
                "dataset_id": dataset_id,
                "error": f"Actor run ended with status: {status}",
            }

        # Fetch results
        items = await self.get_dataset_items(dataset_id, limit=max_items)

        # Cost is already on the run_info we just fetched — no extra
        # round-trip needed.
        cost_usd = float(run_info.get("usageTotalUsd") or 0.0)

        return {
            "status": "SUCCEEDED",
            "items": items,
            "run_id": run_id,
            "dataset_id": dataset_id,
            "cost_usd": cost_usd,
        }

    async def get_dataset_items(
        self,
        dataset_id: str,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Get items from an Apify dataset.

        Args:
            dataset_id: Dataset ID from a completed actor run
            limit: Max items to fetch (None = all, up to 10000)

        Returns list of result dicts.
        """
        params = {}
        if limit:
            params["limit"] = limit

        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.get(
                f"{BASE_URL}/datasets/{dataset_id}/items",
                params=params,
                headers=self._headers,
            )

        if resp.status_code != 200:
            logger.warning(f"[Apify] Failed to fetch dataset {dataset_id}: {resp.status_code}")
            return []

        return resp.json() if isinstance(resp.json(), list) else []

    async def close(self) -> None:
        """No persistent connections to close."""
        pass
