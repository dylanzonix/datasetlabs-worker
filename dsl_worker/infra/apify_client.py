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
from typing import Any, Dict, List, Optional

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
        # JSON Schema that the Apify MCP server also uses. We also pull
        # `storages.dataset.views` when available (output column hints).
        input_schema, output_views = await self._fetch_build_schemas(d)
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
            "output_views": output_views,
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
    def _extract_pricing(pricing_info: Any) -> Optional[str]:
        """Extract human-readable pricing.

        Accepts either the full `pricingInfos` list (we use the last entry)
        or a single `currentPricingInfo` dict (as returned by the store
        search endpoint). PAY_PER_EVENT actors can charge multiple events
        with either flat (`eventPriceUsd`) or tiered (`eventTieredPricingUsd`)
        prices — we render all of them, mark the primary event, and use the
        FREE tier price (worst case for a no-plan caller) as the headline.
        """
        if not pricing_info:
            return None
        if isinstance(pricing_info, list):
            latest = pricing_info[-1] if pricing_info else None
        else:
            latest = pricing_info
        if not latest:
            return None

        model = latest.get("pricingModel", "")
        if model == "PAY_PER_EVENT":
            events = (
                latest.get("pricingPerEvent", {})
                .get("actorChargeEvents", {})
            )
            parts = []
            primary_part = None
            for event_info in events.values():
                title = event_info.get("eventTitle", "") or "event"
                tiered = event_info.get("eventTieredPricingUsd")
                if tiered:
                    free = tiered.get("FREE", {}).get("tieredEventPriceUsd")
                    price = free if free is not None else next(
                        (t.get("tieredEventPriceUsd") for t in tiered.values()
                         if t.get("tieredEventPriceUsd") is not None),
                        None,
                    )
                else:
                    price = event_info.get("eventPriceUsd")
                if price is None:
                    continue
                # Express per-result events as $/1k for readability
                if "result" in title.lower() and price < 0.1:
                    rate = f"${price * 1000:.2f}/1k {title.lower()}"
                else:
                    rate = f"${price}/{title}"
                if event_info.get("isPrimaryEvent"):
                    primary_part = rate + " (primary)"
                else:
                    parts.append(rate)
            if primary_part:
                parts.insert(0, primary_part)
            return ", ".join(parts) if parts else None
        elif model == "FLAT_PRICE_PER_MONTH":
            price = latest.get("pricePerUnitUsd", 0)
            return f"${price}/month subscription"
        elif model == "PRICE_PER_DATASET_ITEM":
            price = latest.get("pricePerUnitUsd", 0)
            return f"${price}/result"
        elif model == "FREE":
            return "Free (compute units only)"
        return None

    async def _fetch_build_schemas(
        self, actor_data: Dict[str, Any]
    ) -> tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
        """Fetch input schema and output views from the actor's latest build.

        Returns (input_schema, output_views). Tries taggedBuilds.latest
        first, then falls back to the most recent version's build.
        """
        candidates: List[str] = []
        tagged = actor_data.get("taggedBuilds") or {}
        latest = (tagged.get("latest") or {}).get("buildId") if isinstance(tagged, dict) else None
        if latest:
            candidates.append(latest)
        versions = actor_data.get("versions") or {}
        version_items = versions.get("items") if isinstance(versions, dict) else versions
        for version in version_items or []:
            if not isinstance(version, dict):
                continue
            vid = version.get("buildId")
            if vid and vid not in candidates:
                candidates.append(vid)

        for build_id in candidates:
            input_schema, views = await self._get_build_schemas(build_id)
            if input_schema or views:
                return input_schema, views
        return None, None

    async def _get_build_schemas(
        self, build_id: str
    ) -> tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
        """Fetch a build and pull both input schema and dataset views."""
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                resp = await client.get(
                    f"{BASE_URL}/actor-builds/{build_id}",
                    headers=self._headers,
                )
            if resp.status_code != 200:
                return None, None
            actor_def = resp.json().get("data", {}).get("actorDefinition", {})
            if not isinstance(actor_def, dict):
                return None, None
            input_schema = actor_def.get("input") if isinstance(actor_def.get("input"), dict) else None
            storages = actor_def.get("storages") or {}
            dataset = storages.get("dataset") if isinstance(storages, dict) else None
            views = dataset.get("views") if isinstance(dataset, dict) else None
            if not isinstance(views, dict):
                views = None
            return input_schema, views
        except Exception:
            logger.debug(f"[Apify] Failed to fetch build {build_id}")
            return None, None

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
                "rating": item.get("actorReviewRating"),
                "review_count": item.get("actorReviewCount", 0),
                "creator": item.get("userFullName") or item.get("username", ""),
                "pricing": self._extract_pricing(item.get("currentPricingInfo")),
                "url": f"https://apify.com/{item.get('username')}/{item.get('name')}",
            })

        return results

    async def run_actor(
        self,
        actor_id: str,
        run_input: Dict[str, Any],
        timeout: int = 300,
        max_items: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Run an Apify actor and wait for completion.

        Args:
            actor_id: Actor ID like "neatrat/upwork-job-scraper"
            run_input: Input parameters for the actor
            timeout: Max seconds to wait for completion
            max_items: Max items to return from dataset (None = all)

        Returns dict with: status, items (list of result dicts),
        dataset_id, run_id, cost_usd.
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

        # Poll using Apify's `waitForFinish=N` long-poll: the GET hangs
        # up to N seconds and returns early as soon as the run reaches a
        # terminal status. Way better than fixed-interval polling — we
        # see the finish within ~1s of when it happens. We loop in case
        # the user-requested `timeout` is longer than the per-call cap
        # (Apify caps waitForFinish at 60s).
        WAIT_FOR_FINISH_SECS = 60
        start = asyncio.get_event_loop().time()
        status = "RUNNING"
        run_info: Dict[str, Any] = {}

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
                # Network blip — short backoff, then retry the long-poll.
                await asyncio.sleep(1)
                continue

            if resp.status_code != 200:
                await asyncio.sleep(1)
                continue

            run_info = resp.json().get("data", {})
            status = run_info.get("status", "RUNNING")

            if status in ("SUCCEEDED", "FAILED", "ABORTED", "TIMED-OUT"):
                break
            # Still RUNNING after the long-poll cap — loop and wait again.

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
