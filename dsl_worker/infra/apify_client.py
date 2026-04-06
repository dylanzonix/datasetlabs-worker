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
                "description": (item.get("description") or "")[:200],
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

        # Poll for completion
        start = asyncio.get_event_loop().time()
        status = "RUNNING"

        while (asyncio.get_event_loop().time() - start) < timeout:
            await asyncio.sleep(5)

            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(
                    f"{BASE_URL}/actor-runs/{run_id}",
                    headers=self._headers,
                )

            if resp.status_code != 200:
                continue

            run_info = resp.json().get("data", {})
            status = run_info.get("status", "RUNNING")

            if status in ("SUCCEEDED", "FAILED", "ABORTED", "TIMED-OUT"):
                break

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

        # Get cost info
        cost_usd = 0.0
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(
                    f"{BASE_URL}/actor-runs/{run_id}",
                    headers=self._headers,
                )
            if resp.status_code == 200:
                run_info = resp.json().get("data", {})
                cost_usd = run_info.get("usageTotalUsd", 0.0) or 0.0
        except Exception:
            pass

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
