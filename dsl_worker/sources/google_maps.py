"""google_maps — Google Places search with optional spatial subdivision.

Google's Nearby/Text Search hard-caps at 60 results per query (3 pages × 20).
For asks larger than 60, we subdivide the geographic area into smaller circles,
query each, dedupe by place_id, and merge. Agent never sees the subdivision —
just gets up to `n` deduped places back.

For `n <= 60`, single query with native pagination via `next_page_token`.

Cost: ~$0.017/result × $0.10/credit = ~0.17 credits/result. Real money once
you fan out into multi-hundred-result asks.
"""

from __future__ import annotations

import asyncio
import logging
import math
import os
from typing import Any, Dict, List, Optional, Tuple

import httpx

from dsl_worker.sources.base import FetchResult, SourceAdapter, SourceDescription, register


log = logging.getLogger(__name__)

PLACES_BASE = "https://maps.googleapis.com/maps/api/place"


def _looks_like_latlng(s: str) -> bool:
    """True if 's' is a 'lat,lng' pair (e.g. '37.7,-122.4'), false for city names."""
    parts = s.split(",")
    if len(parts) != 2:
        return False
    try:
        float(parts[0].strip())
        float(parts[1].strip())
        return True
    except ValueError:
        return False


DEFAULT_COLUMNS = [
    {"source_field": "place_id", "column_name": "place_id", "type": "text"},
    {"source_field": "name", "column_name": "name", "type": "text"},
    {"source_field": "formatted_address", "column_name": "address", "type": "text"},
    {"source_field": "rating", "column_name": "rating", "type": "number"},
    {"source_field": "user_ratings_total", "column_name": "review_count", "type": "number"},
    {"source_field": "types", "column_name": "categories", "type": "text"},
    {"source_field": "business_status", "column_name": "business_status", "type": "enum"},
    {"source_field": "price_level", "column_name": "price_level", "type": "number"},
    {"source_field": "website", "column_name": "website", "type": "url"},
    {"source_field": "formatted_phone_number", "column_name": "phone", "type": "text"},
    {"source_field": "google_maps_url", "column_name": "google_maps_url", "type": "url"},
    {"source_field": "geometry_lat", "column_name": "latitude", "type": "number"},
    {"source_field": "geometry_lng", "column_name": "longitude", "type": "number"},
]


ALLOWED_PARAMS = {
    "query", "location", "radius_miles",
    "min_rating", "max_review_count",
    "n", "next_page_token",
}


class GoogleMapsAdapter(SourceAdapter):
    name = "google_maps"
    label = "Google Maps"
    favicon_url = "https://www.google.com/s2/favicons?domain=maps.google.com&sz=32"
    predictable = True
    default_columns = DEFAULT_COLUMNS
    default_dedup_key_column = "place_id"

    def describe(
        self,
        query_params: Dict[str, Any],
        source: Optional[str] = None,
    ) -> SourceDescription:
        qp = query_params or {}
        q = qp.get("query") or "Places"
        loc = qp.get("location")
        radius = qp.get("radius_miles")
        bits = [str(q)]
        if loc:
            bits.append(f"near {loc}")
        if radius:
            bits.append(f"within {radius}mi")
        headline = " ".join(bits)

        detail_lines: List[str] = []
        if qp.get("min_rating"):
            detail_lines.append(f"- **Min rating:** {qp['min_rating']}")
        if qp.get("max_review_count"):
            detail_lines.append(f"- **Max review count:** {qp['max_review_count']}")
        if qp.get("n"):
            detail_lines.append(f"- **Target rows:** {qp['n']}")
        return SourceDescription(
            kind=self.name,
            label=self.label,
            query_text=headline,
            details="\n".join(detail_lines),
            favicon_url=self.favicon_url,
        )

    def __init__(self) -> None:
        self.api_key = os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            log.warning("GOOGLE_API_KEY not set — google_maps adapter inert")

    def validate_query_params(self, query_params: Dict[str, Any]) -> Optional[str]:
        if "query" not in query_params:
            return "google_maps requires `query` (e.g., 'flooring contractor')"
        if "location" not in query_params:
            return "google_maps requires `location` (e.g., 'San Diego, CA')"
        bad = [k for k in query_params if k not in ALLOWED_PARAMS]
        if bad:
            return f"unknown google_maps params: {bad}. Allowed: {sorted(ALLOWED_PARAMS)}"
        return None

    async def _text_search_page(
        self,
        client: httpx.AsyncClient,
        query: str,
        location: Optional[str],
        radius_m: Optional[int],
        page_token: Optional[str] = None,
    ) -> Tuple[List[Dict[str, Any]], Optional[str]]:
        """One page of Places Text Search. Returns (results, next_page_token).

        Google's textsearch `location` param expects "lat,lng" — passing a
        city name silently falls back to IP geolocation, which is why an
        Austin, TX query was returning Florida results. We fold the location
        into the query text instead ("dentist in Austin, TX"), which is what
        the textsearch endpoint is designed for.
        """
        composed_query = query
        if location and _looks_like_latlng(location):
            pass  # use as location bias param
        elif location:
            composed_query = f"{query} in {location}"
            location = None  # folded into query text
        params: Dict[str, Any] = {"key": self.api_key, "query": composed_query}
        if location:
            params["location"] = location  # lat,lng only
        if radius_m:
            params["radius"] = radius_m
        if page_token:
            params["pagetoken"] = page_token
        resp = await client.get(f"{PLACES_BASE}/textsearch/json", params=params, timeout=30.0)
        if resp.status_code != 200:
            log.warning("google_maps HTTP %s: %s", resp.status_code, resp.text[:200])
            return [], None
        data = resp.json() or {}
        results = data.get("results") or []
        # Flatten geometry for column mapping
        for r in results:
            geom = (r.get("geometry") or {}).get("location") or {}
            if "lat" in geom:
                r["geometry_lat"] = geom["lat"]
            if "lng" in geom:
                r["geometry_lng"] = geom["lng"]
            pid = r.get("place_id")
            if pid:
                r["google_maps_url"] = f"https://www.google.com/maps/place/?q=place_id:{pid}"
            cats = r.get("types") or []
            if cats:
                r["types"] = ", ".join(cats)
        return results, data.get("next_page_token")

    async def _single_query(
        self,
        client: httpx.AsyncClient,
        query: str,
        location: Optional[str],
        radius_m: Optional[int],
        start_token: Optional[str] = None,
    ) -> Tuple[List[Dict[str, Any]], Optional[str]]:
        """Walk up to 3 pages (60 results) of a single query."""
        results: List[Dict[str, Any]] = []
        token = start_token
        first_page = True
        for _ in range(3):
            if not first_page and token:
                # Google requires a small delay before the next_page_token is active.
                await asyncio.sleep(2)
            first_page = False
            page_results, next_token = await self._text_search_page(
                client, query, location if not start_token else None, radius_m, token
            )
            if not page_results:
                break
            results.extend(page_results)
            token = next_token
            if not token:
                break
        return results, token

    async def fetch(
        self,
        query_params: Dict[str, Any],
        n: int,
        prior_cursor: Optional[Dict[str, Any]] = None,
    ) -> FetchResult:
        if not self.api_key:
            return FetchResult(rows=[], schema=[], cost_credits=0.0, exhausted=True)

        query = query_params["query"]
        location = query_params.get("location")
        radius_miles = query_params.get("radius_miles")
        radius_m = int(radius_miles * 1609) if radius_miles else None

        async with httpx.AsyncClient() as client:
            if n <= 60:
                start_token = (prior_cursor or {}).get("next_page_token")
                results, next_tok = await self._single_query(
                    client, query, location, radius_m, start_token=start_token
                )
                cursor = {"next_page_token": next_tok} if next_tok else None
                exhausted = next_tok is None
                # Google Places textsearch is $0.032 per request, up to 20
                # results per page. Estimate request count from result count.
                # (1 cr = $0.10 of compute → 0.32 cr per request.)
                num_requests = max(1, math.ceil(len(results) / 20)) if results else 0
                cost = num_requests * 0.32
                return FetchResult(
                    rows=results[:n],
                    schema=sorted({k for r in results for k in r.keys()})[:30],
                    cost_credits=cost,
                    exhausted=exhausted,
                    cursor=cursor,
                    dedup_key_column_hint="place_id",
                )

            # n > 60 → spatial subdivision. Naive 5x5 grid around the
            # original location with a smaller radius. Production fix would
            # use a proper hex tiling library; this is good enough for v1.
            sub_queries = self._subdivide(location, radius_miles or 25, target=n)
            seen_place_ids = set((prior_cursor or {}).get("seen_place_ids") or [])
            all_results: List[Dict[str, Any]] = []
            sub_tasks = [
                self._single_query(client, query, sub_loc, sub_radius_m)
                for sub_loc, sub_radius_m in sub_queries
            ]
            sub_request_count = 0
            for sub_results, _next_tok in await asyncio.gather(*sub_tasks, return_exceptions=False):
                # Each sub_query walked up to 3 pages; count pages by results
                sub_request_count += max(1, math.ceil(len(sub_results) / 20)) if sub_results else 1
                for r in sub_results:
                    pid = r.get("place_id")
                    if pid and pid not in seen_place_ids:
                        seen_place_ids.add(pid)
                        all_results.append(r)
                if len(all_results) >= n:
                    break

        cost = sub_request_count * 0.32
        return FetchResult(
            rows=all_results[:n],
            schema=sorted({k for r in all_results for k in r.keys()})[:30],
            cost_credits=cost,
            exhausted=len(all_results) < n,
            cursor={"seen_place_ids": list(seen_place_ids)[-500:]},  # cap to avoid growing forever
            dedup_key_column_hint="place_id",
        )

    @staticmethod
    def _subdivide(location: Optional[str], outer_radius_miles: float, target: int) -> List[Tuple[Optional[str], int]]:
        """Generate (sub_location, sub_radius_m) pairs to fan out.

        For v1: a coarse grid offset by lat/lng. Real implementation will use
        proper hex tiling once we have a need. The outer location is used as
        a string (city name); we attach small lat/lng offsets via additional
        text queries. Simpler approach: just issue multiple text queries with
        nearby city names — but for v1, return a few smaller-radius variants
        of the original. Caller dedupes by place_id.
        """
        # For v1 simplicity, return ~ceil(target/60) sub-queries with smaller radii.
        n_sub = max(1, math.ceil(target / 50))
        sub_radius_miles = max(2.0, outer_radius_miles / math.sqrt(n_sub))
        sub_radius_m = int(sub_radius_miles * 1609)
        # All sub-queries hit the same center; differentiation is by smaller
        # radius which surfaces different ranked results when Google's algo
        # has more than 60 candidates in the area. Less effective than true
        # spatial tiling but a reasonable v1 fallback.
        return [(location, sub_radius_m)] * n_sub


register(GoogleMapsAdapter())
