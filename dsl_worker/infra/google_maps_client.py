"""
Google Maps Places API client.

Two operations:
- text_search(query) → list of businesses with basic info
- place_details(place_id) → full details (phone, website, etc.)

Cost: ~$0.003/text search, ~$0.003/place details. Pay-as-you-go.
"""

import logging
from typing import Any, Dict, List, Optional

import httpx

logger = logging.getLogger(__name__)

BASE_URL = "https://maps.googleapis.com/maps/api/place"

# Fields to request from Place Details (controls cost — only request what we need)
DETAIL_FIELDS = (
    "name,formatted_address,formatted_phone_number,website,url,"
    "rating,user_ratings_total,business_status,types,opening_hours"
)


class GoogleMapsClient:
    """Google Maps Places API client."""

    def __init__(self, api_key: str) -> None:
        self._api_key = api_key

    async def text_search(
        self,
        query: str,
        page_token: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Search for businesses by text query.

        Returns {"results": [...], "next_page_token": "..." or None}.
        Each result has: name, formatted_address, place_id, rating,
        user_ratings_total, types, business_status, geometry.
        Up to 20 results per page.
        """
        params = {"query": query, "key": self._api_key}
        if page_token:
            params["pagetoken"] = page_token

        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(f"{BASE_URL}/textsearch/json", params=params)
            data = resp.json()

        if data.get("status") != "OK" and data.get("status") != "ZERO_RESULTS":
            logger.warning(f"[GoogleMaps] text_search error: {data.get('status')} - {data.get('error_message', '')}")

        return {
            "results": data.get("results", []),
            "next_page_token": data.get("next_page_token"),
        }

    async def place_details(self, place_id: str) -> Optional[Dict[str, Any]]:
        """Get full details for a place by place_id.

        Returns dict with: name, formatted_address, formatted_phone_number,
        website, url (Google Maps link), rating, user_ratings_total,
        business_status, types, opening_hours.
        """
        params = {
            "place_id": place_id,
            "fields": DETAIL_FIELDS,
            "key": self._api_key,
        }

        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(f"{BASE_URL}/details/json", params=params)
            data = resp.json()

        if data.get("status") != "OK":
            logger.warning(f"[GoogleMaps] place_details error: {data.get('status')}")
            return None

        return data.get("result")
