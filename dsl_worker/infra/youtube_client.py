"""
YouTube Data API v3 client.

Operations:
- search(query, type) → list of videos/channels/playlists
- channel_details(channel_ids) → subscriber count, views, video count, etc.
- video_details(video_ids) → view count, likes, comments, tags, etc.

Cost: Free. 10,000 units/day quota.
"""

import logging
from typing import Any, Dict, List, Optional

import httpx

logger = logging.getLogger(__name__)

BASE_URL = "https://www.googleapis.com/youtube/v3"


class YouTubeClient:
    """YouTube Data API v3 client."""

    def __init__(self, api_key: str) -> None:
        self._api_key = api_key

    async def search(
        self,
        query: str,
        search_type: str = "video",
        max_results: int = 25,
        page_token: Optional[str] = None,
        order: str = "relevance",
        region_code: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Search YouTube for videos, channels, or playlists.

        Args:
            query: Search query text
            search_type: "video", "channel", or "playlist"
            max_results: 1-50 results per page
            page_token: For pagination
            order: "relevance", "date", "viewCount", "rating"
            region_code: ISO 3166-1 alpha-2 country code

        Returns {"items": [...], "next_page_token": "..." or None, "total_results": N}.
        """
        params = {
            "part": "snippet",
            "q": query,
            "type": search_type,
            "maxResults": min(max_results, 50),
            "order": order,
            "key": self._api_key,
        }
        if page_token:
            params["pageToken"] = page_token
        if region_code:
            params["regionCode"] = region_code

        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(f"{BASE_URL}/search", params=params)
            data = resp.json()

        if "error" in data:
            logger.warning(f"[YouTube] search error: {data['error'].get('message', '')}")
            return {"items": [], "next_page_token": None, "total_results": 0}

        items = []
        for item in data.get("items", []):
            entry = {
                "title": item["snippet"]["title"],
                "description": item["snippet"]["description"],
                "channel_title": item["snippet"]["channelTitle"],
                "published_at": item["snippet"]["publishedAt"],
                "thumbnail": item["snippet"]["thumbnails"].get("high", {}).get("url", ""),
            }
            # ID structure differs by type
            if search_type == "video":
                entry["video_id"] = item["id"]["videoId"]
                entry["url"] = f"https://youtube.com/watch?v={item['id']['videoId']}"
            elif search_type == "channel":
                entry["channel_id"] = item["snippet"]["channelId"]
                entry["url"] = f"https://youtube.com/channel/{item['snippet']['channelId']}"
            elif search_type == "playlist":
                entry["playlist_id"] = item["id"]["playlistId"]
                entry["url"] = f"https://youtube.com/playlist?list={item['id']['playlistId']}"
            items.append(entry)

        return {
            "items": items,
            "next_page_token": data.get("nextPageToken"),
            "total_results": data.get("pageInfo", {}).get("totalResults", 0),
        }

    async def channel_details(self, channel_ids: List[str]) -> List[Dict[str, Any]]:
        """Get details for up to 50 channels in one request.

        Returns list of dicts with: name, subscriber_count, view_count,
        video_count, description, country, url, custom_url.
        """
        if not channel_ids:
            return []

        params = {
            "part": "snippet,statistics,brandingSettings",
            "id": ",".join(channel_ids[:50]),
            "key": self._api_key,
        }

        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(f"{BASE_URL}/channels", params=params)
            data = resp.json()

        if "error" in data:
            logger.warning(f"[YouTube] channel_details error: {data['error'].get('message', '')}")
            return []

        results = []
        for item in data.get("items", []):
            stats = item.get("statistics", {})
            snippet = item.get("snippet", {})
            results.append({
                "channel_id": item["id"],
                "name": snippet.get("title", ""),
                "description": snippet.get("description", ""),
                "country": snippet.get("country", ""),
                "custom_url": snippet.get("customUrl", ""),
                "url": f"https://youtube.com/channel/{item['id']}",
                "subscriber_count": int(stats.get("subscriberCount", 0)),
                "view_count": int(stats.get("viewCount", 0)),
                "video_count": int(stats.get("videoCount", 0)),
                "published_at": snippet.get("publishedAt", ""),
            })

        return results

    async def video_details(self, video_ids: List[str]) -> List[Dict[str, Any]]:
        """Get details for up to 50 videos in one request.

        Returns list of dicts with: title, view_count, like_count,
        comment_count, duration, tags, description, channel, url.
        """
        if not video_ids:
            return []

        params = {
            "part": "snippet,statistics,contentDetails",
            "id": ",".join(video_ids[:50]),
            "key": self._api_key,
        }

        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(f"{BASE_URL}/videos", params=params)
            data = resp.json()

        if "error" in data:
            logger.warning(f"[YouTube] video_details error: {data['error'].get('message', '')}")
            return []

        results = []
        for item in data.get("items", []):
            stats = item.get("statistics", {})
            snippet = item.get("snippet", {})
            content = item.get("contentDetails", {})
            results.append({
                "video_id": item["id"],
                "title": snippet.get("title", ""),
                "description": snippet.get("description", ""),
                "channel_title": snippet.get("channelTitle", ""),
                "channel_id": snippet.get("channelId", ""),
                "published_at": snippet.get("publishedAt", ""),
                "tags": snippet.get("tags", []),
                "url": f"https://youtube.com/watch?v={item['id']}",
                "view_count": int(stats.get("viewCount", 0)),
                "like_count": int(stats.get("likeCount", 0)),
                "comment_count": int(stats.get("commentCount", 0)),
                "duration": content.get("duration", ""),
            })

        return results
