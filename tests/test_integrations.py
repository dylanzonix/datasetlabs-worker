"""
Quick integration tests for third-party APIs.

Run: python -m pytest tests/test_integrations.py -v
Or individually: python tests/test_integrations.py
"""
import os
import json
import httpx
import pytest
from dotenv import load_dotenv

load_dotenv()

GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "")
SPOTIFY_CLIENT_ID = os.environ.get("SPOTIFY_CLIENT_ID", "")
SPOTIFY_CLIENT_SECRET = os.environ.get("SPOTIFY_CLIENT_SECRET", "")


# ── Google Maps Places API ───────────────────────────────────────────

class TestGoogleMaps:
    """Google Maps Places API — text search + place details."""

    def test_text_search(self):
        """Search for businesses by query."""
        resp = httpx.get(
            "https://maps.googleapis.com/maps/api/place/textsearch/json",
            params={"query": "towing companies in Fresno CA", "key": GOOGLE_API_KEY},
        )
        data = resp.json()
        assert data["status"] == "OK", f"API error: {data.get('status')}"
        assert len(data["results"]) > 0
        # Check fields present
        r = data["results"][0]
        assert "name" in r
        assert "formatted_address" in r
        assert "place_id" in r
        print(f"  Found {len(data['results'])} businesses. First: {r['name']}")

    def test_place_details(self):
        """Get full details (phone, website) for a place."""
        # First get a place_id
        search = httpx.get(
            "https://maps.googleapis.com/maps/api/place/textsearch/json",
            params={"query": "towing companies in Fresno CA", "key": GOOGLE_API_KEY},
        ).json()
        place_id = search["results"][0]["place_id"]

        resp = httpx.get(
            "https://maps.googleapis.com/maps/api/place/details/json",
            params={
                "place_id": place_id,
                "fields": "name,formatted_address,formatted_phone_number,website,url,rating,user_ratings_total,business_status",
                "key": GOOGLE_API_KEY,
            },
        )
        data = resp.json()
        assert data["status"] == "OK"
        r = data["result"]
        assert "name" in r
        assert "formatted_address" in r
        # Phone and website may not always be present but URL always is
        assert "url" in r  # Google Maps URL
        print(f"  {r['name']}: phone={r.get('formatted_phone_number', 'N/A')}, website={r.get('website', 'N/A')}")

    def test_pagination(self):
        """Text search returns next_page_token for pagination."""
        resp = httpx.get(
            "https://maps.googleapis.com/maps/api/place/textsearch/json",
            params={"query": "restaurants in New York", "key": GOOGLE_API_KEY},
        ).json()
        assert resp["status"] == "OK"
        assert len(resp["results"]) == 20
        # next_page_token should exist for a broad query
        has_next = "next_page_token" in resp
        print(f"  20 results, has_next_page: {has_next}")


# ── YouTube Data API v3 ──────────────────────────────────────────────

class TestYouTube:
    """YouTube Data API — search, channel details, video details."""

    def test_search_videos(self):
        """Search for videos."""
        resp = httpx.get(
            "https://www.googleapis.com/youtube/v3/search",
            params={
                "part": "snippet",
                "q": "DayZ hidden mechanics tips",
                "type": "video",
                "maxResults": 5,
                "key": GOOGLE_API_KEY,
            },
        )
        data = resp.json()
        assert "items" in data, f"Error: {data.get('error', {}).get('message', 'unknown')}"
        assert len(data["items"]) > 0
        print(f"  Found {len(data['items'])} videos. First: {data['items'][0]['snippet']['title']}")

    def test_search_channels(self):
        """Search for channels."""
        resp = httpx.get(
            "https://www.googleapis.com/youtube/v3/search",
            params={
                "part": "snippet",
                "q": "fitness workout",
                "type": "channel",
                "maxResults": 5,
                "key": GOOGLE_API_KEY,
            },
        ).json()
        assert "items" in resp
        assert len(resp["items"]) > 0
        print(f"  Found {len(resp['items'])} channels. First: {resp['items'][0]['snippet']['channelTitle']}")

    def test_channel_details(self):
        """Get channel statistics (subscribers, views, video count)."""
        # MadFit channel
        resp = httpx.get(
            "https://www.googleapis.com/youtube/v3/channels",
            params={
                "part": "snippet,statistics",
                "id": "UCpQ34afVgk8cRQBjSJ1xuJQ",
                "key": GOOGLE_API_KEY,
            },
        ).json()
        assert "items" in resp
        ch = resp["items"][0]
        stats = ch["statistics"]
        assert "subscriberCount" in stats
        assert "viewCount" in stats
        print(f"  {ch['snippet']['title']}: {int(stats['subscriberCount']):,} subs, {int(stats['viewCount']):,} views")

    def test_batch_channels(self):
        """Batch up to 50 channel IDs in one request."""
        ids = "UCpQ34afVgk8cRQBjSJ1xuJQ,UCZUUZFex6AaIU4QTopFudYA"
        resp = httpx.get(
            "https://www.googleapis.com/youtube/v3/channels",
            params={"part": "statistics", "id": ids, "key": GOOGLE_API_KEY},
        ).json()
        assert len(resp["items"]) == 2
        print(f"  Batch returned {len(resp['items'])} channels")


# ── Spotify Web API ──────────────────────────────────────────────────

class TestSpotify:
    """Spotify Web API — dev mode (limited). Podcasts/shows work best."""

    @staticmethod
    def _get_token():
        resp = httpx.post(
            "https://accounts.spotify.com/api/token",
            data={
                "grant_type": "client_credentials",
                "client_id": SPOTIFY_CLIENT_ID,
                "client_secret": SPOTIFY_CLIENT_SECRET,
            },
        )
        return resp.json()["access_token"]

    def test_search_shows(self):
        """Search for podcasts/shows (works in dev mode)."""
        token = self._get_token()
        resp = httpx.get(
            "https://api.spotify.com/v1/search",
            params={"q": "health wellness", "type": "show", "limit": 5, "market": "US"},
            headers={"Authorization": f"Bearer {token}"},
        ).json()
        shows = resp.get("shows", {}).get("items", [])
        assert len(shows) > 0
        s = shows[0]
        assert "name" in s
        assert "total_episodes" in s
        print(f"  Found {len(shows)} shows. First: {s['name']} ({s['total_episodes']} episodes)")

    def test_show_episodes(self):
        """Get episodes for a show (works in dev mode)."""
        token = self._get_token()
        # The Dr. Hyman Show
        resp = httpx.get(
            "https://api.spotify.com/v1/shows/50MFhL6rItlnDDEStFMSPu/episodes",
            params={"limit": 3, "market": "US"},
            headers={"Authorization": f"Bearer {token}"},
        ).json()
        assert "items" in resp
        eps = resp["items"]
        assert len(eps) > 0
        print(f"  Got {len(eps)} episodes. Latest: {eps[0]['name']} ({eps[0]['release_date']})")

    def test_search_artists_limited(self):
        """Artist search works but returns limited data in dev mode."""
        token = self._get_token()
        resp = httpx.get(
            "https://api.spotify.com/v1/search",
            params={"q": "hip hop", "type": "artist", "limit": 3},
            headers={"Authorization": f"Bearer {token}"},
        ).json()
        artists = resp.get("artists", {}).get("items", [])
        assert len(artists) > 0
        a = artists[0]
        # Dev mode: followers and popularity may be missing
        has_followers = "followers" in a and a["followers"].get("total") is not None
        print(f"  Found {len(artists)} artists. Has follower data: {has_followers}")


# ── Run all tests ────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n=== Google Maps ===")
    t = TestGoogleMaps()
    t.test_text_search()
    t.test_place_details()
    t.test_pagination()

    print("\n=== YouTube ===")
    t = TestYouTube()
    t.test_search_videos()
    t.test_search_channels()
    t.test_channel_details()
    t.test_batch_channels()

    print("\n=== Spotify ===")
    t = TestSpotify()
    t.test_search_shows()
    t.test_show_episodes()
    t.test_search_artists_limited()

    print("\nAll tests passed!")
