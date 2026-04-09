"""
Google Maps namespace tools for orchestrator and row generator.

Provides: search_places, search_nearby, place_details, geocode,
reverse_geocode, batch_geocode, distance_matrix, directions,
local_rank_tracker

Modeled on cablate/mcp-google-map quality — compound tools,
conditional field inclusion, workflow coaching in descriptions.
"""

import json
import logging
from pathlib import Path
from typing import Any, Callable, Awaitable, Dict, List, Optional, Tuple

import httpx

logger = logging.getLogger(__name__)

ToolHandler = Callable[[Dict[str, Any]], Awaitable[Tuple[str, float]]]

NAMESPACE_DESCRIPTION = (
    "Google Maps: search for businesses and places, get details (phone, website, "
    "hours, reviews, rating), geocode addresses, calculate distances and directions, "
    "search nearby locations. Best for local business data."
)

PLACES_V1 = "https://places.googleapis.com/v1"
MAPS_API = "https://maps.googleapis.com/maps/api"


def register_google_maps_namespace(
    registry: Any,
    api_key: str,
    workspace_dir: Path,
    file_counter: Optional[List[int]] = None,
) -> None:
    """Register the google_maps namespace on a ToolRegistry."""
    if file_counter is None:
        file_counter = [0]

    def _next_filename(prefix: str) -> Path:
        idx = file_counter[0]
        file_counter[0] += 1
        return workspace_dir / "candidates" / f"{prefix}_{idx}.jsonl"

    headers_new = {
        "X-Goog-Api-Key": api_key,
        "Content-Type": "application/json",
    }

    # ── Helper: Places API (New) request ─────────────────────────────

    async def _places_post(endpoint: str, body: dict, field_mask: str) -> dict:
        h = {**headers_new, "X-Goog-FieldMask": field_mask}
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.post(f"{PLACES_V1}/{endpoint}", json=body, headers=h)
        if resp.status_code != 200:
            return {"error": f"HTTP {resp.status_code}: {resp.text[:200]}"}
        return resp.json()

    async def _places_get(place_id: str, field_mask: str) -> dict:
        h = {**headers_new, "X-Goog-FieldMask": field_mask}
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.get(f"{PLACES_V1}/places/{place_id}", headers=h)
        if resp.status_code != 200:
            return {"error": f"HTTP {resp.status_code}: {resp.text[:200]}"}
        return resp.json()

    async def _maps_get(endpoint: str, params: dict) -> dict:
        params["key"] = api_key
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.get(f"{MAPS_API}/{endpoint}", params=params)
        if resp.status_code != 200:
            return {"error": f"HTTP {resp.status_code}: {resp.text[:200]}"}
        return resp.json()

    # ── Formatters ───────────────────────────────────────────────────

    def _format_place(p: dict) -> dict:
        """Simplify a Places API (New) result."""
        return {
            "name": p.get("displayName", {}).get("text", ""),
            "place_id": p.get("id", ""),
            "address": p.get("formattedAddress", ""),
            "location": p.get("location", {}),
            "types": p.get("types", []),
            "rating": p.get("rating"),
            "total_ratings": p.get("userRatingCount"),
            "price_level": p.get("priceLevel"),
            "phone": p.get("nationalPhoneNumber"),
            "website": p.get("websiteUri"),
            "open_now": p.get("currentOpeningHours", {}).get("openNow") if p.get("currentOpeningHours") else None,
        }

    # ── search_places ────────────────────────────────────────────────

    async def search_places(args: Dict) -> Tuple[str, float]:
        query = args.get("query", "")
        if not query:
            return "Error: query is required.", 0.0

        body: Dict[str, Any] = {
            "textQuery": query,
            "maxResultCount": args.get("limit", 20),
        }

        if args.get("latitude") and args.get("longitude"):
            body["locationBias"] = {
                "circle": {
                    "center": {"latitude": args["latitude"], "longitude": args["longitude"]},
                    "radius": args.get("radius", 5000),
                }
            }

        if args.get("open_now"):
            body["openNow"] = True
        if args.get("min_rating"):
            body["minRating"] = args["min_rating"]
        if args.get("type"):
            body["includedType"] = args["type"]

        field_mask = "places.id,places.displayName,places.formattedAddress,places.location,places.types,places.rating,places.userRatingCount,places.priceLevel,places.nationalPhoneNumber,places.websiteUri,places.currentOpeningHours"
        result = await _places_post("places:searchText", body, field_mask)

        if "error" in result:
            return f"Error: {result['error']}", 0.0

        places = [_format_place(p) for p in result.get("places", [])]
        if not places:
            return "No places found.", 0.0

        # Write to file
        output_path = _next_filename("gmaps_search")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            for p in places:
                f.write(json.dumps(p, ensure_ascii=False) + "\n")

        workspace_path = f"/workspace/candidates/{output_path.name}"
        lines = [f"Found {len(places)} places.\nFile: {workspace_path}\n"]
        for p in places[:5]:
            rating = f" | rating: {p['rating']}" if p.get("rating") else ""
            phone = f" | phone: {p['phone']}" if p.get("phone") else ""
            lines.append(f"  {p['name']} — {p['address']}{rating}{phone}")
        if len(places) > 5:
            lines.append(f"  ... and {len(places) - 5} more in file")

        return "\n".join(lines), 0.0

    # ── search_nearby ────────────────────────────────────────────────

    async def search_nearby(args: Dict) -> Tuple[str, float]:
        lat = args.get("latitude")
        lng = args.get("longitude")
        if lat is None or lng is None:
            return "Error: latitude and longitude are required.", 0.0

        body: Dict[str, Any] = {
            "maxResultCount": args.get("limit", 20),
            "locationRestriction": {
                "circle": {
                    "center": {"latitude": lat, "longitude": lng},
                    "radius": args.get("radius", 1000),
                }
            },
        }

        if args.get("type"):
            body["includedTypes"] = [args["type"]]
        if args.get("keyword"):
            body["includedTypes"] = body.get("includedTypes", [])
            # keyword search via text query not supported in nearby, use type

        field_mask = "places.id,places.displayName,places.formattedAddress,places.location,places.types,places.rating,places.userRatingCount,places.nationalPhoneNumber,places.websiteUri"
        result = await _places_post("places:searchNearby", body, field_mask)

        if "error" in result:
            return f"Error: {result['error']}", 0.0

        places = [_format_place(p) for p in result.get("places", [])]
        if not places:
            return "No places found nearby.", 0.0

        output_path = _next_filename("gmaps_nearby")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            for p in places:
                f.write(json.dumps(p, ensure_ascii=False) + "\n")

        workspace_path = f"/workspace/candidates/{output_path.name}"
        lines = [f"Found {len(places)} places nearby.\nFile: {workspace_path}\n"]
        for p in places[:5]:
            lines.append(f"  {p['name']} — {p['address']}")

        return "\n".join(lines), 0.0

    # ── place_details ────────────────────────────────────────────────

    async def place_details(args: Dict) -> Tuple[str, float]:
        place_id = args.get("place_id", "")
        if not place_id:
            return "Error: place_id is required.", 0.0

        fields = [
            "displayName", "formattedAddress", "location", "types",
            "rating", "userRatingCount", "nationalPhoneNumber",
            "internationalPhoneNumber", "websiteUri",
            "regularOpeningHours", "editorialSummary",
            "priceLevel", "priceRange",
        ]

        if args.get("include_reviews", True):
            fields.append("reviews")

        result = await _places_get(place_id, ",".join(fields))
        if "error" in result:
            return f"Error: {result['error']}", 0.0

        parts = [
            f"**{result.get('displayName', {}).get('text', '?')}**",
            f"Address: {result.get('formattedAddress', '?')}",
        ]
        if result.get("nationalPhoneNumber"):
            parts.append(f"Phone: {result['nationalPhoneNumber']}")
        if result.get("websiteUri"):
            parts.append(f"Website: {result['websiteUri']}")
        if result.get("rating"):
            parts.append(f"Rating: {result['rating']}/5 ({result.get('userRatingCount', 0)} reviews)")
        if result.get("editorialSummary", {}).get("text"):
            parts.append(f"Summary: {result['editorialSummary']['text']}")
        if result.get("regularOpeningHours", {}).get("weekdayDescriptions"):
            hours = result["regularOpeningHours"]["weekdayDescriptions"]
            parts.append(f"Hours: {'; '.join(hours[:3])}")

        reviews = result.get("reviews", [])
        if reviews:
            parts.append(f"\nTop reviews ({len(reviews)}):")
            for r in reviews[:3]:
                author = r.get("authorAttribution", {}).get("displayName", "?")
                text = r.get("text", {}).get("text", "")[:150]
                rating = r.get("rating", "?")
                parts.append(f"  [{rating}/5] {author}: {text}")

        return "\n".join(parts), 0.0

    # ── geocode ──────────────────────────────────────────────────────

    async def geocode(args: Dict) -> Tuple[str, float]:
        address = args.get("address", "")
        if not address:
            return "Error: address is required.", 0.0

        result = await _maps_get("geocode/json", {"address": address})
        if "error" in result:
            return f"Error: {result['error']}", 0.0

        results = result.get("results", [])
        if not results:
            return f"No results for: {address}", 0.0

        r = results[0]
        loc = r.get("geometry", {}).get("location", {})
        return (
            f"Address: {r.get('formatted_address', address)}\n"
            f"Latitude: {loc.get('lat')}\n"
            f"Longitude: {loc.get('lng')}\n"
            f"Place ID: {r.get('place_id', '')}"
        ), 0.0

    # ── reverse_geocode ──────────────────────────────────────────────

    async def reverse_geocode(args: Dict) -> Tuple[str, float]:
        lat = args.get("latitude")
        lng = args.get("longitude")
        if lat is None or lng is None:
            return "Error: latitude and longitude are required.", 0.0

        result = await _maps_get("geocode/json", {"latlng": f"{lat},{lng}"})
        if "error" in result:
            return f"Error: {result['error']}", 0.0

        results = result.get("results", [])
        if not results:
            return f"No results for coordinates: {lat}, {lng}", 0.0

        r = results[0]
        return (
            f"Address: {r.get('formatted_address', '?')}\n"
            f"Place ID: {r.get('place_id', '')}"
        ), 0.0

    # ── batch_geocode ────────────────────────────────────────────────

    async def batch_geocode(args: Dict) -> Tuple[str, float]:
        addresses = args.get("addresses", [])
        if not addresses:
            return "Error: addresses array is required.", 0.0
        if len(addresses) > 50:
            return "Error: max 50 addresses per batch.", 0.0

        results = []
        for addr in addresses:
            r = await _maps_get("geocode/json", {"address": addr})
            geo_results = r.get("results", [])
            if geo_results:
                loc = geo_results[0].get("geometry", {}).get("location", {})
                results.append({
                    "address": addr,
                    "formatted_address": geo_results[0].get("formatted_address", addr),
                    "latitude": loc.get("lat"),
                    "longitude": loc.get("lng"),
                    "place_id": geo_results[0].get("place_id", ""),
                })
            else:
                results.append({"address": addr, "error": "not found"})

        succeeded = sum(1 for r in results if "latitude" in r)
        output_path = _next_filename("gmaps_geocode")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            for r in results:
                f.write(json.dumps(r) + "\n")

        workspace_path = f"/workspace/candidates/{output_path.name}"
        return (
            f"Geocoded {succeeded}/{len(addresses)} addresses.\n"
            f"File: {workspace_path}"
        ), 0.0

    # ── distance_matrix ──────────────────────────────────────────────

    async def distance_matrix(args: Dict) -> Tuple[str, float]:
        origins = args.get("origins", [])
        destinations = args.get("destinations", [])
        if not origins or not destinations:
            return "Error: origins and destinations are required.", 0.0

        result = await _maps_get("distancematrix/json", {
            "origins": "|".join(origins),
            "destinations": "|".join(destinations),
            "mode": args.get("mode", "driving"),
        })

        if "error" in result:
            return f"Error: {result['error']}", 0.0

        rows = result.get("rows", [])
        origin_addrs = result.get("origin_addresses", origins)
        dest_addrs = result.get("destination_addresses", destinations)

        lines = ["Distance matrix:"]
        for i, row in enumerate(rows):
            for j, elem in enumerate(row.get("elements", [])):
                dist = elem.get("distance", {}).get("text", "?")
                dur = elem.get("duration", {}).get("text", "?")
                lines.append(f"  {origin_addrs[i]} → {dest_addrs[j]}: {dist}, {dur}")

        return "\n".join(lines), 0.0

    # ── directions ───────────────────────────────────────────────────

    async def directions(args: Dict) -> Tuple[str, float]:
        origin = args.get("origin", "")
        destination = args.get("destination", "")
        if not origin or not destination:
            return "Error: origin and destination are required.", 0.0

        params = {
            "origin": origin,
            "destination": destination,
            "mode": args.get("mode", "driving"),
        }

        result = await _maps_get("directions/json", params)
        if "error" in result:
            return f"Error: {result['error']}", 0.0

        routes = result.get("routes", [])
        if not routes:
            return "No route found.", 0.0

        route = routes[0]
        leg = route["legs"][0]

        lines = [
            f"Route: {leg['distance']['text']}, {leg['duration']['text']}",
            f"From: {leg.get('start_address', origin)}",
            f"To: {leg.get('end_address', destination)}",
        ]

        steps = leg.get("steps", [])
        if steps and args.get("include_steps", False):
            lines.append("\nSteps:")
            for s in steps[:20]:
                instr = s.get("html_instructions", "").replace("<b>", "").replace("</b>", "").replace("<div>", " ").replace("</div>", "")
                lines.append(f"  {s.get('distance', {}).get('text', '?')} — {instr[:100]}")

        return "\n".join(lines), 0.0

    # ── local_rank_tracker ───────────────────────────────────────────

    async def local_rank_tracker(args: Dict) -> Tuple[str, float]:
        keyword = args.get("keyword", "")
        place_id = args.get("place_id", "")
        lat = args.get("latitude")
        lng = args.get("longitude")

        if not keyword or not place_id or lat is None or lng is None:
            return "Error: keyword, place_id, latitude, and longitude are required.", 0.0

        grid_size = args.get("grid_size", 3)
        spacing = args.get("grid_spacing", 1000)  # meters

        # Build grid of search points
        import math
        points = []
        half = grid_size // 2
        for row in range(-half, half + 1):
            for col in range(-half, half + 1):
                # Approximate lat/lng offset from meters
                dlat = (row * spacing) / 111320
                dlng = (col * spacing) / (111320 * math.cos(math.radians(lat)))
                points.append({"latitude": lat + dlat, "longitude": lng + dlng})

        # Search each point
        results = []
        for pt in points:
            body = {
                "textQuery": keyword,
                "maxResultCount": 5,
                "locationBias": {
                    "circle": {
                        "center": pt,
                        "radius": spacing / 2,
                    }
                },
            }
            r = await _places_post(
                "places:searchText", body,
                "places.id,places.displayName,places.rating",
            )
            places = r.get("places", [])
            rank = None
            for i, p in enumerate(places):
                if p.get("id") == place_id:
                    rank = i + 1
                    break

            top3 = [{"name": p.get("displayName", {}).get("text", "?"), "id": p.get("id")} for p in places[:3]]
            results.append({
                "point": pt,
                "rank": rank,
                "top_3": top3,
            })

        # Calculate metrics
        ranks = [r["rank"] for r in results if r["rank"] is not None]
        avg_rank = sum(ranks) / len(ranks) if ranks else None
        visibility = len(ranks) / len(results) if results else 0

        output_path = _next_filename("gmaps_rank")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            for r in results:
                f.write(json.dumps(r) + "\n")

        workspace_path = f"/workspace/candidates/{output_path.name}"
        return (
            f"Local rank tracker for '{keyword}' ({grid_size}x{grid_size} grid, {spacing}m spacing):\n"
            f"  Average rank: {avg_rank:.1f}\n" if avg_rank else "  Not found in any grid point\n"
            f"  Visibility: {visibility:.0%} ({len(ranks)}/{len(results)} points)\n"
            f"  File: {workspace_path}"
        ), 0.0

    # ── Register namespace ───────────────────────────────────────────

    tools = [
        {
            "name": "search_places",
            "description": (
                "Free-text search for places and businesses (e.g. 'taquerias in San Francisco', "
                "'hardware stores near Austin'). Returns name, address, rating, phone, website. "
                "Results saved to file."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Natural language search query"},
                    "limit": {"type": "integer", "description": "Max results (default 20)"},
                    "latitude": {"type": "number", "description": "Optional: bias results near this latitude"},
                    "longitude": {"type": "number", "description": "Optional: bias results near this longitude"},
                    "radius": {"type": "number", "description": "Search radius in meters (default 5000)"},
                    "open_now": {"type": "boolean", "description": "Only open businesses"},
                    "min_rating": {"type": "number", "description": "Minimum rating (1-5)"},
                    "type": {"type": "string", "description": "Place type filter (e.g. 'restaurant', 'hotel')"},
                },
                "required": ["query"],
            },
        },
        {
            "name": "search_nearby",
            "description": (
                "Find places near a specific location by type and radius. "
                "Requires latitude/longitude center point. Results saved to file."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "latitude": {"type": "number", "description": "Center latitude"},
                    "longitude": {"type": "number", "description": "Center longitude"},
                    "radius": {"type": "number", "description": "Search radius in meters (default 1000)"},
                    "type": {"type": "string", "description": "Place type (e.g. 'restaurant', 'gas_station', 'pharmacy')"},
                    "limit": {"type": "integer", "description": "Max results (default 20)"},
                },
                "required": ["latitude", "longitude"],
            },
        },
        {
            "name": "place_details",
            "description": (
                "Get full details for a place by place_id: name, address, phone, website, "
                "hours, rating, reviews, editorial summary. Use after search_places to get "
                "details for specific results."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "place_id": {"type": "string", "description": "Google Maps place_id"},
                    "include_reviews": {"type": "boolean", "description": "Include reviews (default true)"},
                },
                "required": ["place_id"],
            },
        },
        {
            "name": "geocode",
            "description": "Convert an address or landmark to GPS coordinates (latitude/longitude).",
            "parameters": {
                "type": "object",
                "properties": {
                    "address": {"type": "string", "description": "Address or landmark name"},
                },
                "required": ["address"],
            },
        },
        {
            "name": "reverse_geocode",
            "description": "Convert GPS coordinates to a street address.",
            "parameters": {
                "type": "object",
                "properties": {
                    "latitude": {"type": "number"},
                    "longitude": {"type": "number"},
                },
                "required": ["latitude", "longitude"],
            },
        },
        {
            "name": "batch_geocode",
            "description": "Geocode up to 50 addresses in one call. Returns coordinates for each. Results saved to file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "addresses": {"type": "array", "items": {"type": "string"}, "description": "Addresses to geocode (max 50)"},
                },
                "required": ["addresses"],
            },
        },
        {
            "name": "distance_matrix",
            "description": "Calculate travel distances and times between multiple origins and destinations.",
            "parameters": {
                "type": "object",
                "properties": {
                    "origins": {"type": "array", "items": {"type": "string"}, "description": "Origin addresses or coordinates"},
                    "destinations": {"type": "array", "items": {"type": "string"}, "description": "Destination addresses or coordinates"},
                    "mode": {"type": "string", "description": "Travel mode: driving, walking, bicycling, transit (default driving)"},
                },
                "required": ["origins", "destinations"],
            },
        },
        {
            "name": "directions",
            "description": "Get route and step-by-step directions between two points.",
            "parameters": {
                "type": "object",
                "properties": {
                    "origin": {"type": "string", "description": "Start address or coordinates"},
                    "destination": {"type": "string", "description": "End address or coordinates"},
                    "mode": {"type": "string", "description": "driving, walking, bicycling, transit (default driving)"},
                    "include_steps": {"type": "boolean", "description": "Include step-by-step directions (default false)"},
                },
                "required": ["origin", "destination"],
            },
        },
        {
            "name": "local_rank_tracker",
            "description": (
                "Track a business's local search ranking across a geographic grid. "
                "Shows rank at each point, top-3 competitors, and visibility metrics. "
                "Useful for local SEO analysis."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "keyword": {"type": "string", "description": "Search keyword to rank for"},
                    "place_id": {"type": "string", "description": "Your business's place_id to track"},
                    "latitude": {"type": "number", "description": "Center latitude"},
                    "longitude": {"type": "number", "description": "Center longitude"},
                    "grid_size": {"type": "integer", "description": "Grid dimension (3-7, default 3 = 3x3 grid)"},
                    "grid_spacing": {"type": "integer", "description": "Distance between grid points in meters (default 1000)"},
                },
                "required": ["keyword", "place_id", "latitude", "longitude"],
            },
        },
    ]

    handlers = {
        "search_places": search_places,
        "search_nearby": search_nearby,
        "place_details": place_details,
        "geocode": geocode,
        "reverse_geocode": reverse_geocode,
        "batch_geocode": batch_geocode,
        "distance_matrix": distance_matrix,
        "directions": directions,
        "local_rank_tracker": local_rank_tracker,
    }

    registry.add_namespace(
        name="google_maps",
        description=NAMESPACE_DESCRIPTION,
        tools=tools,
        handlers=handlers,
    )
