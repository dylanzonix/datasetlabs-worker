"""
FullEnrich API client.

Handles people search, company search, and contact enrichment via
FullEnrich's waterfall enrichment (20+ providers).

Search is nearly free (fractional credits). Enrichment costs 1 credit
per valid email, 10 credits per valid phone.
"""

import asyncio
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx

logger = logging.getLogger(__name__)

BASE_URL = "https://app.fullenrich.com"


class FullEnrichClient:
    def __init__(self, api_key: str, timeout: float = 30.0) -> None:
        self._api_key = api_key
        self._timeout = timeout
        self._headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

    # ── Account ──────────────────────────────────────────────────────

    async def get_credits(self) -> float:
        """Get current credit balance."""
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            resp = await client.get(
                f"{BASE_URL}/api/v2/account/credits",
                headers=self._headers,
            )
        if resp.status_code != 200:
            return 0.0
        return resp.json().get("balance", 0.0)

    # ── People Search ────────────────────────────────────────────────

    async def search_people(
        self,
        filters: Dict[str, Any],
        limit: int = 25,
        offset: int = 0,
        search_after: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Search for people with advanced filters.

        Filters are passed through to FullEnrich's API. The main filters:
        - current_position_titles: [{"value": "CEO", "exact_match": false, "exclude": false}]
        - person_locations: [{"value": "San Francisco", ...}]
        - current_company_industries: [{"value": "Software", ...}]
        - current_position_seniority_level: [{"value": "C-Level", ...}]
        - current_company_headcounts: [{"min": 50, "max": 500, "exclude": false}]
        - person_skills, person_universities, current_company_names, etc.

        Returns dict with 'people' list and 'metadata' (total, credits, offset).
        """
        body: Dict[str, Any] = {**filters, "limit": limit, "offset": offset}
        if search_after:
            body["search_after"] = search_after

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            resp = await client.post(
                f"{BASE_URL}/api/v2/people/search",
                headers=self._headers,
                json=body,
            )

        if resp.status_code != 200:
            return {"error": f"HTTP {resp.status_code}: {resp.text[:200]}", "people": [], "metadata": {}}

        return resp.json()

    async def search_people_to_file(
        self,
        filters: Dict[str, Any],
        output_path: Path,
        max_results: int = 100,
    ) -> Dict[str, Any]:
        """Search people and write all results to a JSONL file.

        Auto-paginates using search_after cursor. Stops at max_results.
        """
        all_people: List[Dict] = []
        search_after = None
        page_size = min(max_results, 100)

        while len(all_people) < max_results:
            remaining = max_results - len(all_people)
            limit = min(page_size, remaining)

            result = await self.search_people(
                filters, limit=limit, search_after=search_after,
            )

            people = result.get("people", [])
            if not people:
                break

            all_people.extend(people)
            metadata = result.get("metadata", {})
            search_after = metadata.get("search_after")

            if not search_after or len(people) < limit:
                break

        if not all_people:
            return {"item_count": 0, "total": result.get("metadata", {}).get("total", 0)}

        # Write JSONL
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            for person in all_people:
                f.write(json.dumps(person, ensure_ascii=False, default=str) + "\n")

        total = result.get("metadata", {}).get("total", len(all_people))
        credits_used = sum(
            r.get("metadata", {}).get("credits", 0)
            for r in [result]  # last page metadata
        )

        return {
            "item_count": len(all_people),
            "total": total,
            "file_path": str(output_path),
            "credits_used": credits_used,
            "fields": _extract_fields(all_people[0]) if all_people else [],
        }

    # ── Company Search ───────────────────────────────────────────────

    async def search_companies(
        self,
        filters: Dict[str, Any],
        limit: int = 25,
        offset: int = 0,
        search_after: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Search for companies with filters.

        Filters:
        - industries: [{"value": "Software Development", ...}]
        - headquarters_locations: [{"value": "California", ...}]
        - headcounts: [{"min": 50, "max": 500, "exclude": false}]
        - specialties, keywords, names, domains, types, founded_years, etc.
        """
        body: Dict[str, Any] = {**filters, "limit": limit, "offset": offset}
        if search_after:
            body["search_after"] = search_after

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            resp = await client.post(
                f"{BASE_URL}/api/v2/company/search",
                headers=self._headers,
                json=body,
            )

        if resp.status_code != 200:
            return {"error": f"HTTP {resp.status_code}: {resp.text[:200]}", "companies": [], "metadata": {}}

        return resp.json()

    async def search_companies_to_file(
        self,
        filters: Dict[str, Any],
        output_path: Path,
        max_results: int = 100,
    ) -> Dict[str, Any]:
        """Search companies and write all results to a JSONL file."""
        all_companies: List[Dict] = []
        search_after = None
        page_size = min(max_results, 100)

        while len(all_companies) < max_results:
            remaining = max_results - len(all_companies)
            limit = min(page_size, remaining)

            result = await self.search_companies(
                filters, limit=limit, search_after=search_after,
            )

            companies = result.get("companies", [])
            if not companies:
                break

            all_companies.extend(companies)
            metadata = result.get("metadata", {})
            search_after = metadata.get("search_after")

            if not search_after or len(companies) < limit:
                break

        if not all_companies:
            return {"item_count": 0, "total": result.get("metadata", {}).get("total", 0)}

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            for company in all_companies:
                f.write(json.dumps(company, ensure_ascii=False, default=str) + "\n")

        total = result.get("metadata", {}).get("total", len(all_companies))

        return {
            "item_count": len(all_companies),
            "total": total,
            "file_path": str(output_path),
            "fields": _extract_fields(all_companies[0]) if all_companies else [],
        }

    # ── Contact Enrichment ───────────────────────────────────────────

    async def enrich_contacts(
        self,
        contacts: List[Dict[str, Any]],
        name: str = "api_enrichment",
        enrich_fields: Optional[List[str]] = None,
        poll_interval: int = 5,
        poll_timeout: int = 120,
    ) -> Dict[str, Any]:
        """Enrich contacts via waterfall (20+ providers).

        Each contact needs either:
        - first_name + last_name + (domain or company_name), OR
        - linkedin_url

        enrich_fields defaults to ["contact.emails", "contact.phones"].

        Returns the full enrichment result with contact_info and profile data.
        """
        if not enrich_fields:
            enrich_fields = ["contact.emails", "contact.phones"]

        # Add enrich_fields to each contact if not already set
        data = []
        for contact in contacts:
            entry = {**contact}
            if "enrich_fields" not in entry:
                entry["enrich_fields"] = enrich_fields
            data.append(entry)

        # Submit enrichment
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            resp = await client.post(
                f"{BASE_URL}/api/v2/contact/enrich/bulk",
                headers=self._headers,
                json={"name": name, "data": data},
            )

        if resp.status_code != 200:
            return {"error": f"HTTP {resp.status_code}: {resp.text[:200]}"}

        enrichment_id = resp.json().get("enrichment_id")
        if not enrichment_id:
            return {"error": "No enrichment_id returned", "raw": resp.json()}

        # Poll for results
        elapsed = 0
        while elapsed < poll_timeout:
            await asyncio.sleep(poll_interval)
            elapsed += poll_interval

            async with httpx.AsyncClient(timeout=self._timeout) as client:
                r = await client.get(
                    f"{BASE_URL}/api/v2/contact/enrich/bulk/{enrichment_id}",
                    headers=self._headers,
                )

            if r.status_code != 200:
                continue

            result = r.json()
            status = result.get("status", "")

            if status == "FINISHED":
                return result
            if status in ("CANCELED", "CREDITS_INSUFFICIENT", "RATE_LIMIT"):
                return {"error": f"Enrichment {status}", **result}

        return {"error": f"Timed out after {poll_timeout}s", "enrichment_id": enrichment_id}

    async def enrich_contacts_to_file(
        self,
        contacts: List[Dict[str, Any]],
        output_path: Path,
        name: str = "api_enrichment",
        enrich_fields: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Enrich contacts and write results to a JSONL file."""
        result = await self.enrich_contacts(
            contacts, name=name, enrich_fields=enrich_fields,
        )

        if "error" in result:
            return result

        data = result.get("data", [])
        if not data:
            return {"item_count": 0, "credits_used": result.get("cost", {}).get("credits", 0)}

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            for contact in data:
                f.write(json.dumps(contact, ensure_ascii=False, default=str) + "\n")

        credits_used = result.get("cost", {}).get("credits", 0)

        return {
            "item_count": len(data),
            "file_path": str(output_path),
            "credits_used": credits_used,
            "enrichment_id": result.get("id"),
        }


def _extract_fields(item: Dict[str, Any], max_fields: int = 15) -> List[str]:
    """Extract top-level field names from a result item."""
    if not item:
        return []
    return sorted(item.keys())[:max_fields]
