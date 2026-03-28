"""
Apollo.io API client — thin async wrapper for people/company search and enrichment.

Key design:
- People Search (mixed_people/api_search) is FREE — no credits consumed
- Enrichment costs 1 export credit per person/company
- Company Search costs credits
- Rate limits: 50-200 req/min depending on plan (response headers tell you)
- Pagination: page 1-500, per_page max 100, 50K display limit

All filters from Apollo's API are exposed as kwargs. The client builds the
payload dict from non-None values, so callers only pass what they need.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import httpx

logger = logging.getLogger(__name__)

APOLLO_BASE = "https://api.apollo.io/api/v1"

# Default timeout for API calls (seconds)
DEFAULT_TIMEOUT = 30.0


def _set_if(payload: Dict, key: str, value: Any) -> None:
    """Set payload[key] = value if value is truthy."""
    if value:
        payload[key] = value


class ApolloClient:
    """Async Apollo.io API client."""

    def __init__(self, api_key: str, timeout: float = DEFAULT_TIMEOUT) -> None:
        self._api_key = api_key
        self._http = httpx.AsyncClient(
            timeout=timeout,
            headers={
                "x-api-key": api_key,
                "Content-Type": "application/json",
            },
        )

    # ── People Search (FREE — no credits) ─────────────────────────────

    async def search_people(
        self,
        # Person filters
        person_titles: Optional[List[str]] = None,
        person_seniorities: Optional[List[str]] = None,
        person_locations: Optional[List[str]] = None,
        person_names: Optional[List[str]] = None,
        contact_email_status: Optional[List[str]] = None,
        department_ids: Optional[List[str]] = None,
        include_similar_titles: Optional[bool] = None,
        # Organization filters
        organization_keywords: Optional[List[str]] = None,
        organization_name: Optional[str] = None,
        organization_locations: Optional[List[str]] = None,
        organization_not_locations: Optional[List[str]] = None,
        organization_num_employees_ranges: Optional[List[str]] = None,
        organization_ids: Optional[List[str]] = None,
        organization_domains: Optional[List[str]] = None,
        organization_revenue_ranges: Optional[List[str]] = None,
        industry_tag_ids: Optional[List[str]] = None,
        technology_uids: Optional[List[str]] = None,
        # Free text
        q_keywords: Optional[str] = None,
        # Pagination
        per_page: int = 100,
        page: int = 1,
    ) -> Tuple[List[Dict[str, Any]], int]:
        """
        Search Apollo's 210M+ contact database. FREE — no credits consumed.

        Returns (people_list, total_entries).
        People have basic info (name, title, company, LinkedIn, location)
        but NOT emails or phone numbers — use enrich_person() for that.
        """
        payload: Dict[str, Any] = {
            "per_page": min(per_page, 100),
            "page": page,
        }
        _set_if(payload, "person_titles", person_titles)
        _set_if(payload, "person_seniorities", person_seniorities)
        _set_if(payload, "person_locations", person_locations)
        _set_if(payload, "person_names", person_names)
        _set_if(payload, "contact_email_status", contact_email_status)
        _set_if(payload, "department_ids", department_ids)
        if include_similar_titles is not None:
            payload["include_similar_titles"] = include_similar_titles
        _set_if(payload, "q_organization_keyword_tags", organization_keywords)
        _set_if(payload, "q_organization_name", organization_name)
        _set_if(payload, "organization_locations", organization_locations)
        _set_if(payload, "organization_not_locations", organization_not_locations)
        _set_if(payload, "organization_num_employees_ranges", organization_num_employees_ranges)
        _set_if(payload, "organization_ids", organization_ids)
        _set_if(payload, "q_organization_domains_list", organization_domains)
        _set_if(payload, "organization_revenue_ranges", organization_revenue_ranges)
        _set_if(payload, "industry_tag_ids", industry_tag_ids)
        _set_if(payload, "currently_using_any_of_technology_uids", technology_uids)
        _set_if(payload, "q_keywords", q_keywords)

        try:
            resp = await self._http.post(
                f"{APOLLO_BASE}/mixed_people/api_search",
                json=payload,
            )
            resp.raise_for_status()
            data = resp.json()

            people = data.get("people", [])
            total = data.get("pagination", {}).get("total_entries", 0)

            logger.info(
                f"[Apollo] search_people: page {page}, "
                f"{len(people)} results, {total} total"
            )
            return people, total

        except httpx.HTTPStatusError as e:
            logger.error(f"[Apollo] search_people HTTP {e.response.status_code}: {e.response.text[:300]}")
            raise
        except Exception as e:
            logger.error(f"[Apollo] search_people error: {e}")
            raise

    # ── Company Search (costs credits) ────────────────────────────────

    async def search_companies(
        self,
        organization_keywords: Optional[List[str]] = None,
        organization_name: Optional[str] = None,
        organization_locations: Optional[List[str]] = None,
        organization_not_locations: Optional[List[str]] = None,
        organization_num_employees_ranges: Optional[List[str]] = None,
        organization_revenue_ranges: Optional[List[str]] = None,
        organization_latest_funding_stage_cd: Optional[List[str]] = None,
        technology_uids: Optional[List[str]] = None,
        website_urls: Optional[List[str]] = None,
        industry_tag_ids: Optional[List[str]] = None,
        founded_year_min: Optional[int] = None,
        founded_year_max: Optional[int] = None,
        publicly_traded: Optional[bool] = None,
        per_page: int = 100,
        page: int = 1,
    ) -> Tuple[List[Dict[str, Any]], int]:
        """
        Search Apollo's 30M+ company database. Costs credits.

        Returns (organizations_list, total_entries).
        """
        payload: Dict[str, Any] = {
            "per_page": min(per_page, 100),
            "page": page,
        }
        _set_if(payload, "q_organization_keyword_tags", organization_keywords)
        _set_if(payload, "q_organization_name", organization_name)
        _set_if(payload, "organization_locations", organization_locations)
        _set_if(payload, "organization_not_locations", organization_not_locations)
        _set_if(payload, "organization_num_employees_ranges", organization_num_employees_ranges)
        _set_if(payload, "organization_revenue_ranges", organization_revenue_ranges)
        _set_if(payload, "organization_latest_funding_stage_cd", organization_latest_funding_stage_cd)
        _set_if(payload, "currently_using_any_of_technology_uids", technology_uids)
        _set_if(payload, "website_urls", website_urls)
        _set_if(payload, "industry_tag_ids", industry_tag_ids)
        if founded_year_min is not None:
            payload["founded_year_min"] = founded_year_min
        if founded_year_max is not None:
            payload["founded_year_max"] = founded_year_max
        if publicly_traded is not None:
            payload["publicly_traded"] = publicly_traded

        try:
            resp = await self._http.post(
                f"{APOLLO_BASE}/mixed_companies/search",
                json=payload,
            )
            resp.raise_for_status()
            data = resp.json()

            orgs = data.get("organizations", [])
            total = data.get("pagination", {}).get("total_entries", 0)

            logger.info(
                f"[Apollo] search_companies: page {page}, "
                f"{len(orgs)} results, {total} total"
            )
            return orgs, total

        except httpx.HTTPStatusError as e:
            logger.error(f"[Apollo] search_companies HTTP {e.response.status_code}: {e.response.text[:300]}")
            raise
        except Exception as e:
            logger.error(f"[Apollo] search_companies error: {e}")
            raise

    # ── People Enrichment (1 credit per person) ───────────────────────

    async def enrich_person(
        self,
        apollo_id: Optional[str] = None,
        first_name: Optional[str] = None,
        last_name: Optional[str] = None,
        name: Optional[str] = None,
        email: Optional[str] = None,
        organization_name: Optional[str] = None,
        domain: Optional[str] = None,
        linkedin_url: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Enrich a person with full contact data. Costs 1 export credit.

        Match strength: email > domain+name > LinkedIn URL > name alone.
        Returns the enriched person dict, or None if not found.
        """
        payload: Dict[str, Any] = {}
        _set_if(payload, "id", apollo_id)
        _set_if(payload, "first_name", first_name)
        _set_if(payload, "last_name", last_name)
        _set_if(payload, "name", name)
        _set_if(payload, "email", email)
        _set_if(payload, "organization_name", organization_name)
        _set_if(payload, "domain", domain)
        _set_if(payload, "linkedin_url", linkedin_url)

        try:
            resp = await self._http.post(
                f"{APOLLO_BASE}/people/match",
                json=payload,
            )
            resp.raise_for_status()
            data = resp.json()
            person = data.get("person")

            if person:
                logger.info(
                    f"[Apollo] enrich_person: {person.get('name', '?')} "
                    f"at {person.get('organization', {}).get('name', '?')}"
                )
            else:
                logger.info("[Apollo] enrich_person: no match found")

            return person

        except httpx.HTTPStatusError as e:
            logger.error(f"[Apollo] enrich_person HTTP {e.response.status_code}: {e.response.text[:300]}")
            return None
        except Exception as e:
            logger.error(f"[Apollo] enrich_person error: {e}")
            return None

    # ── Company Enrichment (1 credit per company) ─────────────────────

    async def enrich_company(
        self,
        domain: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Enrich a company by domain. Costs 1 export credit.

        Returns full organization data or None.
        """
        try:
            resp = await self._http.get(
                f"{APOLLO_BASE}/organizations/enrich",
                params={"domain": domain},
            )
            resp.raise_for_status()
            data = resp.json()
            org = data.get("organization")

            if org:
                logger.info(f"[Apollo] enrich_company: {org.get('name', '?')}")
            else:
                logger.info(f"[Apollo] enrich_company: no match for {domain}")

            return org

        except httpx.HTTPStatusError as e:
            logger.error(f"[Apollo] enrich_company HTTP {e.response.status_code}: {e.response.text[:300]}")
            return None
        except Exception as e:
            logger.error(f"[Apollo] enrich_company error: {e}")
            return None

    # ── Job Postings ──────────────────────────────────────────────────

    async def get_job_postings(
        self,
        organization_id: str,
        per_page: int = 100,
        page: int = 1,
    ) -> Tuple[List[Dict[str, Any]], int]:
        """
        Get job postings for an organization. Display limit 10K.

        Returns (postings_list, total_entries).
        """
        try:
            resp = await self._http.get(
                f"{APOLLO_BASE}/organizations/{organization_id}/job_postings",
                params={"per_page": min(per_page, 100), "page": page},
            )
            resp.raise_for_status()
            data = resp.json()

            postings = data.get("job_postings", [])
            total = data.get("pagination", {}).get("total_entries", 0)

            logger.info(f"[Apollo] job_postings: {len(postings)} results, {total} total")
            return postings, total

        except httpx.HTTPStatusError as e:
            logger.error(f"[Apollo] job_postings HTTP {e.response.status_code}: {e.response.text[:300]}")
            return [], 0
        except Exception as e:
            logger.error(f"[Apollo] job_postings error: {e}")
            return [], 0

    # ── Cleanup ───────────────────────────────────────────────────────

    async def close(self) -> None:
        """Close the HTTP client."""
        await self._http.aclose()
