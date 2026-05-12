"""fullenrich_people — FullEnrich people search.

People with full names (no obfuscation), structured filters (titles,
seniorities, geo, company filters, tech stack, etc). ~0.25 credit per match
on first page. Pagination via `search_after` cursor (preferred) or `offset`
(capped at 10k).
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

import httpx

from dsl_worker.sources_v2.base import FetchResult, SourceAdapter, register


log = logging.getLogger(__name__)

FE_BASE = "https://app.fullenrich.com"


DEFAULT_COLUMNS = [
    {"source_field": "first_name", "column_name": "first_name", "type": "text"},
    {"source_field": "last_name", "column_name": "last_name", "type": "text"},
    {"source_field": "title", "column_name": "title", "type": "text"},
    {"source_field": "headline", "column_name": "headline", "type": "text"},
    {"source_field": "linkedin_url", "column_name": "linkedin_url", "type": "url"},
    {"source_field": "company_name", "column_name": "company", "type": "text"},
    {"source_field": "company_domain", "column_name": "company_domain", "type": "url"},
    {"source_field": "city", "column_name": "city", "type": "text"},
    {"source_field": "state", "column_name": "state", "type": "text"},
    {"source_field": "country", "column_name": "country", "type": "text"},
    {"source_field": "seniority", "column_name": "seniority", "type": "enum"},
    {"source_field": "departments", "column_name": "departments", "type": "text"},
]


ALLOWED_PARAMS = {
    "job_titles", "seniorities", "departments",
    "person_locations", "company_locations",
    "company_names", "company_domains",
    "company_industries", "company_headcounts",
    "currently_using_any_of_technology_uids",
    "contact_email_status",
    "limit", "offset", "search_after",
}


class FullEnrichPeopleAdapter(SourceAdapter):
    name = "fullenrich_people"
    predictable = True
    default_columns = DEFAULT_COLUMNS
    default_dedup_key_column = "linkedin_url"

    def __init__(self) -> None:
        self.api_key = os.getenv("FULLENRICH_API_KEY")
        if not self.api_key:
            log.warning("FULLENRICH_API_KEY not set — fullenrich_people adapter inert")

    def validate_query_params(self, query_params: Dict[str, Any]) -> Optional[str]:
        bad = [k for k in query_params if k not in ALLOWED_PARAMS]
        if bad:
            return (
                f"unknown fullenrich_people params: {bad}. "
                f"Allowed: {sorted(ALLOWED_PARAMS)}"
            )
        return None

    async def fetch(
        self,
        query_params: Dict[str, Any],
        n: int,
        prior_cursor: Optional[Dict[str, Any]] = None,
    ) -> FetchResult:
        if not self.api_key:
            return FetchResult(rows=[], schema=[], cost_credits=0.0, exhausted=True)

        # FE caps `limit` at 100 per call. If n>100, paginate using
        # search_after cursor when the API returns one; else fall back to
        # offset.
        page_size = min(100, max(1, n))
        search_after = (prior_cursor or {}).get("search_after")
        offset = int((prior_cursor or {}).get("offset", 0))
        all_rows: List[Dict[str, Any]] = []

        async with httpx.AsyncClient(timeout=30.0) as client:
            while len(all_rows) < n:
                body: Dict[str, Any] = {
                    **{k: v for k, v in query_params.items() if k not in ("limit", "offset", "search_after")},
                    "limit": min(page_size, n - len(all_rows)),
                }
                if search_after:
                    body["search_after"] = search_after
                else:
                    body["offset"] = offset

                resp = await client.post(
                    f"{FE_BASE}/api/v2/people/search",
                    headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
                    json=body,
                )
                if resp.status_code != 200:
                    log.warning("fullenrich_people HTTP %s: %s", resp.status_code, resp.text[:200])
                    break
                data = resp.json() or {}
                people = data.get("people") or []
                if not people:
                    break
                all_rows.extend(people)
                meta = data.get("metadata") or {}
                next_after = meta.get("search_after")
                if next_after:
                    search_after = next_after
                else:
                    offset += len(people)
                if len(people) < body["limit"]:
                    break

        # Cost: 0.25 credit per match in search
        cost_credits = 0.25 * len(all_rows)
        return FetchResult(
            rows=all_rows[:n],
            schema=sorted({k for r in all_rows for k in r.keys()})[:60],
            cost_credits=cost_credits,
            exhausted=len(all_rows) < n,
            cursor={"search_after": search_after, "offset": offset},
            dedup_key_column_hint="linkedin_url",
        )


register(FullEnrichPeopleAdapter())
