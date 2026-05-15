"""apollo_companies — Apollo organization search.

The biggest free-data win in the redesign. Apollo's /mixed_companies/search
returns ~30 fields per company including revenue, phone, growth metrics,
funding signals, NAICS/SIC codes. Free in practice within our plan
(empirically ~1 credit per 25 results, ~$0.001/company).

Pagination: standard `page` + `per_page`. Cursor stores the next page number.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

import httpx

from dsl_worker.sources_v2.base import FetchResult, SourceAdapter, register


log = logging.getLogger(__name__)

APOLLO_BASE = "https://api.apollo.io/api/v1"


# Field map for the search response → table columns. Field names match the
# Apollo response shape; column names are user-friendly.
DEFAULT_COLUMNS = [
    {"source_field": "name", "column_name": "name", "type": "text"},
    {"source_field": "primary_domain", "column_name": "domain", "type": "url"},
    {"source_field": "website_url", "column_name": "website", "type": "url"},
    {"source_field": "estimated_num_employees", "column_name": "employees", "type": "number"},
    {"source_field": "organization_revenue_printed", "column_name": "revenue", "type": "text"},
    {"source_field": "industry", "column_name": "industry", "type": "text"},
    {"source_field": "phone", "column_name": "phone", "type": "text"},
    {"source_field": "linkedin_url", "column_name": "linkedin_url", "type": "url"},
    {"source_field": "twitter_url", "column_name": "twitter_url", "type": "url"},
    {"source_field": "founded_year", "column_name": "founded_year", "type": "number"},
    {"source_field": "city", "column_name": "city", "type": "text"},
    {"source_field": "state", "column_name": "state", "type": "text"},
    {"source_field": "country", "column_name": "country", "type": "text"},
    {"source_field": "organization_headcount_six_month_growth", "column_name": "headcount_growth_6m", "type": "number"},
    {"source_field": "organization_headcount_twelve_month_growth", "column_name": "headcount_growth_12m", "type": "number"},
    {"source_field": "naics_codes", "column_name": "naics_codes", "type": "text"},
    {"source_field": "logo_url", "column_name": "logo_url", "type": "url"},
]


# Fields the agent can use in query_params. Anything not in this set is
# rejected at validation time with an actionable hint.
ALLOWED_PARAMS = {
    "organization_locations",
    "organization_not_locations",
    "organization_num_employees_ranges",
    "revenue_range",
    "q_organization_keyword_tags",
    "currently_using_any_of_technology_uids",
    "latest_funding_amount_range",
    "latest_funding_date_range",
    "total_funding_range",
    "q_organization_job_titles",
    "organization_num_jobs_range",
    "organization_job_posted_at_range",
    "q_organization_domains_list",
    "q_organization_name",
    "page",
    "per_page",
}


class ApolloCompaniesAdapter(SourceAdapter):
    name = "apollo_companies"
    predictable = True
    default_columns = DEFAULT_COLUMNS
    default_dedup_key_column = "domain"

    def __init__(self) -> None:
        self.api_key = os.getenv("APOLLO_API_KEY")
        if not self.api_key:
            log.warning("APOLLO_API_KEY not set — apollo_companies adapter inert")

    def validate_query_params(self, query_params: Dict[str, Any]) -> Optional[str]:
        bad = [k for k in query_params if k not in ALLOWED_PARAMS]
        if bad:
            return (
                f"unknown apollo_companies params: {bad}. "
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

        # Apollo caps per_page at 100. If n>100, paginate internally to fill n.
        per_page = min(100, max(1, n))
        # Resume from prior_cursor.page if present (1-indexed).
        page = int((prior_cursor or {}).get("page", 1))
        target = n

        all_rows: List[Dict[str, Any]] = []
        total_entries: Optional[int] = None

        async with httpx.AsyncClient(timeout=30.0) as client:
            while len(all_rows) < target:
                body = {
                    **{k: v for k, v in query_params.items() if k not in ("page", "per_page")},
                    "page": page,
                    "per_page": min(per_page, target - len(all_rows)),
                }
                resp = await client.post(
                    f"{APOLLO_BASE}/mixed_companies/search",
                    headers={"X-Api-Key": self.api_key, "Content-Type": "application/json"},
                    json=body,
                )
                if resp.status_code != 200:
                    log.warning("apollo_companies HTTP %s: %s", resp.status_code, resp.text[:200])
                    break
                data = resp.json() or {}
                orgs = data.get("organizations") or []
                if not orgs:
                    break
                all_rows.extend(orgs)
                pagination = data.get("pagination") or {}
                total_entries = pagination.get("total_entries")
                page += 1
                # Apollo caps at 500 pages (50k records); also stop when fewer
                # rows came back than requested.
                if len(orgs) < body["per_page"]:
                    break
                if page > 500:
                    break

        # Trim to exactly n requested
        all_rows = all_rows[:target]

        # exhausted: either page>500 OR total_entries says we've seen them all
        exhausted = page > 500 or (
            total_entries is not None and len(all_rows) + ((prior_cursor or {}).get("seen", 0)) >= total_entries
        )

        # Cost: Apollo doesn't include per-call cost in the response.
        # Estimate from plan economics: typical Apollo plan ~$0.05/org returned
        # (mixed_companies_search consumes ~1 Apollo-credit per org;
        # Apollo plans range $0.03-0.07/credit). 1 our-credit = $0.10 of
        # compute, so 0.05/org → 0.5 our-credit per org. Override via env.
        cost_per_org_usd = float(os.getenv("APOLLO_COST_USD_PER_ORG", "0.05"))
        cost_credits = len(all_rows) * cost_per_org_usd * 10.0

        return FetchResult(
            rows=all_rows,
            schema=sorted({k for r in all_rows for k in r.keys()})[:60],
            cost_credits=cost_credits,
            exhausted=exhausted,
            cursor={"page": page, "seen": ((prior_cursor or {}).get("seen", 0)) + len(all_rows)},
            dedup_key_column_hint="domain",
        )


register(ApolloCompaniesAdapter())
