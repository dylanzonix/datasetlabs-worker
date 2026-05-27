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

from dsl_worker.sources.base import FetchResult, SourceAdapter, SourceDescription, register


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
#
# Each param below has been empirically confirmed via Apollo's API
# (verified by checking that the param actually changes total_entries).
# Apollo's public docs list ~17 params; the rest are undocumented but
# functional — discovered by probing the /mixed_companies/search endpoint.
ALLOWED_PARAMS = {
    # --- documented & widely used ---
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
    "organization_ids",
    "organization_job_locations",
    "page",
    "per_page",
    # --- undocumented but empirically working (added 2026-05-26) ---
    # Strict industry filters — vastly more precise than the loose
    # q_organization_keyword_tags. Use these whenever the user names
    # a specific industry/sector. ID-form takes Apollo hash IDs.
    "organization_industries",
    "organization_industry_tag_ids",
    # Founding-year window. {min: "2010", max: "2020"} string years.
    "organization_founded_year_range",
    # NAICS / SIC industry codes — public standardized classification.
    # Strict and reliable when the user supplies/asks for code-based
    # industry targeting.
    "organization_naics_codes",
    "organization_sic_codes",
    # Latest funding stage as Apollo string code (e.g. "0"=Seed,
    # "1"=Series A, "2"=Series B, ...). Use sparingly until we ship a
    # friendly name → code resolver.
    "organization_latest_funding_stage_cd",
    # AND-semantics tech filter — every company must use every listed
    # technology. Pair with the OR variant (currently_using_any_of_...)
    # when you want strict-fit vs broad-match.
    "currently_using_all_of_technology_uids",
}


class ApolloCompaniesAdapter(SourceAdapter):
    name = "apollo_companies"
    label = "Apollo Companies"
    favicon_url = "https://www.google.com/s2/favicons?domain=apollo.io&sz=32"
    predictable = True
    default_columns = DEFAULT_COLUMNS
    default_dedup_key_column = "domain"

    def describe(
        self,
        query_params: Dict[str, Any],
        source: Optional[str] = None,
    ) -> SourceDescription:
        qp = query_params or {}
        # Headline = the dominant filter (keyword > job titles > name > locations).
        headline_parts: List[str] = []
        kw = qp.get("q_organization_keyword_tags")
        if kw:
            headline_parts.append(", ".join(map(str, kw)) if isinstance(kw, list) else str(kw))
        titles = qp.get("q_organization_job_titles")
        if titles:
            headline_parts.append("hiring " + (", ".join(map(str, titles)) if isinstance(titles, list) else str(titles)))
        name = qp.get("q_organization_name")
        if name:
            headline_parts.append(f'named "{name}"')
        locs = qp.get("organization_locations")
        if locs:
            headline_parts.append("in " + (", ".join(map(str, locs[:3])) if isinstance(locs, list) else str(locs)))
        headline = "Companies — " + " · ".join(headline_parts) if headline_parts else "Apollo company search"

        # Details: every applied filter, one bullet each.
        detail_lines: List[str] = []
        FRIENDLY = {
            "organization_locations": "Locations",
            "organization_not_locations": "Excluded locations",
            "organization_num_employees_ranges": "Headcount",
            "revenue_range": "Revenue",
            "q_organization_keyword_tags": "Keywords",
            "currently_using_any_of_technology_uids": "Tech stack (any of)",
            "currently_using_all_of_technology_uids": "Tech stack (all of)",
            "latest_funding_amount_range": "Latest funding amount",
            "latest_funding_date_range": "Latest funding date",
            "total_funding_range": "Total funding",
            "q_organization_job_titles": "Open job titles",
            "organization_num_jobs_range": "Open jobs",
            "organization_job_posted_at_range": "Jobs posted",
            "q_organization_domains_list": "Domains",
            "q_organization_name": "Name match",
            "organization_ids": "Apollo org IDs",
            "organization_job_locations": "Job posting locations",
            "organization_industries": "Industries",
            "organization_industry_tag_ids": "Industry tag IDs",
            "organization_founded_year_range": "Founded year",
            "organization_naics_codes": "NAICS codes",
            "organization_sic_codes": "SIC codes",
            "organization_latest_funding_stage_cd": "Funding stage",
        }
        for k, v in qp.items():
            if k in ("page", "per_page"):
                continue
            label_k = FRIENDLY.get(k, k)
            if isinstance(v, list):
                val = ", ".join(map(str, v))
            else:
                val = str(v)
            detail_lines.append(f"- **{label_k}:** {val}")
        return SourceDescription(
            kind=self.name,
            label=self.label,
            query_text=headline,
            details="\n".join(detail_lines),
            favicon_url=self.favicon_url,
        )

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
                total_pages = pagination.get("total_pages")
                # Apollo can return slightly fewer than per_page (often 98-99
                # at per_page=100) even mid-stream — they appear to dedupe or
                # filter within a page. So we CANNOT use "len(orgs) < per_page"
                # as end-of-results: that fired on every Apollo pull and
                # silently capped them at ~98 rows. Correct end-of-results
                # signal is the pagination block itself.
                page += 1
                if total_pages is not None and page > total_pages:
                    break
                if page > 500:  # Apollo's hard cap (50k records)
                    break

        # Trim to exactly n requested
        all_rows = all_rows[:target]

        # exhausted: either page>500 OR total_entries says we've seen them all
        exhausted = page > 500 or (
            total_entries is not None and len(all_rows) + ((prior_cursor or {}).get("seen", 0)) >= total_entries
        )

        # Cost: mixed_companies_search is REQUEST-QUOTA-LIMITED, not
        # credit-billed. Apollo's response headers confirm this:
        #
        #   x-rate-limit-24-hour: 50000   (50k requests/day on this plan)
        #   x-24-hour-requests-left: 49997
        #   x-rate-limit-hourly:    6000
        #   x-rate-limit-minute:    200
        #
        # No header indicates per-call "credit" spend, because the user's
        # Apollo monthly credit allowance (e.g. 5,090 credits/mo on $109)
        # is reserved for EXPORT / EMAIL+PHONE REVEAL endpoints — NOT for
        # search. mixed_companies_search returns company data within the
        # request quota at no per-call charge.
        #
        # So this fetch path is effectively free. We record $0 cost.
        # The per-row enrichment path (cell agent → fullenrich / apollo
        # people-match with reveal_personal_emails=true) IS where credit
        # billing lives, and is tracked separately at that handler.
        #
        # Env knobs kept in case we ever want to add a flat infra fee
        # per-fetch (e.g. to cover our own model + parse compute), but
        # default 0 means we don't double-bill the user.
        usd_per_apollo_credit = float(os.getenv("APOLLO_USD_PER_CREDIT", "0.02141"))
        apollo_credits_per_org = float(os.getenv("APOLLO_CREDITS_PER_ORG", "0"))
        cost_usd = len(all_rows) * apollo_credits_per_org * usd_per_apollo_credit
        cost_credits = cost_usd * 10.0

        return FetchResult(
            rows=all_rows,
            schema=sorted({k for r in all_rows for k in r.keys()})[:60],
            cost_credits=cost_credits,
            exhausted=exhausted,
            cursor={"page": page, "seen": ((prior_cursor or {}).get("seen", 0)) + len(all_rows)},
            dedup_key_column_hint="domain",
        )


register(ApolloCompaniesAdapter())
