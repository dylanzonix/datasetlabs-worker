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

from dsl_worker.sources.base import FetchResult, SourceAdapter, SourceDescription, register


log = logging.getLogger(__name__)

FE_BASE = "https://app.fullenrich.com"


# FullEnrich's people-search response wraps everything in nested objects:
#   { id, full_name, first_name, last_name,
#     location: { country, country_code, city?, region? },
#     employment: { current: { title, seniority, company: { name, domain, ... } } },
#     social_profiles: { professional_network: { url } },  # LinkedIn
#     educations: [...], skills: [...] }
# These default mappings use dotted paths so the agent (and table_create's
# row mapper) reach into the nested shape correctly.
DEFAULT_COLUMNS = [
    {"source_field": "full_name", "column_name": "full_name", "type": "text"},
    {"source_field": "first_name", "column_name": "first_name", "type": "text"},
    {"source_field": "last_name", "column_name": "last_name", "type": "text"},
    {"source_field": "employment.current.title", "column_name": "title", "type": "text"},
    {"source_field": "employment.current.seniority", "column_name": "seniority", "type": "enum"},
    {"source_field": "employment.current.company.name", "column_name": "company", "type": "text"},
    {"source_field": "employment.current.company.domain", "column_name": "company_domain", "type": "url"},
    {"source_field": "social_profiles.professional_network.url", "column_name": "linkedin_url", "type": "url"},
    {"source_field": "location.city", "column_name": "city", "type": "text"},
    {"source_field": "location.region", "column_name": "state", "type": "text"},
    {"source_field": "location.country", "column_name": "country", "type": "text"},
]


ALLOWED_PARAMS = {
    # FE-canonical param names (see FE /api/v2/people/search):
    "current_position_titles",
    "current_position_seniority_level",
    "current_position_departments",
    "person_locations",
    "current_company_names",
    "current_company_domains",
    "current_company_industries",
    "current_company_headcounts",
    "current_company_locations",
    "person_skills",
    "person_universities",
    "currently_using_any_of_technology_uids",
    "contact_email_status",
    "limit", "offset", "search_after",
    # Friendly aliases the agent may use — adapter normalizes them:
    "job_titles", "titles",
    "seniorities", "seniority", "seniority_levels",
    "company_names", "company_domains",
    "company_headcounts", "headcounts",
    "company_industries", "industries",
    "departments",
}


# Friendly → canonical param name aliasing so the agent can use either.
PARAM_ALIASES = {
    "job_titles": "current_position_titles",
    "titles": "current_position_titles",
    "seniorities": "current_position_seniority_level",
    "seniority": "current_position_seniority_level",
    "seniority_levels": "current_position_seniority_level",
    "company_names": "current_company_names",
    "company_domains": "current_company_domains",
    "company_industries": "current_company_industries",
    "industries": "current_company_industries",
    "company_headcounts": "current_company_headcounts",
    "headcounts": "current_company_headcounts",
    "departments": "current_position_departments",
}


class FullEnrichPeopleAdapter(SourceAdapter):
    name = "fullenrich_people"
    label = "FullEnrich People"
    favicon_url = "https://www.google.com/s2/favicons?domain=fullenrich.com&sz=32"
    predictable = True
    default_columns = DEFAULT_COLUMNS
    default_dedup_key_column = "linkedin_url"

    @staticmethod
    def _stringify_filter(v: Any) -> str:
        """Pretty-print a FE filter value (may be a list of strings or
        {value, exact_match, exclude} dicts)."""
        if isinstance(v, list):
            parts = []
            for item in v:
                if isinstance(item, dict):
                    val = item.get("value")
                    if val is not None:
                        parts.append(("¬" if item.get("exclude") else "") + str(val))
                    elif "min" in item or "max" in item:
                        parts.append(f"{item.get('min', '')}–{item.get('max', '')}")
                else:
                    parts.append(str(item))
            return ", ".join(parts)
        return str(v)

    def describe(
        self,
        query_params: Dict[str, Any],
        source: Optional[str] = None,
    ) -> SourceDescription:
        qp = {PARAM_ALIASES.get(k, k): v for k, v in (query_params or {}).items()}
        headline_parts: List[str] = []
        titles = qp.get("current_position_titles")
        if titles:
            headline_parts.append(self._stringify_filter(titles))
        sen = qp.get("current_position_seniority_level")
        if sen:
            headline_parts.append(self._stringify_filter(sen))
        comp = qp.get("current_company_names") or qp.get("current_company_domains")
        if comp:
            headline_parts.append("at " + self._stringify_filter(comp))
        ind = qp.get("current_company_industries")
        if ind:
            headline_parts.append("in " + self._stringify_filter(ind))
        loc = qp.get("person_locations") or qp.get("current_company_locations")
        if loc:
            headline_parts.append("based in " + self._stringify_filter(loc))
        headline = "People — " + " · ".join(headline_parts) if headline_parts else "FullEnrich people search"

        FRIENDLY = {
            "current_position_titles": "Titles",
            "current_position_seniority_level": "Seniority",
            "current_position_departments": "Departments",
            "person_locations": "Person locations",
            "current_company_names": "Companies",
            "current_company_domains": "Company domains",
            "current_company_industries": "Industries",
            "current_company_headcounts": "Company headcount",
            "current_company_locations": "Company locations",
            "person_skills": "Skills",
            "person_universities": "Education",
            "currently_using_any_of_technology_uids": "Tech stack",
            "contact_email_status": "Email status",
        }
        detail_lines: List[str] = []
        for k, v in qp.items():
            if k in ("limit", "offset", "search_after"):
                continue
            label_k = FRIENDLY.get(k, k)
            detail_lines.append(f"- **{label_k}:** {self._stringify_filter(v)}")
        return SourceDescription(
            kind=self.name,
            label=self.label,
            query_text=headline,
            details="\n".join(detail_lines),
            favicon_url=self.favicon_url,
        )

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

    @staticmethod
    def _normalize_filter(value: Any) -> Any:
        """FE's API wants array filters as [{value, exact_match, exclude}, ...].
        The agent often sends bare strings like ["VP Sales"]; wrap them.

        For range-typed filters (headcounts), wrap each item as {min, max, exclude}
        if it's a bare {min, max} dict.
        """
        if isinstance(value, list):
            out = []
            for item in value:
                if isinstance(item, str):
                    out.append({"value": item, "exact_match": False, "exclude": False})
                elif isinstance(item, dict):
                    # Could be either a value-wrapper or a range-wrapper; pass through.
                    if "value" in item or "min" in item or "max" in item:
                        item.setdefault("exclude", False)
                        if "value" in item:
                            item.setdefault("exact_match", False)
                        out.append(item)
                    else:
                        out.append(item)
                else:
                    out.append(item)
            return out
        return value

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

        # Resolve friendly aliases → canonical FE param names.
        canonical = {PARAM_ALIASES.get(k, k): v for k, v in query_params.items()}

        # Normalize bare-array filters → FE's {value, exact_match, exclude} shape.
        NORMALIZE_KEYS = {
            "current_position_titles", "current_position_seniority_level",
            "current_position_departments", "person_locations",
            "current_company_names", "current_company_domains",
            "current_company_industries", "current_company_headcounts",
            "current_company_locations",
            "person_skills", "person_universities",
            "currently_using_any_of_technology_uids", "contact_email_status",
        }
        normalized = {
            k: (self._normalize_filter(v) if k in NORMALIZE_KEYS else v)
            for k, v in canonical.items()
        }

        async with httpx.AsyncClient(timeout=30.0) as client:
            while len(all_rows) < n:
                body: Dict[str, Any] = {
                    **{k: v for k, v in normalized.items() if k not in ("limit", "offset", "search_after")},
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

        # FE charges per match. Their published pricing varies by plan
        # (~$0.01-0.05/match for people-search). Use env-tunable per-match
        # USD; default $0.025/match. 1 our-credit = $0.10 of compute, so
        # $0.025 → 0.25 credits/match (matches the historical estimate).
        cost_per_match_usd = float(os.getenv("FULLENRICH_COST_USD_PER_MATCH", "0.025"))
        cost_credits = len(all_rows) * cost_per_match_usd * 10.0
        return FetchResult(
            rows=all_rows[:n],
            schema=sorted({k for r in all_rows for k in r.keys()})[:60],
            cost_credits=cost_credits,
            exhausted=len(all_rows) < n,
            cursor={"search_after": search_after, "offset": offset},
            dedup_key_column_hint="linkedin_url",
        )


register(FullEnrichPeopleAdapter())
