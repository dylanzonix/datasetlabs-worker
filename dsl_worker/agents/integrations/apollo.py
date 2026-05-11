"""
Apollo namespace tools for orchestrator and row generator.

Provides: search_companies, enrich_person, enrich_company, bulk_enrich_people,
org_job_postings

Modeled on thevgergroup/apollo-io-mcp — response simplification,
selectCompaniesArray for response inconsistency, name splitting,
filter best practices in descriptions.

Note: search_people is BLOCKED on Basic plan (requires Organization).
Use FullEnrich for people search instead.
"""

import json
import logging
import time
from pathlib import Path
from typing import Any, Callable, Awaitable, Dict, List, Optional, Tuple

from dsl_worker.infra.apollo_client import ApolloClient

logger = logging.getLogger(__name__)

ToolHandler = Callable[[Dict[str, Any]], Awaitable[Tuple[str, float]]]

NAMESPACE_DESCRIPTION = (
    "Apollo.io: search companies by industry/location/size/tech stack, "
    "enrich people and companies with email/phone/LinkedIn/revenue. "
    "Free for search, credits for enrichment. Good for B2B company data."
)


def _simplify_person(person: dict) -> dict:
    """Strip raw Apollo person to essential fields.

    Apollo's FREE search_people endpoint returns REDACTED person objects:
    `name` is None, `last_name` is masked (e.g. "Vo***a"), most org
    fields are stripped to `has_*` booleans, no LinkedIn URL. We fall
    back to assembling a display name from `first_name` + obfuscated
    last so the agent can still surface "Ami V." to the user. To get
    the unredacted record (full name, email, LinkedIn), the agent must
    call enrich_person with the Apollo `id`."""
    org = person.get("organization") or {}
    phones = person.get("phone_numbers") or []
    # Prefer the full name when present; assemble from parts otherwise.
    name = person.get("name")
    if not name:
        first = person.get("first_name") or ""
        last = (
            person.get("last_name")
            or person.get("last_name_obfuscated")
            or ""
        )
        joined = f"{first} {last}".strip()
        name = joined or None
    return {
        "id": person.get("id"),
        "name": name,
        "first_name": person.get("first_name"),
        "last_name_obfuscated": person.get("last_name_obfuscated"),
        "title": person.get("title"),
        "email": person.get("email"),
        "email_status": person.get("email_status"),
        "phone": phones[0].get("sanitized_number") if phones else None,
        "linkedin": person.get("linkedin_url"),
        "location": person.get("city"),
        "seniority": person.get("seniority"),
        "company": org.get("name"),
        "company_domain": org.get("website_url") or org.get("primary_domain"),
        "company_industry": org.get("industry"),
        "company_size": org.get("estimated_num_employees"),
    }


def _simplify_company(company: dict) -> dict:
    """Strip raw Apollo company to essential fields."""
    return {
        "id": company.get("id"),
        "name": company.get("name"),
        "website": company.get("website_url"),
        "industry": company.get("industry"),
        "employee_count": company.get("estimated_num_employees"),
        "location": (
            company.get("raw_address")
            or f"{company.get('city', '')}, {company.get('state', '')}, {company.get('country', '')}".strip(", ")
        ),
        "linkedin": company.get("linkedin_url"),
        "founded_year": company.get("founded_year"),
        "phone": (company.get("primary_phone") or {}).get("sanitized_number")
                 if isinstance(company.get("primary_phone"), dict)
                 else company.get("primary_phone"),
        "revenue": company.get("organization_revenue_printed"),
        "description": (company.get("short_description") or "")[:200],
    }


def _select_companies(result: dict) -> list:
    """Apollo returns companies in 'organizations' or 'accounts' depending on query."""
    orgs = result.get("organizations", [])
    if orgs:
        return orgs
    return result.get("accounts", [])


def register_apollo_namespace(
    registry: Any,
    client: ApolloClient,
    workspace_dir: Path,
    file_counter: Optional[List[int]] = None,
    cost_per_credit: float = 0.024,
    on_file_written: Optional[Callable] = None,
) -> None:
    """Register the apollo namespace on a ToolRegistry."""
    if file_counter is None:
        file_counter = [0]

    def _next_filename(prefix: str) -> Path:
        idx = file_counter[0]
        file_counter[0] += 1
        return workspace_dir / "candidates" / f"{prefix}_{idx}.jsonl"

    # ── search_people ────────────────────────────────────────────────

    async def search_people(args: Dict) -> Tuple[str, float]:
        """Free people search across Apollo's 210M+ contact DB. Returns
        basic info (name, title, company, LinkedIn, location) — no emails
        or phones. Pair with enrich_person to get contact data."""
        page = args.get("page", 1)
        per_page = args.get("per_page", 25)
        try:
            people, total = await client.search_people(
                person_titles=args.get("titles") or None,
                person_seniorities=args.get("seniorities") or None,
                person_locations=args.get("locations") or None,
                person_names=args.get("names") or None,
                organization_keywords=args.get("company_keywords") or None,
                organization_name=args.get("company_name") or None,
                organization_locations=args.get("company_locations") or None,
                organization_num_employees_ranges=args.get("company_employee_ranges") or None,
                organization_domains=args.get("company_domains") or None,
                organization_revenue_ranges=args.get("company_revenue_ranges") or None,
                technology_uids=args.get("technologies") or None,
                q_keywords=args.get("keywords") or None,
                include_similar_titles=args.get("include_similar_titles"),
                per_page=per_page,
                page=page,
            )
        except Exception as e:
            return f"Apollo search_people error: {e}", 0.0

        if not people:
            return "No people found. Try broader filters.", 0.0

        # Defensive client-side filter on domain — Apollo's
        # q_organization_domains_list is forgiving, so if the agent
        # passed a specific domain, drop rows whose current org domain
        # doesn't substring-match it. Same rationale as the FE post-filter.
        wanted_domains = [
            str(d).strip().lower()
            for d in (args.get("company_domains") or [])
            if isinstance(d, str) and d.strip()
        ]
        filter_dropped = 0
        if wanted_domains:
            kept = []
            for p in people:
                org = p.get("organization") or {}
                dom = (org.get("website_url") or org.get("primary_domain") or "").lower()
                if any(w in dom for w in wanted_domains):
                    kept.append(p)
                else:
                    filter_dropped += 1
            if filter_dropped > 0:
                logger.info(
                    "apollo.search_people: post-filter dropped %d/%d rows by domain %s",
                    filter_dropped, len(people), wanted_domains,
                )
            people = kept

        if not people:
            return (
                f"Apollo returned rows but none matched domain {wanted_domains}. "
                f"Try search_companies first to find the right org, or pass "
                f"company_name instead of company_domains."
            ), 0.0

        simplified = [_simplify_person(p) for p in people]
        output_path = _next_filename("apollo_people")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            for p in simplified:
                f.write(json.dumps(p, ensure_ascii=False, default=str) + "\n")

        if on_file_written:
            on_file_written(output_path)

        workspace_path = f"/workspace/candidates/{output_path.name}"
        total_pages = (total + per_page - 1) // per_page if total else "?"
        lines = [
            f"Apollo people search: {total:,} total results (page {page}/{total_pages}).",
            f"File: {workspace_path} ({len(simplified)} saved)",
        ]
        if filter_dropped > 0:
            lines.append(
                f"(post-filter dropped {filter_dropped} rows whose org domain didn't match)"
            )
        for s in simplified[:3]:
            lines.append(
                f"  {s.get('name', '?')} — {s.get('title', '?')} @ "
                f"{s.get('company', '?')} | {s.get('linkedin', 'no linkedin')}"
            )
        if total > per_page:
            lines.append(f"\nMore: search_people(..., page={page + 1})")
        lines.append(
            "\nNote: emails NOT in these results. Pair with enrich_person "
            "(1 credit/person) or fullenrich_enrich_contacts for contact data."
        )
        return "\n".join(lines), 0.0

    # ── search_companies ─────────────────────────────────────────────

    async def search_companies(args: Dict) -> Tuple[str, float]:
        page = args.get("page", 1)
        per_page = args.get("per_page", 25)

        try:
            orgs, total = await client.search_companies(
                organization_keywords=args.get("keywords") or None,
                organization_name=args.get("name") or None,
                organization_locations=args.get("locations") or None,
                organization_not_locations=args.get("not_locations") or None,
                organization_num_employees_ranges=args.get("employee_ranges") or None,
                organization_revenue_ranges=args.get("revenue_ranges") or None,
                organization_domains=args.get("domains") or None,
                industry_tag_ids=args.get("industries") or None,
                technology_uids=args.get("technologies") or None,
                per_page=per_page,
                page=page,
            )
        except Exception as e:
            return f"Apollo search error: {e}", 0.0

        if not orgs:
            return f"No companies found. Try broader filters.", 0.0

        simplified = [_simplify_company(o) for o in orgs]

        output_path = _next_filename("apollo_companies")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            for c in simplified:
                f.write(json.dumps(c, ensure_ascii=False, default=str) + "\n")

        if on_file_written:
            on_file_written(output_path)

        workspace_path = f"/workspace/candidates/{output_path.name}"
        total_pages = (total + per_page - 1) // per_page if total else "?"

        lines = [
            f"Apollo company search: {total:,} total results (page {page}/{total_pages}).",
            f"File: {workspace_path} ({len(simplified)} saved)",
        ]
        for c in simplified[:3]:
            lines.append(f"  {c['name']} — {c.get('industry', '?')} | {c.get('employee_count', '?')} employees | {c.get('revenue', '?')}")

        if total > per_page:
            lines.append(f"\nMore: search_companies(..., page={page + 1})")

        return "\n".join(lines), 0.0

    # ── enrich_person ────────────────────────────────────────────────

    async def enrich_person(args: Dict) -> Tuple[str, float]:
        try:
            person = await client.enrich_person(
                apollo_id=args.get("apollo_id") or None,
                first_name=args.get("first_name") or None,
                last_name=args.get("last_name") or None,
                name=args.get("name") or None,
                email=args.get("email") or None,
                organization_name=args.get("company") or None,
                domain=args.get("domain") or None,
                linkedin_url=args.get("linkedin_url") or None,
            )
        except Exception as e:
            return f"Apollo enrichment error: {e}", 0.0

        if not person:
            return "No match found in Apollo.", 0.0

        s = _simplify_person(person)

        parts = [
            f"**{s.get('name', '?')}**",
            f"Title: {s.get('title', 'N/A')}",
            f"Email: {s.get('email', 'N/A')} (status: {s.get('email_status', '?')})",
            f"Phone: {s.get('phone', 'N/A')}",
            f"LinkedIn: {s.get('linkedin', 'N/A')}",
            f"Location: {s.get('location', 'N/A')}",
            f"Seniority: {s.get('seniority', 'N/A')}",
            f"Company: {s.get('company', 'N/A')} ({s.get('company_domain', '')})",
            f"Industry: {s.get('company_industry', 'N/A')}",
            f"Company Size: {s.get('company_size', 'N/A')}",
        ]
        # Person enrichment costs 1 Apollo credit
        return "\n".join(parts), cost_per_credit

    # ── enrich_company ───────────────────────────────────────────────

    async def enrich_company(args: Dict) -> Tuple[str, float]:
        domain = args.get("domain") or None

        if not domain:
            return "Error: provide domain (call search_companies if you only have a name).", 0.0

        try:
            org = await client.enrich_company(domain=domain)
        except Exception as e:
            return f"Apollo enrichment error: {e}", 0.0

        if not org:
            return "No match found in Apollo.", 0.0

        s = _simplify_company(org)

        parts = [
            f"**{s.get('name', '?')}**",
            f"Website: {s.get('website', 'N/A')}",
            f"Industry: {s.get('industry', 'N/A')}",
            f"Employees: {s.get('employee_count', 'N/A')}",
            f"Revenue: {s.get('revenue', 'N/A')}",
            f"Location: {s.get('location', 'N/A')}",
            f"LinkedIn: {s.get('linkedin', 'N/A')}",
            f"Founded: {s.get('founded_year', 'N/A')}",
            f"Phone: {s.get('phone', 'N/A')}",
        ]
        if s.get("description"):
            parts.append(f"Description: {s['description']}")
        return "\n".join(parts), 0.0

    # ── bulk_enrich_people ───────────────────────────────────────────

    async def bulk_enrich_people(args: Dict) -> Tuple[str, float]:
        people = args.get("people", [])
        if not people:
            return "Error: people array is required.", 0.0

        # Apollo's bulk match endpoint
        import httpx
        headers = {
            "Content-Type": "application/json",
            "X-Api-Key": client._api_key,
        }

        # Convert our simplified format to Apollo's expected format
        details = []
        for p in people:
            entry: Dict[str, Any] = {}
            if p.get("name"):
                parts = p["name"].strip().split(None, 1)
                entry["first_name"] = parts[0]
                entry["last_name"] = parts[1] if len(parts) > 1 else ""
            if p.get("first_name"):
                entry["first_name"] = p["first_name"]
            if p.get("last_name"):
                entry["last_name"] = p["last_name"]
            if p.get("email"):
                entry["email"] = p["email"]
            if p.get("company"):
                entry["organization_name"] = p["company"]
            if p.get("domain"):
                entry["organization_domain"] = p["domain"]
            if p.get("linkedin_url"):
                entry["linkedin_url"] = p["linkedin_url"]
            details.append(entry)

        try:
            async with httpx.AsyncClient(timeout=30.0) as http:
                resp = await http.post(
                    "https://api.apollo.io/api/v1/people/bulk_match",
                    headers=headers,
                    json={"details": details},
                )
            if resp.status_code != 200:
                return f"Apollo bulk enrich error: HTTP {resp.status_code}", 0.0

            matches = resp.json().get("matches", [])
        except Exception as e:
            return f"Apollo bulk enrich error: {e}", 0.0

        results = []
        for m in matches:
            if m:
                results.append(_simplify_person(m))
            else:
                results.append({"error": "no match"})

        output_path = _next_filename("apollo_enriched")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")

        if on_file_written:
            on_file_written(output_path)

        matched = sum(1 for r in results if "name" in r)
        workspace_path = f"/workspace/candidates/{output_path.name}"

        # 1 credit per matched person
        bulk_cost = matched * cost_per_credit
        return (
            f"Bulk enriched {matched}/{len(people)} people.\n"
            f"File: {workspace_path}\n"
            f"Cost: ${bulk_cost:.4f} ({matched} credits)"
        ), bulk_cost

    # ── org_job_postings ─────────────────────────────────────────────

    async def org_job_postings(args: Dict) -> Tuple[str, float]:
        org_id = args.get("organization_id", "")
        if not org_id:
            return "Error: organization_id is required.", 0.0

        import httpx
        headers = {"Content-Type": "application/json", "X-Api-Key": client._api_key}

        try:
            async with httpx.AsyncClient(timeout=15.0) as http:
                resp = await http.get(
                    f"https://api.apollo.io/api/v1/organizations/{org_id}/job_postings",
                    headers=headers,
                )
            if resp.status_code != 200:
                return f"Apollo error: HTTP {resp.status_code}", 0.0

            postings = resp.json().get("job_postings", [])
        except Exception as e:
            return f"Apollo error: {e}", 0.0

        if not postings:
            return f"No job postings found for organization {org_id}.", 0.0

        lines = [f"Found {len(postings)} job postings:"]
        for jp in postings[:10]:
            lines.append(f"  {jp.get('title', '?')} — {jp.get('location', '?')}")

        return "\n".join(lines), 0.0

    # ── Register namespace ───────────────────────────────────────────

    tools = [
        {
            "name": "search_people",
            "description": (
                "Search Apollo's 210M+ contact database for people. FREE — no "
                "credits consumed. Returns name, title, company, LinkedIn, "
                "location, seniority. Does NOT return emails or phones — pair "
                "with enrich_person (1 credit each) or fullenrich_enrich_contacts.\n\n"
                "Best practices:\n"
                "- For 'all engineers at company X' → set `company_domains` "
                "(or `company_name`) + leave titles broad / use `include_similar_titles=true`\n"
                "- For 'CTOs in NYC' → titles + locations + seniorities\n"
                "- Use `q_keywords` for free-text matching when title shape is unknown"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "titles": {"type": "array", "items": {"type": "string"}, "description": "Person job titles (e.g. ['Software Engineer', 'Member of Technical Staff'])"},
                    "seniorities": {"type": "array", "items": {"type": "string"}, "description": "Seniority levels (e.g. ['c_suite', 'vp', 'director', 'manager', 'senior', 'entry'])"},
                    "locations": {"type": "array", "items": {"type": "string"}, "description": "Person locations (cities, states, countries)"},
                    "names": {"type": "array", "items": {"type": "string"}, "description": "Specific person names"},
                    "company_keywords": {"type": "array", "items": {"type": "string"}, "description": "Industry/keyword filter on the org"},
                    "company_name": {"type": "string", "description": "Specific company name"},
                    "company_locations": {"type": "array", "items": {"type": "string"}, "description": "Org locations"},
                    "company_employee_ranges": {"type": "array", "items": {"type": "string"}, "description": "Org headcount ranges (comma format: ['51,200', '201,500'])"},
                    "company_domains": {"type": "array", "items": {"type": "string"}, "description": "Org domains (e.g. ['anthropic.com'])"},
                    "company_revenue_ranges": {"type": "array", "items": {"type": "string"}, "description": "Org revenue ranges"},
                    "technologies": {"type": "array", "items": {"type": "string"}, "description": "Tech stack the org uses"},
                    "keywords": {"type": "string", "description": "Free-text keyword search"},
                    "include_similar_titles": {"type": "boolean", "description": "Expand title matching to semantic neighbors (default false)"},
                    "per_page": {"type": "integer", "description": "Results per page (default 25, max 100)"},
                    "page": {"type": "integer", "description": "Page number (default 1)"},
                },
            },
        },
        {
            "name": "search_companies",
            "description": (
                "Search Apollo's company database with filters. Returns company profiles "
                "with industry, size, revenue, location, website, LinkedIn. Results saved to file.\n\n"
                "Best practices:\n"
                "- Start with location + keywords — most effective combo\n"
                "- Employee ranges use comma format: '11,20' not '11-20'\n"
                "- Employee filters are restrictive — try without if 0 results\n"
                "- Combine 2-3 filters, add more to narrow"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "keywords": {"type": "array", "items": {"type": "string"}, "description": "Industry keywords (e.g. ['saas', 'fintech', 'edtech'])"},
                    "name": {"type": "string", "description": "Specific company name"},
                    "locations": {"type": "array", "items": {"type": "string"}, "description": "Company locations (most effective filter)"},
                    "not_locations": {"type": "array", "items": {"type": "string"}, "description": "Exclude these locations"},
                    "industries": {"type": "array", "items": {"type": "string"}, "description": "Industry IDs"},
                    "employee_ranges": {"type": "array", "items": {"type": "string"}, "description": "Employee count ranges in comma format (e.g. ['11,20', '21,50', '51,200'])"},
                    "revenue_ranges": {"type": "array", "items": {"type": "string"}, "description": "Revenue ranges"},
                    "domains": {"type": "array", "items": {"type": "string"}, "description": "Company domains"},
                    "technologies": {"type": "array", "items": {"type": "string"}, "description": "Tech stack (e.g. ['salesforce', 'react', 'aws'])"},
                    "per_page": {"type": "integer", "description": "Results per page (default 25, max 100)"},
                    "page": {"type": "integer", "description": "Page number (default 1)"},
                },
            },
        },
        {
            "name": "enrich_person",
            "description": (
                "Enrich a person via Apollo — get email, phone, title, company, LinkedIn. "
                "Provide at least one of: name+company, email, linkedin_url, or apollo_id."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Full name (auto-split into first/last)"},
                    "first_name": {"type": "string"},
                    "last_name": {"type": "string"},
                    "email": {"type": "string"},
                    "company": {"type": "string", "description": "Company name"},
                    "domain": {"type": "string", "description": "Company domain"},
                    "linkedin_url": {"type": "string"},
                    "apollo_id": {"type": "string"},
                },
            },
        },
        {
            "name": "enrich_company",
            "description": (
                "Enrich a company via Apollo by DOMAIN — get industry, size, "
                "revenue, phone, location, LinkedIn, founding year. If you "
                "only have a company name, call search_companies first to "
                "get the domain."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "domain": {"type": "string", "description": "Company domain (e.g. 'stripe.com')"},
                },
                "required": ["domain"],
            },
        },
        {
            "name": "bulk_enrich_people",
            "description": (
                "Enrich multiple people at once. Provide array of people with name+company "
                "or email or linkedin_url. Results saved to file."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "people": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string"},
                                "first_name": {"type": "string"},
                                "last_name": {"type": "string"},
                                "email": {"type": "string"},
                                "company": {"type": "string"},
                                "domain": {"type": "string"},
                                "linkedin_url": {"type": "string"},
                            },
                        },
                        "description": "People to enrich",
                    },
                },
                "required": ["people"],
            },
        },
        {
            "name": "org_job_postings",
            "description": "Get active job postings for a company by Apollo organization_id.",
            "parameters": {
                "type": "object",
                "properties": {
                    "organization_id": {"type": "string", "description": "Apollo organization ID"},
                },
                "required": ["organization_id"],
            },
        },
    ]

    handlers = {
        "search_people": search_people,
        "search_companies": search_companies,
        "enrich_person": enrich_person,
        "enrich_company": enrich_company,
        "bulk_enrich_people": bulk_enrich_people,
        "org_job_postings": org_job_postings,
    }

    registry.add_namespace(
        name="apollo",
        description=NAMESPACE_DESCRIPTION,
        tools=tools,
        handlers=handlers,
    )
