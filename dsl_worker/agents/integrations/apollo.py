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
    """Strip raw Apollo person to essential fields."""
    org = person.get("organization") or {}
    phones = person.get("phone_numbers") or []
    return {
        "id": person.get("id"),
        "name": person.get("name"),
        "title": person.get("title"),
        "email": person.get("email"),
        "email_status": person.get("email_status"),
        "phone": phones[0].get("sanitized_number") if phones else None,
        "linkedin": person.get("linkedin_url"),
        "location": person.get("city"),
        "seniority": person.get("seniority"),
        "company": org.get("name"),
        "company_domain": org.get("website_url"),
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
        name = args.get("name") or None

        if not domain and not name:
            return "Error: provide domain or name.", 0.0

        try:
            org = await client.enrich_company(domain=domain, name=name)
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
                "Enrich a company via Apollo — get industry, size, revenue, phone, "
                "location, LinkedIn, founding year. Provide domain or name."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "domain": {"type": "string", "description": "Company domain (e.g. 'stripe.com')"},
                    "name": {"type": "string", "description": "Company name"},
                },
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
