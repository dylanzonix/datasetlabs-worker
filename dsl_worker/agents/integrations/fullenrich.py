"""
FullEnrich namespace tools for orchestrator and row generator.

Provides: search_people, search_companies, enrich_contacts
All with defer_loading for on-demand discovery via tool_search.
"""

import json
import logging
import time
from pathlib import Path
from typing import Any, Callable, Awaitable, Dict, List, Optional, Tuple

from dsl_worker.infra.fullenrich_client import FullEnrichClient

logger = logging.getLogger(__name__)

ToolHandler = Callable[[Dict[str, Any]], Awaitable[Tuple[str, float]]]

NAMESPACE_DESCRIPTION = (
    "Search for people and companies with advanced filters (title, location, "
    "industry, seniority, skills, company size, etc). Enrich contacts with "
    "verified work emails and phone numbers via waterfall across 20+ data "
    "providers. Only charged for verified results."
)


def build_filter(values: List[str], exact_match: bool = False, exclude: bool = False) -> List[Dict]:
    """Convert simple string list to FullEnrich filter format."""
    return [{"value": v, "exact_match": exact_match, "exclude": exclude} for v in values]


def build_range_filter(min_val: Optional[int] = None, max_val: Optional[int] = None, exclude: bool = False) -> List[Dict]:
    """Convert min/max to FullEnrich range filter format."""
    entry: Dict[str, Any] = {"exclude": exclude}
    if min_val is not None:
        entry["min"] = min_val
    if max_val is not None:
        entry["max"] = max_val
    return [entry]


def register_fullenrich_namespace(
    registry: Any,
    client: FullEnrichClient,
    workspace_dir: Path,
    file_counter: Optional[List[int]] = None,
    cost_per_credit: float = 0.055,
    on_file_written: Optional[Callable] = None,
) -> None:
    """Register the fullenrich namespace on a ToolRegistry.

    Args:
        registry: ToolRegistry instance
        client: FullEnrichClient
        workspace_dir: For writing result files
        file_counter: Mutable [int] for unique filenames (shared across calls)
        on_file_written: Callback(Path) called after each candidate file is written
    """
    if file_counter is None:
        file_counter = [0]

    def _next_filename(prefix: str) -> Path:
        idx = file_counter[0]
        file_counter[0] += 1
        path = workspace_dir / "candidates" / f"{prefix}_{idx}.jsonl"
        return path

    # ── search_people handler ────────────────────────────────────────

    async def search_people(args: Dict) -> Tuple[str, float]:
        # Convert simplified params to FullEnrich filter format
        filters: Dict[str, Any] = {}

        # String filters (value/exact_match/exclude format)
        for param, api_key in [
            ("titles", "current_position_titles"),
            ("locations", "person_locations"),
            ("industries", "current_company_industries"),
            ("seniority", "current_position_seniority_level"),
            ("skills", "person_skills"),
            ("universities", "person_universities"),
            ("company_names", "current_company_names"),
            ("company_domains", "current_company_domains"),
            ("company_specialties", "current_company_specialties"),
            ("company_types", "current_company_types"),
            ("company_hq", "current_company_headquarters"),
            ("names", "person_names"),
            ("linkedin_urls", "person_linkedin_urls"),
            ("past_companies", "past_company_names"),
            ("past_titles", "past_position_titles"),
        ]:
            val = args.get(param)
            if val:
                if isinstance(val, list):
                    filters[api_key] = build_filter(val)
                elif isinstance(val, str):
                    filters[api_key] = build_filter([val])

        # Range filters
        for param_min, param_max, api_key in [
            ("company_headcount_min", "company_headcount_max", "current_company_headcounts"),
            ("years_in_position_min", "years_in_position_max", "current_position_years_in"),
            ("years_at_company_min", "years_at_company_max", "current_company_years_at"),
            ("company_founded_min", "company_founded_max", "current_company_founded_years"),
            ("days_since_job_change_min", "days_since_job_change_max", "current_company_days_since_last_job_change"),
        ]:
            min_val = args.get(param_min)
            max_val = args.get(param_max)
            if min_val is not None or max_val is not None:
                filters[api_key] = build_range_filter(min_val, max_val)

        limit = args.get("limit", 25)
        max_results = args.get("max_results", limit)
        output_path = _next_filename("fullenrich_people")

        result = await client.search_people_to_file(
            filters=filters,
            output_path=output_path,
            max_results=max_results,
        )

        if "error" in result:
            return f"Error: {result['error']}", 0.0

        if result.get("item_count", 0) == 0:
            return f"No people found matching filters. Total in database: {result.get('total', 0)}. Try broader filters.", 0.0

        if on_file_written:
            on_file_written(output_path)

        workspace_path = f"/workspace/candidates/{output_path.name}"

        credits_used = result.get("credits_used", 0)
        cost_usd = credits_used * cost_per_credit

        return (
            f"Found {result['total']:,} people matching filters.\n"
            f"File: {workspace_path} ({result['item_count']} saved)\n"
            f"Fields: {', '.join(result.get('fields', []))}\n"
            f"Cost: ${cost_usd:.4f} ({credits_used:.1f} credits)\n\n"
            f"Next: submit_candidates with this file, or inspect with code_exec."
        ), cost_usd

    # ── search_companies handler ─────────────────────────────────────

    async def search_companies(args: Dict) -> Tuple[str, float]:
        filters: Dict[str, Any] = {}

        for param, api_key in [
            ("industries", "industries"),
            ("locations", "headquarters_locations"),
            ("specialties", "specialties"),
            ("keywords", "keywords"),
            ("names", "names"),
            ("domains", "domains"),
            ("types", "types"),
        ]:
            val = args.get(param)
            if val:
                if isinstance(val, list):
                    filters[api_key] = build_filter(val)
                elif isinstance(val, str):
                    filters[api_key] = build_filter([val])

        for param_min, param_max, api_key in [
            ("headcount_min", "headcount_max", "headcounts"),
            ("founded_min", "founded_max", "founded_years"),
        ]:
            min_val = args.get(param_min)
            max_val = args.get(param_max)
            if min_val is not None or max_val is not None:
                filters[api_key] = build_range_filter(min_val, max_val)

        limit = args.get("limit", 25)
        max_results = args.get("max_results", limit)
        output_path = _next_filename("fullenrich_companies")

        result = await client.search_companies_to_file(
            filters=filters,
            output_path=output_path,
            max_results=max_results,
        )

        if "error" in result:
            return f"Error: {result['error']}", 0.0

        if result.get("item_count", 0) == 0:
            return f"No companies found matching filters. Total in database: {result.get('total', 0)}. Try broader filters.", 0.0

        if on_file_written:
            on_file_written(output_path)

        workspace_path = f"/workspace/candidates/{output_path.name}"
        credits_used = result.get("credits_used", 0) if "credits_used" in result else 0
        cost_usd = credits_used * cost_per_credit

        return (
            f"Found {result['total']:,} companies matching filters.\n"
            f"File: {workspace_path} ({result['item_count']} saved)\n"
            f"Fields: {', '.join(result.get('fields', []))}\n"
            f"Cost: ${cost_usd:.4f} ({credits_used:.1f} credits)\n\n"
            f"Next: submit_candidates with this file, or inspect with code_exec."
        ), cost_usd

    # ── enrich_contacts handler ──────────────────────────────────────

    async def enrich_contacts(args: Dict) -> Tuple[str, float]:
        contacts = args.get("contacts", [])
        if not contacts:
            return "Error: contacts array is required.", 0.0

        fields = args.get("fields", ["emails", "phones"])
        enrich_fields = []
        for f in fields:
            if "email" in f.lower():
                enrich_fields.append("contact.emails")
            if "phone" in f.lower():
                enrich_fields.append("contact.phones")
            if "personal" in f.lower():
                enrich_fields.append("contact.personal_emails")
        if not enrich_fields:
            enrich_fields = ["contact.emails", "contact.phones"]

        # For small batches (≤5), return in context
        # For larger batches, write to file
        if len(contacts) <= 5:
            result = await client.enrich_contacts(
                contacts=contacts,
                name=f"enrich_{int(time.time())}",
                enrich_fields=enrich_fields,
            )

            if "error" in result:
                return f"Enrichment error: {result['error']}", 0.0

            data = result.get("data", [])
            credits = result.get("cost", {}).get("credits", 0)

            cost_usd = credits * cost_per_credit
            parts = [f"Enriched {len(data)} contacts. Cost: ${cost_usd:.4f} ({credits} credits).\n"]
            for entry in data:
                inp = entry.get("input", {})
                name = inp.get("full_name") or f"{inp.get('first_name', '')} {inp.get('last_name', '')}".strip()
                contact_info = entry.get("contact_info", {})

                email_obj = contact_info.get("most_probable_work_email", {})
                email = email_obj.get("email", "Not found") if email_obj else "Not found"
                email_status = email_obj.get("status", "") if email_obj else ""

                phone_obj = contact_info.get("most_probable_phone", {})
                phone = phone_obj.get("number", "Not found") if phone_obj else "Not found"
                phone_region = phone_obj.get("region", "") if phone_obj else ""

                parts.append(
                    f"**{name or 'Unknown'}**\n"
                    f"  Email: {email}"
                    + (f" [{email_status}]" if email_status else "")
                    + f"\n  Phone: {phone}"
                    + (f" [{phone_region}]" if phone_region else "")
                )

                # Include extra emails/phones if found
                extra_emails = contact_info.get("work_emails", [])
                if len(extra_emails) > 1:
                    others = [e.get("email") for e in extra_emails[1:] if e.get("email")]
                    if others:
                        parts.append(f"  Other emails: {', '.join(others)}")

                extra_phones = contact_info.get("phones", [])
                if len(extra_phones) > 1:
                    others = [p.get("number") for p in extra_phones[1:] if p.get("number")]
                    if others:
                        parts.append(f"  Other phones: {', '.join(others)}")

            return "\n".join(parts), cost_usd

        else:
            # Bulk — write to file
            output_path = _next_filename("fullenrich_enriched")
            result = await client.enrich_contacts_to_file(
                contacts=contacts,
                output_path=output_path,
                name=f"enrich_{int(time.time())}",
                enrich_fields=enrich_fields,
            )

            if "error" in result:
                return f"Enrichment error: {result['error']}", 0.0

            if on_file_written:
                on_file_written(output_path)

            workspace_path = f"/workspace/candidates/{output_path.name}"
            credits_used = result.get("credits_used", 0)
            cost_usd = credits_used * cost_per_credit
            return (
                f"Enriched {result.get('item_count', 0)} contacts.\n"
                f"File: {workspace_path}\n"
                f"Cost: ${cost_usd:.4f} ({credits_used} credits)"
            ), cost_usd

    # ── Register namespace ───────────────────────────────────────────

    tools = [
        {
            "name": "search_people",
            "description": (
                "Search for people by job title, location, company, industry, seniority, "
                "skills, education, and more. Returns professional profiles with employment "
                "history, education, skills, and social profiles. Results saved to file.\n\n"
                "Key filters: titles, locations, industries, seniority (Entry/Mid/Director/"
                "VP/C-Level), company_names, company_domains, skills, universities, "
                "company_headcount_min/max, years_in_position_min/max, past_companies, "
                "past_titles.\n\n"
                "Start with 2-3 filters and broaden if too few results."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "titles": {"type": "array", "items": {"type": "string"}, "description": "Job titles (e.g. ['CEO', 'CTO', 'VP Marketing'])"},
                    "locations": {"type": "array", "items": {"type": "string"}, "description": "Person locations — city, state, or country (e.g. ['San Francisco', 'California', 'United States'])"},
                    "industries": {"type": "array", "items": {"type": "string"}, "description": "Company industry (e.g. ['Software Development', 'Financial Services'])"},
                    "seniority": {"type": "array", "items": {"type": "string"}, "description": "Seniority level: Entry, Mid, Director, VP, C-Level, Owner"},
                    "skills": {"type": "array", "items": {"type": "string"}, "description": "Professional skills (e.g. ['Python', 'Sales', 'Product Management'])"},
                    "company_names": {"type": "array", "items": {"type": "string"}, "description": "Current employer names"},
                    "company_domains": {"type": "array", "items": {"type": "string"}, "description": "Current employer domains (e.g. ['google.com'])"},
                    "company_specialties": {"type": "array", "items": {"type": "string"}, "description": "Company focus areas (e.g. ['SaaS', 'AI'])"},
                    "company_hq": {"type": "array", "items": {"type": "string"}, "description": "Company HQ location"},
                    "company_types": {"type": "array", "items": {"type": "string"}, "description": "Public Company, Privately Held, Nonprofit, etc."},
                    "company_headcount_min": {"type": "integer", "description": "Minimum company employees"},
                    "company_headcount_max": {"type": "integer", "description": "Maximum company employees"},
                    "universities": {"type": "array", "items": {"type": "string"}, "description": "Educational institutions"},
                    "past_companies": {"type": "array", "items": {"type": "string"}, "description": "Previous employer names"},
                    "past_titles": {"type": "array", "items": {"type": "string"}, "description": "Previous job titles"},
                    "years_in_position_min": {"type": "integer", "description": "Min years in current role"},
                    "years_in_position_max": {"type": "integer", "description": "Max years in current role"},
                    "days_since_job_change_min": {"type": "integer", "description": "Min days since last job change (e.g. 0 for recent changers)"},
                    "days_since_job_change_max": {"type": "integer", "description": "Max days since last job change (e.g. 90 for recent changers)"},
                    "max_results": {"type": "integer", "description": "Max people to return (default 25). Results saved to file."},
                },
            },
        },
        {
            "name": "search_companies",
            "description": (
                "Search for companies by industry, location, size, specialties, and more. "
                "Returns company profiles with description, headcount, founding year, "
                "locations, industry, and social profiles. Results saved to file.\n\n"
                "Key filters: industries, locations (HQ), specialties, keywords, names, "
                "domains, types, headcount_min/max, founded_min/max."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "industries": {"type": "array", "items": {"type": "string"}, "description": "Industries (e.g. ['Software Development', 'Healthcare'])"},
                    "locations": {"type": "array", "items": {"type": "string"}, "description": "HQ locations — city, state, or country"},
                    "specialties": {"type": "array", "items": {"type": "string"}, "description": "Company specialties (e.g. ['SaaS', 'machine learning'])"},
                    "keywords": {"type": "array", "items": {"type": "string"}, "description": "Search company descriptions"},
                    "names": {"type": "array", "items": {"type": "string"}, "description": "Company names"},
                    "domains": {"type": "array", "items": {"type": "string"}, "description": "Company domains"},
                    "types": {"type": "array", "items": {"type": "string"}, "description": "Public Company, Privately Held, Nonprofit, etc."},
                    "headcount_min": {"type": "integer", "description": "Minimum employees"},
                    "headcount_max": {"type": "integer", "description": "Maximum employees"},
                    "founded_min": {"type": "integer", "description": "Earliest founding year"},
                    "founded_max": {"type": "integer", "description": "Latest founding year"},
                    "max_results": {"type": "integer", "description": "Max companies to return (default 25). Results saved to file."},
                },
            },
        },
        {
            "name": "enrich_contacts",
            "description": (
                "Get verified work emails and phone numbers for contacts via waterfall "
                "enrichment across 20+ data providers. Only charged for verified results.\n\n"
                "Each contact needs either: first_name + last_name + company (name or domain), "
                "OR linkedin_url. Adding linkedin_url improves accuracy significantly.\n\n"
                "Returns email with verification status (DELIVERABLE, HIGH_PROBABILITY, "
                "CATCH_ALL, INVALID) and phone in E.164 format with region.\n\n"
                "Cost: ~1 credit per valid email, ~10 credits per valid phone."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "contacts": {
                        "type": "array",
                        "description": "Contacts to enrich (max 100 per call)",
                        "items": {
                            "type": "object",
                            "properties": {
                                "first_name": {"type": "string"},
                                "last_name": {"type": "string"},
                                "company_name": {"type": "string", "description": "Company name (if no domain)"},
                                "domain": {"type": "string", "description": "Company domain (if no company_name)"},
                                "linkedin_url": {"type": "string", "description": "LinkedIn profile URL (improves accuracy)"},
                            },
                        },
                    },
                    "fields": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "What to enrich: 'emails', 'phones', 'personal_emails'. Default: ['emails', 'phones']",
                    },
                },
                "required": ["contacts"],
            },
        },
    ]

    handlers = {
        "search_people": search_people,
        "search_companies": search_companies,
        "enrich_contacts": enrich_contacts,
    }

    registry.add_namespace(
        name="fullenrich",
        description=NAMESPACE_DESCRIPTION,
        tools=tools,
        handlers=handlers,
    )
