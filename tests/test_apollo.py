"""
Quick Apollo API test — verify search + enrichment work before integrating.

Usage:
    APOLLO_API_KEY=xxx .venv/bin/python3 tests/test_apollo.py
"""

import asyncio
import json
import os
import sys

import httpx


APOLLO_BASE = "https://api.apollo.io/api/v1"


async def test_search(api_key: str):
    """Test People Search — FREE, no credits consumed."""
    print("=" * 60)
    print("TEST 1: People Search (free, no credits)")
    print("=" * 60)

    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(
            f"{APOLLO_BASE}/mixed_people/search",
            headers={"x-api-key": api_key, "Content-Type": "application/json"},
            json={
                "person_titles": ["Vendor Coordinator", "Maintenance Supervisor", "Property Manager"],
                "person_locations": ["Seattle, Washington, United States"],
                "organization_industry_tag_ids": [],  # empty = all industries
                "q_organization_name": "property management",
                "per_page": 10,
                "page": 1,
            },
        )

        print(f"Status: {resp.status_code}")
        if resp.status_code != 200:
            print(f"Error: {resp.text[:500]}")
            return None

        data = resp.json()
        people = data.get("people", [])
        total = data.get("pagination", {}).get("total_entries", 0)

        print(f"Total matches: {total}")
        print(f"Returned: {len(people)}")
        print()

        for i, person in enumerate(people[:5]):
            print(f"  [{i+1}] {person.get('name', '?')}")
            print(f"      Title: {person.get('title', '?')}")
            print(f"      Company: {person.get('organization', {}).get('name', '?') if person.get('organization') else person.get('organization_name', '?')}")
            print(f"      Location: {person.get('city', '?')}, {person.get('state', '?')}")
            print(f"      LinkedIn: {person.get('linkedin_url', 'N/A')}")
            print(f"      Email: {person.get('email', 'N/A (search does not return emails)')}")
            print(f"      Apollo ID: {person.get('id', '?')}")
            print()

        return people


async def test_enrich(api_key: str, person: dict):
    """Test People Enrichment — consumes 1 credit."""
    print("=" * 60)
    print("TEST 2: People Enrichment (costs 1 credit)")
    print("=" * 60)

    name = person.get("name", "")
    org = person.get("organization", {})
    org_name = org.get("name", "") if org else person.get("organization_name", "")
    apollo_id = person.get("id", "")

    print(f"Enriching: {name} at {org_name} (ID: {apollo_id})")

    async with httpx.AsyncClient(timeout=30) as client:
        # Try by Apollo ID first (most reliable)
        payload = {
            "id": apollo_id,
            "reveal_personal_emails": False,
            "reveal_phone_number": True,
        }

        resp = await client.post(
            f"{APOLLO_BASE}/people/match",
            headers={"x-api-key": api_key, "Content-Type": "application/json"},
            json=payload,
        )

        print(f"Status: {resp.status_code}")
        if resp.status_code != 200:
            print(f"Error: {resp.text[:500]}")
            return

        data = resp.json()
        enriched = data.get("person", {})

        print(f"\n  Name: {enriched.get('name', '?')}")
        print(f"  Title: {enriched.get('title', '?')}")
        print(f"  Email: {enriched.get('email', 'N/A')}")
        print(f"  Email status: {enriched.get('email_status', '?')}")
        print(f"  LinkedIn: {enriched.get('linkedin_url', 'N/A')}")

        phones = enriched.get("phone_numbers", [])
        if phones:
            for p in phones:
                print(f"  Phone: {p.get('sanitized_number', '?')} ({p.get('type', '?')})")
        else:
            print(f"  Phone: N/A")

        org_data = enriched.get("organization", {})
        if org_data:
            print(f"  Company: {org_data.get('name', '?')}")
            print(f"  Website: {org_data.get('website_url', 'N/A')}")
            print(f"  Industry: {org_data.get('industry', 'N/A')}")
            print(f"  Employees: {org_data.get('estimated_num_employees', 'N/A')}")

        print()
        return enriched


async def main():
    api_key = os.environ.get("APOLLO_API_KEY", "")
    if not api_key:
        print("Set APOLLO_API_KEY environment variable")
        sys.exit(1)

    # Test 1: Search (free)
    people = await test_search(api_key)
    if not people:
        print("Search returned no results, can't test enrichment")
        return

    # Test 2: Enrich first result (costs 1 credit)
    print("\nThis will consume 1 Apollo credit. Press Enter to continue or Ctrl+C to skip.")
    try:
        input()
    except KeyboardInterrupt:
        print("\nSkipped enrichment test.")
        return

    await test_enrich(api_key, people[0])


if __name__ == "__main__":
    asyncio.run(main())
