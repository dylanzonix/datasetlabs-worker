---
name: find_linkedin
description: Finding LinkedIn profile URLs for individuals
applies_to: [cell_agent]
triggers:
  - linkedin
  - linkedin url
  - linkedin handle
  - linkedin profile
  - linkedin_url
  - linkedin.com/in
---

## Finding someone's LinkedIn URL

LinkedIn is high-precision but identity collisions are common. Many
people share a name; committing the wrong profile is worse than null.

**Tool order:**
1. **FullEnrich** (`fullenrich_search_people` or `fullenrich_enrich_contacts`)
   when you have name + company. Cheapest verified path.
2. **Apollo** (`apollo_enrich_person`) as fallback for B2B people, especially
   if FE returned nothing.
3. **web_search** with `"Full Name" company linkedin.com/in`. The URL pattern
   `linkedin.com/in/<slug>` shows up in result snippets even when the
   profile itself isn't crawled.

**Verify identity before committing:**
- Profile's CURRENT company matches the company you have for this person.
  Past-company-only matches are weaker.
- Title/role roughly aligns with what you know.
- Location aligns if you have it.
- If two profiles tie on name + company, pick the one whose role best
  matches the source. If still ambiguous, return null.

**Don't commit:**
- A LinkedIn URL of a person whose company doesn't match.
- The COMPANY's LinkedIn page (`linkedin.com/company/...`) when the
  column wants a person.
- A profile with no overlap with the row's known fields.
