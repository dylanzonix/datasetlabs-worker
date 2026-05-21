---
name: find_linkedin
description: Finding LinkedIn profile URLs for individuals.
applies_to: [cell_agent]
---

## Finding someone's LinkedIn URL

LinkedIn is high-precision but identity collisions are common. Many
people share a name; committing the wrong profile is worse than null.

### See also: find_x_handles routing rules

Read `find_x_handles` for the column-level routing guidance — the
"one column, not two" rule likely applies here too. LinkedIn URL
discovery has the same shape as X-URL discovery: a known person, a
public profile to verify, and per-cell web_search burns multiple
billable calls before it can confidently null. If FullEnrich/Apollo
(the cheap precision sources for LinkedIn) return nothing on a batch,
prefer a bulk-flavored fill over web_search loops — bulk browser_use
sees patterns across people in the same task and avoids the per-cell
4-search hard floor.

LinkedIn-specific caveat: bulk BU sessions sometimes hit LinkedIn
auth walls / login prompts that web_search results don't. If the bulk
task fails with auth errors on a batch, fall back to per-cell which
can use FullEnrich/Apollo first.

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
