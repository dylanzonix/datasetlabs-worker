---
name: find_people
description: Finding a person (decision maker, manager, founder, owner, contact) at a company.
applies_to: [cell_agent]
---

## Finding a person at a company

The row gives you a company / org / business name (and usually a
domain or website). The column wants the *person* — a decision-maker,
manager, founder, owner, billing contact, etc.

**Use `fullenrich_search_people` first. Don't default to web_search.**
FE is a structured LinkedIn-derived people index — you get name, title,
seniority, employer, LinkedIn URL directly. Web_search means parsing
snippets, which is slower, more expensive across multiple calls, and
prone to fabricated LinkedIn slugs from misleading SERP fragments.

### See also: find_emails

Once you've found the person, the same row likely has an Email
column too. Read `find_emails` for the email-side flow — finding
the person and finding their email are usually one connected pass:
search → pick → enrich.

### Cost discipline — `limit` is the cost knob

FE charges ~1 credit (~$0.055) per RETURNED person, NOT per call. So
the `limit` on `fullenrich_search_people` IS your bill.

- **Column wants ONE specific person** (founder / CEO / owner): `limit=2-3`.
  You want a small candidate pool to pick from, not a directory dump.
- **Column wants any person at a given level** (e.g. "any SDR-level role"):
  `limit=5-10`. Slightly bigger pool so you have title-matching options.
- **Never `limit=25` or higher** for a per-row lookup. That's a $1+ row
  for a single founder name — completely out of proportion.

The cell handler hard-caps `limit` at 10. Even at 10, that's $0.55 — so
default to 3 and only bump when you have a reason.

### Critical rule: keep FE filters MINIMAL

The dominant failure mode across past projects (51dbf993 Apartment
Operator Leads, 1219684b Florida GYN, 64f9b81c NY NJ Med Spa) was
over-filtering `fullenrich_search_people` so it returned 0:

  - `company_names=[<list>] + titles=["Property Manager",...] + locations=[...] + industries=[...]` → **0 results**
  - `company_domains=[<list>] + titles=[...] + seniority=[...] + headcount range` → **0 results**

**Lesson:** every additional filter narrows the index. For a
"find a person at THIS company" lookup, start with `company_names`
or `company_domains` ALONE — no titles, no seniority, no industries.
Look at what comes back, then narrow downstream (in your head, not in
the API).

If you must add a filter, add ONE — most often `locations` to narrow
a national parent to the row's city / region. Don't combine three on
the first call.

### Strip corporate suffixes from company names

FE matches `company_names` against the LinkedIn-listed employer name,
which often omits the suffix the user has. Project efc17fa4 (Apartment
Operator Leads):

  - `company_names=["Village Green Companies"]` → **0 results**
  - `company_names=["Village Green"]` → **18 results**, 7 of them
    Property Managers in Columbus, all at the actual employer domain
    `villagegreenmgt.com`.

Before the FE call, strip these from the row's company string:
"Companies", "Company", "Group", "Holdings", "Properties", "Inc",
"Inc.", "LLC", "Corp", "Corporation", "Ltd", "Limited". If the bare
form returns 0, then try the original. Almost never the other way
around.

### Tool order — by company TYPE

The right path depends on what kind of company is in the row.

#### B2B / professional / tech / SaaS / large org companies

**FullEnrich is the whole game here.** FE indexes LinkedIn-derived data
and is strong on white-collar B2B targets.

1. `fullenrich_search_people(company_names=["<bare org name>"], limit=3)`
   — single call, no other filters. Strip corporate suffixes per the
   rule above.
2. **Pick from the result list** based on the column's intent (most
   senior person, person whose title says "Manager", etc.). Each result
   has `title` and `seniority`.
3. If you got the person but `linkedin_url` is null, commit the name +
   title; leave LinkedIn null. Don't burn web_search trying to guess a
   slug — see the "no fabricated slugs" warning below.
4. If the column also wants Email, chain `fullenrich_enrich_email` on
   the chosen person (`first_name`, `last_name`, `domain`, and pass
   `linkedin_url` if FE returned one — it gates a deeper waterfall).

If `company_names` returned 0, try `company_domains=["<the domain>"]`
once. If both return 0, FE has no LinkedIn entry for this company —
fall to the local-business path below.

#### Local businesses / single-location practices / consumer-facing shops

**FE will usually return 0 for these.** Med spas, single-location
gyms, dentists, OB-GYN practices, salons, restaurants, single-property
apartment complexes, small law firms — typically not LinkedIn-indexed
at the company level.

Fallback path:

1. **One targeted `web_search`** for `"<business name>" owner` or
   `"<business name>" medical director` / `founder`. The name often
   appears in directory listings, LinkedIn snippets, or the "About"
   page. ONE search; if it doesn't surface a name, move on.
2. If web_search surfaced a name, commit the name. Leave LinkedIn /
   email null unless web_search ALSO clearly surfaced them on a
   verifiable page (a real LinkedIn URL from the search result, not
   a fabricated slug from the snippet).
3. If web_search surfaced nothing identifiable, null.

#### Multi-property operators / brand vs. parent company

Apartment complexes, hotel chains, franchise locations, dealerships
— the company name in the row (e.g. "The View on Grant", "Lane
Lofts") is a *property* or *brand*, not the LinkedIn-indexed
company. The actual employer of the property/community manager is
the parent management company (e.g. "Coastal Ridge", "Flaherty &
Collins Properties", "Wilcox Communities").

The pattern (project 51dbf993):

1. Find the parent management company first. Often surfaces from
   `google_maps_place_details` (the website's About page links to
   the operator) or a single web_search for `"<property name>"
   management company`.
2. Then `fullenrich_search_people(company_names=["<parent>"], limit=5)`
   — slightly bigger pool because there will be multiple managers
   across properties. Filter by location if the parent is national.
3. Pick the person whose *current title* matches the column's intent
   AND who appears to cover this property (sometimes evident from
   `title` mentioning the specific community).

A single property manager often manages multiple properties; matching
"this exact property" is a bonus, not a requirement. If the user
asked for "decision-maker at this complex" and we surface a regional
PM at the parent, that IS the decision-maker.

**Prefer property-specific candidates over generic regional contacts.**
When FE search at the parent returns multiple managers, read each
one's title — many property managers have a title that names the
specific community they run. A title-match for THIS row's property
name, address, or unit count is strictly better than a regional/
portfolio contact who doesn't mention any specific community.

Per-cell rule:
1. Read every result's `title` — pick a description-match for THIS
   row before falling to anyone else.
2. If no match, prefer the lowest-seniority person whose title is
   `Property Manager` / `Community Manager` / `General Manager` over
   a `Regional` / `Portfolio` / `VP` title.
3. Only commit a regional/portfolio contact (Director / VP / EVP)
   when steps 1 and 2 yield nothing. When you do, note it in
   `reason`: "no property-specific manager found in FE; using
   regional portfolio contact at <parent>" — that signals to the
   user this row is weaker than ones with site-level managers.

### Don't fabricate LinkedIn URLs from web snippets

When FE returns the person but no LinkedIn URL, do NOT construct a
slug from a web_search snippet. Project efc17fa4 committed a wrong
`linkedin.com/in/angela-piek` slug fabricated from a SERP fragment;
the correct profile was `linkedin.com/in/angela-piek-arm-5bb24213`.
A LinkedIn URL must come from a result FE returned OR from clicking
through to a real linkedin.com page that explicitly belongs to the
named person. Otherwise leave it null.

### Don't fan out org-level info into per-person columns

Already covered in the cell-agent base prompt, but it bites here in
particular: if the FE result returns the COMPANY's main phone or
`info@` mailbox (not a personal one), do NOT paste it into the
per-person Email/Phone columns. Per-person columns require per-person
evidence. Generic mailboxes belong in a *Company Email* column.

### Cost-aware bail criteria

- 1 FE search_people call with `limit=3` → ~$0.15 if 3 hits, free if 0.
- 1 FE enrich_email on the chosen person → ~$0.055 on a hit.
- 1 web_search for local-business fallback → ~$0.025.
- Browser_use is rarely worth it for finding a person — bail and
  commit null with a reason that names what you tried.

The right total spend for "find decision-maker + email" on a B2B
target is **$0.05–$0.20** (FE search + FE enrich), not $0.30+ of
web_searches.

### When null IS the right answer

- FE search by `company_names` AND by `company_domains` returned 0,
  AND
- One web_search for `"<company>" owner|founder|manager` surfaced no
  identifiable person.

Then null. Commit the name column null and skip the email/LinkedIn
columns (don't fabricate). In `reason`, name the company filters and
the web_search you tried — that's how the next iteration knows
which approach to drop.
