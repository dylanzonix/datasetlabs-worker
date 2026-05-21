---
name: find_people
description: Finding a person (decision maker, manager, founder, owner, contact) at a company.
applies_to: [cell_agent]
---

## Finding a person at a company

The row gives you a company / org / business name (and usually a
domain or website). The column wants the *person* — a decision-maker,
manager, founder, owner, billing contact, etc. **You almost always
have a structured-source path here. Don't default to web_search.**

### See also: find_emails

Once you've found the person, the same row likely has an Email
column too. Read `find_emails` for the email-side flow — finding
the person and finding their email are usually one connected pass:
search → pick → enrich.

### Critical rule: keep FE filters MINIMAL

The dominant failure mode across past projects (51dbf993 Apartment
Operator Leads, 1219684b Florida GYN, 64f9b81c NY NJ Med Spa, the
first two FE calls in cbdb7add Org Contact Discovery) was
over-filtering `fullenrich_search_people` so it returned 0:

  - 51dbf993: `company_names=[<list of 10>] + titles=["Property Manager",...] + locations=[...] + industries=[...]` → **0 results**
  - 64f9b81c: `company_domains=[<list>] + titles=[...] + seniority=[...] + headcount range` → **0 results**
  - cbdb7add: `company_domains=[<list>] + titles=[8 of them] + seniority=[...]` → 90 random results
    that didn't actually match the orgs; same `company_names` ALONE
    returned **71 actual people from those orgs**.

**Lesson:** every additional filter narrows the index. For a
"find a person at THIS company" lookup, start with `company_names`
or `company_domains` ALONE — no titles, no seniority, no industries.
Look at what comes back, then narrow downstream (in your head, not in
the API).

If you must add a filter, add ONE — most often `person_locations` to
narrow a national parent's footprint to the row's city / region. Don't
combine three on the first call.

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

The right tool depends on what kind of company is in the row.

#### B2B / professional / tech / SaaS / large org companies

**Use FullEnrich.** FE indexes LinkedIn-derived data and is strong on
white-collar B2B targets.

1. `fullenrich_search_people(company_names=["<bare org name>"], limit=10)`
   — single call, no other filters. Strip corporate suffixes per the
   rule above. Returns up to 10 employees with their names, titles,
   employer domains, location, and *sometimes* a LinkedIn URL.
2. **Pick from the result list** based on the column's intent (most
   senior person, person whose title says "Manager", etc.). The
   result has `employment.current.title` and `employment.current.seniority`.
3. **If FE returned the person but `linkedin: None`, run
   `apollo_enrich_person(first_name, last_name, domain, company)`
   on them** before committing. FE and Apollo have different LinkedIn
   coverage; Apollo frequently has a profile when FE doesn't.
   Project efc17fa4: FE found Angela Piek at Oakwood Management with
   `linkedin: None`; Apollo enrich_person on the same name returned
   `linkedin.com/in/angela-piek-arm-5bb24213` plus a verified email.
   The cell agent there skipped this step and committed a fabricated
   `linkedin.com/in/angela-piek` slug from a web_search snippet — wrong
   profile.
4. If the column also wants Email, chain
   `fullenrich_enrich_contacts(contacts=[{first_name, last_name,
   domain, linkedin_url}], fields=["emails"])` for that person — pass
   `fields=["emails"]` explicitly to avoid the 10-credit phone hit.
   See `find_emails`.

If `company_names` returned 0, try `company_domains=["<the domain>"]`
once. If both return 0, FE has no LinkedIn entry for this company —
fall to the local-business path below.

#### Local businesses / single-location practices / consumer-facing shops

**FE will usually return 0 for these.** Med spas, single-location
gyms, dentists, OB-GYN practices, salons, restaurants, single-property
apartment complexes, small law firms — these are typically not
LinkedIn-indexed at the company level.

**Use Apollo with a guessed name + domain.** Apollo's coverage for
local-business owners is meaningfully broader than FE's. The catch:
Apollo `enrich_person` needs a *name* to look up — there's no
people-search on our plan. So the play is:

1. **Surface a candidate name first** — one targeted `web_search`
   for `"<business name>" owner` or `"<business name>" medical
   director` or `"<business name>" founder`. The name often appears
   in directory listings, LinkedIn snippets, or the business's
   "About" page. ONE search; if it doesn't surface a name, move on.
2. `apollo_enrich_person(first_name="<found>", last_name="<found>",
   domain="<row domain>", company="<row company>")`. ~$0.024 per
   match. Apollo returns a `linkedin` URL and sometimes an `email`
   with `email_status: "verified"` for owners of small businesses
   that FE has nothing on (project 64f9b81c examples:
   `owner@example.com` for Sandra Frayna at Aspen Prime,
   LinkedIn URL for Edward Fruitman at Trifecta Med Spa).
3. If Apollo returns `matched: true` but the person record is mostly
   nulls (no title, no LinkedIn, no email), that means Apollo *has*
   the person but doesn't have the contact data — committing the
   name alone (without LinkedIn/email) is fine if the column is just
   the name; for downstream Email enrichment, retry through
   `fullenrich_enrich_contacts` with that name + domain.

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
2. Then `fullenrich_search_people(company_names=["<parent>"])` —
   plural results. Filter by location if the parent is national
   (Columbus-area community managers vs. their nationwide footprint).
3. Pick the person whose *current title* matches the column's intent
   AND who appears to cover this property (sometimes evident from
   `employment.current.description`).

A single property manager often manages multiple properties; matching
"this exact property" is a bonus, not a requirement. If the user
asked for "decision-maker at this complex" and we surface a regional
PM at the parent, that IS the decision-maker.

**But prefer property-specific candidates over generic regional
contacts.** When FE search at the parent returns multiple managers,
read each one's `employment.current.description` — many property
managers have a description that names the specific community they
run ("Property Manager at <community name>", "Oversees daily
operations for a 300-unit apartment community", etc.). A
description-match for THIS row's property name, address, or unit
count is a strictly better commit than a regional/portfolio contact
who doesn't mention any specific community.

Project fafed105 (Apartment Operator Leads) committed "Trevor Brown
/ contact@example.com" — a senior Coastal Ridge portfolio
contact — to FOUR consecutive rows because the cell agents grabbed
the most senior-looking person for each small property without
checking descriptions for site-level matches. Same FE search would
have surfaced site-specific managers (Aimee Richards, Tabitha
Meyers, Carrie O'Callaghan); those got picked for the larger
properties but the smaller ones got dup'd to Trevor.

Per-cell rule:
1. Read every result's `employment.current.description` — pick a
   description-match for THIS row before falling to anyone else.
2. If no description-match, prefer the lowest-seniority person whose
   title is `Property Manager` / `Community Manager` / `General
   Manager` over a `Regional` / `Portfolio` / `VP` title.
3. Only commit a regional/portfolio contact (Director / VP / EVP)
   when steps 1 and 2 yield nothing. When you do, note it in
   `reason`: "no property-specific manager found in FE; using
   regional portfolio contact at <parent>" — that signals to the
   user this row is weaker than ones with site-level managers.

### First-name variants between FE and Apollo

The same person is often indexed under different first-name forms in
FE vs. Apollo. Project efc17fa4: FE has "Angela Fuller" at Wallick
with no LinkedIn; Apollo has the same person as "Angie Fuller" with
a LinkedIn URL. Same lesson as `find_x_handles`: Andrew↔Andy,
Michael↔Mike, William↔Will/Bill, Elizabeth↔Liz/Beth,
Robert↔Rob/Bob, Daniel↔Dan, Jonathan↔Jon, etc.

When FE found the person but Apollo enrich_person returns no match
on the same first name, retry Apollo with the variant. Don't burn
multiple credits on the same name in two forms — usually one form
will hit; if the first one's record is empty (matched but all
nulls), try the variant before falling to web_search.

### Don't fan out org-level info into per-person columns

Already covered in the cell-agent base prompt, but it bites here in
particular: if the FE / Apollo result returns the COMPANY's main
phone or `info@` mailbox (not a personal one), do NOT paste it into
the per-person Email/Phone columns. Per-person columns require
per-person evidence. Generic mailboxes belong in a *Company Email*
column.

### Cost-aware bail criteria

- 1 FE search_people call (free / fractional credit) → if 0, proceed.
- 1 Apollo enrich_person call ($0.024) → if no useful data, proceed.
- 1 web_search to surface a name (built-in; ~$0.025) → if no name,
  consider null.
- Beyond that, escalating to browser_use ($0.10–$0.50) is rarely
  worth it for finding a person — bail and commit null with a
  reason that names what you tried, so the next pass can adjust the
  approach.

The Apartment Operator Leads anti-pattern was 4–5 web_searches per
cell averaging $0.10–$0.16, finding a name + LinkedIn but never
running FE enrich on it for the email. The right total spend for
"find decision-maker + email" on a B2B target is **$0.05–$0.10**
(FE search + FE enrich), not $0.15+ of web_searches.

### When null IS the right answer

- FE search by `company_names` AND by `company_domains` returned 0,
  AND
- Apollo enrich_person on a guessed name returned no useful contact
  data, AND
- One web_search for `"<company>" owner|founder|manager` surfaced no
  identifiable person.

Then null. Commit the name column null and skip the email/LinkedIn
columns (don't fabricate). In `reason`, name the company filters and
the apollo guess you tried — that's how the next iteration knows
which approach to drop.
