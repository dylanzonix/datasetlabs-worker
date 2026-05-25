---
name: find_person_at_company
description: Given a company, find a specific person (founder, owner, decision-maker, contact, manager) who works there.
applies_to: [cell_agent]
---

## Finding a person at a company

The row gives you a company name (and usually a domain). The column wants
the *person* — founder, CEO, VP Sales, head of recruiting, owner, etc.

**Use `fullenrich_search_people` first.** LinkedIn-derived structured DB;
single call returns name + title + seniority + LinkedIn URL directly.
Web_search is the fallback when FE has nothing.

FE is authoritative for who works where. When the response has people
at the matching company with the title the column wants, pick one and
commit. The LinkedIn URL and title in the response ARE the verification.

### The cost shape

FE charges 0.25 credit (~$0.013) per RETURNED person. So `limit` IS the
cost knob. Default limit=3 is enough for almost every per-row lookup;
~$0.04/call. Skip the urge to set limit=10 to "be safe" — that's $0.13
per call and rarely catches anything limit=3 missed.

### Pick filters by what the column wants

The picker is **about the column's intent**, not about FE's options:

**Specific role** (Founder, CEO, VP Sales, CMO, CTO, Recruiter, Head of X, etc.)

```
fullenrich_search_people(
  company_names=["<company>"],
  titles=["Founder","Co-Founder","CEO","Owner"]    # or whatever role you want
  limit=3,
)
```

Pass title VARIANTS, not just one. For "VP Sales" pass
`["VP Sales","Vice President Sales","VP of Sales","Chief Revenue Officer","Head of Sales"]`.
FE matches the literal string; the same person at the same job has 3-4 ways
they may have written it.

For founder specifically use the broad set
`["Founder","Co-Founder","CEO","Owner","Managing Partner","President"]`
since the actual founder's current title can be any of these.

**Any senior person** (column says "decision-maker" / "key contact" without naming a role)

```
fullenrich_search_people(
  company_names=["<company>"],
  seniority=["C-Level","VP","Director"],
  limit=5,
)
```

Seniority filter beats `titles` when you don't have a specific title in mind —
seniority is FE's structured classification, doesn't depend on the string match.

**Specific named person** (column wants something about a person already named on the row)

Skip search entirely — go straight to `fullenrich_enrich_email`
with first_name + last_name + domain.

### Read the results — the wrapper post-filters but you still pick

The wrapper auto-drops results whose CURRENT employer name doesn't contain
your `company_names` string. This catches the noisy cross-company matches
(e.g. searching "Purple Sales" returns Sam Balzan at "Purple Sales" PLUS a
"Kimberly Soldau at Purple Zebra Sales" — the wrapper drops Kimberly).
You still pick from the filtered list:

- Multiple founders → pick the one whose title matches what the column wants
  (Co-Founder + CEO over plain Co-Founder, etc.)
- Title-match for a description-specific role (e.g. property-specific PM
  over a regional VP) wins.
- If results look reasonable, commit the first qualifying one. Don't
  web_search to "verify" — FE returned the linkedin URL and title; that
  IS the verification.

### When the first call comes back empty (or wrong)

In order of cheapness:

1. **Try `company_domains`** if you have the domain
   ```
   fullenrich_search_people(company_domains=["<domain>"], titles=[...], limit=3)
   ```
   Different index path. Often hits when name search missed (especially
   when the company name in your row differs from how they self-list on
   LinkedIn — e.g. "Acme Companies" in your data vs "Acme" on LinkedIn).

2. **Try stripping the suffix** off the name (LLC, Inc, Corp, Companies,
   Group, Holdings, Co.) — but only AFTER the full name and the domain
   have both missed. Stripping pre-emptively can introduce false matches
   (e.g. "NU Advisory" matches both the real company AND a different
   "NU Advisory Office").

3. **Drop the title filter** — call again with `company_names` alone +
   `limit=5`. The company is in FE but no one is titled with your
   target role. Look at what comes back; sometimes the founder is
   listed under a non-obvious title ("Principal", "Partner", "Director
   of Operations"). If you see someone clearly senior in the results,
   commit them.

4. **Web_search once**. `"<company>" founder` or `"<company>" "<role>"`.
   If web surfaces an identifiable name, commit it. Don't fabricate a
   LinkedIn slug from a search snippet — only commit a LinkedIn URL if
   you got it from FE or from clicking through to a real linkedin.com
   page that explicitly names the person.

5. **Null is the right answer** when steps 1-4 produce nothing. In
   `reason`, name what you tried so the next iteration knows which
   approach failed.

### Small companies and local businesses

FE indexes white-collar B2B targets well. Local businesses
(single-location med spas, dentists, small property managers,
restaurants) frequently return 0 on `fullenrich_search_people` because
they're not in LinkedIn's company graph. Skip steps 1-3 and go
directly to web_search with a name-targeted query, then commit
whatever name surfaces.

### Multi-property operators / brand vs parent

For apartment complexes, hotel chains, franchise locations: the row's
"company" is usually a property/brand. The actual employer is the
parent management company. Find the parent first
(`google_maps_place_details` or one web_search for `"<property>"
management company`), then `fullenrich_search_people` against the
parent with location filter to narrow to the right region.

### Don't fan out org-level info into per-person columns

If FE returns a company main phone or `info@` mailbox, don't paste
it into per-person Email/Phone columns. Per-person columns need
per-person evidence. Company contacts belong in a *Company Email*
column.

### Typical cost shape

- 1 FE search at limit=3 → $0.04. Most cells finish here.
- 2 FE searches (titles → domain fallback) → $0.08.
- + 1-2 web_searches if FE missed entirely → $0.05-0.075.
- Realistic per-cell total: $0.04-$0.15.
- Set `per_row_credit_cap: 3` for these enrichments to leave margin.
