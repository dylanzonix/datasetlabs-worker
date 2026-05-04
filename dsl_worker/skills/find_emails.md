---
name: find_emails
description: Finding work emails for individuals
applies_to: [cell_agent]
triggers:
  - email
  - work email
  - business email
  - contact email
  - verified email
  - email_address
  - email address
---

## Finding someone's work email

### See also: find_people

If the row tells you a *company / org / business* but no *person*, the
column you're filling is implicitly a person + email lookup. Read
`find_people.md` first — it covers how to surface a name (which then
unlocks this email-finding flow). Don't web_search the website hoping
an email shows up; emails on practice / med-spa / property pages are
almost always `info@` (which is a different column from a personal
email anyway).

### Cheapest-to-most-expensive ladder

1. **`fullenrich_enrich_contacts`** — the right answer in ~80% of B2B
   cases. Pass `linkedin_url` when you have one (significantly improves
   the match) AND `first_name` + `last_name` + `domain`. One call,
   one credit per verified email (~$0.055).
   **Always pass `fields=["emails"]` explicitly** when you only need
   email. The default is now emails-only, but being explicit guards
   against a stray phone hit costing 10 credits when FE happens to
   have a phone for that person.
2. **`apollo_enrich_person(name=..., domain=..., company=...)`** — the
   right answer for **local-business owners** (med spas, dentists,
   gyms, salons, small clinics, single-location practices) where FE
   has no LinkedIn entry. Apollo's local coverage is meaningfully
   broader than FE's for these segments. Often returns a verified
   email when FE returns nothing — see project 64f9b81c (NY NJ Med Spa
   Leads): FE search returned 0 across the whole batch, Apollo
   enrich_person matched 10/10 with several verified emails
   (`owner@example.com` etc.).
3. **`apollo_enrich_person(linkedin_url=...)`** — when you have a
   LinkedIn URL but FE struck out anyway. Apollo and FE pull from
   different waterfalls; it's worth the second look.
4. **Pattern guess** — only with strong evidence of the org's email
   format (e.g. you've seen `firstname@company.com` confirmed for
   another employee at the same org in this same fill). When you
   guess, run `fullenrich_enrich_contacts` on the guessed
   first/last/domain to verify the pattern matches a real mailbox
   before committing. Don't commit a pattern guess as "verified".

### Critical rules

- **A confident null beats a wrong email.** The user can't audit a
  bounced email until it bounces — that's worse than empty. But null
  is also wasteful when the email was findable; check this skill's
  ladder before nulling.
- If FE returns `no_data` / `low_probability` / a missing
  `most_probable_work_email`, **try Apollo as the next step** (not
  null). Apollo and FE waterfall different providers; one finding
  nothing doesn't mean the other will.
- If FE returns `CATCH_ALL` status, that's not a real verification —
  treat it as a guess. If Apollo also can't verify, decide based on
  how much weaker `CATCH_ALL` than `DELIVERABLE` is for the user's
  use case (cold-email send: weaker; manual outreach: still useful).
- **Don't commit Email=null after web_search found a name + LinkedIn.**
  This was the dominant failure mode in 51dbf993 (Apartment Operator
  Leads): cells web_searched 4–5x, found the property manager's name
  + LinkedIn URL, then committed Email=null without ever calling
  `fullenrich_enrich_contacts(linkedin_url=...)`. That single missing
  call lost the entire fill's email yield. If you have a name and
  either a LinkedIn or domain, run FE enrich BEFORE nulling.
- Match the column format ("lowercase email or null", etc.) exactly.
  Strip trailing whitespace and normalize case before committing.

### Source signals worth chasing

- **A LinkedIn URL in another column** is the highest-yield input —
  pass it as `linkedin_url` to `fullenrich_enrich_contacts` (or
  `apollo_enrich_person`). FE's match rate jumps significantly with
  a LinkedIn URL vs. just name+domain.
- **A company website** → derive the email domain from it (about
  page, footer). If the row already has `Domain` or `Website`,
  parse that for the FE/Apollo `domain` arg before doing any new
  search. Never fall through to `@gmail.com` style guesses.
- **An `info@` / `contact@` / `hello@` mailbox you saw on the website**
  belongs in a *Company Email* column, NOT a *Contact Email* column.
  Don't fan a generic mailbox into per-person email slots — see the
  cell-agent base prompt.

### When null IS the right answer

After all of: (a) FE enrich on name+domain (with linkedin_url if
available), (b) Apollo enrich_person, (c) one targeted web_search
for a personal email pattern on the company site — if every step
returned no email or `INVALID`, null is correct. Cite which sources
you tried in the `reason` field so the trace is useful next pass.
