---
name: find_emails
description: Finding a work email for a known person via FullEnrich.
applies_to: [cell_agent]
---

## Finding someone's work email

Inputs: a person (first_name + last_name) and a company domain. Optionally
a LinkedIn URL (significantly improves match rate). Pass all of these to
`fullenrich_enrich_email` in one call.

If you have a company but no person yet, load `find_person_at_company`
first — name + LinkedIn typically come from that skill, then this skill
runs the email step.

### Trust FE. One call → commit.

```
fullenrich_enrich_email(
  first_name=...,
  last_name=...,
  domain=...,
  linkedin_url=...,   # pass when the row has it — much better match rate
)
```

The response gives `{email, verification_status}`. Outcomes:

- **email + status=DELIVERABLE** → commit the email. Done.
- **email + status=CATCH_ALL** → commit the email; this is the company's
  catch-all — works for outreach but isn't strictly verified.
- **email + status=INVALID** → commit null. FE found a candidate
  address but it bounces.
- **email=null** → commit null. FE's waterfall ran through all its
  providers and didn't find a personal email. Don't web_search hoping
  the company's "About" page exposes one (it almost never does for
  staff emails), and don't pattern-guess `firstname@domain` as
  "verified" — committing a guess gives the user a bouncy list.

### Cost shape

- 1 FE call → $0.055 on a hit, $0 on a miss.
- A clean per-row email lookup is **one call**.
- FE bulk enrich takes 30-60s server-side; that's normal. Don't
  pile on web_searches while waiting.

### When the row gives you a company but no person

Load `find_person_at_company` first. It surfaces a name + LinkedIn URL
via `fullenrich_search_people`; you then feed those into
`fullenrich_enrich_email`. Two calls total: search + enrich, ~$0.05-0.10.

### What null actually means

After FE_enrich_email with name + domain (+ linkedin_url if available)
returns no email or INVALID — null is the right answer. Pattern guesses
(`firstname@domain`, `firstinitial+lastname@domain`) without FE
confirmation are NOT verified emails. Commit null with a `reason` like
"FE returned no email for {Name} @ {domain}" so the next pass knows the
source already failed.
