---
name: find_emails
description: Given a person (name + domain), find their work email. Read BEFORE calling fullenrich_enrich_email — covers the no-pattern-guess rule and what to do when FullEnrich returns nothing.
applies_to: [cell_agent]
---

## Finding someone's work email

Inputs: a person (first_name + last_name) and a company domain. Optionally
a LinkedIn URL (significantly improves match rate). Pass all of these to
`fullenrich_enrich_email` in one call.

If you have a company but no person yet, load `find_person_at_company`
first — name + LinkedIn typically come from that skill, then this skill
runs the email step.

### Trust FullEnrich. One call → commit.

```
fullenrich_enrich_email(
  first_name=...,
  last_name=...,
  domain=...,
  linkedin_url=...,   # pass when the row has it — much better match rate
)
```

The response gives `{email, verification_status, commit}`.

**If `commit: true` — commit the email. Don't web_search to verify.**

The wrapper sets `commit: true` for the two FullEnrich statuses that
are safe to send (DELIVERABLE = 1% bounce, HIGH_PROBABILITY = 9%
bounce). The most common non-DELIVERABLE return is HIGH_PROBABILITY
(FullEnrich found the email via its waterfall but couldn't 100%
verify the mailbox actively accepts mail). Treat both the same:
commit the email, done. Don't web_search to "confirm" — the wrapper
already told you it's safe. Past failure: model got HIGH_PROBABILITY,
didn't trust it, burned 4 web_searches + a browser_use, committed
null. The email was right the whole time.

**If `commit: false`** the candidate is either missing, INVALID
(verified bounce, ~100%), or CATCH_ALL (server accepts everything,
~26% bounce — too risky for cold outbound at scale). Treat as no
email found. ONE targeted web_search (`"<full name>" "<company>"
email`) is fine if you want one more angle. Commit only what
web_search surfaces on a real cite-able page. Otherwise null.

Full status reference (per FullEnrich docs):
- DELIVERABLE — 1% bounce, safe (commit:true)
- HIGH_PROBABILITY — 9% bounce, usually safe (commit:true)
- CATCH_ALL — 26% bounce, too risky for cold outbound (commit:false)
- INVALID — verified bounce (commit:false)

### NEVER commit a pattern guess as verified

After FullEnrich returns email=null, do NOT commit `firstname@domain`
or `firstinitial+lastname@domain` or any other guessed pattern as the
answer. Those addresses LOOK like emails but the user has no idea
they're guesses; they send a campaign and half bounce. A guess
without verification is worse than null.

Examples of BAD commits (do not do this):
- FullEnrich returned null for Abby Murray @ storyarb.com → commit
  `first@example.com`. **NO.** Commit null.
- FullEnrich returned null for Stan @ frontbrick.io → commit
  `third@example.com`. **NO.** Commit null.

The only time `firstname@domain` is a valid commit is when you
ALREADY VERIFIED it — either FullEnrich returned that exact address
with DELIVERABLE/CATCH_ALL status, or web_search surfaced it on a
real page (a public bio, conference speaker page, GitHub profile)
where the person published it themselves.

### Cost shape

- 1 FullEnrich call → $0.055 on a hit, $0 on a miss.
- A clean per-row email lookup is **one call**.
- FullEnrich bulk enrich takes 30-60s server-side; that's normal.
  Don't pile on web_searches while waiting.

### When the row gives you a company but no person

Load `find_person_at_company` first. It surfaces a name + LinkedIn URL
via `fullenrich_search_people`; you then feed those into
`fullenrich_enrich_email`. Two calls total: search + enrich, ~$0.05-0.10.

### What null actually means

After FullEnrich_enrich_email with name + domain (+ linkedin_url if
available) returns no email or INVALID, and an optional single
web_search didn't surface a verified personal email — null is the
right answer. Commit null with a `reason` like "FullEnrich returned
no email for {Name} @ {domain}; one web_search didn't surface one
either" so the next pass knows the source already failed.
