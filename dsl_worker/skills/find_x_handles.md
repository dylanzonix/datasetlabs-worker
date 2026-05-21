---
name: find_x_handles
description: Finding X (Twitter) handles or X URLs for individuals.
applies_to: [cell_agent]
---

## Finding someone's X (Twitter) handle

The same person often shows different names across platforms. "Andy"
on a directory listing may be "Andrew" on X. Maiden vs. married surnames,
nicknames, initials, or a handle that drops the last name entirely.

### One column, not two: X URL is what the user wants

**X Handle and X URL are the same data.** A handle (`@automin`) and a
URL (`https://x.com/automin`) are trivially convertible — the URL is
just `https://x.com/` + the handle without the `@`. The user cares
about the URL because it's clickable; the handle alone is a half-step
they'd have to manually paste into a browser.

**Rules:**
- When the user asks for "Twitter handles" / "X handles" / "X
  accounts" / "their Twitter" / etc., create ONE column named
  `X URL` with format `https://x.com/... or null`. Do NOT also add
  an `X Handle` column. Two columns is duplicate data the user has
  to maintain.
- If a row already has both `X Handle` and `X URL` columns from an
  older harvest, fill `X URL` only and leave `X Handle` alone (or
  ask the user if they want it dropped). Never spend credits filling
  both.
- When committing a value, always write the full URL form
  (`https://x.com/automin`), never the bare handle (`@automin` or
  `automin`). Even if the column happens to be named `X Handle`,
  store the URL — it's what's useful downstream.

### Hard floor: at least 4 distinct searches before any null

**You MUST run at least 4 distinct `web_search` queries with materially
different angles before calling `set_values(null)`.** One or two passes
that return nothing or return only unverifiable matches are NOT grounds
to null this cell — they are the signal to widen the search, not to bail.

Bailing after one or two searches is the failure mode this skill exists
to prevent. The Paul Sawaya regression (real account `@automin`, found
by the user with one search) and the Andrew Baran regression (real
account findable as `Andy Baran`, missed after 2 searches that didn't
try the variant) both happened because the cell agent quit too early.
Don't repeat that.

If after 4 distinct searches you still cannot confirm a handle, then
null is appropriate — but each of the 4 searches must be MEANINGFULLY
different from the others (different angle below), not the same query
re-typed.

### Required search angles (cycle through, don't repeat)

1. **Full name + employer + platform**: `"Andrew Smith" Acme CEO X`
   or `"Andrew Smith" twitter.com Acme`. The X bio surfaces in results
   even when the name on X is shortened.
2. **First-name variants** if the source name is a common one with
   variants: Andrew↔Andy↔Drew, Michael↔Mike, William↔Will/Bill,
   Elizabeth↔Liz/Beth, Robert↔Rob/Bob, Daniel↔Dan, Jonathan↔Jon, etc.
3. **Reverse-anchor — company/domain first**: `Acme founder x.com`,
   `elsewhere.zone twitter`, or `<company-website> twitter`. The right
   person's profile usually lists their company in the bio, which is
   stronger confirmation than a name match.
4. **Direct site query**: `site:x.com "Full Name"` or
   `site:twitter.com "Full Name" Company`.

### Identity confirmation — POSITIVE evidence required

When committing a non-null handle, your `reason` field MUST cite the
SPECIFIC piece of evidence that confirmed identity. Acceptable forms:

- "Bio mentions <Company>" — quote the relevant phrase.
- "Profile bio links to <company-domain>" — name the URL.
- "Pinned tweet is about <Company> product."
- "Linked LinkedIn URL on the profile matches the founder."
- "The company's official X account interacts with this handle as the
  founder."

**FORBIDDEN as reasons for committing a non-null handle:**

- "no conflicting evidence found"
- "consistent with the founder profile"
- "matched via search results"
- "no contradicting information"
- Any phrasing that asserts identity by *absence* of contradiction
  rather than by *presence* of confirming evidence.

If you cannot write a sentence quoting concrete confirming evidence,
the answer is null. The Daniel Hussain regression (committed
`@danialhussain04` for someone whose real handle was
`@DanialHussain_`) happened because the cell reasoning was "no
conflicting evidence found" — that's the trap this rule prevents.

Name match alone is NEVER enough. There are many people with the same
name. If the candidate profile is private / has no bio / no linked
URLs, prefer null.

### LinkedIn as a name-bridge

If a LinkedIn URL is already in the row (or you find one cheaply during
your searches), open it conceptually: people often use a different
first-name form on social platforms than on a directory listing. Andrew
Baran's LinkedIn says "Andy" — that's the version that appears on X. If
the LinkedIn first-name differs from the source name, prefer the
LinkedIn version as the anchor for your X searches. This is the path a
human would take and it works for the cases where pure-source-name
search misses.

### When null IS the right answer

- Four distinct searches above, no candidate whose bio confirms the
  company / role / domain.
- Multiple plausible profiles and bio info can't disambiguate them.

In both cases, in the `reason` field, **state which queries you tried
and what you saw** — not just "couldn't find a profile". This makes
the trace useful for the next pass.
