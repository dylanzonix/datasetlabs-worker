---
name: find_x_handles
description: Finding X (Twitter) handles for individuals
applies_to: [cell_agent]
triggers:
  - x handle
  - x_handle
  - x account
  - x username
  - twitter
  - twitter handle
  - twitter_handle
  - twitter account
  - twitter username
  - x.com
---

## Finding someone's X (Twitter) handle

The same person often shows different names across platforms. "Andy"
on a directory listing may be "Andrew" on X. Maiden vs. married surnames,
nicknames, initials, or a handle that drops the last name entirely.

### Hard floor: at least 2 distinct searches before any null

**You MUST run at least 2 distinct `web_search` queries with materially
different angles before calling `set_values(null)`.** One pass that
returns nothing or returns only unverifiable matches is NOT grounds to
null this cell — it is the signal to widen the search, not to bail.

A null after only one search is the failure mode this skill exists to
prevent. The Paul Sawaya regression (real account `@automin`, found by
the user with one search) happened because the cell agent did exactly
one query and quit. Don't repeat that.

If after 2 distinct searches you still cannot confirm a handle, then
null is appropriate — but the second search must be MEANINGFULLY
different from the first (different angle below), not the same query
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

### Identity confirmation before committing a non-null

- The X bio mentions the company / role / domain you already know.
  **Name match alone is NOT enough** — confirm via bio text or a linked
  website. A LinkedIn or company URL on the X profile counts.
- If the only candidate profile is private / has no bio / no linked
  URLs, prefer null over a guess.

### When null IS the right answer

- Two distinct searches above, no candidate whose bio confirms the
  company / role / domain.
- Multiple plausible profiles and bio info can't disambiguate them.

In both cases, in the `reason` field, **state which queries you tried
and what you saw** — not just "couldn't find a profile". This makes
the trace useful for the next pass.
