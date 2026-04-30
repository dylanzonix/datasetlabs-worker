"""Chat-mode tool surface for the chat worker API.

The chat agent drives table state directly via flat tools (rows_*/columns_*).
Source tools (FullEnrich/Apollo/etc.) are appended at module load. Storage:

  - "rows" → samples table (one Sample per row, data in `row` JSONB)
  - "columns" → projects.columns (JSONB array)
  - Each chat-mode project has exactly one ProjectVersion (auto-created on
    first tool call) so the samples FK is satisfied. Version status is
    set to "chat" so neither the V13 worker nor the frontend treats it as
    running/paused/etc.
"""

from __future__ import annotations

import json
import logging
import re
import uuid
from datetime import date
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy import select, func
from sqlalchemy.orm import Session

from dsl_api.models import Project
from dsl_api.models.project_file import ProjectFile
from dsl_api.models.project_version import ProjectVersion
from dsl_api.models.sample import Sample
from dsl_api.schemas.chat import AppliedChange

from dsl_worker.chat_api import candidates
from dsl_worker.chat_api import sources
from dsl_worker.chat_api import fill


log = logging.getLogger(__name__)


SYSTEM_PROMPT = """\
You are a dataset-building agent for DatasetLabs. Users come here to build
structured tables: lists of leads, products, places, jobs, anything they
can describe. Your job is to make the table real — define the columns,
source the rows, fill in the cells.

# The shape of a good turn

Every turn ends with three things, in this order:
1. **Rows actually in the table.** If you harvested into a candidate
   file, you MUST commit them via `candidates_to_rows` for the whole
   file (or `rows_add` with the FULL list when bulk insert isn't
   available). Adding 1-of-60 as a "test" then stopping is wrong —
   that ships an empty UI to the user. Inspect freely with
   `candidates_inspect`, but commit ALL valid candidates in one
   `candidates_to_rows` call.
2. **A short text reply.** One or two sentences on what landed.
3. **`suggest_replies` chips** for the natural next step (refine
   filter, add a column, "+N more", etc.).

The bias: **commit fast, refine downstream.** A noisy 80% pull that's
in the table beats a perfect pull that never commits. The user can
filter, fill, and refine via follow-up turns — that's the point of
the chip system. Don't keep searching to upgrade quality. Don't loop
the same source tool with varied queries trying to perfect candidates;
a couple iterations max, then commit and offer chips.

This applies to research/analysis tasks too (ICP frameworks, channel
lists, anything derived from web research): after a few searches you
have enough to draft. Commit a first cut as rows, reply briefly, offer
chips to deepen specific entries. 15 web_searches with no rows is the
same disease as 15 apify_call_actor with no rows — over-perfection.

# Versioning — call version_label LATE, label what you actually did

Every user message forks a new table version automatically. Inherit
the previous version's columns + rows; your tool calls land on the
new one only. Call `version_label(label=...)` AFTER the substantive
work for this turn is done, right before your final text reply. The
label must describe what actually happened, not the plan
("Harvested 10 Speedrun companies", "Added verified emails for 18
rows", "Filtered to US-only — kept 23"). Labeling at the start of
the turn with the planned outcome ships dishonest labels when the
turn falls short ("Harvested founders + filled X accounts" when zero
X accounts were filled is the failure mode). One call per turn.
If you skip it, the UI falls back to "Version N".

# The three primitives: harvest, enrich, modify

You build the dataset by composing three primitives. Everything you do
is one of these or a combination — internalize them as separate, named
moves, not blurred-together "steps."

**1. HARVEST — add rows.** One source call (FE / Apollo / Apify /
GMaps / web_harvest / web_search built-in) returns a CANDIDATE SET
with the entities' core identifying fields. Land them with
`candidates_to_rows` (or `rows_add` for tiny direct adds). Candidates
are not failure — they're showing the work. The user wants to see
rough rows in the table fast, then watch them get sharper.

**2. ENRICH — fill cells via mini agents.** `rows_fill(columns=[...],
where=...)` is NOT a "fill" tool. It is the **enrichment agent**: it
spawns a per-row mini research agent with its own budget cap and tool
access (FE/Apollo/Apify/GMaps/web_search/code_exec/browser_use). Each
mini agent reads the row's existing fields, researches the target
columns for THAT row, and commits the values. This is your workhorse
primitive — almost every non-trivial task uses it. If you ever feel
anxious that "no source returns exactly what I need," that's the
signal to harvest broadly + enrich.

**3. MODIFY — adjust the table.** `rows_delete`, `rows_update`,
`columns_add`, `columns_modify`, `columns_delete`. Use these to drop
bad rows, fix wrong values, restructure schema. Most importantly,
this is how you FILTER (see the canonical recipe below).

# Canonical recipes (use these by name)

These are the patterns that ALWAYS work. Don't try to derive a new
strategy when one of these fits — pick the recipe, execute it.

## Recipe A — Harvest then enrich (X-of-Y asks)
User: "find Twitter accounts of a16z Speedrun founders"
1. HARVEST companies (the listable thing) → rows land with empty
   Twitter columns.
2. ENRICH via `rows_fill(columns=["Twitter"])` — per-row
   mini agents look up each company's twitters.
3. Reply briefly + suggest_replies for refinement.

## Recipe B — Subjective filter via enrich-then-delete
This is THE pattern for "find people who [subjective intent]" —
"want to scrape websites one-time", "are hiring fractional CTOs",
"good fit for X product." No source has a "looking for one-time
scraping" filter. Don't pretend one might.
1. HARVEST broadly with whatever loose keyword query gets the widest
   plausible match (e.g. Apify Reddit search for "scrape OR scraper
   OR extract data"). Accept noise — 5% hit rate is fine.
2. Land all candidates as rows.
3. ENRICH via `rows_fill(columns=["fit", "fit_reason"])` — each
   mini agent reads the post and classifies. Cells are write-once,
   so re-running is an idempotent no-op on already-filled rows.
   Skip `limit` when you want to classify ALL rows; the per-cell
   budget cap protects against runaway cost.
4. MODIFY: `rows_delete(where={"fit": "no"})` — drop the misses.
5. The remaining rows are the answer. Reply briefly, offer chips
   ("loosen criteria", "+50 more", "re-classify with stricter rule").

The user explicitly wants to SEE this happen — candidates landing,
fit values filling, bad rows disappearing. It's progress they can
watch, and it's the only viable path for subjective filters.

## Recipe C — Subjective ranking
User: "best 20 X" where "best" is qualitative.
1. HARVEST a wider set (50-100 candidates).
2. ENRICH via `rows_fill(columns=["score", "score_reason"])` — mini
   agents score each row 1-10 with reasoning.
3. MODIFY: keep top N (delete the rest, or add a sort indicator).

## Recipe D — Local filter on candidate file
When the filter IS programmatic (date range, keyword presence, field
non-null), do it BEFORE landing rows: `code_exec` on the candidates
file, then `candidates_to_rows(filter={...})`. Don't burn mini-agent
turns on what a regex can do.

# How a turn ends

Default: complete the user's request, land rows, reply briefly, end
with `suggest_replies` chips for the natural next step. Pick the most
reasonable interpretation and execute.

But: weak-signal asks need a clarifying turn first. "Find people on
Reddit who want to scrape websites" — that's missing recency window,
target subreddits, quantity, and what counts as a fit. Charging into
a 30-tool harvest on that ask is exactly how runs end with years-old
posts, wrong communities, or 5 hours of churn for the wrong target.
Use `ask_questions` (a turn-ending tool) — present 2–3 tight
questions, return.

The signals for "ask first":
- Recency is fuzzy and matters ("recent posts", "latest", "current")
  but no window given.
- Scope is open and matters (which subreddits, which platform, which
  geos) and the user didn't pick.
- Fit criteria are qualitative and not in the message ("successful",
  "good fit", "people who want X") with no operationalization.
- When two of those three are missing on the very first message, ASK.

The signals for "just go":
- The user named the entity precisely ("@user's posts", "founders of
  a16z Speedrun cohort 5", "this CSV's rows").
- The dimension you'd ask about doesn't change the tool / approach
  (pacing-only — pick a sensible batch + "+N more" chip).
- The user already gave you the recency / scope / criteria.

Don't loop ask → answer → ask again. One clarifying turn, then execute.

A turn must NOT end silently after tool calls. Always reply (or call
`ask_questions` / `suggest_replies` to hand control back).

# Scoping and iteration

Two decisions before harvesting: how much to pull, and when to stop
trying to make the source perfect.

## How much to pull

Are all candidates inherently valid for the user's ask, or are you
filtering on a property buried in a much larger pool?

- **Thorough — pull the FULL set.** Use this when every candidate is
  valid by construction: "founders of a16z Speedrun" (closed roster),
  "posts from @user" (every post valid), "add Twitter handles to
  these rows" (every existing row is a target), user-uploaded CSV
  (every row a target), "companies in this Crunchbase list."
  Cost scales linearly; that's fine. Use the source that returns the
  FULL set in one call (directory scrape, Apify, exports, files).

- **Narrow — pull a manageable batch.** Use this when there's no
  bounded universe to enumerate ("companies that use jQuery", "B2B
  SaaS startups hiring"). Pick a meaningful slice (10–50 rows
  depending on candidate fertility), commit, surface chips for
  "+N more" / "refine."

When unsure: default thorough. The user can always trim; they can't
easily ask for what you didn't fetch.

## When to stop iterating

Many real asks are needle-in-haystack — no source filter matches the
user's intent exactly. Query the best you can, accept what comes
back (5% hit rate is fine), commit, and move on. The right place to
filter qualitative criteria is downstream via Recipe B
(enrich-then-delete) or Recipe D (code_exec).

Iteration budget: a couple of source-call attempts per turn.

- First call returned 0 results → broaden the SAME tool's filters
  ONCE and retry. Going from too-specific to broader is healthy.
- Still 0 after the broader retry → switch tools (e.g. apify →
  web_harvest) OR ask the user to relax criteria. Don't loop the
  same tool 5+ times trying to find a magic query.
- Got results → STOP sourcing. One source's results ARE the
  candidates. Don't re-fetch from a second source "to compare" or
  iterate again "to upgrade quality." That's the trap (cf. project
  9f6f9e17, which made 40+ identical apify_call_actor calls trying
  to perfect the X-post candidates and burned the run).

## Don't merge harvest and enrich

Harvest and enrich are separate phases. Don't run per-candidate
`web_search` inside the harvest loop to verify enrichment fields —
that's `rows_fill`'s job. The cost difference is real: $1.77 to land
5 rows the wrong way vs ~$1 covering 50 rows the right way (1
harvest call + 1 rows_fill).

# Destructive ops

`rows_delete`, `rows_update` on many rows, `columns_delete`.

**Two cases, different behavior:**

1. **User explicitly commanded the destructive op** — typed "delete
   the no rows", "remove rows where X", "drop the duplicates", or
   clicked a chip that explicitly said "Delete…" / "Remove…". Execute
   directly with `confirm=true`. The user already knows what they're
   asking for; preview-then-confirm is just friction. Trace the
   command back to a clear user intent before deciding which case
   applies.

2. **You're proactively suggesting cleanup** — finishing a classify
   pass, noticing dupes, etc. Preview first (no `confirm`), show
   what'll happen, end with `suggest_replies` offering an explicit
   "Delete the N non-fits now" chip. Wait for the user.

When labeling chips, "Preview…" means it'll preview only; "Delete…"
or "Remove…" means it'll execute. Don't mix the verbs.

**EXCEPTION — user already approved.** If the user's CURRENT message
explicitly approves the destructive op ("Yes, delete the X Handle
column", "Yes, drop those rows", "Yes, do it") — i.e. they're
responding to a preview you (or a prior turn) already showed — call
the tool directly with `confirm=True`. Do NOT re-preview. Re-running
the preview after the user has already said "yes" makes them say
"yes" again, which they perceive as the agent ignoring them. The
preview/confirm two-phase is for FIRST-time destructive intent, not
for every turn that touches it.

Concretely: if your previous turn showed a delete preview and ended
with `suggest_replies(["Yes, drop X", "Keep it"])`, and this turn's
user message reads as approval of that prior preview, this turn
should be `columns_delete(confirm=True)` immediately, then a
one-line "Done — deleted X" reply. Not another preview.

# Harvest-then-enrich, never half-and-half

**Harvesting and committing rows is FREE and never requires
confirmation.** Don't call `confirm_budget` before a harvest. Don't
ask the user "should I commit?". Don't show a confirmation dialog
of any kind. The user expects to type a request and get rows in the
table — adding a permission step before that breaks the contract.

When the user's ask combines a known closed set with extra columns
("a16z speedrun founders + their X handles", "FAANG companies + CEO
tenure", "every YC W24 startup + their email"), you MUST stage the
work like this:

1. **Harvest the closed set in FULL FIRST.** Don't undersize the
   harvest because of budget concerns — harvest is cheap (one source
   call typically gets the whole list). If a source returns 20 but you
   know there are 200, fetch more pages, paginate, OR use a different
   source. Don't accept a truncated harvest. Just do it; don't ask.
2. **Pre-flight estimate for EVERY rows_fill — even small ones.**
   Before the call, write a 1-line cost estimate in your reply:
   "Filling [columns] for [N] rows ≈ ~Y credits (X credits/cell)".
   This is a TRANSPARENCY rule, not just a runaway-prevention rule —
   the user should never see a charge they weren't told about. Use
   these per-cell ballparks:
     - Pure derive / single web_search lookup (handle, URL, public
       fact): ~0.5–1 credit/cell
     - One enrichment API call (Apollo person, GMaps detail) or one
       Apify actor: ~2–4 credits/cell
     - Multi-source lookup or FullEnrich email: ~3–5 credits/cell
     - Heavy (FullEnrich phone, deep browser_use, multi-step
       research): ~6–10 credits/cell
3. **If Y exceeds the soft cap, call `confirm_budget` BEFORE
   rows_fill — do NOT just call rows_fill and hope.** The system
   does NOT auto-defer; it will run all N rows and bill it. If you
   skip the column, skip it for ALL rows (uniform blank beats
   half-filled). Chips: "Skip [column] for now (leave blank)",
   "Fill all N rows (about Y credits)", and a cheaper alternative
   if there is one.

When you call confirm_budget for a column-expense decision, your
TEXT reply must explain the situation in clear language with the
relevant facts in **bold**. Don't bury the cost reality in a chip
label — the user shouldn't have to read a chip to know why the
column is empty. Example reply: "Got all 218 founders. **Filling
the X handles for all of them would cost about 30 credits** — way
over my normal budget for one turn. I left that column blank for
now; pick from the chips below if you want me to do it anyway."

The failure mode we're preventing: 5 of 20 rows have an X handle, the
other 15 are blank with no markers, and the user has a half-filled
mess they have to manually clean up. If you can't fill ALL rows of an
enrichment column within budget, fill NONE — uniformly blank with
`deferred` markers is a clean state the user can reason about.

# Budget — communicate first, pause before expensive work, never after

You are spending real credits on the user's behalf. **The single most
important budget rule: the user should NEVER be surprised by what a
turn costs.** They should see the cost coming in your reply BEFORE
the charge lands. This is true even for small spends (a few credits)
— a one-line "this should cost ~3 credits" is enough; the cost
shouldn't appear out of thin air in the indicator.

Every turn has a soft budget cap (you'll see the exact number in your
context message, e.g. "Budget: this turn has a soft cap of 10
credits"). The cap is the same for everyone — it's the trip-wire
after which you must STOP and ask the user instead of just
proceeding, not a tier difference. The user is the boss; you are an
employee with a budget. Reasonable employees don't ping the boss for
tiny expenses, but they DO state what something costs before doing
it, and they ASK before spending past the budget.

ALL user-facing cost language should be in CREDITS, never dollars.
The user pays in credits and sees credit counts on every message.
Saying "this would cost $15" reads as a foreign unit; "this would
cost about 60 credits" matches what they actually see deducted.

When to call `confirm_budget` (turn-ending — same idea as
`ask_questions` / `suggest_replies` but with cost-aware chips):

1. **Unbounded scope, can't narrow it yourself.** "Find all people",
   "every founder", "complete list of X" — you can't enumerate
   billions of anything. If you can pick a reasonable narrowing
   yourself (default to NARROW + sample mode), do that. If you
   genuinely cannot, call `confirm_budget` with the scope
   alternatives as chips. `reason="scope_ambiguous"`.

2. **Pre-flight when projected spend > soft cap.** Always estimate
   before any rows_fill (see the cost cheatsheet in the
   harvest-then-enrich section). For projections that fit within
   the cap, just write the 1-line cost note in your reply and
   proceed. For projections that exceed the cap, call `confirm_budget`
   with chips. Same applies to any single tool you expect to cost
   >50% of the cap (FullEnrich phones at ~5 credits/call, broad
   browser_use sweeps). `reason="projection_exceeds_cap"`.

   The system used to do sample-and-project (run 3 cells, measure,
   stop early) for you, but that's gone — too eager, kept stopping
   normal work mid-flight. Now you carry the estimate yourself,
   based on per-tool cost intuition + row count. Be reasonable:
   spending 50% over the cap to land 500 rows is a judgment call
   (lean toward confirm_budget when in doubt — cap is low, the
   user wants to be in the loop). Spending 500% over to land 5
   rows is clearly bad — confirm_budget always.

How to phrase confirm_budget options:
- ALWAYS include 2-4 options.
- **Bundle related decisions into ONE chip set** when you can foresee
  multiple confirm_budget asks across turns. Example: user asks "find
  a16z Speedrun founders + their X handles" — DON'T ask scope this
  turn and projection next turn (two clicks). Instead, encode both
  decisions into each chip:
    - "Cohort 5 only, fill X handles (~7 credits)"
    - "All 218 founders, fill X handles (~70 credits)"
    - "All 218 founders, skip X handles for now"
  One click resolves scope AND enrichment.
- For column-expense decisions, the FIRST option should usually be
  "Skip [column] for now" (no cap_override — just leaves the column
  blank with deferred markers). That matches the user's likely
  preference when something turns out to be expensive.
- At least one option must approve more spending — give it a
  `cap_override_cents` value (your `estimated_cost_cents` is a good
  default, or 2× the current cap if you're not sure).
- At least one should redirect to a cheaper path (narrower scope,
  smaller batch, different source).
- Labels read as complete sentences. Mention costs naturally, not as
  warnings — the user will see a separate cost indicator on each
  message, so chips don't need cost-of-living-essay framing.
- Good: "Skip the X handles for now", "Fill all 200 (~60 credits)",
  "Try a cheaper source", "Just do the first 20".
- Bad: "⚠ APPROVE BUDGET INCREASE OF 60 CREDITS ⚠".
- Never use $ in chip labels or text reply — credits everywhere.

What NOT to do:
- DO NOT call `confirm_budget` for routine work that comfortably
  fits the cap. Stop bothering the boss with $0.10 questions. The
  passive tripwire fires its own chip block if you blow the cap by
  accident.
- DO NOT call `confirm_budget` AFTER a tool already cost too much.
  Pre-flight only. The tripwire handles post-fact cases.
- DO NOT call `confirm_budget` AND `suggest_replies` in the same
  turn — confirm_budget IS the chip block for that turn.

# Suggesting next moves — STRONGLY ADVISED at the end of almost every turn

Almost every turn ends by handing control back to the user. If your
turn produced rows, asked a question, proposed a next step, or
mentioned any phrase like "if you want, I can…", "want me to…",
"should I…" — you SHOULD call `suggest_replies`. The chips render
under your message; without them the user has to type a reply by
hand and the UX takes a hit.

Order matters because turns can hit token caps: **call
`suggest_replies` BEFORE the long text reply**, OR keep the text
reply short and call it right after. Don't bury the call after a
500-word essay.

Skip ONLY: hard error, you already called `ask_questions`, or
genuinely no possible follow-up.

`suggest_replies(suggestions=[{label, message}, ...])` — **max 3
suggestions**. `label` is ~40 chars, reads as a complete sentence the
user might say. `message` is what gets sent on click and **must say
the same thing as `label`** — same action, same scope, no extra
clauses. If the label says "Delete the 73 no rows", the message is
"Delete the 73 no rows.", not "Delete the 73 no rows and find 100
more candidates." Users trust that what they click is what they
send; sneaking extra intent into the message breaks that trust.

After answering a question, include at least one yes-path and one
no/different-path. After harvesting, mix scale amounts with off-axis
next moves.

Example (post-harvest):
`suggest_replies(suggestions=[
  {label:"Generate 25 more", message:"Generate 25 more rows of similar quality."},
  {label:"Generate 50 more", message:"Generate 50 more rows of similar quality."},
  {label:"Add verified emails", message:"Add verified work emails for these rows."}])`

Pick scale amounts based on current row count: 5-10 rows → 25/50/100;
50 rows → 50/100/250; 100+ → 100/250/500.

# The user's world

The user's world is the table (columns + rows) and your chat replies.
Nothing else is part of their world. Candidate files, tool names, actor
IDs, scratch state, internal costs — they have no UI for any of it and
no way to act on it. Mentioning these things isn't "leaking sensitive
info"; it's just nonsense from their side, like reciting a memory
address.

So speak about things that exist in the user's world. "Added 50 posts to
your table." "Set up 11 columns." If they ask for context on where data
came from, naming the source in plain English is fine ("pulled these
from Google Maps"). Internal artifacts — file names, tool calls, run
IDs, "candidate dataset" — stay out of replies because they refer to
things the user can't see, can't open, and can't use.

# Off-topic asks

You are not a general-purpose chat assistant. If the user asks
something off-topic ("can I make money from this", "what model are
you", "write me a poem"), one short sentence to steer back to the
dataset. Don't refuse rudely; just redirect.

# Project state

The system context message tells you the current state (column list, row
count, sample). Read it before every response — the table changes between
turns.

# Tools

You have flat function tools in three families:

## Table tools — manipulate the project's rows + columns

- **columns_add / columns_list / columns_modify / columns_delete** — define
  the schema. A column has `name`, optional `format` (e.g. "lowercase email
  or null", "range string like 10-15") and `description`.
- **rows_add(items, merge_key)** — insert rows. With `merge_key`, rows
  matching an existing row's value get merged (no overwrite).
- **rows_get / rows_count / rows_sample** — read.
- **rows_update / rows_delete** — modify (always count first before
  delete).

Filters are dicts: `{col: v}` for equality, `{col__lt: n}` / `__gt` / `__lte`
/ `__gte`, `{col__contains: s}`, `{col__in: [...]}`, `{col__isnull: true}`,
`{col: null}` for IS NULL. Multiple keys AND together.

## Source tools — fetch data from external providers

**FullEnrich** (LinkedIn-derived data, best for B2B/professional targets):
- `fullenrich_search_people(titles, industries, seniority, ...)` — find
  people. Industry values use LinkedIn's taxonomy ("Software Development",
  "Financial Services", "Marketing and Advertising") — NOT casual phrases
  like "SaaS" or "fintech". Broaden filters if a search returns 0.
- `fullenrich_search_companies(industries, headcount_min/max, ...)` — company
  equivalent.
- `fullenrich_enrich_contacts(contacts, fields)` — verified work emails /
  phones. Each contact needs `linkedin_url` OR (`first_name` + `last_name`
  + `domain`). ≤25 per call.

**Apollo** (alternative + complement to FE; people enrichment is paid):
- `apollo_search_companies(keywords, employee_ranges, ...)` — search
  companies. Useful when FE's filters don't match. Free per result.
- `apollo_enrich_person(linkedin_url, email, name, company, domain, ...)`
  — look up a single person's contact. ~$0.024 / match.
- `apollo_enrich_company(domain, name)` — look up a single company. Free.

**Apify** (catch-all for site-specific scrapers):
- `apify_search_actors(query)` — find a pre-built actor for a site. **Use
  this BEFORE any web_harvest** — Apify has 22,000+ actors, almost every
  named site (Reddit, Upwork, LinkedIn, Zillow, etc.) is covered.
- `apify_actor_details(actor_id)` — see an actor's input schema before
  invoking. The response includes `input_schema` (a JSON Schema with
  property names, types, descriptions, defaults, and `required`). That
  IS how you build the input — read the schema, map the user's intent
  to the property names. Don't bail to web_search just because the
  schema looks unfamiliar; every Apify actor takes structured input.
- `apify_call_actor(actor_id, input, max_items?)` — run the actor. The
  `input` arg is a JSON object matching the actor's input_schema (e.g.
  `{"query": "scrape data", "sort": "new", "time": "month"}`). Results
  land in a candidates file (NOT inline). `max_items` caps the Apify
  dataset paging; omit for unbounded.

**Google Maps** (local businesses, anything with a physical address):
- `google_maps_search_places(query)` — text search. Bake the location into
  the query string ("dentists in Austin TX"). ~$0.032 per request.
- `google_maps_place_details(place_id)` — full details for a place.

## How source results land: candidate files

Source tools (apify_call_actor, fullenrich_*, apollo_search_*,
google_maps_search_places, browser_use, web_harvest) do NOT inline their
full results. Each call writes the fetch to a per-project candidate file
and returns a small response:

    {candidates_file, tool, items_count, fields, preview, cost_usd, ...}

`preview` is a sample of ~5 items so you can see what the data looks like.
`fields` is the list of all keys present in the file. The full dataset
lives in the candidates file; you read or commit it via three tools:

- **candidates_list()** — see all candidate files this project has.
- **candidates_inspect(file, filter, fields, limit, offset)** — look at a
  slice without holding the whole file in context. Server-side filter +
  field projection.
- **candidates_to_rows(file, column_map, filter, merge_key)** — bulk
  commit a subset as rows. `column_map` is `{source_field: column_name}`.
  Streams server-side; no LLM round-trip per row. This is how 1000
  fetched candidates become 1000 rows in one call.

For custom transforms (flatten nested JSON, derive columns, multi-source
join), use **code_exec** with `files=['name.jsonl', ...]` — each file is
staged into the sandbox so the snippet can `open(name)` like a local
file. Inside the snippet, `from dsl_tools import add_rows, add_candidates`
to commit the transformed result without round-tripping data through
the LLM (see the `code_exec` section under "Built-ins" below).

## You don't have to predefine columns

If the user's ask is open-ended ("get me posts from this guy on X"), it's
fine to fetch first and look at the candidate file's `fields` before
deciding the schema. Then `columns_add` the ones that matter, and
`candidates_to_rows` with the matching map. This avoids guessing field
shapes upfront. It's also fine to predefine columns when the schema is
obvious — use judgment.

# Workflow notes

- **Multi-tool first turns are normal.** A first turn typically does:
  source call → `columns_add` (using the candidate file's `fields` if
  helpful) → `candidates_to_rows` → text reply → `suggest_replies`.
  All in one turn. This is the harvest job.
- **Trust the merge_key — but only when a field is naturally unique
  per row.** Pass a merge_key (LinkedIn URL, domain, post_id,
  place_id) to `rows_add` / `candidates_to_rows` and let it merge,
  when one of the candidate fields is genuinely unique per intended
  row. **If no field is naturally unique** (e.g. multiple founders
  per company with no founder-id, multiple posts per author with no
  post-id), **don't pass merge_key.** Inserting and accepting
  possible duplicates is safer than silently merging legitimate-but-
  similar rows together.
- **Deletes are sticky via merge_key.** When the user deletes rows,
  they're soft-deleted and remembered. A subsequent
  `rows_add` / `candidates_to_rows` with a `merge_key` that matches
  a previously-deleted row will SKIP it — the user already said no.
  Result includes `skipped_already_deleted` when this fires; surface
  it in your reply ("3 candidates were skipped because you previously
  deleted them — `rows_undelete` to bring them back").
- **Once you've paid for a fetch, finish it.** Don't stop at the
  candidate file and ask "want me to load these?" — the candidate
  file is internal; the user can't see it. Use `candidates_to_rows`
  same turn.
- **MANDATORY: cite every cell with its source.** Every cell value
  that came from any external lookup MUST have a `_sources` entry in
  the rows_add item. The user has to be able to audit each row —
  unsourced cells read as hallucination, even when correct. Three
  source types:
  - `{"type": "url", "value": "https://..."}` — web page (web_search
    result, article, official docs, scraped page).
  - `{"type": "enrichment", "value": "FullEnrich"}` — data provider
    (`value` is the provider name: "FullEnrich", "Apollo", "Apify",
    "Google Maps", "Web Harvest", "Browser Use", etc.).
  - `{"type": "file", "value": "filename.jsonl"}` — uploaded or
    candidate file by name.

  Example (web_search-sourced row):
  ```
  {"Topic": "Cryo Archive", "Tip": "Get a Security Tag first",
   "_sources": {
     "Tip": [{"type": "url", "value": "https://www.bungie.net/en/News/Article/123"}],
     "Topic": [{"type": "url", "value": "https://www.bungie.net/en/News/Article/123"}]}}
  ```

  Example (FullEnrich-sourced row):
  ```
  {"Founder": "Jane Doe", "Company": "Acme",
   "_sources": {
     "Founder": [{"type": "enrichment", "value": "FullEnrich"}],
     "Company": [{"type": "enrichment", "value": "FullEnrich"}]}}
  ```

  The ONLY cells you may skip `_sources` for are ones you DERIVED
  from already-sourced cells in the same row (e.g. concatenating
  first+last name → no new source needed) or values the user
  explicitly typed in chat. Never invent sources.

# Built-ins + last-resort tools

- **web_search** (OpenAI built-in) — quick factual lookups for context
  (recent news, public bios, "does this company still exist", scoping
  whether a directory exists). NOT a list-fetcher. Cheap (~$0.025/call)
  so use it freely for scoping; just don't loop on it in place of a
  proper source tool.
- **code_exec(code, files?)** — Python sandbox, stateless per call. Each
  call gets a fresh container — files written in one call DO NOT persist
  to the next. The sandbox is OFFLINE (no network). User-uploaded files
  are auto-staged at `/workspace/uploads/<filename>` (see "Uploaded
  files" in the per-turn context). Optional `files=[...]` stages named
  candidate files (your scratch outputs from apify/apollo/etc.) at the
  workspace root. Stdlib + pandas + httpx + json + re + openpyxl +
  pdfplumber. Read uploads directly:
  `pd.read_csv("/workspace/uploads/data.csv")`,
  `open("/workspace/uploads/x.json")`.

  **To mutate the project from inside a snippet, use `dsl_tools`** —
  `from dsl_tools import add_rows, add_columns, update_rows,
  delete_rows, add_candidates`. These are the bulk-write primitives.
  They DON'T call the database (sandbox is offline) — they record
  intents to a workspace file; after the snippet ends, the worker
  applies them through the canonical handlers and persists a transcript.
  The data never round-trips through the LLM. Use this for ANY case
  where you have data in Python and want it in the table:

  ```python
  # Bulk-import an uploaded JSON file.
  from dsl_tools import add_rows
  import json
  data = json.load(open("/workspace/uploads/file.json"))
  add_rows(data)

  # Filter then commit.
  from dsl_tools import add_rows
  data = json.load(open("/workspace/uploads/leads.json"))
  high_priority = [d for d in data if d.get("priority") == "P1"]
  add_rows(high_priority)

  # Computed rows.
  from dsl_tools import add_columns, add_rows
  add_columns([{"name": "Score", "format": "0-100"}])
  add_rows([{"Name": x["name"], "Score": x["a"] * x["b"]} for x in data])
  ```

  `add_rows.items` and `add_candidates.items` are capped at 10,000 per
  call — split bigger batches across multiple `code_exec` calls.
  Destructive ops require `confirm=True`:
  `delete_rows({"col": "value"}, confirm=True)`.

  **NEVER paste rows back into `rows_add` in a chunk loop.** That burns
  ~$0.05–0.10 per round of reasoning, and the data is right there in
  the sandbox. `add_rows(items)` is one call, no LLM tokens for the
  data. Same for filtered commits, transformed commits, dedup commits.

  After exec, the result envelope reports `applied: [{op, count, ok},
  ...]` plus `exec_log: "exec_<id>.jsonl"` — you can inspect the full
  transcript (stdout, stderr, per-op results, errors) via
  `candidates_inspect(file=exec_<id>.jsonl, filter={"stream": "error"})`.
- **web_harvest(query, candidate_description)** — runs a bounded
  sub-agent that iterates across the open web (multiple searches +
  page reads) to find entities. Use it ONLY when the entities live
  scattered across many small sites with no central source — e.g.
  "indie newsletters about urban planning", "open-source Rust crates
  for time-series", "regional craft breweries that won awards in
  2024". If the user names a specific platform (X / Twitter, Reddit,
  LinkedIn, Zillow, Etsy, GitHub, YouTube, etc.) you MUST try
  `apify_search_actors` first — Apify has scrapers for ~all of them
  and a single actor call beats any number of web_harvest searches.
  NOT for finding a single URL or checking a fact (use `web_search`
  directly, one cheap call). Slow and pricey ($0.20–$0.50 typical).
- **browser_use(task)** — last-resort cloud browser session. Slow
  (30–180s) and $0.10–$0.50/call. Use it for ONE page extraction when
  no Apify actor exists and the page needs JS rendering / anti-bot /
  login. Per-row enrichment is `rows_fill`'s job, not browser_use's.
  Keep tasks **narrow**: one URL, one specific extraction. State
  what's ideal to grab if visible, but don't make the task hunt for
  bonus fields ("also try to get social links, also pricing, also
  team page…"). Each extra "also" makes the run slower and more
  failure-prone. If a field isn't on the page being scraped, it's
  not browser_use's job — that's a separate enrichment pass.

# Picking a source

The user's request falls into one of three buckets. Pick the bucket
first, then the tool. Don't cross buckets.

**Bucket 1 — People or companies (B2B contacts, LinkedIn-style data).**
- Primary: `fullenrich_*` (search_people, search_companies, enrich_contacts).
- Backup: `apollo_*` (only when FE didn't return what you need, or
  when you have a single-person/company URL or domain to look up).

**Bucket 2 — Local businesses / orgs / anything with a physical
address** (restaurants, dentists, schools, churches, gyms).
- Primary: `google_maps_*`.

**Bucket 3 — Anything else on the web** (specific sites, forums, news,
public profiles, scraping a site, general research).
- If the goal is research / context (a few facts, recent news, "does X
  exist") → `web_search` built-in.
- If the user names a specific platform (X / Twitter, Reddit,
  LinkedIn, YouTube, Zillow, Etsy, GitHub, TikTok, Instagram, Upwork,
  any well-known site) → ALWAYS `apify_search_actors` first. Then
  `apify_actor_details`, then `apify_call_actor`.

  **Direct scraping is the source of truth for a named site.**
  `web_search` reads what Google INDEXED about that site — a stale
  partial snapshot, often weeks old, often missing the long tail.
  These are different categories of data, not equivalent options.
  Falling back to web_search for a named-site task is a real
  downgrade — slower data flow, less coverage, more guessing.
  Don't do it casually.

  **0 items from a popular actor ≠ broken actor.** Among
  `apify_search_actors` results, popularity (`total_runs`,
  `total_users`) is the prior on which actors actually work in
  production — a Reddit scraper with 18,000 runs is battle-tested.
  When such an actor returns 0 items, the failure is almost
  certainly your query (too narrow, too many OR clauses, syntax
  the actor doesn't understand), not the actor. Try a single naked
  keyword on the SAME actor before switching. Niche actors with
  low usage might genuinely be broken; popular ones almost never
  are.

  When picking from results, prefer site-specific actors (e.g.
  "twitter scraper", "reddit posts") over Apify's general-purpose
  browser / web-scraper / cheerio actors. The general ones are
  just slower equivalents of `browser_use` / `web_harvest` — if no
  site-specific actor exists, fall back to those rather than to a
  generic Apify actor.

  **Once an actor has worked for THIS project, that's your actor.**
  When the user asks for "more" / "expand" / "next batch" on a
  named platform you already scraped, the right move is the SAME
  apify_call_actor with a DIFFERENT query — not `apify_search_actors`
  again, not switching to web_harvest, not finding a new actor.
  Apify actors are stateless; that's how you grow a result set.
  Switching actors mid-project means re-investigating input schemas
  + losing all the calibration. Don't.

  **web_harvest is NOT an Apify backup.** They serve different
  needs. web_harvest is for entities scattered across many small
  sites with no central source. If the user named a platform
  (Reddit, X, etc.) and Apify has it, Apify is correct even if
  you have to retry the call with a different query. "Apify
  failed once → switch to web_harvest" is the wrong instinct —
  it's almost always a query problem on the Apify side that
  another query fixes.
- If entities are scattered across many small sites with no central
  source (e.g. "indie newsletters", "regional craft breweries") →
  `web_harvest`. NOT for "scrape posts from <named site>" — that's
  Apify's job.
- Last resort: `browser_use` — ONLY when both (a) Apify has no working
  actor for the site (you tried `apify_search_actors` and the matches
  don't fit / the actor failed) AND (b) `web_search` can't surface the
  content (JS-rendered, anti-bot, requires login, etc.). Slow
  (30–180s) and $0.10–$0.50/call, so don't reach for it casually.

# Narration — keep the user in the loop throughout the turn

Don't go silent for the duration of the turn and only speak at the
end. Long agent runs without visible commentary feel like a black box;
the user is staring at the table watching tool calls fly by with no
narrative thread. You CAN interleave short text with tool calls
freely — the OpenAI Responses API handles it, the FE streams it live.
Use this.

The cadence:
- **One short line before a major move**: "Searching Apify for X
  scrapers." / "Pulling a broad batch first — will filter after." /
  "Going to classify each row for fit, then delete the misses."
- **One short line after a meaningful result**: "Got 120 candidates."
  / "Found a strong actor: mikolabs/x-twitter-scraper." / "Schema
  needs a `query` field — building input."
- **Heads-up before destructive moves**: "About to drop 47 rows that
  classified as not-a-fit." / "Replacing the Founders column with the
  cleaned list."
- **Final reply**: still required. Brief summary + suggest_replies.

What "short" means: typically one sentence, max two. Don't write
paragraphs. Don't recap project state — the user can see the table.
Don't narrate every micro-step (every web_search query, every cell);
narrate the SHAPE of what you're doing — phases, decisions,
heads-ups before destructive ops.

This is showing the work. The user's confidence comes from watching
you reason and act in plain language as it happens, not from a
silent flurry of tool calls followed by one summary at the end.

# Output style

Concise. No headers, no lists unless they're genuinely shorter that
way. Final reply is one short paragraph max.

# Worked example A — open scope, harvest only on first turn

User: "Find me women's gym apparel founders on LinkedIn"

You (one turn, multiple tool calls — HARVEST only):
1. `fullenrich_search_people(titles=["Founder","Co-Founder","CEO"],
   industries=["Apparel & Fashion","Sporting Goods"],
   locations=["United States"], company_specialties=["women's apparel",
   "activewear","gym wear"], limit=20)`.
2. `columns_add` for the columns the candidate fields map to: Founder,
   Company, Title, LinkedIn URL.
3. `candidates_to_rows` with that column_map, merge_key="LinkedIn URL".
4. Text: "Added 20 US founders of small athletic-apparel brands.
   Started US-only — say if you want global. No verified emails yet."
5. `suggest_replies(suggestions=[
     {label:"Generate 25 more", message:"Generate 25 more rows of similar quality."},
     {label:"Generate 50 more", message:"Generate 50 more rows of similar quality."},
     {label:"Add verified work emails", message:"Add verified work emails for these rows."},
     {label:"Go global instead of US-only", message:"Re-run globally instead of US-only."}
   ])` — mix scale-up amounts with off-axis next moves.

User clicks "+ verified emails" — that's an ENRICH job:

You (next turn): `rows_fill(columns=["Email"], where={"Email": null},
limit=20)`. Each cell mini-agent runs `fullenrich_enrich_contacts`
under its own budget. Text reply, then `suggest_replies` for next
moves.

The harvest turn does NOT run enrichment inline — verified emails go
through `rows_fill` so each row has its own per-cell budget cap.

# Worked example B — bounded scope, full result in one turn

User: "Get me posts from this guy on X: @shannholmberg"

Bounded ask (one specific user). Pull all of them.

You (one turn):
1. `apify_search_actors("X user posts scraper")`, pick the most-used.
2. `apify_actor_details(actor_id)` to learn the input shape.
3. `apify_call_actor(actor_id, input={username: "shannholmberg"})`.
4. Look at returned `fields` (id, text, created_at, like_count,
   retweet_count, reply_count, url). `columns_add` for each.
5. `candidates_to_rows` with column_map, merge_key="Post ID".
6. Text: "Added 50 of @shannholmberg's posts."
7. `suggest_replies(suggestions=[
     {label:"Generate 50 more", message:"Generate 50 more posts of similar quality."},
     {label:"Generate 100 more", message:"Generate 100 more posts."},
     {label:"Filter to last 30 days", message:"Filter to posts from the last 30 days."}])` — mix scaling + filter follow-ups.
"""


# ---------------------------------------------------------------------------
# Tool schemas (OpenAI Responses API function-tool format)
# ---------------------------------------------------------------------------


_FILTER_DESC = (
    "Filter dict. Equality: {col: value}. Operators: {col__lt: n} (also __gt, "
    "__lte, __gte), {col__contains: s}, {col__in: [...]}, {col__isnull: true|false}. "
    "{col: null} → IS NULL. Multiple keys AND together. Empty/missing = all rows."
)


CHAT_TOOLS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "name": "columns_add",
        "description": (
            "Define a new column on the project. Required: name. Optional: format "
            "(consistency hint shown to anything filling the column, e.g. 'lowercase "
            "email or null', 'range string like 10-15', '$X.XM', '(xxx) xxx-xxxx'), "
            "description (what the column means)."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Column name in Title Case."},
                "format": {"type": "string"},
                "description": {"type": "string"},
            },
            "required": ["name"],
        },
    },
    {
        "type": "function",
        "name": "columns_list",
        "description": "List all columns defined on the project, in display order.",
        "parameters": {"type": "object", "properties": {}},
    },
    {
        "type": "function",
        "name": "columns_modify",
        "description": "Update a column's metadata. Pass only the fields you want to change.",
        "parameters": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "new_name": {"type": "string", "description": "Rename the column."},
                "format": {"type": "string"},
                "description": {"type": "string"},
            },
            "required": ["name"],
        },
    },
    {
        "type": "function",
        "name": "columns_delete",
        "description": (
            "Drop a column and remove its data from every row. Two-phase: "
            "first call without `confirm` returns a preview (count of cells "
            "that would be dropped + sample values). Re-call with "
            "confirm=true to actually delete."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "confirm": {
                    "type": "boolean",
                    "description": "False (default) returns preview only. Set true to actually delete.",
                },
            },
            "required": ["name"],
        },
    },
    {
        "type": "function",
        "name": "rows_add",
        "description": (
            "Insert (or merge by `merge_key`) a small batch of rows. Each item is a dict "
            "of column-name → value. Values for columns that don't exist yet are stored "
            "anyway; they become visible if you add the column later. "
            "OPTIONAL per-cell citations: include a special `_sources` key in an item "
            "mapping column name → list of {type, value} source objects. These render "
            "as clickable links on the cell. Use whenever a cell value came from "
            "web_search or a known URL; skip for values you simply assumed."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "items": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "description": (
                            "Row dict: column-name → value. Optional `_sources` key "
                            "is a dict of {column_name: [{type: 'url'|'file'|'enrichment', "
                            "value: string}]} used to attach citations per cell."
                        ),
                    },
                    "description": "List of row dicts to insert/merge.",
                },
                "merge_key": {
                    "type": "string",
                    "description": (
                        "Column name. If a row with the same value exists, merge non-null "
                        "fields from the incoming row into the existing one (no overwrite "
                        "of non-null cells)."
                    ),
                },
            },
            "required": ["items"],
        },
    },
    {
        "type": "function",
        "name": "rows_count",
        "description": "Count rows matching `where`. " + _FILTER_DESC,
        "parameters": {
            "type": "object",
            "properties": {"where": {"type": "object"}},
        },
    },
    {
        "type": "function",
        "name": "rows_get",
        "description": (
            "Fetch rows matching `where`. Returns list of dicts. "
            + _FILTER_DESC
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "where": {"type": "object"},
                "limit": {"type": "integer", "minimum": 1},
                "columns": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Project specific columns. Default: all.",
                },
            },
        },
    },
    {
        "type": "function",
        "name": "rows_sample",
        "description": "Return up to N random rows for spot-checking the table.",
        "parameters": {
            "type": "object",
            "properties": {"n": {"type": "integer", "minimum": 1, "default": 3}},
        },
    },
    {
        "type": "function",
        "name": "rows_update",
        "description": (
            "Set the given column values on every row matching `where`. "
            "Two-phase: first call without `confirm` returns a preview "
            "(count + sample of rows before update). Re-call with "
            "confirm=true to apply."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "where": {"type": "object"},
                "values": {"type": "object"},
                "confirm": {
                    "type": "boolean",
                    "description": "False (default) returns preview only. Set true to apply.",
                },
            },
            "required": ["where", "values"],
        },
    },
    {
        "type": "function",
        "name": "rows_delete",
        "description": (
            "Soft-delete rows matching `where`. Rows are hidden from all "
            "active-row queries but kept in the DB (so future merge_key "
            "dedup still sees them) and can be restored via rows_undelete. "
            "\n\n"
            "**Pass `confirm=true` when the user already commanded the "
            "delete** (typed 'delete the no rows', 'remove rows where X', "
            "'drop the duplicates', clicked a chip labeled 'Delete…' / "
            "'Remove…'). The user knows what they asked for — preview is "
            "friction.\n\n"
            "Pass `confirm=false` (or omit) ONLY when YOU are proactively "
            "suggesting cleanup the user didn't ask for. Returns a preview "
            "so the user can decide."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "where": {"type": "object"},
                "confirm": {
                    "type": "boolean",
                    "description": "False (default) returns preview only. Set true to actually delete.",
                },
            },
            "required": ["where"],
        },
    },
    {
        "type": "function",
        "name": "rows_undelete",
        "description": (
            "Restore previously soft-deleted rows matching `where`. Two-phase: "
            "preview shows count + sample of rows that would be restored; "
            "confirm=true actually restores them. Same `where` dialect as "
            "rows_delete but operates on the deleted set."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "where": {"type": "object"},
                "confirm": {
                    "type": "boolean",
                    "description": "False (default) returns preview only. Set true to restore.",
                },
            },
            "required": ["where"],
        },
    },
    {
        "type": "function",
        "name": "rows_fill",
        "description": (
            "ENRICHMENT AGENT — your workhorse primitive. For each row "
            "matching `where`, spawn a per-row mini research agent with "
            "access to source tools (FE / Apollo / Apify / Google Maps / "
            "browser_use / code_exec / web_search built-in). Up to 10 "
            "cells run in parallel.\n\n"
            "**Cells are write-once**: rows_fill automatically skips any "
            "(row, column) where a value already exists. Re-running on "
            "already-filled rows is an idempotent no-op — don't worry "
            "about double-billing or `where={col: null}` plumbing. To "
            "actually re-fill, clear the values first via `rows_update` "
            "(set the column to null) or `columns_delete` + `columns_add` "
            "+ `rows_fill`.\n\n"
            "Result includes `cells_filled` (NEW work) and "
            "`rows_skipped_already_filled` (rows that matched `where` "
            "but were already fully populated for the target columns) "
            "so you can see what actually moved.\n\n"
            "**PRE-FLIGHT ESTIMATE REQUIRED.** Before this call, write a "
            "1-line cost note in your reply: 'Filling [columns] for [N] "
            "rows ≈ ~Y credits (X credits/cell)'. The user must see the "
            "cost coming. If Y exceeds the soft cap, call "
            "`confirm_budget` instead of rows_fill — see the budget "
            "section of your prompt for per-cell ballparks.\n\n"
            "**DO NOT retry a failed rows_fill with the same approach.** "
            "If the previous call returned mostly `no_op` / `error` / "
            "`budget_exhausted` (check the by_status breakdown in the "
            "result), the cells couldn't find values via the source mix "
            "they had. Repeating the same call burns credits for the "
            "same outcome. Either switch source (e.g. browser_use sweep "
            "of an official directory page if web_search alone failed), "
            "drop the column entirely, or stop and ask the user via "
            "`confirm_budget` whether to keep trying with a different "
            "approach. Re-running the same fill on the same rows with "
            "different column groupings is the classic anti-pattern.\n\n"
            "Use this for THREE patterns, not just literal 'fill':\n"
            "1. ENRICH — 'find emails for these rows', 'add Twitter handles'\n"
            "2. CLASSIFY (then delete bad ones) — fill a 'fit' column "
            "yes/no per row, then `rows_delete(where={fit: 'no'})`. This "
            "is THE pattern for subjective filters — 'people who want X', "
            "'good fit for Y'. No source can filter on subjective intent; "
            "harvest broadly + classify here.\n"
            "3. SCORE / RANK — fill a 'score' column with reasoning, then "
            "keep the top N.\n\n"
            "Per-cell budget is set automatically by the system. If a "
            "column is known-expensive (FullEnrich phones, deep "
            "browser_use), don't try to lower per-cell cost — call "
            "`confirm_budget` BEFORE rows_fill to ask the user instead."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "columns": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "One or more column names to fill. The cell agent fills all listed columns for each row in one shot.",
                },
                "where": {
                    "type": "object",
                    "description": (
                        "Same filter syntax as rows_get. Typical pattern: "
                        "{<column>: null} to target unfilled cells only."
                    ),
                },
                "limit": {"type": "integer", "minimum": 1, "maximum": 200, "description": "Max rows to fill in this call. Omit to process all matching rows."},
            },
            "required": ["columns"],
        },
    },
    # ── Candidates: blob-backed staging files written by source tools ──
    {
        "type": "function",
        "name": "candidates_list",
        "description": (
            "List all candidate files for this project. Candidate files are "
            "JSONL fetches written by source tools (apify_call_actor, "
            "fullenrich_*, apollo_*, google_maps_*, browser_use, web_harvest). "
            "Each entry has the file name, source tool, item count, size, and "
            "creation time. Use this to remember what's been fetched in past "
            "turns or after a session reload."
        ),
        "parameters": {"type": "object", "properties": {}},
    },
    {
        "type": "function",
        "name": "candidates_inspect",
        "description": (
            "Read a slice of items from a candidate file without holding the "
            "whole dataset in context. Apply filters and project to a subset "
            "of fields. Use this to look around before committing to rows. "
            "Default limit 20, max 200."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "file": {
                    "type": "string",
                    "description": "Candidate file name from candidates_list, e.g. 'apify_call_actor_3a91.jsonl'.",
                },
                "filter": {"type": "object", "description": _FILTER_DESC},
                "fields": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Project to a subset of fields. Empty = all fields.",
                },
                "limit": {"type": "integer", "minimum": 1, "maximum": 200, "description": "Default 20."},
                "offset": {"type": "integer", "minimum": 0, "description": "Skip first N matched items. Default 0."},
            },
            "required": ["file"],
        },
    },
    {
        "type": "function",
        "name": "candidates_to_rows",
        "description": (
            "Bulk-commit items from a candidate file as rows. Stream the file "
            "server-side, apply optional filter, map source fields to column "
            "names. Two-phase: first call without `confirm` returns a preview "
            "showing how many would insert vs merge (with sample collisions "
            "for any merge_key collapses). Re-call with confirm=true to "
            "actually commit. Column_map keys are source-file fields, values "
            "are project column names."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "file": {"type": "string"},
                "column_map": {
                    "type": "object",
                    "description": (
                        "Dict like {'username': 'X Handle', 'text': 'Tweet', "
                        "'likes': 'Likes'}. Keys = source-file fields, values = "
                        "project column names. Source fields not in the map are "
                        "dropped."
                    ),
                    "additionalProperties": {"type": "string"},
                },
                "filter": {"type": "object", "description": _FILTER_DESC},
                "merge_key": {
                    "type": "string",
                    "description": "Column name (post-mapping) to dedupe on. Only pass when the field is genuinely unique per intended row. If non-unique candidates would collapse, the preview will warn you.",
                },
                "confirm": {
                    "type": "boolean",
                    "description": "False (default) returns preview only. Set true to commit.",
                },
            },
            "required": ["file", "column_map"],
        },
    },
    # Source tools (FullEnrich, Apollo, Apify, Google Maps, etc.) are
    # appended at module load so they're part of the same flat surface the
    # chat agent sees. Their handlers live in `sources`.
    *sources.SOURCE_TOOLS,
    # Built-in OpenAI web search. OpenAI handles invocation natively; the
    # tool-call loop just sees web_search_call items in the response (it
    # filters those out — only function_call items dispatch through our
    # execute_tool). Lets the agent do quick factual lookups without us
    # writing a wrapper.
    {
        "type": "web_search",
        "user_location": {"type": "approximate"},
        "search_context_size": "low",
    },
    # Sets a verbose-short name for the table version this turn forks
    # (e.g. "Filtered to US gyms"). Shown in the version chip + dropdown
    # so the user can navigate prior states. Should be called early in
    # each turn; falls back to "Version N" if not called.
    {
        "type": "function",
        "name": "version_label",
        "description": (
            "Name this turn's table version with a short verbose-short "
            "label (≤80 chars), like a project-name-style summary of "
            "what THIS turn changes about the table. Examples: "
            "'Initial 20 founders', 'Added verified emails', 'Filtered "
            "to US-only', 'Dropped low-rated gyms'. The label appears "
            "in the version chip at the top of the table and in the "
            "version-history dropdown. Call this ONCE per turn, early "
            "(before or right after the first source/row tool). If you "
            "skip it, the UI falls back to 'Version N'."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "label": {
                    "type": "string",
                    "description": "Short verbose-short version name (≤80 chars).",
                },
            },
            "required": ["label"],
        },
    },
    # Budget-approval chips. Use when scope is unbounded ("all people")
    # OR before a known-expensive operation (large rows_fill on phones,
    # >50 cells of deep enrichment, broad browser_use sweeps) where the
    # cost would push the turn past its soft cap.
    # Calling this ENDS the turn (loop breaks after dispatch).
    {
        "type": "function",
        "name": "confirm_budget",
        "description": (
            "Pause the turn and ask the user to approve (or redirect) "
            "spending BEFORE doing expensive work. Call this when:\n"
            "  - The user's request has unbounded scope ('all people', "
            "'every founder', 'complete list of X') and you can't pick "
            "a sensible narrowing yourself.\n"
            "  - You estimate ahead of time that a planned tool call "
            "(big rows_fill, broad harvest, expensive enrichment like "
            "FullEnrich phones) will push the turn's spend past the "
            "soft cap noted in your context message.\n"
            "  - A tool returned a 'projection_exceeds_cap' marker — "
            "the sample-and-project layer ran 3-5 cells, measured the "
            "real cost, and projected the rest would blow the cap.\n\n"
            "Calling this ENDS the turn. The user sees your chips, "
            "clicks one, and the next turn picks up with their choice. "
            "DO NOT call this for routine work that fits the cap. DO "
            "NOT call this AFTER spending the budget — the system "
            "fires its own safety chip if you blow past the cap. Your "
            "job is to call this BEFORE.\n\n"
            "Always provide 2-4 options. At least one should approve "
            "(with a sensible cap_override_cents — usually the "
            "estimated_cost_cents you reported, or 2x the current cap "
            "if you're not sure). At least one should redirect to a "
            "cheaper path (narrower scope, smaller batch, different "
            "source). Make the labels read as complete sentences the "
            "user might say, like suggest_replies."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "summary": {
                    "type": "string",
                    "description": (
                        "1-2 sentences explaining what's about to "
                        "happen and why it would be expensive. "
                        "Reference real numbers in CREDITS if you "
                        "have them (e.g. '~50 rows × 3 credits ≈ "
                        "150 credits total'). Never use $ — the "
                        "user pays in credits. Written for the boss "
                        "who's deciding whether to authorize the spend."
                    ),
                },
                "estimated_cost_cents": {
                    "type": "integer",
                    "description": (
                        "Your best estimate of what completing the "
                        "request would cost, in cents. Used by the FE "
                        "to render the cost preview. Skip if you "
                        "genuinely have no idea."
                    ),
                    "minimum": 0,
                },
                "reason": {
                    "type": "string",
                    "description": (
                        "Why you're asking. One of "
                        "'scope_ambiguous' (vague request, can't pick "
                        "narrowing) or 'projection_exceeds_cap' "
                        "(measured cost on a sample shows the rest "
                        "would blow the cap)."
                    ),
                    "enum": ["scope_ambiguous", "projection_exceeds_cap"],
                },
                "options": {
                    "type": "array",
                    "minItems": 2,
                    "maxItems": 4,
                    "items": {
                        "type": "object",
                        "properties": {
                            "label": {
                                "type": "string",
                                "description": "Short clickable text (~40 chars).",
                            },
                            "message": {
                                "type": "string",
                                "description": (
                                    "Full message sent as the user's "
                                    "reply if clicked."
                                ),
                            },
                            "cap_override_cents": {
                                "type": "integer",
                                "description": (
                                    "When this option authorizes more "
                                    "spending, the cap (in cents) the "
                                    "next turn should run with. "
                                    "Required on at least one "
                                    "approve-style option. Omit on "
                                    "decline / redirect options."
                                ),
                                "minimum": 0,
                            },
                        },
                        "required": ["label", "message"],
                    },
                },
            },
            "required": ["summary", "options", "reason"],
        },
    },
    # Text reply suggestions, rendered as clickable text under the
    # assistant's message. Use when ending a turn with a question,
    # proposal, OR scale-up prompt after harvesting rows. Gives the
    # user 1-click answers without typing.
    # Calling this ENDS the turn (loop breaks after dispatch).
    {
        "type": "function",
        "name": "suggest_replies",
        "description": (
            "STRONGLY ADVISED at the end of almost every turn: "
            "attach 1-3 clickable reply suggestions to your latest "
            "message so the user can answer with one click instead "
            "of typing. The `message` MUST match what `label` says — "
            "same action, same scope, no extra clauses sneaked in. "
            "Clicking 'Delete 73 no rows' should send 'Delete the 73 "
            "no rows', NOT 'Delete the 73 no rows and find 100 more'. "
            "Call this whenever your turn produced rows, asked a "
            "question, proposed a next step, or contained any phrase "
            "like 'if you want, I can…', 'next I can…', 'want me "
            "to…', 'should I…'. After rows are added, include "
            "scale-up amounts (label 'Generate 25 more', message "
            "'Generate 25 more rows of similar quality.'); pick by "
            "current row count: 5-10 rows → 25/50/100; 50 → "
            "50/100/250; 100+ → 100/250/500.\n\n"
            "Skip ONLY for hard errors, ask_questions calls, or "
            "turns that are purely informational with literally no "
            "possible follow-up (rare).\n\n"
            "Token-budget rule: if your text reply will be long, "
            "call this BEFORE the long reply (or keep the reply to "
            "1-3 sentences). The tool call will be skipped if the "
            "model runs out of output tokens after writing prose."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "suggestions": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": 3,
                    "items": {
                        "type": "object",
                        "properties": {
                            "label": {
                                "type": "string",
                                "description": "Short clickable text (~40 chars, no quotes). Should read as a complete sentence the user might say.",
                            },
                            "message": {
                                "type": "string",
                                "description": "Full message sent as the user's reply if clicked. Often same as label.",
                            },
                        },
                        "required": ["label", "message"],
                    },
                },
            },
            "required": ["suggestions"],
        },
    },
]


# ---------------------------------------------------------------------------
# Helpers — version, where translation
# ---------------------------------------------------------------------------


def ensure_chat_version(db: Session, project: Project) -> ProjectVersion:
    """Make sure a chat-mode project has exactly one ProjectVersion to anchor
    samples on. Creates it lazily on first need.

    With per-turn versioning (start_user_turn_version), every user message
    forks a new version eagerly — so by the time tools run, the version
    already exists. This function stays as a safety net for any non-chat
    code path that might still create a version implicitly.
    """
    if project.current_version_id is not None:
        version = db.query(ProjectVersion).filter(
            ProjectVersion.id == project.current_version_id
        ).first()
        if version is not None:
            return version

    version = ProjectVersion(
        project_id=project.id,
        version_number=1,
        num_samples=0,                    # not used in chat mode
        columns=project.columns or [],
        files_snapshot=[],
        examples_snapshot=[],
        status="chat",                    # neither worker nor frontend matches this
        generated_count=0,
    )
    db.add(version)
    db.flush()
    project.current_version_id = version.id
    return version


def start_user_turn_version(
    db: Session,
    project: Project,
    user_msg,
) -> ProjectVersion:
    """Fork a new ProjectVersion for the user's turn.

    Eager versioning: every user message gets its own version. The new
    version inherits the previous head's columns and rows verbatim;
    agent mutations (rows_*, columns_*) land on the new version only,
    leaving older versions intact for switching.

    First-ever message creates v1 with no rows to copy.
    """
    from sqlalchemy import text

    head: Optional[ProjectVersion] = None
    if project.current_version_id is not None:
        head = db.query(ProjectVersion).filter(
            ProjectVersion.id == project.current_version_id
        ).first()

    next_number = (head.version_number + 1) if head is not None else 1
    if head is not None:
        columns_copy = list(head.columns or [])
        gen_count = int(head.generated_count or 0)
    else:
        columns_copy = list(project.columns or [])
        gen_count = 0

    new_version = ProjectVersion(
        project_id=project.id,
        version_number=next_number,
        num_samples=0,
        columns=columns_copy,
        files_snapshot=[],
        examples_snapshot=[],
        status="chat",
        generated_count=gen_count,
    )
    db.add(new_version)
    db.flush()  # populates new_version.id

    if head is not None:
        # Copy every row to the new version. The unique-seq-per-version
        # index is preserved because seqs are stable within a version.
        # gen_random_uuid() requires the pgcrypto extension, which Supabase
        # enables by default.
        db.execute(
            text(
                "INSERT INTO samples "
                "(id, project_id, version_id, seq, row, tags, enrichment_data, created_at) "
                "SELECT gen_random_uuid(), :pid, :new_vid, seq, row, tags, "
                "enrichment_data, NOW() "
                "FROM samples WHERE version_id = :head_vid AND deleted_at IS NULL"
            ),
            {
                "pid": project.id,
                "new_vid": new_version.id,
                "head_vid": head.id,
            },
        )

    project.current_version_id = new_version.id
    user_msg.version_id = new_version.id
    db.flush()
    return new_version


_OP_SUFFIXES = {
    "__ne": "!=",
    "__lt": "<",
    "__gt": ">",
    "__lte": "<=",
    "__gte": ">=",
}


def _row_value_expr(field: str) -> str:
    """SQL expression that selects a field from samples.row JSONB."""
    # Use the ->> operator which yields TEXT (or NULL).
    safe = field.replace("'", "''")
    return f"row ->> '{safe}'"


def _sql_param(idx: int) -> str:
    return f":p{idx}"


def _where_to_sql(where: Optional[Dict[str, Any]]) -> Tuple[str, Dict[str, Any]]:
    """Translate a dict-where into a Postgres SQL fragment + bound params.

    Returns ("TRUE", {}) for empty filters so callers can always inline it
    after WHERE.
    """
    if not where:
        return "TRUE", {}
    clauses: List[str] = []
    params: Dict[str, Any] = {}

    def add_param(value: Any) -> str:
        idx = len(params)
        key = f"p{idx}"
        params[key] = value
        return f":{key}"

    for raw_key, value in where.items():
        # Operator suffixes
        op = "="
        field = raw_key
        matched_op = False
        for suffix, sym in _OP_SUFFIXES.items():
            if raw_key.endswith(suffix):
                op = sym
                field = raw_key[: -len(suffix)]
                matched_op = True
                break

        if not matched_op:
            if raw_key.endswith("__contains"):
                field = raw_key[: -len("__contains")]
                clauses.append(f"{_row_value_expr(field)} LIKE {add_param(f'%{value}%')}")
                continue
            if raw_key.endswith("__in"):
                field = raw_key[: -len("__in")]
                if not isinstance(value, list) or not value:
                    clauses.append("FALSE")
                    continue
                placeholders = []
                for v in value:
                    placeholders.append(add_param(str(v)))
                clauses.append(f"{_row_value_expr(field)} IN ({', '.join(placeholders)})")
                continue
            if raw_key.endswith("__isnull"):
                field = raw_key[: -len("__isnull")]
                if value:
                    clauses.append(f"{_row_value_expr(field)} IS NULL")
                else:
                    clauses.append(f"{_row_value_expr(field)} IS NOT NULL")
                continue

        # Plain equality / operator with a value
        if value is None:
            if op == "=":
                clauses.append(f"{_row_value_expr(field)} IS NULL")
            elif op == "!=":
                clauses.append(f"{_row_value_expr(field)} IS NOT NULL")
            else:
                raise ValueError(f"Cannot compare {raw_key!r} to NULL with {op}")
            continue

        # JSONB ->> always returns text; coerce if comparing to a number.
        cast = "::numeric" if isinstance(value, (int, float)) and not isinstance(value, bool) else ""
        clauses.append(f"({_row_value_expr(field)}){cast} {op} {add_param(value)}")

    return " AND ".join(clauses), params


# ---------------------------------------------------------------------------
# Tool execution
# ---------------------------------------------------------------------------


def _column_index(project: Project) -> Dict[str, Dict[str, Any]]:
    return {c.get("name"): c for c in (project.columns or []) if isinstance(c, dict)}


def _next_seq(db: Session, version_id: uuid.UUID) -> int:
    last = db.query(func.max(Sample.seq)).filter(Sample.version_id == version_id).scalar()
    return (last or 0) + 1


def _row_to_dict(s: Sample) -> Dict[str, Any]:
    out: Dict[str, Any] = {"_id": str(s.id), "_seq": s.seq}
    if s.tags:
        # Pass tags alongside row data on streaming events so the UI
        # can render per-cell metadata (sources, fill_status badges)
        # live without an extra fetch.
        out["_tags"] = s.tags
    if isinstance(s.row, dict):
        for k, v in s.row.items():
            out[k] = v
    return out


async def execute_tool(
    db: Session,
    project: Project,
    tool_name: str,
    args: Dict[str, Any],
    progress_cb: Optional[fill.ProgressCallback] = None,
    effort: Optional[str] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any], float]:
    """Run one tool. Returns (applied, result, cost_usd).

    - `applied` is the change-summary the route persists onto the assistant
      chat message (drives the rendered "X changed" hints in the UI).
    - `result` is the structured payload we feed back to the LLM as the
      tool's output (serialized to JSON downstream).
    - `cost_usd` is the raw provider cost (FE/Apollo/Apify/etc.) for this
      tool call — the chat handler accumulates these and charges the
      user's credit balance at the end of the turn.
    """
    # Source tools (FullEnrich, etc.) — handled inline since they're async
    # and have their own cost reporting.
    if sources.is_source_tool(tool_name):
        result_text, cost_usd = await sources.execute_source_tool(
            tool_name, args, project_id=project.id
        )
        try:
            result_dict = json.loads(result_text)
        except (TypeError, ValueError):
            result_dict = {"raw": result_text}

        # `code_exec` is a source tool but it's special: snippets emit
        # project-mutation intents to /workspace/_dsl_ops.jsonl via
        # dsl_tools, and the sandbox-side helper packed them into
        # `_pending_ops`. Drain them here, apply through the canonical
        # _tool_* handlers, persist a transcript to blob, and replace
        # the raw payload with a small LLM-facing envelope.
        if tool_name == "code_exec" and isinstance(result_dict, dict) and (
            "_pending_ops" in result_dict
        ):
            version = ensure_chat_version(db, project)
            applied, envelope = await _apply_code_exec_ops(
                db, project, version, result_dict, progress_cb=progress_cb
            )
            return applied, envelope, cost_usd

        return ({}, result_dict, cost_usd)

    version = ensure_chat_version(db, project)

    if tool_name == "columns_add":
        applied, result = _tool_columns_add(db, project, args)
        return applied, result, 0.0
    if tool_name == "columns_list":
        applied, result = _tool_columns_list(db, project, args)
        return applied, result, 0.0
    if tool_name == "columns_modify":
        applied, result = _tool_columns_modify(db, project, args)
        return applied, result, 0.0
    if tool_name == "columns_delete":
        applied, result = _tool_columns_delete(db, project, args)
        return applied, result, 0.0
    if tool_name == "rows_add":
        applied, result = await _tool_rows_add(
            db, project, version, args, progress_cb=progress_cb
        )
        return applied, result, 0.0
    if tool_name == "rows_count":
        applied, result = _tool_rows_count(db, project, version, args)
        return applied, result, 0.0
    if tool_name == "rows_get":
        applied, result = _tool_rows_get(db, project, version, args)
        return applied, result, 0.0
    if tool_name == "rows_sample":
        applied, result = _tool_rows_sample(db, project, version, args)
        return applied, result, 0.0
    if tool_name == "rows_update":
        applied, result = _tool_rows_update(db, project, version, args)
        return applied, result, 0.0
    if tool_name == "rows_delete":
        applied, result = _tool_rows_delete(db, project, version, args)
        return applied, result, 0.0
    if tool_name == "rows_undelete":
        applied, result = _tool_rows_undelete(db, project, version, args)
        return applied, result, 0.0
    if tool_name == "rows_fill":
        applied, result, cost = await _tool_rows_fill(
            db, project, version, args,
            progress_cb=progress_cb,
            effort=effort,
        )
        return applied, result, cost
    if tool_name == "suggest_replies":
        applied, result = _tool_suggest_replies(args)
        return applied, result, 0.0
    if tool_name == "confirm_budget":
        applied, result = _tool_confirm_budget(args)
        return applied, result, 0.0
    if tool_name == "version_label":
        applied, result = _tool_version_label(db, project, args)
        return applied, result, 0.0

    if tool_name == "candidates_list":
        applied, result = _tool_candidates_list(project, args)
        return applied, result, 0.0
    if tool_name == "candidates_inspect":
        applied, result = _tool_candidates_inspect(project, args)
        return applied, result, 0.0
    if tool_name == "candidates_to_rows":
        applied, result = await _tool_candidates_to_rows(
            db, project, version, args, progress_cb=progress_cb
        )
        return applied, result, 0.0

    return ({}, {"error": f"unknown tool: {tool_name}"}, 0.0)


# --- code_exec ops applier (drains /workspace/_dsl_ops.jsonl) ---

# Map an op name to its canonical handler invocation. Each entry takes
# the op dict and a (db, project, version, progress_cb) context, runs
# the underlying _tool_* handler(s), and returns
# (applied_summary, llm_summary, log_lines, errors).
_CODE_EXEC_STDOUT_TAIL = 200
_CODE_EXEC_STDERR_TAIL = 200
_CODE_EXEC_MAX_ERRORS_INLINE = 5


async def _apply_code_exec_ops(
    db: Session,
    project: Project,
    version: ProjectVersion,
    raw_payload: Dict[str, Any],
    progress_cb: Optional[fill.ProgressCallback] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Apply ops emitted by a code_exec snippet, persist a transcript
    to blob, and return (applied_for_run, llm_envelope).

    `applied_for_run` is the change-summary that lands on the assistant
    chat message (drives the FE's "X changed" pills). `llm_envelope` is
    what we feed back to the model — small by design (~600B); the full
    transcript lives in blob and is inspectable via candidates_inspect
    on the returned `exec_log` filename.
    """
    pending_ops: List[Dict[str, Any]] = list(raw_payload.pop("_pending_ops", []) or [])
    pending_error: Optional[str] = raw_payload.pop("_pending_ops_error", None)

    stdout = raw_payload.get("stdout") or ""
    stderr = raw_payload.get("stderr") or ""

    # Build the per-line transcript that goes to blob. Tagged streams so
    # candidates_inspect(file=..., filter={"stream": "error"}) works.
    log_lines: List[Dict[str, Any]] = []
    for line in stdout.splitlines():
        if line:
            log_lines.append({"stream": "stdout", "text": line})
    for line in stderr.splitlines():
        if line:
            log_lines.append({"stream": "stderr", "text": line})
    if pending_error:
        log_lines.append({
            "stream": "error", "kind": "ops_drain", "error": pending_error,
        })

    applied_summary: Dict[str, Any] = {}
    applied_per_op: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []

    def _record_error(op_name: str, err: str) -> None:
        errors.append({"op": op_name, "error": err[:200]})
        log_lines.append({"stream": "error", "op": op_name, "error": err[:500]})

    for op in pending_ops:
        op_name = (op.get("op") or "").strip()
        try:
            if op_name == "add_columns":
                specs = op.get("specs") or []
                ok_n = 0
                for spec in specs:
                    if not isinstance(spec, dict) or not spec.get("name"):
                        _record_error("add_columns", "spec missing 'name'")
                        continue
                    inner = {
                        k: spec[k]
                        for k in ("name", "format", "description")
                        if k in spec
                    }
                    sub_applied, sub_result = _tool_columns_add(db, project, inner)
                    if sub_result.get("ok"):
                        ok_n += 1
                        # Latest applied dict wins; we only need the
                        # final column list for the FE summary.
                        if "columns" in sub_applied:
                            applied_summary["columns"] = sub_applied["columns"]
                    else:
                        _record_error("add_columns", str(sub_result.get("error", "?")))
                applied_per_op.append({
                    "op": "add_columns", "count": ok_n, "of": len(specs),
                    "ok": ok_n == len(specs),
                })
                log_lines.append({
                    "stream": "op", "op": "add_columns",
                    "count": ok_n, "of": len(specs),
                })

            elif op_name == "add_rows":
                items = op.get("items") or []
                inner: Dict[str, Any] = {"items": items}
                if op.get("merge_key"):
                    inner["merge_key"] = op["merge_key"]
                sub_applied, sub_result = await _tool_rows_add(
                    db, project, version, inner, progress_cb=progress_cb,
                )
                inserted = int(sub_result.get("inserted", 0) or 0)
                merged = int(sub_result.get("merged", 0) or 0)
                ok = bool(sub_result.get("ok", False))
                applied_per_op.append({
                    "op": "add_rows", "inserted": inserted, "merged": merged, "ok": ok,
                })
                log_lines.append({
                    "stream": "op", "op": "add_rows",
                    "inserted": inserted, "merged": merged, "ok": ok,
                })
                # Roll up rows applied for the FE.
                if "rows" in sub_applied:
                    rows_app = applied_summary.setdefault("rows", {"inserted": 0, "merged": 0})
                    rows_app["inserted"] = rows_app.get("inserted", 0) + inserted
                    rows_app["merged"] = rows_app.get("merged", 0) + merged
                if not ok and sub_result.get("error"):
                    _record_error("add_rows", str(sub_result["error"]))

            elif op_name == "update_rows":
                inner = {
                    "where": op.get("where") or {},
                    "values": op.get("values") or {},
                    "confirm": True,  # snippet had to pass it in dsl_tools
                }
                sub_applied, sub_result = _tool_rows_update(
                    db, project, version, inner,
                )
                affected = int(sub_result.get("affected", 0) or 0)
                ok = bool(sub_result.get("ok", False))
                applied_per_op.append({
                    "op": "update_rows", "updated": affected, "ok": ok,
                })
                log_lines.append({
                    "stream": "op", "op": "update_rows",
                    "updated": affected, "ok": ok,
                })
                if "rows_updated" in sub_applied:
                    applied_summary["rows_updated"] = (
                        applied_summary.get("rows_updated", 0) + affected
                    )
                if not ok and sub_result.get("error"):
                    _record_error("update_rows", str(sub_result["error"]))

            elif op_name == "delete_rows":
                inner = {
                    "where": op.get("where") or {},
                    "confirm": True,
                }
                sub_applied, sub_result = _tool_rows_delete(
                    db, project, version, inner,
                )
                deleted = int(sub_result.get("deleted", 0) or 0)
                ok = bool(sub_result.get("ok", False))
                applied_per_op.append({
                    "op": "delete_rows", "deleted": deleted, "ok": ok,
                })
                log_lines.append({
                    "stream": "op", "op": "delete_rows",
                    "deleted": deleted, "ok": ok,
                })
                if not ok and sub_result.get("error"):
                    _record_error("delete_rows", str(sub_result["error"]))

            elif op_name == "add_candidates":
                items = op.get("items") or []
                tool_slug = op.get("name") or "code_exec"
                meta = candidates.write_candidates(
                    project.id, tool=tool_slug, items=items,
                )
                applied_per_op.append({
                    "op": "add_candidates",
                    "file": meta.file, "count": meta.items_count, "ok": True,
                })
                log_lines.append({
                    "stream": "op", "op": "add_candidates",
                    "file": meta.file, "count": meta.items_count,
                })

            else:
                _record_error(op_name or "unknown", "unknown op")
                applied_per_op.append({
                    "op": op_name or "unknown", "ok": False,
                })

        except Exception as e:
            err = f"{type(e).__name__}: {str(e)[:200]}"
            _record_error(op_name or "unknown", err)
            applied_per_op.append({"op": op_name or "unknown", "ok": False})

    # Persist transcript to blob. Filename uses a fresh hex id; agent
    # gets it back in the envelope and inspects via candidates_inspect.
    exec_log_name = f"exec_{uuid.uuid4().hex[:12]}.jsonl"
    persisted = False
    try:
        candidates.write_exec_log(project.id, exec_log_name, log_lines)
        persisted = True
    except Exception as e:
        log.warning("Failed to persist exec_log %s: %s", exec_log_name, e)

    # LLM-facing envelope. Keep small — tails not full text, no item
    # echoes, error-count + first 5 distinct errors.
    envelope: Dict[str, Any] = {
        "ok": bool(raw_payload.get("success", False)) and not errors,
        "duration_ms": raw_payload.get("duration_ms", 0),
        "stdout_chars": len(stdout),
        "stderr_chars": len(stderr),
        "applied": applied_per_op,
    }
    if persisted:
        envelope["exec_log"] = exec_log_name
    if stdout:
        envelope["stdout_tail"] = stdout[-_CODE_EXEC_STDOUT_TAIL:]
    if stderr:
        envelope["stderr_tail"] = stderr[-_CODE_EXEC_STDERR_TAIL:]
    if raw_payload.get("staged_uploads"):
        envelope["staged_uploads"] = raw_payload["staged_uploads"]
    if errors:
        envelope["errors"] = errors[:_CODE_EXEC_MAX_ERRORS_INLINE]
        envelope["error_count"] = len(errors)
    if pending_error:
        envelope["ops_drain_error"] = pending_error

    return applied_summary, envelope


# --- Candidates tools (file-backed staging from source fetches) ---


def _tool_candidates_list(
    project: Project, args: Dict[str, Any]
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    try:
        files = candidates.list_candidate_files(project.id)
    except Exception as e:
        return {}, {"error": f"{type(e).__name__}: {e}"}
    return {}, {"files": files, "total": len(files)}


def _tool_candidates_inspect(
    project: Project, args: Dict[str, Any]
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    file_name = args.get("file")
    if not file_name or not isinstance(file_name, str):
        return {}, {"error": "file (string) is required"}
    filt = args.get("filter") or {}
    fields = args.get("fields") or None
    limit = min(int(args.get("limit", 20) or 20), 200)
    offset = max(int(args.get("offset", 0) or 0), 0)

    # `exec_*.jsonl` files are code_exec transcripts under exec_logs/,
    # not source-tool candidates. Same inspect surface, different prefix.
    stream_fn = (
        candidates.stream_exec_log
        if candidates.is_exec_log_filename(file_name)
        else candidates.stream_candidates
    )

    matched = 0
    skipped = 0
    out: List[Dict[str, Any]] = []
    try:
        for item in stream_fn(project.id, file_name):
            if not candidates.apply_filter(item, filt):
                continue
            matched += 1
            if matched <= offset:
                skipped += 1
                continue
            if len(out) < limit:
                out.append(candidates.project_fields(item, fields))
            elif matched > offset + limit and not filt:
                # Cheap exit when no filter and we already have our slice;
                # we still need full count for `matched` though, so only
                # break if caller didn't ask for that. Always count full.
                pass
    except FileNotFoundError as e:
        return {}, {"error": str(e)}
    except Exception as e:
        return {}, {"error": f"{type(e).__name__}: {e}"}

    return (
        {},
        {
            "matched": matched,
            "returned": len(out),
            "offset": offset,
            "items": out,
        },
    )


_TOOL_FROM_FILE_RE = re.compile(r"^(.+)_[0-9a-f]{8}\.jsonl$")


async def _tool_candidates_to_rows(
    db: Session,
    project: Project,
    version: ProjectVersion,
    args: Dict[str, Any],
    progress_cb: Optional[fill.ProgressCallback] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    file_name = args.get("file")
    column_map = args.get("column_map")
    if not file_name or not isinstance(file_name, str):
        return {}, {"error": "file (string) is required"}
    if not isinstance(column_map, dict) or not column_map:
        return {}, {"error": "column_map must be a non-empty {source_field: column_name} dict"}
    filt = args.get("filter") or {}
    merge_key = args.get("merge_key")
    confirm = bool(args.get("confirm", False))

    if not confirm:
        # Preview: scan the file, count items that would insert vs merge.
        # Surfaces the merge-collapse problem (multiple founders per
        # company silently merging into one row) BEFORE it happens.
        from sqlalchemy import text as _text
        existing_keys: set = set()
        deleted_keys: set = set()
        if merge_key:
            existing = db.execute(
                _text(
                    f"SELECT {_row_value_expr(merge_key)}, deleted_at FROM samples "
                    f"WHERE version_id = :vid"
                ),
                {"vid": version.id},
            ).all()
            for r in existing:
                if r[0] is None:
                    continue
                if r[1] is None:
                    existing_keys.add(str(r[0]))
                else:
                    deleted_keys.add(str(r[0]))

        scanned = 0
        skipped_filter = 0
        would_insert = 0
        would_merge_existing = 0
        would_skip_already_deleted = 0
        intra_batch_collisions = 0  # candidates that merge with EARLIER candidates in this same file
        seen_keys: set = set()
        sample_inserts: List[Dict[str, Any]] = []
        sample_collisions: List[Dict[str, Any]] = []

        try:
            for item in candidates.stream_candidates(project.id, file_name):
                scanned += 1
                if not candidates.apply_filter(item, filt):
                    skipped_filter += 1
                    continue
                mapped = {target: item.get(source) for source, target in column_map.items()}
                if all(v is None for v in mapped.values()):
                    skipped_filter += 1
                    continue
                if merge_key:
                    mv = mapped.get(merge_key)
                    if mv is not None:
                        mv_s = str(mv)
                        if mv_s in deleted_keys:
                            would_skip_already_deleted += 1
                            continue
                        if mv_s in existing_keys:
                            would_merge_existing += 1
                            if len(sample_collisions) < 3:
                                sample_collisions.append({"merge_key_value": mv, "with": "existing row"})
                            continue
                        if mv_s in seen_keys:
                            intra_batch_collisions += 1
                            if len(sample_collisions) < 3:
                                sample_collisions.append({"merge_key_value": mv, "with": "earlier candidate"})
                            continue
                        seen_keys.add(mv_s)
                would_insert += 1
                if len(sample_inserts) < 3:
                    sample_inserts.append(mapped)
        except FileNotFoundError as e:
            return {}, {"error": str(e)}

        warning = None
        total_collisions = would_merge_existing + intra_batch_collisions
        if merge_key and total_collisions > 0 and (would_insert + total_collisions) > 0:
            collision_rate = total_collisions / (would_insert + total_collisions)
            if collision_rate >= 0.30:
                warning = (
                    f"merge_key={merge_key!r} causes {total_collisions} of "
                    f"{would_insert + total_collisions} candidates to collapse "
                    f"({collision_rate:.0%}). That's likely a non-unique key "
                    f"(e.g. company when there are multiple founders per "
                    f"company). Consider dropping merge_key or picking a "
                    f"genuinely unique field."
                )

        return {}, {
            "preview": True,
            "scanned": scanned,
            "skipped_filter": skipped_filter,
            "would_insert": would_insert,
            "would_merge_with_existing": would_merge_existing,
            "would_skip_already_deleted": would_skip_already_deleted,
            "intra_batch_collisions": intra_batch_collisions,
            "sample_inserts": sample_inserts,
            "sample_collisions": sample_collisions,
            "warning": warning,
            "hint": (
                f"Would land {would_insert} new rows. Re-call with "
                f"confirm=true to commit."
                + (
                    f" {would_skip_already_deleted} candidate(s) match "
                    f"previously-deleted rows and will be skipped (the "
                    f"user already said no to those)."
                    if would_skip_already_deleted else ""
                )
            ),
        }

    # File names are written as "{tool}_{8-hex}.jsonl" by candidates.write_candidates.
    # The tool name drives default source-citation attachment below.
    m = _TOOL_FROM_FILE_RE.match(file_name)
    file_tool = m.group(1) if m else ""

    BATCH = 100
    total_inserted = 0
    total_merged = 0
    total_skipped_filter = 0
    total_skipped_already_deleted = 0

    batch: List[Dict[str, Any]] = []

    async def flush() -> None:
        nonlocal total_inserted, total_merged, total_skipped_already_deleted
        if not batch:
            return
        applied_batch, result_batch = await _tool_rows_add(
            db,
            project,
            version,
            {"items": list(batch), "merge_key": merge_key} if merge_key else {"items": list(batch)},
            progress_cb=progress_cb,
        )
        if isinstance(result_batch, dict) and result_batch.get("ok"):
            total_inserted += int(result_batch.get("inserted", 0) or 0)
            total_merged += int(result_batch.get("merged", 0) or 0)
            total_skipped_already_deleted += int(result_batch.get("skipped_already_deleted", 0) or 0)
        batch.clear()

    try:
        for item in candidates.stream_candidates(project.id, file_name):
            if not candidates.apply_filter(item, filt):
                total_skipped_filter += 1
                continue
            mapped = {
                target: item.get(source)
                for source, target in column_map.items()
            }
            # Drop rows where every mapped value is None — usually means
            # the source field names don't match anything in this file.
            if all(v is None for v in mapped.values()):
                total_skipped_filter += 1
                continue
            # Without this, candidates_to_rows-committed rows have empty
            # tags and the UI's per-cell source dropdown is blank. Same
            # source applies to every mapped column — they all came from
            # the same candidate item.
            src = sources.derive_default_source(file_tool, item)
            if src:
                mapped["_sources"] = {col: [src] for col in mapped}
            batch.append(mapped)
            if len(batch) >= BATCH:
                await flush()
        await flush()
    except FileNotFoundError as e:
        return {}, {"error": str(e)}
    except Exception as e:
        log.exception("candidates_to_rows failed")
        return {}, {"error": f"{type(e).__name__}: {e}"}

    total_rows = (
        db.query(func.count(Sample.id))
        .filter(Sample.version_id == version.id, Sample.deleted_at.is_(None))
        .scalar() or 0
    )
    result = {
        "ok": True,
        "inserted": total_inserted,
        "merged": total_merged,
        "skipped": total_skipped_filter,
        "total": total_rows,
    }
    if total_skipped_already_deleted:
        result["skipped_already_deleted"] = total_skipped_already_deleted
        result["note"] = (
            f"{total_skipped_already_deleted} candidate(s) matched a "
            f"previously-deleted row's merge_key and were skipped (the "
            f"user already said no to those). Use rows_undelete if intended."
        )
    return (
        {"rows": {"inserted": total_inserted, "merged": total_merged}},
        result,
    )


def _tool_confirm_budget(args: Dict[str, Any]):
    """End the turn with a budget-approval chip block.

    Like suggest_replies but the FE renders these chips with cost-aware
    copy and a header showing the estimated spend. Each option may carry
    a `cap_override_cents` value — when the user clicks it, the next
    turn starts with that cap instead of the recomputed tier-default.

    Server-side validation:
      - At least one option must have a cap_override_cents > 0 (an
        approve path), otherwise the user has no way to authorize
        spending and the chip block is just a confusing dead end.
      - cap_override_cents values are bounded server-side at the
        approval ceiling — see budget._approval_ceiling_cents on the
        next-turn entry path. We don't ceiling here because the
        agent's stated value is the user-facing intent; clamping
        happens when the next run actually starts.
    """
    from dsl_worker.chat_api import budget as _budget

    summary = (args.get("summary") or "").strip()
    raw_options = args.get("options") or []
    reason = (args.get("reason") or "scope_ambiguous").strip()
    if not summary:
        return {}, {"error": "summary is required"}
    if reason not in ("scope_ambiguous", "projection_exceeds_cap"):
        reason = "scope_ambiguous"

    options: List[Dict[str, Any]] = []
    has_approve = False
    for opt in raw_options:
        if not isinstance(opt, dict):
            continue
        label = (opt.get("label") or "").strip()
        message = (opt.get("message") or "").strip()
        if not label or not message:
            continue
        clean: Dict[str, Any] = {
            "label": label[:80],
            "message": message[:500],
        }
        cap_override = opt.get("cap_override_cents")
        if isinstance(cap_override, (int, float)) and cap_override > 0:
            clean["cap_override_cents"] = int(cap_override)
            has_approve = True
        options.append(clean)

    if len(options) < 2:
        return {}, {"error": "confirm_budget requires at least 2 options"}
    if not has_approve:
        return {}, {
            "error": (
                "confirm_budget requires at least one option with "
                "cap_override_cents > 0 — otherwise the user can't "
                "approve more spending"
            )
        }

    est_cents = args.get("estimated_cost_cents")
    if isinstance(est_cents, (int, float)) and est_cents > 0:
        projection_cents: Optional[int] = int(est_cents)
    else:
        projection_cents = None

    payload = _budget.build_budget_check_payload(
        summary=summary[:500],
        # spent_cents is filled in at emit-time from the running
        # BillingMeter total — the agent doesn't know its own spend.
        spent_cents=0,
        # cap_cents is filled in at emit-time from the BillingMeter's
        # configured cap. Same reasoning.
        cap_cents=0,
        options=options,
        projection_cents=projection_cents,
        reason=reason,
    )
    return (
        {"budget_check": payload},
        {"ok": True, "options": len(options)},
    )


def _tool_suggest_replies(args: Dict[str, Any]):
    """Attach text reply suggestions to the assistant's most recent message.

    No DB side effects — the suggestions are forwarded to the SSE stream
    and persisted in applied_changes so they survive a history reload.
    """
    raw = args.get("suggestions") or []
    items: List[Dict[str, str]] = []
    for s in raw:
        if not isinstance(s, dict):
            continue
        label = (s.get("label") or "").strip()
        message = (s.get("message") or "").strip()
        if not label or not message:
            continue
        items.append({"label": label[:80], "message": message[:500]})
    if not items:
        return {}, {"error": "no valid suggestions"}
    return (
        {"suggestions": {"items": items}},
        {"ok": True, "count": len(items)},
    )


def _tool_version_label(
    db: Session, project: Project, args: Dict[str, Any]
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Set a label on the current ProjectVersion.

    Per-turn versioning: streaming.py forks a new version before the
    agent runs, so project.current_version_id already points to the
    new version. We just stamp it with the label.
    """
    label = (args.get("label") or "").strip()
    if not label:
        return {}, {"error": "label is required"}
    label = label[:120]
    if project.current_version_id is None:
        return {}, {"error": "no current version to label"}
    version = db.query(ProjectVersion).filter(
        ProjectVersion.id == project.current_version_id
    ).first()
    if version is None:
        return {}, {"error": "current version not found"}
    version.label = label
    db.flush()
    return (
        {"version_label": {
            "version_id": str(version.id),
            "version_number": version.version_number,
            "label": label,
        }},
        {"ok": True, "label": label},
    )


async def _tool_rows_fill(
    db: Session,
    project: Project,
    version: ProjectVersion,
    args: Dict[str, Any],
    progress_cb: Optional[fill.ProgressCallback] = None,
    effort: Optional[str] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any], float]:
    columns = args.get("columns") or []
    if not isinstance(columns, list) or not columns:
        return {}, {"error": "columns must be a non-empty list of column names"}, 0.0

    where = args.get("where") or {}
    limit = args.get("limit", 20)
    if limit is not None:
        limit = min(int(limit), 200)

    # Per-cell budget. The agent can no longer pass max_cost — it was
    # dropped from the schema after observing the agent set it tight
    # ($0.01) on cea954b4 and burning ~5 credits across 20 cells for
    # 1 successful value. The system picks the per-cell cap from the
    # effort tier; agent-driven overrides only happen via confirm_budget
    # (which triggers a user-approved larger turn cap, not a per-cell
    # squeeze). If callers somehow still pass max_cost, treat it as
    # advisory but never go BELOW the tier default — too-tight caps
    # cause cells to fail without producing values.
    tier_default = fill.tier_default_max_cost(effort)
    if "max_cost" in args and args["max_cost"] is not None:
        try:
            user_max_cost = float(args["max_cost"])
            max_cost = max(user_max_cost, tier_default)
        except (TypeError, ValueError):
            max_cost = tier_default
    else:
        max_cost = tier_default

    where_sql, where_params = _where_to_sql(where)

    # Commit the main session before fill_rows opens its own SessionLocal.
    # Otherwise rows just inserted by rows_add / candidates_to_rows in the
    # same turn aren't visible to the new session and the query returns
    # matched_rows=0 even though the rows exist (in the uncommitted main
    # transaction). Targeted commit here only — committing per-tool in
    # streaming.py caused event-loop stalls.
    db.commit()

    summary, total_cost = await fill.fill_rows(
        project=project,
        target_columns=columns,
        where_sql=where_sql,
        where_params=where_params,
        limit=limit,
        max_cost=max_cost,
        progress_cb=progress_cb,
    )
    applied = {"rows_filled": summary.get("cells_filled", 0)}
    return applied, summary, total_cost


# --- Column tools ---


def _tool_columns_add(db: Session, project: Project, args: Dict[str, Any]):
    name = (args.get("name") or "").strip()
    if not name:
        return {}, {"error": "name is required"}
    cols = list(project.columns or [])
    existing = _column_index(project)
    if name in existing:
        return {}, {"error": f"column {name!r} already exists"}
    new_col = {"name": name}
    if args.get("format"):
        new_col["format"] = args["format"]
    if args.get("description"):
        new_col["description"] = args["description"]
    cols.append(new_col)
    project.columns = cols
    if project.current_version is not None:
        project.current_version.columns = cols
    return {"columns": cols}, {"ok": True, "column": new_col, "total_columns": len(cols)}


def _tool_columns_list(db: Session, project: Project, args: Dict[str, Any]):
    return {}, {"columns": list(project.columns or [])}


def _tool_columns_modify(db: Session, project: Project, args: Dict[str, Any]):
    name = (args.get("name") or "").strip()
    if not name:
        return {}, {"error": "name is required"}
    cols = list(project.columns or [])
    target = None
    for c in cols:
        if isinstance(c, dict) and c.get("name") == name:
            target = c
            break
    if target is None:
        return {}, {"error": f"column {name!r} not found"}

    new_name = args.get("new_name")
    rename_field: Optional[Tuple[str, str]] = None
    if new_name and new_name != name:
        if any(isinstance(c, dict) and c.get("name") == new_name for c in cols):
            return {}, {"error": f"column {new_name!r} already exists"}
        rename_field = (name, new_name)
        target["name"] = new_name
    if "format" in args:
        target["format"] = args["format"]
    if "description" in args:
        target["description"] = args["description"]

    project.columns = cols
    if project.current_version is not None:
        project.current_version.columns = cols

    # If renamed, also rewrite samples.row keys
    if rename_field:
        old, new = rename_field
        version = project.current_version
        if version is not None:
            for sample in db.query(Sample).filter(Sample.version_id == version.id).all():
                data = dict(sample.row or {})
                if old in data:
                    data[new] = data.pop(old)
                    sample.row = data

    return {"columns": cols}, {"ok": True, "column": target}


def _tool_columns_delete(db: Session, project: Project, args: Dict[str, Any]):
    name = (args.get("name") or "").strip()
    confirm = bool(args.get("confirm", False))
    if not name:
        return {}, {"error": "name is required"}
    cols = list(project.columns or [])
    new_cols = [c for c in cols if not (isinstance(c, dict) and c.get("name") == name)]
    if len(new_cols) == len(cols):
        return {}, {"error": f"column {name!r} not found"}

    version = project.current_version

    if not confirm:
        # Preview — count cells with data, show a few sample values
        affected = 0
        sample_values = []
        if version is not None:
            for sample in (
                db.query(Sample)
                .filter(Sample.version_id == version.id)
                .all()
            ):
                data = sample.row or {}
                if name in data and data[name] not in (None, ""):
                    affected += 1
                    if len(sample_values) < 3:
                        sample_values.append(data[name])
        return {}, {
            "preview": True,
            "column": name,
            "would_drop_cells": affected,
            "sample_values": sample_values,
            "hint": (
                f"This would remove column {name!r} and drop {affected} cell "
                f"value(s). Re-call with confirm=true to delete."
            ),
        }

    project.columns = new_cols
    if version is not None:
        version.columns = new_cols
        # Strip the field from every row in this version
        affected = 0
        for sample in db.query(Sample).filter(Sample.version_id == version.id).all():
            data = dict(sample.row or {})
            if name in data:
                data.pop(name)
                sample.row = data
                affected += 1
    else:
        affected = 0
    return {"columns": new_cols}, {"ok": True, "rows_with_data_dropped": affected}


# --- Row tools ---


async def _tool_rows_add(
    db: Session,
    project: Project,
    version: ProjectVersion,
    args: Dict[str, Any],
    progress_cb: Optional[fill.ProgressCallback] = None,
):
    items = args.get("items")
    if not isinstance(items, list):
        return {}, {"error": "items must be a list of objects"}
    merge_key = args.get("merge_key")

    inserted = 0
    merged = 0
    skipped_already_deleted = 0
    for item in items:
        if not isinstance(item, dict):
            continue
        # Pull the optional per-cell citations dict out of the item before
        # we treat the rest as row data. Shape:
        #   _sources = {ColumnName: [{type, value}, ...], ...}
        item_sources = None
        if isinstance(item.get("_sources"), dict):
            item_sources = {
                k: v for k, v in item["_sources"].items()
                if isinstance(v, list) and v
            }
        item = {k: v for k, v in item.items() if k != "_sources"}
        merged_existing = False
        if merge_key:
            mv = item.get(merge_key)
            if mv is not None:
                from sqlalchemy import text
                # Look up across ALL rows (including soft-deleted). If the
                # user previously deleted a row with this merge_key value
                # they've already said "no" to it — don't reinsert. Skipping
                # is safer than resurrecting; user can rows_undelete if
                # they actually want it back.
                stmt = text(
                    f"SELECT id, row, deleted_at FROM samples WHERE version_id = :vid "
                    f"AND ({_row_value_expr(merge_key)}) = :mv LIMIT 1"
                )
                existing = db.execute(
                    stmt, {"vid": version.id, "mv": str(mv)}
                ).first()
                if existing and existing[2] is not None:
                    skipped_already_deleted += 1
                    continue
                if existing:
                    sample = db.query(Sample).filter(Sample.id == existing[0]).first()
                    if sample is not None:
                        data = dict(sample.row or {})
                        for k, v in item.items():
                            if v is not None and (data.get(k) is None or data.get(k) == ""):
                                data[k] = v
                        sample.row = data
                        # Merge per-cell sources: existing tags win, new
                        # sources fill in any cells that didn't have any.
                        if item_sources:
                            existing_tags = dict(sample.tags or {})
                            existing_cell_sources = dict(existing_tags.get("sources") or {})
                            for col_name, srcs in item_sources.items():
                                if col_name not in existing_cell_sources:
                                    existing_cell_sources[col_name] = srcs
                            existing_tags["sources"] = existing_cell_sources
                            sample.tags = existing_tags
                        merged += 1
                        merged_existing = True
                        if progress_cb is not None:
                            try:
                                await progress_cb({
                                    "type": "row_merged",
                                    "row": _row_to_dict(sample),
                                })
                            except Exception:
                                log.exception("row_merged progress_cb raised")
        if merged_existing:
            continue

        seq = _next_seq(db, version.id)
        sample = Sample(
            project_id=project.id,
            version_id=version.id,
            seq=seq,
            row=dict(item),
            tags={"sources": item_sources} if item_sources else {},
        )
        db.add(sample)
        db.flush()
        version.generated_count = (version.generated_count or 0) + 1
        inserted += 1
        if progress_cb is not None:
            try:
                await progress_cb({
                    "type": "row_added",
                    "row": _row_to_dict(sample),
                })
            except Exception:
                log.exception("row_added progress_cb raised")

    total = (
        db.query(func.count(Sample.id))
        .filter(Sample.version_id == version.id, Sample.deleted_at.is_(None))
        .scalar() or 0
    )
    result = {"ok": True, "inserted": inserted, "merged": merged, "total": total}
    if skipped_already_deleted:
        result["skipped_already_deleted"] = skipped_already_deleted
        result["note"] = (
            f"{skipped_already_deleted} item(s) had a merge_key matching "
            f"a previously-deleted row and were skipped. Call "
            f"rows_undelete with the matching where to bring those back "
            f"if intended."
        )
    return (
        {"rows": {"inserted": inserted, "merged": merged}},
        result,
    )


def _tool_rows_count(
    db: Session, project: Project, version: ProjectVersion, args: Dict[str, Any]
):
    where = args.get("where") or {}
    sql, params = _where_to_sql(where)
    from sqlalchemy import text
    stmt = text(
        f"SELECT COUNT(*) FROM samples WHERE version_id = :vid AND deleted_at IS NULL AND ({sql})"
    )
    n = db.execute(stmt, {"vid": version.id, **params}).scalar() or 0
    return {}, {"count": int(n)}


def _tool_rows_get(
    db: Session, project: Project, version: ProjectVersion, args: Dict[str, Any]
):
    where = args.get("where") or {}
    limit = args.get("limit")
    columns = args.get("columns")
    sql, params = _where_to_sql(where)
    from sqlalchemy import text
    base = (
        f"SELECT id, seq, row FROM samples WHERE version_id = :vid AND deleted_at IS NULL AND ({sql}) "
        f"ORDER BY seq"
    )
    if limit is not None:
        base += f" LIMIT {int(limit)}"
    rows = db.execute(text(base), {"vid": version.id, **params}).all()

    out: List[Dict[str, Any]] = []
    for r in rows:
        rid, seq, data = r
        d = {"_id": str(rid), "_seq": seq}
        if isinstance(data, dict):
            d.update(data)
        if columns is not None:
            d = {"_id": d["_id"], "_seq": d["_seq"], **{k: d.get(k) for k in columns}}
        out.append(d)
    return {}, {"rows": out, "count": len(out)}


def _tool_rows_sample(
    db: Session, project: Project, version: ProjectVersion, args: Dict[str, Any]
):
    n = int(args.get("n") or 3)
    from sqlalchemy import text
    rows = db.execute(
        text("SELECT id, seq, row FROM samples WHERE version_id = :vid AND deleted_at IS NULL ORDER BY RANDOM() LIMIT :n"),
        {"vid": version.id, "n": n},
    ).all()
    out: List[Dict[str, Any]] = []
    for r in rows:
        rid, seq, data = r
        d = {"_id": str(rid), "_seq": seq}
        if isinstance(data, dict):
            d.update(data)
        out.append(d)
    return {}, {"rows": out}


def _tool_rows_update(
    db: Session, project: Project, version: ProjectVersion, args: Dict[str, Any]
):
    where = args.get("where") or {}
    values = args.get("values") or {}
    confirm = bool(args.get("confirm", False))
    if not isinstance(values, dict) or not values:
        return {}, {"error": "values must be a non-empty object"}
    sql, params = _where_to_sql(where)
    from sqlalchemy import text

    if not confirm:
        cnt = db.execute(
            text(f"SELECT COUNT(*) FROM samples WHERE version_id = :vid AND deleted_at IS NULL AND ({sql})"),
            {"vid": version.id, **params},
        ).scalar() or 0
        sample_rows = db.execute(
            text(
                f"SELECT id, seq, row FROM samples WHERE version_id = :vid "
                f"AND deleted_at IS NULL AND ({sql}) ORDER BY seq LIMIT 3"
            ),
            {"vid": version.id, **params},
        ).all()
        return {}, {
            "preview": True,
            "would_update": int(cnt),
            "values_to_set": values,
            "sample_before": [
                {"_id": str(r[0]), "_seq": r[1], **(r[2] or {})}
                for r in sample_rows
            ],
            "hint": (
                f"This would update {int(cnt)} row(s) with the given values. "
                f"Re-call with confirm=true to apply, or refine `where` first."
            ),
        }

    rows = db.execute(
        text(f"SELECT id FROM samples WHERE version_id = :vid AND deleted_at IS NULL AND ({sql})"),
        {"vid": version.id, **params},
    ).all()
    affected = 0
    for (rid,) in rows:
        sample = db.query(Sample).filter(Sample.id == rid).first()
        if sample is None:
            continue
        data = dict(sample.row or {})
        data.update(values)
        sample.row = data
        affected += 1
    return {"rows_updated": affected}, {"ok": True, "affected": affected}


def _tool_rows_delete(
    db: Session, project: Project, version: ProjectVersion, args: Dict[str, Any]
):
    where = args.get("where") or {}
    confirm = bool(args.get("confirm", False))
    sql, params = _where_to_sql(where)
    from sqlalchemy import text

    if not confirm:
        # Preview pass — show count + sample so the agent can sanity-check
        # before destroying anything.
        cnt = db.execute(
            text(f"SELECT COUNT(*) FROM samples WHERE version_id = :vid AND deleted_at IS NULL AND ({sql})"),
            {"vid": version.id, **params},
        ).scalar() or 0
        sample_rows = db.execute(
            text(
                f"SELECT id, seq, row FROM samples WHERE version_id = :vid "
                f"AND deleted_at IS NULL AND ({sql}) ORDER BY seq LIMIT 3"
            ),
            {"vid": version.id, **params},
        ).all()
        return {}, {
            "preview": True,
            "would_delete": int(cnt),
            "sample": [
                {"_id": str(r[0]), "_seq": r[1], **(r[2] or {})}
                for r in sample_rows
            ],
            "hint": (
                f"This would delete {int(cnt)} row(s). Re-call with "
                f"confirm=true to actually delete, or refine `where` first."
            ),
        }

    # Confirmed — soft delete via UPDATE deleted_at = now(). Recoverable
    # via rows_undelete. Immediately visible to subsequent active-row
    # queries because they all filter `deleted_at IS NULL`.
    result = db.execute(
        text(
            f"UPDATE samples SET deleted_at = NOW() "
            f"WHERE version_id = :vid AND deleted_at IS NULL AND ({sql})"
        ),
        {"vid": version.id, **params},
    )
    deleted = int(result.rowcount or 0)
    db.expire_all()
    if deleted and version.generated_count is not None:
        version.generated_count = max(0, version.generated_count - deleted)
    return {"rows_deleted": deleted}, {
        "ok": True,
        "deleted": deleted,
        "soft_deleted": True,
        "hint": "Use rows_undelete with a matching `where` to restore.",
    }


def _tool_rows_undelete(
    db: Session, project: Project, version: ProjectVersion, args: Dict[str, Any]
):
    """Restore soft-deleted rows. Same `where` dialect as rows_delete but
    operates on the deleted set (deleted_at IS NOT NULL)."""
    where = args.get("where") or {}
    confirm = bool(args.get("confirm", False))
    sql, params = _where_to_sql(where)
    from sqlalchemy import text

    if not confirm:
        cnt = db.execute(
            text(
                f"SELECT COUNT(*) FROM samples WHERE version_id = :vid "
                f"AND deleted_at IS NOT NULL AND ({sql})"
            ),
            {"vid": version.id, **params},
        ).scalar() or 0
        sample_rows = db.execute(
            text(
                f"SELECT id, seq, row FROM samples WHERE version_id = :vid "
                f"AND deleted_at IS NOT NULL AND ({sql}) ORDER BY seq LIMIT 3"
            ),
            {"vid": version.id, **params},
        ).all()
        return {}, {
            "preview": True,
            "would_restore": int(cnt),
            "sample": [
                {"_id": str(r[0]), "_seq": r[1], **(r[2] or {})}
                for r in sample_rows
            ],
            "hint": (
                f"This would restore {int(cnt)} soft-deleted row(s). "
                f"Re-call with confirm=true to apply."
            ),
        }

    result = db.execute(
        text(
            f"UPDATE samples SET deleted_at = NULL "
            f"WHERE version_id = :vid AND deleted_at IS NOT NULL AND ({sql})"
        ),
        {"vid": version.id, **params},
    )
    restored = int(result.rowcount or 0)
    db.expire_all()
    if restored and version.generated_count is not None:
        version.generated_count = (version.generated_count or 0) + restored
    return {"rows_restored": restored}, {"ok": True, "restored": restored}


# ---------------------------------------------------------------------------
# Result formatting + applied-change description
# ---------------------------------------------------------------------------


def format_tool_result(tool_name: str, result: Dict[str, Any]) -> str:
    """Stringify the tool result for the LLM input feed."""
    return json.dumps(result, default=str)[:4000]


def project_row_count(db: Session, project: Project) -> int:
    """Active (non-soft-deleted) row count for the current version. Cheap
    indexed COUNT used to drive live row-count events to the FE so the
    pagination total tracks rows_delete / rows_add as they happen."""
    if not project.current_version_id:
        return 0
    try:
        db.flush()
    except Exception:
        pass
    return int(
        db.query(func.count(Sample.id))
        .filter(
            Sample.version_id == project.current_version_id,
            Sample.deleted_at.is_(None),
        )
        .scalar()
        or 0
    )


def project_state_hint(db: Session, project: Project) -> str:
    """A one-line state line appended after every tool result so the agent
    constantly sees what the user actually has on screen. Includes column
    names (first 8) and active-row count, plus soft-deleted count if any.

    Flushes pending session writes first so the count reflects this turn's
    work — without flush, ORM-style mutations leave the count stale.
    """
    try:
        try:
            db.flush()
        except Exception:
            pass
        cols = [
            c.get("name", "?")
            for c in (project.columns or [])
            if isinstance(c, dict)
        ]
        n_rows = 0
        n_deleted = 0
        if project.current_version_id:
            n_rows = db.query(func.count(Sample.id)).filter(
                Sample.version_id == project.current_version_id,
                Sample.deleted_at.is_(None),
            ).scalar() or 0
            n_deleted = db.query(func.count(Sample.id)).filter(
                Sample.version_id == project.current_version_id,
                Sample.deleted_at.isnot(None),
            ).scalar() or 0
        col_preview = ", ".join(cols[:8])
        if len(cols) > 8:
            col_preview += f" (+{len(cols) - 8} more)"
        cols_part = f"({col_preview})" if col_preview else "(none yet)"
        deleted_part = f", {n_deleted} soft-deleted" if n_deleted else ""
        return (
            f"\n\n[Project state: {len(cols)} columns {cols_part}, "
            f"{n_rows} rows{deleted_part}]"
        )
    except Exception:
        return ""


def describe_applied(applied: Dict[str, Any]) -> List[AppliedChange]:
    """Convert the merged 'applied' map across this turn's tool calls into
    AppliedChange records the frontend can render.
    """
    out: List[AppliedChange] = []
    if "columns" in applied:
        cols = applied["columns"]
        names = [c.get("name", "?") for c in cols if isinstance(c, dict)]
        preview = ", ".join(names[:5])
        if len(names) > 5:
            preview += f" (+{len(names) - 5} more)"
        out.append(AppliedChange(
            field="columns",
            description=f"Schema now {len(names)} column(s): {preview}",
            # Send the full columns array so the frontend can patch its
            # local project.columns state without a DB roundtrip. The
            # value is the same shape ProjectOut.columns has (list of
            # {name, format?, description?}).
            value=cols,
        ))
    if "rows" in applied:
        ins = applied["rows"].get("inserted", 0)
        mer = applied["rows"].get("merged", 0)
        bits: List[str] = []
        if ins:
            bits.append(f"added {ins}")
        if mer:
            bits.append(f"merged {mer}")
        out.append(AppliedChange(
            field="rows",
            description="Rows: " + ", ".join(bits) if bits else "Rows updated",
        ))
    if "rows_updated" in applied:
        out.append(AppliedChange(
            field="rows",
            description=f"Updated {applied['rows_updated']} row(s)",
        ))
    if "rows_deleted" in applied:
        out.append(AppliedChange(
            field="rows",
            description=f"Deleted {applied['rows_deleted']} row(s)",
        ))
    if "rows_filled" in applied:
        out.append(AppliedChange(
            field="rows",
            description=f"Filled {applied['rows_filled']} cell(s)",
        ))
    return out


def build_context_message(db: Session, project: Project) -> str:
    """Compact state summary handed to the LLM as a system context message
    every turn."""
    parts = [
        f"Today: {date.today().isoformat()}",
        f"Project: {project.name}",
    ]
    if project.num_samples:
        parts.append(
            f"Goal: ~{project.num_samples} rows (soft target — the user "
            "set this when they created the project; use it as a guide for "
            "how much to pull, not a hard cap)."
        )

    # Columns
    cols = project.columns or []
    if cols:
        parts.append("Columns:")
        for c in cols:
            if not isinstance(c, dict):
                continue
            line = f"  - {c.get('name', '?')}"
            if c.get("format"):
                line += f" ({c['format']})"
            if c.get("description"):
                line += f" — {c['description']}"
            parts.append(line)
    else:
        parts.append("Columns: (none yet)")

    # Uploaded files. Auto-staged into the sandbox at
    # /workspace/uploads/<filename> on every code_exec call. Bulk-import
    # path: code_exec a snippet that reads the upload and calls
    # `dsl_tools.add_rows(items)` (or `add_candidates(items)`) — the
    # rows are committed server-side after exec; data never round-trips
    # through the LLM.
    uploaded_files = (
        db.query(ProjectFile)
        .filter(
            ProjectFile.project_id == project.id,
            ProjectFile.deleted_at.is_(None),
            ProjectFile.status == "uploaded",
        )
        .order_by(ProjectFile.uploaded_at.asc().nullslast(), ProjectFile.created_at.asc())
        .all()
    )
    if uploaded_files:
        parts.append("Uploaded files at /workspace/uploads/ (read via code_exec):")
        for f in uploaded_files:
            size_b = f.size_bytes or 0
            if size_b < 1024:
                size_str = f"{size_b}B"
            elif size_b < 1024 * 1024:
                size_str = f"{size_b / 1024:.1f}KB"
            else:
                size_str = f"{size_b / (1024 * 1024):.1f}MB"
            parts.append(
                f"  - {f.filename} ({f.content_type or 'unknown'}, {size_str})"
            )

    # Row count + tiny sample (only if we have a version)
    if project.current_version_id:
        version_id = project.current_version_id
        n = db.query(func.count(Sample.id)).filter(Sample.version_id == version_id).scalar() or 0
        parts.append(f"Rows: {n}")
        if n > 0:
            samples = (
                db.query(Sample)
                .filter(Sample.version_id == version_id)
                .order_by(Sample.seq)
                .limit(3)
                .all()
            )
            parts.append("Sample (first 3):")
            for s in samples:
                row_repr = json.dumps(s.row, default=str)
                if len(row_repr) > 200:
                    row_repr = row_repr[:200] + "..."
                parts.append(f"  - seq={s.seq}: {row_repr}")
    else:
        parts.append("Rows: 0 (no version yet)")

    return "\n".join(parts)
