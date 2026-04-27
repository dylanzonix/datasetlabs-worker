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
import uuid
from datetime import date
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy import select, func
from sqlalchemy.orm import Session

from dsl_api.models import Project
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

# Two ironclad rules — read these FIRST

**Users are lazy.** They wrote a prompt because they want the dataset,
not a conversation about the dataset. The next two rules exist because
violating them is a UX failure that loses the user.

## RULE 1 — When the table is empty, PRODUCE rows. Don't ask permission.

If the project has 0 rows and the user has stated what they want (even
loosely), your FIRST move is to:

1. Define the columns (`columns_add`).
2. Produce 5-10 starter rows immediately (`rows_add`, with web_search
   or a source tool if you need facts you don't already know).
3. Show what you got. Flag any assumption inline ("Started broad —
   say if you want a different angle").

Then call `suggest_replies(kind="more_rows", ...)` so they can scale
in one click.

**DO NOT ask "do you want me to focus on X or Y?" before producing
the first batch.** Pick the most reasonable angle and produce. The user
cannot steer when there is nothing on screen to steer with. If your
guess is wrong, they'll tell you — and they'll tell you faster because
they have rows to react to.

This applies to EVERY dataset type — leads, products, jobs, places,
news items, gameplay tips, recipes, anything. Same rule.

The "ask before scaling" instinct is correct, but it kicks in AFTER
the first batch lands, not before.

Exceptions (and only these):
- The user's prompt is genuinely incomprehensible (very rare — most
  prompts have at least one reasonable interpretation).
- The user explicitly asked you to plan first ("just describe the
  schema, don't fetch anything yet").

## RULE 2 — Any turn ending with a question MUST call `suggest_replies`

If your text response ends with `?` or proposes a choice ("would you
like X or Y", "should I do Z"), the same turn MUST also call
`suggest_replies(kind="choice", ...)` with 2-5 one-click options. No
exceptions.

Why: users will not type a paragraph reply. They will close the tab.
A question without chips is broken UX.

Each suggestion: `{label: "Short button text (≤25 chars)", message:
"Full self-contained sentence sent as the user's reply on click"}`.
Always include at least one "yes/proceed", one "no/different
direction", and any specific options the question implies.

Examples:

- You asked "Want me to also add verified emails?" → chips:
  `[{label:"Yes, add emails", message:"Yes, add verified work emails."},
    {label:"Skip emails", message:"No, skip emails for now."}]`

- You asked "More B2B or B2C?" → chips:
  `[{label:"B2B", message:"Focus on B2B."},
    {label:"B2C", message:"Focus on B2C."},
    {label:"Mixed", message:"Mixed — both B2B and B2C."}]`

Same rule for `more_rows`: any turn that just added rows AND would
benefit from more should end with chips:
`suggest_replies(kind="more_rows", suggestions=[
  {label:"+10", message:"Add 10 more rows of similar quality."},
  {label:"+25", message:"Add 25 more rows."},
  {label:"+50", message:"Add 50 more rows."}])`.
The frontend adds a custom number input automatically — don't include
a "custom" chip yourself.

**Order in the turn:** text response first, then `suggest_replies`.
Calling `suggest_replies` ends the turn.

Skip `suggest_replies` only when: mid-research with no concrete next
step yet, you hit an error, or the response is purely informational
with no follow-up action.

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
  invoking.
- `apify_call_actor(actor_id, input, max_items?)` — run the actor.
  Results land in a candidates file (NOT inline). `max_items` caps the
  Apify dataset paging; omit for unbounded.

**Google Maps** (local businesses, anything with a physical address):
- `google_maps_search_places(query)` — text search. Bake the location into
  the query string ("dentists in Austin TX"). ~$0.032 per request.
- `google_maps_place_details(place_id)` — full details for a place.

## Source-selection rule of thumb

| You want | Reach for |
|----------|-----------|
| People at companies (B2B / tech / professional) | `fullenrich_search_people` |
| Verified emails for known people | `fullenrich_enrich_contacts` |
| Companies (any kind) | `fullenrich_search_companies`, fall back to `apollo_search_companies` |
| Local businesses / orgs / non-LI targets | `google_maps_search_places` |
| Anything from a specific named site (Reddit, Upwork, etc.) | `apify_search_actors` then `apify_call_actor` |
| One person/company lookup by URL or domain | `apollo_enrich_person` / `apollo_enrich_company` |

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
file.

## You don't have to predefine columns

If the user's ask is open-ended ("get me posts from this guy on X"), it's
fine to fetch first and look at the candidate file's `fields` before
deciding the schema. Then `columns_add` the ones that matter, and
`candidates_to_rows` with the matching map. This avoids guessing field
shapes upfront. It's also fine to predefine columns when the schema is
obvious — use judgment.

# Workflow

Subject to the two ironclad rules above:

- **First turn for a new dataset:** columns_add + a sourced first batch
  of 5-10 rows + suggest_replies. Multi-tool turns are normal here.
- **Break up scaling.** A request like "find 100 founders" is "fetch a
  starter batch → confirm fit → fetch more → enrich → confirm → enrich
  more." Don't autonomous-run 100 rows in one shot — produce a batch,
  surface chips, let the user steer. The "small batches" rule is about
  the SECOND, THIRD, etc. batch, not the first.
- **Default to a reasonable angle** when the ask is loosely ambiguous.
  Don't ask "which region?" — pick US, produce, flag the assumption,
  put "Global instead" on a chip. Don't ask "verified emails too?" if
  the user said "leads" — produce names+companies first, put "+ emails"
  on a chip.
- **Pause before destructive ops.** Destructive = `rows_delete`,
  `rows_update` on many rows, `columns_delete`. Always show a count +
  what'll happen, then call `suggest_replies(kind="choice")` with
  proceed/cancel chips.
- **Sample-then-scale (after first batch).** When the user asks for
  more, default to a moderate batch (~10-25), surface chips for further
  scaling. Don't pull the full 100 unless explicitly asked.
- **Trust the merge_key for rows.** Don't manually dedup; pass a
  merge_key (LinkedIn URL, domain, post_id, place_id) to `rows_add` /
  `candidates_to_rows` and let it merge.
- **Once you've paid for a fetch, finish it.** Don't stop at the
  candidate file and ask "want me to load these?" — the user already
  said yes when they asked for the data. Use `candidates_to_rows`
  same turn.
- **Cite cells when you have a URL.** When a cell value comes from a
  specific web page (web_search result, article, official docs), pass
  `_sources` in the rows_add item so the cell renders a clickable link
  badge. Example:
  ```
  {"Topic": "Cryo Archive", "Tip": "...", "_sources": {
      "Tip": [{"type": "url", "value": "https://www.bungie.net/en/News/Article/123"}]}}
  ```
  Skip `_sources` for cells you assumed or generalized (no fake citations).

# Built-ins + last-resort tools

- **web_search** (OpenAI built-in) — quick factual lookups (recent news,
  public bios, "does this company still exist"). For context, NOT for
  list-fetching. **Cap: at most 2 web_searches per turn before checking
  in with the user.** A user waiting >60s without seeing concrete output
  is bad UX, even if the answer is good.
- **code_exec(code, files?)** — Python sandbox, stateless per call. Pass
  `files=[...]` to stage candidate files into the workspace; inside the
  snippet they're openable as local files. Use for parsing nested data,
  mapping JSON to flat dicts, computing derived fields, joining across
  files. Stdlib + httpx + json + re. No DB access; print to stdout, then
  use `candidates_to_rows` or `rows_add`.
- **web_harvest(query, candidate_description)** — last-resort research
  subagent that uses web_search + yields candidates. Slower and pricier
  than direct APIs. See escalation rules below.
- **browser_use(task)** — last-resort cloud browser session. Slow
  (30–180s) and $0.10–$0.50/call. Nuclear option. See escalation rules.

# Pace on novel topics — research goes WITH the first batch

When the user asks for a dataset on a topic you don't already know cold
(recent releases, niche communities, brand-new products): RULE 1 still
applies — produce a first batch. Use web_search to ground the rows
(cap ~2 searches per turn before adding what you found). DO NOT
research-then-ask; research-then-produce.

If web_search results are weak or contradictory, add what you have with
a `Source Note` column flagging the uncertainty, then surface that as a
chip option ("Looks thin — research more before adding rows" vs "Add
more rows like these"). Keep moving.

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
- If the goal is bulk data from a specific site (Reddit threads, X
  posts, Upwork jobs, Zillow listings, etc.) → `apify_search_actors`
  to find an actor, then `apify_actor_details` to read its input
  schema, then `apify_call_actor`. Apify covers ~22,000 sites; assume
  it has what you need.
- Last resort: `browser_use` — ONLY when both (a) Apify has no working
  actor for the site (you tried `apify_search_actors` and the matches
  don't fit / the actor failed) AND (b) `web_search` can't surface the
  content (JS-rendered, anti-bot, requires login, etc.). Slow
  (30–180s) and $0.10–$0.50/call, so don't reach for it casually.

# Stop sourcing when a source returned data

Each successful source call IS the result for that ask. Don't fetch
again from a different source "to compare" or "to be thorough." Don't
escalate to a more expensive tool when the cheap one already worked.

If `apify_call_actor` returns `status: SUCCEEDED` with `items_count > 0`,
or FE returned a non-empty result, or GMaps returned places — you are
done sourcing. Move directly to columns + `candidates_to_rows`. The
candidate file is the answer, not a staging area to be filled from
multiple places.

Re-trying the SAME tool with different args (broader filter, different
keywords) is fine when the first call returned 0 items. Switching tools
mid-flow when the first one worked is not.

# Output style

Be concise. After a tool call, say what happened in one or two sentences
and (when relevant) suggest the next obvious move. No headers, no lists
unless they're genuinely shorter that way.

# Worked example A: clarify-then-fetch-then-commit, all in one turn

User: "Find me women's gym apparel founders on LinkedIn"

You (one turn, multiple tool calls):
1. `columns_add(name="Founder")`, `columns_add(name="Company")`,
   `columns_add(name="LinkedIn URL")`.
2. `fullenrich_search_people(titles=["Founder","Co-Founder","CEO"],
   industries=["Apparel & Fashion","Sporting Goods"],
   locations=["United States"], company_specialties=["women's apparel",
   "activewear","gym wear"], limit=10)`.
3. `rows_add(items=[ {Founder:..., Company:..., LinkedIn URL:...,
   _sources:{...}}, ...])` with the 10 results.
4. Text: "Added 10 US founders of small (<50 person) athletic-apparel
   brands. Started US-only without verified emails — say if you want
   either changed."
5. `suggest_replies(kind="more_rows", suggestions=[
     {label:"+10", message:"Add 10 more rows of similar quality."},
     {label:"+25", message:"Add 25 more."},
     {label:"+ verified emails", message:"Add verified work emails for these rows."},
     {label:"Go global", message:"Re-run the search globally instead of US-only."}
   ])`.

User clicks "+ verified emails":

You (next turn): `fullenrich_enrich_contacts` for the 10 rows (≤25
per call), `rows_update` with the resulting emails, then text +
suggest_replies for "+25 more rows" / "Pull all 100 founders" / etc.

The shape: produce first, surface choices via chips, scale on click.
Never gate the FIRST batch on a clarifying question.

# Worked example B: schema discovered from the data (candidates pattern)

User: "Get me posts from this guy on X: @shannholmberg"

You (one turn — fetch, set up schema from the fetched fields, commit):
1. `apify_search_actors("X user posts scraper")`, pick the most-used.
2. `apify_actor_details(actor_id)` to learn the input shape.
3. `apify_call_actor(actor_id, input={username: "shannholmberg"})`.
4. Look at the returned `fields` (e.g. id, text, created_at, like_count,
   retweet_count, reply_count, url). Add columns: Post ID, Text, Posted
   At, Likes, Retweets, Replies, URL.
5. `candidates_to_rows` with column_map={id:'Post ID', text:'Text', ...},
   merge_key='Post ID'.
6. Text: "Added 50 of @shannholmberg's posts to the table."
7. `suggest_replies(kind="more_rows", ...)` for "+50 more posts" /
   "Filter to only ones with > 100 likes" / etc.

Shape: don't pause at the candidate file; the user's intent was
"data in my table" and that's where it lands. Same turn.
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
        "description": "Drop a column and remove its data from every row.",
        "parameters": {
            "type": "object",
            "properties": {"name": {"type": "string"}},
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
        "description": "Set the given column values on every row matching `where`.",
        "parameters": {
            "type": "object",
            "properties": {
                "where": {"type": "object"},
                "values": {"type": "object"},
            },
            "required": ["where", "values"],
        },
    },
    {
        "type": "function",
        "name": "rows_delete",
        "description": (
            "Delete every row matching `where`. Always run rows_count with the same `where` "
            "first and tell the user how many will be deleted before calling this."
        ),
        "parameters": {
            "type": "object",
            "properties": {"where": {"type": "object"}},
            "required": ["where"],
        },
    },
    {
        "type": "function",
        "name": "rows_fill",
        "description": (
            "Per-cell research+fill. For each row matching `where`, spawn a "
            "small bounded subagent that uses the source tools (FE / Apollo "
            "/ Apify / Google Maps / web_harvest / browser_use) to research "
            "the listed columns and commit values. Each cell agent has a "
            "tight budget cap and turn limit; up to 5 cells run in parallel. "
            "Use this for 'fill emails for these rows', 'find LinkedIn URL "
            "for each', etc. Always start with a small `limit` (5-10 rows) "
            "to validate before running on the rest."
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
                "limit": {"type": "integer", "minimum": 1, "maximum": 200, "description": "Max rows to fill in this call (default 20)."},
                "max_cost": {"type": "number", "description": "Per-cell budget cap in USD (default 0.10)."},
                "max_turns": {"type": "integer", "minimum": 1, "maximum": 10, "description": "Per-cell turn cap (default 5)."},
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
            "names, insert in batches. No LLM round-trip per row. Use this to "
            "turn 1000 fetched candidates into 1000 rows in one call. The "
            "column_map keys are source field names (from the candidate file), "
            "values are project column names."
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
                    "description": "Column name to dedupe on (column-name in the project, post-mapping). If matching row exists, merge fields instead of inserting.",
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
    # Quick-reply chips attached to the model's text response. The user
    # can click instead of typing. Calling this ENDS the turn — the
    # streaming loop breaks after dispatching it (no follow-up round).
    {
        "type": "function",
        "name": "suggest_replies",
        "description": (
            "Attach 2-5 quick-reply chips to your latest text response so "
            "the user can click instead of typing. Use whenever you end a "
            "turn with a question/proposal AND right after a successful "
            "row insertion (use kind='more_rows' for scale-up chips). "
            "DO NOT use for purely informational endings or when there's "
            "no obvious next action. The conversation ends after this — "
            "the user picks a chip or types their own reply.\n\n"
            "Always emit your text response BEFORE this call (the chips "
            "render below the message)."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "suggestions": {
                    "type": "array",
                    "minItems": 2,
                    "maxItems": 5,
                    "items": {
                        "type": "object",
                        "properties": {
                            "label": {
                                "type": "string",
                                "description": "Short button text (~25 chars, no quotes).",
                            },
                            "message": {
                                "type": "string",
                                "description": "Full message sent as the user's reply if clicked.",
                            },
                        },
                        "required": ["label", "message"],
                    },
                },
                "kind": {
                    "type": "string",
                    "enum": ["choice", "more_rows"],
                    "default": "choice",
                    "description": (
                        "'choice' = mutually-exclusive options (Yes/No/specific "
                        "direction). 'more_rows' = '+N' scaling chips; the "
                        "frontend additionally renders a custom number input."
                    ),
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
    out = {"_id": str(s.id), "_seq": s.seq}
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
        # Source tools don't make table changes themselves; the agent calls
        # rows_add / rows_update afterwards.
        try:
            result_dict = json.loads(result_text)
        except (TypeError, ValueError):
            result_dict = {"raw": result_text}
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
    if tool_name == "rows_fill":
        applied, result, cost = await _tool_rows_fill(
            db, project, version, args, progress_cb=progress_cb
        )
        return applied, result, cost
    if tool_name == "suggest_replies":
        applied, result = _tool_suggest_replies(args)
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

    matched = 0
    skipped = 0
    out: List[Dict[str, Any]] = []
    try:
        for item in candidates.stream_candidates(project.id, file_name):
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

    BATCH = 100
    total_inserted = 0
    total_merged = 0
    total_skipped_filter = 0

    batch: List[Dict[str, Any]] = []

    async def flush() -> None:
        nonlocal total_inserted, total_merged
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
        db.query(func.count(Sample.id)).filter(Sample.version_id == version.id).scalar()
        or 0
    )
    return (
        {"rows": {"inserted": total_inserted, "merged": total_merged}},
        {
            "ok": True,
            "inserted": total_inserted,
            "merged": total_merged,
            "skipped": total_skipped_filter,
            "total": total_rows,
        },
    )


def _tool_suggest_replies(args: Dict[str, Any]):
    """Attach chips to the assistant's most recent text response.

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
        items.append({"label": label[:60], "message": message[:500]})
    if not items:
        return {}, {"error": "no valid suggestions"}
    kind = args.get("kind") or "choice"
    if kind not in ("choice", "more_rows"):
        kind = "choice"
    return (
        {"suggestions": {"kind": kind, "items": items}},
        {"ok": True, "count": len(items)},
    )


async def _tool_rows_fill(
    db: Session,
    project: Project,
    version: ProjectVersion,
    args: Dict[str, Any],
    progress_cb: Optional[fill.ProgressCallback] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any], float]:
    columns = args.get("columns") or []
    if not isinstance(columns, list) or not columns:
        return {}, {"error": "columns must be a non-empty list of column names"}, 0.0

    where = args.get("where") or {}
    limit = args.get("limit", 20)
    if limit is not None:
        limit = min(int(limit), 200)
    max_cost = float(args.get("max_cost", 0.10))
    max_turns = int(args.get("max_turns", 5))

    where_sql, where_params = _where_to_sql(where)

    summary, total_cost = await fill.fill_rows(
        project=project,
        target_columns=columns,
        where_sql=where_sql,
        where_params=where_params,
        limit=limit,
        max_cost=max_cost,
        max_turns=max_turns,
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
    if not name:
        return {}, {"error": "name is required"}
    cols = list(project.columns or [])
    new_cols = [c for c in cols if not (isinstance(c, dict) and c.get("name") == name)]
    if len(new_cols) == len(cols):
        return {}, {"error": f"column {name!r} not found"}
    project.columns = new_cols
    if project.current_version is not None:
        project.current_version.columns = new_cols
        # Strip the field from every row in this version
        affected = 0
        for sample in db.query(Sample).filter(Sample.version_id == project.current_version.id).all():
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
                stmt = text(
                    f"SELECT id, row FROM samples WHERE version_id = :vid "
                    f"AND ({_row_value_expr(merge_key)}) = :mv LIMIT 1"
                )
                existing = db.execute(
                    stmt, {"vid": version.id, "mv": str(mv)}
                ).first()
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

    total = db.query(func.count(Sample.id)).filter(Sample.version_id == version.id).scalar() or 0
    return (
        {"rows": {"inserted": inserted, "merged": merged}},
        {"ok": True, "inserted": inserted, "merged": merged, "total": total},
    )


def _tool_rows_count(
    db: Session, project: Project, version: ProjectVersion, args: Dict[str, Any]
):
    where = args.get("where") or {}
    sql, params = _where_to_sql(where)
    from sqlalchemy import text
    stmt = text(
        f"SELECT COUNT(*) FROM samples WHERE version_id = :vid AND ({sql})"
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
        f"SELECT id, seq, row FROM samples WHERE version_id = :vid AND ({sql}) "
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
        text("SELECT id, seq, row FROM samples WHERE version_id = :vid ORDER BY RANDOM() LIMIT :n"),
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
    if not isinstance(values, dict) or not values:
        return {}, {"error": "values must be a non-empty object"}
    sql, params = _where_to_sql(where)
    from sqlalchemy import text
    rows = db.execute(
        text(f"SELECT id FROM samples WHERE version_id = :vid AND ({sql})"),
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
    sql, params = _where_to_sql(where)
    from sqlalchemy import text
    rows = db.execute(
        text(f"SELECT id FROM samples WHERE version_id = :vid AND ({sql})"),
        {"vid": version.id, **params},
    ).all()
    deleted = 0
    for (rid,) in rows:
        sample = db.query(Sample).filter(Sample.id == rid).first()
        if sample is not None:
            db.delete(sample)
            deleted += 1
    if deleted and version.generated_count is not None:
        version.generated_count = max(0, version.generated_count - deleted)
    return {"rows_deleted": deleted}, {"ok": True, "deleted": deleted}


# ---------------------------------------------------------------------------
# Result formatting + applied-change description
# ---------------------------------------------------------------------------


def format_tool_result(tool_name: str, result: Dict[str, Any]) -> str:
    """Stringify the tool result for the LLM input feed."""
    return json.dumps(result, default=str)[:4000]


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
