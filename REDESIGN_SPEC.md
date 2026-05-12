# DatasetLabs v-Next Redesign — Spec

Single source of truth for the v-next redesign. Reviewed and iterated as a doc before any code lands. After lock-in, this drives the implementation.

---

## 1. System Prompt

The text below is the actual prompt the chat agent will see. Target: ~2.5k tokens. Cached via prompt caching since it changes rarely.

````md
# Identity

You are DatasetLabs — an AI that builds structured datasets from natural language. The user describes what they want; you set up tables, fetch candidates, configure enrichments, refine results, and iterate with the user inside a chat interface.

The product exists in the space between general-purpose research and a spreadsheet that already has the data. Most questions users care about can't be answered by an API query alone — Apollo doesn't know which engineers work on Claude Code, Google Maps doesn't know which doctors specialize in oral immunotherapy. Bridging that gap is what makes the product useful: pull a structured candidate pool, enrich the columns that decide fit, let the user filter and iterate.

Your job is to orchestrate the work well — pick the right sources, set up tables that match what the user asked for, define enrichments that succeed for most rows within a sensible budget, and help the user iterate. You handle orchestration and refinement; the user provides intent and feedback.

# Two kinds of work

Everything you do is one of these. They're distinct primitives with different tools and cost shapes. Order varies by project — fetching usually precedes enrichment, but a CSV upload is enrichment-only, a "scrape all posts in r/foo" is fetching-only, and mid-project additions are often enrichment-only.

**Fetching** — pulling candidates into a table from a source. One source-query per table. The candidates fill the user's *scope* (the universe their target lives within); they aren't already the final answer.

**Enrichment** — adding columns that derive info per-row, typically to determine target membership or pull supplemental data. This is where the user-facing filter often gets defined.

# Scope sizing

Before fetching, develop an internal rough sense of how big the candidate pool is. Don't surface numbers or thresholds to the user.

**Pool is comfortably tractable** → fetch the whole scope. Organize into one or more tables along natural query boundaries. Doctors in Istanbul. Anthropic employees. Recent posts in r/AiAutomations.

**Pool is too big to fetch in full** → pivot to a *proxy scope*: a smaller source whose members are a good signal for being in the user's target. "Engineers who use Claude Code" → pivot to people who file issues on the claude-code GitHub repo, or tweet about it. "Taco shell manufacturers" → SERP for the phrase, take the top 30-50 results as the scope.

The trigger for pivoting is *the scope is too big*, not *the user's filter doesn't exist as a source query.* Source-level filters for what the user actually wants almost never exist — that's why enrichment exists.

Good-enough scope coverage is fine. Stop pursuing thoroughness once you're close.

# Tables

One source-query per table. Use as many tables as the project naturally needs — Reddit + Quora are two tables; LinkedIn people + GitHub issue-openers are two tables. Tables are how the system organizes; pick whatever separation is natural.

Defaults:
- **`table_create`** fetches and commits 100 rows
- **`table_fetch_page`** pulls another 100

If the user expects rows landing fast, the system streams them — you don't need to wait for all 100 before continuing.

# Sources

Pick by data shape, not a fixed priority list. Rough strengths:

- **`fullenrich_people`, `fullenrich_companies`** — Apollo-like people/company database. Use when the target is a person or company with structured filters (title, headcount, geo, industry).
- **`google_maps`** — local orgs / places with geographic scope.
- **`apify_actor:<id>`** — vertical scrapers for specific platforms (Reddit, Quora, Indeed, LinkedIn jobs, Twitter, Instagram, etc.). Use `source_search_apify` to discover the right actor — don't assume an actor exists or works without checking.
- **`web_harvest`** — niche topics with no integration coverage, or fragmented web data. A bounded research subagent on a topic.
- **`file`** — uploaded tabular files (CSV/XLSX auto-parsed; other formats fall back to `code_exec` to transform first). Also used for implicit-knowledge or derived data: write a file via `code_exec`, then `table_create` with `source="file"`.

Integrations are preferred over open-web when they cover the data — they're more structured, more thorough at scale, more cost-efficient. But "preferred" isn't "always best." For niche topics with no integration, web is the right call.

This source list will grow as new integrations get added.

# Browser use

`browser_use` is its own source and can also be the underlying tool of an enrichment action. Use cases:
- As a **source** (`table_create(source="browser_use", ...)`) — scrape a specific known site (a directory page, a listings index) into a table.
- As an **enrichment action tool** — per-row scraping of a target URL derived from the row (visit the company's site, extract the careers page link).

It is a **last resort** — only when (a) no Apify actor covers the site and (b) native HTTP fails (JS rendering, antibot, login wall). Never use it for broad multi-site exploration. Sessions are bounded: reliability degrades fast above ~50 items per session, so for larger pulls, break into multiple bounded sessions.

# Working with the table

Run `enrichment_set` to define an enrichment AND run it on the table's first 10 unfilled rows. Inspect the results. If they look right, leave it alone — the enrichment is set up, the user can scale it via "run on rest" UI buttons (or you can via `enrichment_run`). If results are mostly wrong, call `enrichment_set` again with a refined `action` and the same `enrichment_id` to re-test on the same 10. Each cell is enriched once per refinement — no wasted work.

**Refine only when there's a clear problem:**
- Mostly noise, zero results, or wrong entity type (fetching)
- Multiple sample rows fail outright — hallucinated, wrong column, total miss (enrichment)
- Pricing wildly out of expected range

**NOT a clear problem:**
- Mostly right but you'd like cleaner
- Ranking isn't ideal
- One or two outputs look weird
- An enrichment finds X/10 when you'd expect Y/10 and X ≈ Y — that's working

**Cap: 3 refinements per turn.** If confident the query is right, skip refinement entirely and commit. Each refinement is latency to the user — don't take them if they aren't needed. Web (`web_harvest`) and `browser_use` don't need refinement — their subagent loops already handle filtering.

**Somewhat working is a win.** Commit and let the user steer. They iterate with you over many turns; you don't have to nail it in one.

**"Not finding" ≠ "not working."** If an enrichment finds 6/10 Twitter accounts because some people don't have one, that's working. If it finds 1/10 when you'd expect 7, something's off — try once. Don't hide failed results — leaving them visible is how the user (and you) calibrate next steps.

**Budget is a cap, not a target.** Lean generous on `per_row_credit_cap`. Rough rule of thumb:
- 1 credit — classification or extraction from data already in the row
- 5 credits — one tool call (web_search, FullEnrich enrich, simple lookup)
- 10 credits — multi-step research, browser_use, multi-tool cascade

Higher caps are safer. The user adjusts in the UI if rows fail.

# Research and scouting

Use `web_search` freely for quick lookups — figuring out what sources exist for a niche topic, verifying which Apify actor is right, checking what the canonical site is for some data. Soft budget around 5 scouting queries per turn. If it's productive, keep going.

# Decision flow (rough)

1. Understand what the user wants. If the ask is vague enough to risk wasted effort, clarify; otherwise just go.
2. Get an internal rough read on scope size, pick a proxy source if scope is too big.
3. Pick the source for the data shape. Scout briefly if you're unsure what's available.
4. Set up the first table. Set up the most important enrichments on it. Run the first batch.
5. Hand back to the user. They iterate from there — adding tables, refining filters, running more enrichments.

# Thin slice

In a first turn for a new project, aim for one well-set-up table with the most important enrichments configured and a small first batch enriched. Don't try to build out all tables and all enrichments in one turn. The user reacts to what you ship and steers the next move. Subsequent turns expand.

For trivially small projects (one table, no enrichment needed) this collapses naturally. For multi-table projects, pick the most important table for turn 1.

# Filters

Filters surface a slice of the table without destroying data. You can set filters proactively (`filter_set`); the user can toggle them off in the UI. Seeing the full scope is often valuable even when the user only cares about a slice — "15% of engineers work on Claude Code" is information even if the user only wants those 15%.

`row_delete` is for explicit user intent ("delete rows 12-15"), not for narrowing. If a row looks off, set a filter or enrich the determining column — don't delete.

# Approvals

Some tool calls automatically prompt the user for approval — running an enrichment on many rows, deleting tables or rows, large fetches. You call the tool normally; the system pauses, the user sees a card with the action and cost, and approves or denies. You'll see the response in the tool result — proceed if approved, adapt if denied.

# Visibility

Each turn, you'll see a project state block describing tables, columns, row counts, active filters, enrichments configured, and any column class breakdowns (for low-cardinality columns). This is your persistent awareness — refer to it rather than calling tools to re-discover what already exists.

# Voice

Speak in the user's vocabulary. They see a table and what's in it; they don't see source mechanics, schemas, or internal state. Don't reference Apify actors, input schemas, or other internals.

# Anti-patterns

- Optimizing past good-enough. If results are mostly right, commit.
- Predicting cost in dollars or warning the user that something is "expensive." The UI handles cost.
- Re-pulling a source you already covered with a different tool.
- Pre-deleting rows to narrow — filters do this without destroying data.
- Trying to be exhaustive in one turn — the user iterates with you.
- Refining when you're already confident.
````

---

## 2. Tools

13 total, namespaced by prefix. Each tool is single-purpose, atomic. Schemas below in pseudo-JSONSchema for clarity; we'll generate the actual OpenAI tool definitions from these.

### 2.1 `table_create`

```yaml
description: |
  Create a new table backed by one source query, fetch the first batch
  of candidates, and commit them. Use when you've decided on a source
  and query params and want to materialize rows. The first 10 rows
  stream in fast; the remaining 90 land asynchronously.
params:
  name:        string       # user-visible table name
  source:      SourceEnum   # see Sources section
  query_params: object      # shape varies by source; see Source Query Params section
  n:           integer      # default 100, max 1000
returns:
  table_id:    string
  rows_initial: integer     # how many rows committed initially (typically n or less if source ran short)
  preview:     row[]        # first 5 rows of the table for the agent to inspect
approval_required: false
```

### 2.2 `table_update_query`

```yaml
description: |
  Re-run a table's fetch with revised query params, replacing prior
  rows. Use when the initial fetch returned mostly noise, zero
  results, or the wrong entity shape. Soft cap: 3 calls per turn
  per table. Each call replaces the prior rows; no rows are
  duplicated.
params:
  table_id:    string
  new_query_params: object
returns:
  table_id:    string
  rows_replaced: integer  # how many prior rows got replaced
  rows_initial: integer
  preview:     row[]
approval_required: false
```

### 2.3 `table_fetch_page`

```yaml
description: |
  Pull the next page of candidates from the table's source, appending
  rows. Server tracks the cursor per-table per-source. Calling twice
  yields different rows (no duplication). Use when the user wants
  more rows of the same kind.
params:
  table_id:    string
  n:           integer      # default 100
returns:
  rows_added:  integer
  preview:     row[]        # last 5 rows added
approval_required: if n > 100
```

### 2.4 `table_delete`

```yaml
description: |
  Delete a table and all its rows + enrichments. Destructive.
params:
  table_id:    string
returns:
  ok:          boolean
approval_required: true
```

### 2.5 `source_search_apify`

```yaml
description: |
  Discover Apify actors matching a query. Use when you need a source
  for a platform that isn't covered by built-in integrations
  (FullEnrich, Google Maps). Returns top N actors with descriptions,
  recent run count, and pricing per item. Use the actor_id in
  `table_create` as source="apify_actor:<actor_id>".
params:
  query:       string       # e.g. "Reddit posts scraper", "Untappd users"
returns:
  actors:
    - actor_id:         string
      title:            string
      description:      string
      monthly_run_count: integer
      price_model:      string   # e.g. "$3 per 1000 results"
      rating:           number?
approval_required: false
```

### 2.6 `enrichment_set`

```yaml
description: |
  Define a new enrichment (or refine an existing one) and run it on
  the table's first 10 unfilled rows. If enrichment_id is omitted,
  creates new. If provided, updates the action/config and re-runs
  the first 10 rows (overwriting prior values). Each cell is enriched
  once per refinement — no wasted work. Soft cap: 3 sets per
  enrichment per turn.
params:
  table_id:    string
  columns:                  # one or more columns filled by this enrichment
    - name:    string
      type:    ColumnType   # see Column Types section
  action:
    type:      "tool" | "cell_agent"
    tool:      string?      # for type=tool: the source/tool to call (e.g. "fullenrich_enrich_contacts")
    args_template: object?  # for type=tool: param shape with {row.x} placeholders
    prompt:    string?      # for type=cell_agent: natural-language instruction
  per_row_credit_cap: integer
  name:        string       # human-readable enrichment label
  enrichment_id: string?    # if present, refines existing
returns:
  enrichment_id: string
  rows_filled: integer       # of the first 10
  results_preview: object[]  # the filled cells for the agent to inspect
approval_required: false
```

### 2.7 `enrichment_run`

```yaml
description: |
  Extend the enrichment to more rows. By default skips already-filled
  cells (so refinement runs don't re-pay). Background job — returns
  immediately with a job_id; rows fill in live in the UI.
params:
  enrichment_id: string
  scope:
    type: "first_n" | "all_unfilled" | "row_ids" | "filter"
    first_n:   integer?   # for first_n
    row_ids:   string[]?  # for row_ids — explicit
    filter:    object?    # for filter: {column, op, value}
  overwrite:   boolean    # default false; if true, also re-runs filled cells
returns:
  job_id:      string
  rows_queued: integer
approval_required: if scope.first_n > 10 OR scope = all_unfilled OR scope = filter
```

### 2.8 `filter_set`

```yaml
description: |
  Apply a non-destructive filter to a table column. Use proactively
  to surface what the user asked for. Filter is visible in the UI
  as a chip the user can toggle off. Returns the matched count and
  a small sample so you can confirm the filter does what you
  expected.
params:
  table_id:    string
  column:      string
  op:          FilterOp    # depends on column type; see Filter Operators
  value:       any
returns:
  matched:     integer
  total:       integer
  sample:      row[]      # first 5 matching rows
approval_required: false
```

### 2.9 `filter_clear`

```yaml
description: |
  Remove a filter from a column. Rare — users usually clear via UI.
params:
  table_id:    string
  column:      string
returns:
  ok:          boolean
approval_required: false
```

### 2.10 `row_inspect`

```yaml
description: |
  Read-only peek at rows from a table. Use when you need to inspect
  specific cells, take a sample, or check rows matching a condition
  without changing what the user sees. For showing a filtered view
  to the user, use filter_set instead.
params:
  table_id:    string
  filter:      object?     # optional WHERE-clause-like
  n:           integer     # default 10, max 50
  sort_by:     string?     # column name, default "_created_at desc"
returns:
  rows:        row[]
approval_required: false
```

### 2.11 `row_delete`

```yaml
description: |
  Delete rows from a table. For explicit user intent ("delete rows
  12-15", "drop the empty rows"). Never use for narrowing — use
  filter_set instead.
params:
  table_id:    string
  row_ids:     string[]?   # explicit rows
  filter:      object?     # OR: delete all rows matching filter
returns:
  rows_deleted: integer
approval_required: true
```

### 2.12 `code_exec`

```yaml
description: |
  Execute a Python snippet in the sandbox. Use for transforms,
  derived computations, populating manual tables, transforming
  uploaded files, anything that doesn't fit a dedicated tool.
  Helpers available in the sandbox:
    - from dsl_tools import add_rows, get_table, list_tables
    - browser_use (for site-specific scraping; use sparingly)
params:
  code:        string
  files:       string[]?   # file_ids to stage into the sandbox as local files
returns:
  ok:          boolean
  stdout:      string      # truncated to 8000 chars
  stderr:      string
  duration_ms: integer
approval_required: false
```

### 2.13 `web_search`

```yaml
description: |
  Quick lookup. Returns top web search results — use for scouting,
  verifying which Apify actor to use, finding canonical sites for
  niche data, etc. Different from web_harvest source (which builds
  rows in a table); web_search is a thinking tool, not a row source.
params:
  query:       string
returns:
  results:
    - title:   string
      url:     string
      snippet: string
approval_required: false
```

### 2.14 `suggest_replies`

```yaml
description: |
  Emit chip suggestions for the user's next move. Call at the end
  of each turn. Chips are short (~40 chars), action-oriented, and
  represent the natural next options the user might want.
params:
  chips:
    - label:   string       # the chip text
      message: string       # what to send if clicked
returns:
  ok:          boolean
approval_required: false
```

That's 14 (miscounted again earlier — `filter_clear` is the 14th). Acceptable. All atomic, single-purpose, well-named.

---

## 3. Sources & Query Params

### Source enum

```
fullenrich_people
fullenrich_companies
google_maps
apify_actor:<actor_id>     # parameterized; actor_id from source_search_apify
web_harvest
browser_use
file
```

### Per-source `query_params` shapes

**`fullenrich_people`**
```yaml
job_titles: string[]?
seniority_levels: string[]?     # e.g. ["c-level", "vp", "director"]
company_names: string[]?
company_industries: string[]?
company_headcount_min: integer?
company_headcount_max: integer?
locations: string[]?
exclude_titles: string[]?
exclude_companies: string[]?
```

**`fullenrich_companies`**
```yaml
industries: string[]?
headcount_min: integer?
headcount_max: integer?
locations: string[]?
funding_stages: string[]?
keywords: string[]?
```

**`google_maps`**
```yaml
query: string                   # e.g. "flooring contractor"
location: string                # e.g. "San Diego, CA"
radius_miles: integer?
min_rating: number?
max_review_count: integer?
```

**`apify_actor:<actor_id>`**
```yaml
input: object                   # schema is actor-specific; server fetches schema and validates
```
Server auto-fills plumbing fields (proxy etc.) before validation — agent doesn't supply them.

**`web_harvest`**
```yaml
query: string                   # what to look for
candidate_description: string   # what a successful row looks like
max_candidates: integer         # default 30
max_turns: integer?             # how many search/page iterations the subagent gets, default 6
```

**`browser_use`**
```yaml
url: string                     # the page to start at
task: string                    # what to extract or navigate to
candidate_description: string   # what a successful row looks like
max_candidates: integer         # default 30; reliability cliffs above ~50
```
For larger pulls, agent calls `table_create` with `source="browser_use"` multiple times against different starting URLs or sub-pages rather than one huge session.

**`file`**
```yaml
file_id: string                 # from upload, or written via code_exec
```
Server auto-detects CSV/XLSX, parses, returns rows. Other formats → error with hint to use `code_exec` to transform first.

For implicit-knowledge populating, derived data, or pasted content: the agent uses `code_exec` to write a CSV/JSON file in the sandbox, gets back a file_id, then calls `table_create(source="file", query_params={file_id})`. No separate "manual" source needed.

---

## 4. Column Types (soft)

```
text       (default)
number
url
email      → triggers automatic Scrubby verification on filled cells
phone      → triggers automatic phone verification on filled cells
date
bool
enum       → auto-detected when cardinality < 20 unique values
```

Soft means: server stores any value regardless of type. Type only affects:
- UI rendering (URLs as links, dates formatted, etc.)
- Filter operator menu (number gets `<`, `>`, `between`; enum gets value picker)
- Side effects (email triggers verify; phone triggers verify)

Type is set at column declaration time (in `enrichment_set` or auto-inferred when source returns rows).

---

## 5. Filter Operators (per column type)

| Type | Operators |
|---|---|
| `text` | `equals`, `not_equals`, `contains`, `not_contains`, `starts_with`, `ends_with`, `regex`, `is_null`, `is_not_null` |
| `number` | `equals`, `not_equals`, `<`, `>`, `<=`, `>=`, `between`, `is_null`, `is_not_null` |
| `url` | `equals`, `not_equals`, `contains`, `is_null`, `is_not_null` |
| `email` | `equals`, `not_equals`, `contains`, `verification_status`, `is_null`, `is_not_null` |
| `phone` | `equals`, `not_equals`, `verification_status`, `is_null`, `is_not_null` |
| `date` | `equals`, `<`, `>`, `between`, `is_null`, `is_not_null` |
| `bool` | `equals`, `is_null`, `is_not_null` |
| `enum` | `equals`, `not_equals`, `in`, `not_in`, `is_null`, `is_not_null` |

Cell status filter (special): `cell_status equals filled | not_found | failed | pending`.

---

## 6. Cell Status

Per-cell status enum:

- `filled` — has a value
- `not_found` — enrichment ran, legitimate miss (no Twitter exists for this person)
- `failed` — enrichment errored or hit budget cap without result
- `pending` — not yet enriched

UI renders accordingly. Agent sees per-column rollups in the project state banner (`twitter_url: 41 filled, 6 not_found, 0 failed, 0 pending`).

Transitions:
- `pending` → `filled` (success) | `not_found` (legit miss) | `failed` (error)
- `filled` → re-runnable only with `overwrite=true`
- `not_found` / `failed` → re-runnable on subsequent `enrichment_run`

---

## 7. Project State Banner

Auto-injected before every user message in the chat run. Format:

```xml
<project_state>
Tables:
  - reddit_pain_points (47 rows, source: apify_actor:clearpath/reddit-search-scraper)
      Query: { subreddit: "automation", sort: "new", time: "month", limit: 100 }
      Columns: title (text), subreddit (enum), author (text), posted_date (date), url (url), pain_signal (enum)
      Column "pain_signal" classes: clear_need (28), exploring (12), off_topic (7)
      Filters: posted_date > 2026-04-01 (matches 47 of 47)
      Last activity: 2 min ago

  - anthropic_engineers (43 rows, source: fullenrich_people)
      Query: { company_names: ["Anthropic"], seniority_levels: ["senior", "staff"] }
      Columns: name (text), title (text), linkedin_url (url), email (email)
      Column "email" status: 38 filled (38 verified), 5 pending
      Filters: none
      Last activity: 1 hr ago

Enrichments configured:
  - id=e1 "Works on Claude Code" on anthropic_engineers
      Fills: works_on_claude_code (bool), evidence_url (url)
      Cap: 8 cr/row
      Status: 10/43 rows complete (8 yes, 2 no, 0 failed)
  - id=e2 "Verified email" on anthropic_engineers
      Auto-runs via type=email on the "email" column. 38 verified, 5 pending.

Filters active across project: 1 (see per-table above)
</project_state>
```

Token cost: typically 200-800 tokens for a normal project, scales with table count. Cached as part of prompt context for the turn.

Agent uses this as situational awareness. Doesn't need to call any tool to know what tables exist or how many rows are filled.

---

## 8. Approval Allowlist

Server-side config. When a flagged tool call is invoked, server pauses the run, emits `approval_required` event, FE renders card, user approves/denies, server resumes with the tool result (success or `{denied: true, reason}`).

```yaml
- tool: enrichment_run
  approval_when:
    - scope.type == "all_unfilled"
    - scope.type == "filter"
    - scope.type == "first_n" AND scope.first_n > 10
    - scope.type == "row_ids" AND len(row_ids) > 10
- tool: table_delete
  approval_when: always
- tool: row_delete
  approval_when: always
- tool: table_fetch_page
  approval_when: n > 100
```

Card content (FE-rendered):
- Action description ("Run enrichment 'Verified email' on 33 rows")
- Estimated cost (`max 33 × 5 = 165 credits`)
- Buttons: **Approve** / **Cancel**

If denied: agent gets `{ok: false, denied: true, reason: "user_cancelled"}` as tool result and adapts.

---

## 9. Schema Changes

New tables (or extensions to existing ones — TBD per migration ease):

### `tables`
```sql
id              uuid primary key
project_id      uuid references projects
name            text not null
source          text not null            -- enum value
query_params    jsonb not null
source_cursor   jsonb                    -- for pagination
created_at      timestamptz default now()
deleted_at      timestamptz
```

### `enrichments`
```sql
id                  uuid primary key
table_id            uuid references tables
name                text not null
columns             jsonb not null       -- [{name, type}, ...]
action              jsonb not null       -- {type: tool|cell_agent, ...}
per_row_credit_cap  integer not null
created_at          timestamptz default now()
deleted_at          timestamptz
```

### `samples` (existing — extend)
Add:
```sql
table_id  uuid references tables    -- key by table now, not project directly
```
Migration: existing projects get a default "Main" table created; existing samples re-keyed.

### `cells` (or extend `samples.row`)
Per-column cell status. Option A: store `{value, status}` per column in `samples.row` JSONB. Option B: separate `cells` table with `(sample_id, column_name, value, status)`. **Option A is simpler — go with that for v1**, revisit if we need per-cell history.

### `column_definitions` (or store on `tables.columns`)
Per-table column metadata (name, type, source — plain/enriched). Store as JSONB on `tables.columns` for simplicity.

### `filters` (or store on `tables.active_filters`)
Per-table active filters. JSONB on table.

### `project_versions` (existing — extend)
Already forks on every message. Extend to snapshot tables + enrichments + filters as part of version state.

---

## 10. Migration

Existing single-table projects:
- On first chat message after upgrade: create a `tables` row named "Main" with `source = "manual"` and `query_params = {columns: project.columns}`
- Re-key existing `samples` to point at the new table_id
- Project's old `columns` field becomes `tables[0].columns`

Effectively a no-op for the user — they just see their existing single table working as before, named "Main." They can rename it or add more tables.

Old tools we'll remove: `apify_search_actors`, `apify_actor_details`, `apify_call_actor`, `candidates_inspect`, `candidates_to_rows`, `candidates_list`, `rows_add`, `rows_fill`, `rows_delete`, `rows_update`, `rows_count`, `rows_get`, `rows_sample`, `columns_add`, `columns_delete`, `version_label`, `confirm_budget`, `cell_traces_inspect`, `set_instructions`, others as we find them.

---

## 11. Open Items / Defer

- **Cross-table references** (Clay-style formula columns) — v2.
- **Skills** — not in v1. Add when we observe repeated successful patterns.
- **Per-table versioning** — staying with per-project for now.
- **Subagent web research** beyond `web_harvest` source — defer.
- **Custom enrichment runners** (user-defined webhook actions) — defer.
- **`strategy_hint` for cached recipes** — dropped per discussion; `action.prompt` IS the strategy.

---

## 12. Testing Plan

After implementation lands:
- Drive ~10 representative projects end-to-end as if I were the user
- Judge the system, not individual project fit (3+ failures of the same kind = system issue; 1 = accept)
- Fixture pool: the 7 ICP partner projects + a CSV upload test + a manual-table test + a proxy-pivot test
- Report **system-level** findings only (prompt undercaps refinements, approval card unclear, etc.)
- No per-project patches to the prompt or skills based on test results
