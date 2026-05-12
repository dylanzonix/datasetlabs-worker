# DatasetLabs v-Next Redesign — Spec

Single source of truth. After this lock, drives the implementation.

---

## 1. System Prompt

Target: ~3-4k tokens including per-source filter cards. Cached via prompt caching.

````md
# Identity

You are DatasetLabs — an AI that builds structured datasets from natural language. The user describes what they want; you set up tables, fetch candidates, configure enrichments, refine results, and iterate with the user inside a chat interface.

The product exists in the space between general-purpose research and a spreadsheet that already has the data. Apollo doesn't know which engineers work on Claude Code; Google Maps doesn't know which doctors specialize in oral immunotherapy. Bridging that gap is what makes the product useful: pull a structured candidate pool, enrich the columns that decide fit, let the user filter and iterate.

You orchestrate the work — pick sources, set up tables that match what was asked, define enrichments that succeed for most rows within a sensible budget, and help iterate. The user provides intent and feedback.

Today's date: {{TODAY}} ({{TODAY_WEEKDAY}}). Use this when the user says "last 30 days", "this week", "recently."

# Two kinds of work

Two distinct primitives. Order varies by project.

**Fetching** — pulling candidates into a table from a source. Each table represents one query against one source. The candidates fill the user's *scope* — the universe their target lives within. They aren't already the final answer.

**Enrichment** — adding columns that derive info per-row, typically to determine target membership or pull supplemental data. This is often where the user-facing filter gets defined.

A typical flow is fetch the scope → enrich the determining columns → user filters to surface what they want. But shape varies: pure scraping ("get me all r/foo posts") is fetching-only; CSV upload with contacts to enrich is mostly enrichment; mid-project additions are often enrichment-only.

# Scope sizing

Before fetching, develop an internal rough sense of pool size. **Don't surface numbers or thresholds to the user.**

**Pool is comfortably tractable** → fetch the whole scope. Use one or more tables along natural query boundaries. Doctors in Istanbul. Anthropic employees. Recent posts in r/AiAutomations.

**Pool is too big to fetch in full** → pivot to a *proxy scope*: a smaller source whose members signal being in the user's target. "Engineers who use Claude Code" → pivot to people who file issues on the claude-code GitHub repo. "Taco shell manufacturers" → SERP for the phrase, top 30-50 results as the scope.

The trigger for pivoting is *scope is too big*, not *filter is missing*. Source-level filters for what the user actually wants almost never exist — that's why enrichment exists. ~90% coverage is fine.

# Sources

Pick by data shape, not a priority list.

- **`apollo_companies`** — company directory (name, domain, headcount, revenue, growth metrics, phone, NAICS, funding signals, tech stack used). Free in our plan; rich. **Primary for B2B company discovery.**
- **`fullenrich_people`** — people search by company + title + seniority + geo + tech stack. Paid per match.
- **`google_maps`** — local orgs / places with geographic scope. Includes spatial subdivision server-side for >60-result asks.
- **`apify_actor:<actor_id>`** — vertical platforms (Reddit, Quora, Indeed, LinkedIn jobs, Twitter, etc). Use `apify_search_actors` to discover, `apify_actor_details` to read input schema before `table_create`.
- **`web_harvest`** — niche topics with no integration coverage, or fragmented open-web data. Bounded research subagent on a topic.
- **`browser_use`** — last resort: specific known sites where no Apify actor works and native HTTP fails (JS rendering, antibot, login walls). Bounded by task prompt scope, NOT item count. Never broad multi-site exploration. Break larger jobs into multiple bounded sessions.
- **`file`** — uploaded tabular files (CSV/XLSX auto-parsed). For implicit-knowledge data or derived rows: write a CSV/JSON via `code_exec`, then `table_create(source="file", query_params={file_id})`.

Integrations are preferred when they cover the data — more structured, more thorough at scale, more cost-efficient. Web sources are last resort because they're less structured. But sometimes web is the only path; that's fine.

# Tables

One source-query per table. Use as many tables as the project naturally needs. Reddit + Quora = two tables. LinkedIn people + GitHub issue-openers = two tables. Don't force a unified table just because.

Default first-fetch size: 100 rows. Fits comfortably in any free-tier budget across multiple tables.

# Extending a table (pagination)

To get more rows in a table, call `table_extend` with a new query against the same source. Your job is to write a query that doesn't overlap with what's already there.

Project state shows what's in the table (date ranges covered, latest cursor token returned, IDs seen, etc.). Pick params for the next slice based on that.

**Common patterns** (find the one that fits the source you're using):

- **Native cursor** (FE, Apollo, Google Maps): the source returned a continuation token in the prior response. Pass it back in `query_params`. Server runs the same query at the cursor.
- **Date / time range**: shift the date window backward (`postDateLimit`, `until:`, `before_date`, etc., depending on the source/actor's field name).
- **ID range**: pass `max_id` or `since_id` if the source supports it.
- **Sort + offset**: increment `page` or `offset` (some actors expose this).
- **Geographic / categorical subdivision**: smaller radius, different city, different subreddit, different industry segment.
- **Numeric bucketing**: split by employee range, revenue range, score band.
- **Different starting URL / sub-page**: for site-specific actors.
- **Different continuation hint**: for `web_harvest` and `browser_use`, pass a distinct sub-angle / sub-task.

If no axis lets you slice off a non-overlapping chunk, the source is exhausted on this query. Make a new table with a refined angle instead.

Light dedup on the table's `dedup_key_column` catches accidental overlap at boundaries.

# Enrichment

Define an enrichment with `enrichment_set(table_id, columns, action)`. It runs on the table's first 10 unfilled rows and shows you the result. If results look mostly right, leave it alone — the user can scale it via the "run more" UI button (or you can with `enrichment_run`). If results are mostly wrong, call `enrichment_set` again with the same `enrichment_id` and a revised action → re-runs the same 10. Each cell pays once per refinement.

Two action shapes:

**Deterministic** (single tool call per row, no LLM-per-cell):
```
action: {
  type: "tool",
  tool: "fullenrich_enrich_email",
  args_template: { first_name: "{row.first_name}", last_name: "{row.last_name}", company: "{row.company}" },
  output_map: { email: "verified_email_column" }
}
```

**Cell agent** (mini-LLM per row, when reasoning is needed):
```
action: {
  type: "cell_agent",
  prompt: "Find this person's Twitter URL. Try web_search for their name + company. Return null if they don't have one.",
  columns_to_fill: ["twitter_url"],
  per_row_credit_cap: 5
}
```

Prefer deterministic when the row maps cleanly to one tool call. Use cell agent when answer requires search, judgment, or chaining.

# Cell agent toolset (when type=cell_agent)

The cell agent runs per row with these tools, bounded by `per_row_credit_cap`:

- `fullenrich_enrich_email` — name+company → verified email (~$0.055)
- `fullenrich_enrich_phone` — name+company → verified phone (~$0.55, ~10× email)
- `fullenrich_enrich_company` — domain → company fields
- `apollo_org_enrich` — domain → 57 fields (funding, dept headcount, tech stack)
- `google_maps_place_details` — place_id → details
- `apify_search_actors`, `apify_actor_details`, `apify_call_actor` (tight scope, maxItems 1-5)
- `web_search` — quick query for scouting
- `browser_use` — single session, bounded task
- `code_exec` — python in sandbox

Cell agent sees the full current row's data, the columns it needs to fill, the action prompt, available tools, and remaining budget.

# Refinement rules

**Refine only when there's a clear problem:**
- Mostly noise, zero results, wrong entity type (fetching)
- Multiple sample rows fail outright — hallucinated, wrong column, total miss (enrichment)
- Pricing wildly out of expected range

**NOT a clear problem:**
- Mostly right but you'd like cleaner
- Ranking isn't ideal
- 1-2 outputs look weird
- An enrichment finds X/10 when you'd expect Y/10 and X ≈ Y — that's working

**Cap: 3 refinements per turn.** If confident the query is right, skip refinement entirely. **Somewhat working is a win** — commit and let the user steer.

**"Not finding" ≠ "not working."** If an enrichment finds 6/10 Twitter accounts because some people don't have one, that's working. If it finds 1/10 when you'd expect 7/10, something's off — try once. Web sources (`web_harvest`, `browser_use`) don't need refinement — they're already filtered subagents.

# Budget

Per-row credit cap on enrichments. Rough rule of thumb:
- 1 credit — classification or extraction from already-present row data, no tools
- 5 credits — one tool call (web_search, FE enrich, simple lookup)
- 10 credits — multi-step research, browser_use, multi-tool cascade

These are caps, not targets. Lean generous. Don't talk to the user about cost — the UI handles it.

# Approvals

Some tool calls automatically prompt the user before running — running enrichments at scale, extending tables, destructive deletes. You call the tool normally; the system pauses, the user sees a card with the action and estimated cost, and approves or denies. The estimated cost is computed empirically from the previous batch's cost on this table/enrichment. You'll see the response in the tool result — proceed if approved, adapt if denied.

# Visibility

Each turn, the system auto-injects a project state block describing tables, columns, row counts, active filters, enrichments configured, last fetch returned, and column class breakdowns where relevant. This is your situational awareness — refer to it rather than calling tools to re-discover what already exists.

# Voice

Speak in the user's vocabulary. They see a table and what's in it; they don't see source mechanics, schemas, or internal state. Don't reference Apify actors, input schemas, or other internals.

# Decision flow (rough)

1. Understand what the user wants. If the ask is vague enough to risk wasted effort, clarify; otherwise just go.
2. Get an internal rough read on scope size. Pivot to a proxy source if scope is too big.
3. Pick the source for the data shape. Scout briefly via `web_search` (~3-5 calls budget) if unsure what's out there.
4. Set up the first table. Add the most important enrichments. Run the first batch.
5. Hand back. The user iterates from there.

# Thin slice

In a first turn for a new project, aim for one well-set-up table with the most important enrichments configured and the first batch enriched. Don't try to build out all tables and all enrichments in one turn. The user reacts to what you ship and steers the next move.

For multi-source projects (Reddit + Quora + IndieHackers; or multi-vertical SMB lead-gen), build ONE table per turn unless the user signaled "do all of them up front."

# Filters

Filters are non-destructive — they surface a slice of the table without destroying data. You can set them proactively. Seeing the full scope is often valuable even when the user only cares about a slice ("15% of engineers work on Claude Code" is information even if user wants those 15%).

`row_delete` is for explicit user intent ("delete rows 12-15"), not for narrowing. If a row looks off, set a filter or enrich the determining column.

# Anti-patterns

- Optimizing past good-enough. If results are mostly right, commit.
- Predicting cost in dollars or warning the user that something is "expensive." The UI handles cost.
- Re-pulling a source you already covered with a different tool. If Apollo gave you a company, don't also web_harvest for it.
- Pre-deleting rows to narrow — filters do this without destroying data.
- Trying to be exhaustive in one turn — the user iterates with you.
- Refining when you're already confident.
- Asking the user to be sure / confirm what you just did. Just do it.

# Source filter cards

The query_params shape per source. The 80%-common filters listed here; for rare filters call `source_describe` or look at input_schema for Apify actors.

## apollo_companies
```
organization_locations: ["San Francisco", "California"] | exclude via organization_not_locations
organization_num_employees_ranges: ["11,50", "51,200"]
revenue_range: {min: 1000000, max: 50000000}
q_organization_keyword_tags: ["artificial intelligence", "developer tools"]
currently_using_any_of_technology_uids: ["aws", "react", "salesforce"]
latest_funding_amount_range: {min: 1000000, max: 100000000}
latest_funding_date_range: {min: "2024-01-01", max: "2026-01-01"}
total_funding_range: {min, max}
q_organization_job_titles: ["DevOps Engineer", "VP of Sales"]    # active hiring signal
organization_num_jobs_range: {min: 3, max: 50}                    # active hiring signal
q_organization_domains_list: ["anthropic.com", ...]               # up to 1000
page: 1, per_page: 100
```

## fullenrich_people
```
job_titles: ["VP Sales", "Head of Compliance"]
seniorities: ["c_suite", "vp", "director", "manager", "senior"]
person_locations: ["California", "United States"]
company_names: ["Anthropic", "OpenAI"]
company_industries: ["Software Development"]
company_headcounts: [{min: 50, max: 500}]
company_locations: ["San Francisco"]
contact_email_status: ["verified", "likely to engage"]
limit: 100, offset: 0, search_after: <cursor>
```

## google_maps
```
query: "flooring contractor"
location: "San Diego, CA"        # or a polygon via customGeolocation
radius_miles: 25                  # optional
min_rating: 4.0                   # optional
max_review_count: 50              # optional
n: 100                            # server subdivides spatially if > 60
```

## apify_actor:<actor_id>
```
input: { ... }                    # shape is actor-specific
                                  # call apify_actor_details(actor_id) first to see input_schema
                                  # server auto-fills plumbing (proxy etc.) before validation
```

## web_harvest
```
query: string                     # what to look for
candidate_description: string     # what a successful row looks like
max_candidates: 30                # default
max_turns: 6                      # default; how many search/page iterations the subagent gets
```

## browser_use
```
url: string                       # starting page
task: string                      # very specific extraction task, vertical, one site
candidate_description: string
```
No item cap — bound by the task prompt scope. ~50 navigation actions per session before reliability degrades; break larger jobs into multiple bounded sessions.

## file
```
file_id: string                   # from upload OR from code_exec writing a file
```
Server auto-detects CSV/XLSX. Other formats → error with hint to use `code_exec`.

# Pagination per source (rules of thumb)

- **FullEnrich, Apollo, Google Maps**: prior response returned a cursor / page / token. Pass it back in next `table_extend`'s query_params. Server runs same query at the cursor.
- **Apify (time-indexed actors — Reddit, Twitter, Indeed, etc.)**: read project state for oldest-seen-date in this table. Pass next query_params with the actor's date-filter set to that date (look at the actor's input_schema field for the right name: `postDateLimit`, `until`, `before_date`, etc.). Each call is a fresh actor run; cost applies.
- **Apify (non-time-indexed)**: source is one-shot. To get more, create a new table with a different angle (different starting URL, different keyword, different category).
- **web_harvest / browser_use**: pass a `continuation_hint` describing a distinct sub-angle. Each call costs a new subagent / session.

# Dedup

For each table, pick `dedup_key_column` at create time. Rule of thumb: pick an ID-shape or URL-shape column (post_id, place_id, tweet_id, linkedin_url, email). **Never** name or title. If no obvious unique key exists, leave dedup off.

# Cost notes (rules of thumb)

- Apollo company search: ~$0.001/row — free in practice
- Apollo org enrich: ~$0.022/company
- Apollo job postings: ~$0.022/company (returns up to 10k jobs)
- FullEnrich search: ~$0.014/match
- FullEnrich verified email: ~$0.055
- FullEnrich verified phone: ~$0.55 (10× email)
- Google Maps place details: ~$0.017/place
- Apify per-item: actor-specific, ranges $0.001-$0.01 typically
- web_harvest: ~$0.10-0.30 per subagent run (yielding 5-30 candidates)
- browser_use: ~$0.10-0.50 per session
````

---

## 2. Tools

Two surfaces — orchestrator (chat agent) and cell agent (per-row enrichment runs). Both share many of the same underlying integrations; different contexts.

### 2.1 Orchestrator (15 tools)

#### `table_create`
Create a new table backed by one source query and fetch the first batch.

```yaml
params:
  name:         string
  source:       SourceEnum               # see Source enum
  query_params: object                   # source-specific shape (see filter cards)
  n:            integer (default 100)    # natural size cap
returns:
  table_id:               string
  rows_initial:           integer        # actual rows committed
  source_schema_preview:  object         # only for unpredictable sources; first ~5 rows + field list
  exhausted_first_batch:  boolean        # true if source returned fewer than requested AND can't yield more
approval_required: false                 # bounded to n=100 default
```

#### `table_extend`
Run another query against the same source, appending rows.

```yaml
params:
  table_id:     string
  query_params: object                   # the next slice — agent picks based on project state
  n:            integer (default 100)
returns:
  rows_added:   integer
  exhausted:    boolean                  # last fetch returned 0 — soft indicator
approval_required: always                # user sees empirical cost estimate
```

#### `table_delete`
```yaml
params: { table_id: string }
returns: { ok: boolean }
approval_required: always
```

#### `apify_search_actors`
Discover actors. Returns lightweight summary per actor — full schemas via `apify_actor_details`.

```yaml
params: { query: string }
returns:
  actors:
    - actor_id, title, short_description, monthly_run_count, rating, pricing_summary
approval_required: false
```

#### `apify_actor_details`
Full input_schema + output preview + pricing for one actor.

```yaml
params: { actor_id: string }
returns:
  actor_id, title, description, input_schema, output_schema_sample, pricing_full
approval_required: false
```

#### `column_map_set`
Commit (or update) the field→column mapping for a table. Used after seeing source_schema_preview from `table_create` on unpredictable sources, OR to rename/retype columns later.

```yaml
params:
  table_id:           string
  mapping:            { source_field: { column_name, type } }
  dedup_key_column:   string?              # optional, sets the unique key
returns: { ok: boolean }
approval_required: false
```

**Flow for unpredictable sources** (apify_actor, web_harvest, browser_use):
1. `table_create(source, query_params)` → server fetches first ~10 rows synchronously, returns `source_schema_preview` (rows + raw field list). Table is in "schema_pending" state; rows NOT yet committed to the visible table.
2. Agent inspects the preview, calls `column_map_set` to specify field→column mapping + types + dedup_key.
3. Server applies mapping to the 10 preview rows + continues fetching the remaining ~90 in the background with mapping applied. Rows stream into the visible table.

**Flow for predictable sources** (apollo_companies, fullenrich_people, google_maps, file with CSV headers): server has a default column map. `table_create` commits all rows immediately with the default map. No `column_map_set` call needed unless the agent wants to rename/retype later.

#### `enrichment_set`
Define or refine an enrichment. Runs on the table's first 10 unfilled rows.

```yaml
params:
  table_id:           string
  enrichment_id:      string?              # if present, refines existing
  name:               string
  columns:            [{ name, type }]
  action:             object               # type=tool or type=cell_agent (see prompt)
  per_row_credit_cap: integer
returns:
  enrichment_id, rows_filled, results_preview
approval_required: false                   # bounded to 10 rows
```

#### `enrichment_run`
Extend an enrichment to more rows.

```yaml
params:
  enrichment_id: string
  scope:
    type:         "first_n" | "all_unfilled" | "row_ids" | "filter"
    first_n:      integer?
    row_ids:      string[]?
    filter:       object?
  overwrite:    boolean (default false)
returns:
  job_id, rows_queued
approval_required: always
```

#### `filter_set`
Apply a non-destructive filter. Returns matched count + sample so agent can sanity-check.

```yaml
params: { table_id, column, op, value }
returns: { matched, total, sample }
approval_required: false
```

#### `filter_clear`
Remove a filter from a column.

```yaml
params: { table_id, column }
returns: { ok }
approval_required: false
```

#### `row_inspect`
Read-only peek at rows.

```yaml
params: { table_id, filter?, n: default 10, sort_by? }
returns: { rows }
approval_required: false
```

#### `row_delete`
Delete rows by explicit ids or filter.

```yaml
params: { table_id, row_ids? | filter? }
returns: { rows_deleted }
approval_required: always
```

#### `code_exec`
Python in sandbox. Helpers: `from dsl_tools import add_rows, get_table, list_tables`.

```yaml
params: { code, files?: string[] }
returns: { ok, stdout, stderr, duration_ms }
approval_required: false
```

#### `web_search`
Quick lookup. Returns top web search results.

```yaml
params: { query }
returns: { results: [{title, url, snippet}] }
approval_required: false
```

#### `suggest_replies`
Emit chip suggestions for user's next move.

```yaml
params: { chips: [{ label, message }] }
returns: { ok }
approval_required: false
```

### 2.2 Cell agent (11 tools)

Per-row enrichment. Bounded by per_row_credit_cap.

- `fullenrich_enrich_email(first_name, last_name, company)` → verified email + metadata
- `fullenrich_enrich_phone(first_name, last_name, company)` → verified phone (~10× email cost)
- `fullenrich_enrich_company(domain)` → company fields
- `apollo_org_enrich(domain)` → 57 fields including funding events, dept_headcount, tech stack
- `google_maps_place_details(place_id)` → full place
- `apify_search_actors(query)`
- `apify_actor_details(actor_id)`
- `apify_call_actor(actor_id, input)` — tight scope; maxItems 1-5 for per-row
- `web_search(query)`
- `browser_use(url, task)` — single session
- `code_exec(code, files?)`

Cell agent has implicit context: full current row, column(s) to fill, prompt, remaining budget.

---

## 3. Source enum

```
apollo_companies
fullenrich_people
google_maps
apify_actor:<actor_id>
web_harvest
browser_use
file
```

Per-source `query_params` shapes documented in the system prompt's filter cards section. Server validates query_params per source; errors include actionable hints.

---

## 4. Column types

7 types, soft (no errors on mismatch — server stores any value).

| Type | Side effects |
|---|---|
| `text` | Default. Substring filter operators. |
| `number` | `<`, `>`, `between` operators. Histogram in UI. |
| `url` | Renders as clickable link. (Optional URL verification later; metadata `url_status`.) |
| `email` | **Auto-fires Scrubby verification server-side** on filled cells. Verification badge in UI. Filter operator `verification_status equals valid|risky|bounced|unverified`. |
| `date` | Date filter + picker UI. |
| `bool` | Yes/No filter, checkbox UI. |
| `enum` | Auto-detected when column has <20 unique values. Class breakdown in project state banner. Value-picker filter UI. |

(`phone` dropped — phones stored as text. FE/Apollo phone enrichments are pre-verified at-source.)

Cell metadata per cell:
- `value` — the cell's value
- `status` — `filled` | `not_found` | `failed` | `pending`
- `metadata` — type-specific, e.g. `verification: valid|risky|...` for email

---

## 5. Filter operators per type

| Type | Operators |
|---|---|
| text | equals, not_equals, contains, not_contains, starts_with, ends_with, regex, is_null, is_not_null |
| number | equals, not_equals, <, >, <=, >=, between, is_null, is_not_null |
| url | equals, not_equals, contains, is_null, is_not_null |
| email | equals, not_equals, contains, **verification_status**, is_null, is_not_null |
| date | equals, <, >, between, is_null, is_not_null |
| bool | equals, is_null, is_not_null |
| enum | equals, not_equals, in, not_in, is_null, is_not_null |

Plus cell-status filter (orthogonal to type): `cell_status equals filled|not_found|failed|pending`.

---

## 6. Project state banner

Auto-injected before every user message. Token cost ~200-1000 tokens for typical projects.

```xml
<project_state>
Today is {{TODAY}} ({{TODAY_WEEKDAY}}).

Tables:
  - reddit_pain_points (47 rows, source: apify_actor:clearpath/reddit-search-scraper)
      Query: { searchTerms: ["AI automation"], sort: "Latest", maxItems: 100 }
      Columns: title (text), subreddit (enum), author (text), posted_date (date), url (url), pain_signal (enum)
      Class breakdown — pain_signal: clear_need (28), exploring (12), off_topic (7)
      Filters: posted_date > 2026-04-01 (matches 47/47)
      Last fetch: 47 rows, 12 credits, 5 min ago. Oldest seen: 2026-04-12.
      dedup_key_column: post_id

  - anthropic_engineers (43 rows, source: apollo_companies)
      Query: { q_organization_keyword_tags: ["AI"], organization_locations: ["San Francisco"] }
      Columns: name (text), domain (url), employees (number), revenue (number), phone (text), email (email)
      Email column: 38 filled (verified: 30, risky: 5, bounced: 3), 5 pending
      Last fetch: 43 rows, 2 credits, 1 hr ago.

Enrichments configured:
  - id=e1 "Verified email for VPs" on anthropic_engineers
      Fills: vp_email (email)
      Cap: 8 credits/row
      Last run: 10 rows filled (8 verified, 2 bounced), 75 credits

Filters active across project: 1 (see per-table above)
</project_state>
```

---

## 7. Approvals

Server intercepts gated tool calls. Always-gated:
- `table_extend`
- `enrichment_run`
- `table_delete`, `row_delete`

When called, server pauses run, emits `approval_required` event with empirical cost preview. FE renders card; user approves or denies. Server resumes with tool result.

Cost preview = `n × (last_fetch_cost / last_fetch_rows)` for the target table or enrichment. Falls back to coarse range when no prior batch exists.

Card content:
- Action description ("Fetch 100 more rows from Reddit pain points")
- Estimated cost ("~5 credits")
- **Approve** / **Cancel**

If denied: agent gets `{ok: false, denied: true, reason: "user_cancelled"}` and adapts.

---

## 8. Cost tracking (empirical)

Per table:
```sql
last_fetch_returned_rows  integer
last_fetch_cost_credits   numeric
last_fetch_at             timestamptz
```

Per enrichment:
```sql
last_run_filled_rows      integer
last_run_cost_credits     numeric
last_run_at               timestamptz
```

Server updates atomically when a fetch/run completes. Approval cards read these for cost preview. No precomputed pricing tables; adapts automatically as Apify pricing changes etc.

---

## 9. Schema changes

### `tables`
```sql
id                        uuid pk
project_id                uuid fk
name                      text
source                    text                  -- enum value
query_params              jsonb
columns                   jsonb                  -- [{name, type, source_field?}, ...]
dedup_key_column          text                  -- nullable
last_fetch_returned_rows  integer
last_fetch_cost_credits   numeric
last_fetch_at             timestamptz
created_at                timestamptz
deleted_at                timestamptz
```

### `enrichments`
```sql
id                        uuid pk
table_id                  uuid fk
name                      text
columns                   jsonb                  -- [{name, type}, ...]
action                    jsonb                  -- {type, tool|prompt, args_template?, ...}
per_row_credit_cap        integer
last_run_filled_rows      integer
last_run_cost_credits     numeric
last_run_at               timestamptz
created_at                timestamptz
deleted_at                timestamptz
```

### `filters` (per-table active filters)
```sql
id            uuid pk
table_id      uuid fk
column_name   text
op            text
value         jsonb
created_at    timestamptz
```

### `samples` (existing — extend)
Add: `table_id uuid fk` (re-keyed from project_id). Cell-level value + status + metadata stored inline in the `row` jsonb (one map per row keyed by column_name → `{value, status, metadata}`).

### Migration
Existing single-table projects: on first chat after upgrade, create a "Main" table per project. Re-key all existing samples to point at it. No data loss.

---

## 10. Approval allowlist

```yaml
table_extend:    always
enrichment_run:  always (any scope)
table_delete:    always
row_delete:      always
```

Initial calls (`table_create` default n=100, `enrichment_set` first 10) NOT gated — bounded by design.

---

## 11. Exhaustion (soft, not hard)

No hard "table is exhausted" state. Always lets user click "Fetch more."

When last fetch returned 0 new rows:
- Soft UI indicator next to table name ("last fetch: 0 new rows")
- Friendly chat message after the fetch ("Got 0 new rows — source may be tapped out for this query, want me to try a different angle?")
- Button stays clickable; agent gets the 0-rows signal in tool result

No blocking. User has agency.

---

## 12. Streaming

For sources where fetch is incremental (Apify dataset polled during run; web_harvest yielding mid-subagent; FE/Apollo paginating internally): server streams rows into the table as they land. FE updates live.

`table_create` returns synchronously with first ~5 rows + table_id. Rest flow into background. Agent doesn't block.

---

## 13. Open Items / Defer

- Cross-table references (Clay-style formula columns) → v2
- Skills → not pre-authored; emerge from observed repeated patterns
- Per-table versioning → stay with per-project for now
- URL verification on `type=url` columns → v2
- Project-level budget cap mode → not in v1; each operation gets its own approval card
- Apify per-actor input adapters → not in v1; agent learns from input_schema, server validates with helpful errors

---

## 14. Testing plan

- Set up eval harness with the 7 ICP partner projects + 3 fresh fixtures (CSV upload, manual-table-via-code, proxy pivot)
- Drive end-to-end as if I were the user
- Judge the system, not individual projects (3+ failures of same kind = system issue; 1 = accept)
- Report system-level findings only, no per-project prompt patches
