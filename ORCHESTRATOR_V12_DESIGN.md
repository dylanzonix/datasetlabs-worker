# Orchestrator V12 Design

## Problem

Given a project description, target row count, and credits budget, produce the most
rows at acceptable quality for the least cost.

The orchestrator must learn as it goes — cost-per-row varies wildly by source and
project type, sources change mid-run (dry out, get blocked, degrade), and the optimal
strategy differs per project. The orchestrator observes, adapts, and optimizes.

## Architecture

### Core Loop

The orchestrator is async. Harvesters and row generators run in the background.
The orchestrator checks in periodically, sees a dashboard, makes decisions, and
goes back to sleep.

**Check-in intervals:**
- Pre-first-row: every $0.50 cost OR 60 seconds, whichever first
- Post-first-row: orchestrator sets its own interval (guided to be conservative)
- Structural events force immediate check-in: target reached, budget exhausted,
  all sources dead + pipeline empty

### Orchestrator Tools

- `create_harvester(source, description)` — creates and starts a harvester immediately
- `stop_harvester(source_id, reason)` — kills a harvester, remaining buffer still processed
- `inspect(source_id, [candidate], [step])` — drill down into hierarchy
- `finish(reason)` — end job early
- Built-in `web_search` — research what sources to try
- `code_exec` — peek at uploaded files, understand schemas

No `process()` — processing is automatic.
No `browse` — orchestrator doesn't need a browser.
No Apollo directly — Apollo is a harvester tool.
Dashboard is auto-injected at every check-in, not a tool.

### Dashboard (auto-injected at check-in)

```
PROGRESS
  Rows: 34/100 | Cost: $1.82/$10.00 | Avg: $0.054/row | Elapsed: 4m 12s

SOURCES                                          candidates  processed  pending  rows  skip  dupe  err  $/row   status
  harvest:0 "upwork: dataset"     [ws,bu,ce]          18        18        0      12     4     1    1   $0.031  active
  harvest:1 "upwork: lead list"   [ws,bu,ce]          25        20        5      16     2     5    0   $0.028  active
  harvest:2 "upwork: scraping"    [ws,bu,ce]           3         3        0       0     3     0    0     —     active (0 rows)
  harvest:3 "apollo: data cos"    [apollo]             40        32        8       6     0     0    0   $0.019  exhausted

COST BREAKDOWN (per source)
  harvest:0  harvest: $0.12 (ws:$0.02, bu:$0.08, llm:$0.02)  row_gen: $0.25  total: $0.37  → $0.031/row
  harvest:1  harvest: $0.08 (ws:$0.03, bu:$0.00, llm:$0.05)  row_gen: $0.37  total: $0.45  → $0.028/row
  harvest:2  harvest: $0.41 (ws:$0.01, bu:$0.38, llm:$0.02)  row_gen: $0.09  total: $0.50  → no rows
  harvest:3  harvest: $0.00 (apollo:$0.00)                    row_gen: $0.11  total: $0.11  → $0.019/row

OUTCOMES (since last check-in)
  Rows: +8 ($0.041/row this interval)
  Skipped: +3 (harvest:0 "posted 3 weeks ago" x2, harvest:2 "not a dataset job" x1)
  Dupes: +2 (harvest:1 candidates matched rows from harvest:0)
  Errors: +0

PIPELINE
  Buffer: 13 waiting | Row generators: 7/10 active | Rate: ~2.1 rows/min

HARVESTER REPORTS (latest per source)
  harvest:0: "Extracted 8 from page 2. Pagination shows 5 pages. Some old listings mixed in."
  harvest:1: "Found 12 lead-gen jobs. 4 already seen from harvest:0. Trying new keywords."
  harvest:2: "BU spent 180s, extracted 3. All devops roles. Query too broad."
  harvest:3: "(exhausted) Apollo returned 40 results, no more pages."
```

### Drill-Down (via inspect tool)

Hierarchy using existing conversation data:

```
Dashboard (all sources summary)
  └→ inspect("harvest:0")
       Harvester tool calls in order: name, cost, duration
       Candidates: outcome (row/skip/dupe/error), cost, reason
       └→ inspect("harvest:0", step=3)
            Full input/output of that tool call
       └→ inspect("harvest:0", candidate=5)
            Row generator tool calls: name, cost, duration
            └→ inspect("harvest:0", candidate=5, step=2)
                 Full input/output of that row gen step
```

Each level is truncated for context management. Orchestrator drills in when needed.

## Harvesters

### Philosophy

Harvesters are **slices** — one per query/approach/source, not per depth-level.
"Upwork: dataset" and "upwork: lead list" are two harvesters.
Page 1 and page 2 of the same search are NOT separate harvesters — that's depth
within one harvester's slice.

### Lifecycle

```
Harvester started (by create_harvester)
  → produces batch (agent loop until text response = natural boundary)
  → candidates enter source buffer
  → WAIT: candidates processed by row generators (backpressure)
  → batch results collected
  → results fed back as context for next batch
  → produce next batch (or exhaust)
  → ...until killed by orchestrator or source exhausted
```

### Batch Boundaries

A batch = everything you can get without taking a new navigational action.
- File: whole file = 1 batch
- List page via BU: one page = 1 batch
- API query: one query's results = 1 batch
- Web search discovery: one research pass = 1 batch (fuzzier)

Current run_batch() already captures this — agent loop runs until text response.

### Tool Loadout

Default: web_search + browse + code_exec + list_files.
All harvesters get the same tools by default. Don't overthink restrictions.
Apollo is also a harvester tool (not orchestrator-level).

### Backpressure

Harvesters cannot get ahead of processing. After producing a batch, the harvester
waits until all its candidates from that batch are processed. This:
- Prevents unvalidated sources from running up tabs
- Ensures every source gets evaluated before harvesting more
- Keeps buffers small and manageable

## Processing

### Automatic

Processing starts automatically as candidates appear in source buffers.
No orchestrator involvement needed.

### Round-Robin

Row generator slots (10 concurrent) are shared equally across active sources.
Dispatcher pulls candidates round-robin: one from source 0, one from source 1, etc.
No source can hog all slots.

### Row Generators

Same as current: one RowGeneratorAgent per candidate, parallel via semaphore.
No changes to row gen internals, dedup, or billing.

## Budget Hint

Orchestrator system prompt includes rough guidance:
- "Aim for ~$2 to produce the first row (exploration budget)"
- "Once you know cost-per-row, optimize. Compare sources against each other."
- "A source is expensive or cheap RELATIVE to other sources, not in absolute terms."

## What Changes vs Current

**Changes:**
- Orchestrator: async check-in loop instead of blocking tool calls
- create_harvester: starts harvesting immediately in background
- No process() tool: processing is automatic with round-robin
- New dispatcher: round-robin candidate feeding across sources
- Backpressure: harvesters wait for batch processing before continuing
- Dashboard: auto-injected status at each check-in
- inspect() tool: hierarchical drill-down
- stop_harvester() tool: kill a source
- finish() tool: end job early
- Orchestrator prompt: teaches the optimization game

**Unchanged:**
- Row generator internals (agents, dedup, tools)
- Billing / cost tracking
- Checkpoint / resume
- BU client (already updated with page-scoped instructions)
- Harvester agent internals (prompt, tools, batch loop)
- DedupStore
- Database / API layer
