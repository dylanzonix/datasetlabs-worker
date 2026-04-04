# V13 Architecture

## Overview

Three agents: Orchestrator, Web Research, Row Generator. No harvesters. File-based candidate flow. Blocking tool calls. Orchestrator has direct control over everything.

## Agents

### Orchestrator (gpt-5.4, long-lived)

Controls the entire job. Generates candidates via APIs/code/subagents, writes them to files, submits batches for processing, iterates based on feedback.

**Tools:**

| Tool | Blocking | Output |
|------|----------|--------|
| `code_exec(script)` | Yes (fast) | stdout |
| `web_research(query, candidate_description)` | Yes (~30s) | Summary + candidates written to file |
| `bu_extract(task)` | Yes (1-5min) | Summary + results written to file. Task describes what to do — include URL in the task text. This delegates to a BU cloud agent (a capable browser agent with anti-bot bypass, resi proxies, captcha solving). Use as last resort — most expensive option. |
| `submit_candidates(file, note, preset_fields, checkin_after)` | Yes (blocks until checkin) | Feedback report |
| `continue_processing(checkin_after)` | Yes (blocks until checkin) | Feedback report |
| `apollo_search(...)` | Yes (fast) | Summary + results written to file |
| `google_maps_search(...)` | Yes (fast) | Summary + results written to file |
| `finish(reason)` | No | Abort. ONLY when task is genuinely impossible — all feasible approaches exhausted, 100% certainty there's no viable path. |

All candidate-producing tools write results to `/workspace/candidates/` as JSONL. Orchestrator gets a summary (count, sample, file path) — never raw data in context.

**System prompt includes:**
- Role and tools description
- How to work (find candidates → prep → test batch → iterate)
- Cost awareness (minimize waste, start small, scale up — dollar amounts, not credits)
- Schema (columns with types/formats)
- Target row count
- Full conversation history (user messages with timestamps for date context)
- Uploaded files description

**System prompt does NOT include:**
- User credit balance or budget
- Internal architecture details
- Row generator instructions
- Prescriptive strategies

### Web Research Agent (gpt-5.4, per-query)

Searches the web for a query, opens pages, yields candidates. No BU. Returns summary + file.

**Tools:**
- `web_search(query)` — native web search
- `open(url)` — open and read a page
- `find(ref_id, pattern)` — search within a page
- `click(ref_id, link_id)` — follow a link
- `yield_candidate(data)` — appends to output JSONL file

**Prompt:** The query, what candidates should look like, instructions to search and open promising results (roughly 5-15 pages depending on what's needed), yield what's found. No BU — native web only.

### Row Generator (gpt-5.4, per-candidate)

Receives a candidate + note + preset fields. Produces one row or skips.

**Tools:**
- `set_column(name, value, sources)` — set a column value
- `submit_row()` — submit the completed row (works whether columns were pre-filled or manually set)
- `skip_row(reason)` — skip this candidate
- `mark_duplicate(reason)` — skip as duplicate
- `web_search(query)`, `open(url)` — for research
- `apollo_enrich(...)`, `google_maps_search(...)` — for enrichment
- `code_exec(script)` — if needed
- `bu_extract(task)` — BU browser, last resort

**Prompt includes:**
- Role, tools, how to work
- Schema (columns)
- Candidate values
- Pre-filled columns (already set via preset_fields mapping, dedup already checked on these — LLM can override if something looks wrong)
- Note from orchestrator (where candidates came from, what's trustworthy, what to look for, any heads up)
- Full conversation history (user messages with timestamps)
- Dedup warnings if similar rows exist

## Candidate Flow

```
/workspace/
  uploads/              ← user files (read-only)
  candidates/           ← all candidate-producing tools write here
    apollo_results.jsonl
    google_maps_seattle.jsonl
    reddit_startups.jsonl
    bu_abc123.json
```

All tools that produce candidates write JSONL to `/workspace/candidates/`. One JSON object per line, whatever fields the source provides. No enforced schema — candidates are messy, row gen normalizes them.

Orchestrator uses `code_exec` to inspect, filter, dedupe, sample, merge files as needed before submitting.

## submit_candidates

Native orchestrator tool call (NOT a sandbox function).

**Parameters:**
- `file` — path to JSONL file in /workspace/candidates/
- `note` — handoff briefing for row gen (where data came from, what's trustworthy, what to look for)
- `preset_fields` — mapping of `{schema_column: candidate_field}` to pre-fill row columns from candidate data
- `checkin_after` — how many processed candidates before returning feedback

**What happens internally:**
1. Worker reads the JSONL file
2. For each candidate: apply preset_fields mapping → pre-fill columns → run dedup check on pre-filled values
3. Start processing candidates (10 concurrent max)
4. Process up to `checkin_after + floor(checkin_after * 0.5)` candidates (the extra is an optimism buffer — if results are good, those extras aren't wasted time)
5. Block until `checkin_after` candidates are processed
6. Return feedback report

**Feedback report:**
```
Processed: 10/15 (5 in flight)
Rows: 7 | Skipped: 2 | Dupes: 1
Skip reasons: "org appears defunct", "no public contacts"
Avg cost: $0.08/row
Total project: $3.41 spent, 47/100 rows
```

## continue_processing

Native orchestrator tool call. Resumes processing remaining candidates from the last submitted file.

**Parameters:**
- `checkin_after` — next checkin threshold

Same optimism buffer. Blocks until checkin. Returns same feedback report format.

If orchestrator doesn't call continue_processing, the in-flight candidates (from the optimism buffer) finish but no new ones start. Remaining unprocessed candidates sit idle.

## Stopping

**Target reached:** Checked after each row gen completion. When `generated >= target`, cancel all in-flight processing. Pipeline stops. No extra orchestrator turn.

**Credits exhausted:** Same. Processing stops. Next feedback report says stopped.

**finish(reason):** Circuit breaker. Absolute last resort.

**User pause:** External signal. Processing stops, state persisted, resume picks up seamlessly.

## Checkpoint / Resume

**Persisted:**
- Orchestrator conversation (full message history)
- All files in /workspace/
- Generation stats (rows, skipped, dupes, errors, cost)
- Which candidates have been processed (per-file tracking)
- Dedup store state

**On resume:**
- Restore everything above
- Seed dedup store with existing rows from DB
- Discard any pending tool responses
- Orchestrator continues from its last completed turn
- No "you were paused" message, no special resume context, just seamless continuation

## dsl_tools in sandbox

Upload `/workspace/dsl_tools.py` to sandbox on session init. Real Python module, real imports.

Contains utility functions: `list_files()`, `file_info()`, `read_jsonl()`, `write_jsonl()`.

`submit_candidates` is NOT in dsl_tools — it's a native orchestrator tool. Sandbox does data manipulation, tools do actions.

## Integration tools (Apollo, Google Maps, etc.)

All follow the same pattern:
1. Orchestrator makes tool call
2. Worker calls the API
3. Worker writes results to `/workspace/candidates/{source}_{timestamp}.jsonl`
4. Worker returns summary to orchestrator (count, sample, file path)

Orchestrator never sees raw API responses in context. Just summary and file path.

Future Apify follows the same pattern.

## Implementation order

1. New orchestrator agent — new prompt, new tool set, blocking tool calls
2. submit_candidates + continue_processing — tool handlers, dispatcher wiring, feedback reports
3. Web research subagent — new agent with yield_candidate, file output
4. Row generator updates — preset_fields, note, dedup on pre-filled values
5. BU extract tool — wrap BU client, file download from BU file space
6. Integration tools rewrite — Apollo/Google Maps/YouTube write to files
7. dsl_tools module — upload to sandbox on init, real imports
8. Checkpoint/resume — workspace persistence, conversation restore
9. Remove old code — HarvesterAgent, old dispatcher streaming, old check-in loop
10. Run evals — compare V13 vs V12 baseline
