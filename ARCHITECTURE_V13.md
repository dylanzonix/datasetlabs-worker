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
| `bu_extract(url, task)` | Yes (1-5min) | Summary + results written to file |
| `submit_candidates(file, note, preset_fields, checkin_after)` | Yes (blocks until checkin) | Feedback report |
| `continue_processing(checkin_after)` | Yes (blocks until checkin) | Feedback report |
| `apollo_search(...)` | Yes (fast) | Summary + results written to file |
| `google_maps_search(...)` | Yes (fast) | Summary + results written to file |
| `finish(reason)` | No | Ends the job. Only when task is genuinely impossible. |

All candidate-producing tools write results to `/workspace/candidates/` as JSONL. Orchestrator gets a summary (count, sample, file path) — never raw data in context.

**System prompt includes:**
- Role and tools description
- How to work (find candidates → prep → test batch → iterate)
- Cost awareness (minimize waste, start small, scale up)
- Schema (columns with types/formats)
- Target row count
- Full conversation history (user messages timestamped)
- Uploaded files description

**System prompt does NOT include:**
- User credit balance or budget
- Internal architecture details
- Row generator instructions
- Prescriptive strategies

### Web Research Agent (gpt-5.4, short-lived, per-query)

Searches the web for a query, opens pages, yields candidates. No BU. Scoped to ~5 pages. Returns summary + file.

**Tools:**
- `brave_search(query)`
- `open(url)`
- `find(ref_id, pattern)`
- `click(ref_id, link_id)`
- `yield_candidate(data)` — appends to output JSONL file

**Prompt:** The query, what candidates should look like, and basic instructions to search, open promising results, yield what's found. Short and tight.

### Row Generator (gpt-5-mini, short-lived, per-candidate)

Receives a candidate + note + preset fields. Produces one row or skips.

**Tools:**
- `set_column(name, value, sources)`
- `approve_row()` — fast path when all columns pre-filled and look good
- `submit_row()`
- `skip_row(reason)`
- `mark_duplicate(reason)`
- `web_search(query)`, `open(url)` — for research
- `apollo_enrich(...)`, `google_maps_search(...)` — for enrichment
- `code_exec(script)` — if needed
- `browse(url, task)` — BU, last resort

**Prompt includes:**
- Role, tools, how to work
- Schema (columns)
- Candidate values
- Pre-filled columns (already set, dedup already checked, LLM can override or approve)
- Note from orchestrator (source, context, what to expect)
- Full conversation history (timestamped user messages for ground truth)
- Dedup warnings if similar rows exist

### BU Agent (not a separate agent — just a tool call)

`bu_extract(url, task)` is a blocking tool on the orchestrator. Under the hood it runs a BU cloud session with the task description. After completion, downloads output files from BU file space to `/workspace/candidates/bu_{session_id}.json`. Returns BU's summary text to orchestrator.

The task should be tightly scoped: specific URL, specific extraction, smash and grab.

## Candidate Flow

```
/workspace/
  uploads/              ← user files (read-only)
  candidates/           ← orchestrator writes here
    apollo_results.jsonl
    google_maps_seattle.jsonl
    reddit_startups.jsonl
    bu_abc123.json
  (no processing/ or completed/ dirs — track state in memory)
```

All tools that produce candidates write JSONL to `/workspace/candidates/`. Format: one JSON object per line, whatever fields the source provides. No enforced schema — candidates are messy, row gen normalizes them.

Orchestrator uses `code_exec` to inspect, filter, dedupe, sample, merge files as needed before submitting.

## submit_candidates mechanics

```python
submit_candidates(
    file: str,              # path to JSONL file
    note: str,              # handoff briefing for row gen
    preset_fields: dict,    # {schema_col: candidate_field} mapping
    checkin_after: int,      # unblock after N processed
)
```

**What happens:**
1. Worker reads the JSONL file
2. For each candidate: apply preset_fields mapping → pre-fill columns → run dedup check on pre-filled values
3. Queue all candidates for row generation (10 concurrent max)
4. Additionally queue `floor(checkin_after * 0.5)` extra candidates beyond checkin_after (the optimism buffer — always, no LLM config needed)
5. Block until `checkin_after` candidates are processed
6. Build feedback report and return it

**Feedback report format:**
```
Processed: 10/15 (5 in flight)
Rows: 7 | Skipped: 2 | Dupes: 1
Skip reasons: "org appears defunct", "no public contacts"
Avg cost: $0.08/row
Total project: $3.41 spent, 47/100 rows
```

## continue_processing mechanics

```python
continue_processing(
    checkin_after: int    # next checkin threshold
)
```

Resumes processing remaining candidates from the last submitted file. Same optimism buffer. Blocks until checkin. Returns same feedback report format.

If orchestrator doesn't call continue_processing, the in-flight candidates (optimism buffer) finish but no new ones start. The remaining unprocessed candidates in the file sit idle until orchestrator either continues or submits a different file.

## Stopping

**Target reached:** After each row gen completion, check `generated >= target`. If met, cancel all in-flight processing, return to orchestrator with "target_reached" in next feedback report. No extra orchestrator turn needed — pipeline just stops.

**Credits exhausted:** Same pattern. Processing stops, next feedback report says "credits_exhausted." Orchestrator doesn't need to know about credits beforehand.

**finish(reason):** Circuit breaker for impossible tasks. Used only when orchestrator has exhausted all feasible approaches and there's no viable path forward.

**User pause:** External signal. Processing stops, state saved, resume picks up where it left off.

## Checkpoint / Resume

**What gets persisted:**
- Orchestrator conversation (full message history)
- All files in /workspace/ (candidates, uploads, everything)
- Generation stats (rows generated, skipped, dupes, errors, cost)
- Which candidates from which files have been processed (file path + line numbers or candidate IDs)
- Dispatcher state (what's pending, what's done)

**On resume:**
- Restore orchestrator conversation
- Restore workspace files
- Restore generation stats
- Seed dedup store with existing rows from DB
- Discard any pending tool responses (orchestrator picks up from last completed turn)
- Orchestrator continues naturally — it has its conversation, it has its files, it knows where it was

No "you were paused" message. No special resume context. Just restore state and let it continue. If it was mid-submit_candidates, the pending candidates get re-queued. If it was mid-code_exec, it just gets a fresh turn.

## dsl_tools in sandbox

Upload `/workspace/dsl_tools.py` to sandbox on session init. Real Python module, real imports.

```python
# /workspace/dsl_tools.py
"""Tools available in the code execution sandbox."""

import json
import os
from pathlib import Path

WORKSPACE = "/workspace"

def list_files(directory="all"):
    """List files in workspace."""
    ...

def file_info(path):
    """Get file metadata."""
    ...

def read_jsonl(path):
    """Read a JSONL file, return list of dicts."""
    ...

def write_jsonl(path, data):
    """Write list of dicts to JSONL file."""
    ...
```

Note: `submit_candidates` is NOT in dsl_tools. It's a native orchestrator tool call. The sandbox just preps files. Clean separation: sandbox does data manipulation, tools do actions.

## Integration tools (Apollo, Google Maps, etc.)

All follow the same pattern:
1. Tool call from orchestrator
2. Worker calls the API
3. Worker writes results to `/workspace/candidates/{source}_{timestamp}.jsonl`
4. Worker returns summary to orchestrator ("Found 23 results. Written to /workspace/candidates/apollo_seattle.jsonl")

Orchestrator never sees raw API responses in its context. Just the summary and file path.

Future Apify integration follows the same pattern — tool call, API, results to file, summary back.

## Implementation order

1. **New orchestrator agent** — new prompt, new tool set, blocking tool call model
2. **submit_candidates + continue_processing** — tool handlers, dispatcher integration, feedback reports
3. **Web research subagent** — new agent with yield_candidate tool, file output
4. **Row generator updates** — approve_row, preset_fields, note from orchestrator
5. **BU extract tool** — wrap existing BU client, file download from BU file space
6. **Integration tools rewrite** — Apollo/Google Maps/YouTube write to files
7. **dsl_tools module** — upload to sandbox, real imports
8. **Checkpoint/resume** — workspace persistence, conversation restore
9. **Remove old code** — HarvesterAgent, dispatcher streaming, harvester loop, old check-in system
10. **Run evals** — compare V13 baseline against V12 baseline
