# V-Next Design

Working draft. Replaces the V13 orchestrator pipeline with a Claude-Code-style
chat agent that builds the dataset incrementally, with the user in the loop.

## Goal

The user describes what they want, the agent acts in small batches, reports
back, and asks before scaling. No big abstracted runs that consume credits in
the dark. No upfront pipeline. No orchestrator-vs-row-generator split.

## Core Principles

1. **User in the loop.** Default behavior: do a small batch (~10 rows or
   ≤$0.10 estimate), report, ask before bulk.
2. **Casual chat for everything.** No structured Question card. Confirmations
   are inline Yes/Cancel buttons attached to the agent's message.
3. **One agent.** No row generators. Per-cell mini-agents are short, bounded,
   spawned inline by `rows_fill` when a column needs judgment/research.
4. **Files for raw, SQLite for resolved.** Source tools write JSONL to
   workspace. Commits transform JSONL → rows in the project SQLite.

## Persistence

- **One SQLite file per project.** Stored in blob storage. Worker downloads
  on job start, syncs back at save points.
- **Tables:** `rows` (project columns), `columns` (column definitions),
  `cell_meta` (per-cell status/budget/error sidecar), `_versions` (snapshot
  pointers).
- **No formal migrations.** Schema lives in the SQLite.
- **Versioning = SQLite file snapshots** before destructive turns
  (`delete`, `update`/`fill` on ≥5 rows, `column.delete`). User can
  `versions_checkout` any prior snapshot. Linear history, no branching.

## LLM Tool Surface

Flat function-call tools — no chained Python at the LLM level. Internally the
worker uses a real SQLAlchemy ORM; the LLM sees flat tools.

**Sources** (write to file, no DB touch, no approval):

```
fullenrich.search_companies / search_people / enrich_email
apollo.search_companies / enrich_person
google_maps.search_places
apify.search_actors / call_actor
browser_use(task)            # last resort, prompted
web_harvest(query)            # subagent
```

**Rows:**

```
rows_get(where, limit, columns)
rows_count(where)
rows_sample(n)
rows_add(items, merge_key)                   # tiny commits
rows_fill(columns, where, max_cost, limit)   # cell agent OR direct_call
rows_update(where, values)
rows_delete(where)
```

**Columns:**

```
columns_list()
columns_add(name, format, description, direct_call=None, max_cost=0.15)
columns_modify(name, ...)
columns_delete(name)
```

**Versions:**

```
versions_list()
versions_checkout(snapshot_id)
```

**Escape hatch:**

```
code_exec(python)    # full ORM access; for nested commits or rare complex queries
```

## Column Definition

```jsonc
{
  "name": "Verified Email",
  "format": "lowercase email or null",
  "description": "Find the verified work email for this person.",
  "direct_call": {                       // optional fast path; if present, no LLM per cell
    "tool": "fullenrich.enrich_email",
    "args": {"first_name": "{Founder Name}", "company": "{Company}"},
    "extract": "result.most_probable_work_email.email"
  },
  "max_cost": 0.15
}
```

- **`direct_call` set** → fills run as pure code (template substitution → tool
  call → jq-like extract → cell write). Predictable cost, no LLM.
- **`direct_call` null** → fills spawn a cell mini-agent with row context, all
  source tools, turn cap (~5), budget cap (`max_cost`), and a system prompt
  with the rules-of-thumb (BU last resort, Apify for bulk-scrape, etc.).

Multi-column fills supported — `rows_fill(columns=["Founder Name", "Founder LinkedIn"])`
runs ONE cell agent per row that produces all listed columns in one shot.

## Cell Metadata

Sidecar `cell_meta(row_id, column_name)` per cell:

```
status: "filled" | "null_legitimate" | "budget_exhausted" | "error"
budget_used: float
last_error: string | null
last_attempt_at: timestamp
```

Targeted retries via where-clause:

```
rows_fill(
  columns=["Verified Email"],
  where={"_meta.Verified Email.status": "budget_exhausted"},
  max_cost=0.30
)
```

## Confirmation UX

- **Small actions** (≤10 rows, ≤$0.10): just do, then report.
- **Larger / destructive**: agent stops and asks in plain chat. No structured
  Yes/Cancel widget — the user replies however they want. Behavior is entirely
  in the system prompt.
- **Always sample first** for bulk fills/commits. Hard rule in system prompt.

## Worker Architecture

One chat agent loop:

1. Receive user message via Service Bus.
2. Download project SQLite from blob; mount.
3. Run agent turn(s); execute tool calls; spawn cell mini-agents inline as
   needed.
4. Snapshot before destructive turns.
5. Sync SQLite → blob.
6. Stream response back to chat.

No separate orchestrator. No row-generator agents. Cell mini-agents are
bounded and short-lived.

## What Dies vs V13

| V13 | V-next |
|---|---|
| Orchestrator LLM 40-turn loop | Chat agent — ≤2 tool calls per turn before pausing |
| Workshop "try one row before delegate" | Gone; agent always acts in small batches first |
| Row generators as separate agents | Per-cell mini-agents, ≤5 turns each |
| `submit_candidates` / `process_candidate` / `set_column` / `submit_row` / `skip_row` | `rows_add`, `rows_fill`, `rows_update` |
| `plan()` tool | Gone — agent works conversationally |
| `ask_questions` tool | Gone — casual chat |
| Pipeline as data structure | None — table state IS the project; chat history is the audit log |
| Filter/skip during fill | Separate `rows_delete` after a classifier column |

## What Survives

- Source tools (FE, Apollo, Apify, Google Maps, BU, web_harvest) — unchanged
- Cost tracker — unchanged
- Service Bus transport — unchanged
- Rules-of-thumb prompt content — moves into cell agent system prompt
- Dedup — moves to commit-time `merge_key` on `rows_add`

## Implementation Stages

1. SQLite schema + project-vs-blob sync.
2. ORM + flat tool implementations (Python class-per-tool, dispatched).
3. Cell mini-agent (bounded subagent spec, system prompt, budget enforcement).
4. Direct-call fast path (template substitution + jq-extract).
5. Chat agent (system prompt for do-then-ask, confirmation markers).
6. Frontend: inline Yes/Cancel buttons, chat tool result rendering.
7. Snapshots + version checkout.
8. New-projects-only rollout. Old V13 projects keep their pipeline path.

## Open / TBD

- Default `max_cost` per column type (rough numbers in dollars).
- `direct_call` template syntax — f-string–style with `{Column Name}` works.
- Snapshot retention policy.
- Whether `direct_call` columns can have a fallback to cell agent if the
  direct call returns null/error.
