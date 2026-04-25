# V-Next Handoff

**Read this first.** You are continuing work on a major architectural rewrite
of DatasetLabs's worker. Most of the design is already locked in. Your job is
to iterate on the prototype on the `vnext` branch and either tune it until
it feels right or extend it (sources, per-cell agents).

This file is what the previous Claude Code session would tell you over a beer.
Read it top to bottom before touching code.

---

## 1. The product, in 90 seconds

DatasetLabs lets users describe a dataset in natural language and the system
builds it. Today's main use case is **lead generation** — "find me 100
founders of B2B SaaS companies, 2-20 people" — but we keep the system
general enough to scrape Upwork jobs, Reddit posts, etc. We compete with
Clay/Apollo on UX (we accept arbitrary natural-language requests; they only
support pre-defined filters).

Three repos:

| Repo | Path | Stack | What it does |
|------|------|-------|--------------|
| **frontend** | `/home/user/datasetlabs/frontend/` | Vite + React + TS | UI. Deployed to Cloudflare Pages at datasetlabs.ai |
| **api** | `/home/user/datasetlabs/api/` | FastAPI + Postgres (Supabase) | HTTP endpoints. The chat "waiter" LLM (talks to user, sets columns/num_samples, calls Service Bus to wake worker) |
| **worker** | `/home/user/datasetlabs/worker/` | Python async, OpenAI Responses API, Azure Service Bus | The agentic core — generates rows. Deployed as Azure Container App `datasetlabs-worker-prod` |

The three are loosely coupled via Postgres + Azure Service Bus + Azure Blob.
API never blocks on worker.

---

## 2. What's in production right now

- **API**: Supabase Postgres (dev URL: `db.kimyhpfvictnunsxsptg.supabase.co`).
  The "chat waiter" LLM uses OpenAI Responses API. Tools today:
  `set_columns`, `set_num_samples`, `present_summary`, `ask_questions`.
  See `/home/user/datasetlabs/api/dsl_api/routers/chat.py`.
- **Frontend**: live at datasetlabs.ai via Cloudflare Pages. Deployed from
  `main`. Most recent additions: resizable chat sidebar, MS Clarity
  first-party proxy (`functions/c/[[path]].ts` — defeats domain-list ad
  blockers, NOT URL-pattern ones), SEO prerender script.
- **Worker (V13)**: deployed as `datasetlabs-worker-prod` rev `--0000030`
  in Azure RG `datasetlabs-rg`. KEDA-scaled 0–10 replicas off Service Bus
  queue depth. Currently still serving all real users.
  - **Branch in prod**: `main` at commit `9f63373`
  - **Deploy command**: `cd worker && ./deploy.sh --production`
  - **Important env vars** (in `worker/.env.prod`): `OPENAI_API_KEY`,
    `LANGFUSE_*`, `BROWSER_USE_API_KEY`, `APOLLO_API_KEY`, `GOOGLE_API_KEY`,
    `BRAVE_API_KEY`, `AZURE_*`, `DATABASE_URL`. FullEnrich key is in
    config too (search for `fullenrich` in worker/.env.prod).

**Production worker still uses V13** (the orchestrator-pipeline architecture).
The `vnext` branch is parallel work that does NOT touch prod yet.

---

## 3. Why V-next exists — V13's failure modes

Background reading: `/home/user/datasetlabs/VNEXT.md` (the design doc).

We diagnosed V13 via Langfuse traces. Two specific failure patterns kept
showing up:

1. **Orchestrator overhead is huge.** On a representative trace
   (`005f9a5d6800c1ae4f56374551e90438`), 45 orchestrator-LLM turns produced
   exactly 1 row directly + delegated to row generators that produced ~5
   more. Total cost: $0.84. **69% of cost was orchestrator overhead** vs
   actual row production work. Orchestrator was workshopping rows itself
   *and* spawning row generators in parallel — two redundant pipelines.

2. **Per-candidate web_search paranoia.** When schema demands a "Founder
   LinkedIn URL" and FullEnrich returns mostly CEO titles at small startups,
   the orchestrator doesn't trust "CEO of 13-person company == founder" and
   fires 15-20 OpenAI built-in `web_search` queries in a single LLM turn to
   verify each candidate. Each verify-batch is $0.20 and 3 minutes wall time.
   You can't rate-limit OpenAI's built-in tools per-turn via prompt.

Other systemic issues:
- The 40-turn orchestrator loop has no clean termination after `submit_candidates`. It keeps running forever in parallel with the row generators it spawned.
- "Workshop then delegate" forces the orchestrator to act as a row generator that then teaches its behavior to other row generators. Two jobs, one agent.
- Every prompt patch we tried (FE-first, Apify-first, stick-with-source, vary-FE-queries) only addresses surface symptoms. The root cause is structural: orchestrator-as-LLM driving a pipeline with tools will always lean toward more verification and hedging.

**Conclusion: prompt patches won't fix V13. Need an architectural rewrite.**
That's V-next.

---

## 4. V-next design — locked in

Read `worker/VNEXT.md` (already on `vnext` branch). Key shape:

- **Claude-Code-style chat agent.** User chats in natural language. Agent
  acts in small batches and asks before bulk operations. No upfront pipeline,
  no plan-approval card, no setup screen.
- **Per-project SQLite file** as the table substrate. Stored in blob storage,
  worker downloads + syncs. Real DB capabilities; isolated per project.
- **Flat tool surface.** No chained Python at the LLM level. Each tool is
  a discrete OpenAI function call. Internally a SQLAlchemy-style ORM does
  the work, but the LLM only sees flat tools with structured args.
- **Per-cell mini-agents** (NOT one big row-generator). When `rows_fill`
  runs on a column without a `direct_call`, it spawns a small bounded
  agent per cell with row context, source tools, turn cap, and a budget cap.
- **`direct_call` fast path.** Columns can declare a typed integration call
  (e.g. `fullenrich.enrich_email`) with template-substituted args. Pure
  code, no LLM per cell. Cheaper.
- **No formal migrations.** Versioning = SQLite file snapshots before
  destructive turns. User can `versions_checkout` to roll back.
- **No structured questions or approval cards.** Plain chat for everything.
  Agent asks "want me to run on the rest? ~$0.84 for 84 rows" in chat. User
  responds however.

**What dies vs V13:**

| V13 | V-next |
|---|---|
| Orchestrator LLM 40-turn loop | Chat agent, ≤2 tool calls per turn before pausing |
| "Workshop one row before delegate" bumper | Gone — small batches always |
| Row generators as separate agents | Per-cell mini-agents, ≤5 turns each |
| `submit_candidates` / `process_candidate` / `set_column` / `submit_row` / `skip_row` | `rows_add`, `rows_fill`, `rows_update` |
| `plan()` tool | Gone — agent works conversationally |
| `ask_questions` tool with structured options | Gone — casual chat |
| Pipeline-as-data-structure | None — table state IS the project; chat history is the audit log |
| Filter/skip during fill | Filter is a separate `rows_delete` after a classifier column |

**What survives:**
- Source tools (FE, Apollo, Apify, Google Maps, BU, web_harvest) — unchanged at the integration layer
- Cost tracker, Service Bus, Langfuse tracing — unchanged
- Rules-of-thumb prompt content (BU last resort etc.) — moves into cell agent prompt
- Dedup — moves to commit-time `merge_key` on `rows_add`

---

## 5. What's built on the `vnext` branch

Branch: `vnext`. Tip commit: `3a93c69` (this handoff is the next commit).
Lives entirely in `worker/dsl_worker/vnext/`. Does NOT modify any V13 code.

| File | LoC | What it does |
|------|-----|--------------|
| `db.py` | ~500 | SQLite schema (rows/columns/cell_meta/snapshots/turns), CRUD ops, Django-ish where-dict translation, file-level snapshots. **Tested.** |
| `tools.py` | ~480 | 12 flat tools wired to db.py: `rows_add/get/count/sample/update/delete`, `columns_add/list/modify/delete`, `versions_list/checkout`. Each has an OpenAI Responses API tool schema. Dispatch via `call_tool(name, args, ...)`. |
| `agent.py` | ~235 | `ChatAgent` class. Runs the OpenAI Responses API loop, executes tool calls, snapshots before destructive ops. Has a system prompt at `agent.py:26`. |
| `cli.py` | ~120 | REPL: `python -m dsl_worker.vnext.cli ./path/to/project.sqlite`. Auto-loads `.env.prod`. Renders tool calls + results inline. |

**End-to-end smoke tested with real OpenAI.** Agent successfully:
- Adds columns with format hints
- Adds rows
- Updates rows by `where` filter
- Deletes rows
- Renders the table in chat
- Self-corrects when it makes bad tool calls (e.g. forgets required args)

---

## 6. What's NOT built yet (next stages)

These are the stage-2 deliverables. None of them are blockers — the chat
behavior should be tuned first on the existing tools. Don't start these
until the user (Dylan) has played with the CLI and feels good about
behavior.

### 6a. Source tools

The flat tool surface needs to add tools that wrap each integration source:

```
fullenrich.search_companies(industries, headcount_min, headcount_max, ...)
fullenrich.search_people(titles, industries, seniority, ...)
fullenrich.enrich_email(first_name, last_name, company, domain, linkedin_url)
apollo.search_companies(...)
apollo.enrich_person(...)
google_maps.search_places(query, location)
apify.search_actors(query)
apify.call_actor(actor_id, args)
browser_use(task)
web_harvest(query)
```

Existing implementations live in `worker/dsl_worker/agents/integrations/`
and are already used by V13. Approach: write thin wrappers that:
1. Call the existing integration client
2. Write raw output to `workspace/candidates/{source}_{idx}.jsonl`
3. Return a sample (3-5 records) + the file path to the agent

Then the agent uses `code_exec` (a tool we don't have yet — see below) to
flatten the JSONL into rows and call `rows_add(items, merge_key=...)`.

### 6b. `code_exec` tool

For the nested-source-mapping case (Apify returns deeply-nested objects),
the agent needs to write a small Python snippet that flattens the JSONL
into a list of dicts and calls `rows_add`. Existing V13 has a sandbox
service at `worker/sandbox_service/` that does this — reuse it. The
sandbox exposes a `commit_rows(...)` function that wraps `rows_add` for
ergonomics.

### 6c. Cell mini-agent + `rows_fill`

When `rows_fill(columns=["X"], where={...})` runs, for each matching row
either:
- If column has `direct_call` → execute as pure code (template substitution
  → tool call → jq-like extract → cell write). No LLM.
- Else → spawn a small bounded subagent: row context as input, source tools
  available, turn cap (~5), budget cap (`column.max_cost`, default 0.15).
  System prompt with the rules-of-thumb (BU last resort, Apify for bulk-
  scrape, etc.). Agent calls a `set_value(value)` tool to commit the cell
  and exit.
- Track outcome in `cell_meta(row_id, column_name)`: `filled` / `null_legitimate`
  / `budget_exhausted` / `error`. Later retries can target specific status:
  `where={"_meta.X.status": "budget_exhausted"}`.

### 6d. Frontend integration (much later)

Eventually the chat agent moves out of the CLI and into the actual product.
But for now keep iterating in the CLI. Don't touch frontend until the
behavior is solid.

---

## 7. How to run / iterate

```bash
cd /home/user/datasetlabs/worker
source .venv/bin/activate
git checkout vnext
python -m dsl_worker.vnext.cli ./projects/test1.sqlite
```

`OPENAI_API_KEY` auto-loads from `worker/.env.prod`. Each project = one
SQLite file plus a sibling `<name>_snapshots/` dir.

In the REPL:
- Type messages naturally. Agent responds, executes tool calls inline (gray
  `> tool(args)` and `→ result` lines), then assistant text in cyan.
- `--quiet-tools` flag hides tool execution lines.
- `exit` / Ctrl-D to quit.

Because we don't have source tools yet, you can only:
- Add columns
- `rows_add` directly with literal data
- Read / count / sample / update / delete rows
- Snapshot + checkout

That's enough to iterate on the **chat behavior** — does the agent ask
appropriately, sample first, etc. Once that feels right, layer on sources.

---

## 8. Behavior to watch for + tune

The system prompt is at `agent.py:26`. When you see the agent doing
something dumb, edit it. Things to specifically watch for:

- **Lunging at the full task.** User says "find 100 founders" — does the
  agent try to add 100 rows in one shot, or does it start with 10 and ask?
  Should be the latter. (For now there's no source so it can't actually
  fetch — but the equivalent test is: when you say "add 100 fake rows
  X,Y,Z..." does it just do all 100 or does it sample first?)
- **Format hint adherence.** When you add a column with `format="range
  string like 10-15"` and ask the agent to set values, does it emit
  strings or integers?
- **Tool call chattiness.** Should be 1-2 tool calls per turn. If you see
  5+ in a single response, the prompt is letting it ramble.
- **Confirmation in chat.** Before any `rows_delete` it should say "this
  will delete N rows, confirm?" and wait for a reply. Currently this is
  loose — tighten in the system prompt as needed.
- **Self-correction.** If a tool call errors, agent should retry once
  with fixed args, not give up.
- **Stuck loops.** `max_turns = 12` in `ChatAgent`. If it ever hits that,
  the prompt is letting it spiral.

When tuning the prompt: keep it **short**. Long prompts dilute. Add concrete
rules with concrete examples. Reference the actual tool names.

---

## 9. Open design questions (still need a call)

These are flagged in `VNEXT.md` and were left for "we'll see when we get there":

1. **Default `max_cost` per column type.** Currently 0.15 for everything.
   Probably needs to vary: $0.05 for direct_call, $0.20 for research,
   $0.50 for hard scrape.
2. **`direct_call` template syntax.** Probably `{Column Name}` f-string-ish
   substitution. Not implemented yet.
3. **`direct_call` fallback to cell agent on null/error?** Unclear if useful.
4. **Snapshot retention policy.** Currently keeps all forever. Probably
   need to prune after N or after T days.

Don't try to answer these in code unless Dylan explicitly asks. Keep
designing as you go.

---

## 10. Operational stuff you'll need

- **Worker prod**: `datasetlabs-worker-prod` in Azure RG `datasetlabs-rg`.
  Deploy via `cd worker && ./deploy.sh --production`. Don't deploy without
  Dylan's explicit say-so.
- **Local worker run** (for V13 testing): `cd worker && source .venv/bin/activate && python -m dsl_worker.main`. Connects to Service Bus, picks up real jobs from Dylan's dev account.
- **Langfuse traces**: env vars in `.env.prod`. CLI: `npx langfuse-cli api traces get <id> --json`. Useful for diagnosing V13 behavior. Use sparingly — V13 is not the focus anymore.
- **Eval suite**: `worker/tests/eval/run_eval.py`. 8 fixtures, real APIs, no mocks. Each gets 50 credits = $5 (matches new-user free tier). Last run was 2026-04-16; results in `worker/tests/eval/results/<timestamp>/`. Most fixtures hit `insufficient_balance` — that's by design (it's measuring "what does a new user see for $5"). **Eval suite uses V13.** Don't try to run it against vnext — different code paths entirely.
- **Frontend deploy**: pushes to `main` auto-trigger Cloudflare Pages build. Don't push without coordination if there's risk.

---

## 11. Recent context (last 48 hours)

A few things shipped in the last couple sessions that you should know
about so you don't accidentally undo them:

- **Worker** (still on V13, branch `main`):
  - `252e00b` — Orchestrator now registers rows in DedupStore (was missing → row gens were duplicating orchestrator-direct rows)
  - `7bd4358` — Prompt: prefer FullEnrich for people/company lists, stick with working sources
  - `db38d4a` — Prompt: vary FE queries before web_harvest
  - `9f63373` — Reverted a brief web_search removal experiment (Dylan wanted it back)
  - `636a362` — Pipeline early-exit now keeps projects resumable (was incorrectly marking them succeeded with partial rows)
  - `abbf67b` — Cofounder's commit: post finish-reason to chat when orchestrator aborts with 0 rows
- **Frontend** (branch `main`):
  - Resizable chat sidebar (collapsible + draggable, button in app bar)
  - DataTable: hide "Get" enrichment button if row has no enrichment context
  - Project page: usage refreshes on poll tick (was stale until refresh)
  - SEO prerender script (puppeteer-based, retries flaky pages, tolerates small failures)
  - Microsoft Clarity first-party proxy at `/c/sub/*` via Pages Functions
  - Misc paddings / SharedProject width fix
- **Cofounder is "nlp"** — they push to `nlp/homepage-design` and merge to `main` periodically. Don't be surprised by their commits.

---

## 12. Conventions

- Don't run destructive git operations without asking (force push, hard reset, branch delete, etc.).
- Don't commit `.env.prod` or any keys. Already in `.gitignore` but be careful.
- Worker uses async Python heavily; SQLAlchemy with the sync API is fine for vnext.
- Prefer extending `vnext/` over modifying `agents/orchestrator_v13.py` or other V13 code. V13 stays frozen except for critical bugs.
- Dylan's style: terse messages, ok with profanity, prefers small commits with clear messages, dislikes over-engineering. If you're proposing 5 layers of abstraction, you're wrong.

---

## 13. What to do first

1. Pull and check out the `vnext` branch.
2. Run the CLI on a fresh SQLite. Type a few things. Read the system prompt
   at `agent.py:26` and skim `tools.py` so you understand the surface.
3. Read `VNEXT.md` for the design doc.
4. Wait for Dylan to tell you what's next. Likely "this thing it does is
   dumb, fix the prompt" or "ok let's add FullEnrich source." Don't start
   stage 2 (sources / cell agent) without his go-ahead.

If Dylan says "go" without specifics, start with the FullEnrich source
wrapper since it's the most-used integration and `direct_call` columns
for `enrich_email` are the simplest fast-path test case.

Good luck.
