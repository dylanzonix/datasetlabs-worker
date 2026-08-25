# Dataset Labs — worker

The agent that did the work behind Dataset Labs: describe a dataset in plain language, and agents
build the table, research each row, verify what they find, and hand back something you can act on.

This is the real product codebase from a company that shut down, not a curated sample. It is
published as-is. What follows is a map, including which parts are dead.

## Where the live code is

```
dsl_worker/chat/          the shipped system
  agent.py                the conversational agent that owns a project
  cell_agent.py           per-cell research agent, budget-aware, writes to a row dossier
  enrichment.py           column enrichment: scope resolution, batching, provider waterfalls
  approvals.py            approval cards for work that costs money
  chat_run_tasks.py       durable runs: leases, resume, replay after a worker restart
dsl_worker/sources/       data providers behind one namespaced tool interface
dsl_worker/skills/        per-pattern playbooks loaded on demand
                          ("how to find Reddit leads", "classify by LinkedIn company")
dsl_worker/billing/       per-call cost tracking and credit enforcement
dsl_worker/infra/         sandboxing, browser client, artifacts, verification
```

Two ideas in here worth keeping:

**Verification gates billing.** If a row's website, email or phone cannot be confirmed, the row
does not ship and the customer is not charged for the attempt. That constraint shaped the whole
enrichment path.

**Conclusions are not facts.** Each row accumulates a dossier as agents research it. An early
version let one run's conclusions enter the next run's evidence, so a wrong guess could harden
into a fact nothing downstream would question. Dossiers now separate what was observed from what
was concluded.

## What is dead

Left in place because the history is the honest version, but do not read these as current:

- `ARCHITECTURE_V13.md` — last true in April 2026, superseded by `dsl_worker/chat/`
- `ORCHESTRATOR_V12_DESIGN.md` — one architecture older than that
- `dsl_worker/agents/orchestrator_v13.py` — the previous orchestrator
- `dsl_worker/infra/candidate_pool.py` — multi-armed bandit source allocation: Thompson sampling
  over per-source Beta posteriors, weighted by cost per usable row. It worked. Only the retired
  orchestrator imports it.
- `VNEXT.md`, `HANDOFF.md`, `REDESIGN_SPEC.md`, `RECIPE_REDESIGN.md` — design notes at various
  dates, useful as history, not as documentation

The architecture changed twice while the product was looking for its market. These docs record
where it had got to at the time and were not rewritten afterwards.

## Not included

Evaluation fixtures, which held real contact data for real people, and anything
customer-identifying. Removed from the full history rather than deleted in a later commit.

## Setup

```
pip install -r requirements.txt
cp .env.example .env    # then add API keys
```
