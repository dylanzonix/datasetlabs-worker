# Dataset Recipe Redesign — Queries, Enrichments, Filters

## Background and motivation

datasetlabs is a chat-driven dataset-generation platform. Users submit
prompts ("find me 50 VP Sales at B2B SaaS companies", "create a math SFT
dataset", "scrape posts from this Reddit thread"), and today the system
relies on relatively open-ended agentic exploration to produce rows.

That open-endedness has two costs:

1. **Inefficiency.** The agent re-derives the workflow per request — what to
   search, where to enrich from, how to filter — even when the underlying
   shape repeats across users. Lots of orchestration is duplicated.
2. **Opacity.** The user can't preview *what they're about to get* before
   tokens are spent. Provenance of each row is fuzzy. Cost is unpredictable.

The hypothesis we wanted to test: every dataset request can be expressed as
a fixed **3-slot recipe**, executed deterministically once filled. If the
recipe covers the space, we replace open-ended agentic exploration with a
structured pipeline that is faster, cheaper, and inspectable end-to-end.

## The original recipe (the hypothesis)

```
QUERIES       iterable, source-isolated searches.
              Each query binds to ONE external source (Apollo, Google Maps,
              Reddit, LinkedIn, web_search, etc.). It returns a sub-table of
              candidates. Multiple queries union into the full candidate pool.

ENRICHMENTS   per-candidate targeted lookups. Each adds a column to the
              candidate row. Same enrichment can apply across candidates
              sourced from different queries.

FILTERS       predicates applied to enriched rows. Drop rows that don't
              meet criteria.
```

Mental model: queries define the rows, enrichments define the columns,
filters define the keep-or-drop. Output is one flat table.

The point of constraining to these three slots is operational:
- Candidates are explicit, so users can audit them before paying for
  enrichment.
- Enrichment cost is linear in `|candidates| × |enrichments|`, so it's
  budgetable in advance.
- Filtering is the cheap final step that determines deliverables.

## Empirical analysis we ran

To stress-test the recipe, we scraped the first user message of every
project in our dev + prod databases (~256 unique messages after dedup
across both environments), then asked an LLM to decompose each one into
the 3 slots and rate the fit. Categories used:

- `lead_gen` — people/companies to contact
- `research_dataset` — factual rows from real external sources (news,
  transcripts, listings, posts)
- `synthetic_generation` — LLM-generated training data with no external
  candidate source
- `invalid` — chitchat / test / not a dataset request

Fit scores: 0 (N/A), 1 (wrong shape), 3 (works but loses nuance),
5 (clean map). Output saved to `scripts/breakdown_first_messages.csv`.

### Initial findings

| Category               | Count | Fit |
|------------------------|-------|-----|
| lead_gen               | 129   | mostly 4–5 |
| research_dataset       | 54    | mostly 4–5 |
| synthetic_generation   | 39    | mostly 0   |
| invalid                | 34    | 0          |

Clean fit (score 4–5): 163 / 256 = **64%**
No fit (score 0, valid): 39 / 222 valid = **~18%**
Partial fit (score 1–3): 21 / 222 valid = **~9%**

So out-of-the-box: about **73% clean coverage** of valid requests.

### Where the gaps were

**Gap 1: synthetic generation (39 rows).** Pure LLM-generated training data:
"Create a roleplaying dataset", "Create a dataset for math (GSM8K / AMC /
AIME style)", "Generate a 100k-row Jarvis SFT", "DayZ pro player tips",
multiple Polish legal SFT requests. The recipe assumed an external
candidate source; these have none.

**Gap 2: 21 partial fits.** Three sub-patterns emerged:

- **2.1 Single-target lookups.** "Find LinkedIn for Brett in Texas who works
  for the government, stationed in Istanbul." The analyzer flagged these as
  "wrong shape, recipe is table-oriented." Eight rows like this.
- **2.2 Inferential / strategic.** "Who's paying mercor.com",
  "find my ICP and outreach channels", "find me customers for
  https://...". The analyzer worried that enrichments here return
  probabilities rather than hard facts.
- **2.3 Underspecified.** "Show me vertical saas in healthcare", "create a
  dataset about cars and all their specs", "create a dataset for
  manufacturing — CNC, purchasing, logistics". The analyzer noted that the
  table schema isn't decided.

## Reasoning through each gap

### Gap 2.1: not actually a gap, just bad scoring

"Find Brett in Texas, in the government, stationed in Istanbul" decomposes
cleanly:

```
query:        linkedin_search → people named Brett in Texas
enrichments:  works_for_government   (LLM + profile inspection)
              stationed_in_istanbul  (LLM + profile inspection + web)
filter:       both flags must be true
```

Result table happens to contain one row. "Single-target" isn't a different
recipe shape — it's a query whose filter set is so restrictive that the
output is small. The recipe accepts iterables of size 1 just fine. The
analyzer conflated "narrow output" with "wrong primitive." **Recipe fits.**

### Gap 2.2: fits if LLM can be a query source

Looking at the analyzer's own decompositions, most of 2.2 already produced
sensible query + enrichment + filter breakdowns — they were just scored
harshly because enrichments returned probability/confidence rather than
deterministic answers. Filtering on a confidence column is perfectly
expressible within the recipe; the recipe doesn't require deterministic
enrichments.

The genuinely hard cases were "find my ICPs" — where each *row* of the
dataset is a synthetic ICP definition, not an external entity. These map
onto the recipe **only if `llm_generation` is allowed as a query source**:

```
query:        llm_generation → "generate 5 ICPs for this business given context"
enrichments:  outreach_channels  (LLM, optionally web-grounded)
              example_companies  (Apollo / web_search, validates the ICP)
filter:       none, or "ICP confidence > threshold"
```

With LLM-as-source, gap 2.2 collapses.

### Gap 2.3: not a recipe gap, a UX gap

"Show me vertical saas in healthcare" decomposes fine the moment the user
picks a source and a column set. The recipe is universal here; what's
missing is *which Crunchbase fields*, *which geography*, *what defines
"vertical"*. That's an onboarding / clarification problem, not a coverage
problem. The recipe is fine; the elicitation step is what needs work.

### Gap 1 revisited: also fits with LLM as a query source

If `llm_generation` is a first-class source, synthetic generation isn't a
separate path:

```
query:        llm_generation → "generate math word problems, GSM8K/AMC style"
              batch_size: 100, dedup_key: problem_hash,
              stop_when: novelty_rate_in_last_batch < 0.4
enrichments:  difficulty_grade        (LLM)
              topic                   (LLM)
              solution_check_passed   (code_exec — runs the solution in Python)
              ambiguity_score         (LLM)
filter:       solution_check_passed = true
              ambiguity_score < 0.2
              topic distribution roughly matches GSM8K target mix
```

Structurally identical to a lead-gen recipe. The enrichment
`solution_check_passed` uses `code_exec`, which is the same enrichment
shape as "verify this person's email" in lead-gen — take a candidate, run
an external check, return a column. The provenance of the candidate is
opaque to the enrichment step.

## The two generalizations that close coverage

### 1. `llm_generation` is a first-class query source

LLM is iterable like any other source — it just paginates by **novelty**
instead of by cursor:

| Source type | Iteration mechanic        | Stop condition                  |
|-------------|---------------------------|----------------------------------|
| Pull        | cursor / offset           | no more matches in universe     |
| Stream      | time window               | window exhausted / freshness    |
| Generate    | "different from these N"  | novelty rate drops below floor  |

All three return `Iterable[Candidate]`. Downstream pipeline doesn't care
which one produced the row.

### 2. Enrichments can return scalar | list | table (fan-out)

The "for each company, find 3 decision-makers" pattern is everywhere in the
data — not unique to LLM-source queries. It's a 1-to-N relationship that
the original recipe couldn't express cleanly. Fix: an enrichment can return
multiple values or a sub-table, which auto-explodes the parent row.

A table-returning enrichment is itself a mini-recipe applied per parent:

```
fan-out enrichment {
  shape: table
  field: yc_matches
  sub-recipe:
    sub-query:    web_search + crunchbase → "YC companies in: {parent.category}"
    sub-enrich:   yc_batch, url, founders
    sub-filter:   exists_in_yc_directory = true
}
```

Recipe is fractal — `query + enrichments + filters`, where any enrichment
slot can itself be a `query + enrichments + filters`. Bounded recursion,
typically 1–2 levels in practice. **Output is still one flat denormalized
table** — explode + join collapses the tree, parent columns repeat across
child rows.

## End-to-end example: hybrid LLM-seed + real-source verification

User: "Find niche YC startups solving boring B2B problems."

```
QUERIES
  [ { source: llm_generation,
      query:  "Generate 50 unsexy B2B problem categories. Return
               {category, example_pitch}." } ]

ENRICHMENTS
  [ { field: yc_matches,      shape: table,
      source: web_search + crunchbase,
      desc:   "find up to 10 real YC companies matching each category;
               return rows of {company_name, yc_batch, url, founders[]}" },
    { field: founder_email,   shape: scalar,
      source: fullenrich,
      desc:   "verified work email for the founder
               (applied per child row after explode)" },
    { field: last_funding_at, shape: scalar,
      source: crunchbase,
      desc:   "most recent funding date" } ]

FILTERS
  [ "yc_matches not empty",                # parent-level
    "founder_email is verified",           # child-level
    "last_funding_at within 24 months" ]   # child-level
```

50 LLM-generated parents × ~5 YC matches each = ~250 verified-real child
rows. Final flat table:

```
category | example_pitch | company_name | yc_batch | founder_name | founder_email | last_funding_at
```

LLM brainstormed the search vocabulary (which Apollo can't do — you can't
ask Apollo "what categories should I search?"). Real sources verified and
contact-enriched. Recipe handles it.

## Coverage after refinement

| Path                          | Before | After |
|-------------------------------|--------|-------|
| Clean lead-gen + research     | 73%    | 73%   |
| Synthetic SFT (gap 1)         | 0%     | ✓ via llm source |
| Strategic/inferential (2.2)   | partial| ✓ via llm source |
| Needle-in-haystack (2.1)      | partial| ✓ already fit, scoring fixed |
| Fan-outs (1-to-N)             | leaky  | ✓ via table-shaped enrichments |
| Underspecified (2.3)          | partial| handled at UX layer, not recipe |
| Chitchat / invalid            | 13%    | 13% — never serve |

Net: **~73% → ~95%+ clean coverage** of valid requests, with the same
three slots. The remaining ~5% is genuine non-requests.

## What stays the same

- Three slots: `query / enrichment / filter`.
- Output is one flat denormalized table.
- Existing lead-gen and research-dataset recipes work unchanged.
- Cost is still linear in `|candidates| × |enrichments|`, just with
  fan-out multipliers when table-shape enrichments are used.

## What changes

1. **Query sources expand** from `{Pull, Stream}` to `{Pull, Stream, Generate}`.
   `Generate` (LLM) is iterated by novelty-driven re-prompting against a
   running dedup set, stopped when novelty rate falls below a threshold.

2. **Enrichment return shapes expand** from `scalar` to
   `{scalar, list, table}`. List and table outputs auto-explode the
   parent row; table outputs are themselves sub-recipes with their own
   query/enrichments/filters.

3. **Synthetic SFT is no longer a separate path** — it's the same recipe
   with `source: llm_generation` in the query slot and `code_exec` /
   `llm` in the enrichment slots for verification.

## Operational implications

- **Budgeting**: fan-out multiplies cost (`parents × children_per_parent ×
  enrichments`). Need to estimate child cardinality before kicking off, or
  stream the explode with a parent-row cost cap. Same problem the system
  already has for Apollo → decision-maker fan-outs, just generalized.
- **Termination**: pull sources terminate by exhaustion, generate sources
  terminate by novelty. Need a uniform `done?` signal across source types
  so the orchestrator can stop iteration without source-specific logic.
- **Provenance**: pull sources cite URLs, generate sources cite prompts +
  model. Both need a provenance column on every candidate row so the
  user can audit row-by-row.
- **Quality gates**: filters and enrichments together act as the quality
  gate. With LLM-as-source, more rows fail filters (lower precision per
  candidate), so the budget calculation should assume a kept-rate
  multiplier per source.

## Reference: source data and analysis script

- `scripts/dump_first_messages.py` — pulls first user message per project
  from dev + prod Postgres into CSVs.
- `scripts/first_messages_{dev,prod}.csv` — deduped, latest-first, prompt-only.
- `scripts/first_messages_{dev,prod}_full.csv` — same with project + user
  metadata.
- `scripts/breakdown_first_messages.py` — calls Azure OpenAI (gpt-5.4) on
  each unique message with a structured-output schema and writes the
  per-row decomposition.
- `scripts/breakdown_first_messages.csv` — final 256-row analysis with
  `category`, `fits_recipe`, `fit_score`, `queries`, `enrichments`,
  `filters`, `notes` columns. This is the empirical evidence base for the
  coverage numbers above.
