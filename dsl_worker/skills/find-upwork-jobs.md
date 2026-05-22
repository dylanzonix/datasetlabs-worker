---
name: find-upwork-jobs
description: Finding active Upwork jobs (lead list / data scraping / research) — for fulfilling them as a service.
applies_to: [orchestrator]
---

## Finding Upwork jobs for dataset/research work

Skip if the ask isn't about Upwork or finding jobs to fulfill.

### The actor — use `apify_actor:devcake/upwork-jobs-scraper`

Empirical race on "lead list" + "lead generation" + "data scraping" queries:

| Actor | Items | Time | Cost |
|---|---:|---:|---:|
| **devcake/upwork-jobs-scraper** | 50 | **25s** | $0.11 (pay-per-event, ~$0.002/job) |
| neatrat/upwork-job-scraper | — | — | requires monthly subscription, returned 0 rows on trial |
| getdataforme/upwork-actor | — | — | subscription model, observed 0 rows in dev |
| flash_mage/upwork | — | — | per-item pricing, second-tier choice |

Don't waste a turn on `apify_search_actors` — go straight to devcake.

### Call shape

```
table_create(
  source="apify_actor:devcake/upwork-jobs-scraper",
  query_params={
    "input": {
      "searchQueries": ["lead list", "lead generation", "data scraping"],
      "maxItems": 100,
      "sort": "recency+desc",
    },
    "maxItems": 100,
  },
  name="Upwork Dataset Jobs"
)
```

### Sort values (must match exactly)

The actor rejects free-form sort strings. Use one of:

- `"recency+desc"` — newest first. **Default for fulfilling jobs** (fresh = unbid).
- `"relevance+desc"` — Upwork's own relevance score on the query.
- `"spend+desc"` — clients who've spent the most on Upwork first (higher quality, less haggling).

### Query patterns that find dataset-buildable jobs

The product fulfills "build me a structured list" jobs. The keywords that surface those jobs:

**High-signal (run these first):**
- `"lead list"`, `"lead lists"`
- `"prospect list"`, `"prospect lists"`
- `"contact list"`
- `"lead generation"` (broad, lots of volume)
- `"data scraping"`
- `"data mining"`
- `"web scraping"`
- `"list building"`

**Adjacent (run if first batch is thin):**
- `"company research"`
- `"email list"` / `"email research"` / `"email extraction"`
- `"contact research"`
- `"data extraction"`
- `"database research"`
- `"market research"` (broader — needs filtering)

**Pass `searchQueries` as an array — the actor unions results.** Fewer Apify runs, less cost.

### Column map after fetch

The actor returns rich job data. Useful columns:

```
column_map_set(table_id="t1", columns=[
  {"name": "Title",         "source_field": "title",          "type": "text"},
  {"name": "Description",   "source_field": "description",    "type": "text"},
  {"name": "Job Type",      "source_field": "jobType",        "type": "enum"},  # HOURLY | FIXED
  {"name": "Budget",        "source_field": "budget",         "type": "text"},
  {"name": "Hourly Min",    "source_field": "hourlyMin",      "type": "number", "format": "currency"},
  {"name": "Hourly Max",    "source_field": "hourlyMax",      "type": "number", "format": "currency"},
  {"name": "Fixed Amount",  "source_field": "fixedAmount",    "type": "number", "format": "currency"},
  {"name": "Tier",          "source_field": "contractorTier", "type": "enum"},  # EntryLevel | IntermediateLevel | ExpertLevel
  {"name": "Duration",      "source_field": "duration",       "type": "text"},
  {"name": "Skills",        "source_field": "skills[]",       "type": "text"},
  {"name": "Posted",        "source_field": "publishTime",    "type": "date"},
  {"name": "URL",           "source_field": "url",            "type": "url"},
], dedup_key_column="URL")
```

`fixedAmount` is null for hourly jobs; `hourlyMin`/`hourlyMax` are null for fixed jobs. That's fine — let the FE render nulls.

### Filtering the noise

Lead-generation / data-scraping queries on Upwork return **a lot of jobs we can't fulfill**: commission-only sales gigs, cold-call jobs, sales coaching, etc. Filter via classify-tier enrichment:

```
enrichment_set(
  name="Fulfillable Tag",
  columns=[{"name": "Fulfillable", "type": "enum"}, {"name": "Why", "type": "text"}],
  action={
    "research": "classify",
    "prompt": (
      "Is this Upwork job something we can fulfill by building a structured dataset/list (rows + columns) "
      "and delivering as a CSV or spreadsheet? YES if the deliverable is a list of leads, companies, contacts, "
      "products, places, posts, etc. with researched columns. NO if it's a commission-only sales role, "
      "ongoing cold-calling, sales coaching, full-time hire, or work requiring a human (calls, meetings, "
      "negotiations). Output Fulfillable: Yes|No and Why: one short sentence."
    ),
    "depends_on": ["Title", "Description"],
    "per_row_credit_cap": 0.3,
  }
)
```

Then `filter_set(column="Fulfillable", op="is_any_of", value=["Yes"])`.

Typical yield: 30-50% of fetched jobs are fulfillable. The rest are sales/commission gigs.

### What a good "fulfillable job" looks like

The dream signals when scanning the table:
- **Budget present and reasonable** — fixed $20-200 or hourly $10-40 (margin space, not a race-to-zero gig).
- **Description has a concrete deliverable** — "100 leads", "list of HVAC companies in TX", "200 founders' emails". Vague jobs ("looking for someone to grow my business") are not it.
- **Tier: IntermediateLevel or ExpertLevel** — Entry-level jobs attract 50+ bidders at $5/hour. Skip.
- **Posted recently** — within 24h is golden. After 48h, expect 30+ bids already.
- **No "must speak X language fluently"** — those are human-required.

### Quirks to know

- **The actor scrapes search results, not the full job page.** Description is truncated (~500 chars). Enough to triage; open the URL for the full brief before bidding.
- **`recency+desc` returns Upwork's "newest" tab.** Some jobs are hours old, some are minutes. Fresh = better for bidding.
- **Pagination via `maxItems`.** Bumping to 200+ extends the time window scanned. ~30s per 50 items.
- **Skills array often empty** for older jobs or specific categories. Don't depend on it for filtering.

### Common followup pattern

Find jobs → filter to fulfillable → for each, decide whether to bid:

1. Sort the filtered table by Posted DESC.
2. Open the top 5-10 URLs in tabs, read full briefs.
3. For ones that match our capability (dataset shape, reasonable budget), bid + queue the fulfillment.
4. The agent can spin up a separate project per job using the brief as the chat input.
