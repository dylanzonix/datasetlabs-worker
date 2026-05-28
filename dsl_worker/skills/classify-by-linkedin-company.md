---
name: classify-by-linkedin-company
description: Qualifying a list of companies by a fuzzy descriptive criterion (is this an X) — scrape LinkedIn company About data, then classify against it. ~$0.004/row + nano. 10-15x cheaper than per-row web_search.
applies_to: [orchestrator]
---

## When to use

The user has a list of companies (typically Apollo) and wants to filter/label each by a descriptive criterion the firmographic fields can't express. Examples:

- "Is this an outbound agency?"
- "Is this a B2B SaaS company?"
- "Is this a healthcare practice?"
- "Industry vertical: pick from [eCommerce | Fintech | Healthcare | Other]"
- "Does this company sell to enterprise?"

Apollo's `/mixed_companies/search` returns firmographics — name, domain, headcount, NAICS — but NOT the company's self-described identity (tagline / description / industries). The cell agent's instinct is to `web_search` per row to fill that gap. That's a 10-15x cost mistake.

The right pattern: scrape each company's LinkedIn About via `apify_call_actor(harvestapi/linkedin-company)`, then classify against the scraped fields.

## Verified cost (empirical, not docs)

`harvestapi/linkedin-company` is **PAY_PER_EVENT at $0.004 per company** (verified by reading the actor's `pricingInfos`). Returns structured fields including `tagline`, `description`, `industries`. Bulk-capable but per-row calls are fine.

For 1000 companies:
- Apify scrape: $0.004 × 1000 = **$4**
- Classify-tier nano against scraped fields: ~$0.0005 × 1000 = **$0.50**
- **Total: ~$4.50 vs ~$50-100 for per-row web_search**

## Pattern (two enrichments)

**Enrichment 1 — LinkedIn Profile** (research-tier, ~$0.005/row including apify cost):

```
enrichment_set(
  name="LinkedIn Profile",
  columns=[
    {"name": "LinkedIn Tagline",     "type": "text"},
    {"name": "LinkedIn Description", "type": "text"},
    {"name": "LinkedIn Industries",  "type": "text"},
  ],
  action={
    "research": "research",
    "prompt": (
      "Call apify_call_actor with actor_id='harvestapi/linkedin-company' and input "
      "{\"companies\": [Company LinkedIn]} (a single-element list with the row's "
      "LinkedIn URL). Read the returned row's `tagline`, `description`, and "
      "`industries` (array of objects — join their names with ', '). Fill the "
      "three columns. Return null on any field the actor doesn't provide."
    ),
    "depends_on": ["Company LinkedIn"],
    "per_row_credit_cap": 0.5
  }
)
```

Note: Apollo's `mixed_companies/search` raw rows include a `linkedin_url` field — column_map_set it as "Company LinkedIn" before this enrichment runs. If the row only has Domain, fall back to `apify_call_actor` with the domain — harvestapi/linkedin-company accepts domains too.

**Enrichment 2 — qualification** (classify-tier, ~$0.0005/row):

```
enrichment_set(
  name="Is Outbound Agency",     # or whatever the qualification is
  columns=[{"name": "Is Outbound Agency", "type": "enum"}],
  action={
    "research": "classify",
    "prompt": (
      "Read LinkedIn Tagline + LinkedIn Description + LinkedIn Industries. "
      "Return Yes if this is a B2B outbound agency / lead-gen / appointment-setting / "
      "SDR-as-a-service business. Return No otherwise. When the three fields are all "
      "empty, return null."
    ),
    "depends_on": ["LinkedIn Description"],
    "per_row_credit_cap": 0.05
  }
)
```

Classify auto-runs once `LinkedIn Description` is populated (dep-readiness gate). Total wall-time: ~2-3 minutes for 1000 rows at typical concurrency.

## Sample harvestapi output

Verified live on these domains:

```
DM Lead Generation
  tagline: "The best BookIn Software ever!"
  description: "What would it be like if you magically had more scheduled sales appointments..."
  industries: [{name: "Marketing & Advertising", title: "Advertising Services"}]

Kozmoze
  tagline: "Your remote Sales Prospecting team..."
  description: "Are you looking for a sustainable way to generate B2B leads?..."
  industries: [{name: "Marketing & Advertising", title: "Advertising Services"}]

HubSpot
  tagline: "The agentic customer platform to scale your business."
  description: "HubSpot is a leading agentic customer platform..."
  industries: [{name: "Computer Software"}]
```

Three fields is enough signal for almost any "is this X" qualification.

## Why not Apollo's organizations/enrich endpoint

Apollo's docs say `/organizations/enrich` "consumes credits as part of your Apollo pricing plan" but won't tell you how many per call without checking the auth-walled `app.apollo.io/#/settings/credits/about` page. Until that's verified, we don't use it — harvestapi LinkedIn is a known cost ($0.004) and similarly clean data.

If/when Apollo enrich cost is verified at < $0.004, swap to it (it's slightly richer — includes `keywords` array and `current_technologies`).

## Don't reach for web_search

For "is this company X" questions answerable from a LinkedIn About page (which is most B2B classification), web_search at ~$0.025/call is a 6x cost mistake. The pattern above is the cheaper, more reliable path. Web_search is only correct when LinkedIn doesn't have the company (rare) or the question can't be answered from a profile (e.g., "did they raise in the last 30 days").
