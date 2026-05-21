---
name: scrape-g2-reviews
description: Scraping G2 product reviews (and using the reviewers as lead sources).
applies_to: [orchestrator]
---

## Scraping G2 reviews

Skip if the ask isn't about G2.com reviews or using G2 reviewers as a lead source.

### The actor — use `apify_actor:azzouzana/g2-products-reviews-scraper-pro`

Empirically the fastest + richest. Race I ran on Clay.com (May 2026):

| Actor | Items | Time | Cost |
|---|---:|---:|---:|
| **azzouzana/g2-products-reviews-scraper-pro** | 7 | **6s** | $0 (trial) / $0.001/review after |
| zen-studio/g2-reviews-scraper | 7 | 117s | $0.027 |
| focused_vanguard/g2-reviews-scraper | 0 | 19s | failed |

Don't use `apify_search_actors` to discover G2 actors — its top result for "G2 reviews scraper" is consistently `automation-lab/reddit-scraper` (a Reddit actor), not anything G2-related. Go straight to `azzouzana/g2-products-reviews-scraper-pro`.

### Find the product slug first

G2 URLs are `https://www.g2.com/products/<slug>/reviews`. The slug is rarely the product's marketing name. **Always confirm the slug via `web_search` before calling the actor** — guessing wastes a full ~60s actor run and 0 rows. Patterns to know:

- Brand-takes-over-the-name disambiguation: `clay` is the personal-CRM Clay; the lead-data Clay.com is `clay-com-clay`.
- Company name with TLD: `monday.com` is `monday-com`.
- Multi-product brands: `slack` is `slack` but `slack-connect` is separate.

Quick confirm:
```
web_search("site:g2.com/products clay.com")    # cheap, $0.025
```
Open the result, copy the slug.

### Call shape

```
table_create(
  source="apify_actor:azzouzana/g2-products-reviews-scraper-pro",
  query_params={
    "input": {
      "productUrl": "https://www.g2.com/products/<slug>/reviews",
      "maxItems": 200,
      "starFilterCondition": "maximum",   # "maximum" or "minimum"
      "starFilterValue": 3,               # integer 1-5
      "lookbackDays": 365,                # OPTIONAL — server-side date cutoff
      "sortOrder": "most_recent",
    },
    "maxItems": 200,
  },
  name="<Product> G2 Reviews"
)
```

Filter semantics:
- `starFilterCondition="maximum", starFilterValue=3` → stars ≤ 3 (negative reviews — most common ask: "find customers complaining about X").
- `starFilterCondition="minimum", starFilterValue=4` → stars ≥ 4 (positive reviews — for "find happy customers of competitor X").
- Omit both fields for ALL ratings.

Use `lookbackDays` for server-side date filtering — much more reliable than trying `filter_set` on the date column after the fact (G2's published_at is an ISO timestamp; FE filter on a text-typed column matches lexically and is finicky).

### Column map after fetch

The actor returns 47 fields per review. Keep the useful ones:

```
column_map_set(table_id="t1", columns=[
  {"name": "Reviewer Name",   "source_field": "user_name",                 "type": "text"},
  {"name": "Title",           "source_field": "title",                     "type": "text"},
  {"name": "Company Segment", "source_field": "company_segment_label",     "type": "enum"},
  {"name": "Country",         "source_field": "country",                   "type": "text"},
  {"name": "Rating",          "source_field": "rating",                    "type": "number"},
  {"name": "Published",       "source_field": "published_at",              "type": "date"},
  {"name": "What They Like",  "source_field": "whatDoYouLike",             "type": "text"},
  {"name": "What They Dislike","source_field": "whatDoYouDislike",         "type": "text"},
  {"name": "Problem Solved",  "source_field": "whatProblemsOrBenefits",    "type": "text"},
  {"name": "Categories",      "source_field": "categories[].name",         "type": "text"},
  {"name": "Review URL",      "source_field": "reviewUrl",                 "type": "url"},
])
```

Reviewer identity is **anonymized on G2** ("Verified User in Financial Services"). G2 doesn't expose real names or emails. Don't promise the user reviewer-name-resolution — it's not in the data. The product is the reviewer's role + company segment + country + the review text itself.

### Cost / volume notes

- Per-review pricing: **$0.001/review** ($1 per 1K reviews). Negligible for typical asks (10-200 reviews).
- For high-volume products (e.g. Salesforce — 25k+ reviews), use `maxItems` to bound spend.
- One actor run scrapes one product URL. To compare multiple products, run multiple `table_create`s (one table per product) and union later via "Product" column.

### Quirks to know

- **Some G2 products have only a handful of reviews** matching tight filters. Clay.com 1-3 star: 7 results. If you expected 50, lower the star bar to all-stars or drop `lookbackDays`.
- **`reviewerName` is always anonymized.** Don't ask for it.
- **`switched_from_products` is populated when the reviewer says they switched.** Useful signal for finding competitor users.
- **`incentivized` flag distinguishes paid/gift-card reviews.** Drop incentivized reviews if you're looking for genuine signal.

### Common followup pattern

G2 reviewers are a candidate pool — they're real software buyers complaining about a specific product. Typical downstream:

1. Fetch 1-3 star reviews of competitor X → that's the candidate list.
2. Enrich each row: `apollo_org_enrich` on the company (from `company_segment_label` + `categories`), `whatProblemsOrBenefits` is the talking point.
3. The reviewer is anonymized but the COMPANY SEGMENT + INDUSTRY + COUNTRY narrows down ICP — often combine with a separate Apollo people search to find the actual contact.
