---
name: find-reddit-leads
description: "Find me leads on Reddit for <company/website>" — derive the ICP from the site, broad-scrape a month of Reddit on the customer's pain, then classify every post to surface the people actually asking. Thorough, cheap (~$0.50/run), beats web_search.
applies_to: [orchestrator]
---

## Finding leads on Reddit for a company

Trigger: "find leads on reddit for stripe.com", "who on reddit needs <product>", "reddit leads for my SaaS <url>". The user often gives **just a website** — derive everything from it.

This uses the [[fetch-reddit]] actor for the scrape; this skill is the **lead-gen strategy** on top of it.

### The big idea (why this beats web_search)

Scrape the whole last month of Reddit broadly on the **pain**, then let a cheap classify-tier enrichment separate the people *asking* from the noise. You get every real person who voiced the problem this month, curated — thorough, not the 10 shallow links web_search returns. Empirically: a popular B2B category yields **~28 explicit asks + ~160 pain posts per month**, at ~$0.50/run. Do NOT use web_search for this.

The single thing that makes or breaks a run is **query derivation**. Get the pain language + incumbent right and it's magic; use product jargon and it drifts.

### Step 1 — derive the ICP from the site (one turn, primary ICP only)

Read the site (it's usually enough from the homepage). Extract, in the **customer's words, not the product's**:

- **The pain** people would complain about ("GA4 is confusing", "I forget to follow up with leads", "calorie counting is tedious").
- **The incumbent / alternative** they'd switch from — this is the highest-signal term. Plausible's best query was literally `"google analytics alternative"`. Name competitors too (PostHog, Fathom, etc.).
- **The ICP** — who has this pain (indie devs, founders, agencies, dieters…). If there are several, **pick the primary one** and build one table this turn. Don't fan out into multiple tables unprompted.

Don't query the product's category jargon ("personal CRM") — query what the sufferer types ("how do I keep track of who I've talked to"). This is the #1 lever.

### Step 2 — build a broad, multi-angle query

OR together every angle: `"alternative to <incumbent>"`, the incumbent/competitor names, the pain phrases, the category. Add `NOT` for obvious noise (`NOT hiring NOT job`). Breadth here is good — precision comes from the classifier, not the query.

```
("google analytics alternative" OR "alternative to google analytics" OR "GA4"
 OR "ditch google analytics" OR "privacy analytics" OR "cookieless analytics"
 OR "GDPR analytics" OR "simple analytics" OR "self hosted analytics"
 OR "PostHog" OR "Fathom analytics") NOT hiring NOT job
```

### Step 3 — scrape (thorough defaults)

```
table_create(
  source="apify_actor:clearpath/reddit-search-scraper",
  query_params={"input": {
    "query": "<the OR query>",
    "contentType": "both",            # posts AND comments — comments hold candid intent
    "sort": "new",
    "timeFilter": "month",            # DEFAULT: last 30 days unless the user says otherwise
    "autoDiscoverSubreddits": true,   # finds the right communities itself — works well
    "maxSubreddits": 30,
    "maxResults": 1000                # thorough default; the whole month for most niches
  }},
  columns=[
    {"name": "Type",      "source_field": "_type",        "type": "enum"},
    {"name": "Title",     "source_field": "title",        "type": "text"},
    {"name": "Body",      "source_field": "body",         "type": "text"},
    {"name": "Subreddit", "source_field": "subreddit",    "type": "text"},
    {"name": "Author",    "source_field": "author",       "type": "text"},
    {"name": "Score",     "source_field": "score",        "type": "number"},
    {"name": "Date",      "source_field": "createdAt",    "type": "date"},
    {"name": "URL",       "source_field": "url",          "type": "url"},
  ],
  name="Reddit leads — <company>"
)
```

**If it returns 0 rows, retry the SAME call** — clearpath flakes to 0 occasionally. Don't switch actors. If it's *still* thin after a retry, the niche is genuinely small (see "thin niche" below).

### Step 4 — classify every row for lead quality (this is the magic)

A `classify`-tier enrichment over all rows. Nano is enough — it correctly dumps ~85% as noise and nails the asker-vs-answerer distinction. Surface 3s first.

```
enrichment_set(
  name="Lead Signal",
  columns=[
    {"name": "Lead Score", "type": "enum"},   # 3 | 2 | 1 | 0
    {"name": "Signal",     "type": "enum"},    # hot | warm | weak | noise
    {"name": "Why",        "type": "text"},
  ],
  action={
    "research": "classify",
    "depends_on": ["Title", "Body", "Subreddit"],
    "per_row_credit_cap": 0.05,
    "prompt": (
      "Score this Reddit post/comment as a SALES LEAD for: <ICP one-liner>.\n"
      "A LEAD is someone who could BUY this product and shows fit + intent.\n"
      "3 = HOT: explicitly asking for a tool/solution in this category, asking for an "
      "alternative to a named competitor, or 'what should I use for <what this does>'.\n"
      "2 = WARM: clearly describes the exact pain this solves, or complains about the "
      "incumbent/status quo, and is plausibly the ICP.\n"
      "1 = WEAK: in the space / ICP-adjacent but no buying need.\n"
      "0 = NOISE: not a prospect — people ANSWERING/recommending (not asking), vendors "
      "promoting their own product, news, memes, already-solved, or not the ICP.\n"
      "Fill Lead Score (3/2/1/0), Signal (hot/warm/weak/noise), and Why (<=12 words)."
    )
  }
)
```

You (the orchestrator) are free to run this classify enrichment yourself — it's the filtering step that makes the table useful. Then `filter_set` Lead Score `is_any_of` [3, 2] and `sort_set` by Lead Score desc. Lead the user's attention to the 3s (DM-ready today); 2s are warmer-engagement plays.

### Thin niche — fail gracefully

If the product is a brand-new category nobody posts about (e.g. "Docker on Ethereum" → 1 lead), the method can't manufacture demand. Don't pad the table with garbage. Tell the user plainly: "Reddit's thin for this exact niche — only N real signals this month," and offer to **broaden to the adjacent bigger category** (e.g. "decentralized hosting" → "cloud hosting / self-hosting pains") or widen the time window to `year`.

### Numbers to set expectations

- Popular B2B/prosumer pain (analytics, GTM, dev tools, finance, fitness): hundreds of rows/month, ~20% land as leads, ~25–30 explicit asks. Excellent.
- Mass consumer (calorie app): huge volume but fuzzy leads — great for *audience research*, less for cold DMs (DMing 60 strangers a consumer app reads as spam). Set that expectation.
- Ultra-niche/novel: few or none. That's signal, not failure.
- Cost: ~$0.01 scrape (1000 rows) + ~$0.50 nano classify ≈ **$0.50/run**.

### Other channels

Same pattern (broad-scrape the pain → classify) is the play for LinkedIn/X/etc. when those sources exist. Reddit is the cleanest because people post raw, searchable pain there.
