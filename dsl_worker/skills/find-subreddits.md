---
name: find-subreddits
description: Listing subreddits relevant to a topic or audience.
applies_to: [orchestrator]
---

## Finding subreddits for a topic

Skip if the user's ask isn't about subreddits or Reddit communities.

### The actor

`apify_actor:solidcode/reddit-scraper` is the working path. Native Reddit search via `web_search` is unreliable; other community-search actors I tested either return bad data or zero results. This one returns clean structured rows with subscriber counts, descriptions, and URLs.

### Call shape

```
table_create(
  source="apify_actor:solidcode/reddit-scraper",
  query_params={
    "searches": ["<topic>"],
    "searchCommunities": true,
    "searchPosts": false,
    "searchComments": false,
    "searchUsers": false
  },
  columns=[
    {"name": "Subreddit",    "source_field": "displayName",        "type": "text"},
    {"name": "Subscribers",  "source_field": "subscribers",        "type": "number"},
    {"name": "Description",  "source_field": "publicDescription",  "type": "text"},
    {"name": "URL",          "source_field": "url",                "type": "url"},
    {"name": "Created",      "source_field": "createdAt",          "type": "date"},
    {"name": "NSFW",         "source_field": "isNsfw",             "type": "enum"},
  ],
  name="Subreddits about <topic>"
)
```

Returns 70–100 communities for typical queries. ~$0.08 per query ($0.01 start + $0.001/item × ~80).

### Quirks to know

- **Short queries work best.** "GTM", "founders", "B2B SaaS" → 70–100 results. Longer multi-word queries ("ecommerce marketing") sometimes return zero. If a query returns zero, retry with a shorter / different keyword (e.g. "ecom" + "marketing" as two queries, or "DTC").
- **Reddit's search returns some noise.** Ambiguous abbreviations like "GTM" pull in r/GuessTheMovie + r/ClimateGTM (16 subs, dead). Filter by subscriber count and topic relevance after the fetch.
- **The actor doesn't take a maxItems cap.** It returns whatever Reddit serves. To trim, use `filter_set` on `Subscribers >= 1000` after the table lands.
- **"Similar to r/X" is not a built-in feature.** If the user wants "subreddits like r/Entrepreneur", extract topic keywords from r/Entrepreneur's description and run the topic search yourself. Don't promise a similarity feature you don't have.

### Followups

Common next steps once the table is in place:

- Filter by subscriber count (skip dead subs with <500 subs; skip mega-subs like r/todayilearned with 41M unless the user wants them).
- Filter by language (`Lang = en` for English-speaking audience).
- Drop NSFW unless asked.
- Use the resulting list as a **scope** for downstream enrichments (e.g. "for each of these subs, fetch top 100 posts last 30 days").

### When zero results come back

If `searches: ["<topic>"]` returns 0 rows:

1. Try a single-word version of the topic.
2. Try a near-synonym (e.g. "DTC" instead of "direct-to-consumer ecommerce").
3. As a last resort, ask the user to name 2–3 communities they already know about and pivot to "more like these" via descriptions.

Do NOT switch sources just because one query was weak. The actor is the right path; the query is the lever.
