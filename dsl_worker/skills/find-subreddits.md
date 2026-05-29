---
name: find-subreddits
description: Finding the subreddits where a topic/audience lives — "what subreddits should I post my <X> in", "where's my audience on Reddit". Scrapes posts on the audience's topics and surfaces the communities they actually post in.
applies_to: [orchestrator]
---

## Finding subreddits for a topic / audience

Trigger: "what subreddits should I post my B2B SaaS in", "find subreddits for <topic>", "where's my audience on Reddit". Uses the [[fetch-reddit]] actor.

### Don't use solidcode/trudax community-search — it's dead

The old recipe (`solidcode/reddit-scraper` with `searchCommunities: true`) now returns **0 rows** — verified across retries, proxy, and `trudax/reddit-scraper-lite`. Reddit community-search actors are broken. Do not use them.

### The working method: scrape posts on the topic, the subreddits fall out

The subreddits where a topic is *actively posted* ARE the answer to "where should I post" — better than a name-catalog because it's ranked by where the audience actually is, and dead/irrelevant subs self-filter. Use `clearpath/reddit-search-scraper` (the verified actor) over the audience's topics, map the **Subreddit** column, and the community distribution is right there.

```
table_create(
  source="apify_actor:clearpath/reddit-search-scraper",
  query_params={"input": {
    "query": "(\"B2B SaaS\" OR \"saas founder\" OR \"lead generation\" OR \"cold outreach\" OR \"indie hacker\" OR \"startup marketing\")",
    "contentType": "posts",
    "sort": "top",              # top = established, high-signal posts → the real communities
    "timeFilter": "year",       # wide window so you see durable subs, not just this week's
    "autoDiscoverSubreddits": true,
    "maxSubreddits": 50,
    "maxResults": 400
  }},
  columns=[
    {"name": "Subreddit", "source_field": "subreddit", "type": "text"},
    {"name": "Title",     "source_field": "title",     "type": "text"},
    {"name": "Score",     "source_field": "score",     "type": "number"},
    {"name": "URL",       "source_field": "url",        "type": "url"},
  ],
  name="Subreddits for <topic>"
)
```

Build the `query` from the **audience's topics**, not the product name — what your target users post about. For a B2B SaaS: "B2B SaaS", "saas founder", "lead generation", "indie hacker", "startup marketing". For a fitness app: "calorie counting", "weight loss", "running". Cast a few OR-terms wide.

If it returns 0, **retry the same call** (clearpath flakes to 0 occasionally) — don't fall back to the dead community actors.

### Surfacing the ranked communities

Once the posts table lands, the **Subreddit** column's value frequency = the ranked community list (it shows in the filter panel's distinct-value counts; `sort_set` by Score for high-signal posts). Tell the user the top communities by post volume — e.g. a B2B-SaaS query surfaces r/SaasDevelopers, r/b2b_sales, r/AskMarketing, r/micro_saas, r/startup, r/SaaS, r/buildinpublic, r/Entrepreneurs. Those are where to post.

### Quirks

- **Some noise is normal.** Broad topic queries pull a few off-target subs (r/ProgrammerHumor, r/AITAH). The top ~10 by volume are the real targets; ignore the long tail.
- **Read the subreddit rules before posting.** Many (r/SaaS, r/Entrepreneur) restrict self-promo — a human step, not something to scrape.
- **For "subreddits like r/X":** extract the topics r/X is about and run the topic query — there's no built-in similarity lookup.

### Followups

- Use the resulting subreddit list as the audience for [[find-reddit-leads]] (scrape those subs for buying-intent posts).
- Filter the posts table by Score to see what content performs in each community before you post.
