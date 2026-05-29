---
name: find-subreddits
description: Finding the subreddits where a topic/audience lives — "what subreddits should I post my <X> in", "where's my audience on Reddit". Uses browser_use to read Reddit's own community directory (the only thing that reaches it).
applies_to: [orchestrator]
---

## Finding subreddits for a topic / audience

Trigger: "what subreddits should I post my B2B SaaS in", "find subreddits for <topic>", "where's my audience on Reddit".

### Use `browser_use` — it's the only thing that reaches Reddit's community directory

Empirically tested head-to-head:

- Reddit hard-blocks datacenter IPs (403) → direct JSON and Firecrawl both fail.
- Community-search actors are **dead**: `solidcode/reddit-scraper` and `trudax/reddit-scraper-lite` with `searchCommunities` return 0 rows. Do NOT use them.
- Aggregating posts from `clearpath` gives "subreddits where the topic gets posted" — broad and noisy (pulls r/AITAH, r/ProgrammerHumor), and **misses the purpose-built niche subs** that are the best post targets.
- `browser_use` searches Reddit's actual **Communities** directory in a real browser, so it finds the targeted niche communities (r/SaaSSales, r/B2BSaaSGrowthTips, r/leadgeninsiders, r/leadgen, r/sales_intelligence) alongside the big ones (r/SaaS, r/startups, r/sales). For "where do I post," targeted wins. This is the method.

### The call

```
table_create(
  source="browser_use",
  query_params={
    "url": "https://www.reddit.com/search/?q=<primary topic, url-encoded>&type=communities",
    "task": (
      "Find subreddits where <PRODUCT>'s audience hangs out (<ICP, e.g. 'B2B SaaS founders, "
      "startups, sales teams, lead-gen'>). On www.reddit.com search Communities for terms like "
      "'<topic1>', '<topic2>', '<topic3>'. For EACH relevant community return these fields: "
      "subreddit (the name WITHOUT the 'r/' prefix, e.g. 'SaaS'), "
      "members (member count as a plain integer, read from the search-results listing — do NOT open each community one-by-one, that's too slow at this scale; 0 if not shown), "
      "description (its one-line description), "
      "url (full https URL). "
      "Search ALL of the topic terms above (and obvious adjacent ones), scroll / click "
      "'load more' on each communities search, and accumulate the results. Return up to 100 "
      "distinct relevant communities (dedupe by name), most relevant first. If fewer than 100 "
      "genuinely relevant communities exist for this niche, return all you find — don't pad with "
      "irrelevant subs."
    )
  },
  name="Subreddits for <topic>"
)
```

- **Do NOT pre-declare `columns`** — browser_use rejects it (row shape comes from the task). Map columns with `column_map_set` after the rows land if needed.
- Build the topic terms from the **audience**, not the product name: a B2B SaaS → "B2B SaaS", "lead generation", "sales", "startups"; a fitness app → "calorie counting", "weight loss", "running".
- `start_url` / `url` = Reddit's `type=communities` search so the agent lands on the directory, not posts.

### browser_use is intermittently flaky — RETRY, don't fall back

BU cloud occasionally fails a session-start or returns a 502; you'll see 0 rows or an error. **Retry the same call once or twice** — it succeeds on retry (verified: ~1 in 3 attempts hiccups). Do NOT fall back to the dead community actors. The account/method are fine; the failure is transient cloud plumbing.

### Data caveats (set expectations, don't post-process)

- **Names + descriptions are reliable.** Member counts are best-effort — Reddit sometimes shows "weekly visitors" instead of total members, and a few come back 0/unknown. Don't promise exact subscriber numbers; the value is the right *list of communities*.
- A couple of marginally-relevant subs slip in; the top of the list is solid. Sort/scan by relevance.
- **Count scales with niche breadth.** A broad space (B2B SaaS, fitness) yields ~50-100; a narrow one returns whatever genuinely exists (don't expect 100 for a hyper-niche product). The task targets up to 100 — it won't pad with junk to hit the number.
- Cost/time scale with the target: ~100 communities across several searches is ~4-8 min and a few $ of BU session time (vs ~2 min / ~$0.20 for 20). Fine for a deliberate one-off; not for high-frequency use.

### Followups

- Read each subreddit's posting rules before promoting (many restrict self-promo) — a human step.
- Feed the resulting subreddit list into [[find-reddit-leads]] (scrape those subs for buying-intent posts) or [[fetch-reddit]].
