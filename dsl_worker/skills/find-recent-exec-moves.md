---
name: find-recent-exec-moves
description: Finding companies where a senior exec (CFO/COO/CTO/CHRO/VP Sales etc.) joined in the last 90 days — based on the new exec's own LinkedIn announcement post.
applies_to: [orchestrator]
---

## Finding recent C-suite / VP-level moves

Use this when the user wants companies where a specific senior role was filled recently — "companies that just hired a new CFO," "VPs of Engineering who started in the last 60 days," "founders who announced a new COO," etc.

### Why post search beats every alternative

Tried and rejected:
- **`web_harvest`** — biased to PR Newswire / Business Wire press releases. Maxes out at ~10 results per query. Misses the silent majority of moves that never hit a press release.
- **`fullenrich_people`** — has `employment.current.start_at` on each row but **does not expose it as a filter or sort param**. Hit rate on random sweep is ~3% (90-day window). Burns budget filtering after the fact.
- **`apollo_companies` → enrichment-verify per row** — brute force. ~5% hit rate, expensive at scale.

The winning source: the new exec's own LinkedIn announcement. They almost always post "thrilled to share I've joined…" within days of starting. The post is dated, names the company + role, and the poster's profile gives you the verified LinkedIn URL.

### The actor

`apify_actor:harvestapi/linkedin-post-search` — 1.8M total runs, $0.002/result. Searches LinkedIn posts by keyword. No cookies / no account needed.

### Call shape

**Two-step pattern.** `apify_actor:harvestapi/linkedin-post-search` rejects pre-declared `columns=[...]` on `table_create` — its row shape comes from the actor, not your schema. So:

```
# Step 1: fetch with no columns. LinkedIn search recency only stays
# clean at maxPosts <= 30/query — past that the actor digs into
# relevance-ranked OLD viral posts (6mo+ ago) that defeat the whole
# "recent" intent. Get more breadth via MORE queries, not more per
# query.
table_create(
  source="apify_actor:harvestapi/linkedin-post-search",
  query_params={
    "input": {
      "searchQueries": [
        "joined as Chief Financial Officer",
        "joined as Chief Operating Officer",
        "joined as Chief Technology Officer",
        "joined as Chief Human Resources Officer",
        "joined as Chief Marketing Officer",
        "joined as Chief Revenue Officer",
        "joined as Chief People Officer",
        "joined as General Counsel",
        "thrilled to share I've joined",
        "excited to share I've joined",
        "I am pleased to announce I joined",
        "starting a new position as Chief",
        "I'm thrilled to announce I joined"
      ],
      "maxPosts": 30,
      "postedLimit": "3months",
      "profileScraperMode": "short",
      "scrapeReactions": false,
      "scrapeComments": false
    },
    "maxItems": 400
  },
  name="Recent C-Suite Move Announcements"
)

# Step 2: AFTER the fetch lands, run column_map_set to clean up the
# raw schema. The actor returns ~25 fields per post; map only what
# downstream enrichments need.
column_map_set(
  table_id="t1",
  columns=[
    {"name": "Author Name",     "source_field": "author.name",         "type": "text", "pinned": true},
    {"name": "Author LinkedIn", "source_field": "author.linkedinUrl",  "type": "url"},
    {"name": "Author Type",     "source_field": "author.type",         "type": "enum"},
    {"name": "Post Date",       "source_field": "postedAt.date",       "type": "date"},
    {"name": "Post Content",    "source_field": "content",             "type": "text"},
    {"name": "Post URL",        "source_field": "linkedinUrl",         "type": "url"},
  ],
)

# Step 3 (recommended): filter Post Date >= today - 90 days and
# filter Author Type = 'profile' (drops corporate-page reposts /
# news-bot accounts that don't represent real exec moves).
filter_set(table_id="t1", column="Post Date", op="gte", value="<90-days-ago ISO>")
filter_set(table_id="t1", column="Author Type", op="is_any_of", value=["profile"])
```

Hard input: `searchQueries` is an **array** (plural) — the actor returns 400 if you pass `search` or `query` singular. The actor IGNORES strict freshness filters like `postedLimit`; sort by `Post Date` desc + filter post-fetch.

### Query patterns that work

| Role | Best query strings |
|---|---|
| New CFO | `"joined as Chief Financial Officer"`, `"thrilled to share new CFO"`, `"hired our new CFO"` |
| New COO | `"joined as Chief Operating Officer"`, `"named our COO"` |
| New CTO | `"joined as Chief Technology Officer"`, `"thrilled to announce new CTO"` |
| New CHRO | `"joined as Chief Human Resources Officer"`, `"joined as Chief People Officer"` |
| Generic | `"I am pleased to announce I joined"`, `"thrilled to share I've joined"` |

Run 6-10 queries in one call for coverage. The actor de-dupes by post URL.

### Noise + how to handle it

Raw stream is **~40% real founder/exec announcements, ~60% noise**. Noise types:
- Press-release reposts ("$FWDI Adds Mark Brazier as CFO") — usually posted by news bots like NetworkNewsWire, Small Cap Society, Home Run Stocks
- Engagement bait ("I had a great chat with Sarah our new CFO")
- Anniversary posts ("It's been 1 year since I joined as CFO")
- Cross-posts of the same announcement

**Always follow `table_create` with a `classify`-tier enrichment to filter:**

```
enrichment_set(
  table_id="t1",
  name="Is Real Recent Move",
  columns=[{"name":"Is Real Move", "type":"enum"}, {"name":"Verified Role","type":"text"}],
  action={
    "research": "classify",
    "prompt": "Read Post Content + Author Position + Post Date. Return 'Yes' if this is a first-person announcement of joining a company in a senior role (CFO/COO/CTO/CHRO/VP/SVP/Head of X) within the last 90 days. Return 'No' if the post is: a press release repost from a news/bot account, a third-party comment on someone else's move, an anniversary/throwback post, a re-share, or a different topic. When Yes, extract the role into Verified Role (e.g. 'Chief Financial Officer'). When No, set Verified Role to null.",
    "per_row_credit_cap": 0.5
  }
)
```

`classify` runs nano with no tool calls — costs ~$0.0002/row. Pure text reasoning on Post Content. Then `filter_set` on `Is Real Move = Yes` and the table is clean.

### Extracting company name + start date

After the classify pass, add a `research`-tier enrichment that opens the author's LinkedIn profile to pull the verified current company:

```
enrichment_set(
  name="Current Company",
  columns=[
    {"name":"Company","type":"text"},
    {"name":"Company Domain","type":"text"},
    {"name":"Company LinkedIn","type":"url"},
    {"name":"Start Date","type":"date"}
  ],
  action={
    "research": "research",
    "prompt": "Open Author LinkedIn. Read the most recent role on the profile — that's the new position. Fill Company (org name), Company Domain (from the org's About), Company LinkedIn (the company's /company/ URL), and Start Date (when the new role started — visible on the experience entry).",
    "depends_on": ["Author LinkedIn"],
    "per_row_credit_cap": 5
  }
)
```

LinkedIn job tenures are reliable on the poster's own profile (they keep their current role accurate). The Post Date is a useful start-date FALLBACK if the profile is private — within a few days either way.

### Qualifying mid-market / B2B / location

`harvestapi/linkedin-post-search` doesn't filter by company size, geography, or B2B-ness. Two paths to qualify:

1. **After-the-fact via Apollo:** add an `apollo_org_enrich` column keyed on Company Domain — returns headcount, location, industry. Then filter rows where headcount ∈ [200, 2000] and location matches.
2. **Free-text classify:** ask a classify enrichment "Is this a US-based B2B company with 200-2000 employees" — cheaper but less reliable.

Prefer #1. Apollo enrich is free on our plan and authoritative.

### Verified email

Once Author Name + Company Domain are known, add a FullEnrich email column — same shape as the [[find_emails]] skill.

### Cost model

For a 400-row pull (default; 13 queries × 30 posts):
- Apify post search: $0.002 × 400 = **$0.80**
- Classify filter: $0.0002 × 400 = **$0.08**
- Filter result: ~160 real moves (40% signal); after Post Date >= 90d filter, ~80-100
- Research (current company / start date): $0.05 × 100 = **$5.00**
- Apollo enrich (qualify): free, ×100
- FullEnrich email: $0.05 × 100 = **$5.00**

Total ≈ **$11 for ~80-100 qualified recent-move leads with founder LinkedIn + verified email**.

### Why we don't scale by raising `maxPosts`

Burned by this once (project bbcbdbc2): bumping `maxPosts` from 30 → 100 returned 508 posts of which **zero were in the last 90 days**. LinkedIn's search ranks by relevance × engagement past the first ~30 results, so deeper paginating surfaces 6+ month-old viral posts that match the keywords but defeat the "recent" intent.

Scale via more `searchQueries`, not more per query. Each query is a fresh LinkedIn search whose top 30 are recency-dominant. 20 queries × 30 = 600 posts all in the recency zone; far better than 5 queries × 200 = 1000 posts mostly stale.

### Quirks to know

- **Author Position is often null** for poster profiles — don't depend on it. The classify pass should read Post Content, not Author Position.
- **Post Date format** is `postedAt.date` (ISO string under the `postedAt` object). Use `source_field="postedAt.date"`, not `postedAt`.
- **`searchQueries` not `search`** — singular `search` returns 400.
- **`postedLimit` / `sortBy`** params are accepted but unreliable — the actor often ignores `sortBy`. Always sort the table client-side after the fetch.
- **Same post can appear under multiple search queries** — the actor de-dupes by post URL within one run, but if you table_extend later you may see duplicates. Dedup_key should be Post URL.
