---
name: fetch-reddit
description: Fetching Reddit posts and/or comments — by keyword across Reddit, within one subreddit, or across a known set of subreddits. The general-purpose Reddit source.
applies_to: [orchestrator]
---

## Fetching from Reddit

Use this whenever the ask involves pulling Reddit content: "find posts about X", "what are people saying about Y", "complaints in r/Z", "recent posts asking for W", "comments mentioning V". For *listing communities* (subreddits about a topic, with subscriber counts) use [[find-subreddits]] instead — different job, different actor.

### The actor — use `apify_actor:clearpath/reddit-search-scraper`

This is the verified working path. It returns clean structured rows (posts and/or comments) with title, body, author, subreddit, score, comment count, timestamp, and URL. PAY_PER_EVENT and dirt cheap: ~$0.001 actor start + $0.00001 per result. 200 results costs about a cent.

**Do NOT cycle through other Reddit actors.** Every one of these returned **zero rows** on real lead-gen queries in the same session, burning turns:

- `trudax/reddit-scraper-lite` — 0 rows (the popular one; its keyword search just doesn't return)
- `trudax/reddit-scraper` — 403, requires paid rental
- `solidcode/reddit-scraper` — 0 rows for posts (it's only reliable for *communities* — that's what [[find-subreddits]] uses it for)
- `truefetch/reddit-post-search` — 0 rows
- `vulnv/reddit-posts-search-scraper`, `igview-owner/reddit-post-viewer` — unreliable
- `browser_use` on `reddit.com/search` — 0 rows (Reddit blocks it)
- `web_harvest` with `site:reddit.com` — thin/garbage results

Go straight to `clearpath/reddit-search-scraper`. If it returns zero, the lever is the **query**, not the actor (see below).

### Call shape

```
table_create(
  source="apify_actor:clearpath/reddit-search-scraper",
  query_params={
    "input": {
      "query": "(\"lead list\" OR \"leads list\" OR \"looking for leads\") NOT survey NOT academic",
      "maxResults": 200,
      "contentType": "both",          # posts | comments | both
      "sort": "new",                  # relevance | new | top | hot | comments
      "timeFilter": "month",          # '' | hour | day | week | month | year
      "autoDiscoverSubreddits": true,
      "maxSubreddits": 30
    }
  },
  columns=[
    {"name": "Type",        "source_field": "_type",        "type": "enum"},   # post | comment
    {"name": "Title",       "source_field": "title",        "type": "text"},
    {"name": "Body",        "source_field": "body",         "type": "text"},
    {"name": "Subreddit",   "source_field": "subreddit",    "type": "text"},
    {"name": "Author",      "source_field": "author",       "type": "text"},
    {"name": "Score",       "source_field": "score",        "type": "number"},
    {"name": "Comments",    "source_field": "commentCount", "type": "number"},
    {"name": "Date",        "source_field": "createdAt",    "type": "date"},
    {"name": "URL",         "source_field": "url",          "type": "url"},
  ],
  name="<topic> on Reddit"
)
```

Other output fields available to map if needed: `id`, `permalink`, `upvoteRatio`, `domain`, `isSelfPost`, `isNsfw`, `isSpoiler`, `isLocked`, `isStickied`, `flair`.

### The input knobs — cover every case with these

| Field | Values | Use it for |
|---|---|---|
| `query` | keywords; supports `"quoted phrases"` and boolean `(A OR B) NOT C`. Max 700 chars. | The search itself. Leave broad; filter after. |
| `contentType` | `posts` \| `comments` \| `both` | `posts` for threads/asks; `comments` for opinions/complaints/replies; `both` to catch everything. |
| `sort` | `relevance` \| `new` \| `top` \| `hot` \| `comments` | `new` = freshest (best for "who's asking right now"). `top`/`hot` = highest signal. `relevance` = best keyword match. |
| `timeFilter` | `''` \| `hour` \| `day` \| `week` \| `month` \| `year` | Recency window. `''` = all time. Tighten to `week`/`month` for intent/leads. |
| `subreddit` | e.g. `"python"` or `"r/python"` | Scope to ONE subreddit (browse/search inside it). Disables auto-discover. |
| `subreddits` | `["sales", "Entrepreneur", "msp"]` | Search a KNOWN set of subreddits in parallel. Disables auto-discover. |
| `autoDiscoverSubreddits` | `true`/`false` | `true` (default): find relevant communities for the query and search across them. The right choice when you don't know which subs to hit. |
| `maxSubreddits` | int (default 20) | How many communities auto-discovery hits. More = more results, slower. 30 is a good ceiling. |
| `maxResults` | int; `0` = unlimited | Cap. With auto-discovery a single keyword can return thousands, so set a real cap (200–500) unless the user wants everything. |

### Case → config

- **"Find posts about X across Reddit"** → `query=X`, `autoDiscoverSubreddits=true`, `contentType=posts`, `sort=new` or `top`.
- **"What are people saying about X / complaints about X"** → `contentType=both` or `comments`, `sort=top`. Comments carry the candid opinions.
- **"Recent posts asking for X right now"** (lead intent) → `sort=new`, `timeFilter=week`/`month`, tight boolean query with `NOT` exclusions to cut noise.
- **"Posts in r/Z"** → `subreddit="Z"` (auto-discover off automatically).
- **"Across r/A, r/B, r/C"** → `subreddits=["A","B","C"]`.
- **"Everything ever about X"** → `timeFilter=''`, `maxResults=0`, raise `maxSubreddits`.

### Query craft (this is the lever, not the actor)

- **Boolean OR to widen, quotes for phrases:** `("lead list" OR "leads list" OR "looking for leads")`. Unquoted multi-word terms match loosely.
- **`NOT` to cut predictable noise:** `... NOT survey NOT academic NOT hiring`. Reddit keyword search pulls adjacent junk; exclude it in the query rather than fetching then filtering when the noise term is obvious.
- **If a query returns 0 rows:** shorten it, drop the most niche OR-term, or widen `timeFilter`. Do NOT switch sources — the actor works; the query was too narrow.
- Keep the query broad-ish and do the precise membership call with a cheap classify-tier enrichment afterward (fetch wide → classify → filter). See the noise→signal pattern: fetch, classify Yes/No, `filter_set` on Yes.

### After the fetch

- Dedupe-by-author or by-subreddit if the user wants distinct people/communities.
- Classify intent with a `classify`-tier enrichment ("Is this person asking to BUY X? Yes/No") then filter — turns a noisy keyword pull into a clean lead list without losing data.
- `filter_set` on `Score >= N` or `Type = post` to trim low-signal rows.
- Drop NSFW (`isNsfw`) unless asked.
