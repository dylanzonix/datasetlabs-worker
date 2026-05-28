---
name: find-ad-activity
description: Whether a company runs paid ads and which platforms. Includes Apollo-source filter for narrowing the pool upfront.
applies_to: [orchestrator, cell_agent]
---

## Filter at Apollo FIRST when "runs paid ads" is a hard filter

When the user wants companies *that run paid ads* (not just *enriched with ad info*), don't set up a per-row verification enrichment over a broad Apollo pool — that wastes ~$0.05/row to drop 90% of the table. Filter at the Apollo source instead.

Apollo's `currently_using_any_of_technology_uids` accepts ad-tech UIDs and narrows total_entries in one call. Verified working UIDs (probed against US + 100-1000-employee B2B baseline of ~48k companies):

```
google_ads      ~2% of base   "Running Google paid search" — cleanest signal
doubleclick     ~13%          Google's ad-tech backbone, broader (overlaps google_ads)
linkedin_ads    ~4%           Has LinkedIn Insight Tag installed
bing_ads        ~6%           Microsoft Ads
criteo          ~0.7%         Retargeting (DTC-heavy)
adroll          ~0.5%         Retargeting (B2B SaaS-heavy)
```

For a "running paid search ads" filter, **`["google_ads"]` is the right primary** — narrowest, cleanest, no ambiguity.
For "running ads on Google in any form," **`["google_ads", "doubleclick"]`** (OR semantics — Apollo dedupes overlap).
For "running ads anywhere," **`["google_ads", "doubleclick", "linkedin_ads", "bing_ads"]`**.

**Do NOT use**: `facebook_pixel` (real UID but only 22 hits across 48k — Apollo has very thin coverage of Meta-side ad-tech). `facebook_ads`, `meta_pixel`, `linkedin_insight_tag`, `tiktok_pixel`, `twitter_pixel`, `hubspot_ads` — all return 0 (not valid Apollo UIDs).

Empirical narrowing on a real prompt: US + 100-1000 employees + B2B + SDR jobs last 30d = **1,495 companies**. Add `["google_ads"]` → **149 companies** (90% drop, free). Skip this filter and you'd pay ~$0.05 verify × 1495 rows = ~$75 to do the same drop downstream via per-row enrichment.

When to fall back to the per-row enrichment below:
- The user explicitly asked to *enrich* an existing table with ad info (column on a non-Apollo table)
- Apollo's tech filter returned 0 — try broadening from `google_ads` → `doubleclick`
- The user wants the ad **platforms list** as an output column (per-row enrichment fills the list; the Apollo filter just gates membership)
- The user wants Meta/Facebook ad detection — Apollo's coverage is too thin; per-row BuiltWith is the path (it has full Facebook Pixel coverage)

## Detecting paid-ad activity for a company

Skip if this doesn't fit the column you're filling.

### Inputs

A `Domain` or `Website` in the row. Without one, return null and stop — no domain means nothing to inspect.

**Use the row's domain only.** Derive `<domain>` directly from the row's `Website URL` / `Domain` field (strip protocol + `www.`). Do NOT BuiltWith adjacent or competitor companies surfaced by web_search — those are different businesses. One call, on the row's domain.

### Step 1: BuiltWith Apify actor (preferred)

Call `apify_call_actor`:

- `actor_id`: `builtwith/builtwith-official-technology-scraper`
- `input`: `{"startDomains": ["<domain>"]}`
- Cost: ~$0.002/row, ~2–3s

Response shape:

```json
{"domain": "weremoto.com", "techs": [
  {"name": "Facebook Pixel", "tag": "analytics", "categories": ["Retargeting"], ...},
  {"name": "Google Tag Manager", "tag": "widgets", "categories": ["Tag Management"], ...}
]}
```

### Classifying "runs paid ads"

**YES** if `techs` contains any entry whose `name` matches one of these (case-insensitive substring is fine):

- Facebook Pixel, Facebook Custom Audiences, Facebook Conversion Tracking, Facebook Domain Insights
- Google AdWords Conversion, Google Conversion Tracking, Google Remarketing, Google Dynamic Remarketing, DoubleClick
- LinkedIn Ads, LinkedIn Insights
- TikTok Conversion Tracking Pixel
- Twitter Conversion Tracking
- Reddit Conversion Tracking
- Pinterest Conversion Tracking
- Bing Universal Event Tracking, Microsoft Advertising
- Criteo, AdRoll, Simpli.fi, Hubspot Ads, Broad Street Ads

Otherwise **NO**.

**Do NOT** count as ad activity:

- *Google Analytics, Google Tag Manager, Global Site Tag* — these are tag/analytics infrastructure. Present on most sites whether they advertise or not.
- *Google AdSense, Google Publisher Tag, Media.net* — these are ad **monetization** (the site displays ads), not running ads.

### "Ad Platforms" column

If filling a list-of-platforms column, extract platform names from matched entries and normalize:

- Facebook Pixel / Facebook Custom Audiences / Facebook Conversion → **Meta**
- Google AdWords / Google Remarketing / DoubleClick → **Google**
- LinkedIn Ads / LinkedIn Insights → **LinkedIn**
- TikTok Conversion → **TikTok**
- Twitter Conversion → **X / Twitter**
- Reddit Conversion → **Reddit**
- Pinterest Conversion → **Pinterest**
- Bing Universal Event Tracking / Microsoft Advertising → **Microsoft**

Dedupe; return as a list.

### Step 2 (fallback): Apollo current_technologies

If BuiltWith fails or returns zero techs, call `apollo_org_enrich(domain=<domain>)`. Response includes `current_technologies` — a similar array. Apply the same classification list. Apollo's tech taxonomy is narrower than BuiltWith's, so BuiltWith is preferred when both succeed.

### Strict no-pixel rule

If BuiltWith returns zero techs from the strict ad-name list, **the answer is NO** (or null if even non-ad techs are empty). Do not write YES based on:

- Marketing language on the homepage ("we advertise locally").
- Web_search snippets that mention the company runs ads.
- Google Maps "ads enabled" indicators or Local Services Ads badges.
- Inference from the company's category ("HVAC companies usually advertise").

Pixel presence is the contract. No pixel → no YES. The whole point of this skill is to replace inference with ground-truth detection.

### When null is right

After both BuiltWith and Apollo return no matches (or both fail), null is correct. Don't web_search for ads — public ad libraries (Meta Ad Library, Google Ads Transparency Center) require platform-specific calls and don't give a clean Yes/No fast enough at cell scale.

### Source citation

When committing the cell, cite the BuiltWith actor as the source:

```
{"type": "source_record", "source": "apify_actor:builtwith/builtwith-official-technology-scraper"}
```

Or Apollo if that's what answered:

```
{"type": "source_record", "source": "apollo_companies"}
```
