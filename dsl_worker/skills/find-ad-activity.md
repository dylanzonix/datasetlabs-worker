---
name: find-ad-activity
description: Whether a company runs paid ads and which platforms.
applies_to: [cell_agent]
---

## Detecting paid-ad activity for a company

Skip if this doesn't fit the column you're filling.

### Inputs

A `Domain` or `Website` in the row. Without one, return null and stop — no domain means nothing to inspect.

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
