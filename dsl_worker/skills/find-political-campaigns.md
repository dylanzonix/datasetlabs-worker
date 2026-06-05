---
name: find-political-campaigns
description: Building lists of political campaigns and the people/vendors around them — candidates, their committees, treasurers, contact emails, financials, and (via enrichment) campaign staff and consultants. Covers federal (FEC) and notes the state/local long tail.
applies_to: [orchestrator]
---

## Finding political campaigns + their people

Use this when the user wants campaigns as leads — "Arizona 2026 campaign leads,"
"funded House challengers in swing districts," "campaign managers to sell our
tool to," "political consultants working 2026 races." The lead can be the
**campaign** (candidate + committee + treasurer email), the **staff** (campaign
manager, finance/data director — via enrichment), or the **vendors** (media,
fundraising, data firms — via disbursements).

### The trick: FEC has a free JSON API. Read it with `browser_use`.

Federal campaign finance is a complete, free REST API at `api.open.fec.gov`.
Do NOT `web_harvest` campaign lists (it fails — one real attempt ran 4.4 min,
cost $0.61, returned 0 rows). Instead, point a `browser_use` session straight
at an FEC **API URL** — the page renders as JSON and Browser Use parses it
cleanly. Verified: one session reads a 100-candidate page in ~30s for ~$0.17.

The API key goes **in the URL** (`api_key=...`). Use `DEMO_KEY` for light pulls
(it works from Browser Use's cloud IP), or a free api.data.gov key for heavy use.

### TAM (sanity-check coverage)

Federal, per 2-year cycle (`cycle=2026` = the 2025–2026 cycle):
- ~8,000 candidates file; **~1,700 are seriously funded (>=$25k receipts)** —
  that's the real addressable list. Always filter with `min_receipts` so you
  get campaigns, not 6,000 paper filings.
- Per state it scales down: AZ 2026 = 180 candidates → **33 funded**.
- PACs / party committees / super PACs are a *separate* universe — ask first.

State + local (governor, AG, state-leg, county, municipal) is a much larger
long tail but has **no unified API** — 50 separate state portals (AZ "See The
Money", CA Cal-Access, TX TEC…). Not covered here. If the user wants state/local,
say so and treat it as a per-state browser job, one portal at a time.

### Step 1 — fetch the campaign spine (`browser_use` on the FEC API URL)

Build the API URL from the user's filters, then `table_create`:

```
source = "browser_use"
url    = "https://api.open.fec.gov/v1/candidates/totals/?api_key=DEMO_KEY"
         "&cycle=2026&state=AZ&office=H&min_receipts=25000"
         "&is_active_candidate=true&election_full=true"
         "&sort=-receipts&per_page=100"
task   = "This page is a JSON API response from the FEC. Read results[]. "
         "Return one row per candidate with fields: name, office_full, party, "
         "incumbent_challenge_full, receipts, disbursements, "
         "last_cash_on_hand_end_period, district, candidate_id. "
         "Just parse the JSON shown on the page; do not click anything."
```

URL filters (all optional except `cycle`):
- `cycle` — even election year (required), e.g. `2026`
- `state` — two-letter, e.g. `AZ`. Omit for national.
- `office` — `H` House / `S` Senate / `P` President (repeat the param for several)
- `party` — `DEM` / `REP` / `IND` …
- `min_receipts` — **use this**, the viability floor (e.g. `25000`)
- `incumbent_challenge` — `I` incumbent / `C` challenger / `O` open-seat
- `q` — name search
- always add `is_active_candidate=true&election_full=true&sort=-receipts&per_page=100`

Each `browser_use` session is **one-shot = one page**. For >100 results,
`table_extend` with the same URL but `&page=2`, `&page=3`, … Stop when a page
returns fewer than 100 rows. (National funded ≈ 1,700 = ~18 pages ≈ ~$3.)

The keys above are the exact JSON field names — name them in the task so the
returned columns are predictable, then `column_map_set` to friendly labels
(Candidate, Office, Party, Status, Receipts, Cash on Hand, District, FEC ID).
Set the dedup key to `candidate_id`.

### Step 2 — funnel, THEN enrich the survivors

The spine is cheap; contact/staff lookups are per-row, so filter first (by
Receipts / Status — see the funnel discipline in the main prompt) and only
enrich the rows that matter.

**Committee contact (treasurer + email + website).** A per-row `browser_use`
enrichment keyed on `candidate_id`. One session, two hops, ~$0.08/row:
```
task = "Navigate to https://api.open.fec.gov/v1/candidate/{candidate_id}/"
       "committees/?api_key=DEMO_KEY&designation=P — from results[0] read "
       "committee_id. Then navigate to https://api.open.fec.gov/v1/committee/"
       "{committee_id}/?api_key=DEMO_KEY and return treasurer_name, email, "
       "website, city, state from results[0]. Parse JSON only."
```
Verified output: treasurer name, a real contact email, the campaign website.
(The email is often the treasurer/compliance firm — still reachable.)

**Campaign staff (manager, finance/data director, comms).** FEC payroll is
inconsistent, so get people from the **Website** (from the contact step): a
`fetch_url` enrichment on the campaign site's Team/About/Contact page, then
extract names + roles. Fall back to a LinkedIn enrichment ("campaign manager
at {Committee}"). Then run the email enrichment (FullEnrich) on
`{first} {last} @ {campaign domain}`.

**Vendors / consultants** (media, canvassing, fundraising, digital, data, legal
firms — themselves a B2B lead pool). A per-row `browser_use` enrichment on
`https://api.open.fec.gov/v1/schedules/schedule_b/?api_key=DEMO_KEY&committee_id={committee_id}&sort=-disbursement_amount&per_page=100`
— each record has `recipient_name` + `disbursement_description` + amount. Dedup
recipients by category keyword (MEDIA, CONSULT, FUNDRAIS, DIGITAL, DATA,
CANVASS). Only do this if the user wants vendors.

### Honesty

If a fetch or extend adds 0 rows, **say so** — relay the true count the tool
returns, never a number you didn't add.
