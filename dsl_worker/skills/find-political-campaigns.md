---
name: find-political-campaigns
description: Building lists of political campaigns and the people/vendors around them — candidates, their committees, treasurers, contact emails, financials, and (via enrichment) campaign staff and consultants. Covers federal (FEC, full API) and notes the state/local long tail.
applies_to: [orchestrator]
---

## Finding political campaigns + their people

Use this when the user wants campaigns as leads — "Arizona 2026 campaign leads,"
"funded House challengers in swing districts," "campaign managers to sell our
tool to," "political consultants / media firms working 2026 races." The lead
can be the **campaign** (candidate + committee + treasurer), the **staff**
(campaign manager, finance/data director — via enrichment), or the **vendors**
(media, canvassing, fundraising, data firms — via disbursements).

### Do NOT web_harvest this. It is a free API.

The single most important rule. `web_harvest` / browser on campaign data fails
(real example: a "Arizona Campaign Leads" extend ran 4.4 min, cost $0.61, and
returned **0 rows**). Federal campaign finance is a clean, complete, free REST
API. Always use the `fec_candidates` source.

### TAM (so you can sanity-check coverage)

Federal, per 2-year cycle (`cycle=2026` = the 2025–2026 cycle):
- ~8,000 candidates file; ~7,900 "active."
- **~1,700 are seriously funded (>=$25k receipts)** — that's the real
  addressable federal list. Filter with `min_receipts` so you get campaigns,
  not 6,000 paper filings.
- Per state it scales down: AZ 2026 had 180 candidates → **33 funded** active.
- Plus thousands of PACs, party committees, leadership PACs, super PACs (not
  candidate campaigns — different universe, ask before pulling).

State + local (governor, AG, state-leg, county, municipal) is a *much larger*
long tail but has **no unified API** — each of the 50 states runs its own
disclosure portal (AZ "See The Money", CA Cal-Access, TX TEC, NY BOE…). Out of
scope for the `fec_candidates` source. If the user explicitly wants state/local,
say so plainly and treat it as a per-state browser job, one portal at a time.

### Setup

Needs `FEC_API_KEY` (a free api.data.gov key, 1,000 calls/hour) in the worker
env. Without it the adapter falls back to `DEMO_KEY` (40 calls/hour) — only
enough for a tiny preview. If pulls come back empty with rate-limit warnings,
that's a missing key, not a bad query.

### The source: `fec_candidates`

`table_create(source="fec_candidates", query_params={...})`. Predictable
schema — rows commit immediately, no `column_map_set` round-trip.

Each row already includes: **Candidate, Office, District, Party, Status
(incumbent/challenger/open), Receipts, Disbursements, Cash on Hand, Committee,
Treasurer, Committee Email, Website, City, State, FEC Profile, FEC Candidate
ID.** Sorted by receipts (biggest campaigns first). The Committee Email is the
treasurer/compliance contact — a real reachable address (often a compliance
firm, not the candidate). The Website is the campaign site — the seed for
finding staff.

Query params (all optional except `cycle`):

| param | example | notes |
|---|---|---|
| `cycle` | `2026` | **required.** Even election year. |
| `state` | `"AZ"` | two-letter. Omit for national. |
| `office` | `"H"` / `"S"` / `"P"` | House / Senate / President. List ok. |
| `party` | `"DEM"` / `"REP"` | list ok. |
| `min_receipts` | `25000` | **use this** — the viability floor that turns 8k filings into real campaigns. |
| `incumbent_challenge` | `"C"` | `I` incumbent, `C` challenger, `O` open-seat. |
| `q` | `"Ansari"` | name search. |
| `include_contact` | `false` | default true. Set false for a fast spine-only pull (skips the per-candidate committee lookup). |

Example call shapes:
- "Funded AZ 2026 House campaigns" → `{cycle:2026, state:"AZ", office:"H", min_receipts:25000}`
- "Democratic Senate challengers nationally, 2026" → `{cycle:2026, office:"S", party:"DEM", incumbent_challenge:"C", min_receipts:50000}`

### Funnel: cheap fetch → enrich the survivors

The fetch is the spine. Layer enrichments only on the rows that matter (filter
by Receipts / Status first — see the funnel discipline in the main prompt):

1. **Campaign staff (manager, finance/data director, comms).** FEC payroll is
   inconsistent, so get people from the **Website**: a `fetch_url` enrichment on
   the campaign site's "Team"/"About"/"Contact" page, then a classify/extract to
   pull names + roles. Fall back to a LinkedIn enrichment ("campaign manager at
   {Committee}").
2. **Staff emails.** Once you have a name + the campaign domain, run the email
   enrichment (FullEnrich) on `{first} {last} @ {campaign domain}`.
3. **Vendors / consultants** (media, canvassing, fundraising, digital, data,
   legal firms). These are themselves a B2B lead universe. They come from the
   committee's itemized disbursements:
   `GET /schedules/schedule_b/?committee_id={id}&sort=-disbursement_amount` —
   each record has `recipient_name` + `disbursement_description` + amount. If the
   user wants vendors, pull schedule_b per committee (code_exec against the FEC
   API) and dedup recipients by category keyword (MEDIA, CONSULT, FUNDRAIS,
   DIGITAL, DATA, CANVASS).

### Honesty

If a fetch or extend adds 0 rows, **say so** — don't report a count you didn't
add. (The adapter and table_extend now report true counts; relay them as-is.)
