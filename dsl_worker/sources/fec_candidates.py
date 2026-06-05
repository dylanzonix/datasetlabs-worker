"""fec_candidates — US federal campaign finance (FEC) candidate search.

The authoritative, complete, FREE source for federal political campaigns
(House / Senate / President). The FEC publishes every candidate, their
principal campaign committee, the committee's treasurer + contact email +
address + website, and full financials (receipts, disbursements, cash on
hand). No scraping, no browser — a clean public REST API at
api.open.fec.gov.

Why this exists: the orchestrator kept trying to `web_harvest` for campaign
lists (project 83698b95 "Arizona Campaign Leads" — 4.4min, $0.61, 0 rows).
Campaign data is a free API, not something to browse for. This adapter makes
it a first-class source.

Coverage: federal only. Per 2-year cycle there are ~8,000 candidates, of
which ~1,700 are seriously funded (>=$25k receipts). State / local races
(governor, state-leg, county, municipal) live in 50 separate state portals
with no unified API — out of scope here; see the find-political-campaigns
skill for the state-portal long tail.

Two-stage fetch:
  1. /candidates/totals/  — the spine: candidate + office + party + financials,
     in one paginated call. Great filters (state, cycle, office, party,
     min_receipts, incumbent/challenger), sorted by receipts so real
     campaigns come first.
  2. /committee/{id}/     — concurrent per-candidate contact join: principal
     committee's treasurer_name, email, website, city/state. Bounded
     concurrency, graceful — a row still returns its spine if the contact
     lookup fails or rate-limits.

API key: set FEC_API_KEY to a free api.data.gov key (1,000 calls/hour).
Falls back to DEMO_KEY (40/hour) with a loud warning — fine for a tiny
preview, useless for a real pull.
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any, Dict, List, Optional

import httpx

from dsl_worker.sources.base import FetchResult, SourceAdapter, SourceDescription, register


log = logging.getLogger(__name__)

FEC_BASE = "https://api.open.fec.gov/v1"

# Bounded concurrency for the per-candidate contact join. api.data.gov allows
# 1,000 calls/hour on a personal key; 8 in flight keeps us well clear while
# still resolving a 50-row page in a couple seconds.
_CONTACT_CONCURRENCY = 8


# Predictable column map — every row has this exact shape. source_field names
# are flattened keys we build in fetch() (FEC nests financials under the totals
# row and contact under the committee detail; we merge both into one flat row).
DEFAULT_COLUMNS = [
    {"source_field": "candidate_name", "column_name": "Candidate", "type": "text"},
    {"source_field": "office_full", "column_name": "Office", "type": "text"},
    {"source_field": "district", "column_name": "District", "type": "text"},
    {"source_field": "party_full", "column_name": "Party", "type": "text"},
    {"source_field": "incumbent_challenge_full", "column_name": "Status", "type": "text"},
    {"source_field": "receipts", "column_name": "Receipts", "type": "number"},
    {"source_field": "disbursements", "column_name": "Disbursements", "type": "number"},
    {"source_field": "cash_on_hand", "column_name": "Cash on Hand", "type": "number"},
    {"source_field": "committee_name", "column_name": "Committee", "type": "text"},
    {"source_field": "treasurer_name", "column_name": "Treasurer", "type": "text"},
    {"source_field": "contact_email", "column_name": "Committee Email", "type": "email"},
    {"source_field": "website", "column_name": "Website", "type": "url"},
    {"source_field": "city", "column_name": "City", "type": "text"},
    {"source_field": "state", "column_name": "State", "type": "text"},
    {"source_field": "fec_url", "column_name": "FEC Profile", "type": "url"},
    {"source_field": "candidate_id", "column_name": "FEC Candidate ID", "type": "text"},
]


# query_params the agent may pass. These map onto /candidates/totals/ filters,
# all confirmed against the live API. Anything else is rejected with a hint.
ALLOWED_PARAMS = {
    "state",                  # two-letter, e.g. "AZ"
    "cycle",                  # even election year, e.g. 2026
    "office",                 # "H" | "S" | "P" (House/Senate/President); list ok
    "party",                  # "DEM" | "REP" | "IND" | ...; list ok
    "min_receipts",           # viability floor, e.g. 25000
    "max_receipts",
    "incumbent_challenge",    # "I" incumbent | "C" challenger | "O" open-seat; list ok
    "is_active_candidate",    # bool; default true (drops withdrawn/paper candidates)
    "election_full",          # bool; default true (full election period, not just one year)
    "q",                      # name search
    "include_contact",        # bool; default true — set false to skip the committee join
}


class FECCandidatesAdapter(SourceAdapter):
    name = "fec_candidates"
    label = "FEC (federal campaigns)"
    favicon_url = "https://www.google.com/s2/favicons?domain=fec.gov&sz=32"
    predictable = True
    default_columns = DEFAULT_COLUMNS
    default_dedup_key_column = "candidate_id"

    def __init__(self) -> None:
        self.api_key = os.getenv("FEC_API_KEY")
        if not self.api_key:
            log.warning(
                "FEC_API_KEY not set — fec_candidates falling back to DEMO_KEY "
                "(40 calls/hour). Get a free key at https://api.data.gov/signup/"
            )
            self.api_key = "DEMO_KEY"

    def validate_query_params(self, query_params: Dict[str, Any]) -> Optional[str]:
        bad = [k for k in query_params if k not in ALLOWED_PARAMS]
        if bad:
            return (
                f"unknown fec_candidates params: {bad}. "
                f"Allowed: {sorted(ALLOWED_PARAMS)}"
            )
        if not query_params.get("cycle"):
            return "fec_candidates requires `cycle` (an even election year, e.g. 2026)."
        return None

    def describe(
        self,
        query_params: Dict[str, Any],
        source: Optional[str] = None,
    ) -> SourceDescription:
        qp = query_params or {}
        bits: List[str] = []
        office = qp.get("office")
        office_map = {"H": "House", "S": "Senate", "P": "President"}
        if office:
            offs = office if isinstance(office, list) else [office]
            bits.append("/".join(office_map.get(o, str(o)) for o in offs))
        if qp.get("party"):
            p = qp["party"]
            bits.append(", ".join(p) if isinstance(p, list) else str(p))
        if qp.get("state"):
            bits.append(str(qp["state"]))
        if qp.get("cycle"):
            bits.append(f"{qp['cycle']} cycle")
        headline = "Federal campaigns — " + " · ".join(bits) if bits else "FEC candidate search"

        FRIENDLY = {
            "state": "State", "cycle": "Cycle", "office": "Office", "party": "Party",
            "min_receipts": "Min receipts ($)", "max_receipts": "Max receipts ($)",
            "incumbent_challenge": "Incumbent/Challenger", "is_active_candidate": "Active only",
            "election_full": "Full election period", "q": "Name search",
            "include_contact": "Resolve committee contact",
        }
        details = "\n".join(
            f"- **{FRIENDLY.get(k, k)}:** {', '.join(map(str, v)) if isinstance(v, list) else v}"
            for k, v in qp.items()
        )
        return SourceDescription(
            kind=self.name, label=self.label, query_text=headline,
            details=details, favicon_url=self.favicon_url,
        )

    async def fetch(
        self,
        query_params: Dict[str, Any],
        n: int,
        prior_cursor: Optional[Dict[str, Any]] = None,
    ) -> FetchResult:
        qp = dict(query_params or {})
        include_contact = qp.pop("include_contact", True)
        # Defaults that make the result a clean "real campaigns" list rather
        # than 8,000 paper filings.
        qp.setdefault("is_active_candidate", True)
        qp.setdefault("election_full", True)

        per_page = min(100, max(1, n))
        page = int((prior_cursor or {}).get("page", 1))
        target = n

        spine: List[Dict[str, Any]] = []
        total_entries: Optional[int] = None

        async with httpx.AsyncClient(timeout=30.0) as client:
            while len(spine) < target:
                params = {
                    **qp,
                    "api_key": self.api_key,
                    "page": page,
                    "per_page": min(per_page, target - len(spine)),
                    "sort": "-receipts",
                    "sort_hide_null": "false",
                }
                resp = await client.get(f"{FEC_BASE}/candidates/totals/", params=params)
                if resp.status_code != 200:
                    log.warning("fec_candidates totals HTTP %s: %s", resp.status_code, resp.text[:200])
                    break
                data = resp.json() or {}
                results = data.get("results") or []
                if not results:
                    break
                for r in results:
                    spine.append(self._flatten_spine(r))
                pg = data.get("pagination") or {}
                total_entries = pg.get("count")
                page += 1
                if pg.get("pages") is not None and page > pg["pages"]:
                    break

            spine = spine[:target]

            # Concurrent contact join — resolve each candidate's principal
            # committee detail (treasurer, email, website, city/state).
            if include_contact and spine:
                sem = asyncio.Semaphore(_CONTACT_CONCURRENCY)

                async def _join(row: Dict[str, Any]) -> None:
                    async with sem:
                        try:
                            contact = await self._resolve_contact(
                                client, row["candidate_id"], qp.get("cycle")
                            )
                            row.update(contact)
                        except Exception as e:  # graceful: keep the spine
                            log.debug("fec contact join failed for %s: %s", row.get("candidate_id"), e)

                await asyncio.gather(*[_join(r) for r in spine])

        exhausted = total_entries is not None and (
            len(spine) + ((prior_cursor or {}).get("seen", 0)) >= total_entries
        )
        return FetchResult(
            rows=spine,
            schema=[c["source_field"] for c in DEFAULT_COLUMNS],
            cost_credits=0.0,  # FEC is free
            exhausted=exhausted,
            cursor={"page": page, "seen": ((prior_cursor or {}).get("seen", 0)) + len(spine)},
            dedup_key_column_hint="candidate_id",
            total_entries=total_entries,
        )

    def _flatten_spine(self, r: Dict[str, Any]) -> Dict[str, Any]:
        cid = r.get("candidate_id")
        district = r.get("district")
        # House rows carry a district number; Senate/President don't.
        district_str = ""
        if district and str(district) not in ("00", "0"):
            district_str = str(district).lstrip("0") or district
        return {
            "candidate_id": cid,
            "candidate_name": r.get("name"),
            "office_full": r.get("office_full"),
            "district": district_str,
            "party_full": r.get("party_full") or r.get("party"),
            "incumbent_challenge_full": r.get("incumbent_challenge_full"),
            "receipts": r.get("receipts"),
            "disbursements": r.get("disbursements"),
            "cash_on_hand": r.get("last_cash_on_hand_end_period"),
            "fec_url": f"https://www.fec.gov/data/candidate/{cid}/" if cid else None,
            # contact fields filled by _resolve_contact; defaults so the
            # column map always finds the key even when the join is skipped.
            "committee_name": None,
            "treasurer_name": None,
            "contact_email": None,
            "website": None,
            "city": None,
            "state": r.get("state"),
        }

    async def _resolve_contact(
        self, client: httpx.AsyncClient, candidate_id: str, cycle: Any
    ) -> Dict[str, Any]:
        """Resolve a candidate's principal campaign committee contact.

        Step 1: /committees/?candidate_id=&designation=P → principal committee
                 id (+ treasurer_name from the list row).
        Step 2: /committee/{id}/ → email, website, address (only on detail).
        """
        params = {
            "api_key": self.api_key,
            "candidate_id": candidate_id,
            "designation": "P",  # principal campaign committee
            "per_page": 1,
            "sort": "-last_file_date",
        }
        if cycle:
            params["cycle"] = cycle
        r1 = await client.get(f"{FEC_BASE}/committees/", params=params)
        if r1.status_code != 200:
            return {}
        results = (r1.json() or {}).get("results") or []
        if not results:
            return {}
        cm = results[0]
        cid = cm.get("committee_id")
        out: Dict[str, Any] = {
            "committee_name": cm.get("name"),
            "treasurer_name": cm.get("treasurer_name"),
        }
        if not cid:
            return out
        r2 = await client.get(f"{FEC_BASE}/committee/{cid}/", params={"api_key": self.api_key})
        if r2.status_code == 200:
            det = ((r2.json() or {}).get("results") or [{}])[0]
            out.update({
                "contact_email": det.get("email"),
                "website": det.get("website"),
                "city": det.get("city"),
                "state": det.get("state") or out.get("state"),
                "committee_name": det.get("name") or out["committee_name"],
                "treasurer_name": det.get("treasurer_name") or out["treasurer_name"],
            })
        return out


register(FECCandidatesAdapter())
