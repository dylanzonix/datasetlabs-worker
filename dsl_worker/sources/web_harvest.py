"""web_harvest — bounded open-web research that returns rows.

For niche topics where no API integration covers the data shape. Uses OpenAI's
Responses API with the native web_search tool to run a bounded research turn
and return up to N candidates as JSON rows. Unpredictable source — the row
schema depends on candidate_description, so agent calls column_map_set after
inspecting the preview.
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Dict, List, Optional

from dsl_worker.sources.base import FetchResult, SourceAdapter, SourceDescription, register


log = logging.getLogger(__name__)


class WebHarvestAdapter(SourceAdapter):
    name = "web_harvest"
    label = "Web Research"
    favicon_url = "https://www.google.com/s2/favicons?domain=google.com&sz=32"
    predictable = False

    def describe(
        self,
        query_params: Dict[str, Any],
        source: Optional[str] = None,
    ) -> SourceDescription:
        qp = query_params or {}
        query = str(qp.get("query") or "Web research")
        desc = str(qp.get("candidate_description") or "")
        details_parts = []
        if desc:
            details_parts.append(f"**What a row looks like:** {desc}")
        if qp.get("max_candidates"):
            details_parts.append(f"**Max candidates:** {qp['max_candidates']}")
        if qp.get("continuation_hint"):
            details_parts.append(f"**Avoid prior coverage:** {qp['continuation_hint']}")
        return SourceDescription(
            kind=self.name,
            label=self.label,
            query_text=query,
            details="\n\n".join(details_parts),
            favicon_url=self.favicon_url,
        )

    def validate_query_params(self, query_params: Dict[str, Any]) -> Optional[str]:
        required = {"query", "candidate_description"}
        missing = required - set(query_params)
        if missing:
            return f"web_harvest requires {sorted(required)}; missing: {sorted(missing)}"
        return None

    async def fetch(
        self,
        query_params: Dict[str, Any],
        n: int,
        prior_cursor: Optional[Dict[str, Any]] = None,
    ) -> FetchResult:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            log.warning("OPENAI_API_KEY not set — web_harvest adapter inert")
            return FetchResult(rows=[], schema=[], cost_credits=0.0, exhausted=True)

        query = query_params["query"]
        candidate_description = query_params["candidate_description"]
        max_candidates = int(query_params.get("max_candidates", min(n, 30)))
        continuation_hint = query_params.get("continuation_hint")
        # Structured exclude list — same shape as the llm source. Each
        # entry is a unique identifier for a row already in the table
        # (typically the dedup-key value). Cleaner than free-form
        # continuation_hint for the common "extend without re-pulling"
        # case: the agent passes the existing row names and the LLM
        # skips them by exact-match.
        exclude_list = query_params.get("exclude") or []
        if not isinstance(exclude_list, list):
            exclude_list = []
        # When table_extend reuses this adapter, the table's existing
        # column_map is reapplied to every new row. If the LLM picks
        # DIFFERENT keys this time, the mapped cells are mostly empty
        # (the column map's source_field paths won't find the new keys).
        # tools.table_extend stuffs the prior keys here so the prompt
        # can pin the schema. Empty/missing on first-fetch table_create.
        existing_schema = query_params.get("__existing_schema") or []

        from dsl_worker.billing.tracked_client import TrackedOpenAIClient
        from openai import AsyncOpenAI
        client = TrackedOpenAIClient(AsyncOpenAI(api_key=api_key))
        model = os.getenv("OPENAI_MODEL_MINI", "gpt-5.4-mini")

        prompt_parts = [
            f"Research the web and return up to {max_candidates} candidates for the following query.",
            f"\nQuery: {query}",
            f"\nWhat a successful row looks like: {candidate_description}",
            "\nReturn ONLY a JSON array of objects. Each object is one candidate row.",
            "Use consistent keys across rows. Include a `url` key if you have a source URL for the candidate.",
            "No prose, no markdown, no preamble — just the JSON array.",
        ]
        if existing_schema:
            # Lock the schema to the prior batch's keys. The downstream
            # column_map_set is reused on extends, so any key the LLM
            # invents that's NOT in this list will land in raw_row but
            # not in the mapped row.
            prompt_parts.append(
                "\nMANDATORY SCHEMA — every row MUST use these EXACT keys "
                f"(no renames, no synonyms): {existing_schema}.\n"
                "If a value is unknown for a row, return null for that key. "
                "Do not add extra keys."
            )
        if continuation_hint:
            prompt_parts.append(f"\nAvoid candidates from this prior coverage: {continuation_hint}")
        if exclude_list:
            ex_sample = exclude_list[:80]
            prompt_parts.append(
                "\nDO NOT include any of these (already in the table — duplicates wasted): "
                + ", ".join(str(x) for x in ex_sample)
                + (
                    f" — and {len(exclude_list) - 80} more not listed."
                    if len(exclude_list) > 80 else ""
                )
            )
        prompt = "\n".join(prompt_parts)

        try:
            resp, usage_cost = await client.responses_create(
                model=model,
                input=[{"role": "user", "content": prompt}],
                tools=[{"type": "web_search"}],
            )
            llm_cost_usd = float(getattr(usage_cost, "total_cost_usd", 0.0) or 0.0)
        except Exception as e:
            log.exception("web_harvest LLM call failed: %s", e)
            return FetchResult(rows=[], schema=[], cost_credits=0.0, exhausted=True)

        # OpenAI's hosted web_search tool is billed per call separately
        # from token cost. TrackedClient only counts the token side, so
        # we sum the per-call fee here. See dsl_worker/billing/web_search.py
        # for the full billing model (advertised rate, sub-search
        # multiplier, why we use 0.025).
        try:
            from dsl_worker.billing.web_search import web_search_cost_usd
            web_search_call_cost_usd = web_search_cost_usd(resp.output or [])
        except Exception:
            # Best-effort; never crash the harvest on a billing tally.
            log.exception("web_harvest: web_search_call cost tally failed")
            web_search_call_cost_usd = 0.0

        text = (resp.output_text or "").strip()
        if text.startswith("```"):
            text = text.strip("`")
            if text.lower().startswith("json"):
                text = text[4:].strip()

        rows: List[Dict[str, Any]] = []
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                rows = [r for r in parsed if isinstance(r, dict)]
        except Exception:
            m = re.search(r"\[\s*\{.*\}\s*\]", text, re.DOTALL)
            if m:
                try:
                    parsed = json.loads(m.group(0))
                    if isinstance(parsed, list):
                        rows = [r for r in parsed if isinstance(r, dict)]
                except Exception:
                    pass

        if not rows:
            log.warning("web_harvest: no parsable rows from model output")

        schema_keys = sorted({k for r in rows for k in r.keys()})[:40]
        # Total cost = token cost from TrackedOpenAIClient + per-call
        # web_search fee (hosted tool, billed separately by OpenAI).
        # Convert USD → credits at 1 credit = $0.10 of compute.
        total_cost_usd = llm_cost_usd + web_search_call_cost_usd
        return FetchResult(
            rows=rows[:n],
            schema=schema_keys,
            cost_credits=total_cost_usd * 10.0,
            exhausted=True,  # one-shot; agent passes a fresh continuation_hint for more
            cursor=None,
            dedup_key_column_hint="url" if "url" in schema_keys else None,
        )


register(WebHarvestAdapter())
