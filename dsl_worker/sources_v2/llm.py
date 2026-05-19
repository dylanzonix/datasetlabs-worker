"""llm — pure-LLM row generation.

For ideation, brainstorming, and pattern-continuation use cases where the user
wants the model itself to produce candidates rather than retrieving from a
directory or scraping. No tools — straight Responses API call returning a
JSON array of rows.

Use cases that fit this source:

- "Come up with 30 ICP ideas for a developer-tools startup."
- "Generate 50 angle/hook ideas for a cold email sequence."
- "Continue this list of categories" (table_extend with an `exclude` hint).
- "Come up with ideas then look for companies matching those ideas" — first
  table is llm (ideas), then the agent creates downstream apollo/web tables
  feeding off the ideas table's rows.

Unpredictable source — the row schema follows `candidate_description` /
`columns_hint`, so `table_create` returns a preview and the agent calls
`column_map_set` to lock the column names if the passthrough isn't right.
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Dict, List, Optional

from dsl_worker.sources_v2.base import FetchResult, SourceAdapter, SourceDescription, register


log = logging.getLogger(__name__)


class LLMAdapter(SourceAdapter):
    name = "llm"
    label = "LLM Generation"
    favicon_url = "https://www.google.com/s2/favicons?domain=openai.com&sz=32"
    predictable = False

    def describe(
        self,
        query_params: Dict[str, Any],
        source: Optional[str] = None,
    ) -> SourceDescription:
        qp = query_params or {}
        query = str(qp.get("prompt") or "LLM generation")
        details_parts = []
        if qp.get("candidate_description"):
            details_parts.append(f"**What a row looks like:** {qp['candidate_description']}")
        if qp.get("columns_hint"):
            details_parts.append(f"**Columns:** {', '.join(qp['columns_hint'])}")
        if qp.get("exclude"):
            sample = qp["exclude"] if isinstance(qp["exclude"], list) else [qp["exclude"]]
            details_parts.append(f"**Excluding prior:** {', '.join(str(x) for x in sample[:5])}{'…' if len(sample) > 5 else ''}")
        return SourceDescription(
            kind=self.name,
            label=self.label,
            query_text=query[:200],
            details="\n\n".join(details_parts),
            favicon_url=self.favicon_url,
        )

    def validate_query_params(self, query_params: Dict[str, Any]) -> Optional[str]:
        if not query_params.get("prompt"):
            return "llm requires a `prompt` describing what rows to generate"
        return None

    @classmethod
    def query_params_schema(cls) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": "What to generate. Be explicit about the entity type and any constraints (e.g. 'B2B SaaS ICP archetypes for a Series-A startup selling to engineering teams').",
                },
                "candidate_description": {
                    "type": "string",
                    "description": "Optional shape of one row in plain words (e.g. 'an ICP archetype with company size, vertical, pain, and example companies').",
                },
                "columns_hint": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional list of column names to bias the model toward. The model will use these as JSON keys.",
                },
                "examples": {
                    "type": "array",
                    "items": {"type": "object"},
                    "description": "Optional few-shot rows to anchor the schema and style.",
                },
                "exclude": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional list of items to skip (e.g. names already in the table on table_extend).",
                },
                "model": {
                    "type": "string",
                    "description": "Optional model override. Defaults to OPENAI_MODEL_MINI.",
                },
                "temperature": {
                    "type": "number",
                    "description": "Optional sampling temperature. Higher for more diversity on extends.",
                },
            },
            "required": ["prompt"],
            "additionalProperties": True,
        }

    @classmethod
    def tool_description(cls) -> str:
        return (
            "Generate rows from the model directly. No retrieval — pure synthesis. "
            "Use for ideation/brainstorming where the answer is the model's structured guess, "
            "not a lookup against an external source."
        )

    async def fetch(
        self,
        query_params: Dict[str, Any],
        n: int,
        prior_cursor: Optional[Dict[str, Any]] = None,
    ) -> FetchResult:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            log.warning("OPENAI_API_KEY not set — llm adapter inert")
            return FetchResult(rows=[], schema=[], cost_credits=0.0, exhausted=True)

        prompt = query_params["prompt"]
        candidate_description = query_params.get("candidate_description") or ""
        columns_hint = query_params.get("columns_hint") or []
        examples = query_params.get("examples") or []
        exclude = query_params.get("exclude") or []
        temperature = query_params.get("temperature")

        from dsl_worker.billing.tracked_client import TrackedOpenAIClient
        from openai import AsyncOpenAI
        client = TrackedOpenAIClient(AsyncOpenAI(api_key=api_key))
        model = query_params.get("model") or os.getenv("OPENAI_MODEL_MINI", "gpt-5.4-mini")

        prompt_parts = [
            f"Generate up to {n} rows for the following request.",
            f"\nRequest: {prompt}",
        ]
        if candidate_description:
            prompt_parts.append(f"\nWhat one row looks like: {candidate_description}")
        if columns_hint:
            prompt_parts.append(f"\nUse these JSON keys for each row: {', '.join(columns_hint)}")
        if examples:
            prompt_parts.append(
                "\nExample rows (match this schema / style):\n"
                + json.dumps(examples[:5], indent=2)
            )
        if exclude:
            ex_sample = exclude[:50]
            prompt_parts.append(
                "\nDo NOT include any of these (already covered): "
                + ", ".join(str(x) for x in ex_sample)
            )
        prompt_parts.append(
            "\nReturn ONLY a JSON array of objects. Each object is one row. "
            "Use the SAME keys across every row so they form a clean table. "
            "No prose, no markdown, no preamble — just the JSON array."
        )
        full_prompt = "\n".join(prompt_parts)

        kwargs: Dict[str, Any] = {
            "model": model,
            "input": [{"role": "user", "content": full_prompt}],
        }
        if temperature is not None:
            kwargs["temperature"] = float(temperature)

        try:
            resp, usage_cost = await client.responses_create(**kwargs)
            llm_cost_usd = float(getattr(usage_cost, "total_cost_usd", 0.0) or 0.0)
        except Exception as e:
            log.exception("llm source call failed: %s", e)
            return FetchResult(rows=[], schema=[], cost_credits=0.0, exhausted=True)

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
            log.warning("llm source: no parsable rows from model output")

        schema_keys = sorted({k for r in rows for k in r.keys()})[:40]
        # 1 credit = $0.10 of compute.
        return FetchResult(
            rows=rows[:n],
            schema=schema_keys,
            cost_credits=llm_cost_usd * 10.0,
            exhausted=True,  # one-shot; agent extends by passing `exclude` from current rows
            cursor=None,
        )


register(LLMAdapter())
