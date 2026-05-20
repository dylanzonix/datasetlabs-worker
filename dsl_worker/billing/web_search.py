"""OpenAI hosted web_search tool — billing helper.

Centralizes the per-call fee for the hosted `web_search` tool so all
three callsites (orchestrator agent, cell agent, web_harvest source)
read the same constant and use the same accounting logic. Before this
existed each site duplicated the loop and (in web_harvest's case)
silently skipped the per-call fee entirely.

# OpenAI's web_search billing model (as of May 2026)

The hosted `web_search` tool in the Responses API has TWO cost
components:

1. **Per-call tool fee** — billed separately from token cost. OpenAI's
   public rates as of this writing:
     - `{"type": "web_search"}` (standard, reasoning models):  $10/1K calls
     - `{"type": "web_search_preview"}` (preview, reasoning):  $10/1K calls
     - `{"type": "web_search_preview"}` (preview, non-reasoning, legacy): $25/1K calls

2. **Search content tokens** — the model reads search results as input
   tokens. For most current models these show up in
   `response.usage.input_tokens` and ARE captured by TrackedClient's
   token-cost path. For gpt-4o-mini / gpt-4.1-mini specifically, OpenAI
   bills a fixed 8K-token block per call regardless of actual content
   size (also accounted via input_tokens). So we do NOT need to add a
   separate token line for search content — the token path already has
   it.

# The hidden sub-search problem

OpenAI's web_search tool routinely runs MULTIPLE internal sub-searches
per visible call. The user's prompt is one `web_search_call` item in
`response.output`, but the tool may dispatch several queries
internally to satisfy it. Each sub-search is billed.

Discovered by community via dashboard reconciliation (e.g.
community.openai.com/t/1236954): observed cost is typically 2-3x the
advertised per-call rate. The sub-searches are NOT surfaced in
response.output OR response.usage.

# Our rate choice

We bill `WEB_SEARCH_CALL_COST_USD = 0.025` per VISIBLE call. That's
~2.5x the advertised $0.010 standard rate — an empirical
approximation of the sub-search multiplier so our cost ledger lines
up with OpenAI's actual bill instead of undercharging. If OpenAI
later exposes per-sub-search counts in response.usage we should drop
the multiplier and read the truth.

Tunable from one place. If you observe consistent over- or under-
billing against the OpenAI dashboard, edit this constant.
"""

from __future__ import annotations

from typing import Iterable


# Per visible web_search_call in response.output. See module docstring
# for the derivation — this bakes in OpenAI's sub-search multiplier
# observed empirically by the community.
WEB_SEARCH_CALL_COST_USD: float = 0.025


def count_web_search_calls(response_output: Iterable) -> int:
    """Return the number of web_search_call items in a Responses API
    output array. Robust to missing `.type` (treats those as not a
    web_search_call). Sub-searches inside a single visible call are
    not counted here — the constant above bakes them into the per-call
    rate."""
    n = 0
    for item in response_output or ():
        if getattr(item, "type", None) == "web_search_call":
            n += 1
    return n


def web_search_cost_usd(response_output: Iterable) -> float:
    """Convenience: count visible web_search_call items × per-call fee.
    Call after `client.responses_create(...)` and ADD the result to
    whatever the token-cost path already computed.

    Returns 0.0 if the response had no web_search_call items, so this
    is safe to call on every responses.create regardless of whether
    web_search was passed in tools."""
    return count_web_search_calls(response_output) * WEB_SEARCH_CALL_COST_USD
