"""
Tracked Anthropic client — Claude-native equivalent of TrackedOpenAIClient.

Thin wrapper over the Anthropic SDK: executes messages.create (or beta for MCP),
computes cost from the response's usage, returns (response, UsageCost).

Retries are handled by the Anthropic SDK itself (default max_retries=2).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

from anthropic import AsyncAnthropic

from dsl_worker.billing.pricing import UsageCost, get_pricing_config

logger = logging.getLogger(__name__)


class TrackedAnthropicClient:
    """
    Wrapper around AsyncAnthropic that tracks cost for every call.

    Usage:
        client = TrackedAnthropicClient(AsyncAnthropic(api_key=...))
        response, cost = await client.messages_create(
            model="claude-opus-4-7",
            system="You are...",
            messages=[{"role": "user", "content": "..."}],
            tools=[...],
            max_tokens=16_000,
        )
    """

    def __init__(self, client: AsyncAnthropic):
        self._client = client
        self._pricing = get_pricing_config()
        self._total_requests = 0

    @property
    def raw_client(self) -> AsyncAnthropic:
        return self._client

    async def messages_create(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        system: Optional[Any] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        max_tokens: int = 16_000,
        thinking: Optional[Dict[str, Any]] = None,
        output_config: Optional[Dict[str, Any]] = None,
        mcp_servers: Optional[List[Dict[str, Any]]] = None,
        betas: Optional[List[str]] = None,
        **extra_kwargs,
    ) -> Tuple[Any, UsageCost]:
        """Call messages.create and return (response, cost).

        If mcp_servers is provided, routes through client.beta.messages.create
        with the mcp-client beta header.
        """
        kwargs: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
        }
        if system is not None:
            kwargs["system"] = system
        if tools:
            kwargs["tools"] = tools
        if thinking is not None:
            kwargs["thinking"] = thinking
        if output_config is not None:
            kwargs["output_config"] = output_config
        kwargs.update(extra_kwargs)

        self._total_requests += 1

        if mcp_servers:
            mcp_betas = list(betas or [])
            if "mcp-client-2025-11-20" not in mcp_betas:
                mcp_betas.append("mcp-client-2025-11-20")
            kwargs["mcp_servers"] = mcp_servers
            kwargs["betas"] = mcp_betas
            response = await self._client.beta.messages.create(**kwargs)
        elif betas:
            kwargs["betas"] = betas
            response = await self._client.beta.messages.create(**kwargs)
        else:
            response = await self._client.messages.create(**kwargs)

        cost = self._compute_cost(model, response)
        logger.debug(
            f"Anthropic: model={model}, "
            f"input={cost.input_tokens}, cached_read={cost.cached_input_tokens}, "
            f"output={cost.output_tokens}, cost=${cost.total_cost_usd:.6f}"
        )
        return response, cost

    # Multiplier applied to the base input rate for cache-creation tokens
    # (5-minute TTL). 1-hour TTL would be 2.0 — we don't use that yet.
    _CACHE_WRITE_MULTIPLIER = 1.25

    def _compute_cost(self, model: str, response: Any) -> UsageCost:
        """Compute UsageCost from Anthropic usage fields.

        Claude usage fields:
          input_tokens              — uncached input tokens (full price)
          output_tokens             — output tokens
          cache_read_input_tokens   — tokens read from cache (~10% of input)
          cache_creation_input_tokens — tokens written to cache (~125% of input)
        """
        usage = getattr(response, "usage", None)
        if usage is None:
            return self._pricing.calculate_cost(
                model=model, input_tokens=0, output_tokens=0
            )

        input_tokens = getattr(usage, "input_tokens", 0) or 0
        output_tokens = getattr(usage, "output_tokens", 0) or 0
        cache_read = getattr(usage, "cache_read_input_tokens", 0) or 0
        cache_creation = getattr(usage, "cache_creation_input_tokens", 0) or 0

        # Base cost for uncached input + output + cache reads.
        base = self._pricing.calculate_cost(
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cached_input_tokens=cache_read,
        )

        if cache_creation == 0:
            return base

        # Add cache-write cost at 1.25× the base input rate. Pricing lookup is
        # identical to what calculate_cost did internally; accessing _models
        # directly is fine (same module).
        pricing = self._pricing._models.get(model)
        if pricing is None:
            return base
        cache_write_cost = (
            cache_creation * pricing.input_per_token * self._CACHE_WRITE_MULTIPLIER
        )
        return UsageCost(
            model=base.model,
            input_tokens=base.input_tokens + cache_creation,
            output_tokens=base.output_tokens,
            input_cost_usd=base.input_cost_usd + cache_write_cost,
            output_cost_usd=base.output_cost_usd,
            cached_input_tokens=base.cached_input_tokens,
            cached_input_cost_usd=base.cached_input_cost_usd,
        )

    def get_metrics(self) -> Dict[str, int]:
        return {"total_requests": self._total_requests}
