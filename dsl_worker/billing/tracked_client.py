"""
Tracked OpenAI client that calculates costs for all API calls.
"""

import logging
from typing import List, Optional, Any, Dict, Tuple

from openai import AsyncOpenAI

from dsl_worker.billing.pricing import get_pricing_config, UsageCost
from dsl_worker.billing.rate_limiter import RateLimiter
from dsl_worker.billing.resilient_client import ResilientClient, RetryConfig

logger = logging.getLogger(__name__)


class TrackedOpenAIClient:
    """
    Wrapper around AsyncOpenAI that tracks costs for all API calls.
    """

    def __init__(
        self,
        client: AsyncOpenAI,
        rate_limiter: Optional[RateLimiter] = None,
        retry_config: Optional[RetryConfig] = None,
    ):
        self._pricing = get_pricing_config()
        self._rate_limiter = rate_limiter

        self._resilient = ResilientClient(
            openai_client=client,
            rate_limiter=rate_limiter,
            retry_config=retry_config,
        )

    @property
    def raw_client(self) -> AsyncOpenAI:
        """Access the underlying OpenAI client."""
        return self._resilient.raw_client

    @property
    def rate_limiter(self) -> Optional[RateLimiter]:
        """Access the rate limiter."""
        return self._rate_limiter

    async def responses_create(
        self,
        model: str,
        input: List[Any],
        prompt: Optional[Dict[str, Any]] = None,
        tools: Optional[List[Dict]] = None,
        estimated_tokens: int = 0,
        **kwargs,
    ) -> Tuple[Any, UsageCost]:
        """
        Create a response using the Responses API.

        Returns:
            Tuple of (response, cost)
        """
        create_kwargs = {
            "model": model,
            "input": input,
            **kwargs,
        }

        if prompt is not None:
            create_kwargs["prompt"] = prompt

        if tools is not None:
            create_kwargs["tools"] = tools

        response = await self._resilient.responses_create(
            model=model,
            input=input,
            estimated_tokens=estimated_tokens,
            **{k: v for k, v in create_kwargs.items() if k not in ("model", "input")},
        )

        input_tokens = 0
        output_tokens = 0
        cached_input_tokens = 0

        if hasattr(response, 'usage') and response.usage:
            input_tokens = getattr(response.usage, 'input_tokens', 0)
            output_tokens = getattr(response.usage, 'output_tokens', 0)

            details = getattr(response.usage, 'input_tokens_details', None)
            if details:
                cached_input_tokens = getattr(details, 'cached_tokens', 0) or 0

        # input_tokens from the API includes cached tokens, so subtract them
        non_cached_input_tokens = input_tokens - cached_input_tokens

        cost = self._pricing.calculate_cost(
            model=model,
            input_tokens=non_cached_input_tokens,
            output_tokens=output_tokens,
            cached_input_tokens=cached_input_tokens,
        )

        logger.debug(
            f"Responses API: model={model}, "
            f"input={non_cached_input_tokens}, cached={cached_input_tokens}, output={output_tokens}, "
            f"cost=${cost.total_cost_usd:.6f}"
        )

        return response, cost

    def get_metrics(self) -> Dict[str, Any]:
        """Get client metrics."""
        return self._resilient.get_metrics()