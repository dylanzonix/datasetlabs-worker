"""
Tracked OpenAI client that calculates costs for all API calls.

Wraps the ResilientClient and returns cost information alongside responses.
Includes rate limiting and retry logic via the underlying resilient client.
"""

import logging
from dataclasses import dataclass
from typing import List, Optional, Any, Dict, Tuple

from openai import AsyncOpenAI
from openai.types.chat import ChatCompletion
from openai.types import CreateEmbeddingResponse

from dsl_worker.billing.pricing import get_pricing_config, UsageCost
from dsl_worker.billing.rate_limiter import RateLimiter
from dsl_worker.billing.resilient_client import ResilientClient, RetryConfig

logger = logging.getLogger(__name__)


@dataclass
class TrackedChatCompletion:
    """Chat completion response with cost tracking."""
    response: ChatCompletion
    cost: UsageCost


@dataclass
class TrackedEmbeddingResponse:
    """Embedding response with cost tracking."""
    response: CreateEmbeddingResponse
    cost: UsageCost


class TrackedOpenAIClient:
    """
    Wrapper around AsyncOpenAI that tracks costs for all API calls.

    Includes:
    - Cost tracking with configurable pricing
    - Rate limiting (proactive)
    - Automatic retries with exponential backoff
    - Graceful error handling

    Usage:
        rate_limiter = RateLimiter(db)
        client = TrackedOpenAIClient(AsyncOpenAI(), rate_limiter=rate_limiter)

        # Chat completion
        result = await client.chat_completion(model="gpt-4o", messages=[...])
        print(result.cost.total_cost_usd)

        # Embeddings
        result = await client.create_embeddings(model="text-embedding-3-large", input=[...])
        print(result.cost.total_cost_usd)
    """

    def __init__(
        self,
        client: AsyncOpenAI,
        rate_limiter: Optional[RateLimiter] = None,
        retry_config: Optional[RetryConfig] = None,
    ):
        self._pricing = get_pricing_config()
        self._rate_limiter = rate_limiter

        # Create resilient client wrapper
        self._resilient = ResilientClient(
            openai_client=client,
            rate_limiter=rate_limiter,
            retry_config=retry_config,
        )

    @property
    def raw_client(self) -> AsyncOpenAI:
        """Access the underlying OpenAI client for unsupported operations."""
        return self._resilient.raw_client

    @property
    def rate_limiter(self) -> Optional[RateLimiter]:
        """Access the rate limiter."""
        return self._rate_limiter

    def _estimate_tokens(self, messages: List[Dict[str, Any]]) -> int:
        """Estimate token count for messages (rough approximation)."""
        total_chars = sum(
            len(str(msg.get("content", "")))
            for msg in messages
        )
        # Rough estimate: 1 token per 4 characters, plus overhead
        return (total_chars // 4) + (len(messages) * 4)

    async def chat_completion(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict]] = None,
        tool_choice: Optional[str] = None,
        **kwargs,
    ) -> TrackedChatCompletion:
        """
        Create a chat completion and track the cost.

        Returns:
            TrackedChatCompletion with response and cost
        """
        estimated_tokens = self._estimate_tokens(messages)

        response = await self._resilient.chat_completion(
            model=model,
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
            estimated_tokens=estimated_tokens,
            **kwargs,
        )

        # Extract token usage
        usage = response.usage
        input_tokens = usage.prompt_tokens if usage else 0
        output_tokens = usage.completion_tokens if usage else 0

        cost = self._pricing.calculate_cost(
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )

        logger.debug(
            f"Chat completion: model={model}, "
            f"input={input_tokens}, output={output_tokens}, "
            f"cost=${cost.total_cost_usd:.6f}"
        )

        return TrackedChatCompletion(response=response, cost=cost)

    async def create_embeddings(
        self,
        model: str,
        input: List[str],
        **kwargs,
    ) -> TrackedEmbeddingResponse:
        """
        Create embeddings and track the cost.

        Returns:
            TrackedEmbeddingResponse with response and cost
        """
        response = await self._resilient.embeddings_create(
            model=model,
            input=input,
            **kwargs,
        )

        # Extract token usage
        usage = response.usage
        input_tokens = usage.total_tokens if usage else 0

        cost = self._pricing.calculate_cost(
            model=model,
            input_tokens=input_tokens,
            output_tokens=0,  # Embeddings have no output tokens
        )

        logger.debug(
            f"Embeddings: model={model}, "
            f"input={input_tokens}, "
            f"cost=${cost.total_cost_usd:.6f}"
        )

        return TrackedEmbeddingResponse(response=response, cost=cost)

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

        Supports both:
        - Stored prompts (with prompt parameter)
        - Direct tool-calling (with tools parameter)

        Args:
            model: Model to use (e.g., "gpt-4o", "o1")
            input: List of input messages/items
            prompt: Optional stored prompt config
            tools: Optional list of tool definitions
            estimated_tokens: Estimated tokens for rate limiting
            **kwargs: Additional parameters (max_output_tokens, reasoning, etc.)

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

        # Extract token usage
        input_tokens = 0
        output_tokens = 0

        if hasattr(response, 'usage') and response.usage:
            input_tokens = getattr(response.usage, 'input_tokens', 0)
            output_tokens = getattr(response.usage, 'output_tokens', 0)

        cost = self._pricing.calculate_cost(
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )

        logger.debug(
            f"Responses API: model={model}, "
            f"input={input_tokens}, output={output_tokens}, "
            f"cost=${cost.total_cost_usd:.6f}"
        )

        return response, cost

    async def embeddings_create(
        self,
        model: str,
        input: List[str],
        **kwargs,
    ) -> Tuple[CreateEmbeddingResponse, UsageCost]:
        """
        Create embeddings (alternative signature returning tuple).

        This matches the signature expected by some phases.
        """
        result = await self.create_embeddings(model=model, input=input, **kwargs)
        return result.response, result.cost

    def get_metrics(self) -> Dict[str, Any]:
        """Get client metrics including retry stats."""
        return self._resilient.get_metrics()