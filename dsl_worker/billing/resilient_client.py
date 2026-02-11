"""
Resilient OpenAI client with rate limiting.

OpenAI's client has built-in retry logic, so we don't duplicate that.
We add:
- Proactive rate limiting (avoid hitting limits)
- Metrics tracking
- Optional additional backoff for persistent rate limits
"""

import asyncio
import logging
import random
from dataclasses import dataclass
from typing import Any, Optional, Dict, List

import openai
from openai import AsyncOpenAI

from dsl_worker.billing.rate_limiter import RateLimiter

logger = logging.getLogger(__name__)


@dataclass
class RetryConfig:
    """Configuration for additional retry behavior (after OpenAI's built-in retries fail)."""

    # Extra retries after OpenAI gives up (for persistent rate limits)
    extra_retries_on_rate_limit: int = 3
    extra_retry_base_delay: float = (
        30.0  # Start with longer delay since OpenAI already tried short ones
    )
    extra_retry_max_delay: float = 120.0


def calculate_backoff(
    attempt: int,
    base_delay: float,
    max_delay: float,
    jitter: float = 0.3,
) -> float:
    """Calculate backoff delay with exponential growth and jitter."""
    delay = base_delay * (2**attempt)
    delay = min(delay, max_delay)
    delay += delay * jitter * random.random()
    return delay


class ResilientClient:
    """
    Wrapper around AsyncOpenAI with rate limiting.

    OpenAI's client handles transient errors and short retries.
    We add:
    - Proactive rate limiting to avoid hitting limits
    - Extra retries with longer backoff for persistent rate limits
    - Metrics tracking

    Usage:
        client = ResilientClient(
            openai_client=AsyncOpenAI(),
            rate_limiter=RateLimiter(),
        )

        response = await client.embeddings_create(
            model="text-embedding-3-large",
            input=["hello"],
        )
    """

    def __init__(
        self,
        openai_client: AsyncOpenAI,
        rate_limiter: Optional[RateLimiter] = None,
        retry_config: Optional[RetryConfig] = None,
    ):
        self._client = openai_client
        self._rate_limiter = rate_limiter
        self._retry_config = retry_config or RetryConfig()

        # Metrics
        self._total_requests = 0
        self._total_rate_limit_waits = 0
        self._total_extra_retries = 0

    @property
    def raw_client(self) -> AsyncOpenAI:
        """Access underlying client for unsupported operations."""
        return self._client

    async def _execute_with_rate_limiting(
        self,
        operation,
        estimated_tokens: int = 0,
        **api_kwargs,
    ) -> Any:
        """
        Execute an API operation with rate limiting.

        OpenAI's client handles transient retries.
        We add proactive rate limiting and extra retries for persistent 429s.
        """
        model = api_kwargs.get("model", "unknown")
        self._total_requests += 1
        config = self._retry_config

        for extra_attempt in range(config.extra_retries_on_rate_limit + 1):
            try:
                # Proactive rate limiting - wait if we're near limits
                if self._rate_limiter:
                    wait_time = await self._rate_limiter.acquire(
                        model, estimated_tokens
                    )
                    if wait_time > 0:
                        self._total_rate_limit_waits += 1

                # Execute - OpenAI client handles transient retries internally
                result = await operation(**api_kwargs)

                # Record successful request for rate tracking
                if self._rate_limiter and hasattr(result, "usage"):
                    total_tokens = 0
                    if hasattr(result.usage, "total_tokens"):
                        total_tokens = result.usage.total_tokens
                    elif hasattr(result.usage, "input_tokens"):
                        total_tokens = getattr(
                            result.usage, "input_tokens", 0
                        ) + getattr(result.usage, "output_tokens", 0)
                    await self._rate_limiter.record(model, total_tokens)

                return result

            except openai.RateLimitError as e:
                # OpenAI's retries failed, try extra backoff for persistent rate limits
                if extra_attempt >= config.extra_retries_on_rate_limit:
                    logger.error(
                        f"Rate limit persists for {model} after {extra_attempt + 1} extra attempts, giving up"
                    )
                    raise

                self._total_extra_retries += 1
                delay = calculate_backoff(
                    extra_attempt,
                    config.extra_retry_base_delay,
                    config.extra_retry_max_delay,
                )

                logger.warning(
                    f"Persistent rate limit for {model}, extra attempt {extra_attempt + 1}/"
                    f"{config.extra_retries_on_rate_limit}, waiting {delay:.1f}s"
                )
                await asyncio.sleep(delay)

            except openai.APIStatusError as e:
                # Check if it's a rate limit that wasn't caught as RateLimitError
                if getattr(e, "status_code", 0) == 429:
                    if extra_attempt >= config.extra_retries_on_rate_limit:
                        logger.error(
                            f"Rate limit (429) persists for {model}, giving up"
                        )
                        raise

                    self._total_extra_retries += 1
                    delay = calculate_backoff(
                        extra_attempt,
                        config.extra_retry_base_delay,
                        config.extra_retry_max_delay,
                    )
                    logger.warning(
                        f"Rate limit (429) for {model}, waiting {delay:.1f}s"
                    )
                    await asyncio.sleep(delay)
                else:
                    # Not a rate limit, don't retry
                    raise

    async def chat_completion(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        estimated_tokens: int = 0,
        **kwargs,
    ):
        """Create chat completion with rate limiting."""
        return await self._execute_with_rate_limiting(
            self._client.chat.completions.create,
            estimated_tokens=estimated_tokens,
            model=model,
            messages=messages,
            **kwargs,
        )

    async def embeddings_create(
        self,
        model: str,
        input: List[str],
        estimated_tokens: int = 0,
        **kwargs,
    ):
        """Create embeddings with rate limiting."""
        if estimated_tokens == 0:
            estimated_tokens = sum(len(text) // 4 for text in input)

        return await self._execute_with_rate_limiting(
            self._client.embeddings.create,
            estimated_tokens=estimated_tokens,
            model=model,
            input=input,
            encoding_format="float",
            **kwargs,
        )

    async def responses_create(
        self,
        model: str,
        input: List[Any],
        estimated_tokens: int = 0,
        **kwargs,
    ):
        """Create response (Responses API) with rate limiting."""
        return await self._execute_with_rate_limiting(
            self._client.responses.create,
            estimated_tokens=estimated_tokens,
            model=model,
            input=input,
            **kwargs,
        )

    def get_metrics(self) -> Dict[str, int]:
        """Get client metrics."""
        return {
            "total_requests": self._total_requests,
            "total_rate_limit_waits": self._total_rate_limit_waits,
            "total_extra_retries": self._total_extra_retries,
        }
