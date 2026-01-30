"""
Rate limiter with sliding window tracking for RPM and TPM.
"""

import asyncio
import logging
import time
from collections import deque
from dataclasses import dataclass
from typing import Dict, Tuple, Deque

logger = logging.getLogger(__name__)


@dataclass
class RateLimitConfig:
    """Rate limit configuration for a model."""
    model: str
    rpm_limit: int
    tpm_limit: int


@dataclass
class UsageRecord:
    """A single usage record in the sliding window."""
    timestamp: float
    tokens: int


class SlidingWindowCounter:
    """
    Sliding window counter for rate limiting.
    
    Tracks requests and tokens over a sliding 60-second window.
    """

    def __init__(self, rpm_limit: int, tpm_limit: int, window_seconds: float = 60.0):
        self.rpm_limit = rpm_limit
        self.tpm_limit = tpm_limit
        self.window_seconds = window_seconds

        self._requests: Deque[float] = deque()
        self._tokens: Deque[UsageRecord] = deque()
        self._lock = asyncio.Lock()

    def _prune_old(self, now: float) -> None:
        """Remove entries outside the sliding window."""
        cutoff = now - self.window_seconds

        while self._requests and self._requests[0] < cutoff:
            self._requests.popleft()

        while self._tokens and self._tokens[0].timestamp < cutoff:
            self._tokens.popleft()

    async def acquire(self, tokens: int = 0) -> float:
        """
        Acquire permission to make a request.
        
        Returns wait time in seconds (0 if no wait needed).
        """
        async with self._lock:
            now = time.monotonic()
            self._prune_old(now)

            current_rpm = len(self._requests)
            current_tpm = sum(r.tokens for r in self._tokens)

            wait_time = 0.0

            if current_rpm >= self.rpm_limit and self._requests:
                oldest = self._requests[0]
                wait_for_rpm = (oldest + self.window_seconds) - now
                wait_time = max(wait_time, wait_for_rpm)

            if current_tpm + tokens > self.tpm_limit and self._tokens:
                needed = (current_tpm + tokens) - self.tpm_limit
                accumulated = 0
                for record in self._tokens:
                    accumulated += record.tokens
                    if accumulated >= needed:
                        wait_for_tpm = (record.timestamp + self.window_seconds) - now
                        wait_time = max(wait_time, wait_for_tpm)
                        break

            if wait_time > 0:
                logger.debug(
                    f"Rate limit: waiting {wait_time:.2f}s "
                    f"(rpm={current_rpm}/{self.rpm_limit}, tpm={current_tpm}/{self.tpm_limit})"
                )

            record_time = now + wait_time
            self._requests.append(record_time)
            self._tokens.append(UsageRecord(timestamp=record_time, tokens=tokens))

            return wait_time

    async def get_usage(self) -> Tuple[int, int]:
        """Get current (rpm, tpm) usage."""
        async with self._lock:
            now = time.monotonic()
            self._prune_old(now)
            return len(self._requests), sum(r.tokens for r in self._tokens)


class RateLimiter:
    """
    Multi-model rate limiter.
    
    Manages separate sliding windows per model.
    """

    DEFAULT_LIMITS = {
        "gpt-4o": RateLimitConfig("gpt-4o", rpm_limit=500, tpm_limit=30_000),
        "gpt-4o-mini": RateLimitConfig("gpt-4o-mini", rpm_limit=500, tpm_limit=200_000),
        "gpt-5.2": RateLimitConfig("gpt-5.2", rpm_limit=500, tpm_limit=30_000),
        "gpt-5-mini": RateLimitConfig("gpt-5-mini", rpm_limit=500, tpm_limit=200_000),
        "o1": RateLimitConfig("o1", rpm_limit=500, tpm_limit=30_000),
        "o1-mini": RateLimitConfig("o1-mini", rpm_limit=500, tpm_limit=150_000),
        "text-embedding-3-large": RateLimitConfig("text-embedding-3-large", rpm_limit=500, tpm_limit=1_000_000),
        "text-embedding-3-small": RateLimitConfig("text-embedding-3-small", rpm_limit=500, tpm_limit=1_000_000),
    }

    def __init__(
        self,
        default_rpm: int = 100,
        default_tpm: int = 100_000,
    ):
        self.default_rpm = default_rpm
        self.default_tpm = default_tpm

        self._counters: Dict[str, SlidingWindowCounter] = {}
        self._configs: Dict[str, RateLimitConfig] = dict(self.DEFAULT_LIMITS)

    def _get_counter(self, model: str) -> SlidingWindowCounter:
        """Get or create counter for a model."""
        if model not in self._counters:
            config = self._configs.get(model)
            if config:
                rpm, tpm = config.rpm_limit, config.tpm_limit
            else:
                rpm, tpm = self.default_rpm, self.default_tpm
                logger.warning(f"No rate limit config for {model}, using defaults")

            self._counters[model] = SlidingWindowCounter(rpm, tpm)

        return self._counters[model]

    async def acquire(self, model: str, estimated_tokens: int = 0) -> float:
        """Acquire permission to make an API call."""
        counter = self._get_counter(model)
        wait_time = await counter.acquire(estimated_tokens)

        if wait_time > 0:
            logger.info(f"Rate limit for {model}: waiting {wait_time:.2f}s")
            await asyncio.sleep(wait_time)

        return wait_time

    async def get_usage(self, model: str) -> Tuple[int, int]:
        """Get current (rpm, tpm) usage for a model."""
        counter = self._get_counter(model)
        return await counter.get_usage()

    def update_limits(self, model: str, rpm_limit: int, tpm_limit: int) -> None:
        """Dynamically update limits for a model."""
        self._configs[model] = RateLimitConfig(model, rpm_limit, tpm_limit)

        if model in self._counters:
            counter = self._counters[model]
            counter.rpm_limit = rpm_limit
            counter.tpm_limit = tpm_limit

        logger.info(f"Updated rate limits for {model}: {rpm_limit} rpm, {tpm_limit} tpm")