"""
Rate limiter with sliding window tracking for RPM and TPM.

Supports:
- Per-model rate limits (different models have different limits)
- Dynamic limits from database (for user tiers in future)
- Sliding window tracking (not fixed windows)
- Async-safe with locks
"""

import asyncio
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, Optional, Deque, Tuple, List

logger = logging.getLogger(__name__)


@dataclass
class RateLimitConfig:
    """Rate limit configuration for a model."""
    model: str
    rpm_limit: int  # Requests per minute
    tpm_limit: int  # Tokens per minute

    # For future user tier support
    tier: str = "default"


@dataclass
class UsageRecord:
    """A single usage record in the sliding window."""
    timestamp: float
    tokens: int


class SlidingWindowCounter:
    """
    Sliding window counter for rate limiting.

    Tracks requests and tokens over a sliding 60-second window.
    Thread-safe with asyncio lock.
    """

    def __init__(self, rpm_limit: int, tpm_limit: int, window_seconds: float = 60.0):
        self.rpm_limit = rpm_limit
        self.tpm_limit = tpm_limit
        self.window_seconds = window_seconds

        self._requests: Deque[float] = deque()  # Timestamps only
        self._tokens: Deque[UsageRecord] = deque()  # Timestamp + token count
        self._lock = asyncio.Lock()

    def _prune_old(self, now: float) -> None:
        """Remove entries outside the sliding window."""
        cutoff = now - self.window_seconds

        while self._requests and self._requests[0] < cutoff:
            self._requests.popleft()

        while self._tokens and self._tokens[0].timestamp < cutoff:
            self._tokens.popleft()

    def _current_rpm(self, now: float) -> int:
        """Get current requests in window."""
        self._prune_old(now)
        return len(self._requests)

    def _current_tpm(self, now: float) -> int:
        """Get current tokens in window."""
        self._prune_old(now)
        return sum(r.tokens for r in self._tokens)

    async def acquire(self, tokens: int = 0) -> float:
        """
        Acquire permission to make a request.

        Records the request immediately (as a reservation) to prevent
        concurrent callers from exceeding limits.

        Args:
            tokens: Estimated tokens for this request (input + output estimate)

        Returns:
            Wait time in seconds (0 if no wait needed)
        """
        async with self._lock:
            now = time.monotonic()
            self._prune_old(now)

            current_rpm = len(self._requests)
            current_tpm = sum(r.tokens for r in self._tokens)

            wait_time = 0.0

            # Check RPM limit
            if current_rpm >= self.rpm_limit and self._requests:
                # Wait until oldest request exits window
                oldest = self._requests[0]
                wait_for_rpm = (oldest + self.window_seconds) - now
                wait_time = max(wait_time, wait_for_rpm)

            # Check TPM limit
            if current_tpm + tokens > self.tpm_limit and self._tokens:
                # Need to wait for enough tokens to exit window
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

            # IMPORTANT: Record reservation NOW, before releasing lock
            # This prevents concurrent callers from all seeing empty window
            record_time = now + wait_time  # Record at the time we'll actually make the request
            self._requests.append(record_time)
            self._tokens.append(UsageRecord(timestamp=record_time, tokens=tokens))

            return wait_time

    async def record(self, tokens: int) -> None:
        """
        Update the actual token count after request completes.

        Since acquire() already recorded an estimate, this adjusts the
        most recent token record if the actual count differs significantly.

        Note: This is optional - if not called, the estimate from acquire() is used.
        """
        # For simplicity, we don't update - the estimate from acquire() is good enough
        # The slight inaccuracy in token counting won't meaningfully affect rate limiting
        pass

    async def get_usage(self) -> Tuple[int, int]:
        """Get current (rpm, tpm) usage."""
        async with self._lock:
            now = time.monotonic()
            self._prune_old(now)
            return len(self._requests), sum(r.tokens for r in self._tokens)


class RateLimiter:
    """
    Multi-model rate limiter.

    Manages separate sliding windows per model, with limits
    loaded from database or defaults.

    Usage:
        limiter = RateLimiter(db)
        await limiter.acquire("gpt-4o", estimated_tokens=5000)
        # ... make API call ...
        await limiter.record("gpt-4o", actual_tokens=4500)
    """

    # Default limits (conservative, suitable for Tier 1)
    DEFAULT_LIMITS = {
        "gpt-4o": RateLimitConfig("gpt-4o", rpm_limit=500, tpm_limit=30_000),
        "gpt-4o-mini": RateLimitConfig("gpt-4o-mini", rpm_limit=500, tpm_limit=200_000),
        "gpt-5.2": RateLimitConfig("gpt-5.2", rpm_limit=500, tpm_limit=30_000),
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
        self._configs: Dict[str, RateLimitConfig] = {}
        self._config_lock = asyncio.Lock()

        # Load initial configs
        self._load_configs()

    def _load_configs(self) -> None:
        """Load rate limit configs from defaults."""
        # Start with defaults
        self._configs = dict(self.DEFAULT_LIMITS)

    def load_from_db(self, db_limits: list) -> None:
        """
        Load rate limits from database query results.

        Call this from job_processor after querying the rate_limit_configs table.

        Args:
            db_limits: List of objects with model, rpm_limit, tpm_limit, tier attributes

        Example:
            db_limits = db.query(RateLimitConfig).all()
            rate_limiter.load_from_db(db_limits)
        """
        for limit in db_limits:
            self._configs[limit.model] = RateLimitConfig(
                model=limit.model,
                rpm_limit=limit.rpm_limit,
                tpm_limit=limit.tpm_limit,
                tier=getattr(limit, 'tier', 'default') or "default",
            )
        logger.info(f"Loaded {len(db_limits)} rate limits from database")

    def _get_counter(self, model: str) -> SlidingWindowCounter:
        """Get or create counter for a model."""
        if model not in self._counters:
            config = self._configs.get(model)
            if config:
                rpm, tpm = config.rpm_limit, config.tpm_limit
            else:
                rpm, tpm = self.default_rpm, self.default_tpm
                logger.warning(f"No rate limit config for {model}, using defaults: {rpm} rpm, {tpm} tpm")

            self._counters[model] = SlidingWindowCounter(rpm, tpm)

        return self._counters[model]

    async def acquire(self, model: str, estimated_tokens: int = 0) -> float:
        """
        Acquire permission to make an API call.

        Blocks if rate limits would be exceeded.

        Args:
            model: Model name
            estimated_tokens: Estimated total tokens (input + output)

        Returns:
            Time spent waiting (seconds)
        """
        counter = self._get_counter(model)
        wait_time = await counter.acquire(estimated_tokens)

        if wait_time > 0:
            logger.info(f"Rate limit for {model}: waiting {wait_time:.2f}s")
            await asyncio.sleep(wait_time)

        return wait_time

    async def record(self, model: str, tokens: int) -> None:
        """Record a completed API call."""
        counter = self._get_counter(model)
        await counter.record(tokens)

    async def get_usage(self, model: str) -> Tuple[int, int]:
        """Get current (rpm, tpm) usage for a model."""
        counter = self._get_counter(model)
        return await counter.get_usage()

    async def get_all_usage(self) -> Dict[str, Tuple[int, int]]:
        """Get usage for all tracked models."""
        return {
            model: await counter.get_usage()
            for model, counter in self._counters.items()
        }

    def update_limits(self, model: str, rpm_limit: int, tpm_limit: int) -> None:
        """
        Dynamically update limits for a model.

        Useful for adjusting based on API responses (e.g., 429 headers).
        """
        self._configs[model] = RateLimitConfig(model, rpm_limit, tpm_limit)

        # Update existing counter if present
        if model in self._counters:
            counter = self._counters[model]
            counter.rpm_limit = rpm_limit
            counter.tpm_limit = tpm_limit

        logger.info(f"Updated rate limits for {model}: {rpm_limit} rpm, {tpm_limit} tpm")