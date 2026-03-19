"""
Candidate pool with Thompson Sampling source allocation.

V10: Replaces the FIFO SeedProcessor queue. Each source (harvester) is an "arm"
in a multi-armed bandit. Thompson Sampling selects which source to pull
candidates from, optimizing for cost-per-successful-row.

The pool automatically favors sources with higher fertility rates and lower
processing costs. No absolute thresholds — everything is relative between sources.

Dedup remains at row level via DedupStore (unchanged). The pool handles
source-level optimization only.
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Awaitable, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


class OutcomeType(Enum):
    SUCCESS = "success"
    DUPLICATE = "duplicate"
    FILTERED = "filtered"
    ERROR = "error"


@dataclass
class SourceStats:
    """Per-source statistics for the multi-armed bandit."""

    source_id: str
    label: str = ""  # human-readable (e.g. "upwork.com/search?q=dataset")

    # Beta distribution parameters (Thompson Sampling)
    alpha: float = 1.0  # prior + successes
    beta: float = 1.0   # prior + failures

    # Candidate counts
    candidates_produced: int = 0
    candidates_processed: int = 0
    candidates_pending: int = 0

    # Outcome counts
    rows_produced: int = 0
    duplicates: int = 0
    filtered: int = 0
    errors: int = 0

    # Cost tracking
    total_process_cost: float = 0.0
    production_cost: float = 0.0

    # Source state
    exhausted: bool = False
    created_at: float = field(default_factory=time.time)

    @property
    def fertility_rate(self) -> float:
        if self.candidates_processed == 0:
            return 0.0
        return self.rows_produced / self.candidates_processed

    @property
    def avg_process_cost(self) -> float:
        if self.candidates_processed == 0:
            return 0.01  # small prior to avoid div-by-zero in scoring
        return self.total_process_cost / self.candidates_processed

    @property
    def cost_per_row(self) -> float:
        if self.rows_produced == 0:
            return float("inf")
        return self.total_process_cost / self.rows_produced

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_id": self.source_id,
            "label": self.label,
            "alpha": self.alpha,
            "beta": self.beta,
            "candidates_produced": self.candidates_produced,
            "candidates_processed": self.candidates_processed,
            "candidates_pending": self.candidates_pending,
            "rows_produced": self.rows_produced,
            "duplicates": self.duplicates,
            "filtered": self.filtered,
            "errors": self.errors,
            "total_process_cost": self.total_process_cost,
            "production_cost": self.production_cost,
            "exhausted": self.exhausted,
            "fertility_rate": self.fertility_rate,
            "cost_per_row": self.cost_per_row if self.rows_produced > 0 else None,
        }


@dataclass
class Candidate:
    """A candidate item from a source, ready for row generation."""

    values: Any  # raw candidate: string, dict, or structured data
    source_id: str
    source_context: str = ""  # human-readable scope description
    metadata: Dict[str, Any] = field(default_factory=dict)


class CandidatePool:
    """
    Pool of candidates from multiple sources with Thompson Sampling selection.

    Sources register themselves and submit candidates. The generation consumer
    pulls candidates weighted by source performance (fertility × 1/cost).
    Outcomes are recorded to update each source's Beta distribution.

    No quotas, no FIFO — the bandit naturally favors better sources.
    """

    def __init__(self, target_rows: int, generation_stats: Optional[Dict[str, Any]] = None) -> None:
        self._target_rows = target_rows
        self._generation_stats = generation_stats or {}

        self._sources: Dict[str, SourceStats] = {}
        self._queues: Dict[str, asyncio.Queue] = {}  # source_id → candidate queue

        self._lock = asyncio.Lock()
        self._condition = asyncio.Condition(self._lock)

        # Global counters
        self._total_submitted = 0
        self._total_pulled = 0

    def register_source(self, source_id: str, label: str = "") -> None:
        """Register a new source arm. Idempotent."""
        if source_id not in self._sources:
            self._sources[source_id] = SourceStats(source_id=source_id, label=label)
            self._queues[source_id] = asyncio.Queue()
            logger.info(f"[CandidatePool] Registered source: {source_id} ({label})")

    async def submit(self, candidate: Candidate) -> None:
        """
        Add a candidate to the pool. Non-blocking.

        Called by harvesters (via BU submit_seed or code_exec callback).
        """
        source_id = candidate.source_id

        async with self._condition:
            if source_id not in self._sources:
                self._sources[source_id] = SourceStats(source_id=source_id)
                self._queues[source_id] = asyncio.Queue()

            self._sources[source_id].candidates_produced += 1
            self._sources[source_id].candidates_pending += 1
            self._total_submitted += 1

            await self._queues[source_id].put(candidate)
            self._condition.notify_all()  # wake up any blocked pull()

        if self._total_submitted % 50 == 0:
            logger.info(
                f"[CandidatePool] {self._total_submitted} candidates submitted, "
                f"{self._total_pulled} pulled, {len(self._sources)} sources"
            )

    async def pull(self, timeout: float = 30.0) -> Optional[Candidate]:
        """
        Pull next candidate using Thompson Sampling.

        Blocks if no candidates are available. Returns None when all sources
        are exhausted and all queues are drained (signals consumer to exit).
        """
        deadline = time.monotonic() + timeout

        while True:
            async with self._condition:
                # Check terminal condition: all exhausted + all drained
                if self._is_fully_drained():
                    return None

                # Check if target reached
                rows_done = self._generation_stats.get("rows_generated", 0)
                if rows_done >= self._target_rows:
                    return None

                # Find sources with available candidates
                available = {
                    sid: q for sid, q in self._queues.items()
                    if not q.empty()
                }

                if not available:
                    # Wait for new candidates or timeout
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        return None
                    try:
                        await asyncio.wait_for(
                            self._condition.wait(),
                            timeout=min(remaining, 5.0),
                        )
                    except asyncio.TimeoutError:
                        pass
                    continue

                # Thompson Sampling: pick source
                best_source = self._thompson_select(available)

                try:
                    candidate = self._queues[best_source].get_nowait()
                    self._sources[best_source].candidates_pending -= 1
                    self._total_pulled += 1
                    return candidate
                except asyncio.QueueEmpty:
                    continue

    def _thompson_select(self, available: Dict[str, asyncio.Queue]) -> str:
        """
        Thompson Sampling with cost adjustment.

        For each available source:
          1. Sample from Beta(alpha, beta)
          2. Divide by avg_process_cost (cost-weighted)
          3. Pick highest score

        New sources (few samples) have wide Beta distributions → natural exploration.
        Proven sources have tight distributions → exploitation.
        All relative — no absolute thresholds.
        """
        scores: Dict[str, float] = {}

        for sid in available:
            stats = self._sources[sid]
            # Sample from Beta distribution
            sample = random.betavariate(stats.alpha, stats.beta)
            # Cost-weight: higher score for cheaper sources
            avg_cost = stats.avg_process_cost
            scores[sid] = sample / avg_cost

        return max(scores, key=scores.get)

    def _is_fully_drained(self) -> bool:
        """True when all sources exhausted AND all queues empty."""
        if not self._sources:
            return False  # no sources registered yet

        for sid, stats in self._sources.items():
            if not stats.exhausted:
                return False  # at least one source still active
            if not self._queues.get(sid, asyncio.Queue()).empty():
                return False  # still has pending candidates

        return True

    def record_outcome(
        self,
        source_id: str,
        outcome: OutcomeType,
        cost: float,
    ) -> None:
        """
        Record a processing outcome for a candidate.

        Updates the source's Beta distribution parameters:
          - SUCCESS: alpha += 1 (full reward)
          - DUPLICATE: beta += 0.5 (partial penalty — cheap to detect)
          - FILTERED: beta += 1 (full penalty)
          - ERROR: beta += 1 (full penalty)
        """
        stats = self._sources.get(source_id)
        if not stats:
            logger.warning(f"[CandidatePool] Unknown source_id: {source_id}")
            return

        stats.candidates_processed += 1
        stats.total_process_cost += cost

        if outcome == OutcomeType.SUCCESS:
            stats.rows_produced += 1
            stats.alpha += 1
        elif outcome == OutcomeType.DUPLICATE:
            stats.duplicates += 1
            stats.beta += 0.5
        elif outcome == OutcomeType.FILTERED:
            stats.filtered += 1
            stats.beta += 1
        elif outcome == OutcomeType.ERROR:
            stats.errors += 1
            stats.beta += 1

    def mark_exhausted(self, source_id: str) -> None:
        """Mark a source as exhausted (harvester finished)."""
        stats = self._sources.get(source_id)
        if stats:
            stats.exhausted = True
            logger.info(
                f"[CandidatePool] Source exhausted: {source_id} "
                f"({stats.candidates_produced} produced, "
                f"{stats.rows_produced} rows, "
                f"{stats.fertility_rate:.0%} fertility)"
            )

    def add_production_cost(self, source_id: str, cost: float) -> None:
        """Track cost of producing candidates (crawling/parsing), separate from processing."""
        stats = self._sources.get(source_id)
        if stats:
            stats.production_cost += cost

    @property
    def total_rows(self) -> int:
        return sum(s.rows_produced for s in self._sources.values())

    @property
    def total_pending(self) -> int:
        return sum(s.candidates_pending for s in self._sources.values())

    @property
    def target_rows(self) -> int:
        return self._target_rows

    def get_source_stats(self) -> Dict[str, SourceStats]:
        return dict(self._sources)

    def format_status(self) -> str:
        """Format pool stats for orchestrator consumption."""
        rows_done = self._generation_stats.get("rows_generated", 0)
        skipped = self._generation_stats.get("skipped", 0)
        errors = self._generation_stats.get("errors", 0)

        lines = [
            f"Progress: {rows_done}/{self._target_rows} rows "
            f"({skipped} skipped, {errors} errors)"
        ]

        if not self._sources:
            lines.append("  No sources registered yet.")
            return "\n".join(lines)

        for sid, stats in self._sources.items():
            label = stats.label or sid
            fertility = f"{stats.fertility_rate:.0%}" if stats.candidates_processed > 0 else "N/A"
            cpr = f"${stats.cost_per_row:.3f}" if stats.rows_produced > 0 else "N/A"
            status = "exhausted" if stats.exhausted else f"{stats.candidates_pending} pending"

            lines.append(
                f"  {label}: {fertility} fertility, "
                f"cost/row={cpr}, "
                f"{stats.rows_produced} rows from {stats.candidates_processed} processed, "
                f"{status}"
            )

        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize pool state for checkpointing."""
        return {
            "target_rows": self._target_rows,
            "total_submitted": self._total_submitted,
            "total_pulled": self._total_pulled,
            "sources": {sid: s.to_dict() for sid, s in self._sources.items()},
        }


# ── Strategy Monitor ─────────────────────────────────────────────────


@dataclass
class StrategyEvent:
    """A strategic event for the orchestrator."""

    type: str  # source_exhausted, fertility_shift, milestone, stall, pool_empty, target_reached
    message: str  # formatted for orchestrator consumption
    source_id: Optional[str] = None
    data: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


class StrategyMonitor:
    """
    Watches CandidatePool stats and generates strategic events for the orchestrator.

    Events are meaningful inflection points — not operational noise. The orchestrator
    only wakes up when there's a genuine strategic decision to make.
    """

    def __init__(
        self,
        pool: CandidatePool,
        target_rows: int,
        generation_stats: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._pool = pool
        self._target_rows = target_rows
        self._generation_stats = generation_stats or {}
        self._event_queue: asyncio.Queue[StrategyEvent] = asyncio.Queue()

        # Milestone tracking (fire once each)
        self._milestones_fired: set = set()  # {0.25, 0.50, 0.75, 1.0}

        # Stall tracking (cumulative, each supersedes last)
        self._stall_intervals = [5, 10, 20, 50]
        self._stall_index = 0  # which interval we're on
        self._consecutive_failures = 0

        # Fertility shift tracking (per-source baselines)
        self._baselines: Dict[str, float] = {}  # source_id → baseline fertility
        self._baseline_sample_count = 10  # samples before establishing baseline
        self._last_fertility_event: Dict[str, int] = {}  # source_id → processed count at last event
        self._fertility_debounce = 10  # min candidates between fertility events

        # Pool empty debounce
        self._last_pool_empty_time: float = 0
        self._pool_empty_debounce = 30.0  # seconds

    async def on_outcome(
        self,
        source_id: str,
        outcome: OutcomeType,
        cost: float,
    ) -> None:
        """
        Record an outcome and check for strategic events.

        Called by the generation consumer after each candidate is processed.
        """
        self._pool.record_outcome(source_id, outcome, cost)

        # Update stall counter
        if outcome == OutcomeType.SUCCESS:
            self._consecutive_failures = 0
        else:
            self._consecutive_failures += 1

        # Check all event conditions
        await self._check_target_reached()
        await self._check_milestone()
        await self._check_stall()
        await self._check_fertility_shift(source_id)

    async def on_source_done(self, source_id: str) -> None:
        """Called when a harvester finishes its source."""
        self._pool.mark_exhausted(source_id)

        stats = self._pool.get_source_stats().get(source_id)
        if not stats:
            return

        status = self._pool.format_status()
        await self._fire_event(StrategyEvent(
            type="source_exhausted",
            source_id=source_id,
            message=(
                f"Source exhausted: {stats.label or source_id}\n"
                f"  Produced {stats.candidates_produced} candidates, "
                f"{stats.rows_produced} rows, "
                f"{stats.fertility_rate:.0%} fertility\n\n"
                f"{status}"
            ),
            data=stats.to_dict(),
        ))

        # Also check if pool is now empty
        await self._check_pool_empty()

    async def wait_for_event(self, timeout: float = 120.0) -> Optional[StrategyEvent]:
        """
        Block until a strategic event occurs.

        Returns None if timeout expires (caller should inject status update).
        """
        try:
            return await asyncio.wait_for(self._event_queue.get(), timeout=timeout)
        except asyncio.TimeoutError:
            return None

    # ── Internal event checks ──────────────────────────────────────

    async def _check_target_reached(self) -> None:
        rows_done = self._generation_stats.get("rows_generated", 0)
        if rows_done >= self._target_rows and 1.0 not in self._milestones_fired:
            self._milestones_fired.add(1.0)
            await self._fire_event(StrategyEvent(
                type="target_reached",
                message=(
                    f"Target reached: {rows_done}/{self._target_rows} rows generated.\n\n"
                    f"{self._pool.format_status()}"
                ),
            ))

    async def _check_milestone(self) -> None:
        rows_done = self._generation_stats.get("rows_generated", 0)
        if self._target_rows <= 0:
            return

        pct = rows_done / self._target_rows
        for threshold in [0.25, 0.50, 0.75]:
            if pct >= threshold and threshold not in self._milestones_fired:
                self._milestones_fired.add(threshold)
                await self._fire_event(StrategyEvent(
                    type="milestone",
                    message=(
                        f"Milestone: {threshold:.0%} complete — "
                        f"{rows_done}/{self._target_rows} rows.\n\n"
                        f"{self._pool.format_status()}"
                    ),
                    data={"threshold": threshold, "rows": rows_done},
                ))

    async def _check_stall(self) -> None:
        if self._stall_index >= len(self._stall_intervals):
            return

        threshold = self._stall_intervals[self._stall_index]
        if self._consecutive_failures >= threshold:
            self._stall_index += 1  # next interval (supersedes)
            await self._fire_event(StrategyEvent(
                type="stall",
                message=(
                    f"Stall: {self._consecutive_failures} consecutive candidates "
                    f"without a successful row.\n\n"
                    f"{self._pool.format_status()}"
                ),
                data={"consecutive_failures": self._consecutive_failures},
            ))

    async def _check_fertility_shift(self, source_id: str) -> None:
        stats = self._pool.get_source_stats().get(source_id)
        if not stats or stats.candidates_processed < self._baseline_sample_count:
            return

        # Establish baseline on first check
        if source_id not in self._baselines:
            self._baselines[source_id] = stats.fertility_rate
            self._last_fertility_event[source_id] = stats.candidates_processed
            return

        # Debounce
        last_event_at = self._last_fertility_event.get(source_id, 0)
        if stats.candidates_processed - last_event_at < self._fertility_debounce:
            return

        baseline = self._baselines[source_id]
        current = stats.fertility_rate

        # Relative shift > 30% from baseline
        if baseline > 0:
            shift = abs(current - baseline) / baseline
        elif current > 0:
            shift = 1.0  # went from 0 to something
        else:
            return

        if shift >= 0.3:
            direction = "improved" if current > baseline else "declined"
            self._baselines[source_id] = current  # update baseline
            self._last_fertility_event[source_id] = stats.candidates_processed

            await self._fire_event(StrategyEvent(
                type="fertility_shift",
                source_id=source_id,
                message=(
                    f"Fertility shift on {stats.label or source_id}: "
                    f"{direction} from {baseline:.0%} to {current:.0%}\n\n"
                    f"{self._pool.format_status()}"
                ),
                data={"baseline": baseline, "current": current, "direction": direction},
            ))

    async def _check_pool_empty(self) -> None:
        rows_done = self._generation_stats.get("rows_generated", 0)
        if rows_done >= self._target_rows:
            return

        if self._pool.total_pending > 0:
            return

        now = time.time()
        if now - self._last_pool_empty_time < self._pool_empty_debounce:
            return

        self._last_pool_empty_time = now

        await self._fire_event(StrategyEvent(
            type="pool_empty",
            message=(
                f"Pool empty: all candidates consumed but "
                f"{self._target_rows - rows_done} more rows needed.\n\n"
                f"{self._pool.format_status()}"
            ),
        ))

    async def _fire_event(self, event: StrategyEvent) -> None:
        logger.info(f"[StrategyMonitor] Event: {event.type} — {event.message[:120]}")
        await self._event_queue.put(event)
