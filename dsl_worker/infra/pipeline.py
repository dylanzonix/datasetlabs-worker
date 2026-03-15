"""
Pipeline configuration and seed processing.

V8: Simplified. SeedProcessor accepts raw candidates (no variable declarations,
no template interpolation, no seed-level dedup). Candidates flow directly to the
work queue. Row generators receive the raw candidate + instructions.

Dedup happens at row level via set_column() in the row generator.
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class Seed:
    """A candidate item that will become one row."""
    values: Any = None              # raw candidate: string, dict, or any extracted value
    metadata: Dict[str, Any] = field(default_factory=dict)


OVERSHOOT_FACTOR = 3  # accept up to 3x the remaining rows needed as in-flight seeds


class SeedProcessor:
    """
    Accepts candidates from harvesters and dispatches them to the work queue.

    Configured incrementally by the orchestrator:
      - set_instructions(instructions, candidate_description)
      - set_identity_columns(columns)

    No seed-level dedup — dedup happens at row level inside the row generator.

    Backpressure: stops accepting seeds when in-flight seeds exceed
    remaining rows needed * OVERSHOOT_FACTOR, preventing crawlers from
    producing far more seeds than the target requires.
    """

    def __init__(
        self,
        work_queue: asyncio.Queue,
        on_checkpoint: Optional[Callable[[Dict], Awaitable[None]]] = None,
        target_rows: int = 100,
        generation_stats: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._work_queue = work_queue
        self._on_checkpoint = on_checkpoint
        self._target_rows = target_rows
        self._generation_stats = generation_stats or {}

        # Set incrementally by orchestrator tools
        self._instructions: Optional[str] = None
        self._candidate_description: str = ""
        self._research_context: str = ""          # orchestrator note for row generators
        self._identity_columns: List[str] = []    # columns that trigger dedup check

        self._lock = asyncio.Lock()
        self._accepted = 0
        self._submitted_total = 0

    # --- Incremental configuration ---

    def set_instructions(self, instructions: str, candidate_description: str = "") -> None:
        self._instructions = instructions
        if candidate_description:
            self._candidate_description = candidate_description

    def set_identity_columns(self, columns: List[str]) -> None:
        self._identity_columns = list(columns)

    def set_research_context(self, context: str) -> None:
        self._research_context = context

    @property
    def identity_columns(self) -> List[str]:
        return self._identity_columns

    @property
    def accepted_count(self) -> int:
        return self._accepted

    @property
    def stats(self) -> Dict[str, Any]:
        rows_done = self._generation_stats.get("rows_generated", 0)
        return {
            "accepted": self._accepted,
            "submitted_total": self._submitted_total,
            "target": self._target_rows,
            "remaining": max(0, self._target_rows - rows_done),
        }

    def _is_backpressured(self) -> bool:
        """Return True if we have enough seeds in flight to hit the target."""
        rows_done = self._generation_stats.get("rows_generated", 0)
        rows_skipped = self._generation_stats.get("skipped", 0)
        processed = rows_done + rows_skipped
        in_flight = self._accepted - processed
        rows_needed = max(0, self._target_rows - rows_done)
        if rows_needed == 0:
            return True
        return in_flight > rows_needed * OVERSHOOT_FACTOR

    async def submit_seed(self, seed: Seed, harvester_id: str = "") -> Dict[str, Any]:
        """
        Accept a candidate and queue it for row generation.

        Returns status dict: {accepted, stats}.
        Rejects with backpressure if in-flight seeds already cover the target.
        """
        if not self._instructions:
            return {"accepted": False, "reason": "no instructions set", "stats": self.stats}

        async with self._lock:
            self._submitted_total += 1
            if self._is_backpressured():
                logger.debug(
                    f"[SeedProcessor] Backpressure: {self._accepted} accepted, "
                    f"{self._generation_stats.get('rows_generated', 0)} rows done"
                )
                return {"accepted": False, "reason": "backpressure", "stats": self.stats}
            self._accepted += 1

        work_item = self._build_work_item(seed)

        if self._on_checkpoint:
            await self._on_checkpoint(work_item)

        await self._work_queue.put(work_item)

        logger.info(
            f"[SeedProcessor] Candidate accepted ({self._accepted}, "
            f"target={self._target_rows}, "
            f"rows_done={self._generation_stats.get('rows_generated', 0)})"
        )

        return {"accepted": True, "stats": self.stats}

    def _build_work_item(self, seed: Seed) -> Dict[str, Any]:
        """Build a work item for the row generator."""
        return {
            "instructions": self._instructions,
            "candidate": seed.values,
            "research_context": self._research_context,
            "tags": seed.metadata.get("tags", {}),
        }
