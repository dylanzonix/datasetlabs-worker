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


class SeedProcessor:
    """
    Accepts candidates from harvesters and dispatches them to the work queue.

    Configured incrementally by the orchestrator:
      - set_instructions(instructions, candidate_description)

    No seed-level dedup — dedup happens at row level inside the row generator
    via token Jaccard similarity on all columns.

    Backpressure is organic: when the bounded work queue is full, submit_seed()
    blocks until row generators consume items, which throttles extractors and
    crawlers upstream.
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

        self._lock = asyncio.Lock()
        self._accepted = 0
        self._submitted_total = 0

    # --- Incremental configuration ---

    def set_instructions(self, instructions: str, candidate_description: str = "") -> None:
        self._instructions = instructions
        if candidate_description:
            self._candidate_description = candidate_description

    def set_research_context(self, context: str) -> None:
        self._research_context = context

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

    def _is_full(self) -> bool:
        """Return True if target rows have been generated."""
        rows_done = self._generation_stats.get("rows_generated", 0)
        return rows_done >= self._target_rows

    async def submit_seed(self, seed: Seed, harvester_id: str = "") -> Dict[str, Any]:
        """
        Accept a candidate and queue it for row generation.

        Returns status dict: {accepted, stats}.
        Blocks if the work queue is full (organic backpressure) — extractors
        and crawlers slow down naturally until row generators catch up.
        """
        if not self._instructions:
            return {"accepted": False, "reason": "no instructions set", "stats": self.stats}

        async with self._lock:
            self._submitted_total += 1
            if self._is_full():
                logger.debug(
                    f"[SeedProcessor] Full: {self._accepted} accepted, "
                    f"{self._generation_stats.get('rows_generated', 0)} rows done"
                )
                return {"accepted": False, "reason": "target_reached", "stats": self.stats}
            self._accepted += 1

        work_item = self._build_work_item(seed)

        if self._on_checkpoint:
            await self._on_checkpoint(work_item)

        # Backpressure loop: if the bounded queue is full, wait with a timeout
        # so we can re-check _is_full() and bail if quota was reached while
        # we were blocked (prevents deadlock if consumer stops).
        while True:
            if self._is_full():
                return {"accepted": True, "stats": self.stats}
            try:
                await asyncio.wait_for(self._work_queue.put(work_item), timeout=2.0)
                break
            except asyncio.TimeoutError:
                continue

        logger.info(
            f"[SeedProcessor] Candidate accepted ({self._accepted}, "
            f"target={self._target_rows}, "
            f"rows_done={self._generation_stats.get('rows_generated', 0)})"
        )

        return {"accepted": True, "stats": self.stats}

    def _build_work_item(self, seed: Seed) -> Dict[str, Any]:
        """Build a work item for the row generator."""
        item: Dict[str, Any] = {
            "instructions": self._instructions,
            "candidate": seed.values,
            "research_context": self._research_context,
            "tags": seed.metadata.get("tags", {}),
        }
        if seed.metadata.get("source_url"):
            item["source_url"] = seed.metadata["source_url"]
        return item
