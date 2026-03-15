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
      - set_identity_columns(columns)

    No seed-level dedup — dedup happens at row level inside the row generator.
    """

    def __init__(
        self,
        work_queue: asyncio.Queue,
        on_checkpoint: Optional[Callable[[Dict], Awaitable[None]]] = None,
        target_rows: int = 100,
    ) -> None:
        self._work_queue = work_queue
        self._on_checkpoint = on_checkpoint
        self._target_rows = target_rows

        # Set incrementally by orchestrator tools
        self._instructions: Optional[str] = None
        self._candidate_description: str = ""
        self._research_context: str = ""          # orchestrator note for row generators
        self._identity_columns: List[str] = []    # columns that trigger dedup check

        self._lock = asyncio.Lock()
        self._accepted = 0
        self._submitted_total = 0

        # Per-harvester contribution tracking (elastic fair share)
        self._harvester_contributions: Dict[str, int] = {}

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
        return {
            "accepted": self._accepted,
            "submitted_total": self._submitted_total,
            "target": self._target_rows,
            "remaining": max(0, self._target_rows - self._accepted),
        }

    def _is_over_fair_share(self, harvester_id: str) -> bool:
        """Check if a harvester has contributed more than its elastic fair share."""
        if not harvester_id or not self._harvester_contributions:
            return False
        num_harvesters = len(self._harvester_contributions)
        if num_harvesters <= 1:
            return False
        fair_share = self._target_rows / num_harvesters
        return self._harvester_contributions.get(harvester_id, 0) > fair_share * 1.5

    async def submit_seed(self, seed: Seed, harvester_id: str = "") -> Dict[str, Any]:
        """
        Accept a candidate and queue it for row generation.

        Returns status dict: {accepted, stats, over_fair_share?}.
        """
        if not self._instructions:
            return {"accepted": False, "reason": "no instructions set", "stats": self.stats}

        async with self._lock:
            self._submitted_total += 1
            if harvester_id and harvester_id not in self._harvester_contributions:
                self._harvester_contributions[harvester_id] = 0

        work_item = self._build_work_item(seed)

        async with self._lock:
            self._accepted += 1
            if harvester_id:
                self._harvester_contributions[harvester_id] = (
                    self._harvester_contributions.get(harvester_id, 0) + 1
                )

        if self._on_checkpoint:
            await self._on_checkpoint(work_item)

        await self._work_queue.put(work_item)

        over_share = self._is_over_fair_share(harvester_id) if harvester_id else False

        logger.info(
            f"[SeedProcessor] Candidate accepted ({self._accepted}/{self._target_rows})"
            f"{' [over fair share]' if over_share else ''}"
        )

        status = {"accepted": True, "stats": self.stats}
        if over_share:
            status["over_fair_share"] = True
        return status

    def _build_work_item(self, seed: Seed) -> Dict[str, Any]:
        """Build a work item for the row generator."""
        return {
            "instructions": self._instructions,
            "candidate": seed.values,
            "research_context": self._research_context,
            "tags": seed.metadata.get("tags", {}),
        }
