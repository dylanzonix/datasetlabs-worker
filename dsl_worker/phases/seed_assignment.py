"""
Phase: Seed Assignment (Streaming)

Assigns seeds to diversity slots using Jonker-Volgenant algorithm.
Supports streaming - can assign in batches as seeds become available.

Key design:
- Can run incrementally (batch_mode=True) or wait for all seeds (batch_mode=False)
- Tracks which seeds have been assigned
- Generation can start as soon as first batch is assigned
- Re-assigns when new batches arrive for better global optimization

Resume behavior:
- Assignment state is ephemeral (not persisted)
- On resume, re-computes from scored seeds
"""

import logging
import hashlib
import json
import numpy as np
from typing import List, Dict, Tuple, Set, Optional
from dataclasses import dataclass
from scipy.optimize import linear_sum_assignment
from uuid import UUID

from dsl_worker.phases.base import Phase, PhaseResult
from dsl_api.models.project_seed import ProjectSeed

logger = logging.getLogger(__name__)


@dataclass
class QuotaSlot:
    """A slot in the diversity quota matrix."""
    assignments: Dict[str, str]  # {axis_name: axis_value}
    count_needed: int


@dataclass
class AssignedSeed:
    """A seed with its diversity assignment."""
    seed_id: str
    seed_text: str
    diversity_assignments: Dict[str, str]
    score: float


class SeedAssignmentPhase(Phase):
    """
    Assign seeds to diversity slots.

    Uses Jonker-Volgenant (Hungarian) algorithm for optimal assignment.

    Modes:
    - streaming (batch_mode=True): Assign in batches as seeds become available
    - complete (batch_mode=False): Wait for all seeds before assigning

    The streaming mode allows generation to start earlier at the cost of
    potentially suboptimal assignments (since we don't see all candidates).
    """

    def __init__(
        self,
        *args,
        batch_mode: bool = False,  # Set True for streaming
        batch_threshold: int = 100,  # Min seeds before first batch assignment
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.batch_mode = batch_mode
        self.batch_threshold = batch_threshold

        # Assignment state
        self._assigned_seeds: List[AssignedSeed] = []
        self._assigned_seed_ids: Set[str] = set()
        self._last_scored_count: int = 0
        self._assignment_complete: bool = False

        # For detecting when we need to re-assign
        self._last_seed_hash: str = ""

    def should_run(self) -> bool:
        """
        Determine if assignment should run.

        Flow control:
        - If no seeds exist and all preprocessing complete: mark done
        - Preview mode: run as soon as any seeds are scored
        - Batch mode: run when batch_threshold new seeds are scored
        - Normal mode: wait for ALL preprocessing to complete
        """
        if self._assignment_complete:
            # Check if we need to re-assign (new seeds scored since last run)
            if self.batch_mode and self._should_reassign():
                return True
            return False

        # Handle no-files/no-seeds case
        if self.state.seeds_scored == 0 and not self.state.has_unscored_seeds():
            if not self.state.has_unprocessed_files() and not self.state.has_chunks_without_seeds():
                logger.info("No seeds available (no files or empty files), marking assignment complete")
                self._assignment_complete = True
                self._assigned_seeds = []
                return False

        # Need at least some scored seeds
        if self.state.seeds_scored == 0:
            return False

        # Preview mode: assign immediately
        if self.state.preview_mode:
            return True

        # Batch mode: assign when we have enough new seeds
        if self.batch_mode:
            new_seeds = self.state.seeds_scored - self._last_scored_count
            if new_seeds >= self.batch_threshold or self._preprocessing_complete():
                return True
            return False

        # Normal mode: wait for ALL preprocessing to complete
        # This is crucial for resume scenarios with new files
        if self.state.has_unprocessed_files():
            return False

        if self.state.has_chunks_without_seeds():
            return False

        if self.state.has_unscored_seeds():
            return False

        return True

    def _preprocessing_complete(self) -> bool:
        """Check if all preprocessing phases are done."""
        return (
            not self.state.has_unprocessed_files() and
            not self.state.has_chunks_without_seeds() and
            not self.state.has_unscored_seeds()
        )

    def _should_reassign(self) -> bool:
        """Check if we should re-run assignment (new seeds available)."""
        if not self.batch_mode:
            return False

        current_count = self.state.seeds_scored
        new_seeds = current_count - self._last_scored_count

        # Re-assign if significant new seeds or preprocessing complete
        if new_seeds >= self.batch_threshold:
            return True

        if self._preprocessing_complete() and new_seeds > 0:
            return True

        return False

    async def execute_once(self) -> PhaseResult:
        """Run assignment on scored seeds."""
        seeds = self.state.get_scored_seeds()
        if not seeds:
            return PhaseResult.no_work()

        logger.info(f"[{self.name}] Assigning {len(seeds)} seeds to diversity slots")

        try:
            # Run the assignment algorithm
            assigned = self._run_assignment(seeds)

            # Update state
            self._assigned_seeds = assigned
            self._assigned_seed_ids = {s.seed_id for s in assigned}
            self._last_scored_count = self.state.seeds_scored

            # Mark complete if preprocessing is done
            if self._preprocessing_complete():
                self._assignment_complete = True

            logger.info(f"[{self.name}] Assigned {len(assigned)} seeds to diversity slots")
            return PhaseResult.work_done(cost_usd=0.0)

        except Exception as e:
            logger.error(f"Assignment failed: {e}", exc_info=True)
            return PhaseResult.no_work()

    def _run_assignment(self, seeds: List[ProjectSeed]) -> List[AssignedSeed]:
        """Run the Jonker-Volgenant assignment algorithm."""
        axes = self._parse_axes()
        slots = self._compute_slots(axes, self.state.num_samples)

        if not slots:
            # No diversity spec - assign all seeds to default
            return [
                AssignedSeed(
                    seed_id=str(seed.id),
                    seed_text=seed.text,
                    diversity_assignments={},
                    score=1.0
                )
                for seed in seeds
            ]

        # Build position list (expand slots by count_needed)
        positions: List[Tuple[int, QuotaSlot]] = []
        for slot_idx, slot in enumerate(slots):
            for _ in range(slot.count_needed):
                positions.append((slot_idx, slot))

        # Build cost matrix: seeds x positions
        n_seeds = len(seeds)
        n_positions = len(positions)

        cost_matrix = np.full((n_seeds, n_positions), 1000.0)

        for i, seed in enumerate(seeds):
            for j, (slot_idx, slot) in enumerate(positions):
                score = self._compute_slot_score(seed, slot)
                cost_matrix[i, j] = -score  # Negate for minimization

        # Run Jonker-Volgenant algorithm
        seed_indices, pos_indices = linear_sum_assignment(cost_matrix)

        # Build assigned seeds list
        assigned = []
        for seed_idx, pos_idx in zip(seed_indices, pos_indices):
            if cost_matrix[seed_idx, pos_idx] >= 999:
                continue  # Skip unassigned

            seed = seeds[seed_idx]
            _, slot = positions[pos_idx]

            assigned.append(AssignedSeed(
                seed_id=str(seed.id),
                seed_text=seed.text,
                diversity_assignments=slot.assignments.copy(),
                score=-cost_matrix[seed_idx, pos_idx]
            ))

        return assigned

    def _parse_axes(self) -> List[Dict]:
        """Parse diversity axes from config."""
        if not self.state.diversity_spec:
            return []

        return [
            {
                "name": axis.get("name"),
                "weights": {v.get("value"): v.get("weight", 1.0) for v in axis.get("values", [])}
            }
            for axis in self.state.diversity_spec
        ]

    def _compute_slots(self, axes: List[Dict], target_count: int) -> List[QuotaSlot]:
        """Compute all cross-axis combinations and their target counts."""
        if not axes:
            return []

        # Start with first axis
        slots = [
            {"assignments": {axes[0]["name"]: v}, "weight": w}
            for v, w in axes[0]["weights"].items()
        ]

        # Cross with remaining axes
        for axis in axes[1:]:
            slots = [
                {
                    "assignments": {**s["assignments"], axis["name"]: v},
                    "weight": s["weight"] * w
                }
                for s in slots
                for v, w in axis["weights"].items()
            ]

        # Convert weights to counts
        total_weight = sum(s["weight"] for s in slots)
        return [
            QuotaSlot(
                assignments=s["assignments"],
                count_needed=max(1, round((s["weight"] / total_weight) * target_count))
            )
            for s in slots
        ]

    def _compute_slot_score(self, seed: ProjectSeed, slot: QuotaSlot) -> float:
        """Compute how well a seed matches a slot."""
        if not seed.scores:
            return 0.01

        score = 1.0
        for axis_name, target_value in slot.assignments.items():
            if axis_name in seed.scores and target_value in seed.scores[axis_name]:
                score *= seed.scores[axis_name][target_value]
            else:
                score *= 0.01

        return score

    def get_assigned_seeds(self) -> List[AssignedSeed]:
        """Get the assigned seeds for generation phase."""
        return self._assigned_seeds

    def get_assigned_count(self) -> int:
        """Get count of assigned seeds."""
        return len(self._assigned_seeds)

    def is_complete(self) -> bool:
        """
        Complete when:
        - Batch mode: Have at least one batch assigned
        - Normal mode: Assignment has run on all seeds
        """
        if self.batch_mode:
            # In batch mode, complete once we have ANY assigned seeds
            # (generation can proceed while we continue assigning)
            return len(self._assigned_seeds) > 0 or self._assignment_complete

        return self._assignment_complete

    def reset(self):
        """Reset assignment state (for fresh start on resume)."""
        self._assigned_seeds = []
        self._assigned_seed_ids = set()
        self._last_scored_count = 0
        self._assignment_complete = False
        self._last_seed_hash = ""