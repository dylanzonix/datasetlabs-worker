"""
Phase: Seed Assignment

Assigns seeds to diversity slots using Jonker-Volgenant algorithm.
This phase is EPHEMERAL - it runs fresh on every start/resume.

Key design:
- Does NOT persist assignment state to database
- Computes assignments in memory from scored seeds
- Passes assignments to generation phase via shared state
- Algorithm requires seeing ALL candidates, so execute_once() does everything
- No API calls, so no costs
"""

import logging
import numpy as np
from typing import List, Dict, Tuple
from dataclasses import dataclass
from scipy.optimize import linear_sum_assignment

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
    This phase is ephemeral - results are computed fresh each time.

    One execute_once() processes ALL seeds at once (algorithm requirement).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._assignment_done = False
        self._assigned_seeds: List[AssignedSeed] = []

    def should_run(self) -> bool:
        """
        Run when scoring is complete and assignment not done yet.

        Flow control:
        - If no seeds exist and scoring is complete: mark done, skip to generation
        - Preview mode: run as soon as any seeds are scored
        - Normal mode: wait for scoring to complete
        """
        if self._assignment_done:
            return False

        # Handle no-files/no-seeds case:
        # If there are no seeds AND no unscored seeds (meaning file processing is done
        # but produced nothing), mark assignment as complete so generation can proceed.
        if self.state.seeds_scored == 0 and not self.state.has_unscored_seeds():
            # Check if file processing is also complete (no unprocessed files)
            if not self.state.has_unprocessed_files():
                logger.info("No seeds available (no files or empty files), marking assignment complete")
                self._assignment_done = True
                self._assigned_seeds = []  # Empty list - generation will create synthetic seeds
                return False

        # Need at least some scored seeds to assign
        if self.state.seeds_scored == 0:
            return False

        # Preview mode: assign as soon as any scored
        if self.state.preview_mode:
            return True

        # Normal mode: wait for scoring to complete
        if self.state.has_unscored_seeds():
            return False

        return True

    async def execute_once(self) -> PhaseResult:
        """Run assignment on ALL scored seeds."""
        seeds = self.state.get_scored_seeds()
        if not seeds:
            return PhaseResult.no_work()

        logger.info(f"[{self.name}] Assigning {len(seeds)} seeds to diversity slots")

        try:
            axes = self._parse_axes()
            slots = self._compute_slots(axes, self.state.num_samples)

            if not slots:
                # No diversity spec, assign all seeds to default
                self._assigned_seeds = [
                    AssignedSeed(
                        seed_id=str(seed.id),
                        seed_text=seed.text,
                        diversity_assignments={},
                        score=1.0
                    )
                    for seed in seeds
                ]
                self._assignment_done = True
                logger.info(f"No diversity spec, assigned all {len(seeds)} seeds to default")
                return PhaseResult.work_done(cost_usd=0.0)

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
            self._assigned_seeds = []
            for seed_idx, pos_idx in zip(seed_indices, pos_indices):
                if cost_matrix[seed_idx, pos_idx] >= 999:
                    continue  # Skip unassigned

                seed = seeds[seed_idx]
                _, slot = positions[pos_idx]

                self._assigned_seeds.append(AssignedSeed(
                    seed_id=str(seed.id),
                    seed_text=seed.text,
                    diversity_assignments=slot.assignments.copy(),
                    score=-cost_matrix[seed_idx, pos_idx]
                ))

            self._assignment_done = True
            logger.info(f"[{self.name}] Assigned {len(self._assigned_seeds)} seeds to diversity slots")
            return PhaseResult.work_done(cost_usd=0.0)

        except Exception as e:
            logger.error(f"Assignment failed: {e}", exc_info=True)
            return PhaseResult.no_work()

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

    def is_complete(self) -> bool:
        """Complete when assignment has been done."""
        return self._assignment_done

    def reset(self):
        """Reset assignment state (for fresh start on resume)."""
        self._assignment_done = False
        self._assigned_seeds = []