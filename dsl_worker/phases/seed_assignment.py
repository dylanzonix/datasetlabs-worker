"""
Phase: Seed Assignment

Assigns seeds to diversity slot COMBINATIONS for generation.

Each row fills one value from EACH diversity axis simultaneously.
E.g., a row is both domain:customer_support AND instruction_type:tone_style.

30 rows = 30 assignments, each filling one slot per axis.

Uses batch ranking:
- Show LLM the remaining quotas per axis
- Ask which combination fits best for each seed
- Prioritize filling slots that are falling behind
"""

import asyncio
import json
import logging
import os
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple

from sqlalchemy import func as sql_func

from dsl_worker.phases.base import Phase, PhaseResult
from dsl_api.models.project_seed import ProjectSeed

logger = logging.getLogger(__name__)

ASSIGNMENT_MODEL = os.getenv("ASSIGNMENT_MODEL", "gpt-5-nano")
BATCH_SIZE = 20


@dataclass
class AssignedSeed:
    """A seed with diversity assignment."""
    seed_id: uuid.UUID
    source: str  # File pointer
    note: str    # Description
    quality_score: float
    diversity_slot: Dict[str, str]  # {axis_name: value} - ALL axes filled


class SeedAssignmentPhase(Phase):
    """
    Assign seeds to diversity slot combinations.
    
    Each assignment fills one value per axis. Quotas are tracked per-axis
    but assignments are combinations.
    """
    
    def __init__(
        self,
        *args,
        stop_checker: Optional[Callable[[], bool]] = None,
        cost_tracker: Optional[Any] = None,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        
        self.stop_checker = stop_checker
        self.cost_tracker = cost_tracker
        
        # Quota tracking per axis: {axis_name: {value: remaining_count}}
        self._quotas: Dict[str, Dict[str, int]] = {}
        
        # Axis names for iteration
        self._axis_names: List[str] = []
        
        # Assigned seeds (ready for generation)
        self._assigned_seeds: List[AssignedSeed] = []
        self._assigned_seed_ids: set = set()
        
        # Total assignments made
        self._assignments_made = 0
        
        # Lock for sequential processing
        self._lock = asyncio.Lock()
        
        # Initialize quotas
        self._init_quotas()
        
        self._total_cost = 0.0
    
    def _init_quotas(self):
        """Initialize quota targets from diversity spec."""
        if not self.state.diversity_spec:
            return
        
        target = self.state.num_samples
        
        for axis in self.state.diversity_spec:
            axis_name = axis.get("name")
            self._axis_names.append(axis_name)
            values = axis.get("values", [])
            
            if not values:
                continue
            
            total_weight = sum(v.get("weight", 1.0) for v in values)
            
            # Calculate quotas with remainder distribution
            quotas = []
            for v in values:
                value_name = v.get("value")
                weight = v.get("weight", 1.0)
                exact = (weight / total_weight) * target
                quotas.append({
                    "value": value_name,
                    "count": int(exact),
                    "remainder": exact - int(exact),
                })
            
            # Distribute remainder to highest-remainder values
            total_assigned = sum(q["count"] for q in quotas)
            remainder = target - total_assigned
            
            quotas.sort(key=lambda x: x["remainder"], reverse=True)
            for i in range(min(remainder, len(quotas))):
                quotas[i]["count"] += 1
            
            self._quotas[axis_name] = {q["value"]: q["count"] for q in quotas}
        
        logger.info(f"[Assignment] Quotas initialized for {len(self._axis_names)} axes, target={target}")
        for axis_name, values in self._quotas.items():
            logger.info(f"[Assignment]   {axis_name}: {values}")
    
    def get_remaining_quotas(self) -> Dict[str, int]:
        """Get remaining quota per axis:value (for display)."""
        remaining = {}
        for axis_name, values in self._quotas.items():
            for value_name, count in values.items():
                if count > 0:
                    remaining[f"{axis_name}:{value_name}"] = count
        return remaining
    
    def get_remaining_by_axis(self) -> Dict[str, Dict[str, int]]:
        """Get remaining quotas organized by axis."""
        return {
            axis: {v: c for v, c in values.items() if c > 0}
            for axis, values in self._quotas.items()
        }
    
    def get_total_remaining(self) -> int:
        """Get total remaining rows needed."""
        return self.state.num_samples - self._assignments_made
    
    def should_run(self) -> bool:
        """Run if there are unassigned seeds and we need more rows."""
        if self.get_total_remaining() <= 0:
            return False
        return self._has_unassigned_seeds()
    
    def _has_unassigned_seeds(self) -> bool:
        """Check if there are unassigned seeds in DB."""
        count = (
            self.db.query(sql_func.count(ProjectSeed.id))
            .filter(
                ProjectSeed.project_id == self.state.project_id,
                ProjectSeed.version_id == self.state.version_id,
                ProjectSeed.deleted_at.is_(None),
            )
            .scalar() or 0
        )
        return count > len(self._assigned_seed_ids)
    
    async def execute_once(self) -> PhaseResult:
        """Process a batch of seeds with ranking."""
        
        async with self._lock:
            if self.get_total_remaining() <= 0:
                logger.info("[Assignment] Target reached!")
                return PhaseResult.no_work()
            
            # Get unassigned seeds
            seeds = self._get_unassigned_seeds(limit=BATCH_SIZE)
            
            if not seeds:
                return PhaseResult.no_work()
            
            remaining_by_axis = self.get_remaining_by_axis()
            
            logger.info(f"[Assignment] Scoring {len(seeds)} seeds, {self.get_total_remaining()} rows remaining")
            
            # Score and rank seeds
            scored_seeds, cost = await self._score_batch(seeds, remaining_by_axis)
            self._total_cost += cost
            
            if self.cost_tracker and cost > 0:
                self.cost_tracker.add_cost(
                    phase=self.name,
                    cost_usd=cost,
                    model=ASSIGNMENT_MODEL,
                )
            
            # Assign top performers
            assigned_count = 0
            for seed_id, slot_combo, score in scored_seeds:
                if self.stop_checker and self.stop_checker():
                    break
                
                if self.get_total_remaining() <= 0:
                    break
                
                # Validate all axes have room in the proposed combo
                can_assign = True
                for axis_name, value in slot_combo.items():
                    if self._quotas.get(axis_name, {}).get(value, 0) <= 0:
                        can_assign = False
                        break
                
                if not can_assign:
                    continue
                
                # Find the seed
                seed = next((s for s in seeds if s.id == seed_id), None)
                if not seed:
                    continue
                
                # Assign - decrement ALL axes
                for axis_name, value in slot_combo.items():
                    self._quotas[axis_name][value] -= 1
                
                metadata = seed.extraction_metadata or {}
                assigned = AssignedSeed(
                    seed_id=seed.id,
                    source=seed.text,
                    note=metadata.get("note", ""),
                    quality_score=score,
                    diversity_slot=slot_combo,  # Full combination
                )
                
                self._assigned_seeds.append(assigned)
                self._assigned_seed_ids.add(seed.id)
                self._assignments_made += 1
                assigned_count += 1
            
            logger.info(f"[Assignment] Assigned {assigned_count} seeds. Remaining: {self.get_total_remaining()}")
            
            return PhaseResult.work_done(cost_usd=cost)
    
    def _get_unassigned_seeds(self, limit: int = 20) -> List[ProjectSeed]:
        """Get unassigned seeds from DB."""
        all_seeds = (
            self.db.query(ProjectSeed)
            .filter(
                ProjectSeed.project_id == self.state.project_id,
                ProjectSeed.version_id == self.state.version_id,
                ProjectSeed.deleted_at.is_(None),
            )
            .all()
        )
        
        unassigned = [s for s in all_seeds if s.id not in self._assigned_seed_ids]
        return unassigned[:limit]
    
    async def _score_batch(
        self,
        seeds: List[ProjectSeed],
        remaining_by_axis: Dict[str, Dict[str, int]],
    ) -> Tuple[List[Tuple[uuid.UUID, Dict[str, str], float]], float]:
        """
        Score a batch of seeds against remaining quotas.
        
        Returns list of (seed_id, slot_combo, score) sorted by score descending.
        slot_combo is {axis: value} for ALL axes.
        """
        
        # Build seeds info
        seeds_info = []
        for seed in seeds:
            metadata = seed.extraction_metadata or {}
            seeds_info.append({
                "id": str(seed.id),
                "source": seed.text,
                "note": metadata.get("note", ""),
            })
        
        # Format quotas by axis
        quotas_formatted = []
        for axis_name, values in remaining_by_axis.items():
            if values:
                values_str = ", ".join([f"{v}: {c}" for v, c in values.items() if c > 0])
                quotas_formatted.append(f"**{axis_name}**: {values_str}")
        
        prompt = f"""Assign diversity slots to these seeds for dataset generation.

## Dataset Goal
{self.state.generation_prompt}

## Remaining Quotas by Axis
Each row needs ONE value from EACH axis. Prioritize filling slots that need more.

{chr(10).join(quotas_formatted)}

## Seeds to Assign
{json.dumps(seeds_info, indent=2)}

## Your Task
For each seed, pick the BEST combination that:
1. Fits the seed's content naturally
2. Helps fill quotas that are falling behind

Return JSON:
{{
  "assignments": [
    {{
      "id": "seed-uuid",
      "slot": {{{", ".join([f'"{axis}": "value"' for axis in self._axis_names])}}},
      "score": 8.5
    }}
  ]
}}

Score 0-10 based on how well the seed fits the assigned combination.
Only use values that exist in the quotas above.
If a seed doesn't fit anything well, give it score 0 and skip it."""

        try:
            response, cost = await self.openai_client.responses_create(
                model=ASSIGNMENT_MODEL,
                input=[{"role": "user", "content": prompt}],
            )
            
            # Parse response
            response_text = response.output_text.strip()
            if response_text.startswith("```"):
                response_text = response_text.split("```")[1]
                if response_text.startswith("json"):
                    response_text = response_text[4:]
            
            data = json.loads(response_text)
            
            # Build scored list
            scored = []
            for item in data.get("assignments", []):
                seed_id = item.get("id")
                slot = item.get("slot", {})
                score = float(item.get("score", 0))
                
                if score <= 0:
                    continue
                
                # Validate slot has all axes
                if not all(axis in slot for axis in self._axis_names):
                    continue
                
                # Validate all values exist in quotas
                valid = True
                for axis, value in slot.items():
                    if axis not in self._quotas or value not in self._quotas[axis]:
                        valid = False
                        break
                if not valid:
                    continue
                
                try:
                    scored.append((uuid.UUID(seed_id), slot, score))
                except ValueError:
                    continue
            
            # Sort by score descending
            scored.sort(key=lambda x: x[2], reverse=True)
            
            return scored, cost.total_cost_usd
            
        except Exception as e:
            logger.error(f"[Assignment] Scoring failed: {e}")
            
            # Fallback: try to assign with most-needed slots
            scored = []
            for seed in seeds:
                # Pick highest-remaining value from each axis
                slot = {}
                for axis_name, values in remaining_by_axis.items():
                    if values:
                        best_value = max(values.items(), key=lambda x: x[1])[0]
                        slot[axis_name] = best_value
                
                if len(slot) == len(self._axis_names):
                    scored.append((seed.id, slot, 5.0))
            
            return scored, 0.0
    
    def get_assigned_seeds(self, limit: Optional[int] = None) -> List[AssignedSeed]:
        """Get assigned seeds ready for generation."""
        if limit:
            return self._assigned_seeds[:limit]
        return self._assigned_seeds.copy()
    
    def get_top_seeds(self, fraction: float = 0.1) -> List[AssignedSeed]:
        """Get top seeds by quality."""
        sorted_seeds = sorted(
            self._assigned_seeds,
            key=lambda s: s.quality_score,
            reverse=True
        )
        count = max(1, int(len(sorted_seeds) * fraction))
        return sorted_seeds[:count]
    
    def mark_seed_used(self, seed_id: uuid.UUID):
        """Mark a seed as used (generated into row)."""
        self._assigned_seeds = [s for s in self._assigned_seeds if s.seed_id != seed_id]
    
    def get_stats(self) -> Dict:
        """Get assignment stats."""
        return {
            "assigned_count": len(self._assigned_seeds),
            "total_assigned": self._assignments_made,
            "remaining_quotas": self.get_remaining_quotas(),
            "total_remaining": self.get_total_remaining(),
        }
    
    def is_complete(self) -> bool:
        """Complete when target rows assigned."""
        return self.get_total_remaining() <= 0
    
    def get_status(self):
        """Get current status."""
        from dsl_worker.phases.base import PhaseStatus
        
        return PhaseStatus(
            phase_name=self.name,
            status="complete" if self.is_complete() else "active",
            progress=f"{self._assignments_made}/{self.state.num_samples} assigned"
        )