import numpy as np
from typing import List, Dict, Optional
from dataclasses import dataclass, field
from collections import defaultdict
import random


@dataclass
class ScoredSeed:
    id: str
    text: str
    source_origin: str  # "upload" or "web"
    scores: Dict[str, Dict[str, float]]  # {axis_name: {value: score}}


@dataclass
class DiversityAxis:
    name: str
    weights: Dict[str, float]  # {value: weight} e.g., {"horror": 0.3, "romance": 0.7}
    source_rule: str = "any"  # "uploads_only" | "web_only" | "any"


@dataclass
class QuotaSlot:
    """Represents one slot to fill in the dataset."""
    assignments: Dict[str, str]  # {axis_name: value}
    count_needed: int
    count_filled: int = 0


@dataclass 
class SelectedSeed:
    seed_id: str
    assigned: Dict[str, str]


def compute_quota_slots(
    axes: List[DiversityAxis],
    target_count: int
) -> List[QuotaSlot]:
    """
    Compute all cross-axis combinations and their target counts.
    """
    if not axes:
        return []
    
    slots = []
    first_axis = axes[0]
    for value, weight in first_axis.weights.items():
        slots.append({
            "assignments": {first_axis.name: value},
            "weight": weight
        })
    
    for axis in axes[1:]:
        new_slots = []
        for slot in slots:
            for value, weight in axis.weights.items():
                new_slots.append({
                    "assignments": {**slot["assignments"], axis.name: value},
                    "weight": slot["weight"] * weight
                })
        slots = new_slots
    
    quota_slots = []
    for slot in slots:
        count = max(1, round(slot["weight"] * target_count))
        quota_slots.append(QuotaSlot(
            assignments=slot["assignments"],
            count_needed=count
        ))
    
    return quota_slots


def seed_passes_source_rules(
    seed: ScoredSeed,
    axes: List[DiversityAxis]
) -> bool:
    """Check if seed's source origin passes all axis source rules."""
    for axis in axes:
        if axis.source_rule == "uploads_only" and seed.source_origin != "upload":
            return False
        if axis.source_rule == "web_only" and seed.source_origin != "web":
            return False
    return True


def compute_seed_score_for_slot(
    seed: ScoredSeed,
    slot: QuotaSlot
) -> float:
    """
    Compute how well a seed fits a slot.
    Multiplies scores across all assigned axes.
    """
    total_score = 1.0
    for axis_name, target_value in slot.assignments.items():
        if axis_name in seed.scores and target_value in seed.scores[axis_name]:
            total_score *= seed.scores[axis_name][target_value]
        else:
            total_score *= 0.01
    return total_score


def select_seeds(
    seeds: List[ScoredSeed],
    axes: List[DiversityAxis],
    target_count: int,
    prioritize_uploads: bool = True
) -> List[SelectedSeed]:
    """
    Select seeds to fill diversity quotas.
    
    Current: Greedy per-slot selection.
    
    ============================================================================
    FUTURE ENHANCEMENTS
    ============================================================================
    
    1. DIVERSITY PENALTY
       Problem: Greedy may pick semantically similar seeds (e.g., 5 "mafia boss" scenarios)
       Solution: After each selection, penalize seeds similar to already-selected:
       
           final_score = base_score - (max_similarity_to_selected * penalty_weight)
       
       Requires: Keep seed embeddings through selection phase
       Adds: ~20 lines, need embeddings in ScoredSeed dataclass
    
    2. GLOBAL OPTIMIZATION  
       Problem: Greedy can waste good seeds. Seed A might be the only decent 
                "romance" option but gets grabbed for "crime" first.
       Solution: Use Hungarian algorithm (scipy.optimize.linear_sum_assignment)
                 to find globally optimal seed-to-slot assignment.
       
           cost_matrix[i][j] = -score(seed_i, slot_j)
           seed_indices, slot_indices = linear_sum_assignment(cost_matrix)
       
       Requires: scipy
       Adds: ~15 lines, O(n³) vs current O(n²)
    
    3. RE-BALANCING / GAP HANDLING
       Problem: Currently just warns "X slots unfilled". User has no recourse.
       Solution: Return structured gap report:
       
           {
               "selected": [...],
               "gaps": [
                   {"slot": {"genre": "romance"}, "needed": 20, "search_query": "romance scenario"}
               ],
               "achievable_distribution": {"horror": 0.7, "romance": 0.3},
               "requested_distribution": {"horror": 0.5, "romance": 0.5}
           }
       
       Then either:
       A. Auto-trigger web search for gap categories
       B. Ask user to approve relaxed quotas
       C. Fill with lower-scoring seeds (relax threshold)
       
       Adds: ~30 lines, improves UX significantly
    
    ============================================================================
    """
    # Step 1: Compute quota slots
    slots = compute_quota_slots(axes, target_count)
    
    # Step 2: Filter seeds by source rules
    eligible_seeds = [s for s in seeds if seed_passes_source_rules(s, axes)]
    
    # Step 3: Separate by source if prioritizing uploads
    if prioritize_uploads:
        upload_seeds = [s for s in eligible_seeds if s.source_origin == "upload"]
        web_seeds = [s for s in eligible_seeds if s.source_origin == "web"]
    else:
        upload_seeds = eligible_seeds
        web_seeds = []
    
    used_seed_ids = set()
    selected = []
    
    # Step 4: Fill slots, prioritizing uploads
    for seed_pool, pool_name in [(upload_seeds, "upload"), (web_seeds, "web")]:
        if not seed_pool:
            continue
            
        for slot in slots:
            while slot.count_filled < slot.count_needed:
                best_seed = None
                best_score = -1
                
                for seed in seed_pool:
                    if seed.id in used_seed_ids:
                        continue
                    
                    score = compute_seed_score_for_slot(seed, slot)
                    
                    # FUTURE: Add diversity penalty here
                    # if selected_embeddings:
                    #     max_sim = max(cosine_sim(seed.embedding, e) for e in selected_embeddings)
                    #     score -= max_sim * diversity_penalty_weight
                    
                    if score > best_score:
                        best_score = score
                        best_seed = seed
                
                if best_seed is None:
                    break
                
                used_seed_ids.add(best_seed.id)
                selected.append(SelectedSeed(
                    seed_id=best_seed.id,
                    assigned=slot.assignments.copy()
                ))
                slot.count_filled += 1
    
    # Step 5: Report unfilled slots
    # FUTURE: Return structured gap report instead of just warning
    unfilled = sum(slot.count_needed - slot.count_filled for slot in slots)
    if unfilled > 0:
        print(f"Warning: {unfilled} slots could not be filled due to insufficient seeds")
        # FUTURE: Return which slots are unfilled and suggested search queries
    
    return selected


# Example usage
if __name__ == "__main__":
    # Mock scored seeds (in reality, from Phase 2)
    seeds = [
        ScoredSeed(
            id="seed_1",
            text="Damian mafia boss...",
            source_origin="upload",
            scores={
                "genre": {"romance": 0.12, "action": 0.14, "horror": 0.19, "crime": 0.27},
                "tone": {"dark": 0.19, "lighthearted": 0.08}
            }
        ),
        ScoredSeed(
            id="seed_2",
            text="Jayden secret agent...",
            source_origin="upload",
            scores={
                "genre": {"romance": 0.06, "action": 0.17, "horror": 0.07, "crime": 0.13},
                "tone": {"dark": 0.15, "lighthearted": 0.05}
            }
        ),
        ScoredSeed(
            id="seed_3",
            text="Web sourced romance story...",
            source_origin="web",
            scores={
                "genre": {"romance": 0.45, "action": 0.05, "horror": 0.03, "crime": 0.04},
                "tone": {"dark": 0.05, "lighthearted": 0.35}
            }
        ),
    ]
    
    # Diversity axes with quotas
    axes = [
        DiversityAxis(
            name="genre",
            weights={"crime": 0.5, "romance": 0.5},
            source_rule="any"
        ),
        DiversityAxis(
            name="tone", 
            weights={"dark": 0.5, "lighthearted": 0.5},
            source_rule="any"
        ),
    ]
    
    # Select seeds for 4 rows
    selected = select_seeds(
        seeds=seeds,
        axes=axes,
        target_count=4,
        prioritize_uploads=True
    )
    
    # Pretty print
    import json
    print(json.dumps([{"seed_id": s.seed_id, "assigned": s.assigned} for s in selected], indent=2))
    
    # Show slot fill status
    print("\nQuota slots:")
    slots = compute_quota_slots(axes, 4)
    for slot in slots:
        print(f"  {slot.assignments}: need {slot.count_needed}")