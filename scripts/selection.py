import numpy as np
from typing import List, Dict, Tuple
from dataclasses import dataclass
from collections import defaultdict
from openai import OpenAI

client = OpenAI()


@dataclass
class ScoredSeed:
    id: str
    text: str
    source_origin: str  # "upload" or "web"
    scores: Dict[str, Dict[str, float]]  # {axis_name: {value: score}}
    embedding: np.ndarray = None


@dataclass
class DiversityAxis:
    name: str
    weights: Dict[str, float]
    source_rule: str = "any"  # "uploads_only" | "web_only" | "any"


@dataclass
class QuotaSlot:
    assignments: Dict[str, str]
    count_needed: int
    count_filled: int = 0


@dataclass
class SelectedSeed:
    seed_id: str
    assigned: Dict[str, str]
    score: float


@dataclass
class GapInfo:
    slot_assignments: Dict[str, str]
    needed: int
    filled: int
    suggested_search_query: str


@dataclass
class SelectionResult:
    selected: List[SelectedSeed]
    gaps: List[GapInfo]
    total_requested: int
    total_filled: int


def compute_embeddings(texts: List[str]) -> List[np.ndarray]:
    """Embed texts via OpenAI API."""
    if not texts:
        return []
    
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=texts,
        encoding_format="float"
    )
    sorted_data = sorted(response.data, key=lambda d: d.index)
    return [np.array(item.embedding, dtype=np.float32) for item in sorted_data]


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def compute_quota_slots(axes: List[DiversityAxis], target_count: int) -> List[QuotaSlot]:
    """Compute all cross-axis combinations and their target counts."""
    if not axes:
        return []
    
    slots = [{"assignments": {axes[0].name: v}, "weight": w} for v, w in axes[0].weights.items()]
    
    for axis in axes[1:]:
        slots = [
            {"assignments": {**s["assignments"], axis.name: v}, "weight": s["weight"] * w}
            for s in slots for v, w in axis.weights.items()
        ]
    
    total_weight = sum(s["weight"] for s in slots)
    return [
        QuotaSlot(
            assignments=s["assignments"],
            count_needed=max(1, round((s["weight"] / total_weight) * target_count))
        )
        for s in slots
    ]


def seed_passes_source_rules(seed: ScoredSeed, axes: List[DiversityAxis]) -> bool:
    for axis in axes:
        if axis.source_rule == "uploads_only" and seed.source_origin != "upload":
            return False
        if axis.source_rule == "web_only" and seed.source_origin != "web":
            return False
    return True


def compute_seed_score_for_slot(seed: ScoredSeed, slot: QuotaSlot) -> float:
    score = 1.0
    for axis_name, target_value in slot.assignments.items():
        if axis_name in seed.scores and target_value in seed.scores[axis_name]:
            score *= seed.scores[axis_name][target_value]
        else:
            score *= 0.01
    return score


def generate_search_query(assignments: Dict[str, str]) -> str:
    return " ".join(assignments.values()) + " scenario"


def select_seeds(
    seeds: List[ScoredSeed],
    axes: List[DiversityAxis],
    target_count: int,
    prioritize_uploads: bool = True,
    top_k_per_slot: int = 100
) -> SelectionResult:
    """
    Select seeds to fill diversity quotas.
    
    Uses pre-filter + global optimization:
    1. For each slot, get top K candidates
    2. Pool all candidates
    3. Run Hungarian algorithm on pool
    4. Apply diversity penalty adaptively
    """
    from scipy.optimize import linear_sum_assignment
    
    # Compute slots
    slots = compute_quota_slots(axes, target_count)
    
    # Filter by source rules
    eligible = [s for s in seeds if seed_passes_source_rules(s, axes)]
    
    # Ensure all seeds have embeddings
    needs_embedding = [s for s in eligible if s.embedding is None]
    if needs_embedding:
        texts = [s.text for s in needs_embedding]
        embeddings = compute_embeddings(texts)
        for seed, emb in zip(needs_embedding, embeddings):
            seed.embedding = emb
    
    # Separate by source priority
    if prioritize_uploads:
        upload_seeds = [s for s in eligible if s.source_origin == "upload"]
        web_seeds = [s for s in eligible if s.source_origin == "web"]
    else:
        upload_seeds = eligible
        web_seeds = []
    
    # Pre-filter: get top K candidates per slot
    candidate_ids = set()
    for slot in slots:
        for seed_pool in [upload_seeds, web_seeds]:
            scored = [(s, compute_seed_score_for_slot(s, slot)) for s in seed_pool]
            scored.sort(key=lambda x: -x[1])
            for seed, _ in scored[:top_k_per_slot]:
                candidate_ids.add(seed.id)
    
    candidates = [s for s in eligible if s.id in candidate_ids]
    
    if not candidates:
        return SelectionResult(
            selected=[],
            gaps=[GapInfo(s.assignments, s.count_needed, 0, generate_search_query(s.assignments)) for s in slots],
            total_requested=sum(s.count_needed for s in slots),
            total_filled=0
        )
    
    # Expand slots into positions
    positions = [(slot_idx, slot) for slot_idx, slot in enumerate(slots) for _ in range(slot.count_needed)]
    
    n_candidates = len(candidates)
    n_positions = len(positions)
    
    # Build cost matrix
    cost_matrix = np.full((n_candidates, n_positions), 1000.0)
    
    for i, seed in enumerate(candidates):
        for j, (slot_idx, slot) in enumerate(positions):
            score = compute_seed_score_for_slot(seed, slot)
            if prioritize_uploads and seed.source_origin == "upload":
                score += 0.001
            cost_matrix[i, j] = -score
    
    # Hungarian algorithm
    seed_indices, position_indices = linear_sum_assignment(cost_matrix)
    
    # Build initial selection
    raw_selected = []
    for seed_idx, pos_idx in zip(seed_indices, position_indices):
        if cost_matrix[seed_idx, pos_idx] >= 999:
            continue
        slot_idx, slot = positions[pos_idx]
        seed = candidates[seed_idx]
        raw_selected.append((seed, slot.assignments.copy(), -cost_matrix[seed_idx, pos_idx]))
    
    # Apply diversity penalty: re-rank and potentially swap
    # Adaptive: score *= (1 - max_similarity_to_already_selected)
    selected = []
    selected_embeddings = []
    
    for seed, assigned, base_score in raw_selected:
        if selected_embeddings:
            max_sim = max(cosine_similarity(seed.embedding, e) for e in selected_embeddings)
            final_score = base_score * (1 - max_sim)
        else:
            final_score = base_score
        
        selected.append(SelectedSeed(
            seed_id=seed.id,
            assigned=assigned,
            score=final_score
        ))
        selected_embeddings.append(seed.embedding)
    
    # Update slot fill counts
    slot_fills = defaultdict(int)
    for s in selected:
        key = tuple(sorted(s.assigned.items()))
        slot_fills[key] += 1
    
    for slot in slots:
        key = tuple(sorted(slot.assignments.items()))
        slot.count_filled = slot_fills.get(key, 0)
    
    # Identify gaps
    gaps = [
        GapInfo(
            slot_assignments=slot.assignments.copy(),
            needed=slot.count_needed,
            filled=slot.count_filled,
            suggested_search_query=generate_search_query(slot.assignments)
        )
        for slot in slots if slot.count_filled < slot.count_needed
    ]
    
    return SelectionResult(
        selected=selected,
        gaps=gaps,
        total_requested=sum(s.count_needed for s in slots),
        total_filled=len(selected)
    )


if __name__ == "__main__":
    seeds = [
        ScoredSeed(
            id="seed_1",
            text="Damian mafia boss meets new neighbor with cookies",
            source_origin="upload",
            scores={
                "genre": {"romance": 0.12, "crime": 0.27},
                "tone": {"dark": 0.19, "lighthearted": 0.08}
            }
        ),
        ScoredSeed(
            id="seed_2",
            text="Jayden and rival agent at gala, who dies first",
            source_origin="upload",
            scores={
                "genre": {"romance": 0.06, "crime": 0.13},
                "tone": {"dark": 0.15, "lighthearted": 0.05}
            }
        ),
        ScoredSeed(
            id="seed_3",
            text="Sweet summer romance at the beach",
            source_origin="web",
            scores={
                "genre": {"romance": 0.45, "crime": 0.04},
                "tone": {"dark": 0.05, "lighthearted": 0.35}
            }
        ),
    ]
    
    axes = [
        DiversityAxis(name="genre", weights={"crime": 0.5, "romance": 0.5}),
        DiversityAxis(name="tone", weights={"dark": 0.5, "lighthearted": 0.5}),
    ]
    
    result = select_seeds(seeds=seeds, axes=axes, target_count=4, prioritize_uploads=True)
    
    print("=== SELECTED ===")
    for s in result.selected:
        print(f"  {s.seed_id}: {s.assigned} (score: {s.score:.4f})")
    
    print(f"\n=== FILL: {result.total_filled}/{result.total_requested} ===")
    
    if result.gaps:
        print(f"\n=== GAPS ===")
        for g in result.gaps:
            print(f"  {g.slot_assignments}: {g.filled}/{g.needed}")
            print(f"    → \"{g.suggested_search_query}\"")