import numpy as np
from typing import List, Dict
from dataclasses import dataclass
from openai import OpenAI

client = OpenAI()


@dataclass
class Seed:
    id: str
    text: str
    source_chunk_id: str


@dataclass
class DiversityAxis:
    name: str
    values: List[str]  # e.g., ["horror", "romance", "comedy"]


def compute_embeddings(texts: List[str], model: str = "text-embedding-3-large") -> List[np.ndarray]:
    """Embed a list of texts. Returns list of numpy arrays."""
    if not texts:
        return []
    
    response = client.embeddings.create(
        model=model,
        input=texts,
        encoding_format="float"
    )
    
    sorted_data = sorted(response.data, key=lambda d: d.index)
    return [np.array(item.embedding, dtype=np.float32) for item in sorted_data]


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def score_seeds(
    seeds: List[Seed],
    axes: List[DiversityAxis]
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """
    Score all seeds against all diversity axis values.
    
    Returns:
        {
            "seed_id": {
                "axis_name": {"value1": 0.82, "value2": 0.65},
                ...
            }
        }
    """
    # Step 1: Embed all axis values (once)
    axis_embeddings = {}
    for axis in axes:
        embeddings = compute_embeddings(axis.values)
        axis_embeddings[axis.name] = {
            value: emb for value, emb in zip(axis.values, embeddings)
        }
    
    # Step 2: Embed all seeds
    seed_texts = [seed.text for seed in seeds]
    seed_embeddings = compute_embeddings(seed_texts)
    
    # Step 3: Compute scores
    results = {}
    for seed, seed_emb in zip(seeds, seed_embeddings):
        results[seed.id] = {}
        
        for axis in axes:
            results[seed.id][axis.name] = {}
            for value, axis_emb in axis_embeddings[axis.name].items():
                score = cosine_similarity(seed_emb, axis_emb)
                results[seed.id][axis.name][value] = round(score, 4)
    
    return results


# Example usage
if __name__ == "__main__":
    # Mock seeds (in reality, from Phase 1)
    seeds = [
        Seed(
            id="seed_1",
            text="*His name is Damian, he's a dangerous criminal knows also as a mafia boss in the town. You just moved in the town and wanted to know your neighbors...",
            source_chunk_id="chunk_1"
        ),
        Seed(
            id="seed_2", 
            text="You and Jayden are both agents for different organizations. He works for The New Day, his code name is ND 471...",
            source_chunk_id="chunk_2"
        ),
    ]
    
    # Mock diversity axes (in reality, from user config)
    axes = [
        DiversityAxis(name="genre", values=["romance", "action", "horror", "comedy", "drama", "secret agents", "crime", "Agent Jayden", "ND 471"]),
        DiversityAxis(name="tone", values=["dark", "lighthearted", "serious", "playful"]),
    ]
    
    scores = score_seeds(seeds, axes)
    
    # Pretty print
    import json
    print(json.dumps(scores, indent=2))
