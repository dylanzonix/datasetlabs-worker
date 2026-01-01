"""
Phase: Seed Scoring

Scores seeds against diversity axis values using embeddings.
Stores results in project_seeds.scores column.

Resume logic:
- Checks which seeds don't have scored_at set
- Only processes unscored seeds
- If diversity_spec changes, all seeds are invalidated (scores cleared by ProjectState)
"""

import logging
import numpy as np
from typing import List, Dict
from datetime import datetime, timezone

from dsl_worker.phases.base import Phase
from dsl_api.models.project_seed import ProjectSeed

logger = logging.getLogger(__name__)


class SeedScoringPhase(Phase):
    """
    Score seeds for diversity distribution.

    Compares each seed to diversity axis values using cosine similarity.

    One execute_once() processes a batch of seeds.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.batch_size = 20  # Score 20 seeds per iteration
        self._axis_embeddings_cache = None  # Cache embeddings for axis values

    def should_run(self) -> bool:
        """
        Run if there are unscored seeds.

        Flow control:
        - Preview mode: score eagerly whenever seeds are available
        - Normal mode: wait for extraction to complete first
        """
        # No seeds = nothing to score
        if self.state.seeds_extracted == 0:
            return False

        # Check if there's actual work to do
        if not self.state.has_unscored_seeds():
            return False

        return True

    async def execute_once(self) -> bool:
        """Score a batch of seeds."""
        seeds = self.state.get_unscored_seeds(limit=self.batch_size)
        if not seeds:
            return False

        logger.info(f"[{self.name}] Scoring {len(seeds)} seeds")

        # Get diversity axes
        diversity_axes = self._parse_diversity_spec()

        if not diversity_axes:
            # No diversity spec, mark all as scored with empty scores
            logger.info("No diversity axes defined, marking seeds as scored")
            now = datetime.now(timezone.utc)
            for seed in seeds:
                seed.scores = {}
                seed.scored_at = now
            self.db.commit()
            return True

        try:
            # Ensure we have axis embeddings cached
            if self._axis_embeddings_cache is None:
                self._axis_embeddings_cache = await self._compute_axis_embeddings(diversity_axes)

            # Compute seed embeddings
            seed_texts = [seed.text for seed in seeds]
            seed_embeddings = await self._compute_embeddings(seed_texts)

            # Score each seed against all axes
            now = datetime.now(timezone.utc)
            for seed, seed_emb in zip(seeds, seed_embeddings):
                scores = {}
                for axis in diversity_axes:
                    scores[axis["name"]] = {}
                    for value in axis["values"]:
                        axis_emb = self._axis_embeddings_cache[axis["name"]][value]
                        score = self._cosine_similarity(seed_emb, axis_emb)
                        scores[axis["name"]][value] = round(score, 4)

                seed.scores = scores
                seed.scored_at = now

            self.db.commit()
            logger.info(f"[{self.name}] Scored {len(seeds)} seeds against {len(diversity_axes)} axes")
            return True

        except Exception as e:
            logger.error(f"Scoring failed: {e}", exc_info=True)
            return False

    def _parse_diversity_spec(self) -> List[Dict]:
        """Parse diversity_spec from project config."""
        if not self.state.diversity_spec:
            return []

        axes = []
        for axis in self.state.diversity_spec:
            axis_name = axis.get("name")
            value_objs = axis.get("values", [])

            # Extract value names and build weights dict
            values = [v.get("value") for v in value_objs]
            weights = {v.get("value"): v.get("weight", 1.0) for v in value_objs}

            axes.append({
                "name": axis_name,
                "values": values,
                "weights": weights
            })

        return axes

    async def _compute_axis_embeddings(self, diversity_axes: List[Dict]) -> Dict[str, Dict[str, np.ndarray]]:
        """Compute embeddings for all axis values (cached)."""
        axis_embeddings = {}

        for axis in diversity_axes:
            embeddings = await self._compute_embeddings(axis["values"])
            axis_embeddings[axis["name"]] = {
                value: emb for value, emb in zip(axis["values"], embeddings)
            }

        logger.info(f"Cached embeddings for {len(diversity_axes)} diversity axes")
        return axis_embeddings

    async def _compute_embeddings(
            self,
            texts: List[str],
            model: str = "text-embedding-3-large"
    ) -> List[np.ndarray]:
        """Embed a list of texts using OpenAI API."""
        if not texts:
            return []

        response = await self.openai_client.embeddings.create(
            model=model,
            input=texts,
            encoding_format="float"
        )

        sorted_data = sorted(response.data, key=lambda d: d.index)
        return [np.array(item.embedding, dtype=np.float32) for item in sorted_data]

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return float(np.dot(a, b) / (norm_a * norm_b))

    def is_complete(self) -> bool:
        """Complete when all seeds are scored."""
        return not self.state.has_unscored_seeds()