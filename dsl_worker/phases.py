"""
Concrete implementations of all processing phases.
"""

import logging
from typing import List, Optional

from dsl_worker.phase_base import Phase
from dsl_worker.file_processor import FileProcessor

# Import chunking strategies when implementing FileProcessingPhase
# from dsl_worker.chunker import chunk_csv, chunk_jsonl, chunk_json_array, chunk_text_by_tokens

logger = logging.getLogger(__name__)


class FileProcessingPhase(Phase):
    """
    Phase 1: Process uploaded files - chunking and embedding.

    This phase:
    - Loads files from blob storage
    - Chunks text content using dsl_worker.chunker strategies
    - Generates embeddings for chunks
    - Stores chunks and embeddings in database
    """

    def should_run(self) -> bool:
        """Run if there are unprocessed files."""
        total = self.state.stats.get('files_total', 0)
        processed = self.state.stats.get('files_processed', 0)

        # Run if we have files and haven't processed them all
        return total > 0 and processed < total

    async def execute_batch(self, batch_size: int = 5) -> int:
        """
        Process a batch of files.

        For each file:
        1. Download from blob storage
        2. Chunk using appropriate strategy (CSV, JSONL, JSON, or text)
        3. Generate embeddings for each chunk
        4. Store chunks with embeddings in database
        5. Mark file as processed
        """
        logger.info(f"[{self.name}] Processing batch of files (max {batch_size})")

        # TODO: Get unprocessed files from database
        # Pseudocode:
        # from dsl_worker.chunker import chunk_csv, chunk_jsonl, chunk_json_array, chunk_text_by_tokens
        #
        # files = (
        #     self.db.query(ProjectFile)
        #     .filter(
        #         ProjectFile.project_id == self.state.project_id,
        #         ProjectFile.run_id == self.state.run_id,
        #         ProjectFile.processing_status == 'pending'
        #     )
        #     .limit(batch_size)
        #     .all()
        # )
        #
        # for file in files:
        #     try:
        #         # 1. Download file from blob storage
        #         blob_client = self.blob_service_client.get_blob_client(
        #             container="uploads",
        #             blob=file.blob_path
        #         )
        #         content = blob_client.download_blob().readall()
        #
        #         # 2. Chunk based on file type
        #         if file.filename.endswith('.csv'):
        #             chunks = chunk_csv(content, max_tokens=7000)
        #         elif file.filename.endswith('.jsonl'):
        #             chunks = chunk_jsonl(content, max_tokens=8000)
        #         elif file.filename.endswith('.json'):
        #             chunks = chunk_json_array(content, max_tokens=8000)
        #         else:
        #             # Default: text chunking with overlap
        #             chunks = chunk_text_by_tokens(content, chunk_size=4096, overlap=512)
        #
        #         # 3. Generate embeddings for all chunks
        #         embeddings = []
        #         for chunk_text in chunks:
        #             response = await self.openai_client.embeddings.create(
        #                 model="text-embedding-ada-002",
        #                 input=chunk_text
        #             )
        #             embeddings.append(response.data[0].embedding)
        #
        #         # 4. Store chunks with embeddings
        #         for idx, (chunk_text, embedding) in enumerate(zip(chunks, embeddings)):
        #             chunk = Chunk(
        #                 project_id=self.state.project_id,
        #                 run_id=self.state.run_id,
        #                 file_id=file.id,
        #                 text=chunk_text,
        #                 chunk_index=idx,
        #                 embedding=embedding,
        #                 metadata={'filename': file.filename}
        #             )
        #             self.db.add(chunk)
        #
        #         # 5. Mark file as processed
        #         file.processing_status = 'completed'
        #         file.processed_at = datetime.now(timezone.utc)
        #
        #         logger.info(f"Processed file {file.filename}: {len(chunks)} chunks")
        #
        #     except Exception as e:
        #         logger.error(f"Failed to process file {file.filename}: {e}")
        #         file.processing_status = 'failed'
        #         file.error = str(e)
        #
        # return len(files)

        logger.warning(f"[{self.name}] Not implemented yet - chunker integration needed")
        return 0

    def is_complete(self) -> bool:
        """Complete when all files are processed."""
        total = self.state.stats.get('files_total', 0)
        processed = self.state.stats.get('files_processed', 0)

        # Complete if no files or all files processed
        return total == 0 or processed >= total


class SeedExtractionPhase(Phase):
    """
    Phase 2: Extract recipe seeds from embedded chunks.

    Seeds are sub-chunks or chunk combinations that can be used
    directly for sample generation.
    """

    def should_run(self) -> bool:
        """
        Run if:
        - We have embedded chunks
        - Not all chunks have been converted to seeds
        """
        chunks_embedded = self.state.stats.get('chunks_embedded', 0)
        seeds_extracted = self.state.stats.get('seeds_extracted', 0)

        # Need some embedded chunks to extract seeds from
        if chunks_embedded == 0:
            return False

        # In preview mode, extract eagerly as chunks become available
        if self.state.preview_mode:
            return seeds_extracted < chunks_embedded

        # Normal mode: wait for all files to be processed first
        files_complete = self.state.stats.get('files_processed', 0) >= self.state.stats.get('files_total', 1)
        return files_complete and seeds_extracted < chunks_embedded

    async def execute_batch(self, batch_size: int = 10) -> int:
        """Extract seeds from a batch of chunks."""
        logger.info(f"[{self.name}] Extracting seeds from chunks (max {batch_size})")

        # TODO: Implement seed extraction logic
        # Pseudocode:
        # chunks = get_next_unprocessed_chunks(batch_size)
        # for chunk in chunks:
        #     seeds = await extract_seeds_from_chunk(chunk)
        #     save_seeds(seeds)
        #     mark_chunk_processed(chunk)
        # return len(chunks)

        logger.warning(f"[{self.name}] Not implemented yet")
        return 0

    def is_complete(self) -> bool:
        """Complete when all chunks have been processed into seeds."""
        chunks_embedded = self.state.stats.get('chunks_embedded', 0)
        seeds_extracted = self.state.stats.get('seeds_extracted', 0)

        return chunks_embedded > 0 and seeds_extracted >= chunks_embedded


class SeedScoringPhase(Phase):
    """
    Phase 3: Score seeds for generation distribution.

    Compares each seed to a list of categories/topics and assigns
    relevance scores.
    """

    def should_run(self) -> bool:
        """
        Run if:
        - We have extracted seeds
        - Not all seeds have been scored
        """
        seeds_extracted = self.state.stats.get('seeds_extracted', 0)
        seeds_scored = self.state.stats.get('seeds_scored', 0)

        if seeds_extracted == 0:
            return False

        # In preview mode, score eagerly
        if self.state.preview_mode:
            return seeds_scored < seeds_extracted

        # Normal mode: wait for all seeds to be extracted
        extraction_complete = self.state.stats.get('seeds_extracted', 0) >= self.state.stats.get('chunks_embedded', 1)
        return extraction_complete and seeds_scored < seeds_extracted

    async def execute_batch(self, batch_size: int = 10) -> int:
        """Score a batch of seeds."""
        logger.info(f"[{self.name}] Scoring seeds (max {batch_size})")

        # TODO: Implement seed scoring logic
        # Pseudocode:
        # seeds = get_next_unscored_seeds(batch_size)
        # categories = get_project_categories()
        #
        # for seed in seeds:
        #     scores = await llm_score_seed_against_categories(seed, categories)
        #     save_seed_scores(seed, scores)
        #
        # return len(seeds)

        logger.warning(f"[{self.name}] Not implemented yet")
        return 0

    def is_complete(self) -> bool:
        """Complete when all seeds are scored."""
        seeds_extracted = self.state.stats.get('seeds_extracted', 0)
        seeds_scored = self.state.stats.get('seeds_scored', 0)

        return seeds_extracted > 0 and seeds_scored >= seeds_extracted


class SeedAssignmentPhase(Phase):
    """
    Phase 4: Assign seeds to diversity axes.

    Based on scores, select the most relevant diversity category
    for each seed.
    """

    def should_run(self) -> bool:
        """
        Run if:
        - We have scored seeds
        - Not all seeds have been assigned
        """
        seeds_scored = self.state.stats.get('seeds_scored', 0)
        seeds_assigned = self.state.stats.get('seeds_assigned', 0)

        if seeds_scored == 0:
            return False

        # In preview mode, assign eagerly
        if self.state.preview_mode:
            return seeds_assigned < seeds_scored

        # Normal mode: wait for all seeds to be scored
        scoring_complete = self.state.stats.get('seeds_scored', 0) >= self.state.stats.get('seeds_extracted', 1)
        return scoring_complete and seeds_assigned < seeds_scored

    async def execute_batch(self, batch_size: int = 10) -> int:
        """Assign a batch of seeds to diversity axes."""
        logger.info(f"[{self.name}] Assigning seeds to diversity axes (max {batch_size})")

        # TODO: Implement seed assignment logic
        # Pseudocode:
        # seeds = get_next_unassigned_seeds(batch_size)
        #
        # for seed in seeds:
        #     best_axis = find_best_diversity_axis(seed.scores)
        #     assign_seed_to_axis(seed, best_axis)
        #
        # return len(seeds)

        logger.warning(f"[{self.name}] Not implemented yet")
        return 0

    def is_complete(self) -> bool:
        """Complete when all seeds are assigned."""
        seeds_scored = self.state.stats.get('seeds_scored', 0)
        seeds_assigned = self.state.stats.get('seeds_assigned', 0)

        return seeds_scored > 0 and seeds_assigned >= seeds_scored


class RecipeBuildingPhase(Phase):
    """
    Phase 5: Build recipes with RAG context.

    For each seed, determine relevant RAG queries and execute them
    to enrich the seed with additional context.
    """

    def should_run(self) -> bool:
        """
        Run if:
        - We have assigned seeds
        - Not all seeds have been built into recipes
        """
        seeds_assigned = self.state.stats.get('seeds_assigned', 0)
        recipes_built = self.state.stats.get('recipes_built', 0)

        if seeds_assigned == 0:
            return False

        # In preview mode, build recipes eagerly
        if self.state.preview_mode:
            return recipes_built < seeds_assigned

        # Normal mode: wait for all seeds to be assigned
        assignment_complete = self.state.stats.get('seeds_assigned', 0) >= self.state.stats.get('seeds_scored', 1)
        return assignment_complete and recipes_built < seeds_assigned

    async def execute_batch(self, batch_size: int = 10) -> int:
        """Build recipes for a batch of seeds."""
        logger.info(f"[{self.name}] Building recipes with RAG context (max {batch_size})")

        # TODO: Implement recipe building logic
        # Pseudocode:
        # seeds = get_next_seeds_without_recipes(batch_size)
        #
        # for seed in seeds:
        #     # Determine RAG queries
        #     rag_queries = await generate_rag_queries_for_seed(seed)
        #
        #     # Execute RAG queries (vector similarity search)
        #     rag_results = []
        #     for query in rag_queries:
        #         results = await vector_search(query, seed.diversity_axis)
        #         rag_results.extend(results)
        #
        #     # Build recipe
        #     recipe = create_recipe(seed, rag_results)
        #     save_recipe(recipe)
        #
        # return len(seeds)

        logger.warning(f"[{self.name}] Not implemented yet")
        return 0

    def is_complete(self) -> bool:
        """Complete when all seeds have recipes."""
        seeds_assigned = self.state.stats.get('seeds_assigned', 0)
        recipes_built = self.state.stats.get('recipes_built', 0)

        return seeds_assigned > 0 and recipes_built >= seeds_assigned


class GenerationPhase(Phase):
    """
    Phase 6: Generate samples from recipes.

    Uses recipes (seeds + RAG context) to generate actual dataset samples.
    """

    def should_run(self) -> bool:
        """
        Run if:
        - In preview mode: ANY recipes available
        - Normal mode: ALL prep phases complete
        - Haven't reached target sample count
        """
        samples_generated = self.state.stats.get('samples_generated', 0)
        target_count = self.state.num_samples

        # Check if we've hit target
        if samples_generated >= target_count:
            return False

        recipes_built = self.state.stats.get('recipes_built', 0)

        # In preview mode: generate as soon as ANY recipes available
        if self.state.preview_mode:
            return recipes_built > 0

        # Normal mode: wait for all recipes to be built
        all_files_done = self.state.stats.get('files_processed', 0) >= self.state.stats.get('files_total', 1)
        all_recipes_done = recipes_built >= self.state.stats.get('seeds_assigned', 1)

        return all_files_done and all_recipes_done

    async def execute_batch(self, batch_size: int = 10) -> int:
        """Generate a batch of samples."""
        logger.info(f"[{self.name}] Generating samples (max {batch_size})")

        # Determine how many samples we can generate
        samples_generated = self.state.stats.get('samples_generated', 0)
        target_count = self.state.num_samples
        remaining = target_count - samples_generated

        # Don't generate more than needed
        actual_batch_size = min(batch_size, remaining)

        # TODO: Implement sample generation logic
        # Pseudocode:
        # if self.state.preview_mode:
        #     # Grab any available recipes
        #     recipes = get_any_available_recipes(actual_batch_size)
        # else:
        #     # Systematic generation respecting diversity quotas
        #     recipes = get_next_batch_recipes(actual_batch_size, diversity_spec)
        #
        # samples = []
        # for recipe in recipes:
        #     sample = await generate_sample_from_recipe(
        #         recipe=recipe,
        #         prompt=self.state.generation_prompt,
        #         columns=self.state.columns,
        #         use_internet=self.state.use_internet
        #     )
        #     samples.append(sample)
        #
        # save_samples(samples)
        # update_project_generated_count(len(samples))
        #
        # return len(samples)

        logger.warning(f"[{self.name}] Not implemented yet")
        return 0

    def is_complete(self) -> bool:
        """Complete when we've generated target number of samples."""
        samples_generated = self.state.stats.get('samples_generated', 0)
        target_count = self.state.num_samples

        return samples_generated >= target_count


class ValidationPhase(Phase):
    """
    Phase 7: Validate generated samples.

    Checks sample quality and may trigger re-generation if quality
    is insufficient.
    """

    def should_run(self) -> bool:
        """
        Run if:
        - We have generated samples
        - Not all samples have been validated
        """
        samples_generated = self.state.stats.get('samples_generated', 0)
        samples_validated = self.state.stats.get('samples_validated', 0)

        # Only run if we have samples to validate
        if samples_generated == 0:
            return False

        # In preview mode, validate eagerly
        if self.state.preview_mode:
            return samples_validated < samples_generated

        # Normal mode: wait for generation to complete
        generation_complete = samples_generated >= self.state.num_samples
        return generation_complete and samples_validated < samples_generated

    async def execute_batch(self, batch_size: int = 10) -> int:
        """Validate a batch of samples."""
        logger.info(f"[{self.name}] Validating samples (max {batch_size})")

        # TODO: Implement sample validation logic
        # Pseudocode:
        # samples = get_next_unvalidated_samples(batch_size)
        #
        # for sample in samples:
        #     # Check quality (e.g., format compliance, content quality)
        #     validation_result = await validate_sample(sample)
        #
        #     if validation_result.passed:
        #         mark_sample_validated(sample)
        #     else:
        #         # Mark for regeneration
        #         mark_sample_for_regeneration(sample, validation_result.reason)
        #         # TODO: Trigger regeneration (might need to go back to Phase 6)
        #
        # return len(samples)

        logger.warning(f"[{self.name}] Not implemented yet")
        return 0

    def is_complete(self) -> bool:
        """Complete when all samples are validated."""
        samples_generated = self.state.stats.get('samples_generated', 0)
        samples_validated = self.state.stats.get('samples_validated', 0)

        # All samples validated or no samples to validate
        return samples_generated > 0 and samples_validated >= samples_generated