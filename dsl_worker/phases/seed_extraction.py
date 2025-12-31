"""
Phase: Seed Extraction

Extracts seeds from embedded chunks using LLM-based extraction.
Stores results in project_seeds table.

Resume logic:
- Checks which chunks already have seeds
- Only processes chunks without seeds
"""

import logging
import json
import uuid
from typing import List
from datetime import datetime, timezone
from pydantic import BaseModel, ValidationError

from dsl_worker.phases.base import Phase
from dsl_api.models.project_rag_chunk import ProjectRagChunk
from dsl_api.models.project_seed import ProjectSeed

logger = logging.getLogger(__name__)


class SeedMarker(BaseModel):
    """Marker for seed boundaries in chunk text."""
    start: str
    end: str


class ExtractionResponse(BaseModel):
    """Expected response format from LLM."""
    seeds: List[SeedMarker]


class SeedExtractionPhase(Phase):
    """
    Extract seeds from embedded chunks.

    Seeds are sub-chunks that can be used directly for sample generation.
    Uses LLM to identify seed boundaries within chunks.

    One execute_once() processes a small batch of chunks.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.batch_size = 5  # Process 5 chunks per iteration

    def should_run(self) -> bool:
        """Run if there are chunks without seeds."""
        chunks_total = self.state.stats.get('chunks_total', 0)
        seeds_extracted = self.state.stats.get('seeds_extracted', 0)

        if chunks_total == 0:
            return False

        # In preview mode, extract eagerly
        if self.state.preview_mode:
            return seeds_extracted < chunks_total

        # Normal mode: wait for all files to be processed first
        files_total = self.state.stats.get('files_total', 0)
        files_processed = self.state.stats.get('files_processed', 0)
        files_complete = files_total == 0 or files_processed >= files_total

        return files_complete and seeds_extracted < chunks_total

    async def execute_once(self) -> bool:
        """Extract seeds from a batch of chunks."""
        chunks = self.state.get_chunks_without_seeds(limit=self.batch_size)
        if not chunks:
            return False

        logger.info(f"[{self.name}] Extracting seeds from {len(chunks)} chunks")

        extracted_count = 0
        for chunk in chunks:
            try:
                seed_texts = await self._extract_seeds_from_chunk(chunk)

                for seed_text in seed_texts:
                    seed = ProjectSeed(
                        id=uuid.uuid4(),
                        project_id=self.state.project_id,
                        run_id=self.state.run_id,
                        chunk_id=chunk.id,
                        file_id=chunk.file_id,
                        text=seed_text,
                        extraction_metadata={
                            'extraction_method': 'llm',
                            'chunk_idx': chunk.chunk_idx,
                            'extracted_at': datetime.now(timezone.utc).isoformat()
                        }
                    )
                    self.db.add(seed)

                extracted_count += len(seed_texts)
                logger.debug(f"Extracted {len(seed_texts)} seeds from chunk {chunk.id}")

            except Exception as e:
                logger.error(f"Failed to extract seeds from chunk {chunk.id}: {e}")
                continue

        self.db.commit()
        logger.info(f"[{self.name}] Extracted {extracted_count} seeds from {len(chunks)} chunks")
        return True

    async def _extract_seeds_from_chunk(self, chunk: ProjectRagChunk) -> List[str]:
        """Use LLM to extract seeds from a chunk."""
        system_prompt = """You are a data extraction assistant. Your job is to identify discrete, self-contained pieces of content that can serve as seeds for generating synthetic data.

Given a chunk of source data, identify the boundaries of each seed within the text. Each seed should be:
- Self-contained and meaningful on its own
- Suitable as a starting point for generating a complete data sample
- Clearly demarcated by a start and end phrase from the original text

Return a JSON object with this structure:
{
    "seeds": [
        {"start": "first few words of seed 1", "end": "last few words of seed 1"},
        {"start": "first few words of seed 2", "end": "last few words of seed 2"}
    ]
}

If the entire chunk is one cohesive unit, return it as a single seed.
If no valid seeds can be extracted, return {"seeds": []}.

Important: The start and end strings must be EXACT substrings from the chunk text."""

        user_prompt = f"""Extract seeds from this chunk:

<chunk>
{chunk.text}
</chunk>

Row generation instructions (context for what makes a good seed):
{self.state.generation_prompt}

Column schema:
{self._format_column_schema()}

Return JSON with seed boundaries."""

        try:
            response = await self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.3
            )

            raw_output = response.choices[0].message.content
            return self._process_extraction_response(raw_output, chunk.text)

        except Exception as e:
            logger.error(f"LLM extraction failed: {e}")
            # Fallback: treat entire chunk as one seed
            return [chunk.text]

    def _format_column_schema(self) -> str:
        """Format the column schema for the LLM prompt."""
        if not self.state.columns:
            return "No specific schema defined"

        lines = []
        for col in self.state.columns:
            col_name = col.get('name', 'unknown')
            col_type = col.get('type', 'string')
            col_desc = col.get('description', '')
            lines.append(f"{col_name} ({col_type}): {col_desc}")

        return "\n".join(lines)

    def _process_extraction_response(self, raw_response: str, chunk_text: str) -> List[str]:
        """Validate LLM response and extract seed texts from chunk."""
        try:
            data = json.loads(raw_response)
        except json.JSONDecodeError as e:
            logger.error(f"JSON parse error in seed extraction: {e}")
            return [chunk_text]  # Fallback

        try:
            response = ExtractionResponse(**data)
        except ValidationError as e:
            logger.error(f"Validation error in seed extraction: {e}")
            return [chunk_text]  # Fallback

        if not response.seeds:
            # No seeds found, use entire chunk
            return [chunk_text]

        extracted_seeds = []
        for marker in response.seeds:
            seed_text = self._extract_seed_text(chunk_text, marker)
            if seed_text:
                extracted_seeds.append(seed_text)
            else:
                logger.warning(f"Could not locate seed with start='{marker.start[:30]}...'")

        return extracted_seeds if extracted_seeds else [chunk_text]

    def _extract_seed_text(self, chunk: str, marker: SeedMarker) -> str | None:
        """Find the continuous span between start and end markers."""
        start_idx = chunk.find(marker.start)
        if start_idx == -1:
            return None

        end_idx = chunk.find(marker.end, start_idx)
        if end_idx == -1:
            return None

        return chunk[start_idx:end_idx + len(marker.end)]

    def is_complete(self) -> bool:
        """Complete when all chunks have been processed into seeds."""
        # We check if there are any chunks without seeds
        chunks_without_seeds = self.state.get_chunks_without_seeds(limit=1)
        return len(chunks_without_seeds) == 0