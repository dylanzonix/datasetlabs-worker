"""
Phase: Seed Extraction

Extracts seeds from embedded chunks using LLM-based extraction.
Stores results in project_seeds table.

VERSION SEMANTICS:
- Seeds are scoped to a specific version_id
- When a new version is created, seeds are extracted fresh
- This ensures config changes (prompt, columns) result in fresh seeds

Resume logic:
- Checks which chunks already have seeds FOR THIS VERSION
- Only processes chunks without seeds

Passthrough mode (EXTRACTION_PASSTHROUGH=true):
- Skips LLM extraction entirely
- Uses whole chunk as seed (with size limits)
- Much faster and cheaper for testing
"""

import asyncio
import logging
import json
import os
import uuid
from typing import List, Tuple
from datetime import datetime, timezone

import tiktoken
from pydantic import BaseModel, ValidationError

from dsl_worker.phases.base import Phase, PhaseResult
from dsl_worker.chunker import chunk_text_by_tokens
from dsl_api.models.project_rag_chunk import ProjectRagChunk
from dsl_api.models.project_seed import ProjectSeed

logger = logging.getLogger(__name__)

# Max tokens for a seed (fallback chunking threshold)
MAX_SEED_TOKENS = 4096

# Timeout for a single extraction request (seconds)
EXTRACTION_TIMEOUT = 120.0

# Passthrough mode - skip LLM extraction, use chunks directly as seeds
EXTRACTION_PASSTHROUGH = os.getenv("EXTRACTION_PASSTHROUGH", "").lower() in ("true", "1", "yes")


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

    Seeds are scoped to the current version_id - a new version means
    re-extracting seeds from scratch, even if the same chunks exist.

    Uses a semaphore-based pool for continuous processing - when one request
    finishes, another starts immediately without waiting for the whole batch.
    """

    def __init__(self, *args, parallel_extractions: int = 10, **kwargs):
        super().__init__(*args, **kwargs)
        self.parallel_extractions = parallel_extractions
        self._semaphore = asyncio.Semaphore(parallel_extractions)

        # Larger batches in passthrough mode since there's no LLM bottleneck
        self.batch_size = 500 if EXTRACTION_PASSTHROUGH else 50

        try:
            self._encoding = tiktoken.get_encoding("cl100k_base")
        except Exception:
            self._encoding = None

        if EXTRACTION_PASSTHROUGH:
            logger.info(f"[{self.name}] ⚡ Passthrough mode enabled - skipping LLM extraction")

    def should_run(self) -> bool:
        """
        Run if there are chunks without seeds for this version.

        Runs eagerly - doesn't wait for file processing to complete.
        """
        # No chunks = nothing to extract from
        if self.state.chunks_total == 0:
            return False

        # Check if there's actual work to do for this version
        return self.state.has_chunks_without_seeds()

    async def execute_once(self) -> PhaseResult:
        """Extract seeds from a batch of chunks using concurrent pool."""
        chunks = self.state.get_chunks_without_seeds(limit=self.batch_size)
        if not chunks:
            return PhaseResult.no_work()

        # Passthrough mode - skip LLM, use chunks directly as seeds
        if EXTRACTION_PASSTHROUGH:
            return self._passthrough_extract(chunks)

        logger.info(f"[{self.name}] Extracting seeds from {len(chunks)} chunks (max {self.parallel_extractions} concurrent)")

        # Create tasks - each will acquire semaphore independently
        tasks = [
            self._extract_with_semaphore(chunk)
            for chunk in chunks
        ]

        # Run all tasks - semaphore ensures only N run concurrently
        # When one finishes, another starts immediately
        results = await asyncio.gather(*tasks, return_exceptions=True)

        extracted_count = 0
        total_cost_usd = 0.0

        for chunk, result in zip(chunks, results):
            if isinstance(result, Exception):
                logger.error(f"Failed to extract seeds from chunk {chunk.id}: {result}")
                # On exception, use fallback
                seed_texts = self._fallback_chunk(chunk.text)
                cost_usd = 0.0
            else:
                seed_texts, cost_usd = result

            total_cost_usd += cost_usd

            for seed_text in seed_texts:
                seed = ProjectSeed(
                    id=uuid.uuid4(),
                    project_id=self.state.project_id,
                    version_id=self.state.version_id,  # Scoped to version
                    chunk_id=chunk.id,
                    file_id=chunk.file_id,
                    text=seed_text,
                    extraction_metadata={
                        "extraction_method": "llm",
                        "chunk_idx": chunk.chunk_idx,
                        "extracted_at": datetime.now(timezone.utc).isoformat(),
                    },
                )
                self.db.add(seed)

            extracted_count += len(seed_texts)

        self.db.commit()
        logger.info(
            f"[{self.name}] Extracted {extracted_count} seeds from {len(chunks)} chunks"
        )
        return PhaseResult.work_done(cost_usd=total_cost_usd)

    def _passthrough_extract(self, chunks: List[ProjectRagChunk]) -> PhaseResult:
        """
        Passthrough mode: use chunks directly as seeds without LLM.

        Much faster and cheaper - useful for testing.
        Still respects MAX_SEED_TOKENS limit (splits if needed).
        """
        logger.info(f"[{self.name}] Passthrough mode: converting {len(chunks)} chunks to seeds directly")

        extracted_count = 0

        for chunk in chunks:
            # Apply size limits (split if too large)
            seed_texts = self._ensure_size_limit(chunk.text)

            for seed_text in seed_texts:
                seed = ProjectSeed(
                    id=uuid.uuid4(),
                    project_id=self.state.project_id,
                    version_id=self.state.version_id,  # Scoped to version
                    chunk_id=chunk.id,
                    file_id=chunk.file_id,
                    text=seed_text,
                    extraction_metadata={
                        "extraction_method": "passthrough",
                        "chunk_idx": chunk.chunk_idx,
                        "extracted_at": datetime.now(timezone.utc).isoformat(),
                    },
                )
                self.db.add(seed)

            extracted_count += len(seed_texts)

        self.db.commit()
        logger.info(f"[{self.name}] Passthrough: created {extracted_count} seeds from {len(chunks)} chunks")

        return PhaseResult.work_done(cost_usd=0.0)

    async def _extract_with_semaphore(
        self, chunk: ProjectRagChunk
    ) -> Tuple[List[str], float]:
        """
        Extract seeds with semaphore-controlled concurrency and timeout.

        On timeout, falls back to using the whole chunk as seed(s).
        """
        async with self._semaphore:
            try:
                return await asyncio.wait_for(
                    self._extract_seeds_from_chunk(chunk),
                    timeout=EXTRACTION_TIMEOUT
                )
            except asyncio.TimeoutError:
                logger.warning(
                    f"Seed extraction timed out for chunk {chunk.id} after {EXTRACTION_TIMEOUT}s, using fallback"
                )
                return self._fallback_chunk(chunk.text), 0.0

    async def _extract_seeds_from_chunk(
        self, chunk: ProjectRagChunk
    ) -> Tuple[List[str], float]:
        """
        Use LLM to extract seeds from a chunk.

        Returns:
            Tuple of (seed_texts, cost_usd)
        """
        try:
            response, cost = await self.openai_client.responses_create(
                model="gpt-5-nano",
                input=[],
                prompt={
                    "id": "pmpt_69508e29f514819693d017e0848e223406fd27a87843182b",
                    "version": "5",
                    "variables": {
                        "row_instructions": self.state.generation_prompt,
                        "column_schema": self._format_column_schema(),
                        "source_chunk": chunk.text,
                    },
                },
                reasoning={"summary": "auto"},
                store=True,
            )

            raw_output = response.output_text
            seeds = self._process_extraction_response(raw_output, chunk.text)
            return seeds, cost.total_cost_usd

        except Exception as e:
            logger.error(f"LLM extraction failed: {e}")
            return self._fallback_chunk(chunk.text), 0.0

    def _format_column_schema(self) -> str:
        """Format the column schema for the LLM prompt."""
        if not self.state.columns:
            return "No specific schema defined"

        lines = []
        for col in self.state.columns:
            col_name = col.get("name", "unknown")
            col_type = col.get("type", "string")
            col_desc = col.get("description", "")
            lines.append(f"{col_name} ({col_type}): {col_desc}")

        return "\n".join(lines)

    def _process_extraction_response(
        self, raw_response: str, chunk_text: str
    ) -> List[str]:
        """Validate LLM response and extract seed texts from chunk."""
        try:
            data = json.loads(raw_response)
        except json.JSONDecodeError as e:
            logger.error(f"JSON parse error in seed extraction: {e}")
            return self._fallback_chunk(chunk_text)

        try:
            response = ExtractionResponse(**data)
        except ValidationError as e:
            logger.error(f"Validation error in seed extraction: {e}")
            return self._fallback_chunk(chunk_text)

        if not response.seeds:
            # No seeds found, use entire chunk (with fallback splitting)
            return self._fallback_chunk(chunk_text)

        extracted_seeds = []
        for marker in response.seeds:
            seed_text = self._extract_seed_text(chunk_text, marker)
            if seed_text:
                # Apply size limit to extracted seeds too
                extracted_seeds.extend(self._ensure_size_limit(seed_text))
            else:
                logger.warning(
                    f"Could not locate seed with start='{marker.start[:30]}...'"
                )

        return extracted_seeds if extracted_seeds else self._fallback_chunk(chunk_text)

    def _extract_seed_text(self, chunk: str, marker: SeedMarker) -> str | None:
        """Find the continuous span between start and end markers."""
        start_idx = chunk.find(marker.start)
        if start_idx == -1:
            return None

        end_idx = chunk.find(marker.end, start_idx)
        if end_idx == -1:
            return None

        return chunk[start_idx : end_idx + len(marker.end)]

    def _fallback_chunk(self, text: str) -> List[str]:
        """
        Fallback: return chunk as seed(s), splitting if too large.

        If the text exceeds MAX_SEED_TOKENS, split it using token-based chunking.
        """
        return self._ensure_size_limit(text)

    def _ensure_size_limit(self, text: str) -> List[str]:
        """
        Ensure text fits within MAX_SEED_TOKENS, splitting if necessary.
        """
        token_count = self._count_tokens(text)

        if token_count <= MAX_SEED_TOKENS:
            return [text]

        # Text is too large, split it
        logger.warning(
            f"Seed text exceeds {MAX_SEED_TOKENS} tokens ({token_count} tokens), "
            f"splitting by tokens"
        )
        return chunk_text_by_tokens(
            text.encode("utf-8"), chunk_size=MAX_SEED_TOKENS, overlap=200
        )

    def _count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        if self._encoding is None:
            # Fallback: rough estimate (4 chars per token)
            return len(text) // 4

        try:
            return len(self._encoding.encode(text))
        except Exception:
            return len(text) // 4

    def is_complete(self) -> bool:
        """Complete when all chunks have seeds for this version."""
        return not self.state.has_chunks_without_seeds()

    def get_status(self) -> "PhaseStatus":
        """Get current progress of seed extraction."""
        from dsl_worker.phases.base import PhaseStatus

        if self.is_complete():
            status = "complete"
        elif self.should_run():
            status = "active"
        else:
            status = "pending"

        progress = f"{self.state.seeds_extracted} seeds from {self.state.chunks_total} chunks"
        if EXTRACTION_PASSTHROUGH:
            progress += " [passthrough]"

        return PhaseStatus(
            phase_name=self.name,
            status=status,
            progress=progress
        )