"""
Phase: Seed Extraction

Extracts seeds from embedded chunks using LLM-based extraction.
Stores results in project_seeds table.

Resume logic:
- Checks which chunks already have seeds
- Only processes chunks without seeds
"""

import asyncio
import logging
import json
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
        self.batch_size = 20  # Process 20 chunks per iteration
        try:
            self._encoding = tiktoken.get_encoding("cl100k_base")
        except Exception:
            self._encoding = None

    def should_run(self) -> bool:
        """
        Run if there are chunks without seeds.

        Runs eagerly - doesn't wait for file processing to complete.
        """
        # No chunks = nothing to extract from
        if self.state.chunks_total == 0:
            return False

        # Check if there's actual work to do
        return self.state.has_chunks_without_seeds()

    async def execute_once(self) -> PhaseResult:
        """Extract seeds from a batch of chunks."""
        chunks = self.state.get_chunks_without_seeds(limit=self.batch_size)
        if not chunks:
            return PhaseResult.no_work()

        logger.info(f"[{self.name}] Extracting seeds from {len(chunks)} chunks")

        # Fire all requests concurrently
        results = await asyncio.gather(
            *[self._extract_seeds_from_chunk(chunk) for chunk in chunks],
            return_exceptions=True,
        )

        extracted_count = 0
        total_cost_usd = 0.0

        for chunk, result in zip(chunks, results):
            if isinstance(result, Exception):
                logger.error(f"Failed to extract seeds from chunk {chunk.id}: {result}")
                continue

            seed_texts, cost_usd = result
            total_cost_usd += cost_usd

            for seed_text in seed_texts:
                seed = ProjectSeed(
                    id=uuid.uuid4(),
                    project_id=self.state.project_id,
                    run_id=self.state.run_id,
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
            logger.debug(f"Extracted {len(seed_texts)} seeds from chunk {chunk.id}")

        self.db.commit()
        logger.info(
            f"[{self.name}] Extracted {extracted_count} seeds from {len(chunks)} chunks"
        )
        return PhaseResult.work_done(cost_usd=total_cost_usd)

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
                model="o1",
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
        return len(self._encoding.encode(text))

    def is_complete(self) -> bool:
        """Complete when all chunks have seeds."""
        return not self.state.has_chunks_without_seeds()
