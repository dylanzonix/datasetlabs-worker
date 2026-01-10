"""
Phase: File Processing (Parallel)

Processes uploaded files by chunking and embedding them.
Stores results in project_rag_chunks table.

VERSION SEMANTICS:
- Files come from the version's files_snapshot (immutable at version creation)
- Chunks are NOT scoped to version - they can be reused across versions
- The state tracks which snapshot files have chunks

Optimizations:
- Parallel embedding computation (multiple concurrent API calls)
- Optional multi-file processing in single execute_once()

Resume logic:
- Checks which files from snapshot already have chunks
- Only processes files without chunks
- Atomic: either all chunks for a file are stored, or none
"""

import asyncio
import logging
import uuid
from typing import List, Tuple, Dict, Any
from uuid import UUID

import numpy as np
from sqlalchemy.dialects.postgresql import insert

from dsl_worker.phases.base import Phase, PhaseResult
from dsl_worker.chunker import chunk_csv, chunk_jsonl, chunk_json_array, chunk_text_by_tokens
from dsl_api.models.project_rag_chunk import ProjectRagChunk

logger = logging.getLogger(__name__)


class FileProcessingPhase(Phase):
    """
    Process files: chunk and embed.

    Works with file info dicts from the version's files_snapshot.

    Supports:
    - Parallel embedding computation for large files
    - Optional multi-file processing per execute_once()

    One execute_once() call processes ONE file completely (by default).
    This ensures atomicity - either all chunks for a file exist, or none.
    """

    def __init__(
        self,
        *args,
        max_concurrent_embeddings: int = 10,
        embedding_batch_size: int = 100,
        files_per_execution: int = 1,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.max_concurrent_embeddings = max_concurrent_embeddings
        self.embedding_batch_size = embedding_batch_size
        self.files_per_execution = files_per_execution

    def should_run(self) -> bool:
        """Run if there are unprocessed files in the version snapshot."""
        return self.state.has_unprocessed_files()

    async def execute_once(self) -> PhaseResult:
        """Process file(s) completely (chunk + embed)."""
        file_infos = self.state.get_unprocessed_files(limit=self.files_per_execution)
        if not file_infos:
            return PhaseResult.no_work()

        if len(file_infos) == 1:
            return await self._process_single_file(file_infos[0])

        # Process multiple files concurrently
        tasks = [self._process_single_file(f) for f in file_infos]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        total_cost = 0.0
        any_work = False

        for file_info, result in zip(file_infos, results):
            if isinstance(result, Exception):
                logger.error(f"File processing failed for {file_info.get('filename', 'unknown')}: {result}")
            elif result.did_work:
                total_cost += result.cost_usd
                any_work = True

        if any_work:
            return PhaseResult.work_done(cost_usd=total_cost)
        return PhaseResult.no_work()

    async def _process_single_file(self, file_info: Dict[str, Any]) -> PhaseResult:
        """
        Process ONE file completely (chunk + embed).

        Args:
            file_info: Dict with keys: id, filename, blob_path, content_type, size_bytes
        """
        filename = file_info.get("filename", "unknown")
        file_id = UUID(file_info["id"])
        blob_path = file_info["blob_path"]
        content_type = file_info.get("content_type", "text/plain")
        size_bytes = file_info.get("size_bytes", 0)

        logger.info(f"Processing file: {filename} ({size_bytes} bytes)")

        total_cost_usd = 0.0

        try:
            # Download from blob storage
            blob_client = self.blob_service_client.get_blob_client(
                container="uploads",
                blob=blob_path
            )
            content = blob_client.download_blob().readall()

            # Chunk based on file type
            chunks = self._chunk_content(content, filename, content_type)
            if not chunks:
                logger.warning(f"No chunks extracted from {filename}")
                # Store a single empty-ish chunk to mark file as processed
                # This prevents infinite retry on files that legitimately produce no chunks
                chunks = ["[Empty or unparseable file]"]

            logger.info(f"Created {len(chunks)} chunks from {filename}")

            # Compute embeddings in parallel batches (with cost tracking)
            embeddings, embedding_cost = await self._compute_embeddings_parallel(chunks)
            total_cost_usd += embedding_cost

            if len(embeddings) != len(chunks):
                logger.error(f"Embedding count mismatch: {len(embeddings)} vs {len(chunks)} chunks")
                return PhaseResult.no_work()

            # Store chunks with embeddings (bulk insert for performance)
            self._store_chunks(file_id, filename, chunks, embeddings)

            logger.info(f"Successfully processed {filename}: {len(chunks)} chunks stored")
            return PhaseResult.work_done(cost_usd=total_cost_usd)

        except Exception as e:
            logger.error(f"File processing failed for {filename}: {e}", exc_info=True)
            return PhaseResult.no_work()

    def _chunk_content(self, content: bytes, filename: str, content_type: str) -> List[str]:
        """Chunk content based on file type."""
        filename_lower = filename.lower()

        if filename_lower.endswith('.csv') or 'csv' in content_type:
            return chunk_csv(content, max_tokens=4096)
        elif filename_lower.endswith('.jsonl'):
            return chunk_jsonl(content, max_tokens=4096)
        elif filename_lower.endswith('.json'):
            return chunk_json_array(content, max_tokens=4096)
        else:
            # Plain text or unknown - use token-based chunking
            return chunk_text_by_tokens(content, chunk_size=4096, overlap=512)

    async def _compute_embeddings_parallel(
        self,
        chunks: List[str],
    ) -> Tuple[List[np.ndarray], float]:
        """
        Compute embeddings in parallel batches.

        Args:
            chunks: Text chunks to embed

        Returns:
            Tuple of (embeddings, total_cost_usd)
        """
        if not chunks:
            return [], 0.0

        semaphore = asyncio.Semaphore(self.max_concurrent_embeddings)
        batch_size = self.embedding_batch_size

        # Split into batches with indices to maintain order
        batches = [
            (i, chunks[i:i + batch_size])
            for i in range(0, len(chunks), batch_size)
        ]

        async def process_batch(
            batch_idx: int,
            batch: List[str]
        ) -> Tuple[int, List[np.ndarray], float]:
            async with semaphore:
                result = await self.openai_client.create_embeddings(
                    model="text-embedding-3-small",
                    input=batch,
                )
                sorted_data = sorted(result.response.data, key=lambda d: d.index)
                embeddings = [
                    np.array(item.embedding, dtype=np.float32)
                    for item in sorted_data
                ]
                return batch_idx, embeddings, result.cost.total_cost_usd

        # Run all batches concurrently
        results = await asyncio.gather(*[
            process_batch(idx, batch) for idx, batch in batches
        ])

        # Sort by batch index and flatten
        results = sorted(results, key=lambda x: x[0])

        all_embeddings = []
        total_cost = 0.0
        for _, embeddings, cost in results:
            all_embeddings.extend(embeddings)
            total_cost += cost

        logger.info(
            f"Computed {len(all_embeddings)} embeddings in {len(batches)} batches "
            f"({self.max_concurrent_embeddings} concurrent)"
        )

        return all_embeddings, total_cost

    def _store_chunks(
        self,
        file_id: UUID,
        filename: str,
        chunks: List[str],
        embeddings: List[np.ndarray]
    ) -> int:
        """
        Store chunks with embeddings in database.

        Uses bulk insert for performance. This is significantly faster than
        creating ORM objects one by one for large files.

        NOTE: Chunks are stored at the project level, NOT version level.
        This allows chunks to be reused across versions if the same files are used.
        """
        rows = []
        for idx, (text, embedding) in enumerate(zip(chunks, embeddings)):
            rows.append({
                "id": uuid.uuid4(),
                "project_id": self.state.project_id,
                "file_id": file_id,
                "chunk_idx": idx,
                "text": text,
                "chunk_metadata": {
                    "chunk_idx": idx,
                    "total_chunks": len(chunks),
                    "filename": filename
                },
                "embedding": embedding.tolist(),
            })

        if rows:
            stmt = insert(ProjectRagChunk).values(rows)
            self.db.execute(stmt)
            self.db.commit()

        return len(rows)

    def is_complete(self) -> bool:
        """Complete when all files from snapshot have been processed."""
        return not self.state.has_unprocessed_files()

    def get_status(self) -> "PhaseStatus":
        """Get current progress of file processing."""
        from dsl_worker.phases.base import PhaseStatus

        if self.is_complete():
            status = "complete"
        elif self.should_run():
            status = "active"
        else:
            status = "pending"

        return PhaseStatus(
            phase_name=self.name,
            status=status,
            progress=f"{self.state.files_processed}/{self.state.files_total} files"
        )