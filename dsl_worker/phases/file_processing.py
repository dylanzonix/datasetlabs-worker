"""
Phase: File Processing

Processes uploaded files by chunking and embedding them.
Stores results in project_rag_chunks table.

Resume logic:
- Checks which files already have chunks
- Only processes files without chunks
- Atomic: either all chunks for a file are stored, or none
"""

import logging
import uuid
from typing import List

import numpy as np
from sqlalchemy.dialects.postgresql import insert

from dsl_worker.phases.base import Phase, PhaseResult
from dsl_worker.chunker import chunk_csv, chunk_jsonl, chunk_json_array, chunk_text_by_tokens
from dsl_api.models.project_file import ProjectFile
from dsl_api.models.project_rag_chunk import ProjectRagChunk

logger = logging.getLogger(__name__)


class FileProcessingPhase(Phase):
    """
    Process files: chunk and embed.

    One execute_once() call processes ONE file completely.
    This ensures atomicity - either all chunks for a file exist, or none.
    """

    def should_run(self) -> bool:
        """Run if there are unprocessed files."""
        return self.state.has_unprocessed_files()

    async def execute_once(self) -> PhaseResult:
        """Process ONE file completely (chunk + embed)."""
        files = self.state.get_unprocessed_files(limit=1)
        if not files:
            return PhaseResult.no_work()

        file = files[0]
        logger.info(f"Processing file: {file.filename} ({file.size_bytes} bytes)")

        total_cost_usd = 0.0

        try:
            # Download from blob storage
            blob_client = self.blob_service_client.get_blob_client(
                container="uploads",
                blob=file.blob_path
            )
            content = blob_client.download_blob().readall()

            # Chunk based on file type
            chunks = self._chunk_content(content, file.filename, file.content_type)
            if not chunks:
                logger.warning(f"No chunks extracted from {file.filename}")
                # Store a single empty-ish chunk to mark file as processed
                # This prevents infinite retry on files that legitimately produce no chunks
                chunks = ["[Empty or unparseable file]"]

            logger.info(f"Created {len(chunks)} chunks from {file.filename}")

            # Compute embeddings in batches (with cost tracking)
            embeddings, embedding_cost = await self._compute_embeddings_batched(chunks)
            total_cost_usd += embedding_cost

            if len(embeddings) != len(chunks):
                logger.error(f"Embedding count mismatch: {len(embeddings)} vs {len(chunks)} chunks")
                return PhaseResult.no_work()

            # Store chunks with embeddings (bulk insert for performance)
            self._store_chunks(file, chunks, embeddings)

            logger.info(f"Successfully processed {file.filename}: {len(chunks)} chunks stored")
            return PhaseResult.work_done(cost_usd=total_cost_usd)

        except Exception as e:
            logger.error(f"File processing failed for {file.filename}: {e}", exc_info=True)
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

    async def _compute_embeddings_batched(
            self,
            chunks: List[str],
            batch_size: int = 100
    ) -> tuple[List[np.ndarray], float]:
        """
        Compute embeddings in batches to avoid API limits.

        Returns:
            Tuple of (embeddings, total_cost_usd)
        """
        embeddings = []
        total_cost_usd = 0.0

        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]

            result = await self.openai_client.create_embeddings(
                model="text-embedding-3-large",
                input=batch,
            )

            # Track cost
            total_cost_usd += result.cost.total_cost_usd

            # Sort by index to maintain order
            sorted_data = sorted(result.response.data, key=lambda d: d.index)
            batch_embeddings = [np.array(item.embedding, dtype=np.float32) for item in sorted_data]
            embeddings.extend(batch_embeddings)

            logger.debug(f"Computed embeddings for batch {i // batch_size + 1}")

        return embeddings, total_cost_usd

    def _store_chunks(
            self,
            file: ProjectFile,
            chunks: List[str],
            embeddings: List[np.ndarray]
    ) -> int:
        """
        Store chunks with embeddings in database.

        Uses bulk insert for performance. This is significantly faster than
        creating ORM objects one by one for large files.
        """
        rows = []
        for idx, (text, embedding) in enumerate(zip(chunks, embeddings)):
            rows.append({
                "id": uuid.uuid4(),
                "project_id": file.project_id,
                "file_id": file.id,
                "chunk_idx": idx,
                "text": text,
                "chunk_metadata": {
                    "chunk_idx": idx,
                    "total_chunks": len(chunks),
                    "filename": file.filename
                },
                "embedding": embedding.tolist(),
            })

        if rows:
            stmt = insert(ProjectRagChunk).values(rows)
            self.db.execute(stmt)
            self.db.commit()

        return len(rows)

    def is_complete(self) -> bool:
        """Complete when all files have been processed."""
        return not self.state.has_unprocessed_files()