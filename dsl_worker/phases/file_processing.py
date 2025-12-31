"""
Phase: File Processing

Processes uploaded files by chunking and embedding them.
Stores results in project_rag_chunks table.

Resume logic:
- Checks which files already have chunks
- Only processes files without chunks
"""

import logging
import uuid
from datetime import datetime, timezone
from typing import List, Optional

import numpy as np
from sqlalchemy.dialects.postgresql import insert

from dsl_worker.phases.base import Phase
from dsl_worker.chunker import chunk_csv, chunk_jsonl, chunk_json_array, chunk_text_by_tokens
from dsl_api.models.project_file import ProjectFile
from dsl_api.models.project_rag_chunk import ProjectRagChunk

logger = logging.getLogger(__name__)


class FileProcessingPhase(Phase):
    """
    Process files: chunk and embed.

    One execute_once() call processes ONE file completely.
    """

    def should_run(self) -> bool:
        """Run if there are unprocessed files."""
        total = self.state.stats.get('files_total', 0)
        processed = self.state.stats.get('files_processed', 0)
        return total > 0 and processed < total

    async def execute_once(self) -> bool:
        """Process ONE file completely (chunk + embed)."""
        # Get one unprocessed file
        files = self.state.get_unprocessed_files(limit=1)
        if not files:
            return False

        file = files[0]
        logger.info(f"Processing file: {file.filename} ({file.size_bytes} bytes)")

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
                return True  # Still counts as processed

            logger.info(f"Created {len(chunks)} chunks from {file.filename}")

            # Compute embeddings in batches
            embeddings = await self._compute_embeddings_batched(chunks)
            if len(embeddings) != len(chunks):
                logger.error(f"Embedding count mismatch: {len(embeddings)} vs {len(chunks)} chunks")
                return False

            # Store chunks with embeddings
            await self._store_chunks(file, chunks, embeddings)

            logger.info(f"Successfully processed {file.filename}: {len(chunks)} chunks stored")
            return True

        except Exception as e:
            logger.error(f"File processing failed for {file.filename}: {e}", exc_info=True)
            return False

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
    ) -> List[np.ndarray]:
        """Compute embeddings in batches to avoid API limits."""
        embeddings = []

        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]

            response = await self.openai_client.embeddings.create(
                model="text-embedding-3-large",
                input=batch,
                encoding_format="float"
            )

            # Sort by index to maintain order
            sorted_data = sorted(response.data, key=lambda d: d.index)
            batch_embeddings = [np.array(item.embedding, dtype=np.float32) for item in sorted_data]
            embeddings.extend(batch_embeddings)

            logger.debug(f"Computed embeddings for batch {i // batch_size + 1}")

        return embeddings

    async def _store_chunks(
            self,
            file: ProjectFile,
            chunks: List[str],
            embeddings: List[np.ndarray]
    ) -> int:
        """Store chunks with embeddings in database."""
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
        total = self.state.stats.get('files_total', 0)
        processed = self.state.stats.get('files_processed', 0)
        return total == 0 or processed >= total