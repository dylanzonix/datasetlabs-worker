import logging
from typing import List, Optional
from uuid import UUID
import uuid
from pathlib import Path

from azure.storage.blob import BlobServiceClient
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session
import numpy as np

from dsl_api.models import RagChunk

logger = logging.getLogger(__name__)


class FileProcessor:
    """
    Handles file processing operations for a project:
    - Fetching files from database
    - Downloading from blob storage
    - Chunking content
    - Computing embeddings
    - Storing chunks in vector database
    """

    def __init__(
            self,
            blob_service_client: BlobServiceClient,
            openai_client,
            db_session: Session
    ):
        self.blob_service_client = blob_service_client
        self.openai_client = openai_client
        self.db = db_session

    async def get_project_files(self, project_id: UUID) -> List:
        """
        Query project_files table for all files belonging to this project.

        Returns only files with status='uploaded' and deleted_at=None.
        """
        from dsl_api.models.project_file import ProjectFile

        files = self.db.query(ProjectFile).filter(
            ProjectFile.project_id == project_id,
            ProjectFile.status == "uploaded",
            ProjectFile.deleted_at == None
        ).all()

        logger.info(f"Found {len(files)} uploaded files for project {project_id}")
        return files

    async def download_file(self, project_file) -> Optional[bytes]:
        """
        Download a file from Azure Blob Storage.

        Args:
            project_file: ProjectFile ORM object with blob_path

        Returns:
            File content as bytes, or None if download failed

        Note:
            blob_path from DB is relative (e.g., "project_id/filename")
            We prepend "uploads/" prefix to get actual blob location
        """
        try:
            from dsl_worker.config import settings

            blob_path = project_file.blob_path
            logger.info(f"Downloading file: {project_file.filename} from {blob_path}")

            # Container is from settings, prepend "uploads/" to blob path
            container_name = settings.azure_storage_container_name
            blob_name = f"uploads/{blob_path}"

            logger.debug(f"Container: {container_name}, Blob: {blob_name}")

            # Get blob client and download
            blob_client = self.blob_service_client.get_blob_client(
                container=container_name,
                blob=blob_path
            )

            # Download to memory
            download_stream = blob_client.download_blob()
            content = download_stream.readall()

            actual_size = len(content)
            logger.info(
                f"Downloaded {project_file.filename}: "
                f"{actual_size:,} bytes (expected: {project_file.size_bytes:,})"
            )

            # Verify size matches
            if actual_size != project_file.size_bytes:
                logger.warning(
                    f"Size mismatch for {project_file.filename}: "
                    f"downloaded {actual_size:,} != expected {project_file.size_bytes:,}"
                )

            return content

        except Exception as e:
            logger.error(f"Failed to download {project_file.filename}: {e}", exc_info=True)
            return None

    async def chunk_content(
            self,
            content: bytes,
            content_type: str,
            filename: str
    ) -> List[str]:
        """
        Chunk file content based on file type.

        All chunking respects token limits for embeddings:
        - Target: ~2000 tokens per chunk (safe for text-embedding-3-large)
        - Max: 8192 tokens (API hard limit)

        Args:
            content: Raw file content
            content_type: MIME type
            filename: Original filename

        Returns:
            List of text chunks
        """
        from dsl_worker.chunker import (
            chunk_csv,
            chunk_jsonl,
            chunk_json_array,
            chunk_text_by_tokens
        )

        extension = Path(filename).suffix.lower()

        try:
            if extension == '.csv':
                # One row per chunk (semantically coherent for embeddings)
                # Each chunk = header + one data row
                chunks = chunk_csv(
                    content,
                    max_tokens=7000,  # Conservative (API limit is 8192)
                    encoding_name="cl100k_base"
                )

            elif extension == '.jsonl':
                chunks = chunk_jsonl(content)

            elif extension == '.json':
                # Try JSON array first, fall back to JSONL
                chunks = chunk_json_array(content)
                if not chunks:
                    chunks = chunk_jsonl(content)

            else:
                # Everything else: token-based chunking with overlap
                chunks = chunk_text_by_tokens(
                    content,
                    chunk_size=2000,  # Increased from 512
                    overlap=100  # Increased from 50
                )

            logger.info(f"Chunked {filename} into {len(chunks)} chunks")
            return chunks

        except Exception as e:
            logger.error(f"Error chunking {filename}: {e}", exc_info=True)
            # Fallback: entire content as single chunk
            try:
                text = content.decode('utf-8', errors='ignore')
                return [text] if text.strip() else []
            except:
                logger.error(f"Failed to decode {filename}")
                return []

    async def get_existing_chunks(self, project_id: UUID, file_id: UUID) -> set:
        """
        Query existing chunk indices for a file to support resume.

        Returns:
            Set of chunk_idx values that already exist in database
        """
        from dsl_api.models.rag_chunk import RagChunk

        existing = self.db.query(RagChunk.chunk_idx).filter(
            RagChunk.project_id == project_id,
            RagChunk.file_id == file_id
        ).all()

        indices = {row.chunk_idx for row in existing}
        logger.info(f"Found {len(indices)} existing chunks for file {file_id}")
        return indices

    async def compute_embeddings(self, chunks: List[str]) -> List[np.ndarray]:
        """
        Compute embeddings for text chunks using OpenAI API.

        Uses text-embedding-3-large (3072 dimensions).
        Respects API limits:
        - Each chunk: ≤ 8192 tokens
        - Total per request: ≤ 300k tokens

        Args:
            chunks: List of text chunks

        Returns:
            List of numpy arrays (embedding vectors, one per chunk)
        """
        if not chunks:
            return []

        embeddings = []

        try:
            logger.info(f"Computing embeddings for {len(chunks)} chunks")

            # Conservative batch size to stay under 300k token limit
            # Assuming ~2000 tokens per chunk on average:
            # 100 chunks * 2000 = 200k tokens (safe margin)
            batch_size = 25

            for i in range(0, len(chunks), batch_size):
                batch = chunks[i:i + batch_size]

                response = await self.openai_client.embeddings.create(
                    model="text-embedding-3-large",
                    input=batch,
                    encoding_format="float"  # Explicit float format
                )

                # Sort by index to maintain order (API doesn't guarantee ordering)
                sorted_data = sorted(response.data, key=lambda d: d.index)

                # Convert to numpy arrays for pgvector
                batch_embeddings = [
                    np.asarray(item.embedding, dtype=np.float32)
                    for item in sorted_data
                ]
                embeddings.extend(batch_embeddings)

                logger.info(f"  Processed batch {i // batch_size + 1}: {len(batch)} chunks")

            logger.info(f"Successfully computed {len(embeddings)} embeddings")

        except Exception as e:
            logger.error(f"Failed to compute embeddings: {e}", exc_info=True)
            return []

        return embeddings

    async def store_chunks(
            self,
            project_id: UUID,
            file_id: UUID,
            chunks: List[str],
            embeddings: List[np.ndarray],
            existing_indices: set = None
    ) -> int:
        if existing_indices is None:
            existing_indices = set()

        rows = []
        for idx, (text, embedding) in enumerate(zip(chunks, embeddings)):
            if idx in existing_indices:
                continue

            rows.append({
                "id": uuid.uuid4(),
                "project_id": project_id,
                "file_id": file_id,
                "chunk_idx": idx,
                "text": text,
                "chunk_metadata": {"chunk_idx": idx, "total_chunks": len(chunks)},
                "embedding": embedding.tolist(),
            })

        if rows:
            stmt = insert(RagChunk).values(rows)
            self.db.execute(stmt)
            self.db.commit()

        return len(rows)

    async def process_file(self, project_file) -> bool:
        """
        Process a single file: download, chunk, compute embeddings, store.

        Supports resume: skips chunks that already exist in database.

        Returns:
            True if processing succeeded, False otherwise
        """
        logger.info(f"Processing file: {project_file.filename}")

        try:
            # Step 1: Check for existing chunks (resume support)
            existing_indices = await self.get_existing_chunks(
                project_file.project_id,
                project_file.id
            )

            # Step 2: Download file
            content = await self.download_file(project_file)
            if content is None:
                logger.error(f"Failed to download {project_file.filename}")
                return False

            # Step 3: Chunk the content
            chunks = await self.chunk_content(
                content,
                project_file.content_type,
                project_file.filename
            )
            if not chunks:
                logger.error(f"Failed to chunk {project_file.filename}")
                return False

            logger.info(f"Created {len(chunks)} chunks from {project_file.filename}")

            # Skip already-processed chunks
            chunks_to_process = [
                (idx, chunk) for idx, chunk in enumerate(chunks)
                if idx not in existing_indices
            ]

            if not chunks_to_process:
                logger.info(f"All chunks already processed for {project_file.filename}")
                return True

            logger.info(
                f"Processing {len(chunks_to_process)} new chunks "
                f"(skipping {len(existing_indices)} existing)"
            )

            # Extract just the text for embedding
            chunks_text = [chunk for idx, chunk in chunks_to_process]

            # Step 4: Compute embeddings for new chunks only
            embeddings = await self.compute_embeddings(chunks_text)
            if not embeddings:
                logger.error(f"Failed to compute embeddings for {project_file.filename}")
                return False

            if len(embeddings) != len(chunks_text):
                logger.error(
                    f"Embedding count mismatch for {project_file.filename}: "
                    f"{len(embeddings)} embeddings vs {len(chunks_text)} chunks"
                )
                return False

            logger.info(f"Computed {len(embeddings)} embeddings for {project_file.filename}")

            # Step 5: Store chunks and embeddings
            # Need to reconstruct full chunks list with proper indices
            full_chunks = []
            full_embeddings = []
            new_chunk_iter = iter(zip(chunks_text, embeddings))

            for idx in range(len(chunks)):
                if idx in existing_indices:
                    # Skip - already in DB
                    continue
                else:
                    # Add new chunk
                    chunk_text, embedding = next(new_chunk_iter)
                    full_chunks.append(chunk_text)
                    full_embeddings.append(embedding)

            stored = await self.store_chunks(
                project_id=project_file.project_id,
                file_id=project_file.id,
                chunks=chunks,  # Pass full list for metadata
                embeddings=full_embeddings,
                existing_indices=existing_indices
            )

            logger.info(
                f"Successfully processed {project_file.filename}: "
                f"stored {stored} chunks"
            )
            return True

        except Exception as e:
            logger.error(
                f"Error processing {project_file.filename}: {e}",
                exc_info=True
            )
            return False

    async def process_all_files(self, project_id: UUID) -> dict:
        """
        Process all files for a project.

        Returns:
            Dict with processing stats: {
                'total': int,
                'succeeded': int,
                'failed': int,
                'files': [...]
            }
        """
        files = await self.get_project_files(project_id)

        stats = {
            'total': len(files),
            'succeeded': 0,
            'failed': 0,
            'files': []
        }

        for project_file in files:
            try:
                success = await self.process_file(project_file)
                if success:
                    stats['succeeded'] += 1
                else:
                    stats['failed'] += 1

                stats['files'].append({
                    'filename': project_file.filename,
                    'size': project_file.size_bytes,
                    'success': success
                })

            except Exception as e:
                logger.error(f"Error processing {project_file.filename}: {e}", exc_info=True)
                stats['failed'] += 1
                stats['files'].append({
                    'filename': project_file.filename,
                    'size': project_file.size_bytes,
                    'success': False,
                    'error': str(e)
                })

        logger.info(
            f"File processing complete for project {project_id}: "
            f"{stats['succeeded']}/{stats['total']} succeeded, "
            f"{stats['failed']} failed"
        )

        return stats