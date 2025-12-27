import logging
from typing import List, Optional
from uuid import UUID
from pathlib import Path

from azure.storage.blob import BlobServiceClient
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)


class FileProcessor:
    """
    Handles file processing operations for a project:
    - Fetching files from database
    - Downloading from blob storage
    - Chunking content
    - Computing embeddings
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

        - CSV/JSONL/JSON: Row-wise chunking
        - Text files: Token-based chunking with overlap

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
                chunks = chunk_csv(content)

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
                    chunk_size=512,
                    overlap=50
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

    async def compute_embeddings(self, chunks: List[str]) -> List[List[float]]:
        """
        Compute embeddings for text chunks using OpenAI API.

        Args:
            chunks: List of text chunks

        Returns:
            List of embedding vectors (one per chunk)
        """
        if not chunks:
            return []

        embeddings = []

        try:
            logger.info(f"Computing embeddings for {len(chunks)} chunks")

            # OpenAI embeddings API
            # Using text-embedding-3-small (cheaper, still good quality)
            # Can batch up to 2048 inputs per request

            batch_size = 100  # Process in batches

            for i in range(0, len(chunks), batch_size):
                batch = chunks[i:i + batch_size]

                response = await self.openai_client.embeddings.create(
                    model="text-embedding-3-small",
                    input=batch
                )

                batch_embeddings = [item.embedding for item in response.data]
                embeddings.extend(batch_embeddings)

                logger.info(f"  Processed batch {i // batch_size + 1}: {len(batch)} chunks")

            logger.info(f"Successfully computed {len(embeddings)} embeddings")

        except Exception as e:
            logger.error(f"Failed to compute embeddings: {e}", exc_info=True)
            return []

        return embeddings

    async def process_file(self, project_file) -> bool:
        """
        Process a single file: download, chunk, compute embeddings.

        Returns:
            True if processing succeeded, False otherwise
        """
        logger.info(f"Processing file: {project_file.filename}")

        # Step 1: Download file
        content = await self.download_file(project_file)
        if content is None:
            logger.error(f"Failed to download {project_file.filename}")
            return False

        # Step 2: Chunk the content
        chunks = await self.chunk_content(content, project_file.content_type, project_file.filename)
        if not chunks:
            logger.error(f"Failed to chunk {project_file.filename}")
            return False

        logger.info(f"Created {len(chunks)} chunks from {project_file.filename}")

        # Step 3: Compute embeddings for each chunk
        embeddings = await self.compute_embeddings(chunks)
        if not embeddings:
            logger.error(f"Failed to compute embeddings for {project_file.filename}")
            return False

        if len(embeddings) != len(chunks):
            logger.error(
                f"Embedding count mismatch for {project_file.filename}: "
                f"{len(embeddings)} embeddings vs {len(chunks)} chunks"
            )
            return False

        logger.info(f"Computed {len(embeddings)} embeddings for {project_file.filename}")

        # TODO: Step 4: Store chunks and embeddings in database
        # For now, we'll just log success
        # await self.store_chunks(project_file.id, chunks, embeddings)

        logger.info(f"Successfully processed {project_file.filename}")
        return True

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