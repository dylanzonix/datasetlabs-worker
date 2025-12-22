"""
Job processor that integrates with existing engines.
Updated to work with the projects table schema.
"""

import asyncio
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from azure.storage.blob import BlobServiceClient
from openai import OpenAI
from sqlalchemy import text
from sqlalchemy.orm import Session

from worker.config import settings
from worker.source_engine.engine import SourceDataEngine
from worker.synthetic_data_engine import SyntheticDataEngine

logger = logging.getLogger(__name__)


class JobContext:
    """
    Context passed to engines for cooperative cancellation and progress reporting.
    """

    def __init__(self, project_id: str, db_session_factory):
        self.project_id = project_id
        self.db_session_factory = db_session_factory
        self.stop_requested = False
        self.last_heartbeat = datetime.now(timezone.utc)
        self.last_checkpoint = datetime.now(timezone.utc)

    def should_stop(self) -> bool:
        """Check if the job should stop (pause/cancel requested)."""
        if self.stop_requested:
            return True

        # Check DB for pause/cancel signals
        with self.db_session_factory() as session:
            result = session.execute(
                text("SELECT desired_state FROM projects WHERE id = :project_id"),
                {"project_id": self.project_id}
            ).fetchone()

            if result:
                desired_state = result[0]
                if desired_state in ('pause', 'cancel'):
                    logger.info(f"Project {self.project_id} received {desired_state} signal")
                    self.stop_requested = True
                    return True

        return False

    def update_progress(self, progress: int, details: Optional[Dict[str, Any]] = None):
        """Update project progress in database."""
        now = datetime.now(timezone.utc)

        # Only update if enough time has passed
        if (now - self.last_heartbeat).total_seconds() < settings.heartbeat_interval_seconds:
            return

        with self.db_session_factory() as session:
            update_dict = {
                "project_id": self.project_id,
                "progress": progress,
                "now": now
            }

            session.execute(
                text("UPDATE projects SET progress = :progress, updated_at = :now WHERE id = :project_id"),
                update_dict
            )

            # Log progress event
            if details:
                session.execute(
                    text("""
                        INSERT INTO project_events (project_id, event_type, details, created_at)
                        VALUES (:project_id, 'progress', :details, :now)
                    """),
                    {
                        "project_id": self.project_id,
                        "details": json.dumps(details),
                        "now": now
                    }
                )

            session.commit()

        self.last_heartbeat = now

    def save_checkpoint(self, checkpoint_data: Dict[str, Any]):
        """Save checkpoint data to database."""
        now = datetime.now(timezone.utc)

        # Rate limit checkpoint saves
        if (now - self.last_checkpoint).total_seconds() < settings.checkpoint_interval_seconds:
            return

        with self.db_session_factory() as session:
            session.execute(
                text("UPDATE projects SET checkpoint = :checkpoint, updated_at = :now WHERE id = :project_id"),
                {
                    "project_id": self.project_id,
                    "checkpoint": json.dumps(checkpoint_data)
                }
            )
            session.commit()

        self.last_checkpoint = now
        logger.debug(f"Saved checkpoint for project {self.project_id}")


class JobProcessor:
    """
    Processes different types of jobs using the appropriate engine.
    """

    def __init__(self, openai_client: OpenAI, db_session_factory):
        self.openai_client = openai_client
        self.db_session_factory = db_session_factory

        # Azure Blob Storage client (optional for local testing)
        self.blob_service_client = None
        self.container_client = None

        # Only init if we have a valid connection string (not just "AccountName=...")
        conn_str = settings.azure_storage_connection_string
        if conn_str and len(conn_str) > 50 and not conn_str.endswith("..."):
            try:
                self.blob_service_client = BlobServiceClient.from_connection_string(conn_str)
                self.container_client = self.blob_service_client.get_container_client(
                    settings.azure_storage_container_name
                )
                self.container_client.create_container()
                logger.info("Azure Blob Storage initialized")
            except Exception as e:
                logger.warning(f"Blob storage unavailable: {e}")
        else:
            logger.info("Running without blob storage (mock mode)")

    def request_stop(self):
        """Request that current job processing stops."""
        # This would be called by signal handlers
        # Implementation depends on how we pass context to engines
        logger.info("Stop requested for current job")

    async def process_job(self, project_id: str, job_data: Dict[str, Any]) -> bool:
        """
        Process a job based on its type.

        Returns:
            bool: True if successful, False otherwise
        """
        job_type = job_data.get("job_type")

        try:
            # MOCK MODE: Just sleep for 60 seconds
            if not job_type:
                logger.info(f"Mock processing project {project_id} - sleeping 45s...")
                await asyncio.sleep(45)

                # Mark as succeeded
                with self.db_session_factory() as session:
                    session.execute(
                        text("""
                            UPDATE projects 
                            SET status = 'succeeded',
                                finished_at = :now,
                                updated_at = :now
                            WHERE id = :project_id
                        """),
                        {
                            "project_id": project_id,
                            "now": datetime.now(timezone.utc)
                        }
                    )
                    session.commit()

                logger.info(f"Mock processing completed for {project_id}")
                return True

            # Real processing
            if job_type == "source_data_processing":
                return await self._process_source_data_job(project_id, job_data)
            elif job_type == "synthetic_data_generation":
                return await self._process_synthetic_data_job(project_id, job_data)
            else:
                logger.error(f"Unknown job type: {job_type}")
                return False
        except Exception as e:
            logger.exception(f"Failed to process project {project_id}: {e}")

            # Update project with error
            with self.db_session_factory() as session:
                session.execute(
                    text("""
                        UPDATE projects 
                        SET status = 'failed', 
                            error = :error,
                            finished_at = :now,
                            updated_at = :now
                        WHERE id = :project_id
                    """),
                    {
                        "project_id": project_id,
                        "error": str(e),
                        "now": datetime.now(timezone.utc)
                    }
                )
                session.commit()

            return False

    async def _process_source_data_job(self, project_id: str, job_data: Dict[str, Any]) -> bool:
        """Process a source data job."""
        context = JobContext(project_id, self.db_session_factory)

        # Get job parameters
        input_blob_uri = job_data.get("input_blob_uri")
        topic_tree_uri = job_data.get("topic_tree_uri")
        params = job_data.get("params", {})

        # Download input files from blob storage
        input_dir = Path("/tmp") / f"project_{project_id}" / "input"
        input_dir.mkdir(parents=True, exist_ok=True)

        # Download source files
        logger.info(f"Downloading input from {input_blob_uri}")
        # TODO: Implement blob download
        # self._download_from_blob(input_blob_uri, input_dir / "sources")

        # Download topic tree
        logger.info(f"Downloading topic tree from {topic_tree_uri}")
        # TODO: Implement blob download
        # self._download_from_blob(topic_tree_uri, input_dir / "topic_tree.json")

        # Load topic tree
        with open(input_dir / "topic_tree.json") as f:
            topic_tree = json.load(f)

        # Initialize engine
        engine = SourceDataEngine(
            topic_tree=topic_tree,
            openai_client=self.openai_client,
            use_web=params.get("use_web", False)
        )

        # Process files with cooperative cancellation
        # TODO: Integrate your existing engine with context.should_stop() checks

        context.update_progress(50, {"status": "processing"})

        # Check for stop signal
        if context.should_stop():
            context.save_checkpoint({"stage": "processing", "processed": 50})
            logger.info(f"Project {project_id} stopped by request")

            # Update project status
            with self.db_session_factory() as session:
                session.execute(
                    text("UPDATE projects SET status = 'paused', updated_at = :now WHERE id = :project_id"),
                    {"project_id": project_id, "now": datetime.now(timezone.utc)}
                )
                session.commit()

            return False

        # Upload results to blob storage
        output_blob_name = f"projects/{project_id}/output.json"
        # TODO: Implement blob upload
        # blob_client = self.container_client.get_blob_client(output_blob_name)
        # blob_client.upload_blob(output_data)

        context.update_progress(100, {"status": "completed"})

        # Update project with output URI and mark as succeeded
        with self.db_session_factory() as session:
            session.execute(
                text("""
                    UPDATE projects 
                    SET output_blob_uri = :uri,
                        status = 'succeeded',
                        finished_at = :now,
                        updated_at = :now
                    WHERE id = :project_id
                """),
                {
                    "project_id": project_id,
                    "uri": output_blob_name,
                    "now": datetime.now(timezone.utc)
                }
            )
            session.commit()

        return True

    async def _process_synthetic_data_job(self, project_id: str, job_data: Dict[str, Any]) -> bool:
        """Process a synthetic data generation job."""
        context = JobContext(project_id, self.db_session_factory)

        # Get job parameters from the projects table
        with self.db_session_factory() as session:
            result = session.execute(
                text("""
                    SELECT num_samples, generation_prompt, diversity_axes, chat_history, checkpoint
                    FROM projects
                    WHERE id = :project_id
                """),
                {"project_id": project_id}
            ).fetchone()

            if not result:
                logger.error(f"Project {project_id} not found")
                return False

            num_samples, generation_prompt, diversity_axes, chat_history, checkpoint = result

        # Initialize SyntheticDataEngine
        engine = SyntheticDataEngine(
            openai_client=self.openai_client,
            # TODO: Add your engine-specific parameters
        )

        # TODO: Integrate your existing engine with:
        # - context.should_stop() checks
        # - context.update_progress() calls
        # - context.save_checkpoint() calls
        # - Use num_samples, generation_prompt, diversity_axes from DB

        context.update_progress(100, {"status": "completed"})

        # Update project status
        with self.db_session_factory() as session:
            session.execute(
                text("""
                    UPDATE projects 
                    SET status = 'succeeded',
                        finished_at = :now,
                        updated_at = :now
                    WHERE id = :project_id
                """),
                {
                    "project_id": project_id,
                    "now": datetime.now(timezone.utc)
                }
            )
            session.commit()

        return True

    def _download_from_blob(self, blob_uri: str, dest_path: Path):
        """Download a file from Azure Blob Storage."""
        # Extract container and blob name from URI
        # Format: https://<account>.blob.core.windows.net/<container>/<blob>
        parts = blob_uri.split('/')
        blob_name = '/'.join(parts[4:])  # Everything after container name

        blob_client = self.container_client.get_blob_client(blob_name)

        with open(dest_path, 'wb') as f:
            download_stream = blob_client.download_blob()
            f.write(download_stream.readall())

        logger.info(f"Downloaded {blob_uri} to {dest_path}")