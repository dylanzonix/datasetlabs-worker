"""
Job processor that integrates with existing engines.
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

    def __init__(self, job_id: str, db_session_factory):
        self.job_id = job_id
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
                text("SELECT desired_state FROM jobs WHERE id = :job_id"),
                {"job_id": self.job_id}
            ).fetchone()

            if result:
                desired_state = result[0]
                if desired_state in ('pause', 'cancel'):
                    logger.info(f"Job {self.job_id} received {desired_state} signal")
                    self.stop_requested = True
                    return True

        return False

    def update_progress(self, progress: int, details: Optional[Dict[str, Any]] = None):
        """Update job progress in database."""
        now = datetime.now(timezone.utc)

        # Only update if enough time has passed
        if (now - self.last_heartbeat).total_seconds() < settings.heartbeat_interval_seconds:
            return

        with self.db_session_factory() as session:
            update_dict = {
                "job_id": self.job_id,
                "progress": progress,
                "now": now
            }

            session.execute(
                text("UPDATE jobs SET progress = :progress, updated_at = :now WHERE id = :job_id"),
                update_dict
            )

            # Log progress event
            if details:
                session.execute(
                    text("""
                        INSERT INTO job_events (job_id, event_type, details, created_at)
                        VALUES (:job_id, 'progress', :details, :now)
                    """),
                    {
                        "job_id": self.job_id,
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
                text("UPDATE jobs SET checkpoint = :checkpoint WHERE id = :job_id"),
                {
                    "job_id": self.job_id,
                    "checkpoint": json.dumps(checkpoint_data)
                }
            )
            session.commit()

        self.last_checkpoint = now
        logger.debug(f"Saved checkpoint for job {self.job_id}")


class JobProcessor:
    """
    Processes different types of jobs using the appropriate engine.
    """

    def __init__(self, openai_client: OpenAI, db_session_factory):
        self.openai_client = openai_client
        self.db_session_factory = db_session_factory

        # Azure Blob Storage client
        self.blob_service_client = BlobServiceClient.from_connection_string(
            settings.azure_storage_connection_string
        )
        self.container_client = self.blob_service_client.get_container_client(
            settings.azure_storage_container_name
        )

        # Ensure container exists
        try:
            self.container_client.create_container()
        except Exception:
            pass  # Container already exists

    def request_stop(self):
        """Request that current job processing stops."""
        # This would be called by signal handlers
        # Implementation depends on how we pass context to engines
        logger.info("Stop requested for current job")

    async def process_job(self, job_id: str, job_data: Dict[str, Any]) -> bool:
        """
        Process a job based on its type.

        Returns:
            bool: True if successful, False otherwise
        """
        job_type = job_data.get("job_type")

        try:
            if job_type == "source_data_processing":
                return await self._process_source_data_job(job_id, job_data)
            elif job_type == "synthetic_data_generation":
                return await self._process_synthetic_data_job(job_id, job_data)
            else:
                logger.error(f"Unknown job type: {job_type}")
                return False
        except Exception as e:
            logger.exception(f"Failed to process job {job_id}: {e}")
            return False

    async def _process_source_data_job(self, job_id: str, job_data: Dict[str, Any]) -> bool:
        """Process a source data job."""
        context = JobContext(job_id, self.db_session_factory)

        # Get job parameters
        input_blob_uri = job_data.get("input_blob_uri")
        topic_tree_uri = job_data.get("topic_tree_uri")
        params = job_data.get("params", {})

        # Download input files from blob storage
        input_dir = Path("/tmp") / f"job_{job_id}" / "input"
        input_dir.mkdir(parents=True, exist_ok=True)

        # Download source files
        # This is a placeholder - you'd implement actual blob download logic
        logger.info(f"Downloading input from {input_blob_uri}")

        # Download topic tree
        logger.info(f"Downloading topic tree from {topic_tree_uri}")

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
        # This is where you'd integrate your existing engine with context.should_stop() checks
        # For now, this is a placeholder

        context.update_progress(50, {"status": "processing"})

        # Check for stop signal
        if context.should_stop():
            context.save_checkpoint({"stage": "processing", "processed": 50})
            logger.info(f"Job {job_id} stopped by request")
            return False

        # Upload results to blob storage
        output_blob_name = f"jobs/{job_id}/output.json"
        # blob_client = self.container_client.get_blob_client(output_blob_name)
        # blob_client.upload_blob(output_data)

        context.update_progress(100, {"status": "completed"})

        # Update job with output URI
        with self.db_session_factory() as session:
            session.execute(
                text("UPDATE jobs SET output_blob = :uri WHERE id = :job_id"),
                {"job_id": job_id, "uri": output_blob_name}
            )
            session.commit()

        return True

    async def _process_synthetic_data_job(self, job_id: str, job_data: Dict[str, Any]) -> bool:
        """Process a synthetic data generation job."""
        context = JobContext(job_id, self.db_session_factory)

        # Similar structure to source data processing
        # Initialize SyntheticDataEngine and process

        context.update_progress(100, {"status": "completed"})
        return True