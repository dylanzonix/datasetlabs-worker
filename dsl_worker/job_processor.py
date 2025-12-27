import logging
from typing import Any, Dict
from uuid import UUID

from sqlalchemy.orm import Session
from openai import AsyncOpenAI
from azure.storage.blob import BlobServiceClient

from dsl_api.azure.service_bus import ProjectPoke
from dsl_worker.file_processor import FileProcessor
from dsl_worker.synthetic_data_engine import SyntheticDataEngine
from dsl_api.models.project import Project

logger = logging.getLogger(__name__)


class JobProcessor:
    """
    Processes dataset generation jobs.

    Main workflow:
    1. Load project from database
    2. Process uploaded files (if any)
    3. Generate synthetic data based on project config
    4. Export results
    """

    def __init__(
            self,
            db_session_factory,
            openai_client: AsyncOpenAI,
            blob_service_client: BlobServiceClient,
            synthetic_data_engine: SyntheticDataEngine,
    ):
        self.SessionLocal = db_session_factory
        self.openai_client = openai_client
        self.blob_service_client = blob_service_client
        self.synthetic_data_engine = synthetic_data_engine
        self.should_stop = False

    def request_stop(self):
        """Request graceful stop of current job."""
        logger.info("Stop requested for current job")
        self.should_stop = True

    async def process_job(self, message_body: Dict[str, Any]) -> bool:
        """
        Process a single job from the queue.

        Args:
            message_body: ProjectPoke message containing project_id and run_id

        Returns:
            True if job completed successfully, False otherwise
        """
        try:
            poke = ProjectPoke.from_dict(message_body)
        except (KeyError, ValueError) as e:
            logger.error(f"Invalid message format: {e}")
            return False

        logger.info(f"Starting job for project {poke.project_id}, run {poke.run_id}")

        db: Session = self.SessionLocal()
        try:
            # Load project
            project = db.query(Project).filter(Project.id == poke.project_id).first()
            if not project:
                logger.error(f"Project {poke.project_id} not found")
                return False

            # Verify run_id matches current_run_id
            if project.current_run_id != poke.run_id:
                logger.warning(
                    f"Run ID mismatch for project {poke.project_id}: "
                    f"message has {poke.run_id}, project has {project.current_run_id}. "
                    f"This is a stale message, ignoring."
                )
                return True  # Return True to acknowledge and discard the message

            logger.info(f"Processing project: {project.name} (run {poke.run_id})")
            logger.info(f"  Samples to generate: {project.num_samples}")
            logger.info(f"  Generation prompt: {project.generation_prompt[:100]}...")

            # Step 1: Process uploaded files (if any)
            file_processor = FileProcessor(
                blob_service_client=self.blob_service_client,
                openai_client=self.openai_client,
                db_session=db
            )

            logger.info("=" * 60)
            logger.info("STEP 1: Processing uploaded files")
            logger.info("=" * 60)

            file_stats = await file_processor.process_all_files(poke.project_id)

            logger.info(f"File processing results:")
            logger.info(f"  Total files: {file_stats['total']}")
            logger.info(f"  Succeeded: {file_stats['succeeded']}")
            logger.info(f"  Failed: {file_stats['failed']}")

            # If we had files but they all failed, maybe fail the job?
            if file_stats['total'] > 0 and file_stats['succeeded'] == 0:
                logger.error("All file processing failed, aborting job")
                project.status = "failed"
                project.error = "Failed to process uploaded files"
                db.commit()
                return False

            # Check for stop signal
            if self.should_stop:
                logger.info("Stop requested, aborting job")
                project.status = "paused"
                db.commit()
                return False

            # TODO: Step 2: Generate synthetic data
            logger.info("=" * 60)
            logger.info("STEP 2: Generating synthetic data")
            logger.info("=" * 60)
            logger.info("TODO: Implement synthetic data generation")

            # TODO: Step 3: Export results
            logger.info("=" * 60)
            logger.info("STEP 3: Exporting results")
            logger.info("=" * 60)
            logger.info("TODO: Implement export")

            # Mark as succeeded for now
            project.status = "succeeded"
            project.generated_count = project.num_samples
            db.commit()

            logger.info(f"Job completed for project {poke.project_id}")
            return True

        except Exception as e:
            logger.exception(f"Error processing job for project {poke.project_id}: {e}")

            # Try to update project status
            try:
                project = db.query(Project).filter(Project.id == poke.project_id).first()
                if project:
                    project.status = "failed"
                    project.error = str(e)
                    db.commit()
            except Exception as db_error:
                logger.error(f"Failed to update project status: {db_error}")

            return False

        finally:
            db.close()