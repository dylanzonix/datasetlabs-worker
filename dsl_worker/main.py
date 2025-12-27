import asyncio
import json
import logging
import os
import signal
import sys
from datetime import datetime, timezone
from typing import Optional
from uuid import UUID

from azure.servicebus.aio import ServiceBusClient
from azure.storage.blob import BlobServiceClient
from openai import AsyncOpenAI
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session

from dsl_worker.config import settings
from dsl_worker.job_processor import JobProcessor
from dsl_worker.logging_setup import setup_logging
from dsl_worker.synthetic_data_engine import SyntheticDataEngine
from dsl_api.models.project import Project

logger = logging.getLogger(__name__)


class WorkerService:
    """
    Background worker service that:
    1. Connects to Azure Service Bus
    2. Polls for job messages
    3. Processes jobs using JobProcessor
    4. Handles graceful shutdown

    Supports "noop mode" via WORKER_MODE=noop environment variable.
    In noop mode, the worker starts successfully but never processes messages,
    allowing local workers to handle all Service Bus messages for debugging.
    """

    def __init__(self):
        self.running = True
        self.current_processor: Optional[JobProcessor] = None
        self.noop_mode = os.getenv("WORKER_MODE", "").lower() == "noop"

        if self.noop_mode:
            logger.info("🔴 NOOP MODE ENABLED - Worker will not process any messages")
            logger.info("Worker will idle indefinitely for Azure health checks")
            # Skip all service initialization in noop mode
            return

        # Database connection
        self.engine = create_engine(settings.database_url, pool_pre_ping=True)
        self.SessionLocal = sessionmaker(bind=self.engine)

        # Azure Service Bus client
        self.service_bus_client = ServiceBusClient.from_connection_string(
            settings.azure_service_bus_connection_string
        )

        # Azure Blob Storage client
        account_url = f"https://{settings.azure_storage_account_name}.blob.core.windows.net"
        self.blob_service_client = BlobServiceClient(
            account_url=account_url,
            credential=settings.azure_storage_account_key
        )

        # OpenAI client (use AsyncOpenAI)
        self.openai_client = AsyncOpenAI(api_key=settings.openai_api_key)

        # Synthetic data engine
        self.synthetic_data_engine = SyntheticDataEngine(
            openai_client=self.openai_client
        )

        # Job processor
        self.job_processor = JobProcessor(
            db_session_factory=self.SessionLocal,
            openai_client=self.openai_client,
            blob_service_client=self.blob_service_client,
            synthetic_data_engine=self.synthetic_data_engine,
        )

        # Setup signal handlers
        signal.signal(signal.SIGTERM, self._handle_shutdown)
        signal.signal(signal.SIGINT, self._handle_shutdown)

    def _handle_shutdown(self, signum, frame):
        """Handle graceful shutdown signals."""
        logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        self.running = False

        # Signal current job to stop
        if self.current_processor:
            self.current_processor.request_stop()

    async def process_message(self, message):
        """Process a single message from the queue."""
        project_id = None
        try:
            # Parse message body
            body = json.loads(str(message))
            project_id_str = body.get("project_id")

            if not project_id_str:
                logger.error("Message missing project_id: %s", body)
                return

            project_id = UUID(project_id_str)
            logger.info(f"Processing project {project_id}")

            # Update project status to running using ORM
            db: Session = self.SessionLocal()
            try:
                project = db.query(Project).filter(Project.id == project_id).first()
                if project:
                    project.status = "running"
                    project.started_at = datetime.now(timezone.utc)
                    project.updated_at = datetime.now(timezone.utc)
                    db.commit()
                else:
                    logger.error(f"Project {project_id} not found")
                    return
            finally:
                db.close()

            # Process the job
            success = await self.job_processor.process_job(body)

            # Note: Final status is set in job_processor.process_job()
            # We only update here if there was an unexpected issue
            if not success:
                logger.warning(f"Project {project_id} completed with success=False")

            logger.info(f"Completed project {project_id} with status: {'succeeded' if success else 'failed'}")

        except Exception as e:
            logger.exception(f"Error processing message: {e}")
            # Update project to failed using ORM
            if project_id:
                db: Session = self.SessionLocal()
                try:
                    project = db.query(Project).filter(Project.id == project_id).first()
                    if project:
                        project.status = "failed"
                        project.error = str(e)
                        project.finished_at = datetime.now(timezone.utc)
                        project.updated_at = datetime.now(timezone.utc)
                        db.commit()
                finally:
                    db.close()

    async def run(self):
        """Main worker loop."""
        logger.info("Worker starting...")

        if self.noop_mode:
            logger.info("🔴 Running in NOOP mode - idling indefinitely")
            # Just sleep forever, keeping the container alive but not processing
            try:
                while self.running:
                    await asyncio.sleep(60)
                    logger.debug("Noop worker heartbeat")
            except asyncio.CancelledError:
                logger.info("Noop worker cancelled")
            logger.info("Noop worker stopped")
            return

        async with self.service_bus_client:
            receiver = self.service_bus_client.get_queue_receiver(
                queue_name=settings.azure_service_bus_queue_name,
                max_wait_time=30,  # Wait up to 30 seconds for messages
                max_lock_renewal_duration=3600  # Auto-renew lock for up to 1 hour
            )

            async with receiver:
                while self.running:
                    try:
                        # Receive messages with timeout
                        messages = await receiver.receive_messages(max_message_count=1, max_wait_time=5)

                        if not messages:
                            # No messages, check if we should keep running
                            if not self.running:
                                break
                            continue

                        for message in messages:
                            if not self.running:
                                # Don't start new work if shutting down
                                break

                            try:
                                await self.process_message(message)
                                # Complete the message (remove from queue)
                                await receiver.complete_message(message)
                            except Exception as e:
                                logger.exception(f"Failed to process message: {e}")
                                # Dead letter the message
                                await receiver.dead_letter_message(message, reason="ProcessingError",
                                                                   error_description=str(e))

                    except Exception as e:
                        logger.exception(f"Error in worker loop: {e}")
                        # Brief pause before retrying
                        await asyncio.sleep(5)

        logger.info("Worker stopped")


async def main():
    """Main entry point."""
    setup_logging()

    worker = WorkerService()

    try:
        await worker.run()
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt, shutting down...")
    except Exception as e:
        logger.exception(f"Worker crashed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())