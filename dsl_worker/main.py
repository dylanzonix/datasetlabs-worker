"""
Worker service main loop.

Handles:
1. Connecting to Azure Service Bus
2. Receiving messages from queue
3. Dispatching to job processor
4. Graceful shutdown
"""

import asyncio
import json
import logging
import signal
import sys
from datetime import datetime, timezone
from uuid import UUID

from sqlalchemy.orm import Session
from openai import AsyncOpenAI
from azure.servicebus.aio import ServiceBusClient
from azure.storage.blob import BlobServiceClient

from dsl_api.db import SessionLocal
from dsl_api.models.project import Project

from dsl_worker.config import settings
from dsl_worker.job_processor import JobProcessor

logger = logging.getLogger(__name__)


def setup_logging():
    """Configure logging for worker."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    # Reduce noise from Azure SDK
    logging.getLogger("azure").setLevel(logging.WARNING)
    logging.getLogger("azure.servicebus").setLevel(logging.INFO)


class WorkerService:
    """
    Main worker service.

    Responsibilities:
    1. Connects to Azure Service Bus
    2. Receives job messages
    3. Dispatches to JobProcessor
    4. Handles graceful shutdown
    """

    def __init__(self, noop_mode: bool = False):
        self.noop_mode = noop_mode
        self.running = True
        self.current_processor: JobProcessor | None = None

        # Database session factory
        self.SessionLocal = SessionLocal

        # Azure Service Bus client
        self.service_bus_client = ServiceBusClient.from_connection_string(
            settings.azure_service_bus_connection_string
        )

        # OpenAI client
        openai_client = AsyncOpenAI(api_key=settings.openai_api_key)

        # Blob storage client
        blob_service_client = BlobServiceClient(
            account_url=f"https://{settings.azure_storage_account_name}.blob.core.windows.net",
            credential=settings.azure_storage_account_key,
        )

        # Job processor
        self.job_processor = JobProcessor(
            db_session_factory=self.SessionLocal,
            openai_client=openai_client,
            blob_service_client=blob_service_client,
        )
        self.current_processor = self.job_processor

        # Setup signal handlers
        signal.signal(signal.SIGTERM, self._handle_shutdown)
        signal.signal(signal.SIGINT, self._handle_shutdown)

    def _handle_shutdown(self, signum, frame):
        """Handle graceful shutdown signals."""
        # CHANGED: Use WARNING level and more descriptive message
        import signal
        signal_name = "SIGTERM" if signum == signal.SIGTERM else "SIGINT" if signum == signal.SIGINT else f"signal {signum}"
        logger.warning(f"⚠️ SHUTDOWN SIGNAL RECEIVED: {signal_name} - initiating graceful shutdown...")
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

            # NOTE: We do NOT set status to "running" here anymore.
            # The job_processor will set it after validating the message is not stale.
            # This prevents stale messages from incorrectly flipping status to "running".

            # Process the job
            success = await self.job_processor.process_job(body)

            # Log result (status is already set correctly by job_processor)
            if success:
                logger.info(f"Completed processing project {project_id}")
            else:
                logger.warning(f"Project {project_id} processing returned False")

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