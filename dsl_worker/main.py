"""
DSL Worker Main Entry Point

Processes jobs from Azure Service Bus queue.
"""

import asyncio
import json
import logging
import os
import signal
from datetime import datetime, timezone
from typing import Optional
from uuid import UUID

from azure.servicebus.aio import ServiceBusClient, AutoLockRenewer
from azure.servicebus.exceptions import MessageLockLostError
from openai import AsyncOpenAI
from sqlalchemy.orm import Session
from azure.storage.blob import BlobServiceClient

from dsl_api.db import SessionLocal
from dsl_api.models.project import Project
from dsl_worker.config import settings
from dsl_worker.job_processor_v3 import JobProcessorV3
from dsl_worker.logging_setup import setup_logging

logger = logging.getLogger(__name__)


class Worker:
    """Main worker class that processes messages from Azure Service Bus."""

    def __init__(self):
        self.running = True
        self.noop_mode = os.getenv("NOOP_MODE", "").lower() in ("true", "1", "yes")

        # Database session factory
        self.SessionLocal = SessionLocal

        # Initialize clients
        self.service_bus_client = ServiceBusClient.from_connection_string(
            settings.azure_service_bus_connection_string
        )

        self.openai_client = AsyncOpenAI(api_key=settings.openai_api_key)

        self.blob_service_client = BlobServiceClient(
            account_url=f"https://{settings.azure_storage_account_name}.blob.core.windows.net",
            credential=settings.azure_storage_account_key,
        )

        # Job processor (v3 - new pipeline)
        self.job_processor = JobProcessorV3(
            db_session_factory=self.SessionLocal,
            openai_client=self.openai_client,
            blob_service_client=self.blob_service_client,
        )
        self.current_processor = self.job_processor

        # Setup signal handlers
        signal.signal(signal.SIGTERM, self._handle_shutdown)
        signal.signal(signal.SIGINT, self._handle_shutdown)

    def _handle_shutdown(self, signum, frame):
        """Handle graceful shutdown signals."""
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

            # Process the job
            success = await self.job_processor.process_job(body)

            if success:
                logger.info(f"Completed processing project {project_id}")
            else:
                logger.warning(f"Project {project_id} processing returned False")

        except Exception as e:
            logger.exception(f"Error processing message: {e}")
            # Update project to failed
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
                max_wait_time=30,
            )

            renewer = AutoLockRenewer(max_lock_renewal_duration=3600)

            async with receiver:
                while self.running:
                    try:
                        messages = await receiver.receive_messages(max_message_count=1, max_wait_time=5)

                        if not messages:
                            if not self.running:
                                break
                            continue

                        for message in messages:
                            if not self.running:
                                break

                            try:
                                renewer.register(receiver, message, max_lock_renewal_duration=3600)
                                logger.debug(f"Registered message for auto-lock renewal")

                                await self.process_message(message)
                                await receiver.complete_message(message)

                            except MessageLockLostError as e:
                                logger.error(f"⚠️ Message lock lost! {e}")

                            except Exception as e:
                                logger.exception(f"Failed to process message: {e}")
                                try:
                                    await receiver.dead_letter_message(
                                        message,
                                        reason="ProcessingError",
                                        error_description=str(e)
                                    )
                                except MessageLockLostError:
                                    logger.warning("Could not dead-letter message - lock already lost")

                    except Exception as e:
                        logger.exception(f"Error in worker loop: {e}")
                        await asyncio.sleep(5)

            await renewer.close()

        logger.info("Worker stopped")


async def main():
    """Main entry point."""
    setup_logging()
    worker = Worker()
    await worker.run()


if __name__ == "__main__":
    asyncio.run(main())