"""
Worker entrypoint for processing jobs from Azure Service Bus.
"""

import asyncio
import json
import logging
import signal
import sys
from datetime import datetime, timezone
from typing import Optional

from azure.servicebus.aio import ServiceBusClient
from azure.servicebus import ServiceBusMessage
from openai import OpenAI
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from worker.config import settings
from worker.job_processor import JobProcessor
from worker.logging_setup import setup_logging

logger = logging.getLogger(__name__)


class WorkerService:
    """
    Main worker service that:
    1. Connects to Azure Service Bus queue
    2. Pulls job messages
    3. Processes jobs using JobProcessor
    4. Handles graceful shutdown
    """

    def __init__(self):
        self.running = True
        self.current_processor: Optional[JobProcessor] = None

        # Database connection
        self.engine = create_engine(settings.database_url, pool_pre_ping=True)
        self.SessionLocal = sessionmaker(bind=self.engine)

        # Azure Service Bus client
        self.service_bus_client = ServiceBusClient.from_connection_string(
            settings.azure_service_bus_connection_string
        )

        # OpenAI client
        self.openai_client = OpenAI(api_key=settings.openai_api_key)

        # Job processor
        self.job_processor = JobProcessor(
            openai_client=self.openai_client,
            db_session_factory=self.SessionLocal,
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
        job_id = None
        try:
            # Parse message body
            body = json.loads(str(message))
            job_id = body.get("job_id")

            if not job_id:
                logger.error("Message missing job_id: %s", body)
                return

            logger.info(f"Processing job {job_id}")

            # Update job status to running
            with self.SessionLocal() as session:
                session.execute(
                    text("UPDATE jobs SET status = 'running', started_at = :now WHERE id = :job_id"),
                    {"job_id": job_id, "now": datetime.now(timezone.utc)}
                )
                session.commit()

            # Process the job
            success = await self.job_processor.process_job(job_id, body)

            # Update final status
            with self.SessionLocal() as session:
                if success:
                    session.execute(
                        text("UPDATE jobs SET status = 'succeeded', finished_at = :now WHERE id = :job_id"),
                        {"job_id": job_id, "now": datetime.now(timezone.utc)}
                    )
                else:
                    session.execute(
                        text("UPDATE jobs SET status = 'failed', finished_at = :now WHERE id = :job_id"),
                        {"job_id": job_id, "now": datetime.now(timezone.utc)}
                    )
                session.commit()

            logger.info(f"Completed job {job_id} with status: {'succeeded' if success else 'failed'}")

        except Exception as e:
            logger.exception(f"Error processing message: {e}")
            # Update job to failed
            if job_id:
                with self.SessionLocal() as session:
                    session.execute(
                        text("UPDATE jobs SET status = 'failed', error = :error, finished_at = :now WHERE id = :job_id"),
                        {"job_id": job_id, "error": str(e), "now": datetime.now(timezone.utc)}
                    )
                    session.commit()

    async def run(self):
        """Main worker loop."""
        logger.info("Worker starting...")

        async with self.service_bus_client:
            receiver = self.service_bus_client.get_queue_receiver(
                queue_name=settings.azure_service_bus_queue_name,
                max_wait_time=30  # Wait up to 30 seconds for messages
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
                                await receiver.dead_letter_message(message, reason="ProcessingError", error_description=str(e))

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