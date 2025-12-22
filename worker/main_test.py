"""
Simplified worker for testing Azure scaling.
Just sleeps for 60 seconds per job.
"""

import asyncio
import json
import logging
import signal
import sys
from datetime import datetime, timezone

from azure.servicebus.aio import ServiceBusClient
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from worker.config import settings
from worker.logging_setup import setup_logging

logger = logging.getLogger(__name__)


class TestWorkerService:
    """Minimal worker that processes messages by sleeping for 60s."""

    def __init__(self):
        self.running = True

        # Database connection
        self.engine = create_engine(settings.database_url, pool_pre_ping=True)
        self.SessionLocal = sessionmaker(bind=self.engine)

        # Azure Service Bus client
        self.service_bus_client = ServiceBusClient.from_connection_string(
            settings.azure_service_bus_connection_string
        )

        # Setup signal handlers
        signal.signal(signal.SIGTERM, self._handle_shutdown)
        signal.signal(signal.SIGINT, self._handle_shutdown)

    def _handle_shutdown(self, signum, frame):
        """Handle graceful shutdown signals."""
        logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        self.running = False

    async def process_message(self, message):
        """Process a single message - just sleep for 60 seconds."""
        project_id = None
        try:
            # Parse message body
            body = json.loads(str(message))
            project_id = body.get("project_id")

            if not project_id:
                logger.error("Message missing project_id: %s", body)
                return

            logger.info(f"🚀 Starting test job for project {project_id}")

            # Update project status to running
            with self.SessionLocal() as session:
                session.execute(
                    text("""
                        UPDATE projects 
                        SET status = 'running', 
                            started_at = :now,
                            updated_at = :now
                        WHERE id = :project_id
                    """),
                    {"project_id": project_id, "now": datetime.now(timezone.utc)}
                )
                session.commit()

            # Sleep for 60 seconds (simulating work)
            for i in range(60):
                await asyncio.sleep(1)
                progress = int((i + 1) / 60 * 100)

                # Update progress every 10 seconds
                if (i + 1) % 10 == 0:
                    with self.SessionLocal() as session:
                        session.execute(
                            text("UPDATE projects SET progress = :progress, updated_at = :now WHERE id = :project_id"),
                            {"project_id": project_id, "progress": progress, "now": datetime.now(timezone.utc)}
                        )
                        session.commit()
                    logger.info(f"⏳ Project {project_id} progress: {progress}%")

            # Mark as succeeded
            with self.SessionLocal() as session:
                session.execute(
                    text("""
                        UPDATE projects 
                        SET status = 'succeeded', 
                            progress = 100,
                            finished_at = :now,
                            updated_at = :now
                        WHERE id = :project_id
                    """),
                    {"project_id": project_id, "now": datetime.now(timezone.utc)}
                )
                session.commit()

            logger.info(f"✅ Completed project {project_id}")

        except Exception as e:
            logger.exception(f"❌ Error processing message: {e}")
            if project_id:
                with self.SessionLocal() as session:
                    session.execute(
                        text("""
                            UPDATE projects 
                            SET status = 'failed', 
                                error = :error, 
                                finished_at = :now,
                                updated_at = :now
                            WHERE id = :project_id
                        """),
                        {"project_id": project_id, "error": str(e), "now": datetime.now(timezone.utc)}
                    )
                    session.commit()

    async def run(self):
        """Main worker loop."""
        logger.info("🔧 Test Worker starting...")

        async with self.service_bus_client:
            receiver = self.service_bus_client.get_queue_receiver(
                queue_name=settings.azure_service_bus_queue_name,
                max_wait_time=30
            )

            async with receiver:
                while self.running:
                    try:
                        # Receive messages with timeout
                        messages = await receiver.receive_messages(max_message_count=1, max_wait_time=5)

                        if not messages:
                            if not self.running:
                                break
                            continue

                        for message in messages:
                            if not self.running:
                                break

                            try:
                                await self.process_message(message)
                                # Complete the message (remove from queue)
                                await receiver.complete_message(message)
                            except Exception as e:
                                logger.exception(f"Failed to process message: {e}")
                                # Dead letter the message
                                await receiver.dead_letter_message(
                                    message,
                                    reason="ProcessingError",
                                    error_description=str(e)
                                )

                    except Exception as e:
                        logger.exception(f"Error in worker loop: {e}")
                        await asyncio.sleep(5)

        logger.info("🛑 Test Worker stopped")


async def main():
    """Main entry point."""
    setup_logging()

    worker = TestWorkerService()

    try:
        await worker.run()
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt, shutting down...")
    except Exception as e:
        logger.exception(f"Worker crashed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())