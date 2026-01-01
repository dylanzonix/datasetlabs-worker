"""
Job processor orchestrator.

Executes phases with pause/resume support.

Key design:
- Persistent phases (file_processing, seed_extraction, seed_scoring) resume from where they left off
- Ephemeral phases (seed_assignment, generation) start fresh on each run
- Pause/resume is handled by database state, not in-memory state
"""

import asyncio
import logging
from typing import Any, Dict
from uuid import UUID
from datetime import datetime, timezone

from sqlalchemy.orm import Session
from openai import AsyncOpenAI
from azure.storage.blob import BlobServiceClient

from dsl_api.models.project import Project
from dsl_api.models.project_event import ProjectEvent

from dsl_worker.project_state import ProjectState
from dsl_worker.phases import (
    FileProcessingPhase,
    SeedExtractionPhase,
    SeedScoringPhase,
    SeedAssignmentPhase,
    GenerationPhase,
)

logger = logging.getLogger(__name__)


class JobProcessor:
    """
    Orchestrates dataset generation using phases.

    Processes work from each active phase in a loop.
    Handles pause requests by immediately stopping and cleaning up.
    """

    def __init__(
        self,
        db_session_factory,
        openai_client: AsyncOpenAI,
        blob_service_client: BlobServiceClient,
    ):
        self.SessionLocal = db_session_factory
        self.openai_client = openai_client
        self.blob_service_client = blob_service_client
        self.should_stop = False

    def request_stop(self):
        """Request graceful stop (SIGTERM/SIGINT)."""
        logger.info("Stop requested")
        self.should_stop = True

    async def process_job(self, message_body: Dict[str, Any]) -> bool:
        """
        Process a job using the phase orchestrator.

        Returns True if job completed or paused successfully.
        """
        project_id_str = message_body.get("project_id")
        run_id_str = message_body.get("run_id")

        if not project_id_str or not run_id_str:
            logger.error("Invalid message: missing project_id or run_id")
            return False

        project_id = UUID(project_id_str)
        run_id = UUID(run_id_str)

        logger.info(f"🚀 Starting orchestrator: project={project_id}, run={run_id}")

        db: Session = self.SessionLocal()
        try:
            project = db.query(Project).filter(Project.id == project_id).first()
            if not project:
                logger.error(f"Project not found: {project_id}")
                return False

            # Check run_id matches - if not, this is a stale message
            if project.current_run_id != run_id:
                logger.warning("Stale message (run mismatch), ignoring")
                return True

            logger.info(f"Project: {project.name}")
            logger.info(f"  Status: {project.status}")
            logger.info(f"  Target: {project.num_samples} samples")

            # Initialize state (no run_id needed - queries don't filter by it)
            state = ProjectState(db, project_id)

            # Create phases
            file_processing = FileProcessingPhase(
                'file_processing', state, db,
                self.openai_client, self.blob_service_client
            )
            seed_extraction = SeedExtractionPhase(
                'seed_extraction', state, db, self.openai_client
            )
            seed_scoring = SeedScoringPhase(
                'seed_scoring', state, db, self.openai_client
            )
            seed_assignment = SeedAssignmentPhase(
                'seed_assignment', state, db, self.openai_client
            )
            generation = GenerationPhase(
                'generation', state, db, self.openai_client,
                assignment_phase=seed_assignment
            )

            phases = [
                file_processing,
                seed_extraction,
                seed_scoring,
                seed_assignment,
                generation,
            ]

            # Emit RUNNING event
            self._emit_event(db, project, "running", "Worker started")

            # Main loop
            iteration = 0
            while True:
                iteration += 1

                # Refresh state from database
                state.refresh()

                # Check for pause
                if state.paused:
                    logger.info("⏸️  Pause detected")
                    self._handle_pause(db, project)
                    return True

                # Check for external stop
                if self.should_stop:
                    logger.info("⏸️  Stop signal received")
                    project.status = "paused"
                    self._emit_event(db, project, "paused", "Stopped by signal")
                    db.commit()
                    return False

                # Find active phases
                active = [p for p in phases if p.should_run()]

                if not active:
                    # Check if all complete
                    if all(p.is_complete() for p in phases):
                        logger.info("✅ All phases complete")
                        self._handle_completion(db, project)
                        return True
                    else:
                        # No work but not complete - wait
                        logger.debug(f"Iteration {iteration}: No active phases, waiting")
                        await asyncio.sleep(1)
                        continue

                # Log active phases periodically
                if iteration % 100 == 1:
                    logger.info(f"Iteration {iteration}: Active={[p.name for p in active]}")

                # Execute ONE unit from EACH active phase concurrently
                tasks = [p.execute_once() for p in active]
                results = await asyncio.gather(*tasks, return_exceptions=True)

                # Check for errors
                for i, result in enumerate(results):
                    if isinstance(result, Exception):
                        logger.error(f"❌ Error in {active[i].name}: {result}")
                        raise result

                # Commit after each iteration
                db.commit()

                # Brief sleep to avoid tight loop
                await asyncio.sleep(0.01)

        except Exception as e:
            logger.exception(f"❌ Orchestrator error: {e}")

            try:
                project = db.query(Project).filter(Project.id == project_id).first()
                if project:
                    project.status = "failed"
                    project.error = str(e)
                    project.finished_at = datetime.now(timezone.utc)
                    self._emit_event(db, project, "failed", "Error", {"error": str(e)})
                    db.commit()
            except Exception as db_err:
                logger.error(f"Failed to update status: {db_err}")

            return False

        finally:
            db.close()

    def _emit_event(
        self,
        db: Session,
        project: Project,
        event_type: str,
        message: str,
        details: dict = None
    ) -> None:
        """Emit project event."""
        event = ProjectEvent(
            project_id=project.id,
            run_id=project.current_run_id,
            event_type=event_type,
            message=message,
            details=details or {}
        )
        db.add(event)
        db.commit()

    def _handle_pause(self, db: Session, project: Project) -> None:
        """Handle pause: update status, emit event."""
        logger.info(f"Pausing project {project.id}")

        project.status = "paused"

        self._emit_event(
            db, project, "paused",
            "Worker paused",
            {
                "paused_at": datetime.now(timezone.utc).isoformat(),
                "generated_count": project.generated_count
            }
        )

        db.commit()
        logger.info("✅ Paused successfully")

    def _handle_completion(self, db: Session, project: Project) -> None:
        """Handle successful completion."""
        logger.info(f"Project {project.id} completed")

        project.status = "succeeded"
        project.finished_at = datetime.now(timezone.utc)

        self._emit_event(
            db, project, "completed",
            "Generation complete",
            {
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "generated_count": project.generated_count
            }
        )

        db.commit()