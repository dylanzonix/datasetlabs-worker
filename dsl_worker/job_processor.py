"""
Job processor with phase-based orchestrator.

Manages the processing pipeline for dataset generation projects using
a flexible phase-based architecture that supports:
- Pause/resume
- Preview mode (concurrent processing)
- Config invalidation
- Easy extensibility
"""

import asyncio
import logging
from typing import Any, Dict, Optional
from uuid import UUID
from datetime import datetime, timezone

from sqlalchemy.orm import Session
from openai import AsyncOpenAI
from azure.storage.blob import BlobServiceClient

from dsl_api.azure.service_bus import ProjectPoke
from dsl_api.models.project import Project
from dsl_api.models.project_event import ProjectEvent

from dsl_worker.project_state import ProjectState
from dsl_worker.phases import (
    FileProcessingPhase,
    SeedExtractionPhase,
    SeedScoringPhase,
    SeedAssignmentPhase,
    RecipeBuildingPhase,
    GenerationPhase,
    ValidationPhase,
)

logger = logging.getLogger(__name__)


class JobProcessor:
    """
    Orchestrates dataset generation using a phase-based pipeline.

    The orchestrator:
    1. Maintains project state by polling the database
    2. Determines which phases should run based on state
    3. Executes active phases concurrently (via asyncio.gather)
    4. Handles pause requests gracefully
    5. Supports preview mode (eager generation)
    """

    def __init__(
            self,
            db_session_factory,
            openai_client: AsyncOpenAI,
            blob_service_client: BlobServiceClient,
            synthetic_data_engine=None,  # Not used in new architecture
    ):
        self.SessionLocal = db_session_factory
        self.openai_client = openai_client
        self.blob_service_client = blob_service_client
        self.should_stop = False

    def request_stop(self):
        """Request graceful stop (for SIGTERM/SIGINT)."""
        logger.info("Stop requested for current job")
        self.should_stop = True

    async def process_job(self, message_body: Dict[str, Any]) -> bool:
        """
        Process a single job from the queue using the orchestrator.

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

        logger.info(f"🚀 Starting orchestrator for project {poke.project_id}, run {poke.run_id}")

        db: Session = self.SessionLocal()
        try:
            # Load project
            project = db.query(Project).filter(Project.id == poke.project_id).first()
            if not project:
                logger.error(f"Project {poke.project_id} not found")
                return False

            # Verify run_id matches
            if project.current_run_id != poke.run_id:
                logger.warning(
                    f"Run ID mismatch for project {poke.project_id}: "
                    f"message has {poke.run_id}, project has {project.current_run_id}. "
                    f"Stale message, ignoring."
                )
                return True  # Acknowledge and discard

            logger.info(f"Processing project: {project.name}")
            logger.info(f"  Target samples: {project.num_samples}")
            logger.info(f"  Current status: {project.status}")
            logger.info(f"  Preview mode: {getattr(project, 'preview_mode', False)}")

            # Initialize state tracker
            state = ProjectState(db, poke.project_id, poke.run_id)

            # Create phases
            phases = [
                FileProcessingPhase('file_processing', state, db, self.openai_client, self.blob_service_client),
                SeedExtractionPhase('seed_extraction', state, db, self.openai_client),
                SeedScoringPhase('seed_scoring', state, db, self.openai_client),
                SeedAssignmentPhase('seed_assignment', state, db, self.openai_client),
                RecipeBuildingPhase('recipe_building', state, db, self.openai_client),
                GenerationPhase('generation', state, db, self.openai_client),
                ValidationPhase('validation', state, db, self.openai_client),
            ]

            # Emit RUNNING event
            self._emit_event(db, project, "running", "Worker started processing")

            # Main orchestrator loop
            iteration = 0
            while True:
                iteration += 1

                # Heartbeat: refresh state from database
                state.refresh()

                # Check for pause request
                if state.paused:
                    logger.info("⏸️  Pause request detected, pausing...")
                    self._handle_pause(db, project)
                    return True

                # Check for external stop signal (SIGTERM/SIGINT)
                if self.should_stop:
                    logger.info("⏸️  Stop signal received, pausing...")
                    project.status = "paused"
                    self._emit_event(db, project, "paused", "Worker stopped by signal")
                    db.commit()
                    return False

                # Find active phases (phases that have work to do)
                active_phases = [p for p in phases if p.should_run()]

                if not active_phases:
                    # Check if all phases are complete
                    all_complete = all(p.is_complete() for p in phases)

                    if all_complete:
                        logger.info("✅ All phases complete!")
                        self._handle_completion(db, project)
                        return True
                    else:
                        # No active phases but not all complete - might be waiting for something
                        logger.debug(f"No active phases, waiting... (iteration {iteration})")
                        await asyncio.sleep(2)
                        continue

                # Log active phases
                phase_names = [p.name for p in active_phases]
                logger.info(f"📊 Iteration {iteration}: Active phases: {phase_names}")

                # Execute all active phases concurrently
                tasks = []
                for phase in active_phases:
                    # Default batch size, but could be configured per phase
                    tasks.append(phase.execute_batch(batch_size=10))

                # Wait for all batches to complete
                results = await asyncio.gather(*tasks, return_exceptions=True)

                # Check for errors
                for i, result in enumerate(results):
                    if isinstance(result, Exception):
                        logger.error(f"❌ Error in {active_phases[i].name}: {result}")
                        raise result
                    else:
                        items_processed = result
                        if items_processed > 0:
                            logger.info(f"  ✓ {active_phases[i].name}: processed {items_processed} items")

                # Commit progress after each iteration
                db.commit()

                # Brief sleep to avoid tight loop
                # (especially when waiting for work to appear in preview mode)
                await asyncio.sleep(0.1)

        except Exception as e:
            logger.exception(f"❌ Error in orchestrator for project {poke.project_id}: {e}")

            # Try to update project status
            try:
                project = db.query(Project).filter(Project.id == poke.project_id).first()
                if project:
                    project.status = "failed"
                    project.error = str(e)
                    project.finished_at = datetime.now(timezone.utc)

                    self._emit_event(
                        db, project, "failed",
                        "Job failed with error",
                        details={"error": str(e)}
                    )
                    db.commit()
            except Exception as db_error:
                logger.error(f"Failed to update project status: {db_error}")

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
        """Emit a project event."""
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
        """
        Handle pausing the job.

        Updates status and emits confirmation event.
        """
        logger.info(f"Pausing project {project.id}")

        project.status = "paused"

        self._emit_event(
            db, project, "paused",
            "Worker paused processing",
            details={
                "paused_at": datetime.now(timezone.utc).isoformat(),
                "generated_count": project.generated_count
            }
        )

        logger.info(f"✅ Project {project.id} successfully paused")

    def _handle_completion(self, db: Session, project: Project) -> None:
        """
        Handle successful completion.

        Updates status and emits completion event.
        """
        logger.info(f"Project {project.id} completed successfully")

        project.status = "succeeded"
        project.finished_at = datetime.now(timezone.utc)

        # Update generated count to reflect actual completion
        # (should already be set by GenerationPhase, but double-check)
        if project.generated_count < project.num_samples:
            logger.warning(
                f"Generated count ({project.generated_count}) is less than target ({project.num_samples}), "
                f"but all phases report complete. Using actual count."
            )

        self._emit_event(
            db, project, "completed",
            "Generation completed successfully",
            details={
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "generated_count": project.generated_count
            }
        )

        db.commit()