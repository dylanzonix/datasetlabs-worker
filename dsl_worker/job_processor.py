"""
Job processor orchestrator.

Executes phases with pause/resume support and cost tracking.

Key design:
- Persistent phases (file_processing, seed_extraction, seed_scoring) resume from where they left off
- Ephemeral phases (seed_assignment, generation) start fresh on each run
- Pause/resume is handled by database state, not in-memory state
- Cost tracking with periodic charging to user balance
- Force-stop when balance depleted or spend limit exceeded
- Only one project can run at a time per user
"""

import asyncio
import logging
from typing import Any, Dict, Optional
from uuid import UUID
from datetime import datetime, timezone

from sqlalchemy.orm import Session
from openai import AsyncOpenAI
from azure.storage.blob import BlobServiceClient

from dsl_api.models.project import Project
from dsl_api.models.project_event import ProjectEvent

from dsl_worker.config import settings
from dsl_worker.project_state import ProjectState
from dsl_worker.billing import TrackedOpenAIClient, CostTracker
from dsl_worker.phases import (
    PhaseResult,
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
    Tracks costs and charges user balance periodically.
    Force-stops projects when balance is depleted or spend limit exceeded.
    """

    def __init__(
        self,
        db_session_factory,
        openai_client: AsyncOpenAI,
        blob_service_client: BlobServiceClient,
    ):
        self.SessionLocal = db_session_factory
        self.raw_openai_client = openai_client
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

            # Check if project is paused - if so, this is a stale message from before the pause
            # The user must explicitly resume (which sends a new message)
            if project.status == "paused":
                logger.warning("Stale message (project is paused), ignoring")
                return True

            # Also reject if already succeeded/failed - no need to reprocess
            if project.status == "failed":
                logger.warning(f"Stale message (project is {project.status}), ignoring")
                return True

            # NOW we can set status to running (after confirming message is valid)
            # This prevents stale messages from incorrectly setting status to running
            project.status = "running"
            project.started_at = datetime.now(timezone.utc)
            project.updated_at = datetime.now(timezone.utc)
            db.commit()

            # Check if another project is already running for this user
            running_project = (
                db.query(Project)
                .filter(
                    Project.user_id == project.user_id,
                    Project.status == "running",
                    Project.id != project_id,
                )
                .first()
            )
            if running_project:
                logger.warning(
                    f"User {project.user_id} already has running project {running_project.id}, "
                    f"cannot start {project_id}"
                )
                # Return False to keep message in queue for retry
                return False

            logger.info(f"Project: {project.name}")
            logger.info(f"  Status: {project.status}")
            logger.info(f"  Target: {project.num_samples} samples")

            # Create tracked OpenAI client
            tracked_client = TrackedOpenAIClient(self.raw_openai_client)

            # Create cost tracker with spend limit
            cost_tracker = CostTracker(
                db=db,
                user_id=project.user_id,
                project_id=project_id,
                margin_multiplier=settings.billing_margin_multiplier,
                charge_threshold_cents=settings.billing_charge_threshold_cents,
                charge_interval_seconds=settings.billing_charge_interval_seconds,
                spend_limit_cents=project.spend_limit_cents,
            )

            # Check initial balance and spend limit
            can_continue, stop_reason = cost_tracker.check_balance_and_charge()
            if not can_continue:
                logger.warning(f"Cannot start project: {stop_reason}")
                self._handle_force_stop(db, project, cost_tracker, stop_reason)
                return False

            # Initialize state (no run_id needed - queries don't filter by it)
            state = ProjectState(db, project_id)

            # Create phases
            file_processing = FileProcessingPhase(
                'file_processing', state, db,
                tracked_client, self.blob_service_client
            )
            seed_extraction = SeedExtractionPhase(
                'seed_extraction', state, db, tracked_client
            )
            seed_scoring = SeedScoringPhase(
                'seed_scoring', state, db, tracked_client
            )
            seed_assignment = SeedAssignmentPhase(
                'seed_assignment', state, db, tracked_client
            )
            generation = GenerationPhase(
                'generation', state, db, tracked_client,
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
                    self._handle_pause(db, project, cost_tracker)
                    return True

                # Check for external stop
                if self.should_stop:
                    logger.info("⏸️  Stop signal received")
                    self._handle_pause(db, project, cost_tracker, "Stopped by signal")
                    return False

                # Check balance and charge if needed
                can_continue, stop_reason = cost_tracker.check_balance_and_charge()
                if not can_continue:
                    logger.warning(f"💸 Stopping: {stop_reason}")
                    self._handle_force_stop(db, project, cost_tracker, stop_reason)
                    return False

                # Find active phases
                active = [p for p in phases if p.should_run()]

                if not active:
                    # Check if all complete
                    if all(p.is_complete() for p in phases):
                        logger.info("✅ All phases complete")
                        self._handle_completion(db, project, cost_tracker)
                        return True
                    else:
                        # No work but not complete - wait with stop check
                        logger.debug(f"Iteration {iteration}: No active phases, waiting")
                        for _ in range(10):  # 10 x 0.1s = 1s total
                            if self.should_stop:
                                logger.info("⏸️  Stop signal received during wait")
                                self._handle_pause(db, project, cost_tracker, "Stopped by signal")
                                return False
                            await asyncio.sleep(0.1)
                        continue

                # Log active phases periodically
                if iteration % 100 == 1:
                    logger.info(f"Iteration {iteration}: Active={[p.name for p in active]}")

                # Execute ONE unit from EACH active phase concurrently
                tasks = [p.execute_once() for p in active]

                try:
                    # Use wait_for with timeout to allow checking should_stop
                    results = await asyncio.wait_for(
                        asyncio.gather(*tasks, return_exceptions=True),
                        timeout=60.0
                    )
                except asyncio.TimeoutError:
                    # Timeout hit - check if we should stop
                    logger.warning("Phase execution timeout (60s)")
                    if self.should_stop:
                        logger.info("⏸️  Stop signal received during timeout")
                        self._handle_pause(db, project, cost_tracker, "Stopped by signal")
                        return False
                    continue

                # Process results and track costs
                for i, result in enumerate(results):
                    if isinstance(result, Exception):
                        logger.error(f"❌ Error in {active[i].name}: {result}")
                        raise result

                    # Track cost from this phase
                    if isinstance(result, PhaseResult) and result.cost_usd > 0:
                        cost_tracker.add_cost(
                            phase=active[i].name,
                            cost_usd=result.cost_usd,
                        )

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

    def _handle_pause(
        self,
        db: Session,
        project: Project,
        cost_tracker: CostTracker,
        message: str = "Worker paused"
    ) -> None:
        """Handle pause: update status, charge remaining costs, emit event."""
        logger.info(f"Pausing project {project.id}")

        # Charge any remaining costs
        cost_tracker.charge_remaining()

        project.status = "paused"

        summary = cost_tracker.get_summary()
        self._emit_event(
            db, project, "paused",
            message,
            {
                "paused_at": datetime.now(timezone.utc).isoformat(),
                "generated_count": project.generated_count,
                "total_cost_cents": summary["total_costs_cents"],
                "preprocessing_cost_cents": summary["preprocessing_costs_cents"],
                "generation_cost_cents": summary["generation_costs_cents"],
            }
        )

        db.commit()
        logger.info("✅ Paused successfully")

    def _handle_completion(
        self,
        db: Session,
        project: Project,
        cost_tracker: CostTracker
    ) -> None:
        """Handle successful completion."""
        logger.info(f"Project {project.id} completed")

        # Charge any remaining costs
        cost_tracker.charge_remaining()

        project.status = "succeeded"
        project.finished_at = datetime.now(timezone.utc)

        summary = cost_tracker.get_summary()
        self._emit_event(
            db, project, "completed",
            "Generation complete",
            {
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "generated_count": project.generated_count,
                "total_cost_cents": summary["total_costs_cents"],
                "preprocessing_cost_cents": summary["preprocessing_costs_cents"],
                "generation_cost_cents": summary["generation_costs_cents"],
                "cumulative_project_spend_cents": summary["cumulative_project_spend_cents"],
            }
        )

        db.commit()
        logger.info(
            f"✅ Completed: {summary['total_costs_cents']}¢ total "
            f"(preprocessing: {summary['preprocessing_costs_cents']}¢, "
            f"generation: {summary['generation_costs_cents']}¢), "
            f"balance: {summary['user_balance_cents']}¢"
        )

    def _handle_force_stop(
        self,
        db: Session,
        project: Project,
        cost_tracker: CostTracker,
        reason: str
    ) -> None:
        """
        Handle force-stop due to balance depletion or spend limit exceeded.

        Uses 'failed' status with descriptive error message.
        """
        logger.warning(f"Force-stopping project {project.id}: {reason}")

        # Charge any remaining costs
        cost_tracker.charge_remaining()

        # Build error message
        if reason == "insufficient_balance":
            error_msg = "Insufficient balance to continue generation"
        elif reason == "spend_limit_exceeded":
            limit = project.spend_limit_cents
            error_msg = f"Project spend limit of ${limit / 100:.2f} exceeded" if limit else "Spend limit exceeded"
        else:
            error_msg = f"Force-stopped: {reason}"

        # Update project
        project.status = "failed"
        project.error = error_msg
        project.finished_at = datetime.now(timezone.utc)

        # Build event details
        summary = cost_tracker.get_summary()
        details = {
            "reason": reason,
            "stopped_at": datetime.now(timezone.utc).isoformat(),
            "generated_count": project.generated_count,
            "total_cost_cents": summary["total_costs_cents"],
            "preprocessing_cost_cents": summary["preprocessing_costs_cents"],
            "generation_cost_cents": summary["generation_costs_cents"],
            "cumulative_project_spend_cents": summary["cumulative_project_spend_cents"],
            "final_balance_cents": summary["user_balance_cents"],
        }

        # Add spend limit info if relevant
        if project.spend_limit_cents is not None:
            details["spend_limit_cents"] = project.spend_limit_cents
            details["remaining_budget_cents"] = summary.get("remaining_budget_cents", 0)

        self._emit_event(db, project, "failed", error_msg, details)

        db.commit()
        logger.info(
            f"🛑 Force-stopped: {reason} "
            f"(preprocessing: {summary['preprocessing_costs_cents']}¢, "
            f"generation: {summary['generation_costs_cents']}¢)"
        )