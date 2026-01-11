"""
Job processor for the DSL worker.

Orchestrates phases and manages project lifecycle:
- File processing → Seed extraction → Seed scoring → Seed assignment → Generation
- Tracks costs and charges user balance incrementally
- Force-stops projects when balance is depleted or spend limit exceeded
- Handles PAUSE immediately - cancels in-flight generation workers

VERSION SEMANTICS:
- Each job is associated with a specific version_id
- (project_id, version_id) is treated as a completely fresh project
- All phases start from scratch for each new version
- Seeds, samples, and events are all scoped to the version
"""

import asyncio
import logging
import time
from datetime import datetime, timezone
from typing import Any, Dict, Optional
from uuid import UUID

from openai import AsyncOpenAI
from sqlalchemy.orm import Session
from azure.storage.blob import BlobServiceClient

from dsl_api.db import SessionLocal
from dsl_api.models.project import Project
from dsl_api.models.project_version import ProjectVersion
from dsl_api.models.project_event import ProjectEvent

from dsl_worker.config import settings
from dsl_worker.project_state import ProjectState
from dsl_worker.billing import CostTracker, TrackedOpenAIClient

from dsl_worker.phases.base import PhaseResult
from dsl_worker.phases.file_processing import FileProcessingPhase
from dsl_worker.phases.seed_extraction import SeedExtractionPhase
from dsl_worker.phases.seed_scoring import SeedScoringPhase
from dsl_worker.phases.seed_assignment import SeedAssignmentPhase
from dsl_worker.phases.generation import GenerationPhase

logger = logging.getLogger(__name__)


class JobProcessor:
    """
    Main job processor.

    Orchestrates all phases for a project VERSION:
    1. File processing (chunking + embedding)
    2. Seed extraction (from chunks)
    3. Seed scoring (against diversity axes)
    4. Seed assignment (to diversity slots)
    5. Sample generation (agentic tool calling)

    IMPORTANT: All work is scoped to the version_id.
    A new version means starting completely fresh.

    Tracks costs and charges user balance periodically.
    Force-stops projects when balance is depleted or spend limit exceeded.

    PAUSE HANDLING:
    - Checks for pause every iteration of the main loop
    - When pause is detected during generation, immediately cancels all workers
    - Costs are charged before pausing
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

        # Reference to generation phase for immediate stop capability
        self._generation_phase: Optional[GenerationPhase] = None

    def request_stop(self):
        """
        Request graceful stop (SIGTERM/SIGINT).

        If generation is running, immediately cancels all workers.
        """
        logger.info("Stop requested")
        self.should_stop = True

        # Immediately cancel generation if running
        if self._generation_phase:
            self._generation_phase.request_stop()

    async def process_job(self, message_body: Dict[str, Any]) -> bool:
        """
        Process a job using the phase orchestrator.

        Returns True if job completed or paused successfully.
        """
        project_id_str = message_body.get("project_id")
        version_id_str = message_body.get("version_id")

        if not project_id_str or not version_id_str:
            logger.error("Invalid message: missing project_id or version_id")
            return False

        project_id = UUID(project_id_str)
        version_id = UUID(version_id_str)

        logger.info(f"🚀 Starting orchestrator: project={project_id}, version={version_id}")

        db: Session = self.SessionLocal()
        try:
            project = db.query(Project).filter(Project.id == project_id).first()
            if not project:
                logger.error(f"Project not found: {project_id}")
                return False

            version = db.query(ProjectVersion).filter(ProjectVersion.id == version_id).first()
            if not version:
                logger.error(f"Version not found: {version_id}")
                return False

            # Check version_id matches current - if not, this is a stale message
            if project.current_version_id != version_id:
                logger.warning(
                    f"Stale message (version mismatch): "
                    f"message version={version_id}, current version={project.current_version_id}"
                )
                # This is not an error - just an outdated message
                return True

            # Check version status - only process if it's in 'running' state
            if version.status not in ("running", "pause_requested"):
                logger.warning(
                    f"Version {version_id} is not in running state (status={version.status}), "
                    f"ignoring message"
                )
                return True

            # Update version to running if it was in pause_requested
            if version.status == "pause_requested":
                # This shouldn't happen normally, but handle gracefully
                logger.info("Version was in pause_requested, treating as pause")
                self._handle_pause_for_version(db, project, version, None, "Pause requested before start")
                return True

            # Set started_at if not already set
            if version.started_at is None:
                version.started_at = datetime.now(timezone.utc)
                db.commit()

            # Check if another project is already running for this user
            running_project = (
                db.query(Project)
                .filter(
                    Project.user_id == project.user_id,
                    Project.id != project_id,
                )
                .join(ProjectVersion, Project.current_version_id == ProjectVersion.id)
                .filter(ProjectVersion.status == "running")
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
            logger.info(f"  Version: {version.version_number}")
            logger.info(f"  Status: {version.status}")
            logger.info(f"  Target: {version.num_samples} samples")

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
                self._handle_force_stop_for_version(db, project, version, cost_tracker, stop_reason)
                return False

            # Initialize state with version_id
            state = ProjectState(db, project_id, version_id)

            # Create phases (all will work with the version-scoped state)
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
                assignment_phase=seed_assignment,
                parallel_samples=settings.generation_parallel_samples,
                cost_tracker=cost_tracker,
                stop_checker=lambda: self.should_stop,
            )

            # Store reference for immediate stop capability
            self._generation_phase = generation

            phases = [
                file_processing,
                seed_extraction,
                seed_scoring,
                seed_assignment,
                generation,
            ]

            # Emit RUNNING event
            self._emit_event(db, project, version, "running", "Worker started")

            # Main loop
            iteration = 0
            last_status_log = time.time()
            STATUS_LOG_INTERVAL = 30.0  # Log status every 30 seconds

            while True:
                iteration += 1

                # Refresh state from database
                state.refresh()

                # Check for pause FIRST - before any other processing
                if state.paused:
                    logger.info("⏸️  Pause detected")
                    # Request immediate stop from generation phase
                    generation.request_stop()
                    self._log_status(phases, cost_tracker)
                    self._handle_pause_for_version(db, project, version, cost_tracker)
                    return True

                # Check for external stop
                if self.should_stop:
                    logger.info("⏸️  Stop signal received")
                    # Request immediate stop from generation phase
                    generation.request_stop()
                    self._log_status(phases, cost_tracker)
                    self._handle_pause_for_version(db, project, version, cost_tracker, "Stopped by signal")
                    return False

                # Check balance and charge if needed
                can_continue, stop_reason = cost_tracker.check_balance_and_charge()
                if not can_continue:
                    logger.warning(f"💸 Stopping: {stop_reason}")
                    generation.request_stop()
                    self._log_status(phases, cost_tracker)
                    self._handle_force_stop_for_version(db, project, version, cost_tracker, stop_reason)
                    return False

                # Periodic status logging
                now = time.time()
                if now - last_status_log >= STATUS_LOG_INTERVAL:
                    self._log_status(phases, cost_tracker)
                    last_status_log = now

                # Find active phases
                active = [p for p in phases if p.should_run()]

                if not active:
                    # Check if all complete
                    if all(p.is_complete() for p in phases):
                        logger.info("✅ All phases complete")
                        self._log_status(phases, cost_tracker)  # Final status
                        self._handle_completion_for_version(db, project, version, cost_tracker)
                        return True
                    else:
                        # No work but not complete - wait with stop check
                        logger.debug(f"Iteration {iteration}: No active phases, waiting")
                        for _ in range(10):  # 10 x 0.1s = 1s total
                            if self.should_stop:
                                logger.info("⏸️  Stop signal received during wait")
                                generation.request_stop()
                                self._handle_pause_for_version(db, project, version, cost_tracker, "Stopped by signal")
                                return False
                            # Also check for pause during wait
                            state.refresh()
                            if state.paused:
                                logger.info("⏸️  Pause detected during wait")
                                generation.request_stop()
                                self._handle_pause_for_version(db, project, version, cost_tracker)
                                return True
                            await asyncio.sleep(0.1)
                        continue

                # Execute ONE unit from EACH active phase concurrently
                # BUT: We need to handle the case where pause/stop happens during execution
                tasks = [p.execute_once() for p in active]

                try:
                    # Use a shorter timeout and check for pause/stop more frequently
                    # For generation, the phase itself handles pause internally
                    results = await asyncio.wait_for(
                        asyncio.gather(*tasks, return_exceptions=True),
                        timeout=300.0
                    )
                except asyncio.TimeoutError:
                    logger.warning("Phase execution timeout (300s)")
                    # On timeout, check if we should stop
                    state.refresh()
                    if self.should_stop or state.paused:
                        logger.info("⏸️  Stop/pause detected after timeout")
                        generation.request_stop()
                        self._handle_pause_for_version(db, project, version, cost_tracker, "Stopped after timeout")
                        return self.should_stop  # False for external stop, True for pause
                    continue
                except asyncio.CancelledError:
                    # The task was cancelled (probably by pause/stop)
                    logger.info("Phase execution cancelled")
                    generation.request_stop()
                    state.refresh()
                    if state.paused:
                        self._handle_pause_for_version(db, project, version, cost_tracker)
                        return True
                    self._handle_pause_for_version(db, project, version, cost_tracker, "Cancelled")
                    return False

                # Process results and track costs
                for i, result in enumerate(results):
                    if isinstance(result, Exception):
                        logger.error(f"❌ Error in {active[i].name}: {result}")
                        raise result

                    if isinstance(result, PhaseResult) and result.cost_usd > 0:
                        cost_tracker.add_cost(
                            phase=active[i].name,
                            cost_usd=result.cost_usd,
                        )

                # Commit after each iteration
                db.commit()

                # Check for pause/stop AFTER execution too
                state.refresh()
                if state.paused:
                    logger.info("⏸️  Pause detected after phase execution")
                    generation.request_stop()
                    self._log_status(phases, cost_tracker)
                    self._handle_pause_for_version(db, project, version, cost_tracker)
                    return True

                if self.should_stop:
                    logger.info("⏸️  Stop signal received after phase execution")
                    generation.request_stop()
                    self._log_status(phases, cost_tracker)
                    self._handle_pause_for_version(db, project, version, cost_tracker, "Stopped by signal")
                    return False

                # Brief sleep to avoid tight loop
                await asyncio.sleep(0.01)

        except Exception as e:
            logger.exception(f"❌ Orchestrator error: {e}")

            # Make sure generation is stopped
            if self._generation_phase:
                self._generation_phase.request_stop()

            try:
                version = db.query(ProjectVersion).filter(ProjectVersion.id == version_id).first()
                project = db.query(Project).filter(Project.id == project_id).first()
                if version:
                    version.status = "failed"
                    version.error = str(e)
                    version.finished_at = datetime.now(timezone.utc)
                    if project:
                        self._emit_event(db, project, version, "failed", "Error", {"error": str(e)})
                    db.commit()
            except Exception as db_err:
                logger.error(f"Failed to update status: {db_err}")

            return False

        finally:
            # Clean up reference
            self._generation_phase = None
            db.close()

    def _emit_event(
        self,
        db: Session,
        project: Project,
        version: ProjectVersion,
        event_type: str,
        message: str,
        details: dict = None
    ) -> None:
        """Emit project event for a specific version."""
        event = ProjectEvent(
            project_id=project.id,
            version_id=version.id,
            event_type=event_type,
            message=message,
            details=details or {}
        )
        db.add(event)
        db.commit()

    def _handle_pause_for_version(
        self,
        db: Session,
        project: Project,
        version: ProjectVersion,
        cost_tracker: Optional[CostTracker],
        message: str = "Worker paused"
    ) -> None:
        """Handle pause: update version status, charge remaining costs, emit event."""
        logger.info(f"Pausing version {version.id} of project {project.id}")

        # Refresh to get accurate generated_count
        db.refresh(version)

        # Charge any remaining costs
        if cost_tracker:
            cost_tracker.charge_remaining()

        version.status = "paused"

        if cost_tracker:
            summary = cost_tracker.get_summary()
            self._emit_event(
                db, project, version, "paused",
                message,
                {
                    "paused_at": datetime.now(timezone.utc).isoformat(),
                    "generated_count": version.generated_count,
                    "total_cost_cents": summary["total_costs_cents"],
                    "preprocessing_cost_cents": summary["preprocessing_costs_cents"],
                    "generation_cost_cents": summary["generation_costs_cents"],
                }
            )
        else:
            self._emit_event(db, project, version, "paused", message)

        db.commit()
        logger.info("✅ Paused successfully")

    def _handle_completion_for_version(
        self,
        db: Session,
        project: Project,
        version: ProjectVersion,
        cost_tracker: CostTracker
    ) -> None:
        """Handle successful completion for a version."""
        logger.info(f"Version {version.id} of project {project.id} completed")

        # Refresh to get accurate generated_count
        db.refresh(version)

        # Charge any remaining costs
        cost_tracker.charge_remaining()

        version.status = "succeeded"
        version.finished_at = datetime.now(timezone.utc)

        summary = cost_tracker.get_summary()
        self._emit_event(
            db, project, version, "completed",
            "Generation complete",
            {
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "generated_count": version.generated_count,
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
            f"tokens: {summary.get('total_tokens', 0):,} "
            f"(in={summary.get('total_input_tokens', 0):,} out={summary.get('total_output_tokens', 0):,}), "
            f"balance: {summary['user_balance_cents']}¢"
        )

    def _handle_force_stop_for_version(
        self,
        db: Session,
        project: Project,
        version: ProjectVersion,
        cost_tracker: CostTracker,
        reason: str
    ) -> None:
        """
        Handle force-stop due to balance depletion or spend limit exceeded.

        Uses 'failed' status with descriptive error message.
        """
        logger.warning(f"Force-stopping version {version.id}: {reason}")

        # Refresh to get accurate generated_count
        db.refresh(version)

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

        # Update version
        version.status = "failed"
        version.error = error_msg
        version.finished_at = datetime.now(timezone.utc)

        # Build event details
        summary = cost_tracker.get_summary()
        details = {
            "reason": reason,
            "stopped_at": datetime.now(timezone.utc).isoformat(),
            "generated_count": version.generated_count,
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

        self._emit_event(db, project, version, "failed", error_msg, details)

        db.commit()
        logger.info(
            f"🛑 Force-stopped: {reason} "
            f"(preprocessing: {summary['preprocessing_costs_cents']}¢, "
            f"generation: {summary['generation_costs_cents']}¢, "
            f"tokens: {summary.get('total_tokens', 0):,})"
        )

    def _log_status(self, phases, cost_tracker) -> None:
        """Log current status of all phases."""
        parts = []
        for p in phases:
            s = p.get_status()
            if s.status == "complete":
                parts.append(f"✓{s.phase_name}")
            elif s.status == "active":
                parts.append(f"▶{s.phase_name}({s.progress})")
            else:
                parts.append(f"○{s.phase_name}")

        summary = cost_tracker.get_summary()
        spent = summary["cumulative_project_spend_cents"] / 100
        total_tokens = summary.get("total_tokens", 0)
        input_tokens = summary.get("total_input_tokens", 0)
        output_tokens = summary.get("total_output_tokens", 0)

        logger.info(
            f"📊 Status: {' | '.join(parts)} | "
            f"spent=${spent:.2f} | tokens={total_tokens:,} (in={input_tokens:,} out={output_tokens:,})"
        )