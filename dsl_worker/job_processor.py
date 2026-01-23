"""
Job processor for the DSL worker.

NEW ARCHITECTURE (v2):
- ResearchPhase: Intelligent seed collection (replaces extraction/scoring/assignment)
- GenerationPhase: Row generation with coverage gap awareness

Research and generation run concurrently:
- Research explores sources, extracts seeds
- Generation consumes seeds as they become available
"""

import asyncio
import logging
import os
import socket
import time
from datetime import datetime, timezone
from typing import Any, Dict, Optional
from uuid import UUID

from openai import AsyncOpenAI
from sqlalchemy.orm import Session
from azure.storage.blob import BlobServiceClient

from dsl_api.models.project import Project
from dsl_api.models.project_version import ProjectVersion
from dsl_api.models.project_event import ProjectEvent

from dsl_worker.config import settings
from dsl_worker.project_state import ProjectState
from dsl_worker.billing import CostTracker, TrackedOpenAIClient

from dsl_worker.phases.base import PhaseResult
from dsl_worker.phases.research import ResearchPhase
from dsl_worker.phases.generation_v2 import GenerationPhase

logger = logging.getLogger(__name__)

WORKER_ID = f"{socket.gethostname()}-{os.getpid()}"

# Minimum seeds before generation starts
MIN_SEEDS_BEFORE_GENERATION = int(os.getenv("MIN_SEEDS_BEFORE_GENERATION", "5"))

# Status log interval
STATUS_LOG_INTERVAL = 30.0


class JobProcessor:
    """
    Job processor using research-based seed collection.
    
    Two phases run concurrently:
    1. Research: Explores sources, extracts seeds intelligently
    2. Generation: Pulls seeds, generates rows with coverage awareness
    
    Generation starts as soon as seeds are available (non-blocking).
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

        # Phase references for stop capability
        self._research_phase: Optional[ResearchPhase] = None
        self._generation_phase: Optional[GenerationPhase] = None

    def request_stop(self):
        """Request graceful stop (SIGTERM/SIGINT)."""
        logger.warning(f"⚠️ Stop requested on worker {WORKER_ID}")
        self.should_stop = True

        if self._generation_phase:
            self._generation_phase._stop_requested = True
        # Research phase checks should_stop via stop_checker

    async def process_job(self, message_body: Dict[str, Any]) -> bool:
        """Process a job using research + generation pipeline."""
        project_id_str = message_body.get("project_id")
        version_id_str = message_body.get("version_id")

        if not project_id_str or not version_id_str:
            logger.error("Invalid message: missing project_id or version_id")
            return False

        project_id = UUID(project_id_str)
        version_id = UUID(version_id_str)

        logger.info(
            f"🚀 Starting job: project={project_id}, version={version_id}, worker={WORKER_ID}"
        )

        db: Session = self.SessionLocal()
        try:
            # Validate project and version
            project = db.query(Project).filter(Project.id == project_id).first()
            if not project:
                logger.error(f"Project not found: {project_id}")
                return False

            version = db.query(ProjectVersion).filter(ProjectVersion.id == version_id).first()
            if not version:
                logger.error(f"Version not found: {version_id}")
                return False

            # Check for stale message
            if project.current_version_id != version_id:
                logger.warning(f"Stale message: version mismatch")
                return True

            # Check version status
            if version.status not in ("running", "pause_requested"):
                logger.warning(f"Version not in running state: {version.status}")
                return True

            if version.status == "pause_requested":
                self._handle_pause_for_version(db, project, version, None, "Pause requested before start")
                return True

            # Set started_at
            if version.started_at is None:
                version.started_at = datetime.now(timezone.utc)
                db.commit()

            logger.info(f"Project: {project.name}")
            logger.info(f"  Version: {version.version_number}")
            logger.info(f"  Target: {version.num_samples} samples")

            # Create tracked client and cost tracker
            tracked_client = TrackedOpenAIClient(self.raw_openai_client)
            
            cost_tracker = CostTracker(
                db=db,
                user_id=project.user_id,
                project_id=project_id,
                margin_multiplier=settings.billing_margin_multiplier,
                charge_threshold_cents=settings.billing_charge_threshold_cents,
                charge_interval_seconds=settings.billing_charge_interval_seconds,
                spend_limit_cents=project.spend_limit_cents,
            )

            # Check initial balance
            can_continue, stop_reason = cost_tracker.check_balance_and_charge()
            if not can_continue:
                logger.warning(f"Cannot start: {stop_reason}")
                self._handle_force_stop_for_version(db, project, version, cost_tracker, stop_reason)
                return False

            # Initialize state
            state = ProjectState(db, project_id, version_id)

            # Create phases
            self._research_phase = ResearchPhase(
                name="research",
                state=state,
                db=db,
                openai_client=tracked_client,
                blob_service_client=self.blob_service_client,
                stop_checker=lambda: self.should_stop or state.paused,
                cost_tracker=cost_tracker,
            )

            self._generation_phase = GenerationPhase(
                name="generation",
                state=state,
                db=db,
                openai_client=tracked_client,
                research_phase=self._research_phase,
                parallel_samples=settings.generation_parallel_samples,
                stop_checker=lambda: self.should_stop or state.paused,
                cost_tracker=cost_tracker,
            )

            # Emit running event
            self._emit_event(db, project, version, "running", "Worker started")

            # Run pipeline
            result = await self._run_pipeline(db, project, version, state, cost_tracker)
            return result

        except Exception as e:
            logger.exception(f"❌ Job error: {e}")
            
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
            # Cleanup browser pool
            if self._research_phase:
                try:
                    await self._research_phase.cleanup()
                except Exception as e:
                    logger.warning(f"Research cleanup error: {e}")
                    
            self._research_phase = None
            self._generation_phase = None
            db.close()

    async def _run_pipeline(
        self,
        db: Session,
        project: Project,
        version: ProjectVersion,
        state: ProjectState,
        cost_tracker: CostTracker,
    ) -> bool:
        """Run research and generation concurrently."""
        
        research_done = asyncio.Event()
        stop_reason: Optional[str] = None
        last_status_log = time.time()
        
        async def research_loop():
            """Run research until complete or stopped."""
            nonlocal stop_reason
            
            try:
                while not self.should_stop:
                    state.refresh()
                    if state.paused:
                        return "paused"
                        
                    if self._research_phase.is_complete():
                        logger.info("✅ Research complete")
                        return "complete"
                        
                    if not self._research_phase.should_run():
                        await asyncio.sleep(1)
                        continue
                        
                    try:
                        result = await self._research_phase.execute_once()
                        
                        # Track cost
                        if result.cost_usd > 0:
                            cost_tracker.add_cost(phase="research", cost_usd=result.cost_usd)
                        
                        # Check balance
                        can_continue, reason = cost_tracker.check_balance_and_charge()
                        if not can_continue:
                            stop_reason = reason
                            return f"stopped:{reason}"
                            
                    except Exception as e:
                        logger.error(f"Research error: {e}")
                        await asyncio.sleep(2)
                        
                    if not result.did_work:
                        await asyncio.sleep(1)
                        
                return "stopped"
                
            finally:
                research_done.set()
                
        async def generation_loop():
            """Run generation, waiting for seeds."""
            nonlocal stop_reason
            
            while not self.should_stop:
                state.refresh()
                if state.paused:
                    return "paused"
                    
                if self._generation_phase.is_complete():
                    logger.info("✅ Generation complete")
                    return "complete"
                    
                # Wait for minimum seeds
                seed_count = self._research_phase.get_seed_count()
                if seed_count < MIN_SEEDS_BEFORE_GENERATION:
                    if research_done.is_set():
                        if seed_count == 0:
                            logger.warning("No seeds collected - cannot generate")
                            return "no_seeds"
                        # Proceed with what we have
                        logger.info(f"Research done, proceeding with {seed_count} seeds")
                    else:
                        logger.debug(f"Waiting for seeds: {seed_count}/{MIN_SEEDS_BEFORE_GENERATION}")
                        await asyncio.sleep(2)
                        continue
                        
                if not self._generation_phase.should_run():
                    if research_done.is_set():
                        return "complete"
                    await asyncio.sleep(1)
                    continue
                    
                try:
                    result = await self._generation_phase.execute_once()
                    
                    # Track cost
                    if result.cost_usd > 0:
                        cost_tracker.add_cost(phase="generation", cost_usd=result.cost_usd)
                    
                    # Check balance
                    can_continue, reason = cost_tracker.check_balance_and_charge()
                    if not can_continue:
                        stop_reason = reason
                        return f"stopped:{reason}"
                        
                except Exception as e:
                    logger.error(f"Generation error: {e}")
                    await asyncio.sleep(2)
                    
                if not result.did_work:
                    if research_done.is_set():
                        return "complete"
                    await asyncio.sleep(2)
                    
            return "stopped"
            
        async def status_loop():
            """Periodic status logging."""
            nonlocal last_status_log
            
            while True:
                state.refresh()
                if state.paused:
                    return
                    
                now = time.time()
                if now - last_status_log >= STATUS_LOG_INTERVAL:
                    self._log_status(cost_tracker)
                    last_status_log = now
                    
                # Check if done
                if self._generation_phase.is_complete():
                    return
                if research_done.is_set() and not self._generation_phase.should_run():
                    return
                    
                await asyncio.sleep(5)

        # Run all loops
        try:
            results = await asyncio.gather(
                research_loop(),
                generation_loop(),
                status_loop(),
                return_exceptions=True,
            )
        except asyncio.CancelledError:
            logger.info("Pipeline cancelled")
            state.refresh()
            if state.paused:
                self._handle_pause_for_version(db, project, version, cost_tracker)
                return True
            self._handle_pause_for_version(db, project, version, cost_tracker, "Cancelled")
            return False

        # Final status log
        self._log_status(cost_tracker)
        
        # Parse results
        research_result = results[0] if not isinstance(results[0], Exception) else f"error:{results[0]}"
        generation_result = results[1] if not isinstance(results[1], Exception) else f"error:{results[1]}"
        
        logger.info(f"Pipeline results: research={research_result}, generation={generation_result}")
        
        # Handle outcomes
        if "paused" in [research_result, generation_result]:
            self._handle_pause_for_version(db, project, version, cost_tracker)
            return True
            
        if generation_result == "no_seeds":
            self._handle_force_stop_for_version(
                db, project, version, cost_tracker, 
                "No seeds could be collected for this dataset"
            )
            return False
            
        if str(research_result).startswith("stopped:") or str(generation_result).startswith("stopped:"):
            reason = stop_reason or "unknown"
            self._handle_force_stop_for_version(db, project, version, cost_tracker, reason)
            return False
            
        if str(research_result).startswith("error:") or str(generation_result).startswith("error:"):
            error = research_result if str(research_result).startswith("error:") else generation_result
            raise Exception(str(error).split(":", 1)[-1])
            
        # Success
        if generation_result == "complete":
            self._handle_completion_for_version(db, project, version, cost_tracker)
            return True
            
        # Fallback
        logger.warning(f"Unexpected pipeline end state: research={research_result}, generation={generation_result}")
        return True

    def _log_status(self, cost_tracker: CostTracker):
        """Log current status."""
        research_seeds = self._research_phase.get_seed_count() if self._research_phase else 0
        research_target = self._research_phase.target_seeds if self._research_phase else 0
        gen_count = self._generation_phase.state.samples_generated if self._generation_phase else 0
        gen_target = self._generation_phase.state.num_samples if self._generation_phase else 0
        
        summary = cost_tracker.get_summary()
        spent = summary["cumulative_project_spend_cents"] / 100
        total_tokens = summary.get("total_tokens", 0)
        
        logger.info(
            f"📊 Status: seeds={research_seeds}/{research_target} | "
            f"samples={gen_count}/{gen_target} | "
            f"spent=${spent:.2f} | tokens={total_tokens:,}"
        )

    def _emit_event(
        self,
        db: Session,
        project: Project,
        version: ProjectVersion,
        event_type: str,
        message: str,
        details: dict = None
    ) -> None:
        """Emit project event."""
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
        """Handle pause."""
        logger.info(f"Pausing version {version.id}")
        db.refresh(version)

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
                    "seeds_collected": self._research_phase.get_seed_count() if self._research_phase else 0,
                    "total_cost_cents": summary["total_costs_cents"],
                    "preprocessing_cost_cents": summary.get("preprocessing_costs_cents", 0),
                    "generation_cost_cents": summary.get("generation_costs_cents", 0),
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
        """Handle successful completion."""
        logger.info(f"Version {version.id} completed")
        db.refresh(version)

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
                "seeds_collected": self._research_phase.get_seed_count() if self._research_phase else 0,
                "total_cost_cents": summary["total_costs_cents"],
                "cumulative_project_spend_cents": summary["cumulative_project_spend_cents"],
            }
        )

        db.commit()
        logger.info(
            f"✅ Completed: {version.generated_count} samples, "
            f"${summary['total_costs_cents']/100:.2f} total"
        )

    def _handle_force_stop_for_version(
        self,
        db: Session,
        project: Project,
        version: ProjectVersion,
        cost_tracker: CostTracker,
        reason: str
    ) -> None:
        """Handle force-stop due to balance/limit/error."""
        logger.warning(f"Force-stopping version {version.id}: {reason}")
        db.refresh(version)

        cost_tracker.charge_remaining()

        if reason == "insufficient_balance":
            error_msg = "Insufficient balance to continue"
        elif reason == "spend_limit_exceeded":
            limit = project.spend_limit_cents
            error_msg = f"Spend limit of ${limit/100:.2f} exceeded" if limit else "Spend limit exceeded"
        elif "seeds" in reason.lower():
            error_msg = "Could not collect seeds for this dataset"
        else:
            error_msg = f"Stopped: {reason}"

        version.status = "failed"
        version.error = error_msg
        version.finished_at = datetime.now(timezone.utc)

        summary = cost_tracker.get_summary()
        self._emit_event(
            db, project, version, "failed",
            error_msg,
            {
                "reason": reason,
                "stopped_at": datetime.now(timezone.utc).isoformat(),
                "generated_count": version.generated_count,
                "seeds_collected": self._research_phase.get_seed_count() if self._research_phase else 0,
                "total_cost_cents": summary["total_costs_cents"],
                "final_balance_cents": summary["user_balance_cents"],
            }
        )

        db.commit()
        logger.info(f"🛑 Force-stopped: {reason}")