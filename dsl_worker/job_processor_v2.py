"""
Job Processor v2 - Orchestrates the research pipeline.

Pipeline:
1. Research Agent - explores sources, dispatches to extraction
2. Seed Extraction - processes files into seeds (source + note)
3. Seed Assignment - assigns seeds to diversity slots (batch ranking)
4. Generation - generates rows from assigned seeds

Concurrent execution with feedback loops.
"""

import asyncio
import logging
import os
import socket
import time
from datetime import datetime, timezone
from pathlib import Path
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

from dsl_worker.phases.research_v2 import ResearchPhaseV2
from dsl_worker.phases.seed_extraction_v2 import SeedExtractionPhase
from dsl_worker.phases.seed_assignment import SeedAssignmentPhase
from dsl_worker.phases.generation_v3 import GenerationPhaseV3

logger = logging.getLogger(__name__)

WORKER_ID = f"{socket.gethostname()}-{os.getpid()}"
STATUS_LOG_INTERVAL = 30.0


class JobProcessorV2:
    """
    Job processor using the research pipeline.
    
    Runs concurrently:
    - Research loop (explores, dispatches)
    - Extraction workers (process queue)
    - Assignment loop (sequential)
    - Generation workers (starts at threshold)
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
        
        # Phase references
        self._research_phase: Optional[ResearchPhaseV2] = None
        self._extraction_phase: Optional[SeedExtractionPhase] = None
        self._assignment_phase: Optional[SeedAssignmentPhase] = None
        self._generation_phase: Optional[GenerationPhaseV3] = None
    
    def request_stop(self):
        """Request graceful stop."""
        logger.warning(f"[Worker] Stop requested")
        self.should_stop = True
        
        if self._generation_phase:
            self._generation_phase._stop_requested = True
    
    async def process_job(self, message_body: Dict[str, Any]) -> bool:
        """Process a job using the pipeline."""
        
        project_id_str = message_body.get("project_id")
        version_id_str = message_body.get("version_id")
        
        if not project_id_str or not version_id_str:
            logger.error("Invalid message: missing IDs")
            return False
        
        project_id = UUID(project_id_str)
        version_id = UUID(version_id_str)
        
        logger.info(f"🚀 Starting job: project={project_id}, version={version_id}")
        
        db: Session = self.SessionLocal()
        
        try:
            # Validate
            project = db.query(Project).filter(Project.id == project_id).first()
            if not project:
                logger.error(f"Project not found: {project_id}")
                return False
            
            version = db.query(ProjectVersion).filter(ProjectVersion.id == version_id).first()
            if not version:
                logger.error(f"Version not found: {version_id}")
                return False
            
            # Check stale
            if project.current_version_id != version_id:
                logger.warning("Stale message")
                return True
            
            # Check status
            if version.status not in ("running", "pause_requested"):
                logger.warning(f"Version not running: {version.status}")
                return True
            
            if version.status == "pause_requested":
                self._handle_pause(db, project, version, None, "Pause before start")
                return True
            
            # Set started
            if version.started_at is None:
                version.started_at = datetime.now(timezone.utc)
                db.commit()
            
            logger.info(f"Project: {project.name}, Version: {version.version_number}")
            logger.info(f"Target: {version.num_samples} samples")
            
            # Create tracked client
            tracked_client = TrackedOpenAIClient(self.raw_openai_client)
            
            # Create cost tracker
            cost_tracker = CostTracker(
                db=db,
                user_id=project.user_id,
                project_id=project_id,
                margin_multiplier=settings.billing_margin_multiplier,
                charge_threshold_cents=settings.billing_charge_threshold_cents,
                charge_interval_seconds=settings.billing_charge_interval_seconds,
                spend_limit_cents=project.spend_limit_cents,
            )
            
            # Check balance
            can_continue, stop_reason = cost_tracker.check_balance_and_charge()
            if not can_continue:
                self._handle_force_stop(db, project, version, cost_tracker, stop_reason)
                return False
            
            # Initialize state
            state = ProjectState(db, project_id, version_id)
            
            # Create workspace
            workspace_dir = Path(f"./workspace_{project_id}_{version_id}")
            workspace_dir.mkdir(parents=True, exist_ok=True)
            (workspace_dir / "uploads").mkdir(exist_ok=True)
            (workspace_dir / "web").mkdir(exist_ok=True)
            (workspace_dir / "extracted").mkdir(exist_ok=True)
            
            # Create extraction queue
            extraction_queue = asyncio.Queue()
            
            # Create phases
            self._extraction_phase = SeedExtractionPhase(
                name="extraction",
                state=state,
                db=db,
                openai_client=tracked_client,
                extraction_queue=extraction_queue,
                stop_checker=lambda: self.should_stop or state.paused,
                cost_tracker=cost_tracker,
            )
            
            self._assignment_phase = SeedAssignmentPhase(
                name="assignment",
                state=state,
                db=db,
                openai_client=tracked_client,
                stop_checker=lambda: self.should_stop or state.paused,
                cost_tracker=cost_tracker,
            )
            
            # Feedback callback for research agent
            def get_feedback() -> Dict:
                extraction_stats = self._extraction_phase.get_stats()
                assignment_stats = self._assignment_phase.get_stats()
                return {
                    "total_seeds": extraction_stats["total_seeds"],
                    "avg_quality": extraction_stats["avg_quality"],
                    "remaining_quotas": assignment_stats["remaining_quotas"],
                    "total_remaining": assignment_stats["total_remaining"],
                }
            
            self._research_phase = ResearchPhaseV2(
                name="research",
                state=state,
                db=db,
                openai_client=tracked_client,
                blob_service_client=self.blob_service_client,
                extraction_queue=extraction_queue,
                feedback_callback=get_feedback,
                stop_checker=lambda: self.should_stop or state.paused,
                cost_tracker=cost_tracker,
                workspace_dir=str(workspace_dir),
                browser_pool_size=int(os.getenv("BROWSER_POOL_SIZE", "5")),
            )
            
            self._generation_phase = GenerationPhaseV3(
                name="generation",
                state=state,
                db=db,
                openai_client=tracked_client,
                assignment_phase=self._assignment_phase,
                workspace_dir=str(workspace_dir),
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
            # Cleanup
            if self._research_phase:
                try:
                    await self._research_phase.cleanup()
                except Exception as e:
                    logger.warning(f"Research cleanup error: {e}")
            
            self._research_phase = None
            self._extraction_phase = None
            self._assignment_phase = None
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
        """Run all pipeline loops concurrently."""
        
        stop_reason: Optional[str] = None
        last_status_log = time.time()
        
        async def research_loop():
            """Research agent loop."""
            nonlocal stop_reason
            
            while not self.should_stop:
                state.refresh()
                if state.paused:
                    return "paused"
                
                # Check if quotas filled
                if self._assignment_phase.is_complete():
                    logger.info("✅ All quotas filled, research done")
                    return "complete"
                
                if not self._research_phase.should_run():
                    await asyncio.sleep(2)
                    continue
                
                try:
                    result = await self._research_phase.execute_once()
                    
                    can_continue, reason = cost_tracker.check_balance_and_charge()
                    if not can_continue:
                        stop_reason = reason
                        return f"stopped:{reason}"
                        
                except Exception as e:
                    logger.error(f"Research error: {e}")
                    await asyncio.sleep(2)
                
                if not result.did_work:
                    await asyncio.sleep(2)
            
            return "stopped"
        
        async def extraction_loop():
            """Extraction worker loop."""
            nonlocal stop_reason
            
            while not self.should_stop:
                state.refresh()
                if state.paused:
                    return "paused"
                
                if not self._extraction_phase.should_run():
                    if self._assignment_phase.is_complete():
                        return "complete"
                    await asyncio.sleep(1)
                    continue
                
                try:
                    result = await self._extraction_phase.execute_once()
                    
                    can_continue, reason = cost_tracker.check_balance_and_charge()
                    if not can_continue:
                        stop_reason = reason
                        return f"stopped:{reason}"
                        
                except Exception as e:
                    logger.error(f"Extraction error: {e}")
                    await asyncio.sleep(2)
                
                if not result.did_work:
                    await asyncio.sleep(1)
            
            return "stopped"
        
        async def assignment_loop():
            """Assignment loop (sequential)."""
            nonlocal stop_reason
            
            while not self.should_stop:
                state.refresh()
                if state.paused:
                    return "paused"
                
                if self._assignment_phase.is_complete():
                    return "complete"
                
                if not self._assignment_phase.should_run():
                    await asyncio.sleep(1)
                    continue
                
                try:
                    result = await self._assignment_phase.execute_once()
                    
                    can_continue, reason = cost_tracker.check_balance_and_charge()
                    if not can_continue:
                        stop_reason = reason
                        return f"stopped:{reason}"
                        
                except Exception as e:
                    logger.error(f"Assignment error: {e}")
                    await asyncio.sleep(2)
                
                if not result.did_work:
                    await asyncio.sleep(1)
            
            return "stopped"
        
        async def generation_loop():
            """Generation worker loop."""
            nonlocal stop_reason
            
            while not self.should_stop:
                state.refresh()
                if state.paused:
                    return "paused"
                
                if self._generation_phase.is_complete():
                    logger.info("✅ Generation complete")
                    return "complete"
                
                if not self._generation_phase.should_run():
                    await asyncio.sleep(2)
                    continue
                
                try:
                    result = await self._generation_phase.execute_once()
                    
                    can_continue, reason = cost_tracker.check_balance_and_charge()
                    if not can_continue:
                        stop_reason = reason
                        return f"stopped:{reason}"
                        
                except Exception as e:
                    logger.error(f"Generation error: {e}")
                    await asyncio.sleep(2)
                
                if not result.did_work:
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
                
                if self._generation_phase.is_complete():
                    return
                
                await asyncio.sleep(5)
        
        # Run all loops
        try:
            results = await asyncio.gather(
                research_loop(),
                extraction_loop(),
                assignment_loop(),
                generation_loop(),
                status_loop(),
                return_exceptions=True,
            )
        except asyncio.CancelledError:
            logger.info("Pipeline cancelled")
            state.refresh()
            if state.paused:
                self._handle_pause(db, project, version, cost_tracker)
                return True
            self._handle_pause(db, project, version, cost_tracker, "Cancelled")
            return False
        
        # Final status
        self._log_status(cost_tracker)
        
        # Parse results
        research_result = results[0] if not isinstance(results[0], Exception) else f"error:{results[0]}"
        extraction_result = results[1] if not isinstance(results[1], Exception) else f"error:{results[1]}"
        assignment_result = results[2] if not isinstance(results[2], Exception) else f"error:{results[2]}"
        generation_result = results[3] if not isinstance(results[3], Exception) else f"error:{results[3]}"
        
        logger.info(
            f"Pipeline results: research={research_result}, extraction={extraction_result}, "
            f"assignment={assignment_result}, generation={generation_result}"
        )
        
        # Handle outcomes
        if "paused" in [research_result, extraction_result, assignment_result, generation_result]:
            self._handle_pause(db, project, version, cost_tracker)
            return True
        
        if any(str(r).startswith("stopped:") for r in [research_result, extraction_result, assignment_result, generation_result]):
            reason = stop_reason or "unknown"
            self._handle_force_stop(db, project, version, cost_tracker, reason)
            return False
        
        if any(str(r).startswith("error:") for r in [research_result, extraction_result, assignment_result, generation_result]):
            errors = [r for r in [research_result, extraction_result, assignment_result, generation_result]
                     if str(r).startswith("error:")]
            raise Exception(str(errors[0]).split(":", 1)[-1])
        
        # Success
        if generation_result == "complete":
            self._handle_completion(db, project, version, cost_tracker)
            return True
        
        logger.warning(f"Unexpected end state")
        return True
    
    def _log_status(self, cost_tracker: CostTracker):
        """Log current status."""
        extraction_stats = self._extraction_phase.get_stats() if self._extraction_phase else {}
        assignment_stats = self._assignment_phase.get_stats() if self._assignment_phase else {}
        gen_count = self._generation_phase.state.samples_generated if self._generation_phase else 0
        gen_target = self._generation_phase.state.num_samples if self._generation_phase else 0
        sources = len(self._research_phase._sources) if self._research_phase else 0
        
        summary = cost_tracker.get_summary()
        spent = summary["cumulative_project_spend_cents"] / 100
        
        logger.info(
            f"📊 Status: sources={sources} | "
            f"seeds={extraction_stats.get('total_seeds', 0)} "
            f"(quality={extraction_stats.get('avg_quality', 0):.1f}) | "
            f"assigned={assignment_stats.get('assigned_count', 0)} | "
            f"samples={gen_count}/{gen_target} | "
            f"spent=${spent:.2f}"
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
    
    def _handle_pause(
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
        
        details = {"paused_at": datetime.now(timezone.utc).isoformat()}
        if cost_tracker:
            summary = cost_tracker.get_summary()
            details["total_cost_cents"] = summary["total_costs_cents"]
        
        self._emit_event(db, project, version, "paused", message, details)
        db.commit()
    
    def _handle_completion(
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
                "total_cost_cents": summary["total_costs_cents"],
            }
        )
        
        db.commit()
        logger.info(f"✅ Completed: {version.generated_count} samples")
    
    def _handle_force_stop(
        self,
        db: Session,
        project: Project,
        version: ProjectVersion,
        cost_tracker: CostTracker,
        reason: str
    ) -> None:
        """Handle force-stop."""
        logger.warning(f"Force-stopping: {reason}")
        db.refresh(version)
        
        cost_tracker.charge_remaining()
        
        version.status = "failed"
        version.error = reason
        version.finished_at = datetime.now(timezone.utc)
        
        self._emit_event(
            db, project, version, "failed",
            reason,
            {"reason": reason}
        )
        
        db.commit()