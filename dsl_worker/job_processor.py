"""
Job Processor - Simplified pipeline

Pipeline:
1. Create root scope from user spec
2. Run scope processor (recursive research/breakdown) - each scope gets own browser
3. Run generation workers on seeds
"""

import asyncio
import logging
import os
import socket
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
from dsl_worker.checkpoint import CheckpointManager, checkpoints_to_seeds

from dsl_worker.phases.scope_processor import ScopeProcessor, Scope
from dsl_worker.phases.row_generator import GenerationWorkerPool
from dsl_worker.phases.sandbox import SandboxExecutor

logger = logging.getLogger(__name__)

WORKER_ID = f"{socket.gethostname()}-{os.getpid()}"


class JobProcessor:
    """
    Job processor for the pipeline.
    
    Each scope creates its own browser (no shared browser).
    Checkpoints are saved to Azure Blob for pause/resume.
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
        
        # Sandbox (shared - it's just a Docker pool)
        self._sandbox: Optional[SandboxExecutor] = None
    
    def request_stop(self):
        """Request graceful stop."""
        logger.warning(f"[Worker] Stop requested")
        self.should_stop = True
    
    def _make_stop_checker(self, state: ProjectState):
        """Create a stop checker that refreshes state before checking."""
        def checker():
            state.refresh()
            return self.should_stop or state.paused
        return checker
    
    async def process_job(self, message_body: Dict[str, Any]) -> bool:
        """Process a job."""
        
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
            
            # Emit running event
            self._emit_event(db, project, version, "running", "Worker started")
            
            # Run pipeline
            result = await self._run_pipeline(
                db, project, version, state, tracked_client, cost_tracker
            )
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
            await self._cleanup()
            db.close()
    
    async def _run_pipeline(
        self,
        db: Session,
        project: Project,
        version: ProjectVersion,
        state: ProjectState,
        tracked_client: Any,
        cost_tracker: CostTracker,
    ) -> bool:
        """Run the main pipeline."""
        
        # Create workspace
        workspace_dir = Path(f"./workspace_{project.id}_{version.id}")
        workspace_dir.mkdir(parents=True, exist_ok=True)
        (workspace_dir / "uploads").mkdir(exist_ok=True)
        (workspace_dir / "downloads").mkdir(exist_ok=True)
        
        # Load uploaded files from blob
        await self._load_uploaded_files(state, workspace_dir)
        
        # Build files metadata for research agent
        files_metadata = []
        for f in state.files_snapshot:
            filename = f.get('filename', '')
            size_bytes = f.get('size_bytes', 0)
            if filename:
                size_str = f"{size_bytes / 1024:.1f}KB" if size_bytes else ""
                files_metadata.append(f"{filename} ({size_str})" if size_str else filename)
        
        # Initialize sandbox (shared across scopes)
        self._sandbox = SandboxExecutor(use_pool=True, pool_size=3)
        
        # Initialize checkpoint manager
        checkpoint_mgr = CheckpointManager(
            blob_service_client=self.blob_service_client,
            container_name=settings.azure_storage_container_name,
            project_id=project.id,
            version_id=version.id,
        )
        checkpoint = await checkpoint_mgr.initialize()
        
        # Check if resuming
        seeds = []
        if checkpoint.current_phase == "generation" and checkpoint.seeds:
            # Resume from generation phase
            logger.info(
                f"[Pipeline] Resuming generation: "
                f"{len(checkpoint.processed_seed_indices)}/{len(checkpoint.seeds)} done"
            )
            seeds = checkpoints_to_seeds(checkpoint.seeds)
            
        elif checkpoint.current_phase == "research":
            # Run or resume research phase
            logger.info("[Pipeline] Starting/resuming research phase...")
            
            # Create root scope
            first_line = state.generation_prompt.strip().split('\n')[0][:100]
            root_description = first_line if first_line else "Root scope"
            
            root_scope = Scope(
                description=root_description,
                quota=state.num_samples,
                notes=[],
            )
            
            # Run scope processor
            # NOTE: No browser passed in - each scope creates its own
            scope_processor = ScopeProcessor(
                workspace_dir=workspace_dir,
                schema=state.columns,
                project_instructions=state.generation_prompt,
                openai_client=tracked_client,
                brave_api_key=settings.brave_api_key,
                sandbox=self._sandbox,
                files_metadata=files_metadata,
                stop_checker=self._make_stop_checker(state),
            )
            
            try:
                seeds = await scope_processor.process(root_scope)
            except Exception as e:
                logger.error(f"[Pipeline] Scope processor failed: {e}")
                raise
            
            logger.info(f"[Pipeline] Research complete: {len(seeds)} seeds")
            
            # Track scope processor cost
            if scope_processor.get_total_cost() > 0:
                cost_tracker.add_cost(
                    phase="scope_processor",
                    cost_usd=scope_processor.get_total_cost(),
                    model=settings.research_model,
                )
                await checkpoint_mgr.add_cost(scope_processor.get_total_cost())
            
            # Save seeds to checkpoint
            await checkpoint_mgr.add_seeds(seeds)
            await checkpoint_mgr.set_phase("generation")
        
        # Check if we should continue
        state.refresh()
        if state.paused:
            await checkpoint_mgr.force_save()
            self._handle_pause(db, project, version, cost_tracker)
            return True
        
        can_continue, stop_reason = cost_tracker.check_balance_and_charge()
        if not can_continue:
            await checkpoint_mgr.force_save()
            self._handle_force_stop(db, project, version, cost_tracker, stop_reason)
            return False
        
        if not seeds:
            logger.warning("[Pipeline] No seeds to process")
            self._handle_force_stop(db, project, version, cost_tracker, "No seeds generated")
            return False
        
        # Filter to pending seeds only
        pending_indices = checkpoint.get_pending_seed_indices()
        pending_seeds = [seeds[i] for i in pending_indices]
        
        logger.info(
            f"[Pipeline] Starting generation: "
            f"{len(pending_seeds)} pending of {len(seeds)} total"
        )
        
        # Run generation
        generation_pool = GenerationWorkerPool(
            workspace_dir=workspace_dir,
            openai_client=tracked_client,
            db_session=db,
            project_id=project.id,
            version_id=version.id,
            brave_api_key=settings.brave_api_key,
            browser=None,  # Each generator can create its own if needed
            sandbox=self._sandbox,
            num_workers=settings.generation_parallel_samples,
            stop_checker=self._make_stop_checker(state),
            cost_tracker=cost_tracker,
            checkpoint_callback=lambda idx, success, row_id: asyncio.create_task(
                checkpoint_mgr.mark_seed_processed(pending_indices[idx], success, row_id)
            ),
        )
        
        total_success, total_errors = await generation_pool.process_seeds(
            pending_seeds, state.columns
        )
        
        # Check balance after generation
        can_continue, stop_reason = cost_tracker.check_balance_and_charge()
        if not can_continue:
            await checkpoint_mgr.force_save()
            self._handle_force_stop(db, project, version, cost_tracker, stop_reason)
            return False
        
        # Log stats
        stats = generation_pool.get_stats()
        logger.info(
            f"[Pipeline] Generation complete: "
            f"{stats['rows_generated']} rows, {stats['errors']} errors"
        )
        
        # Track generation cost
        if stats['total_cost'] > 0:
            cost_tracker.add_cost(
                phase="generation",
                cost_usd=stats['total_cost'],
                model=settings.generation_model,
            )
            await checkpoint_mgr.add_cost(stats['total_cost'])
        
        # Handle final state
        state.refresh()
        if state.paused:
            await checkpoint_mgr.force_save()
            self._handle_pause(db, project, version, cost_tracker)
            return True
        
        # Calculate total success (previous + this run)
        total_processed = len(checkpoint.processed_seed_indices) + total_success
        
        if total_processed >= state.num_samples or total_processed >= len(seeds):
            # Success - delete checkpoint
            await checkpoint_mgr.delete()
            self._handle_completion(db, project, version, cost_tracker)
            return True
        
        if total_success > 0:
            logger.warning(f"[Pipeline] Partial completion: {total_processed}/{state.num_samples}")
            await checkpoint_mgr.delete()
            self._handle_completion(db, project, version, cost_tracker)
            return True
        
        # No rows generated
        await checkpoint_mgr.force_save()
        self._handle_force_stop(db, project, version, cost_tracker, "No rows generated")
        return False
    
    async def _load_uploaded_files(self, state: ProjectState, workspace_dir: Path):
        """Download uploaded files from Azure to workspace."""
        if not state.files_snapshot:
            logger.info("[Pipeline] No uploaded files")
            return
        
        logger.info(f"[Pipeline] Loading {len(state.files_snapshot)} uploaded files...")
        
        uploads_dir = workspace_dir / "uploads"
        
        for f in state.files_snapshot:
            filename = f.get('filename')
            blob_path = f.get('blob_path')
            
            if not filename or not blob_path:
                continue
            
            local_path = uploads_dir / filename
            
            try:
                if self.blob_service_client:
                    container = settings.azure_storage_container_name
                    blob_client = self.blob_service_client.get_blob_client(
                        container=container,
                        blob=blob_path
                    )
                    with open(local_path, "wb") as file:
                        file.write(blob_client.download_blob().readall())
                    logger.info(f"[Pipeline] Downloaded: {filename}")
                else:
                    logger.warning(f"[Pipeline] No blob client, can't download {filename}")
            except Exception as e:
                logger.error(f"[Pipeline] Failed to download {filename}: {e}")
    
    async def _cleanup(self):
        """Cleanup resources."""
        # Note: No browser to cleanup - each scope cleans up its own
        
        if self._sandbox:
            try:
                self._sandbox.close()
            except Exception as e:
                logger.warning(f"Sandbox cleanup error: {e}")
            self._sandbox = None
    
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