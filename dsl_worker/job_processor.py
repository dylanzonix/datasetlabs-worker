"""
Job Processor - Orchestrator-driven pipeline

Pipeline:
1. Load conversation history
2. Run orchestrator (research, generators, recipe)
3. Run generation workers on seeds from queue
"""

import asyncio
import json
import logging
import os
import socket
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import UUID

from openai import AsyncOpenAI
from sqlalchemy.orm import Session
from azure.storage.blob import BlobServiceClient

from dsl_api.models.project import Project
from dsl_api.models.project_version import ProjectVersion
from dsl_api.models.project_event import ProjectEvent
from dsl_api.models.chat_message import ChatMessage

from dsl_worker.config import settings
from dsl_worker.project_state import ProjectState
from dsl_worker.billing import CostTracker, TrackedOpenAIClient
from dsl_worker.checkpoint import CheckpointManager, checkpoints_to_seeds

from dsl_worker.agents import OrchestratorAgent
from dsl_worker.phases.research_tools import Seed
from dsl_worker.phases.row_generator import GenerationWorkerPool
from dsl_worker.phases.sandbox import SandboxExecutor

logger = logging.getLogger(__name__)

WORKER_ID = f"{socket.gethostname()}-{os.getpid()}"

# Batch size for queue → generation pool
GENERATION_BATCH_SIZE = 50


class JobProcessor:
    """
    Job processor for the orchestrator-driven pipeline.

    Flow: conversation → orchestrator → seeds → generation
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
        logger.warning("[Worker] Stop requested")
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

        logger.info(f"Starting job: project={project_id}, version={version_id}")

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
            logger.exception(f"Job error: {e}")

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
        tracked_client: TrackedOpenAIClient,
        cost_tracker: CostTracker,
    ) -> bool:
        """Run the orchestrator-driven pipeline."""

        # Create workspace
        workspace_dir = Path(f"./workspace_{project.id}_{version.id}")
        workspace_dir.mkdir(parents=True, exist_ok=True)
        (workspace_dir / "uploads").mkdir(exist_ok=True)
        (workspace_dir / "downloads").mkdir(exist_ok=True)

        # Load uploaded files from blob
        await self._load_uploaded_files(state, workspace_dir)

        # Initialize sandbox
        self._sandbox = SandboxExecutor(use_pool=True, pool_size=3)

        # Initialize checkpoint manager
        checkpoint_mgr = CheckpointManager(
            blob_service_client=self.blob_service_client,
            container_name=settings.azure_storage_container_name,
            project_id=project.id,
            version_id=version.id,
        )
        checkpoint = await checkpoint_mgr.initialize()

        stop_checker = self._make_stop_checker(state)

        # Check if resuming from generation phase
        if checkpoint.current_phase == "generation" and checkpoint.seeds:
            logger.info(
                f"[Pipeline] Resuming generation: "
                f"{len(checkpoint.processed_seed_indices)}/{len(checkpoint.seeds)} done"
            )
            return await self._run_generation_from_checkpoint(
                db, project, version, state, tracked_client, cost_tracker,
                checkpoint_mgr, checkpoint, workspace_dir, stop_checker,
            )

        # --- Fresh run: orchestrator-driven flow ---

        # Load conversation history
        chat_history = self._load_chat_history(db, project.id)

        logger.info(f"[Pipeline] Starting orchestrator with {len(chat_history)} chat messages")

        # State for generation consumer
        generation_task: Optional[asyncio.Task] = None
        generation_result = {"success": 0, "errors": 0}

        # Callbacks for the orchestrator
        async def on_recipe_ready(recipe: str, seed_queue: asyncio.Queue):
            nonlocal generation_task

            # Save recipe to checkpoint
            await checkpoint_mgr.set_recipe(recipe)
            await checkpoint_mgr.set_phase("generation")

            # Start generation consumer in background
            generation_task = asyncio.create_task(
                self._run_generation_from_queue(
                    db=db,
                    project=project,
                    version=version,
                    state=state,
                    tracked_client=tracked_client,
                    cost_tracker=cost_tracker,
                    checkpoint_mgr=checkpoint_mgr,
                    recipe=recipe,
                    seed_queue=seed_queue,
                    workspace_dir=workspace_dir,
                    stop_checker=stop_checker,
                    result_tracker=generation_result,
                    schema=state.columns,
                )
            )

        async def on_done():
            logger.info("[Pipeline] Orchestrator signaled done")

        # Create and run orchestrator
        orchestrator = OrchestratorAgent(
            chat_history=chat_history,
            columns=state.columns,
            num_samples=state.num_samples,
            openai_client=tracked_client,
            model=settings.research_model,
            workspace_dir=workspace_dir,
            brave_api_key=settings.brave_api_key,
            sandbox=self._sandbox,
            stop_checker=stop_checker,
            on_recipe_ready=on_recipe_ready,
            on_done=on_done,
        )

        try:
            result = await orchestrator.run()

            # Track orchestrator cost
            orch_cost = orchestrator.cost_usd
            if orch_cost > 0:
                cost_tracker.add_cost(
                    phase="orchestrator",
                    cost_usd=orch_cost,
                    model=settings.research_model,
                )
                await checkpoint_mgr.add_cost(orch_cost)

            logger.info(
                f"[Pipeline] Orchestrator finished: "
                f"cost=${orch_cost:.4f}, turns={result.turns_taken}"
            )

        finally:
            await orchestrator.cleanup()

        # Wait for generation to complete (if it was started)
        if generation_task:
            try:
                await generation_task
            except Exception as e:
                logger.error(f"[Pipeline] Generation consumer error: {e}")

        # Check final state
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

        total_success = generation_result["success"]

        if total_success > 0:
            await checkpoint_mgr.delete()
            self._handle_completion(db, project, version, cost_tracker)
            return True

        # No rows generated
        await checkpoint_mgr.force_save()
        self._handle_force_stop(db, project, version, cost_tracker, "No rows generated")
        return False

    def _load_chat_history(self, db: Session, project_id: UUID) -> List[Dict[str, str]]:
        """Load chat messages and format for the orchestrator."""
        messages = (
            db.query(ChatMessage)
            .filter(ChatMessage.project_id == project_id)
            .order_by(ChatMessage.created_at.asc())
            .all()
        )

        history = []
        for msg in messages:
            history.append({
                "role": msg.role,
                "content": msg.content,
            })

        return history

    async def _run_generation_from_queue(
        self,
        db: Session,
        project: Project,
        version: ProjectVersion,
        state: ProjectState,
        tracked_client: TrackedOpenAIClient,
        cost_tracker: CostTracker,
        checkpoint_mgr: CheckpointManager,
        recipe: str,
        seed_queue: asyncio.Queue,
        workspace_dir: Path,
        stop_checker,
        result_tracker: Dict,
        schema: List[Dict],
    ) -> None:
        """Consume seeds from the queue and run generation in batches."""

        batch: List[Dict] = []
        seed_index = len(checkpoint_mgr.checkpoint.seeds)  # Start index for new seeds

        while True:
            # Check stop conditions
            if stop_checker():
                break

            can_continue, _ = cost_tracker.check_balance_and_charge()
            if not can_continue:
                break

            try:
                # Wait for seed with timeout
                seed = await asyncio.wait_for(seed_queue.get(), timeout=5.0)
            except asyncio.TimeoutError:
                # Process any accumulated batch on timeout
                if batch:
                    await self._process_seed_batch(
                        batch, schema, recipe, db, project, version,
                        tracked_client, cost_tracker, checkpoint_mgr,
                        workspace_dir, stop_checker, result_tracker,
                        seed_index - len(batch),
                    )
                    batch = []
                continue

            if seed is None:
                # Poison pill from a generator — process remaining batch and exit
                logger.info("[Generation] Received poison pill, flushing remaining seeds")
                if batch:
                    await self._process_seed_batch(
                        batch, schema, recipe, db, project, version,
                        tracked_client, cost_tracker, checkpoint_mgr,
                        workspace_dir, stop_checker, result_tracker,
                        seed_index - len(batch),
                    )
                    batch = []

                # Drain any remaining seeds in the queue before exiting
                while not seed_queue.empty():
                    remaining = seed_queue.get_nowait()
                    if remaining is None:
                        continue
                    await checkpoint_mgr.add_seed_dict(remaining)
                    batch.append(remaining)
                    seed_index += 1

                if batch:
                    await self._process_seed_batch(
                        batch, schema, recipe, db, project, version,
                        tracked_client, cost_tracker, checkpoint_mgr,
                        workspace_dir, stop_checker, result_tracker,
                        seed_index - len(batch),
                    )
                break

            # Checkpoint the seed
            await checkpoint_mgr.add_seed_dict(seed)
            batch.append(seed)
            seed_index += 1

            # Process batch when full
            if len(batch) >= GENERATION_BATCH_SIZE:
                await self._process_seed_batch(
                    batch, schema, recipe, db, project, version,
                    tracked_client, cost_tracker, checkpoint_mgr,
                    workspace_dir, stop_checker, result_tracker,
                    seed_index - len(batch),
                )
                batch = []

        logger.info(
            f"[Generation] Queue consumer done: "
            f"{result_tracker['success']} success, {result_tracker['errors']} errors"
        )

    async def _process_seed_batch(
        self,
        batch: List[Dict],
        schema: List[Dict],
        recipe: str,
        db: Session,
        project: Project,
        version: ProjectVersion,
        tracked_client: TrackedOpenAIClient,
        cost_tracker: CostTracker,
        checkpoint_mgr: CheckpointManager,
        workspace_dir: Path,
        stop_checker,
        result_tracker: Dict,
        start_index: int,
    ) -> None:
        """Process a batch of seed dicts through the generation pool."""

        # Convert seed dicts to Seed objects
        seeds = []
        for seed_dict in batch:
            seed = Seed(
                content=json.dumps(seed_dict) if isinstance(seed_dict, dict) else str(seed_dict),
                scope_id="generator",
                scope_description=recipe[:500],
                notes=[],
                research_summary=None,
                source_ref=None,
                source_url=None,
            )
            seeds.append(seed)

        if not seeds:
            return

        logger.info(f"[Generation] Processing batch of {len(seeds)} seeds")

        pool = GenerationWorkerPool(
            workspace_dir=workspace_dir,
            openai_client=tracked_client,
            db_session=db,
            project_id=project.id,
            version_id=version.id,
            brave_api_key=settings.brave_api_key,
            sandbox=self._sandbox,
            num_workers=settings.generation_parallel_samples,
            stop_checker=stop_checker,
            cost_tracker=cost_tracker,
            checkpoint_callback=lambda idx, success, row_id: asyncio.create_task(
                checkpoint_mgr.mark_seed_processed(start_index + idx, success, row_id)
            ),
        )

        success, errors = await pool.process_seeds(seeds, schema)

        result_tracker["success"] += success
        result_tracker["errors"] += errors

        # Track generation cost
        stats = pool.get_stats()
        if stats["total_cost"] > 0:
            cost_tracker.add_cost(
                phase="generation",
                cost_usd=stats["total_cost"],
                model=settings.generation_model,
            )
            await checkpoint_mgr.add_cost(stats["total_cost"])

    async def _run_generation_from_checkpoint(
        self,
        db: Session,
        project: Project,
        version: ProjectVersion,
        state: ProjectState,
        tracked_client: TrackedOpenAIClient,
        cost_tracker: CostTracker,
        checkpoint_mgr: CheckpointManager,
        checkpoint,
        workspace_dir: Path,
        stop_checker,
    ) -> bool:
        """Resume generation from a saved checkpoint."""

        seeds = checkpoints_to_seeds(checkpoint.seeds)

        # Filter to pending seeds only
        pending_indices = checkpoint.get_pending_seed_indices()
        pending_seeds = [seeds[i] for i in pending_indices]

        if not pending_seeds:
            logger.info("[Pipeline] All seeds already processed")
            await checkpoint_mgr.delete()
            self._handle_completion(db, project, version, cost_tracker)
            return True

        logger.info(
            f"[Pipeline] Resuming generation: "
            f"{len(pending_seeds)} pending of {len(seeds)} total"
        )

        pool = GenerationWorkerPool(
            workspace_dir=workspace_dir,
            openai_client=tracked_client,
            db_session=db,
            project_id=project.id,
            version_id=version.id,
            brave_api_key=settings.brave_api_key,
            sandbox=self._sandbox,
            num_workers=settings.generation_parallel_samples,
            stop_checker=stop_checker,
            cost_tracker=cost_tracker,
            checkpoint_callback=lambda idx, success, row_id: asyncio.create_task(
                checkpoint_mgr.mark_seed_processed(pending_indices[idx], success, row_id)
            ),
        )

        total_success, total_errors = await pool.process_seeds(
            pending_seeds, state.columns
        )

        # Track cost
        stats = pool.get_stats()
        if stats["total_cost"] > 0:
            cost_tracker.add_cost(
                phase="generation",
                cost_usd=stats["total_cost"],
                model=settings.generation_model,
            )
            await checkpoint_mgr.add_cost(stats["total_cost"])

        # Check balance
        can_continue, stop_reason = cost_tracker.check_balance_and_charge()
        if not can_continue:
            await checkpoint_mgr.force_save()
            self._handle_force_stop(db, project, version, cost_tracker, stop_reason)
            return False

        # Handle final state
        state.refresh()
        if state.paused:
            await checkpoint_mgr.force_save()
            self._handle_pause(db, project, version, cost_tracker)
            return True

        total_processed = len(checkpoint.processed_seed_indices) + total_success

        if total_processed > 0:
            await checkpoint_mgr.delete()
            self._handle_completion(db, project, version, cost_tracker)
            return True

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
        logger.info(f"Completed: {version.generated_count} samples")

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
