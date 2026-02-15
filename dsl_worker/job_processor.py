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
import time
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

# Langfuse is optional
try:
    from langfuse import get_client as _get_langfuse_client, propagate_attributes

    def _get_langfuse():
        try:
            return _get_langfuse_client()
        except Exception:
            return None
except ImportError:
    from contextlib import contextmanager

    def _get_langfuse():
        return None

    @contextmanager
    def propagate_attributes(**kwargs):
        yield
from dsl_worker.project_state import ProjectState
from dsl_worker.billing import CostTracker, TrackedOpenAIClient
from dsl_worker.checkpoint import CheckpointManager, checkpoints_to_seeds

from dsl_worker.agents import OrchestratorAgent
from dsl_worker.phases.research_tools import Seed
from dsl_worker.phases.row_generator import GenerationWorkerPool, BucketTracker
from dsl_worker.phases.sandbox import SandboxExecutor

logger = logging.getLogger(__name__)

WORKER_ID = f"{socket.gethostname()}-{os.getpid()}"

# Batch size for queue → generation pool.
# Seeds trickle in slowly from generators, so a small batch ensures rows
# start processing almost immediately (5s timeout flush handles partials).
GENERATION_BATCH_SIZE = 5


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

            # Run pipeline (wrapped in Langfuse trace if available)
            # propagate_attributes tags every Langfuse observation created
            # inside this block with session/user/metadata — so all API calls
            # from all agents are grouped under one session in the Langfuse UI.
            with propagate_attributes(
                session_id=str(project_id),
                user_id=str(project.user_id),
                tags=[project.name, f"v{version.version_number}"],
                metadata={
                    "project_id": str(project_id),
                    "version_id": str(version_id),
                    "num_samples": version.num_samples,
                },
            ):
                langfuse = _get_langfuse()
                if langfuse:
                    with langfuse.start_as_current_observation(
                        as_type="span",
                        name=f"job:{project.name} v{version.version_number}",
                    ):
                        result = await self._run_pipeline(
                            db, project, version, state, tracked_client, cost_tracker
                        )
                else:
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
            # Flush Langfuse traces before cleanup
            langfuse = _get_langfuse()
            if langfuse:
                langfuse.flush()
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
            version.progress_detail = {"phase": "generating"}
            db.commit()
            return await self._run_generation_from_checkpoint(
                db, project, version, state, tracked_client, cost_tracker,
                checkpoint_mgr, checkpoint, workspace_dir, stop_checker,
            )

        # --- Fresh run: orchestrator-driven flow ---

        # Load conversation history
        chat_history = self._load_chat_history(db, project.id)

        # Build uploaded files metadata for orchestrator context
        uploaded_files = []
        if state.files_snapshot:
            for f in state.files_snapshot:
                uploaded_files.append({
                    "filename": f.get("filename", "unknown"),
                    "content_type": f.get("content_type", ""),
                    "size_bytes": f.get("size_bytes", 0),
                })

        logger.info(f"[Pipeline] Starting orchestrator with {len(chat_history)} chat messages")

        # on_generate callback — called when orchestrator's set_plan() fires
        async def on_generate(
            plan: Dict,
            seed_queue: asyncio.Queue,
            result_future: asyncio.Future,
        ):
            nonlocal version, checkpoint_mgr

            # Handle new_version flag — orchestrator decided to create a fresh dataset
            if plan.pop("_new_version", False):
                logger.info("[Pipeline] Orchestrator requested new version")
                from sqlalchemy import func as sql_func
                max_num = (
                    db.query(sql_func.max(ProjectVersion.version_number))
                    .filter(ProjectVersion.project_id == project.id)
                    .scalar()
                    or 0
                )
                new_version = ProjectVersion(
                    project_id=project.id,
                    version_number=max_num + 1,
                    num_samples=state.num_samples,
                    generation_prompt=state.generation_prompt,
                    columns=state.columns,
                    diversity_spec=None,
                    files_snapshot=state.files_snapshot,
                    examples_snapshot=state.examples_snapshot,
                    status="running",
                )
                db.add(new_version)
                project.current_version_id = new_version.id
                db.commit()
                db.refresh(new_version)

                # Switch to the new version
                version = new_version
                checkpoint_mgr = CheckpointManager(
                    blob_service_client=self.blob_service_client,
                    container_name=settings.azure_storage_container_name,
                    project_id=project.id,
                    version_id=version.id,
                )
                await checkpoint_mgr.initialize()
                self._emit_event(db, project, version, "run_started", "New version created by orchestrator")
                logger.info(f"[Pipeline] Created version v{new_version.version_number} (id={new_version.id})")

            # Save plan as JSON to version.recipe and checkpoint
            plan_json = json.dumps(plan, indent=2)
            await checkpoint_mgr.set_recipe(plan_json)
            await checkpoint_mgr.set_phase("generation")
            version.recipe = plan_json
            db.commit()

            # Start generation consumer as a background task
            # The consumer will resolve result_future when done
            asyncio.create_task(
                self._run_generation_from_queue_v2(
                    db=db,
                    project=project,
                    version=version,
                    state=state,
                    tracked_client=tracked_client,
                    cost_tracker=cost_tracker,
                    checkpoint_mgr=checkpoint_mgr,
                    plan=plan,
                    seed_queue=seed_queue,
                    workspace_dir=workspace_dir,
                    stop_checker=stop_checker,
                    schema=state.columns,
                    result_future=result_future,
                    num_generators=len(plan.get("generators", [])),
                )
            )

        # Detect feedback re-run: recipe exists from previous run but we're
        # not resuming from a generation checkpoint (checkpoint was deleted).
        previous_recipe = None
        if version.recipe and checkpoint.current_phase != "generation":
            previous_recipe = version.recipe
            logger.info("[Pipeline] Feedback re-run detected — passing previous recipe to orchestrator")

        # Progress tracking via tool call observation
        progress_counters: Dict[str, Any] = {"phase": "strategizing"}
        last_progress_flush = time.time()

        def on_tool_call(agent_label: str, tool_name: str):
            nonlocal last_progress_flush

            # Phase transitions (orchestrator tools)
            phase_map = {
                "strategy": "strategizing",
                "research": "researching",
                "ask_research": "researching",
                "set_plan": "generating",
                "add_generator": "generating",
            }
            if tool_name in phase_map:
                progress_counters["phase"] = phase_map[tool_name]

            # Counter increments (sub-agent tools)
            counter_map = {
                "brave_search": "searches",
                "open": "sources",
                "click": "sources",
                "code_exec": "analyses",
                "yield_seed": "seeds",
            }
            if tool_name in counter_map:
                key = counter_map[tool_name]
                progress_counters[key] = progress_counters.get(key, 0) + 1

            # Throttled DB flush (every 2s)
            now = time.time()
            if now - last_progress_flush >= 2.0:
                version.progress_detail = dict(progress_counters)
                db.commit()
                last_progress_flush = now

        version.progress_detail = {"phase": "strategizing"}
        db.commit()

        # Create and run orchestrator
        orchestrator = OrchestratorAgent(
            chat_history=chat_history,
            columns=state.columns,
            num_samples=state.num_samples,
            openai_client=tracked_client,
            model=settings.research_model,
            workspace_dir=workspace_dir,
            uploaded_files=uploaded_files if uploaded_files else None,
            brave_api_key=settings.brave_api_key,
            sandbox=self._sandbox,
            stop_checker=stop_checker,
            on_generate=on_generate,
            previous_recipe=previous_recipe,
            blob_service_client=self.blob_service_client,
            project_id=project.id,
            on_tool_call=on_tool_call,
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

        # Check final state — the orchestrator calls done() which ends its loop.
        # Generation has already completed by then (set_plan(start=true) blocks).
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

        # Check if version was auto-paused by the consumer (sample_ready)
        db.refresh(version)
        if version.status == "paused":
            await checkpoint_mgr.force_save()
            return True

        if version.generated_count and version.generated_count > 0:
            await checkpoint_mgr.set_phase("completed")
            await checkpoint_mgr.force_save()
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

    async def _run_generation_from_queue_v2(
        self,
        db: Session,
        project: Project,
        version: ProjectVersion,
        state: ProjectState,
        tracked_client: TrackedOpenAIClient,
        cost_tracker: CostTracker,
        checkpoint_mgr: CheckpointManager,
        plan: Dict,
        seed_queue: asyncio.Queue,
        workspace_dir: Path,
        stop_checker,
        schema: List[Dict],
        result_future: asyncio.Future,
        num_generators: int = 1,
    ) -> None:
        """Consume seeds from the queue and run generation. Resolves result_future when done."""

        SAMPLE_SIZE = 5
        bucket_tracker = BucketTracker(plan, state.num_samples)

        batch: List[Dict] = []
        seed_index = len(checkpoint_mgr.checkpoint.seeds)
        generators_done = 0
        total_success = 0
        total_errors = 0
        is_initial_run = True  # First set_plan(start=true) triggers sampling pause

        try:
            while True:
                # Check stop conditions
                if stop_checker():
                    break

                can_continue, _ = cost_tracker.check_balance_and_charge()
                if not can_continue:
                    break

                # Auto-pause after sample rows on initial run
                if is_initial_run and total_success >= SAMPLE_SIZE:
                    logger.info(
                        f"[Generation] Auto-pausing after {SAMPLE_SIZE} sample rows "
                        f"for user review"
                    )
                    # Drain remaining seeds into checkpoint
                    while not seed_queue.empty():
                        remaining = seed_queue.get_nowait()
                        if remaining is not None:
                            await checkpoint_mgr.add_seed_dict(remaining)
                            seed_index += 1

                    self._handle_sample_pause(db, project, version, cost_tracker, total_success)

                    if not result_future.done():
                        result_future.set_result({
                            "status": "sampling_paused",
                            "rows_generated": total_success,
                            "errors": total_errors,
                        })
                    return

                try:
                    seed = await asyncio.wait_for(seed_queue.get(), timeout=5.0)
                except asyncio.TimeoutError:
                    # Flush batch on timeout
                    if batch:
                        s, e = await self._process_seed_batch_v2(
                            batch, schema, plan, bucket_tracker, db, project, version,
                            tracked_client, cost_tracker, checkpoint_mgr,
                            workspace_dir, stop_checker, seed_index - len(batch),
                        )
                        total_success += s
                        total_errors += e
                        batch = []
                    continue

                if seed is None:
                    # Poison pill from a generator
                    generators_done += 1
                    logger.info(
                        f"[Generation] Generator finished "
                        f"({generators_done}/{num_generators})"
                    )

                    # Flush current batch
                    if batch:
                        s, e = await self._process_seed_batch_v2(
                            batch, schema, plan, bucket_tracker, db, project, version,
                            tracked_client, cost_tracker, checkpoint_mgr,
                            workspace_dir, stop_checker, seed_index - len(batch),
                        )
                        total_success += s
                        total_errors += e
                        batch = []

                    # Drain any remaining seeds before checking if all generators done
                    while not seed_queue.empty():
                        remaining = seed_queue.get_nowait()
                        if remaining is None:
                            generators_done += 1
                            continue
                        await checkpoint_mgr.add_seed_dict(remaining)
                        batch.append(remaining)
                        seed_index += 1

                    if batch:
                        s, e = await self._process_seed_batch_v2(
                            batch, schema, plan, bucket_tracker, db, project, version,
                            tracked_client, cost_tracker, checkpoint_mgr,
                            workspace_dir, stop_checker, seed_index - len(batch),
                        )
                        total_success += s
                        total_errors += e
                        batch = []

                    if generators_done >= num_generators:
                        break
                    continue

                # Checkpoint the seed
                await checkpoint_mgr.add_seed_dict(seed)
                batch.append(seed)
                seed_index += 1

                # Process batch when full
                if len(batch) >= GENERATION_BATCH_SIZE:
                    s, e = await self._process_seed_batch_v2(
                        batch, schema, plan, bucket_tracker, db, project, version,
                        tracked_client, cost_tracker, checkpoint_mgr,
                        workspace_dir, stop_checker, seed_index - len(batch),
                    )
                    total_success += s
                    total_errors += e
                    batch = []

                # Check if all bucket quotas are met
                if await bucket_tracker.is_complete():
                    logger.info("[Generation] All bucket quotas met")
                    break

            # Determine final status
            bucket_status = await bucket_tracker.get_status()

            if await bucket_tracker.is_complete():
                status = "complete"
            elif generators_done >= num_generators:
                status = "shortage"
            else:
                status = "complete"  # Stopped for other reason

            result = {
                "status": status,
                "rows_generated": total_success,
                "errors": total_errors,
                "bucket_status": bucket_status,
            }

            logger.info(
                f"[Generation] Queue consumer done: status={status}, "
                f"{total_success} success, {total_errors} errors"
            )

            if not result_future.done():
                result_future.set_result(result)

        except Exception as e:
            logger.error(f"[Generation] Consumer error: {e}")
            if not result_future.done():
                result_future.set_exception(e)

    async def _process_seed_batch_v2(
        self,
        batch: List[Dict],
        schema: List[Dict],
        plan: Dict,
        bucket_tracker: BucketTracker,
        db: Session,
        project: Project,
        version: ProjectVersion,
        tracked_client: TrackedOpenAIClient,
        cost_tracker: CostTracker,
        checkpoint_mgr: CheckpointManager,
        workspace_dir: Path,
        stop_checker,
        start_index: int,
    ) -> tuple[int, int]:
        """Process a batch of seed envelopes through the generation pool. Returns (success, errors)."""

        # Convert seed envelopes to Seed objects
        seeds = []
        for envelope in batch:
            # Handle envelope format: {"data": ..., "bucket_id": ..., "generator_id": ...}
            if isinstance(envelope, dict) and "data" in envelope:
                seed_data = envelope["data"]
                bucket_id = envelope.get("bucket_id")
                content = json.dumps(seed_data) if isinstance(seed_data, dict) else seed_data
            else:
                content = json.dumps(envelope) if isinstance(envelope, dict) else str(envelope)
                bucket_id = None

            seed = Seed(
                content=content,
                scope_id=envelope.get("generator_id", "generator") if isinstance(envelope, dict) else "generator",
                scope_description="",
                notes=[],
                research_summary=None,
                source_ref=None,
                source_url=None,
                bucket_id=bucket_id,
            )
            seeds.append(seed)

        if not seeds:
            return 0, 0

        logger.info(f"[Generation] Processing batch of {len(seeds)} seeds")

        concurrency = settings.generation_parallel_samples

        pool = GenerationWorkerPool(
            workspace_dir=workspace_dir,
            openai_client=tracked_client,
            db_session=db,
            project_id=project.id,
            version_id=version.id,
            model=settings.generation_model,
            brave_api_key=settings.brave_api_key,
            sandbox=self._sandbox,
            num_workers=concurrency,
            stop_checker=stop_checker,
            cost_tracker=cost_tracker,
            checkpoint_callback=lambda idx, success, row_id: asyncio.create_task(
                checkpoint_mgr.mark_seed_processed(start_index + idx, success, row_id)
            ),
            plan=plan,
            bucket_tracker=bucket_tracker,
            blob_service_client=self.blob_service_client,
        )

        success, errors = await pool.process_seeds(seeds, schema)

        # Track generation cost
        stats = pool.get_stats()
        if stats["total_cost"] > 0:
            cost_tracker.add_cost(
                phase="generation",
                cost_usd=stats["total_cost"],
                model=settings.generation_model,
            )
            await checkpoint_mgr.add_cost(stats["total_cost"])

        return success, errors

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

        # Try to parse plan from checkpoint recipe (JSON).
        # Fall back to legacy string recipe (no plan, no bucket tracking).
        plan = None
        bucket_tracker = None
        if checkpoint.recipe:
            try:
                plan = json.loads(checkpoint.recipe)
                bucket_tracker = BucketTracker(plan, state.num_samples)

                # Pre-populate bucket tracker with already-processed seed counts
                processed_indices = set(checkpoint.processed_seed_indices)
                for idx in processed_indices:
                    if idx < len(seeds):
                        seed = seeds[idx]
                        # Only count successful rows (check checkpoint status)
                        seed_ckpt = checkpoint.seeds[idx]
                        if seed_ckpt.get("status") == "completed":
                            await bucket_tracker.record_success(seed.bucket_id)

                logger.info("[Pipeline] Resumed with plan and bucket tracking")
            except (json.JSONDecodeError, TypeError):
                # Legacy string recipe — no plan structure
                logger.info("[Pipeline] Legacy recipe format, no bucket tracking")
                plan = None
                bucket_tracker = None

        concurrency = settings.generation_parallel_samples

        pool = GenerationWorkerPool(
            workspace_dir=workspace_dir,
            openai_client=tracked_client,
            db_session=db,
            project_id=project.id,
            version_id=version.id,
            model=settings.generation_model,
            brave_api_key=settings.brave_api_key,
            sandbox=self._sandbox,
            num_workers=concurrency,
            stop_checker=stop_checker,
            cost_tracker=cost_tracker,
            checkpoint_callback=lambda idx, success, row_id: asyncio.create_task(
                checkpoint_mgr.mark_seed_processed(pending_indices[idx], success, row_id)
            ),
            plan=plan,
            bucket_tracker=bucket_tracker,
            blob_service_client=self.blob_service_client,
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

    def _handle_sample_pause(
        self,
        db: Session,
        project: Project,
        version: ProjectVersion,
        cost_tracker: Optional[CostTracker],
        sample_count: int = 5,
    ) -> None:
        """Handle auto-pause after sample rows are generated."""
        logger.info(f"Version {version.id} paused for sample review ({sample_count} samples)")
        db.refresh(version)

        if cost_tracker:
            cost_tracker.charge_remaining()

        version.status = "paused"
        version.progress_detail = {"phase": "sample_review"}

        details = {
            "sample_count": sample_count,
            "sample_ready": True,
            "paused_at": datetime.now(timezone.utc).isoformat(),
        }
        if cost_tracker:
            summary = cost_tracker.get_summary()
            details["total_cost_cents"] = summary["total_costs_cents"]

        self._emit_event(
            db, project, version, "sample_ready",
            f"Sample rows ready for review ({sample_count} rows)",
            details,
        )
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
        # Keep progress_detail so frontend can show timeline in paused state

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
        version.progress_detail = None

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
        version.progress_detail = None

        self._emit_event(
            db, project, version, "failed",
            reason,
            {"reason": reason}
        )

        db.commit()
