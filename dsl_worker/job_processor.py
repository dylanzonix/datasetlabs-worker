"""
Job Processor — V3 orchestrator-driven pipeline

Flow:
1. Load conversation history and uploaded files
2. Initialize SourceManager for research accumulation
3. Run orchestrator (loop: research → create work items → generate → monitor)
4. Generation runs in background, consuming work items from queue
5. Orchestrator talks to user via ask_user() during generation
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
from dsl_api.models.project_file import ProjectFile
from dsl_api.models.project_connector import ProjectConnector

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
from dsl_worker.checkpoint import CheckpointManager, checkpoints_to_work_items

from dsl_worker.agents import OrchestratorAgent
from dsl_worker.phases.source_manager import SourceManager
from dsl_worker.phases.row_generator import GenerationWorkerPool
from dsl_worker.phases.sandbox import SandboxExecutor

logger = logging.getLogger(__name__)

WORKER_ID = f"{socket.gethostname()}-{os.getpid()}"

# Batch size for work item queue → generation pool.
GENERATION_BATCH_SIZE = 5


class JobProcessor:
    """
    Job processor for the V3 orchestrator-driven pipeline.

    Flow: conversation → orchestrator (research + work items) → generation
    Checkpoints saved to Azure Blob for pause/resume.
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
        # MCP tools for current job (set during _run_pipeline)
        self._mcp_tools: list = []

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
        """Run the V3 orchestrator-driven pipeline."""

        # Create workspace
        workspace_dir = Path(f"./workspace_{project.id}_{version.id}")
        workspace_dir.mkdir(parents=True, exist_ok=True)
        (workspace_dir / "uploads").mkdir(exist_ok=True)
        (workspace_dir / "downloads").mkdir(exist_ok=True)

        # Load uploaded files from blob
        await self._load_uploaded_files(db, project.id, workspace_dir)

        # Initialize sandbox
        self._sandbox = SandboxExecutor(use_pool=True, pool_size=3)

        # Initialize source manager
        source_manager = SourceManager(workspace_dir)
        await source_manager.initialize()

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
        if checkpoint.current_phase == "generation" and checkpoint.work_items:
            logger.info(
                f"[Pipeline] Resuming generation: "
                f"{len(checkpoint.processed_indices)}/{len(checkpoint.work_items)} done"
            )
            version.progress_detail = {"phase": "generating"}
            db.commit()
            return await self._run_generation_from_checkpoint(
                db, project, version, state, tracked_client, cost_tracker,
                checkpoint_mgr, checkpoint, workspace_dir, stop_checker,
                source_manager,
            )

        # --- Fresh run: orchestrator-driven flow ---

        # Load conversation history
        chat_history = self._load_chat_history(db, project.id)

        # Build uploaded files metadata for orchestrator context
        project_files = (
            db.query(ProjectFile)
            .filter(
                ProjectFile.project_id == project.id,
                ProjectFile.deleted_at.is_(None),
                ProjectFile.status == "uploaded",
            )
            .all()
        )
        uploaded_files = [
            {
                "filename": f.filename,
                "content_type": f.content_type,
                "size_bytes": f.size_bytes,
            }
            for f in project_files
        ]

        # Load connectors for MCP tool injection
        connectors = (
            db.query(ProjectConnector)
            .filter(
                ProjectConnector.project_id == project.id,
                ProjectConnector.deleted_at.is_(None),
            )
            .all()
        )
        mcp_tools = self._build_mcp_tools(connectors)
        self._mcp_tools = mcp_tools

        logger.info(f"[Pipeline] Starting orchestrator with {len(chat_history)} chat messages")

        # === Generation state (shared between orchestrator callbacks and consumer) ===
        work_item_queue: asyncio.Queue = asyncio.Queue()
        generation_stats = {
            "rows_generated": 0,
            "errors": 0,
            "skipped": 0,
            "in_progress": 0,
            "total_cost": 0.0,
        }
        generation_task: Optional[asyncio.Task] = None
        generation_lock = asyncio.Lock()

        # === Callbacks for orchestrator ===

        async def on_create_work_items(items: List[Dict]) -> int:
            nonlocal generation_task

            # Checkpoint each work item
            for item in items:
                await checkpoint_mgr.add_work_item(item)

            # Queue items for generation
            for item in items:
                await work_item_queue.put(item)

            # Start/continue the generation consumer if not running
            async with generation_lock:
                if generation_task is None or generation_task.done():
                    await checkpoint_mgr.set_phase("generation")
                    generation_task = asyncio.create_task(
                        self._run_generation_consumer(
                            db=db,
                            project=project,
                            version=version,
                            tracked_client=tracked_client,
                            cost_tracker=cost_tracker,
                            checkpoint_mgr=checkpoint_mgr,
                            work_item_queue=work_item_queue,
                            workspace_dir=workspace_dir,
                            stop_checker=stop_checker,
                            schema=state.columns,
                            source_manager=source_manager,
                            generation_stats=generation_stats,
                        )
                    )

            return len(items)

        async def on_ask_user(message: str) -> str:
            # Post assistant message to chat
            chat_msg = ChatMessage(
                project_id=project.id,
                role="assistant",
                content=message,
            )
            db.add(chat_msg)
            db.commit()

            # Update progress
            version.progress_detail = {"phase": "waiting_for_user"}
            db.commit()

            logger.info(f"[Pipeline] ask_user: waiting for user response...")

            # Poll for user response (10 minute timeout)
            for _ in range(600):
                await asyncio.sleep(1)
                if stop_checker():
                    version.progress_detail = {"phase": "generating"}
                    db.commit()
                    return "[User did not respond — generation was paused]"

                new_msg = (
                    db.query(ChatMessage)
                    .filter(
                        ChatMessage.project_id == project.id,
                        ChatMessage.role == "user",
                        ChatMessage.created_at > chat_msg.created_at,
                    )
                    .order_by(ChatMessage.created_at.asc())
                    .first()
                )
                if new_msg:
                    version.progress_detail = {"phase": "generating"}
                    db.commit()
                    logger.info(f"[Pipeline] ask_user: got user response ({len(new_msg.content)} chars)")
                    return new_msg.content

            version.progress_detail = {"phase": "generating"}
            db.commit()
            return "[User did not respond within 10 minutes — continuing]"

        def on_check_progress() -> Dict:
            return dict(generation_stats)

        # Progress tracking via tool call observation
        progress_counters: Dict[str, Any] = {"phase": "researching"}
        last_progress_flush = time.time()

        def on_tool_call(agent_label: str, tool_name: str):
            nonlocal last_progress_flush

            # Phase transitions (orchestrator tools)
            phase_map = {
                "run_subagent": "researching",
                "save_source": "researching",
                "create_work_items": "generating",
                "check_progress": "generating",
                "ask_user": "waiting_for_user",
                "done": "completing",
            }
            if tool_name in phase_map:
                progress_counters["phase"] = phase_map[tool_name]

            # Counter increments
            counter_map = {
                "brave_search": "searches",
                "open": "sources",
                "click": "sources",
                "code_exec": "analyses",
                "save_source": "saved_sources",
                "create_work_items": "work_items_created",
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

        version.progress_detail = {"phase": "researching"}
        db.commit()

        # Create and run orchestrator
        orchestrator = OrchestratorAgent(
            chat_history=chat_history,
            columns=state.columns,
            num_samples=state.num_samples,
            openai_client=tracked_client,
            model=settings.research_model,
            workspace_dir=workspace_dir,
            source_manager=source_manager,
            on_create_work_items=on_create_work_items,
            on_ask_user=on_ask_user,
            on_check_progress=on_check_progress,
            uploaded_files=uploaded_files if uploaded_files else None,
            brave_api_key=settings.brave_api_key,
            sandbox=self._sandbox,
            stop_checker=stop_checker,
            blob_service_client=self.blob_service_client,
            project_id=project.id,
            on_tool_call=on_tool_call,
            mcp_tools=mcp_tools,
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

        # Wait for generation consumer to finish processing remaining items
        if generation_task and not generation_task.done():
            # Send poison pill to signal no more items coming
            await work_item_queue.put(None)
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

        db.refresh(version)
        if version.generated_count and version.generated_count > 0:
            await checkpoint_mgr.set_phase("completed")
            await checkpoint_mgr.force_save()
            self._handle_completion(db, project, version, cost_tracker)
            return True

        # No rows generated
        await checkpoint_mgr.force_save()
        self._handle_force_stop(db, project, version, cost_tracker, "No rows generated")
        return False

    async def _run_generation_consumer(
        self,
        db: Session,
        project: Project,
        version: ProjectVersion,
        tracked_client: TrackedOpenAIClient,
        cost_tracker: CostTracker,
        checkpoint_mgr: CheckpointManager,
        work_item_queue: asyncio.Queue,
        workspace_dir: Path,
        stop_checker,
        schema: List[Dict],
        source_manager: SourceManager,
        generation_stats: Dict,
    ) -> None:
        """
        Background consumer: dequeue work items, batch them, and run
        through GenerationWorkerPool. Updates generation_stats in place.
        """
        batch: List[Dict] = []
        batch_start_index = len(checkpoint_mgr.checkpoint.work_items)

        try:
            while True:
                if stop_checker():
                    break

                can_continue, _ = cost_tracker.check_balance_and_charge()
                if not can_continue:
                    break

                try:
                    item = await asyncio.wait_for(work_item_queue.get(), timeout=5.0)
                except asyncio.TimeoutError:
                    # Flush batch on timeout
                    if batch:
                        await self._process_work_item_batch(
                            batch, schema, source_manager, db, project, version,
                            tracked_client, cost_tracker, checkpoint_mgr,
                            workspace_dir, stop_checker, batch_start_index,
                            generation_stats,
                        )
                        batch_start_index += len(batch)
                        batch = []
                    continue

                if item is None:
                    # Poison pill — no more items coming
                    if batch:
                        await self._process_work_item_batch(
                            batch, schema, source_manager, db, project, version,
                            tracked_client, cost_tracker, checkpoint_mgr,
                            workspace_dir, stop_checker, batch_start_index,
                            generation_stats,
                        )
                        batch_start_index += len(batch)
                        batch = []
                    break

                batch.append(item)

                if len(batch) >= GENERATION_BATCH_SIZE:
                    await self._process_work_item_batch(
                        batch, schema, source_manager, db, project, version,
                        tracked_client, cost_tracker, checkpoint_mgr,
                        workspace_dir, stop_checker, batch_start_index,
                        generation_stats,
                    )
                    batch_start_index += len(batch)
                    batch = []

        except Exception as e:
            logger.error(f"[Generation] Consumer error: {e}")

        logger.info(
            f"[Generation] Consumer done: "
            f"{generation_stats['rows_generated']} success, "
            f"{generation_stats['errors']} errors, "
            f"{generation_stats['skipped']} skipped"
        )

    async def _process_work_item_batch(
        self,
        batch: List[Dict],
        schema: List[Dict],
        source_manager: SourceManager,
        db: Session,
        project: Project,
        version: ProjectVersion,
        tracked_client: TrackedOpenAIClient,
        cost_tracker: CostTracker,
        checkpoint_mgr: CheckpointManager,
        workspace_dir: Path,
        stop_checker,
        start_index: int,
        generation_stats: Dict,
    ) -> None:
        """Process a batch of work items through the generation pool."""

        if not batch:
            return

        logger.info(f"[Generation] Processing batch of {len(batch)} work items")

        manifest_summary = source_manager.get_manifest_summary()
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
                checkpoint_mgr.mark_processed(start_index + idx, success, row_id)
            ),
            blob_service_client=self.blob_service_client,
            mcp_tools=self._mcp_tools,
        )

        success, errors = await pool.process_work_items(
            batch, schema, manifest_summary
        )

        # Update shared stats
        stats = pool.get_stats()
        generation_stats["rows_generated"] += success
        generation_stats["errors"] += errors
        generation_stats["skipped"] += stats["skipped"]
        generation_stats["total_cost"] += stats["total_cost"]

        # Track generation cost
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
        source_manager: SourceManager,
    ) -> bool:
        """Resume generation from a saved checkpoint."""

        work_items = checkpoints_to_work_items(checkpoint.work_items)

        # Filter to pending items only
        pending_indices = checkpoint.get_pending_indices()
        pending_items = [work_items[i] for i in pending_indices]

        if not pending_items:
            logger.info("[Pipeline] All work items already processed")
            await checkpoint_mgr.delete()
            self._handle_completion(db, project, version, cost_tracker)
            return True

        logger.info(
            f"[Pipeline] Resuming generation: "
            f"{len(pending_items)} pending of {len(work_items)} total"
        )

        manifest_summary = source_manager.get_manifest_summary()
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
                checkpoint_mgr.mark_processed(pending_indices[idx], success, row_id)
            ),
            blob_service_client=self.blob_service_client,
            mcp_tools=self._mcp_tools,
        )

        total_success, total_errors = await pool.process_work_items(
            pending_items, state.columns, manifest_summary
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

        total_processed = len(checkpoint.processed_indices) + total_success

        if total_processed > 0:
            await checkpoint_mgr.delete()
            self._handle_completion(db, project, version, cost_tracker)
            return True

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

    async def _load_uploaded_files(self, db: Session, project_id, workspace_dir: Path):
        """Download uploaded files from Azure to workspace."""
        files = (
            db.query(ProjectFile)
            .filter(
                ProjectFile.project_id == project_id,
                ProjectFile.deleted_at.is_(None),
                ProjectFile.status == "uploaded",
            )
            .all()
        )

        if not files:
            logger.info("[Pipeline] No uploaded files")
            return

        logger.info(f"[Pipeline] Loading {len(files)} uploaded files...")

        uploads_dir = workspace_dir / "uploads"

        for f in files:
            local_path = uploads_dir / f.filename
            try:
                if self.blob_service_client:
                    container = settings.azure_storage_container_name
                    blob_client = self.blob_service_client.get_blob_client(
                        container=container,
                        blob=f.blob_path,
                    )
                    with open(local_path, "wb") as file:
                        file.write(blob_client.download_blob().readall())
                    logger.info(f"[Pipeline] Downloaded: {f.filename}")
                else:
                    logger.warning(f"[Pipeline] No blob client, can't download {f.filename}")
            except Exception as e:
                logger.error(f"[Pipeline] Failed to download {f.filename}: {e}")

    def _build_mcp_tools(self, connectors) -> list:
        """Build MCP tool definitions from project connectors."""
        from dsl_api.crypto import decrypt_secret

        mcp_tools = []
        for conn in connectors:
            tool_def = {"type": "mcp", "server_label": conn.server_label}

            if conn.server_url:
                tool_def["server_url"] = conn.server_url
            if conn.connector_id:
                tool_def["connector_id"] = conn.connector_id
            if conn.server_description:
                tool_def["server_description"] = conn.server_description
            if conn.allowed_tools is not None:
                tool_def["allowed_tools"] = conn.allowed_tools

            # Decrypt auth
            if conn.authorization_encrypted:
                tool_def["authorization"] = decrypt_secret(conn.authorization_encrypted)
            if conn.headers_encrypted:
                import json as _json
                tool_def["headers"] = _json.loads(decrypt_secret(conn.headers_encrypted))

            tool_def["require_approval"] = "never"
            mcp_tools.append(tool_def)

        return mcp_tools

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
