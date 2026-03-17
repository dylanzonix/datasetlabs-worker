"""
Job Processor — V6 pipeline

Flow:
1. Load conversation history and uploaded files
2. Create shared infrastructure (SeedProcessor, generation consumer)
3. Run orchestrator: research, write template, spawn yielders/synthesizers, submit seeds
4. Drain generation, backfill if needed
5. COMPLETE

V6: Single phase — the orchestrator stays in the loop and coordinates everything.
Generation runs in the background as seeds flow in.
"""

import asyncio
import json
import logging
import os
import shutil
import socket
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
from uuid import UUID

from openai import AsyncOpenAI
from sqlalchemy.orm import Session
from azure.storage.blob import BlobServiceClient, generate_blob_sas, BlobSasPermissions

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
    def _get_langfuse():
        return None

    def propagate_attributes(**kwargs):
        from contextlib import contextmanager
        @contextmanager
        def _noop():
            yield
        return _noop()
from dsl_worker.project_state import ProjectState
from dsl_worker.billing import CostTracker, TrackedOpenAIClient
from dsl_worker.checkpoint import CheckpointManager, checkpoints_to_work_items

from dsl_worker.agents import OrchestratorAgent
from dsl_worker.agents.row import DedupStore, RowGeneratorAgent
from dsl_worker.infra.pipeline import SeedProcessor
from dsl_worker.infra.generation_pool import GenerationWorkerPool
from sandbox_service import SandboxClient

logger = logging.getLogger(__name__)

WORKER_ID = f"{socket.gethostname()}-{os.getpid()}"


class JobProcessor:
    """
    Job processor for the V6 pipeline.

    Flow: orchestrator (dispatches subagents internally) → row generators (parallel)
    The orchestrator stays in the loop, spawning yielders/synthesizers and submitting
    seeds. Generation runs in the background as seeds flow in.
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

        # Sandbox client (shared HTTP connection — each agent creates its own session)
        self._sandbox: Optional[SandboxClient] = None
        self._workspace_dir: Optional[Path] = None
        # MCP tools for current job (set during _run_pipeline)
        self._mcp_tools: list = []

    def request_stop(self):
        """Request graceful stop."""
        logger.warning("[Worker] Stop requested")
        self.should_stop = True

    def _make_stop_checker(self, state: ProjectState, stop_event: asyncio.Event):
        """Create a stop checker that refreshes state before checking.

        Also sets stop_event so all agents wake up instantly once stop is detected,
        rather than each waiting for their own poll cycle.
        """
        def checker():
            state.refresh()
            should_stop = self.should_stop or state.paused
            if should_stop and not stop_event.is_set():
                stop_event.set()
            return should_stop
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
                compute_cost_per_credit=settings.compute_cost_per_credit,
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

            # Run pipeline wrapped in Langfuse trace.
            # propagate_attributes sets session_id/user_id/tags on every
            # observation created within the block, including sub-agent spans
            # in asyncio tasks that inherit the contextvars context.
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
                    ) as job_span:
                        result = await self._run_pipeline(
                            db, project, version, state, tracked_client, cost_tracker,
                            langfuse_parent=job_span,
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
        langfuse_parent: Optional[Any] = None,
    ) -> bool:
        """Run the V5 pipeline: orchestrator → seed yielders → row generators."""

        # Create workspace (only downloads/ needed locally — uploads go direct to sandbox)
        workspace_dir = Path(f"./workspace_{project.id}_{version.id}")
        workspace_dir.mkdir(parents=True, exist_ok=True)
        (workspace_dir / "downloads").mkdir(exist_ok=True)
        self._workspace_dir = workspace_dir

        # Generate SAS URLs for uploaded files (sandbox service fetches them directly)
        uploaded_file_urls = self._generate_file_urls(db, project.id)

        # Initialize sandbox client
        self._sandbox = SandboxClient(settings.sandbox_service_url, timeout=150.0)
        await self._sandbox.__aenter__()

        # Initialize checkpoint manager
        checkpoint_mgr = CheckpointManager(
            blob_service_client=self.blob_service_client,
            container_name=settings.azure_storage_container_name,
            project_id=project.id,
            version_id=version.id,
        )
        checkpoint = await checkpoint_mgr.initialize()

        stop_event = asyncio.Event()
        stop_checker = self._make_stop_checker(state, stop_event)

        # Per-turn cost callback — fires on every API call in every agent.
        async def on_cost(cost_usd: float, label: str):
            try:
                cost_tracker.add_cost(phase=label, cost_usd=cost_usd)
                cost_tracker.charge_if_needed()
                await checkpoint_mgr.add_cost(cost_usd)
            except Exception as e:
                logger.error(f"[Billing] on_cost callback error: {e}")

        # Check if resuming from execution phase
        if checkpoint.current_phase == "execution" and checkpoint.work_items:
            logger.info(
                f"[Pipeline] Resuming execution: "
                f"{len(checkpoint.processed_indices)}/{len(checkpoint.work_items)} done"
            )

            if checkpoint.total_cost_usd > 0:
                cost_tracker.seed_from_checkpoint(checkpoint.total_cost_usd)

            version.progress_detail = {"phase": "generating"}
            db.commit()
            return await self._run_generation_from_checkpoint(
                db, project, version, state, tracked_client, cost_tracker,
                checkpoint_mgr, checkpoint, workspace_dir, stop_checker,
                on_cost=on_cost,
            )

        # --- Fresh run ---

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

        logger.info(f"[Pipeline] Starting V6 pipeline with {len(chat_history)} chat messages")

        # === Shared state ===
        work_item_queue: asyncio.Queue = asyncio.Queue()
        generation_stats = {
            "rows_generated": 0,
            "errors": 0,
            "skipped": 0,
            "in_progress": 0,
            "total_cost": 0.0,
        }
        work_item_counter = [0]

        # Feedback context for re-planning iterations
        feedback_context = None

        # Check if this is a feedback iteration (version has previous config in recipe)
        if checkpoint.recipe:
            try:
                prev_config = json.loads(checkpoint.recipe)
                # Check for user feedback messages since version started
                feedback_msg = (
                    db.query(ChatMessage)
                    .filter(
                        ChatMessage.project_id == project.id,
                        ChatMessage.role == "user",
                        ChatMessage.created_at > version.started_at,
                    )
                    .order_by(ChatMessage.created_at.desc())
                    .first()
                )
                if feedback_msg:
                    feedback_context = {
                        "previous_config": prev_config,
                        "user_feedback": feedback_msg.content,
                    }
            except (json.JSONDecodeError, Exception):
                pass

        # --- Progress tracking ---
        progress_counters: Dict[str, Any] = {"phase": "orchestrating"}
        last_progress_flush = time.time()
        last_langfuse_flush = time.time()

        def on_tool_call(agent_label: str, tool_name: str):
            nonlocal last_progress_flush, last_langfuse_flush

            phase_map = {
                "research": "researching",
                "set_instructions": "designing_template",
                "harvest": "harvesting",
                "done": "finishing",
            }
            if tool_name in phase_map:
                progress_counters["phase"] = phase_map[tool_name]

            counter_map = {
                "brave_search": "searches",
                "open": "pages_viewed",
                "code_exec": "code_runs",
                "research": "research_subagents",
                "harvest": "harvesters_spawned",
                "save_page": "pages_saved",
            }
            if tool_name in counter_map:
                key = counter_map[tool_name]
                progress_counters[key] = progress_counters.get(key, 0) + 1

            now = time.time()
            if now - last_progress_flush >= 2.0:
                merged = {**progress_counters, **generation_stats}
                version.progress_detail = merged
                db.commit()
                last_progress_flush = now
            if now - last_langfuse_flush >= 10.0:
                lf = _get_langfuse()
                if lf:
                    lf.flush()
                last_langfuse_flush = now

        # --- Browser session tracking ---
        browser_sessions: Dict[str, Dict[str, str]] = {}

        def on_browser_started(live_url: str, session_id: str):
            browser_sessions[session_id] = {
                "live_url": live_url,
                "session_id": session_id,
            }
            progress_counters["browser_sessions"] = list(browser_sessions.values())
            logger.info(f"[Pipeline] Cloud browser started: {live_url}")

        def on_browser_stopped(session_id: str):
            browser_sessions.pop(session_id, None)
            progress_counters["browser_sessions"] = list(browser_sessions.values())
            logger.info(f"[Pipeline] Cloud browser stopped: {session_id}")

        version.progress_detail = {"phase": "orchestrating"}
        db.commit()

        # ====================================================================
        # V6: Create shared infrastructure BEFORE orchestrator
        # ====================================================================

        # --- Checkpoint callback for SeedProcessor ---

        async def on_seed_checkpoint(work_item: Dict):
            await checkpoint_mgr.add_work_item(work_item)
            work_item_counter[0] += 1

        # Create seed processor (V6: created before orchestrator, configured incrementally)
        seed_processor = SeedProcessor(
            work_queue=work_item_queue,
            on_checkpoint=on_seed_checkpoint,
            target_rows=state.num_samples,
            generation_stats=generation_stats,
        )

        # Shared dedup store — used by all row generators across the pipeline
        dedup_store = DedupStore()

        # Start generation consumer (runs in background the entire time)
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
                stop_event=stop_event,
                schema=state.columns,
                generation_stats=generation_stats,
                chat_history=chat_history,
                seed_processor=seed_processor,
                uploaded_file_urls=uploaded_file_urls,
                on_cost=on_cost,
                langfuse_parent=langfuse_parent,
                on_browser_started=on_browser_started,
                on_browser_stopped=on_browser_stopped,
                dedup_store=dedup_store,
            )
        )

        # ====================================================================
        # V6: Run orchestrator (it does everything — research, template,
        # spawning yielders/synthesizers, submitting seeds)
        # ====================================================================

        logger.info("[Pipeline] Running V6 orchestrator")

        orchestrator = OrchestratorAgent(
            chat_history=chat_history,
            columns=state.columns,
            num_samples=state.num_samples,
            openai_client=tracked_client,
            model=settings.research_model,
            workspace_dir=workspace_dir,
            seed_processor=seed_processor,
            generation_stats=generation_stats,
            yielder_model=settings.seed_yielder_model,
            uploaded_files=uploaded_files if uploaded_files else None,
            brave_api_key=settings.brave_api_key,
            sandbox=self._sandbox,
            stop_checker=stop_checker,
                stop_event=stop_event,
            blob_service_client=self.blob_service_client,
            project_id=project.id,
            on_tool_call=on_tool_call,
            on_cost=on_cost,
            uploaded_file_urls=uploaded_file_urls if uploaded_file_urls else None,
            mcp_tools=mcp_tools,
            feedback_context=feedback_context,
            langfuse_parent=langfuse_parent,
            on_browser_started=on_browser_started,
            on_browser_stopped=on_browser_stopped,
        )

        try:
            result = await orchestrator.run()
            logger.info(
                f"[Pipeline] Orchestrator finished: "
                f"cost=${orchestrator.cost_usd:.4f}, turns={result.turns_taken}"
            )
        finally:
            await orchestrator.cleanup()

        # Save recipe for checkpoint
        await checkpoint_mgr.set_phase("execution")

        logger.info(
            f"[Pipeline] Orchestrator complete. Seeds: {seed_processor.stats}"
        )

        # Signal generation consumer to drain
        await work_item_queue.put(None)  # poison pill
        try:
            await asyncio.wait_for(generation_task, timeout=300)
        except asyncio.TimeoutError:
            logger.warning("[Pipeline] Generation consumer drain timed out")
        except Exception as e:
            logger.warning(f"[Pipeline] Generation consumer error: {e}")

        # ====================================================================
        # COMPLETE — Check results, backfill if needed
        # ====================================================================

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

        # --- Backfill loop ---
        MAX_BACKFILL_ROUNDS = 3
        for backfill_round in range(MAX_BACKFILL_ROUNDS):
            db.refresh(version)
            generated = version.generated_count or 0
            shortfall = state.num_samples - generated
            if shortfall <= 0:
                break

            logger.info(
                f"[Pipeline] Backfill round {backfill_round + 1}: "
                f"{generated}/{state.num_samples} rows — "
                f"generating {shortfall} more"
            )

            # Build backfill work items
            instructions = seed_processor._instructions or "Generate a row for this dataset."
            backfill_items: List[Dict] = []
            for i in range(shortfall):
                backfill_items.append({
                    "instructions": (
                        f"{instructions}\n\n"
                        f"(Backfill row — generate a unique row not already covered.)"
                    ),
                    "candidate": None,
                    "research_context": seed_processor._research_context or "",
                    "tags": {"backfill": True},
                })

            backfill_start = work_item_counter[0]
            pool = GenerationWorkerPool(
                workspace_dir=workspace_dir,
                openai_client=tracked_client,
                db_session=db,
                project_id=project.id,
                version_id=version.id,
                chat_history=chat_history,
                dedup_store=dedup_store,
                model=settings.generation_model,
                brave_api_key=settings.brave_api_key,
                sandbox=self._sandbox,
                num_workers=settings.generation_parallel_samples,
                stop_checker=stop_checker,
                stop_event=stop_event,
                cost_tracker=cost_tracker,
                checkpoint_callback=lambda idx, success, row_id: asyncio.create_task(
                    checkpoint_mgr.mark_processed(backfill_start + idx, success, row_id)
                ),
                blob_service_client=self.blob_service_client,
                uploaded_file_urls=uploaded_file_urls,
                mcp_tools=self._mcp_tools,
                on_cost=on_cost,
                langfuse_parent=langfuse_parent,
                on_browser_started=on_browser_started,
                on_browser_stopped=on_browser_stopped,
            )
            success, errors = await pool.process_work_items(
                backfill_items, state.columns,
            )
            stats = pool.get_stats()
            generation_stats["rows_generated"] += success
            generation_stats["errors"] += errors
            generation_stats["skipped"] += stats["skipped"]
            work_item_counter[0] += len(backfill_items)

            state.refresh()
            if state.paused:
                await checkpoint_mgr.force_save()
                self._handle_pause(db, project, version, cost_tracker)
                return True

            can_continue, stop_reason = cost_tracker.check_balance_and_charge()
            if not can_continue:
                await checkpoint_mgr.force_save()
                self._handle_force_stop(
                    db, project, version, cost_tracker, stop_reason
                )
                return False

        db.refresh(version)
        generated = version.generated_count or 0
        if generated > 0:
            if generated < state.num_samples:
                logger.warning(
                    f"[Pipeline] Completed with {generated}/{state.num_samples} "
                    f"rows after {MAX_BACKFILL_ROUNDS} backfill rounds"
                )
            await checkpoint_mgr.set_phase("completed")
            await checkpoint_mgr.force_save()
            self._handle_completion(db, project, version, cost_tracker)
            return True

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
        stop_event: asyncio.Event,
        schema: List[Dict],
        generation_stats: Dict,
        chat_history: Optional[List[Dict]] = None,
        seed_processor: Optional[Any] = None,
        uploaded_file_urls: Optional[Dict[str, str]] = None,
        on_cost: Optional[Callable] = None,
        langfuse_parent: Optional[Any] = None,
        on_browser_started: Optional[Callable] = None,
        on_browser_stopped: Optional[Callable] = None,
        dedup_store: Optional[DedupStore] = None,
    ) -> None:
        """
        Background consumer: dequeue work items and process them with a
        semaphore for concurrency control. Each item starts immediately
        when a slot is available — no batching, no head-of-line blocking.
        """
        concurrency = settings.generation_parallel_samples
        semaphore = asyncio.Semaphore(concurrency)
        in_flight: list[asyncio.Task] = []
        item_index = 0
        save_lock = asyncio.Lock()

        dedup_store = dedup_store or DedupStore()

        # Shared row saver (DB writes must be serialized)
        row_saver = GenerationWorkerPool(
            workspace_dir=workspace_dir,
            openai_client=tracked_client,
            db_session=db,
            project_id=project.id,
            version_id=version.id,
        )

        async def process_one(index: int, item: Dict):
            """Process a single work item under the semaphore."""
            async with semaphore:
                if stop_checker():
                    return

                agent = RowGeneratorAgent(
                    openai_client=tracked_client,
                    model=settings.generation_model,
                    workspace_dir=workspace_dir,
                    chat_history=chat_history or [],
                    dedup_store=dedup_store,
                    brave_api_key=settings.brave_api_key,
                    sandbox=self._sandbox,
                    stop_checker=stop_checker,
                    stop_event=stop_event,
                    blob_service_client=self.blob_service_client,
                    project_id=project.id,
                    uploaded_file_urls=uploaded_file_urls,
                    mcp_tools=self._mcp_tools,
                    on_cost=on_cost,
                    langfuse_parent=item.get("langfuse_parent") or langfuse_parent,
                    on_browser_started=on_browser_started,
                    on_browser_stopped=on_browser_stopped,
                )

                try:
                    tags = item.get("tags") or {}
                    instructions = item.get("instructions", "")
                    candidate = item.get("candidate")

                    if not instructions:
                        logger.warning(f"[Generation] Empty instructions at index {index}")
                        generation_stats["errors"] += 1
                        if checkpoint_mgr:
                            await checkpoint_mgr.mark_processed(index, False, None)
                        return

                    max_attempts = 2
                    for attempt in range(max_attempts):
                        try:
                            result = await agent.generate(
                                instructions=instructions,
                                candidate=candidate,
                                schema=schema,
                                source_url=item.get("source_url"),
                                source_content=item.get("source_content"),
                            )

                            if result.success and result.row:
                                async with save_lock:
                                    row_id = await row_saver._save_row(
                                        result.row, tags=tags,
                                    )
                                generation_stats["rows_generated"] += 1

                                if generation_stats["rows_generated"] % 10 == 0:
                                    logger.info(
                                        f"[Generation] Generated "
                                        f"{generation_stats['rows_generated']} rows..."
                                    )

                                if checkpoint_mgr:
                                    await checkpoint_mgr.mark_processed(index, True, row_id)
                                break

                            elif result.skipped:
                                generation_stats["skipped"] += 1
                                logger.info(
                                    f"[Generation] Row skipped at index {index}: "
                                    f"{result.skip_reason}"
                                )
                                if checkpoint_mgr:
                                    await checkpoint_mgr.mark_processed(index, True, None)
                                break

                            else:
                                if attempt < max_attempts - 1:
                                    logger.warning(
                                        f"[Generation] Failed (attempt {attempt + 1}/"
                                        f"{max_attempts}): {result.error} — retrying"
                                    )
                                else:
                                    generation_stats["errors"] += 1
                                    logger.warning(
                                        f"[Generation] Failed after {max_attempts} "
                                        f"attempts: {result.error}"
                                    )
                                    if checkpoint_mgr:
                                        await checkpoint_mgr.mark_processed(
                                            index, False, None,
                                        )

                        except Exception as e:
                            if attempt < max_attempts - 1:
                                logger.warning(
                                    f"[Generation] Error (attempt {attempt + 1}/"
                                    f"{max_attempts}): {e} — retrying"
                                )
                            else:
                                logger.error(
                                    f"[Generation] Error after {max_attempts} "
                                    f"attempts: {e}"
                                )
                                generation_stats["errors"] += 1
                                if checkpoint_mgr:
                                    await checkpoint_mgr.mark_processed(
                                        index, False, None,
                                    )
                finally:
                    await agent.cleanup()

        try:
            while True:
                if stop_checker():
                    break

                try:
                    item = await asyncio.wait_for(work_item_queue.get(), timeout=5.0)
                except asyncio.TimeoutError:
                    continue

                if item is None:
                    break

                # Check balance AFTER dequeuing — don't kill the consumer
                # while it's idle waiting for seeds from the orchestrator.
                can_continue, _ = cost_tracker.check_balance_and_charge()
                if not can_continue:
                    break

                task = asyncio.create_task(process_one(item_index, item))
                in_flight.append(task)
                item_index += 1

                # Clean up completed tasks periodically
                in_flight = [t for t in in_flight if not t.done()]

        except Exception as e:
            logger.error(f"[Generation] Consumer error: {e}")

        # Wait for all in-flight items to finish
        if in_flight:
            logger.info(f"[Generation] Waiting for {len(in_flight)} in-flight items...")
            await asyncio.gather(*in_flight, return_exceptions=True)

        logger.info(
            f"[Generation] Consumer done: "
            f"{generation_stats['rows_generated']} success, "
            f"{generation_stats['errors']} errors, "
            f"{generation_stats['skipped']} skipped"
        )

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
        on_cost: Optional[Callable] = None,
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

        concurrency = settings.generation_parallel_samples

        # Generate SAS URLs for uploaded files (needed by row generators)
        uploaded_file_urls = self._generate_file_urls(db, project.id)

        # Langfuse parent not available for checkpoint resume (no active span)
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
                stop_event=stop_event,
            cost_tracker=cost_tracker,
            checkpoint_callback=lambda idx, success, row_id: asyncio.create_task(
                checkpoint_mgr.mark_processed(pending_indices[idx], success, row_id)
            ),
            blob_service_client=self.blob_service_client,
            uploaded_file_urls=uploaded_file_urls if uploaded_file_urls else None,
            mcp_tools=self._mcp_tools,
            on_cost=on_cost,
        )

        total_success, total_errors = await pool.process_work_items(
            pending_items, state.columns
        )

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

    async def _drain_generation_consumer(
        self,
        work_item_queue: asyncio.Queue,
        generation_task: Optional[asyncio.Task],
    ) -> None:
        """Drain the generation consumer by sending a poison pill and waiting."""
        if generation_task and not generation_task.done():
            await work_item_queue.put(None)  # poison pill
            try:
                await asyncio.wait_for(generation_task, timeout=120)
            except asyncio.TimeoutError:
                logger.warning("[Pipeline] Generation consumer drain timed out")
            except Exception as e:
                logger.warning(f"[Pipeline] Generation consumer drain error: {e}")

    def _create_feedback_version(
        self,
        db: Session,
        project: Project,
        old_version: ProjectVersion,
    ) -> ProjectVersion:
        """Create a new version for a feedback iteration.

        The old version is finalized with status 'succeeded' — its sample
        rows are preserved. The new version starts fresh.
        """
        from sqlalchemy import func as sql_func

        max_num = (
            db.query(sql_func.max(ProjectVersion.version_number))
            .filter(ProjectVersion.project_id == project.id)
            .scalar() or 0
        )

        new_version = ProjectVersion(
            project_id=project.id,
            version_number=max_num + 1,
            num_samples=old_version.num_samples,
            generation_prompt=old_version.generation_prompt,
            columns=old_version.columns,
            files_snapshot=old_version.files_snapshot,
            examples_snapshot=old_version.examples_snapshot,
            status="running",
            started_at=datetime.now(timezone.utc),
        )
        db.add(new_version)
        db.flush()

        old_version.status = "succeeded"
        old_version.finished_at = datetime.now(timezone.utc)
        old_version.progress_detail = None

        project.current_version_id = new_version.id
        db.commit()

        logger.info(
            f"[Pipeline] Created feedback version {new_version.version_number} "
            f"(old v{old_version.version_number} finalized)"
        )

        return new_version

    def _load_chat_history(self, db: Session, project_id: UUID) -> List[Dict[str, str]]:
        """Load chat messages and format for the orchestrator.

        Includes applied_changes data (plan, questions, columns) so the
        orchestrator has full context from the consultation chat.
        """
        messages = (
            db.query(ChatMessage)
            .filter(ChatMessage.project_id == project_id)
            .order_by(ChatMessage.created_at.asc())
            .all()
        )

        history = []
        for msg in messages:
            if msg.role not in ("user", "assistant"):
                continue

            content = msg.content or ""

            # Append structured data from applied_changes so the orchestrator
            # sees the plan, questions, and column definitions from the chat.
            if msg.applied_changes and isinstance(msg.applied_changes, dict):
                changes = msg.applied_changes.get("changes", {})

                if "plan" in changes:
                    plan = changes["plan"]
                    if isinstance(plan, dict):
                        overview = plan.get("overview", "")
                        if overview:
                            content += f"\n\n<plan>\n{overview}\n</plan>"
                    elif isinstance(plan, str):
                        content += f"\n\n<plan>\n{plan}\n</plan>"

                if "columns" in changes:
                    cols = changes["columns"]
                    if isinstance(cols, list) and cols:
                        content += "\n\n<columns>\n" + json.dumps(cols, indent=2) + "\n</columns>"

                if "questions" in changes:
                    qs = changes["questions"]
                    if isinstance(qs, list):
                        q_blocks = []
                        for q in qs:
                            if isinstance(q, dict):
                                question_text = q.get("question", q.get("label", "?"))
                                q_type = q.get("type", "single_choice")
                                options = q.get("options", [])
                                block = question_text
                                if q_type == "multi_choice":
                                    block += " (pick all that apply)"
                                if options:
                                    block += "\n" + "\n".join(
                                        f"  - {opt}" for opt in options
                                    )
                                q_blocks.append(block)
                            elif isinstance(q, str):
                                q_blocks.append(q)
                        if q_blocks:
                            content += "\n\n<questions>\n" + "\n\n".join(q_blocks) + "\n</questions>"

            history.append({
                "role": msg.role,
                "content": content,
            })

        return history

    def _generate_file_urls(self, db: Session, project_id) -> Dict[str, str]:
        """Generate short-lived SAS URLs for uploaded files (no local download needed).

        Returns dict of filename -> SAS URL. The sandbox service will fetch
        these URLs directly into the session workspace.
        """
        project_files = (
            db.query(ProjectFile)
            .filter(
                ProjectFile.project_id == project_id,
                ProjectFile.deleted_at.is_(None),
                ProjectFile.status == "uploaded",
            )
            .all()
        )
        if not project_files:
            logger.info("[Pipeline] No uploaded files")
            return {}

        urls: Dict[str, str] = {}
        container = settings.azure_storage_container_name

        for f in project_files:
            filename = f.filename
            blob_path = f.blob_path
            if not filename or not blob_path:
                continue

            try:
                sas_token = generate_blob_sas(
                    account_name=settings.azure_storage_account_name,
                    account_key=settings.azure_storage_account_key,
                    container_name=container,
                    blob_name=blob_path,
                    permission=BlobSasPermissions(read=True),
                    expiry=datetime.now(timezone.utc) + timedelta(hours=1),
                )
                blob_url = (
                    f"https://{settings.azure_storage_account_name}.blob.core.windows.net"
                    f"/{container}/{blob_path}?{sas_token}"
                )
                urls[filename] = blob_url
                logger.info(f"[Pipeline] Generated SAS URL for: {filename}")
            except Exception as e:
                logger.error(f"[Pipeline] Failed to generate SAS URL for {filename}: {e}")

        return urls

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
                await self._sandbox.__aexit__(None, None, None)
            except Exception as e:
                logger.warning(f"Sandbox cleanup error: {e}")
            self._sandbox = None

        # Clean up workspace directory
        workspace_dir = getattr(self, "_workspace_dir", None)
        if workspace_dir and workspace_dir.exists():
            try:
                shutil.rmtree(workspace_dir)
                logger.info(f"Cleaned up workspace: {workspace_dir}")
            except Exception as e:
                logger.warning(f"Workspace cleanup error: {e}")
            self._workspace_dir = None

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
