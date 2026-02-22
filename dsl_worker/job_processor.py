"""
Job Processor — V4 three-tier pipeline

Flow:
1. Load conversation history and uploaded files
2. Run orchestrator (~5 turns): research, create instruction, delegate_topics
3. SAMPLE phase: each topic agent produces 1 row, system pauses for user review
4. GENERATE phase: topic agents resume, produce remaining rows
5. COMPLETE: all done

Three tiers: orchestrator → topic agents → row generators
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
from typing import Any, Dict, List, Optional
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
from dsl_worker.agents.topic_agent import TopicAgent
from dsl_worker.phases.row_generator import GenerationWorkerPool
from sandbox_service import SandboxClient

logger = logging.getLogger(__name__)

WORKER_ID = f"{socket.gethostname()}-{os.getpid()}"

# Batch size for work item queue → generation pool.
GENERATION_BATCH_SIZE = 5


class JobProcessor:
    """
    Job processor for the V4 three-tier pipeline.

    Flow: orchestrator → topic agents (parallel) → row generators (parallel)
    Phases: PLAN → SAMPLE → GENERATE → COMPLETE
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
        """Run the V4 three-tier pipeline."""

        # Create workspace (only downloads/ needed locally — uploads go direct to sandbox)
        workspace_dir = Path(f"./workspace_{project.id}_{version.id}")
        workspace_dir.mkdir(parents=True, exist_ok=True)
        (workspace_dir / "downloads").mkdir(exist_ok=True)
        self._workspace_dir = workspace_dir

        # Generate SAS URLs for uploaded files (sandbox service fetches them directly)
        uploaded_file_urls = self._generate_file_urls(state)

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
            )

        # --- Fresh run: V4 three-tier flow ---

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

        logger.info(f"[Pipeline] Starting V4 pipeline with {len(chat_history)} chat messages")

        # === Shared state ===
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
        topic_agents: List[TopicAgent] = []
        # Will be set when orchestrator calls delegate_topics
        delegate_config: Optional[Dict] = None
        delegate_event = asyncio.Event()

        # Work item index counter for checkpointing
        work_item_counter = [0]

        # --- Helper: dispatch work items from topic agents ---

        async def dispatch_work_items(items: List[Dict]) -> int:
            """Queue work items for generation, checkpoint them."""
            nonlocal generation_task

            for item in items:
                await checkpoint_mgr.add_work_item(item)
                await work_item_queue.put(item)
                work_item_counter[0] += 1

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
                            generation_stats=generation_stats,
                            uploaded_file_urls=uploaded_file_urls,
                        )
                    )

            return len(items)

        # --- Topic agent dispatch_rows callback ---
        # Called by topic agents when they dispatch rows.
        # Fills instruction template with seeds, adds context, creates work items.

        async def on_dispatch_rows(
            instruction_template: str,
            seeds: List[Dict],
            context: str,
            columns: List[Dict],
            topic_name: str,
        ) -> int:
            """Build work items from seeds and dispatch them."""
            work_items = []
            for seed in seeds:
                # Fill instruction template with seed values
                filled_instruction = instruction_template
                for var_name, var_value in seed.items():
                    filled_instruction = filled_instruction.replace(
                        f"{{{var_name}}}", str(var_value)
                    )

                work_items.append({
                    "instruction": filled_instruction,
                    "context": context,
                    "tags": {"topic": topic_name, **seed},
                })

            return await dispatch_work_items(work_items)

        # --- Orchestrator delegate_topics callback ---

        async def on_delegate_topics(config: Dict) -> Dict:
            """Store the delegation config and signal the pipeline to proceed."""
            nonlocal delegate_config
            delegate_config = config

            # Save the config as recipe for checkpoint debugging
            await checkpoint_mgr.set_recipe(json.dumps(config, indent=2))

            delegate_event.set()
            return {"status": "delegated", "topics": len(config.get("topics", []))}

        # --- Progress tracking ---
        progress_counters: Dict[str, Any] = {"phase": "planning"}
        last_progress_flush = time.time()

        def on_tool_call(agent_label: str, tool_name: str):
            nonlocal last_progress_flush

            # Phase transitions
            phase_map = {
                "run_subagent": "researching",
                "delegate_topics": "delegating",
                "done": "delegating",
                "dispatch_rows": "generating",
            }
            if tool_name in phase_map:
                progress_counters["phase"] = phase_map[tool_name]

            # Counter increments
            counter_map = {
                "brave_search": "searches",
                "open": "pages_viewed",
                "code_exec": "code_runs",
                "dispatch_rows": "topics_dispatched",
            }
            if tool_name in counter_map:
                key = counter_map[tool_name]
                progress_counters[key] = progress_counters.get(key, 0) + 1

            # Throttled DB flush (every 2s)
            now = time.time()
            if now - last_progress_flush >= 2.0:
                merged = {**progress_counters, **generation_stats}
                version.progress_detail = merged
                db.commit()
                last_progress_flush = now

        version.progress_detail = {"phase": "planning"}
        db.commit()

        # ====================================================================
        # Phase 1: PLAN — Run orchestrator (~5 turns)
        # ====================================================================

        logger.info("[Pipeline] Phase 1: PLAN — running orchestrator")

        orchestrator = OrchestratorAgent(
            chat_history=chat_history,
            columns=state.columns,
            num_samples=state.num_samples,
            openai_client=tracked_client,
            model=settings.research_model,
            workspace_dir=workspace_dir,
            on_delegate_topics=on_delegate_topics,
            uploaded_files=uploaded_files if uploaded_files else None,
            brave_api_key=settings.brave_api_key,
            sandbox=self._sandbox,
            stop_checker=stop_checker,
            blob_service_client=self.blob_service_client,
            project_id=project.id,
            on_tool_call=on_tool_call,
            uploaded_file_urls=uploaded_file_urls if uploaded_file_urls else None,
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

        # Check if orchestrator delegated topics
        if delegate_config is None:
            logger.error("[Pipeline] Orchestrator did not call delegate_topics")
            self._handle_force_stop(
                db, project, version, cost_tracker,
                "Orchestrator did not delegate any topics"
            )
            return False

        config = delegate_config
        instruction_template = config["instruction"]
        seed_variables = config["seed_variables"]
        shared_context = config.get("shared_context", "")
        topics = config["topics"]

        logger.info(
            f"[Pipeline] Orchestrator delegated {len(topics)} topics: "
            f"{[t['name'] for t in topics]}"
        )

        # ====================================================================
        # Phase 2: SAMPLE — Each topic produces 1 row, then pause for review
        # ====================================================================

        logger.info("[Pipeline] Phase 2: SAMPLE — running topic agents (1 row each)")
        version.progress_detail = {"phase": "sampling", "topics": len(topics)}
        db.commit()

        # Create topic agents — these are preserved across sample → generate
        topic_agents = []
        for topic in topics:
            agent = TopicAgent(
                topic_name=topic["name"],
                topic_briefing=topic["briefing"],
                instruction_template=instruction_template,
                seed_variables=seed_variables,
                target_count=topic.get("target", 10),
                shared_context=shared_context,
                columns=state.columns,
                openai_client=tracked_client,
                model=settings.research_model,
                workspace_dir=workspace_dir,
                on_dispatch_rows=on_dispatch_rows,
                brave_api_key=settings.brave_api_key,
                sandbox=self._sandbox,
                stop_checker=stop_checker,
                blob_service_client=self.blob_service_client,
                project_id=project.id,
                on_tool_call=on_tool_call,
                mcp_tools=mcp_tools,
            )
            topic_agents.append(agent)

        # Run all topic agents in sample mode (each produces 1 seed)
        sample_tasks = [
            asyncio.create_task(self._run_topic_agent_safe(agent, cost_tracker))
            for agent in topic_agents
        ]
        await asyncio.gather(*sample_tasks)

        # Check if we should stop
        if stop_checker():
            for agent in topic_agents:
                await agent.cleanup()
            await checkpoint_mgr.force_save()
            self._handle_pause(db, project, version, cost_tracker)
            return True

        can_continue, stop_reason = cost_tracker.check_balance_and_charge()
        if not can_continue:
            for agent in topic_agents:
                await agent.cleanup()
            await checkpoint_mgr.force_save()
            self._handle_force_stop(db, project, version, cost_tracker, stop_reason)
            return False

        # Wait for sample rows to be generated
        # (give the generation consumer a moment to process queued items)
        await asyncio.sleep(2)

        # --- Pause for user review ---
        logger.info("[Pipeline] SAMPLE phase complete — pausing for user review")

        # Post sample notification to chat
        db.refresh(version)
        sample_count = version.generated_count or 0
        sample_msg = ChatMessage(
            project_id=project.id,
            role="assistant",
            content=(
                f"Sample phase complete! Generated {sample_count} sample rows "
                f"({len(topics)} topics, 1 row each). "
                f"Review the samples and reply with feedback, or say 'continue' to proceed "
                f"with full generation."
            ),
        )
        db.add(sample_msg)
        db.commit()

        version.progress_detail = {"phase": "waiting_for_review", "sample_count": sample_count}
        db.commit()

        # Poll for user response (30 minute timeout for sample review)
        user_feedback = None
        for _ in range(1800):
            await asyncio.sleep(1)
            if stop_checker():
                for agent in topic_agents:
                    await agent.cleanup()
                await checkpoint_mgr.force_save()
                self._handle_pause(db, project, version, cost_tracker)
                return True

            new_msg = (
                db.query(ChatMessage)
                .filter(
                    ChatMessage.project_id == project.id,
                    ChatMessage.role == "user",
                    ChatMessage.created_at > sample_msg.created_at,
                )
                .order_by(ChatMessage.created_at.asc())
                .first()
            )
            if new_msg:
                user_feedback = new_msg.content
                logger.info(f"[Pipeline] Got user feedback ({len(user_feedback)} chars)")
                break

        if user_feedback is None:
            logger.info("[Pipeline] No user feedback within timeout — continuing")

        # ====================================================================
        # Phase 3: GENERATE — Resume topic agents for remaining rows
        # ====================================================================

        logger.info("[Pipeline] Phase 3: GENERATE — resuming topic agents")
        version.progress_detail = {"phase": "generating", "topics": len(topics)}
        db.commit()

        # Resume the SAME topic agents — all research context is preserved
        resume_tasks = [
            asyncio.create_task(
                self._resume_topic_agent_safe(agent, cost_tracker, user_feedback)
            )
            for agent in topic_agents
        ]
        await asyncio.gather(*resume_tasks)

        # Now clean up topic agents
        for agent in topic_agents:
            await agent.cleanup()

        # ====================================================================
        # Phase 4: COMPLETE — Wait for generation to finish
        # ====================================================================

        logger.info("[Pipeline] Phase 4: COMPLETE — waiting for generation to finish")

        # Wait for remaining generation
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

    async def _run_topic_agent_safe(
        self,
        agent: TopicAgent,
        cost_tracker: CostTracker,
    ) -> None:
        """Run a topic agent (sample phase) with error handling and cost tracking."""
        try:
            result = await agent.run()

            topic_cost = agent.cost_usd
            if topic_cost > 0:
                cost_tracker.add_cost(
                    phase=f"topic:{agent.topic_name}:sample",
                    cost_usd=topic_cost,
                    model=settings.research_model,
                )

            logger.info(
                f"[Pipeline] Topic '{agent.topic_name}' sample done: "
                f"cost=${topic_cost:.4f}, turns={result.turns_taken}"
            )
        except Exception as e:
            logger.error(f"[Pipeline] Topic '{agent.topic_name}' sample error: {e}")

    async def _resume_topic_agent_safe(
        self,
        agent: TopicAgent,
        cost_tracker: CostTracker,
        user_feedback: Optional[str],
    ) -> None:
        """Resume a topic agent (full phase) with error handling and cost tracking."""
        cost_before = agent.cost_usd

        try:
            result = await agent.resume(feedback=user_feedback)

            phase_cost = agent.cost_usd - cost_before
            if phase_cost > 0:
                cost_tracker.add_cost(
                    phase=f"topic:{agent.topic_name}:generate",
                    cost_usd=phase_cost,
                    model=settings.research_model,
                )

            logger.info(
                f"[Pipeline] Topic '{agent.topic_name}' generate done: "
                f"cost=${phase_cost:.4f}, turns={result.turns_taken}"
            )
        except Exception as e:
            logger.error(f"[Pipeline] Topic '{agent.topic_name}' generate error: {e}")

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
        generation_stats: Dict,
            uploaded_file_urls: Optional[Dict[str, str]] = None,
    ) -> None:
        """
        Background consumer: dequeue work items, batch them, and run
        through GenerationWorkerPool. Updates generation_stats in place.
        """
        batch: List[Dict] = []
        batch_start_index = 0

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
                            batch, schema, db, project, version,
                            tracked_client, cost_tracker, checkpoint_mgr,
                            workspace_dir, stop_checker, batch_start_index,
                            generation_stats, uploaded_file_urls=uploaded_file_urls,
                        )
                        batch_start_index += len(batch)
                        batch = []
                    continue

                if item is None:
                    # Poison pill — no more items coming
                    if batch:
                        await self._process_work_item_batch(
                            batch, schema, db, project, version,
                            tracked_client, cost_tracker, checkpoint_mgr,
                            workspace_dir, stop_checker, batch_start_index,
                            generation_stats, uploaded_file_urls=uploaded_file_urls,
                        )
                        batch_start_index += len(batch)
                        batch = []
                    break

                batch.append(item)

                if len(batch) >= GENERATION_BATCH_SIZE:
                    await self._process_work_item_batch(
                        batch, schema, db, project, version,
                        tracked_client, cost_tracker, checkpoint_mgr,
                        workspace_dir, stop_checker, batch_start_index,
                        generation_stats, uploaded_file_urls=uploaded_file_urls,
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
            uploaded_file_urls: Optional[Dict[str, str]] = None,
    ) -> None:
        """Process a batch of work items through the generation pool."""

        if not batch:
            return

        logger.info(f"[Generation] Processing batch of {len(batch)} work items")

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
            uploaded_file_urls=uploaded_file_urls,
            mcp_tools=self._mcp_tools,
        )

        success, errors = await pool.process_work_items(batch, schema)

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
        uploaded_file_urls = self._generate_file_urls(state)

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
            uploaded_file_urls=uploaded_file_urls if uploaded_file_urls else None,
            mcp_tools=self._mcp_tools,
        )

        total_success, total_errors = await pool.process_work_items(
            pending_items, state.columns
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

    def _generate_file_urls(self, state: ProjectState) -> Dict[str, str]:
        """Generate short-lived SAS URLs for uploaded files (no local download needed).

        Returns dict of filename -> SAS URL. The sandbox service will fetch
        these URLs directly into the session workspace.
        """
        if not state.files_snapshot:
            logger.info("[Pipeline] No uploaded files")
            return {}

        urls: Dict[str, str] = {}
        container = settings.azure_storage_container_name

        for f in state.files_snapshot:
            filename = f.get("filename")
            blob_path = f.get("blob_path")
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
