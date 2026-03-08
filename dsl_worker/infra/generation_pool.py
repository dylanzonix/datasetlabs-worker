"""
Row generation — worker pool for the generation phase.

The actual row generation agent is in dsl_worker.agents.row.
This module provides the pool that manages parallel workers consuming
work items from a queue.

V5+: Work items are filled templates from the pipeline. Each has a template
(filled with seed values), seed_values, research_context, and tags.
V4 compat: Also handles V4 format (instruction, context, schema, tags).
"""

import asyncio
import json
import logging
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from dsl_worker.agents.row import GeneratedRow, RowGeneratorAgent
from dsl_worker.config import settings

logger = logging.getLogger(__name__)


class GenerationWorkerPool:
    """Pool of workers that process work items using RowGeneratorAgent."""

    def __init__(
        self,
        workspace_dir: Path,
        openai_client: Any,
        db_session: Any,
        project_id: Any,
        version_id: Any,
        model: str = "",
        brave_api_key: Optional[str] = None,
        sandbox: Optional[Any] = None,
        num_workers: int = 10,
        stop_checker: Optional[Callable[[], bool]] = None,
        cost_tracker: Optional[Any] = None,
        checkpoint_callback: Optional[Callable[[int, bool, Optional[str]], Any]] = None,
        blob_service_client: Optional[Any] = None,
        uploaded_file_urls: Optional[Dict[str, str]] = None,
        mcp_tools: Optional[List[Dict]] = None,
        on_cost: Optional[Callable] = None,
        langfuse_parent: Optional[Any] = None,
        on_browser_started: Optional[Callable] = None,
    ):
        self.workspace_dir = Path(workspace_dir)
        self.openai_client = openai_client
        self.db = db_session
        self.project_id = project_id
        self.version_id = version_id
        self.model = model or settings.generation_model
        self.brave_api_key = brave_api_key
        self.sandbox = sandbox
        self.num_workers = num_workers
        self.stop_checker = stop_checker
        self.cost_tracker = cost_tracker
        self.checkpoint_callback = checkpoint_callback
        self.blob_service_client = blob_service_client
        self.uploaded_file_urls = uploaded_file_urls
        self.mcp_tools = mcp_tools or []
        self.on_cost = on_cost
        self.langfuse_parent = langfuse_parent
        self.on_browser_started = on_browser_started

        self._total_cost = 0.0
        self._rows_generated = 0
        self._skipped = 0
        self._errors = 0

    def _should_stop(self) -> bool:
        return self.stop_checker and self.stop_checker()

    async def process_work_items(
        self,
        work_items: List[Dict],
        default_schema: List[Dict],
    ) -> Tuple[int, int]:
        """
        Process work items in parallel. Returns (success_count, error_count).

        V5 work item format:
            - template: str — filled template (variables substituted)
            - seed_values: Dict — resolved variable values
            - research_context: str — orchestrator research findings
            - tags: Dict — metadata tags

        V4 work item format (backward compat):
            - instruction: str — the filled instruction template
            - context: str — supplementary notes from topic agent
            - schema: List[Dict] — overrides default_schema for this item
            - tags: Dict — metadata tags
        """
        if not work_items:
            logger.warning("[GenerationPool] No work items to process")
            return 0, 0

        logger.info(
            f"[GenerationPool] Processing {len(work_items)} work items "
            f"with {self.num_workers} workers"
        )

        # Build work queue: (index, work_item)
        queue: asyncio.Queue = asyncio.Queue()
        for i, item in enumerate(work_items):
            await queue.put((i, item))

        success_count = 0
        error_count = 0
        skip_count = 0
        lock = asyncio.Lock()

        async def worker():
            nonlocal success_count, error_count, skip_count

            agent = RowGeneratorAgent(
                openai_client=self.openai_client,
                model=self.model,
                workspace_dir=self.workspace_dir,
                brave_api_key=self.brave_api_key,
                sandbox=self.sandbox,
                stop_checker=self.stop_checker,
                blob_service_client=self.blob_service_client,
                project_id=self.project_id,
                uploaded_file_urls=self.uploaded_file_urls,
                mcp_tools=self.mcp_tools,
                on_cost=self.on_cost,
                langfuse_parent=self.langfuse_parent,
                on_browser_started=self.on_browser_started,
            )

            try:
                while True:
                    if self._should_stop():
                        break

                    try:
                        index, item = queue.get_nowait()
                    except asyncio.QueueEmpty:
                        break

                    # Detect V5 vs V4 format
                    is_v5 = "template" in item
                    tags = item.get("tags") or {}

                    # Use per-item langfuse parent so traces nest correctly
                    item_parent = item.get("langfuse_parent")
                    if item_parent is not None:
                        agent.langfuse_parent = item_parent
                    else:
                        agent.langfuse_parent = self.langfuse_parent

                    # Validate work item has content
                    if is_v5 and not item.get("template"):
                        logger.warning(f"[GenerationPool] Empty template at index {index}")
                        async with lock:
                            error_count += 1
                            self._errors += 1
                        if self.checkpoint_callback:
                            await self.checkpoint_callback(index, False, None)
                        continue
                    elif not is_v5 and not item.get("instruction"):
                        logger.warning(f"[GenerationPool] Empty instruction at index {index}")
                        async with lock:
                            error_count += 1
                            self._errors += 1
                        if self.checkpoint_callback:
                            await self.checkpoint_callback(index, False, None)
                        continue

                    max_attempts = 2
                    succeeded = False

                    for attempt in range(max_attempts):
                        try:
                            if is_v5:
                                result = await agent.generate(
                                    template=item["template"],
                                    seed=item.get("seed_values"),
                                    research_context=item.get("research_context"),
                                    schema=default_schema,
                                )
                            else:
                                result = await agent.generate(
                                    assignment=item.get("instruction", ""),
                                    schema=item.get("schema") or default_schema,
                                    dataset_brief=item.get("context", ""),
                                )

                            if result.success and result.row:
                                row_id = await self._save_row(result.row, tags=tags)

                                async with lock:
                                    success_count += 1
                                    self._rows_generated += 1

                                if success_count % 10 == 0:
                                    logger.info(
                                        f"[GenerationPool] Generated {success_count} rows..."
                                    )

                                if self.checkpoint_callback:
                                    await self.checkpoint_callback(index, True, row_id)
                                succeeded = True
                                break
                            elif result.skipped:
                                async with lock:
                                    skip_count += 1
                                    self._skipped += 1
                                logger.info(
                                    f"[GenerationPool] Row skipped at index {index}: "
                                    f"{result.skip_reason}"
                                )
                                if self.checkpoint_callback:
                                    await self.checkpoint_callback(index, True, None)
                                succeeded = True
                                break
                            else:
                                if attempt < max_attempts - 1:
                                    logger.warning(
                                        f"[GenerationPool] Failed (attempt {attempt + 1}/"
                                        f"{max_attempts}): {result.error} — retrying"
                                    )
                                else:
                                    async with lock:
                                        error_count += 1
                                        self._errors += 1
                                    logger.warning(
                                        f"[GenerationPool] Failed after {max_attempts} "
                                        f"attempts: {result.error}"
                                    )
                                    if self.checkpoint_callback:
                                        await self.checkpoint_callback(index, False, None)

                        except Exception as e:
                            if attempt < max_attempts - 1:
                                logger.warning(
                                    f"[GenerationPool] Error (attempt {attempt + 1}/"
                                    f"{max_attempts}): {e} — retrying"
                                )
                            else:
                                logger.error(
                                    f"[GenerationPool] Error after {max_attempts} "
                                    f"attempts: {e}"
                                )
                                async with lock:
                                    error_count += 1
                                    self._errors += 1
                                if self.checkpoint_callback:
                                    await self.checkpoint_callback(index, False, None)
            finally:
                await agent.cleanup()

        workers = [asyncio.create_task(worker()) for _ in range(self.num_workers)]
        await asyncio.gather(*workers)

        logger.info(
            f"[GenerationPool] Completed: {success_count} success, "
            f"{skip_count} skipped, {error_count} errors"
        )

        return success_count, error_count

    async def _save_row(
        self,
        row: Dict,
        tags: Optional[Dict] = None,
    ) -> Optional[str]:
        """Save generated row to database. Returns row ID."""
        from sqlalchemy import func as sql_func
        from dsl_api.models.project_version import ProjectVersion
        from dsl_api.models.sample import Sample

        row_json = json.dumps(row, ensure_ascii=False)
        clean_json = row_json.replace('\\u0000', '').replace('\x00', '')
        clean_row = json.loads(clean_json)

        row_id = str(uuid.uuid4())

        try:
            self.db.query(ProjectVersion).filter(
                ProjectVersion.id == self.version_id
            ).with_for_update().first()

            max_seq = (
                self.db.query(sql_func.max(Sample.seq))
                .filter(Sample.version_id == self.version_id)
                .scalar() or 0
            )

            sample = Sample(
                id=uuid.UUID(row_id),
                project_id=self.project_id,
                version_id=self.version_id,
                seq=max_seq + 1,
                row=clean_row,
                tags=tags or {},
            )
            self.db.add(sample)

            self.db.query(ProjectVersion).filter(
                ProjectVersion.id == self.version_id
            ).update(
                {ProjectVersion.generated_count: ProjectVersion.generated_count + 1},
                synchronize_session=False
            )

            self.db.commit()

            return row_id

        except Exception as e:
            logger.error(f"[GenerationPool] Save failed: {e}")
            self.db.rollback()
            raise

    def get_stats(self) -> Dict:
        """Get current stats."""
        return {
            "rows_generated": self._rows_generated,
            "skipped": self._skipped,
            "errors": self._errors,
            "total_cost": self._total_cost,
        }
