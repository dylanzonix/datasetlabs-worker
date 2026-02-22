"""
Row generation — worker pool for the generation phase.

The actual row generation agent is in dsl_worker.agents.row_generator.
This module provides the pool that manages parallel workers consuming
work items from a queue.

V4: Work items are assignments from topic agents. Each has an instruction
(filled template), optional context (from topic agent), optional schema
override, and optional tags.
"""

import asyncio
import json
import logging
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from dsl_worker.agents.row_generator import GeneratedRow, RowGeneratorAgent
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
        mcp_tools: Optional[List[Dict]] = None,
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
        self.mcp_tools = mcp_tools or []

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

        Each work item is a dict with:
            - instruction: str (required) — the filled instruction template
            - context: str (optional) — supplementary notes from topic agent
            - schema: List[Dict] (optional) — overrides default_schema for this item
            - tags: Dict (optional) — metadata tags to attach to the saved row
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
                mcp_tools=self.mcp_tools,
            )

            try:
                while True:
                    if self._should_stop():
                        break

                    try:
                        index, item = queue.get_nowait()
                    except asyncio.QueueEmpty:
                        break

                    instruction = item.get("instruction", "")
                    context = item.get("context", "")
                    schema = item.get("schema") or default_schema
                    tags = item.get("tags") or {}

                    if not instruction:
                        logger.warning(f"[GenerationPool] Empty instruction at index {index}")
                        async with lock:
                            error_count += 1
                            self._errors += 1
                        if self.checkpoint_callback:
                            await self.checkpoint_callback(index, False, None)
                        continue

                    try:
                        result = await agent.generate(
                            instruction=instruction,
                            schema=schema,
                            context=context,
                        )

                        self._total_cost += result.cost_usd

                        if result.skipped:
                            async with lock:
                                skip_count += 1
                                self._skipped += 1
                            logger.debug(f"[GenerationPool] Skipped: {result.skip_reason}")
                            if self.checkpoint_callback:
                                await self.checkpoint_callback(index, True, None)

                        elif result.success and result.row:
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
                        else:
                            async with lock:
                                error_count += 1
                                self._errors += 1
                            logger.warning(f"[GenerationPool] Failed: {result.error}")

                            if self.checkpoint_callback:
                                await self.checkpoint_callback(index, False, None)

                    except Exception as e:
                        logger.error(f"[GenerationPool] Error processing work item: {e}")
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
