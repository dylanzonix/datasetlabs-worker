"""
Row generation — worker pool and bucket tracking for the generation phase.

The actual row generation agent is in dsl_worker.agents.row_generator.
This module provides the pool that manages parallel workers and the
bucket tracker for quota management.
"""

import asyncio
import json
import logging
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from dsl_worker.agents.row_generator import GeneratedRow, RowGeneratorAgent
from dsl_worker.config import settings
from dsl_worker.phases.research_tools import Seed

logger = logging.getLogger(__name__)


class BucketTracker:
    """
    Tracks per-bucket quotas during generation.

    Quotas are computed from bucket weights and total num_samples.
    Thread-safe via asyncio.Lock.
    """

    def __init__(self, plan: Dict, num_samples: int) -> None:
        self._lock = asyncio.Lock()
        self._quotas: Dict[str, int] = {}
        self._counts: Dict[str, int] = {}
        self._skips: Dict[str, int] = {}

        buckets = plan.get("buckets", [])
        if not buckets:
            # No buckets — single implicit bucket
            self._quotas["_default"] = num_samples
            self._counts["_default"] = 0
            self._skips["_default"] = 0
            return

        # Compute quotas from weights
        total_weight = sum(b.get("weight", 1.0) for b in buckets)
        allocated = 0
        for i, b in enumerate(buckets):
            bid = b.get("id", f"bucket_{i}")
            weight = b.get("weight", 1.0)
            quota = round(num_samples * weight / total_weight)
            self._quotas[bid] = quota
            self._counts[bid] = 0
            self._skips[bid] = 0
            allocated += quota

        # Give remainder to last bucket
        if allocated != num_samples and buckets:
            last_id = buckets[-1].get("id", f"bucket_{len(buckets)-1}")
            self._quotas[last_id] += num_samples - allocated

    async def should_process(self, bucket_id: Optional[str]) -> bool:
        """Check if a bucket still needs rows."""
        async with self._lock:
            bid = bucket_id or "_default"
            if bid not in self._quotas:
                # Unknown bucket — process it (don't silently drop)
                return True
            return self._counts[bid] < self._quotas[bid]

    async def record_success(self, bucket_id: Optional[str]) -> None:
        """Record a successful row for a bucket."""
        async with self._lock:
            bid = bucket_id or "_default"
            if bid not in self._counts:
                self._counts[bid] = 0
            self._counts[bid] += 1

    async def record_skip(self, bucket_id: Optional[str]) -> None:
        """Record a skipped seed for a bucket."""
        async with self._lock:
            bid = bucket_id or "_default"
            if bid not in self._skips:
                self._skips[bid] = 0
            self._skips[bid] += 1

    async def is_complete(self) -> bool:
        """Check if all bucket quotas are met."""
        async with self._lock:
            return all(
                self._counts.get(bid, 0) >= quota
                for bid, quota in self._quotas.items()
            )

    async def get_status(self) -> Dict:
        """Get per-bucket status report."""
        async with self._lock:
            status = {}
            for bid, quota in self._quotas.items():
                status[bid] = {
                    "quota": quota,
                    "completed": self._counts.get(bid, 0),
                    "skipped": self._skips.get(bid, 0),
                    "remaining": max(0, quota - self._counts.get(bid, 0)),
                }
            return status

    async def pre_populate(self, bucket_id: Optional[str], count: int) -> None:
        """Pre-populate counts (for checkpoint resume)."""
        async with self._lock:
            bid = bucket_id or "_default"
            if bid not in self._counts:
                self._counts[bid] = 0
            self._counts[bid] += count


class GenerationWorkerPool:
    """Pool of workers that process seeds using RowGeneratorAgent."""

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
        plan: Optional[Dict] = None,
        bucket_tracker: Optional[BucketTracker] = None,
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
        self.plan = plan
        self.bucket_tracker = bucket_tracker
        self.blob_service_client = blob_service_client
        self.mcp_tools = mcp_tools or []

        self._total_cost = 0.0
        self._rows_generated = 0
        self._skipped = 0
        self._errors = 0

    def _should_stop(self) -> bool:
        return self.stop_checker and self.stop_checker()

    def _get_pipeline_instructions(self, bucket_id: Optional[str]) -> str:
        """Get pipeline instructions for a given bucket from the plan."""
        if not self.plan:
            return ""

        pipelines = self.plan.get("pipelines", {})
        buckets = self.plan.get("buckets", [])

        # Find which pipeline this bucket uses
        pipeline_id = None
        for b in buckets:
            if b.get("id") == bucket_id:
                pipeline_id = b.get("pipeline_id")
                break

        # If no bucket match or no buckets, use first pipeline
        if pipeline_id and pipeline_id in pipelines:
            return pipelines[pipeline_id].get("instructions", "")
        elif pipelines:
            first_key = next(iter(pipelines))
            return pipelines[first_key].get("instructions", "")

        return ""

    async def process_seeds(self, seeds: List[Seed], schema: List[Dict]) -> Tuple[int, int]:
        """Process seeds directly. Returns (success_count, error_count)."""
        if not seeds:
            logger.warning("[GenerationPool] No seeds to process")
            return 0, 0

        logger.info(f"[GenerationPool] Processing {len(seeds)} seeds with {self.num_workers} workers")

        # Build work queue with (index, seed_content, pipeline_instructions, bucket_id, scope_id)
        queue: asyncio.Queue = asyncio.Queue()
        for i, seed in enumerate(seeds):
            pipeline_instructions = self._get_pipeline_instructions(seed.bucket_id)
            await queue.put((i, seed.content, pipeline_instructions, seed.bucket_id, seed.scope_id))

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
                        index, seed_content, pipeline_instructions, bucket_id, scope_id = queue.get_nowait()
                    except asyncio.QueueEmpty:
                        break

                    # Check bucket quota before processing
                    if self.bucket_tracker:
                        if not await self.bucket_tracker.should_process(bucket_id):
                            if self.checkpoint_callback:
                                await self.checkpoint_callback(index, True, None)
                            continue

                    try:
                        result = await agent.generate(
                            seed=seed_content,
                            pipeline_instructions=pipeline_instructions,
                            schema=schema,
                        )

                        self._total_cost += result.cost_usd

                        if result.skipped:
                            async with lock:
                                skip_count += 1
                                self._skipped += 1
                            if self.bucket_tracker:
                                await self.bucket_tracker.record_skip(bucket_id)
                            logger.debug(f"[GenerationPool] Skipped: {result.skip_reason}")
                            if self.checkpoint_callback:
                                await self.checkpoint_callback(index, True, None)

                        elif result.success and result.row:
                            row_id = await self._save_row(
                                result.row, bucket_id=bucket_id, scope_id=scope_id,
                            )

                            if self.bucket_tracker:
                                await self.bucket_tracker.record_success(bucket_id)

                            async with lock:
                                success_count += 1
                                self._rows_generated += 1

                            if success_count % 10 == 0:
                                logger.info(f"[GenerationPool] Generated {success_count} rows...")

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
                        logger.error(f"[GenerationPool] Error processing seed: {e}")
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
        bucket_id: Optional[str] = None,
        scope_id: Optional[str] = None,
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

            tags = {}
            if bucket_id:
                tags["bucket_id"] = bucket_id
            if scope_id:
                tags["scope_id"] = scope_id

            sample = Sample(
                id=uuid.UUID(row_id),
                project_id=self.project_id,
                version_id=self.version_id,
                seq=max_seq + 1,
                row=clean_row,
                tags=tags,
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
