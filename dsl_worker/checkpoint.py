"""
Pipeline Checkpointing

Handles saving and restoring pipeline state for pause/resume.

Checkpoints are stored as JSON in Azure Blob Storage.

V9: Work items stored as-is (pass-through). No format remapping.
Phases: orchestrator | execution | completed.
"""

import asyncio
import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
from uuid import UUID

logger = logging.getLogger(__name__)


@dataclass
class PipelineCheckpoint:
    """
    Full pipeline state.

    Designed to be:
    - Small (no duplicate data)
    - Complete (can fully resume from it)
    - Debuggable (human-readable JSON)
    """

    version: str = "3.0"

    # Identity
    project_id: str = ""
    version_id: str = ""

    # Timestamps
    created_at: str = ""
    updated_at: str = ""

    # Phase tracking: 'orchestrator' | 'execution' | 'completed'
    current_phase: str = "orchestrator"

    # Work items — stored as-is from the pipeline (opaque dicts).
    # Each item gets "status" and "row_id" fields added for tracking.
    work_items: List[Dict] = field(default_factory=list)

    # Generation progress — just indices, actual rows are in DB
    processed_indices: List[int] = field(default_factory=list)

    # Recipe (stored for debugging/resume context)
    recipe: Optional[str] = None

    # Cost tracking
    total_cost_usd: float = 0.0

    # Error tracking
    errors: List[Dict] = field(default_factory=list)

    def to_json(self) -> str:
        """Serialize to JSON string."""
        data = asdict(self)
        return json.dumps(data, indent=2, default=str)

    @classmethod
    def from_json(cls, json_str: str) -> 'PipelineCheckpoint':
        """Deserialize from JSON string. Handles legacy formats."""
        data = json.loads(json_str)

        # --- Legacy migration ---

        # v1: "seeds" field → work_items
        if "seeds" in data and "work_items" not in data:
            data["work_items"] = [
                {"instructions": s.get("content", ""), "status": s.get("status", "pending"), "row_id": s.get("row_id")}
                for s in data.pop("seeds")
            ]

        # Legacy phase names
        phase = data.get("current_phase", "")
        if phase == "research":
            data["current_phase"] = "orchestrator"
        elif phase in ("sample", "generation"):
            data["current_phase"] = "execution"

        # Legacy field names
        if "processed_seed_indices" in data and "processed_indices" not in data:
            data["processed_indices"] = data.pop("processed_seed_indices")

        # Remove any fields not in our dataclass
        known_fields = {f.name for f in cls.__dataclass_fields__.values()}
        for key in list(data.keys()):
            if key not in known_fields:
                data.pop(key)

        # Bump version
        data["version"] = "3.0"

        return cls(**data)

    def add_work_item(self, item: Dict) -> None:
        """Add a work item (stored as-is with status tracking)."""
        item.setdefault("status", "pending")
        item.setdefault("row_id", None)
        self.work_items.append(item)

    def mark_processed(self, index: int, success: bool, row_id: Optional[str] = None) -> None:
        """Mark a work item as processed."""
        if index not in self.processed_indices:
            self.processed_indices.append(index)

        if index < len(self.work_items):
            self.work_items[index]["status"] = "completed" if success else "failed"
            if row_id:
                self.work_items[index]["row_id"] = row_id

    def get_pending_indices(self) -> List[int]:
        """Get indices of work items not yet processed."""
        processed = set(self.processed_indices)
        return [i for i in range(len(self.work_items)) if i not in processed]

    @property
    def stats(self) -> Dict:
        """Get summary stats."""
        return {
            "phase": self.current_phase,
            "total_work_items": len(self.work_items),
            "processed": len(self.processed_indices),
            "pending": len(self.work_items) - len(self.processed_indices),
            "has_recipe": self.recipe is not None,
            "total_cost_usd": self.total_cost_usd,
            "errors": len(self.errors),
        }


class CheckpointManager:
    """
    Manages pipeline checkpoints in Azure Blob Storage.

    Auto-saves based on pending update count or time interval.
    """

    def __init__(
        self,
        blob_service_client: Any,
        container_name: str,
        project_id: UUID,
        version_id: UUID,
        auto_save_interval: int = 30,  # seconds
        auto_save_count: int = 10,     # pending updates
    ):
        self.blob_client = blob_service_client
        self.container_name = container_name
        self.project_id = project_id
        self.version_id = version_id

        self.auto_save_interval = auto_save_interval
        self.auto_save_count = auto_save_count

        # Paths
        self._checkpoint_path = f"checkpoints/{project_id}/{version_id}/state.json"

        # State
        self._checkpoint: Optional[PipelineCheckpoint] = None
        self._lock = asyncio.Lock()
        self._pending_updates = 0
        self._last_save_time = datetime.now(timezone.utc)

    async def initialize(self) -> PipelineCheckpoint:
        """Load existing checkpoint or create new one."""
        existing = await self.load()

        if existing:
            logger.info(
                f"[CheckpointManager] Resuming from checkpoint: "
                f"phase={existing.current_phase}, "
                f"work_items={len(existing.work_items)}, "
                f"processed={len(existing.processed_indices)}"
            )
            self._checkpoint = existing
        else:
            self._checkpoint = PipelineCheckpoint(
                project_id=str(self.project_id),
                version_id=str(self.version_id),
                created_at=datetime.now(timezone.utc).isoformat(),
            )
            logger.info("[CheckpointManager] Created new checkpoint")

        return self._checkpoint

    @property
    def checkpoint(self) -> PipelineCheckpoint:
        if self._checkpoint is None:
            raise RuntimeError("CheckpointManager not initialized")
        return self._checkpoint

    async def load(self) -> Optional[PipelineCheckpoint]:
        """Load checkpoint from blob storage."""
        try:
            blob = self.blob_client.get_blob_client(
                container=self.container_name,
                blob=self._checkpoint_path
            )
            json_data = blob.download_blob().readall().decode('utf-8')
            return PipelineCheckpoint.from_json(json_data)
        except Exception as e:
            logger.debug(f"[CheckpointManager] No checkpoint found: {e}")
            return None

    async def save(self, force: bool = False) -> None:
        """Save checkpoint to blob storage (respects auto-save thresholds)."""
        async with self._lock:
            if self._checkpoint is None:
                return

            now = datetime.now(timezone.utc)
            elapsed = (now - self._last_save_time).total_seconds()

            should_save = (
                force or
                self._pending_updates >= self.auto_save_count or
                elapsed >= self.auto_save_interval
            )

            if not should_save:
                return

            self._checkpoint.updated_at = now.isoformat()
            json_data = self._checkpoint.to_json()

            try:
                blob = self.blob_client.get_blob_client(
                    container=self.container_name,
                    blob=self._checkpoint_path
                )
                blob.upload_blob(json_data, overwrite=True)

                self._pending_updates = 0
                self._last_save_time = now

                stats = self._checkpoint.stats
                logger.info(
                    f"[CheckpointManager] Saved: "
                    f"phase={stats['phase']}, "
                    f"work_items={stats['total_work_items']}, "
                    f"processed={stats['processed']}"
                )

            except Exception as e:
                logger.error(f"[CheckpointManager] Save failed: {e}")
                raise

    async def add_work_item(self, work_item: Dict) -> None:
        """Add a work item to the checkpoint (stored as-is)."""
        async with self._lock:
            self._checkpoint.add_work_item(dict(work_item))
            self._pending_updates += 1

        await self.save()

    async def mark_processed(
        self,
        index: int,
        success: bool,
        row_id: Optional[str] = None,
    ) -> None:
        """Mark a work item as processed."""
        async with self._lock:
            self._checkpoint.mark_processed(index, success, row_id)
            self._pending_updates += 1

        await self.save()

    async def set_phase(self, phase: str) -> None:
        """Update current phase."""
        async with self._lock:
            self._checkpoint.current_phase = phase

        await self.save(force=True)

    async def set_recipe(self, recipe: str) -> None:
        """Store context for resume."""
        async with self._lock:
            self._checkpoint.recipe = recipe

        await self.save(force=True)

    async def add_cost(self, cost_usd: float) -> None:
        """Add to total cost. Bumps pending updates so cost is included in next save."""
        async with self._lock:
            self._checkpoint.total_cost_usd += cost_usd
            self._pending_updates += 1

        await self.save()

    async def add_error(self, error: Dict) -> None:
        """Record an error."""
        async with self._lock:
            self._checkpoint.errors.append({
                **error,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })

    async def delete(self) -> None:
        """Delete checkpoint and any history blobs (after successful completion)."""
        # Delete main checkpoint
        try:
            blob = self.blob_client.get_blob_client(
                container=self.container_name,
                blob=self._checkpoint_path
            )
            blob.delete_blob()
            logger.info("[CheckpointManager] Checkpoint deleted")
        except Exception as e:
            logger.warning(f"[CheckpointManager] Could not delete checkpoint: {e}")

        # Clean up any legacy history blobs
        try:
            container_client = self.blob_client.get_container_client(self.container_name)
            history_prefix = f"checkpoints/{self.project_id}/{self.version_id}/history/"
            blobs = list(container_client.list_blobs(name_starts_with=history_prefix))
            for blob in blobs:
                try:
                    container_client.delete_blob(blob.name)
                except Exception:
                    pass
            if blobs:
                logger.info(f"[CheckpointManager] Cleaned up {len(blobs)} history blobs")
        except Exception as e:
            logger.debug(f"[CheckpointManager] History cleanup skipped: {e}")

    async def force_save(self) -> None:
        """Force immediate save."""
        await self.save(force=True)


def checkpoints_to_work_items(checkpoint_items: List[Dict]) -> List[Dict]:
    """
    Convert checkpoint work item dicts to the format expected by
    GenerationWorkerPool.process_work_items().

    Strips checkpoint-only fields (status, row_id) and returns
    the work items as-is. Handles legacy formats for backward compat.
    """
    work_items = []
    for item in checkpoint_items:
        # Copy and strip checkpoint tracking fields
        wi = {k: v for k, v in item.items() if k not in ("status", "row_id")}

        # Legacy V4/V5 → V9 migration: convert old format to current
        if "template" in wi and "instructions" not in wi:
            # V5 format: template + seed_values → instructions + candidate
            wi = {
                "instructions": wi.get("template", ""),
                "candidate": wi.get("seed_values", {}),
                "research_context": wi.get("filter_findings", ""),
                "tags": wi.get("tags", {}),
            }
        elif "instruction" in wi and "instructions" not in wi:
            # V4 format: instruction + context → instructions + candidate
            wi = {
                "instructions": wi.get("instruction", ""),
                "candidate": wi.get("context", ""),
                "tags": wi.get("tags", {}),
            }

        work_items.append(wi)
    return work_items
