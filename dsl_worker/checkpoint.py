"""
Pipeline Checkpointing

Handles saving and restoring pipeline state for pause/resume.

Checkpoints are stored as JSON in Azure Blob Storage.

V4: Work items include context from topic agents. Backward-compatible
with v2/v3 checkpoints.
"""

import asyncio
import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Set, Any
from uuid import UUID

logger = logging.getLogger(__name__)


@dataclass
class WorkItemCheckpoint:
    """Serializable work item data."""
    instruction: str
    schema: Optional[Dict] = None

    # Generation status
    status: str = "pending"  # 'pending', 'completed', 'failed'
    row_id: Optional[str] = None


@dataclass
class PipelineCheckpoint:
    """
    Full pipeline state.

    Designed to be:
    - Small (no duplicate data)
    - Complete (can fully resume from it)
    - Debuggable (human-readable JSON)
    """

    # Version for future compatibility
    version: str = "2.0"

    # Identity
    project_id: str = ""
    version_id: str = ""

    # Timestamps
    created_at: str = ""
    updated_at: str = ""

    # Phase tracking
    # 'orchestrator' | 'sample' | 'generation' | 'completed'
    current_phase: str = "orchestrator"

    # Work items (output of orchestrator, input to generation)
    work_items: List[Dict] = field(default_factory=list)  # WorkItemCheckpoint as dict

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
        """Deserialize from JSON string. Handles v1 (seeds) and v2 (work_items)."""
        data = json.loads(json_str)

        # Handle legacy v1 checkpoints with "seeds" field
        if "seeds" in data and "work_items" not in data:
            data["work_items"] = _migrate_seeds_to_work_items(data.pop("seeds"))

        # Handle legacy phase names
        if data.get("current_phase") == "research":
            data["current_phase"] = "orchestrator"

        # Handle legacy field names
        if "processed_seed_indices" in data and "processed_indices" not in data:
            data["processed_indices"] = data.pop("processed_seed_indices")

        # Remove legacy fields not in v2
        for legacy_key in [
            "completed_scope_ids", "pending_scopes", "seeds",
            "processed_seed_indices",
        ]:
            data.pop(legacy_key, None)

        # Ensure version is set
        if "version" not in data or data["version"] == "1.0":
            data["version"] = "2.0"

        return cls(**data)

    def add_work_item(self, item: Dict) -> None:
        """Add a work item."""
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

    Thread-safe for concurrent generation workers.
    """

    def __init__(
        self,
        blob_service_client: Any,
        container_name: str,
        project_id: UUID,
        version_id: UUID,
        auto_save_interval: int = 30,  # seconds
        auto_save_count: int = 10,     # work items
    ):
        self.blob_client = blob_service_client
        self.container_name = container_name
        self.project_id = project_id
        self.version_id = version_id

        self.auto_save_interval = auto_save_interval
        self.auto_save_count = auto_save_count

        # Paths
        self._checkpoint_path = f"checkpoints/{project_id}/{version_id}/state.json"
        self._history_prefix = f"checkpoints/{project_id}/{version_id}/history/"

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
        """Get current checkpoint."""
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
            checkpoint = PipelineCheckpoint.from_json(json_data)

            return checkpoint

        except Exception as e:
            logger.debug(f"[CheckpointManager] No checkpoint found: {e}")
            return None

    async def save(self, force: bool = False) -> None:
        """Save checkpoint to blob storage."""
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

                # Save to history (for debugging)
                timestamp = now.strftime("%Y%m%d_%H%M%S")
                history_path = f"{self._history_prefix}{timestamp}.json"
                history_blob = self.blob_client.get_blob_client(
                    container=self.container_name,
                    blob=history_path
                )
                history_blob.upload_blob(json_data, overwrite=True)

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
        """Add a work item to the checkpoint."""
        async with self._lock:
            checkpoint_item = {
                "instruction": work_item.get("instruction", ""),
                "context": work_item.get("context", ""),
                "schema": work_item.get("schema"),
                "tags": work_item.get("tags", {}),
                "status": "pending",
                "row_id": None,
            }
            self._checkpoint.add_work_item(checkpoint_item)
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
        """Add to total cost."""
        async with self._lock:
            self._checkpoint.total_cost_usd += cost_usd

    async def add_error(self, error: Dict) -> None:
        """Record an error."""
        async with self._lock:
            self._checkpoint.errors.append({
                **error,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })

    async def delete(self) -> None:
        """Delete checkpoint (after successful completion)."""
        try:
            blob = self.blob_client.get_blob_client(
                container=self.container_name,
                blob=self._checkpoint_path
            )
            blob.delete_blob()
            logger.info("[CheckpointManager] Checkpoint deleted")
        except Exception as e:
            logger.warning(f"[CheckpointManager] Could not delete checkpoint: {e}")

    async def force_save(self) -> None:
        """Force immediate save."""
        await self.save(force=True)


def checkpoints_to_work_items(checkpoint_items: List[Dict]) -> List[Dict]:
    """
    Convert checkpoint work item dicts to the format expected by
    GenerationWorkerPool.process_work_items().

    Checkpoint format: {instruction, context, schema, tags, status, row_id}
    Pool format: {instruction, context, schema, tags}
    """
    work_items = []
    for item in checkpoint_items:
        work_items.append({
            "instruction": item.get("instruction", ""),
            "context": item.get("context", ""),
            "schema": item.get("schema"),
            "tags": item.get("tags", {}),
        })
    return work_items


def _migrate_seeds_to_work_items(seeds: List[Dict]) -> List[Dict]:
    """Convert v1 seed checkpoint dicts to v2 work item format."""
    work_items = []
    for seed in seeds:
        work_items.append({
            "instruction": seed.get("content", ""),
            "schema": None,
            "status": seed.get("status", "pending"),
            "row_id": seed.get("row_id"),
        })
    return work_items
