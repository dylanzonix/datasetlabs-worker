"""
Pipeline Checkpointing — V11

Saves and restores the full pipeline state for pause/resume.

What we save:
- Orchestrator conversation (messages, cost, turns) — this is the brain
- Each harvester's conversation (messages, cost, turns) — so they pick up mid-source
- Source states (stats, candidates buffer, exhaustion flags)
- Generation stats (rows_generated, skipped, errors)
- Counters (harvester/apollo/research IDs)
- Total cost

What we DON'T save (recreated on resume):
- Async primitives (locks, semaphores, events, tasks)
- HTTP clients (BU, sandbox, OpenAI)
- BU browser sessions (server-side, die on their own — harvesters create new ones)
- Sandbox sessions (recreated on demand)
- Langfuse spans

Storage: JSON in Azure Blob at checkpoints/{project_id}/{version_id}/state.json
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
    Full V11 pipeline state. Everything needed to resume exactly where we left off.
    """

    version: str = "4.0"

    # Identity
    project_id: str = ""
    version_id: str = ""

    # Timestamps
    created_at: str = ""
    updated_at: str = ""

    # Cost tracking
    total_cost_usd: float = 0.0

    # --- V11 state ---

    # Orchestrator conversation (the brain — every decision, tool call, result)
    orchestrator_conversation: Optional[Dict[str, Any]] = None

    # Source states (stats + candidate buffers + harvester conversations)
    sources: List[Dict[str, Any]] = field(default_factory=list)

    # Generation stats
    generation_stats: Optional[Dict[str, Any]] = None

    # Counters for ID generation
    harvester_counter: int = 0
    apollo_counter: int = 0
    research_counter: int = 0

    # --- V13 state ---

    # V13 orchestrator counters (file naming, status line)
    bu_extract_counter: int = 0
    apify_run_counter: int = 0
    candidates_harvested: int = 0
    candidates_submitted: int = 0

    # V13 file processing state (mid-file resume)
    current_file: Optional[Dict[str, Any]] = None

    # --- Legacy fields (kept for backward compat on load) ---
    current_phase: str = "orchestrator"
    work_items: List[Dict] = field(default_factory=list)
    processed_indices: List[int] = field(default_factory=list)
    recipe: Optional[str] = None
    errors: List[Dict] = field(default_factory=list)
    source_stats: Optional[Dict[str, Any]] = None

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

        return cls(**data)

    @property
    def has_v11_state(self) -> bool:
        """Check if this checkpoint has V11 conversation state."""
        return self.orchestrator_conversation is not None

    @property
    def stats(self) -> Dict:
        """Get summary stats."""
        return {
            "phase": self.current_phase,
            "has_v11_state": self.has_v11_state,
            "sources": len(self.sources),
            "total_cost_usd": self.total_cost_usd,
            "orchestrator_messages": len((self.orchestrator_conversation or {}).get("messages", [])),
            # Legacy
            "total_work_items": len(self.work_items),
            "processed": len(self.processed_indices),
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
        auto_save_interval: int = 15,  # seconds (was 30, tightened for V11)
        auto_save_count: int = 5,      # pending updates (was 10)
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
            stats = existing.stats
            logger.info(
                f"[Checkpoint] Resuming: "
                f"v11={stats['has_v11_state']}, "
                f"sources={stats['sources']}, "
                f"orch_msgs={stats['orchestrator_messages']}, "
                f"cost=${stats['total_cost_usd']:.4f}"
            )
            self._checkpoint = existing
        else:
            self._checkpoint = PipelineCheckpoint(
                project_id=str(self.project_id),
                version_id=str(self.version_id),
                created_at=datetime.now(timezone.utc).isoformat(),
            )
            logger.info("[Checkpoint] Created new checkpoint")

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
            logger.debug(f"[Checkpoint] No checkpoint found: {e}")
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

                size_kb = len(json_data) / 1024
                stats = self._checkpoint.stats
                logger.info(
                    f"[Checkpoint] Saved ({size_kb:.1f}KB): "
                    f"sources={stats['sources']}, "
                    f"orch_msgs={stats['orchestrator_messages']}, "
                    f"cost=${stats['total_cost_usd']:.4f}"
                )

            except Exception as e:
                logger.error(f"[Checkpoint] Save failed: {e}")
                raise

    async def save_pipeline_state(
        self,
        orchestrator_messages: List[Dict[str, Any]],
        orchestrator_cost: float,
        orchestrator_turns: int,
        sources: List[Dict[str, Any]],
        generation_stats: Dict[str, Any],
        harvester_counter: int,
        apollo_counter: int,
        research_counter: int,
        *,
        # V13-specific fields (keyword-only so V12 callers don't break)
        bu_extract_counter: int = 0,
        apify_run_counter: int = 0,
        candidates_harvested: int = 0,
        candidates_submitted: int = 0,
        current_file: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Save the full pipeline state. Called after every orchestrator tool call."""
        async with self._lock:
            cp = self._checkpoint
            cp.orchestrator_conversation = {
                "messages": orchestrator_messages,
                "total_cost": orchestrator_cost,
                "total_turns": orchestrator_turns,
            }
            cp.sources = sources
            cp.generation_stats = dict(generation_stats)
            cp.harvester_counter = harvester_counter
            cp.apollo_counter = apollo_counter
            cp.research_counter = research_counter
            # V13 fields
            cp.bu_extract_counter = bu_extract_counter
            cp.apify_run_counter = apify_run_counter
            cp.candidates_harvested = candidates_harvested
            cp.candidates_submitted = candidates_submitted
            cp.current_file = current_file
            self._pending_updates += 1

        await self.save()

    async def add_cost(self, cost_usd: float) -> None:
        """Add to total cost."""
        async with self._lock:
            self._checkpoint.total_cost_usd += cost_usd
            self._pending_updates += 1

        await self.save()

    async def set_phase(self, phase: str) -> None:
        """Update current phase."""
        async with self._lock:
            self._checkpoint.current_phase = phase

        await self.save(force=True)

    async def delete(self) -> None:
        """Delete checkpoint (after successful completion)."""
        try:
            blob = self.blob_client.get_blob_client(
                container=self.container_name,
                blob=self._checkpoint_path
            )
            blob.delete_blob()
            logger.info("[Checkpoint] Deleted")
        except Exception as e:
            logger.warning(f"[Checkpoint] Could not delete: {e}")

        # Clean up legacy history blobs
        try:
            container_client = self.blob_client.get_container_client(self.container_name)
            history_prefix = f"checkpoints/{self.project_id}/{self.version_id}/history/"
            blobs = list(container_client.list_blobs(name_starts_with=history_prefix))
            for blob in blobs:
                try:
                    container_client.delete_blob(blob.name)
                except Exception:
                    pass
        except Exception:
            pass

    async def force_save(self) -> None:
        """Force immediate save."""
        await self.save(force=True)
