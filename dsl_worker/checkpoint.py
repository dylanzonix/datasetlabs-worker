"""
Pipeline Checkpointing

Handles saving and restoring pipeline state for pause/resume.

Checkpoints are stored as JSON in Azure Blob Storage.
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
class SeedCheckpoint:
    """Serializable seed data."""
    content: str
    scope_id: str
    scope_description: str
    notes: List[str]
    research_summary: Optional[str] = None
    source_ref: Optional[str] = None
    source_url: Optional[str] = None
    
    # Generation status
    status: str = "pending"  # 'pending', 'completed', 'failed'
    row_id: Optional[str] = None


@dataclass
class ScopeCheckpoint:
    """Serializable scope data for pending scopes."""
    id: str
    description: str
    quota: int
    depth: int
    notes: List[str]
    parent_notes: List[str]


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
    version: str = "1.0"
    
    # Identity
    project_id: str = ""
    version_id: str = ""
    
    # Timestamps
    created_at: str = ""
    updated_at: str = ""
    
    # Phase tracking
    current_phase: str = "research"  # 'research', 'generation', 'completed'
    
    # Scope tracking
    # We don't store the full tree - just track what's done and what's pending
    completed_scope_ids: List[str] = field(default_factory=list)
    pending_scopes: List[Dict] = field(default_factory=list)  # ScopeCheckpoint as dict
    
    # Seeds (output of research, input to generation)
    seeds: List[Dict] = field(default_factory=list)  # SeedCheckpoint as dict
    
    # Generation progress
    # Just indices - actual rows are in DB
    processed_seed_indices: List[int] = field(default_factory=list)
    
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
        """Deserialize from JSON string."""
        data = json.loads(json_str)
        return cls(**data)
    
    def add_completed_scope(self, scope_id: str):
        """Mark a scope as completed."""
        if scope_id not in self.completed_scope_ids:
            self.completed_scope_ids.append(scope_id)
    
    def add_seed(self, seed: 'SeedCheckpoint'):
        """Add a seed from research."""
        self.seeds.append(asdict(seed))
    
    def mark_seed_processed(self, index: int, success: bool, row_id: Optional[str] = None):
        """Mark a seed as processed."""
        if index not in self.processed_seed_indices:
            self.processed_seed_indices.append(index)
        
        if index < len(self.seeds):
            self.seeds[index]["status"] = "completed" if success else "failed"
            if row_id:
                self.seeds[index]["row_id"] = row_id
    
    def get_pending_seed_indices(self) -> List[int]:
        """Get indices of seeds not yet processed."""
        processed = set(self.processed_seed_indices)
        return [i for i in range(len(self.seeds)) if i not in processed]
    
    @property
    def stats(self) -> Dict:
        """Get summary stats."""
        return {
            "phase": self.current_phase,
            "completed_scopes": len(self.completed_scope_ids),
            "pending_scopes": len(self.pending_scopes),
            "total_seeds": len(self.seeds),
            "processed_seeds": len(self.processed_seed_indices),
            "pending_seeds": len(self.seeds) - len(self.processed_seed_indices),
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
        auto_save_count: int = 10,     # seeds
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
        """
        Load existing checkpoint or create new one.
        
        Returns the checkpoint to use.
        """
        existing = await self.load()
        
        if existing:
            logger.info(
                f"[CheckpointManager] Resuming from checkpoint: "
                f"phase={existing.current_phase}, "
                f"seeds={len(existing.seeds)}, "
                f"processed={len(existing.processed_seed_indices)}"
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
            # Blob doesn't exist or other error
            logger.debug(f"[CheckpointManager] No checkpoint found: {e}")
            return None
    
    async def save(self, force: bool = False) -> None:
        """
        Save checkpoint to blob storage.
        
        Args:
            force: Save immediately, ignoring auto-save thresholds
        """
        async with self._lock:
            if self._checkpoint is None:
                return
            
            # Check if we should save
            now = datetime.now(timezone.utc)
            elapsed = (now - self._last_save_time).total_seconds()
            
            should_save = (
                force or
                self._pending_updates >= self.auto_save_count or
                elapsed >= self.auto_save_interval
            )
            
            if not should_save:
                return
            
            # Update timestamp
            self._checkpoint.updated_at = now.isoformat()
            
            json_data = self._checkpoint.to_json()
            
            try:
                # Save current state
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
                    f"seeds={stats['total_seeds']}, "
                    f"processed={stats['processed_seeds']}"
                )
                
            except Exception as e:
                logger.error(f"[CheckpointManager] Save failed: {e}")
                raise
    
    async def mark_scope_completed(self, scope_id: str):
        """Mark a scope as completed and trigger potential save."""
        async with self._lock:
            self._checkpoint.add_completed_scope(scope_id)
            self._pending_updates += 1
        
        # Always save on scope completion
        await self.save(force=True)
    
    async def add_seeds(self, seeds: List['Seed']):
        """Add seeds from research phase."""
        async with self._lock:
            for seed in seeds:
                seed_checkpoint = SeedCheckpoint(
                    content=seed.content,
                    scope_id=seed.scope_id,
                    scope_description=seed.scope_description,
                    notes=seed.notes,
                    research_summary=seed.research_summary,
                    source_ref=seed.source_ref,
                    source_url=seed.source_url,
                )
                self._checkpoint.add_seed(seed_checkpoint)
            self._pending_updates += len(seeds)
        
        await self.save()
    
    async def mark_seed_processed(
        self,
        index: int,
        success: bool,
        row_id: Optional[str] = None
    ):
        """Mark a seed as processed (row generated or failed)."""
        async with self._lock:
            self._checkpoint.mark_seed_processed(index, success, row_id)
            self._pending_updates += 1
        
        await self.save()
    
    async def set_phase(self, phase: str):
        """Update current phase."""
        async with self._lock:
            self._checkpoint.current_phase = phase
        
        await self.save(force=True)
    
    async def add_cost(self, cost_usd: float):
        """Add to total cost."""
        async with self._lock:
            self._checkpoint.total_cost_usd += cost_usd
    
    async def add_error(self, error: Dict):
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
    
    async def force_save(self):
        """Force immediate save."""
        await self.save(force=True)


def seeds_to_checkpoints(seeds: List['Seed']) -> List[SeedCheckpoint]:
    """Convert Seed objects to SeedCheckpoint objects."""
    return [
        SeedCheckpoint(
            content=s.content,
            scope_id=s.scope_id,
            scope_description=s.scope_description,
            notes=s.notes,
            research_summary=s.research_summary,
            source_ref=s.source_ref,
            source_url=s.source_url,
        )
        for s in seeds
    ]


def checkpoints_to_seeds(checkpoints: List[Dict]) -> List['Seed']:
    """Convert checkpoint dicts back to Seed objects."""
    from dsl_worker.phases.research_tools import Seed
    
    return [
        Seed(
            content=c["content"],
            scope_id=c["scope_id"],
            scope_description=c["scope_description"],
            notes=c["notes"],
            research_summary=c.get("research_summary"),
            source_ref=c.get("source_ref"),
            source_url=c.get("source_url"),
        )
        for c in checkpoints
    ]