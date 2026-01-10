"""
Project state tracker that polls database for current status and progress.

Key design:
- All state is scoped to a specific version_id
- (project_id, version_id) is treated as a completely fresh project
- When version changes, all phases start from scratch
- Tracks file processing, chunks, seeds, and scoring (persistent state)
- Does NOT track assignment or generation counts (ephemeral, restart on resume)
"""

import logging
import json
import hashlib
from datetime import datetime, timezone
from typing import Optional, Set, List
from uuid import UUID

from sqlalchemy import func as sql_func, desc
from sqlalchemy.orm import Session

from dsl_api.models.project import Project
from dsl_api.models.project_version import ProjectVersion
from dsl_api.models.project_event import ProjectEvent
from dsl_api.models.project_file import ProjectFile
from dsl_api.models.project_rag_chunk import ProjectRagChunk
from dsl_api.models.project_seed import ProjectSeed
from dsl_api.models.sample import Sample

logger = logging.getLogger(__name__)


class ProjectState:
    """
    Tracks current state of a project VERSION by polling the database.

    Used by phases to make decisions about what work to do next.
    Refreshed periodically by the orchestrator's heartbeat loop.

    IMPORTANT: All queries are scoped to the current version_id.
    A new version means starting completely fresh - all phases run from scratch.

    Version semantics:
    - File processing: Process files from version's files_snapshot
    - Seed extraction: Extract seeds for this version only
    - Seed scoring: Score seeds for this version only
    - Seed assignment: Assign seeds for this version only
    - Generation: Generate samples for this version only
    """

    def __init__(self, db: Session, project_id: UUID, version_id: UUID):
        self.db = db
        self.project_id = project_id
        self.version_id = version_id

        # State flags
        self.paused = False
        self.preview_mode = False

        # Cost controls
        self.spend_limit_cents: Optional[int] = None

        # Config tracking for invalidation (within a version)
        self._last_diversity_spec_hash: Optional[str] = None

        # Progress statistics - explicit attributes for easy navigation
        self.files_total = 0
        self.files_processed = 0
        self.chunks_total = 0
        self.seeds_extracted = 0
        self.seeds_scored = 0
        self.samples_generated = 0

        # Project/Version metadata
        self.num_samples = 0
        self.generation_prompt = ""
        self.columns = []
        self.diversity_spec = None
        self.use_internet = False

        # Version snapshot data (files and examples at version creation time)
        self.files_snapshot: List[dict] = []
        self.examples_snapshot: List[dict] = []

        # Initial refresh
        self.refresh()

    def refresh(self):
        """
        Poll database for latest state.

        Should be called frequently by the orchestrator (every iteration).
        """
        project = self.db.query(Project).filter(Project.id == self.project_id).first()
        if not project:
            logger.error(f"Project {self.project_id} not found during state refresh")
            return

        version = self.db.query(ProjectVersion).filter(ProjectVersion.id == self.version_id).first()
        if not version:
            logger.error(f"Version {self.version_id} not found during state refresh")
            return

        # Update flags
        self.paused = self._check_pause_requested()
        self.preview_mode = project.preview_mode

        # Update cost controls (from project, not version)
        self.spend_limit_cents = project.spend_limit_cents

        # Update version metadata (from immutable version snapshot)
        self.num_samples = version.num_samples
        self.generation_prompt = version.generation_prompt
        self.columns = version.columns or []
        self.diversity_spec = version.diversity_spec
        self.use_internet = version.use_internet
        self.files_snapshot = version.files_snapshot or []
        self.examples_snapshot = version.examples_snapshot or []

        # Check for diversity spec changes within this version
        # (This handles edge case where diversity spec is modified mid-run)
        self._check_diversity_spec_changes()

        # Update progress statistics (all scoped to this version)
        self.files_total = len(self.files_snapshot)
        self.files_processed = self._count_processed_files()
        self.chunks_total = self._count_total_chunks()
        self.seeds_extracted = self._count_seeds()
        self.seeds_scored = self._count_scored_seeds()
        self.samples_generated = self._count_samples_generated()

        logger.debug(
            f"State refresh [v{self.version_id}]: paused={self.paused}, "
            f"files={self.files_processed}/{self.files_total}, "
            f"chunks={self.chunks_total}, "
            f"seeds={self.seeds_scored}/{self.seeds_extracted}, "
            f"samples={self.samples_generated}"
        )

    def _check_diversity_spec_changes(self):
        """
        Detect diversity spec changes that require re-scoring.

        If diversity_spec changes mid-run, invalidate seed scores.
        """
        current_hash = self._hash_diversity_spec()
        if self._last_diversity_spec_hash is not None and current_hash != self._last_diversity_spec_hash:
            logger.info("Diversity spec changed, invalidating seed scores for this version")
            self._invalidate_seed_scores()
        self._last_diversity_spec_hash = current_hash

    def _hash_diversity_spec(self) -> str:
        """Create a hash of diversity spec for change detection."""
        if not self.diversity_spec:
            return ""
        serialized = json.dumps(self.diversity_spec, sort_keys=True)
        return hashlib.md5(serialized.encode()).hexdigest()

    def _invalidate_seed_scores(self):
        """Clear scores and scored_at for all seeds in this version."""
        self.db.query(ProjectSeed).filter(
            ProjectSeed.project_id == self.project_id,
            ProjectSeed.version_id == self.version_id,
            ProjectSeed.deleted_at.is_(None)
        ).update({
            ProjectSeed.scores: None,
            ProjectSeed.scored_at: None
        }, synchronize_session=False)
        self.db.commit()

    def _check_pause_requested(self) -> bool:
        """
        Check if there's a pending pause request for this version.

        Returns True if:
        - There's a pause_requested event for this version
        - AND no corresponding paused event after it
        """
        pause_request = (
            self.db.query(ProjectEvent)
            .filter(
                ProjectEvent.project_id == self.project_id,
                ProjectEvent.version_id == self.version_id,
                ProjectEvent.event_type == "pause_requested"
            )
            .order_by(desc(ProjectEvent.created_at))
            .first()
        )

        if not pause_request:
            return False

        paused_event = (
            self.db.query(ProjectEvent)
            .filter(
                ProjectEvent.project_id == self.project_id,
                ProjectEvent.version_id == self.version_id,
                ProjectEvent.event_type == "paused",
                ProjectEvent.created_at > pause_request.created_at
            )
            .first()
        )

        return paused_event is None

    # ---- Progress counting methods (all scoped to version) ----

    def _get_file_ids_from_snapshot(self) -> Set[UUID]:
        """Get file IDs from the version's files_snapshot."""
        return {UUID(f["id"]) for f in self.files_snapshot if "id" in f}

    def _count_processed_files(self) -> int:
        """Count files from snapshot that have chunks."""
        file_ids = self._get_file_ids_from_snapshot()
        if not file_ids:
            return 0

        # Files that have at least one chunk
        processed_file_ids = (
            self.db.query(ProjectRagChunk.file_id)
            .filter(
                ProjectRagChunk.project_id == self.project_id,
                ProjectRagChunk.file_id.in_(file_ids)
            )
            .distinct()
            .all()
        )
        return len(processed_file_ids)

    def _count_total_chunks(self) -> int:
        """Count chunks for files in this version's snapshot."""
        file_ids = self._get_file_ids_from_snapshot()
        if not file_ids:
            return 0

        return (
            self.db.query(sql_func.count(ProjectRagChunk.id))
            .filter(
                ProjectRagChunk.project_id == self.project_id,
                ProjectRagChunk.file_id.in_(file_ids)
            )
            .scalar() or 0
        )

    def _count_seeds(self) -> int:
        """Count non-deleted seeds for this version."""
        return (
            self.db.query(sql_func.count(ProjectSeed.id))
            .filter(
                ProjectSeed.project_id == self.project_id,
                ProjectSeed.version_id == self.version_id,
                ProjectSeed.deleted_at.is_(None)
            )
            .scalar() or 0
        )

    def _count_scored_seeds(self) -> int:
        """Count seeds that have been scored for this version."""
        return (
            self.db.query(sql_func.count(ProjectSeed.id))
            .filter(
                ProjectSeed.project_id == self.project_id,
                ProjectSeed.version_id == self.version_id,
                ProjectSeed.deleted_at.is_(None),
                ProjectSeed.scored_at.isnot(None)
            )
            .scalar() or 0
        )

    def _count_samples_generated(self) -> int:
        """Count generated samples for this version."""
        return (
            self.db.query(sql_func.count(Sample.id))
            .filter(
                Sample.project_id == self.project_id,
                Sample.version_id == self.version_id
            )
            .scalar() or 0
        )

    # ---- Work availability queries (used by phases for flow control) ----

    def get_unprocessed_files(self, limit: int = 1) -> List[dict]:
        """
        Get files from snapshot that don't have chunks yet.

        Returns file info dicts from the snapshot, not ProjectFile objects,
        since we're working from the immutable version snapshot.
        """
        file_ids = self._get_file_ids_from_snapshot()
        if not file_ids:
            return []

        # Get IDs of files that already have chunks
        processed_file_ids = set(
            row[0] for row in
            self.db.query(ProjectRagChunk.file_id)
            .filter(
                ProjectRagChunk.project_id == self.project_id,
                ProjectRagChunk.file_id.in_(file_ids)
            )
            .distinct()
            .all()
        )

        # Return snapshot entries for unprocessed files
        unprocessed = []
        for file_info in self.files_snapshot:
            file_id = UUID(file_info["id"])
            if file_id not in processed_file_ids:
                unprocessed.append(file_info)
                if len(unprocessed) >= limit:
                    break

        return unprocessed

    def get_chunks_without_seeds(self, limit: int = 10) -> List[ProjectRagChunk]:
        """Get chunks that haven't been extracted into seeds yet for this version."""
        file_ids = self._get_file_ids_from_snapshot()
        if not file_ids:
            return []

        # Get chunk IDs that already have seeds for this version
        extracted_chunk_ids = (
            self.db.query(ProjectSeed.chunk_id)
            .filter(
                ProjectSeed.project_id == self.project_id,
                ProjectSeed.version_id == self.version_id,
                ProjectSeed.deleted_at.is_(None)
            )
            .distinct()
        )

        return (
            self.db.query(ProjectRagChunk)
            .filter(
                ProjectRagChunk.project_id == self.project_id,
                ProjectRagChunk.file_id.in_(file_ids),
                ~ProjectRagChunk.id.in_(extracted_chunk_ids)
            )
            .limit(limit)
            .all()
        )

    def get_unscored_seeds(self, limit: int = 20) -> List[ProjectSeed]:
        """Get seeds that haven't been scored yet for this version."""
        return (
            self.db.query(ProjectSeed)
            .filter(
                ProjectSeed.project_id == self.project_id,
                ProjectSeed.version_id == self.version_id,
                ProjectSeed.deleted_at.is_(None),
                ProjectSeed.scored_at.is_(None)
            )
            .limit(limit)
            .all()
        )

    def get_scored_seeds(self) -> List[ProjectSeed]:
        """Get all scored (non-deleted) seeds for assignment for this version."""
        return (
            self.db.query(ProjectSeed)
            .filter(
                ProjectSeed.project_id == self.project_id,
                ProjectSeed.version_id == self.version_id,
                ProjectSeed.deleted_at.is_(None),
                ProjectSeed.scored_at.isnot(None)
            )
            .all()
        )

    # ---- Boolean helpers for flow control ----

    def has_unprocessed_files(self) -> bool:
        """Check if there are files without chunks."""
        return len(self.get_unprocessed_files(limit=1)) > 0

    def has_chunks_without_seeds(self) -> bool:
        """Check if there are chunks without seeds for this version."""
        return len(self.get_chunks_without_seeds(limit=1)) > 0

    def has_unscored_seeds(self) -> bool:
        """Check if there are unscored seeds for this version."""
        return len(self.get_unscored_seeds(limit=1)) > 0

    # ---- File access helpers ----

    def get_file_info(self, file_id: UUID) -> Optional[dict]:
        """Get file info from the version snapshot by ID."""
        for file_info in self.files_snapshot:
            if UUID(file_info["id"]) == file_id:
                return file_info
        return None

    def get_all_file_infos(self) -> List[dict]:
        """Get all file infos from the version snapshot."""
        return self.files_snapshot