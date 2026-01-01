"""
Project state tracker that polls database for current status and progress.

Key design:
- Tracks file processing, chunks, seeds, and scoring (persistent state)
- Does NOT track assignment or generation counts (ephemeral, restart on resume)
- Detects config changes that require re-processing
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
from dsl_api.models.project_event import ProjectEvent
from dsl_api.models.project_file import ProjectFile
from dsl_api.models.project_rag_chunk import ProjectRagChunk
from dsl_api.models.project_seed import ProjectSeed
from dsl_api.models.sample import Sample

logger = logging.getLogger(__name__)


class ProjectState:
    """
    Tracks current state of a project by polling the database.

    Used by phases to make decisions about what work to do next.
    Refreshed periodically by the orchestrator's heartbeat loop.

    Resume semantics:
    - File processing: resume from unprocessed files
    - Seed extraction: resume from chunks without seeds
    - Seed scoring: resume from unscored seeds
    - Seed assignment: always start fresh (computed from scored seeds)
    - Generation: always start fresh (uses current assigned seeds)
    """

    def __init__(self, db: Session, project_id: UUID):
        self.db = db
        self.project_id = project_id

        # State flags
        self.paused = False
        self.preview_mode = False

        # Config tracking for invalidation
        self._last_diversity_spec_hash: Optional[str] = None
        self._last_file_ids: Set[UUID] = set()

        # Progress statistics - explicit attributes for easy navigation
        self.files_total = 0
        self.files_processed = 0
        self.chunks_total = 0
        self.seeds_extracted = 0
        self.seeds_scored = 0
        self.samples_generated = 0

        # Project metadata
        self.num_samples = 0
        self.generation_prompt = ""
        self.columns = []
        self.diversity_spec = None
        self.use_internet = False
        self.run_id: Optional[UUID] = None

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

        # Update flags
        self.paused = self._check_pause_requested()
        self.preview_mode = getattr(project, 'preview_mode', False)

        # Update project metadata
        self.num_samples = project.num_samples
        self.generation_prompt = project.generation_prompt
        self.columns = project.columns or []
        self.diversity_spec = project.diversity_spec
        self.use_internet = project.use_internet
        self.run_id = project.current_run_id

        # Check for config changes that require invalidation
        self._check_config_changes()

        # Update progress statistics
        self.files_total = self._count_total_files()
        self.files_processed = self._count_processed_files()
        self.chunks_total = self._count_total_chunks()
        self.seeds_extracted = self._count_seeds()
        self.seeds_scored = self._count_scored_seeds()
        self.samples_generated = self._count_samples_generated()

        logger.debug(
            f"State refresh: paused={self.paused}, "
            f"files={self.files_processed}/{self.files_total}, "
            f"chunks={self.chunks_total}, "
            f"seeds={self.seeds_scored}/{self.seeds_extracted}, "
            f"samples={self.samples_generated}"
        )

    def _check_config_changes(self):
        """
        Detect config changes that require invalidation.

        - If diversity_spec changes: invalidate seed scores (re-score with new axes)
        - If files are deleted: delete chunks and soft-delete seeds
        """
        # Check diversity spec changes
        current_hash = self._hash_diversity_spec()
        if self._last_diversity_spec_hash is not None and current_hash != self._last_diversity_spec_hash:
            logger.info("Diversity spec changed, invalidating seed scores")
            self._invalidate_seed_scores()
        self._last_diversity_spec_hash = current_hash

        # Check for deleted files
        current_file_ids = self._get_active_file_ids()
        deleted_files = self._last_file_ids - current_file_ids
        if deleted_files:
            logger.info(f"Files deleted: {deleted_files}, cleaning up chunks and seeds")
            self._delete_chunks_for_files(deleted_files)
            self._soft_delete_seeds_for_files(deleted_files)
        self._last_file_ids = current_file_ids

    def _hash_diversity_spec(self) -> str:
        """Create a hash of diversity spec for change detection."""
        if not self.diversity_spec:
            return ""
        serialized = json.dumps(self.diversity_spec, sort_keys=True)
        return hashlib.md5(serialized.encode()).hexdigest()

    def _get_active_file_ids(self) -> Set[UUID]:
        """Get IDs of non-deleted files."""
        files = (
            self.db.query(ProjectFile.id)
            .filter(
                ProjectFile.project_id == self.project_id,
                ProjectFile.deleted_at.is_(None),
                ProjectFile.status == 'uploaded'
            )
            .all()
        )
        return {f.id for f in files}

    def _invalidate_seed_scores(self):
        """Clear scores and scored_at for all seeds."""
        self.db.query(ProjectSeed).filter(
            ProjectSeed.project_id == self.project_id,
            ProjectSeed.deleted_at.is_(None)
        ).update({
            ProjectSeed.scores: None,
            ProjectSeed.scored_at: None
        }, synchronize_session=False)
        self.db.commit()

    def _delete_chunks_for_files(self, file_ids: Set[UUID]):
        """Hard delete chunks from deleted files."""
        self.db.query(ProjectRagChunk).filter(
            ProjectRagChunk.project_id == self.project_id,
            ProjectRagChunk.file_id.in_(file_ids)
        ).delete(synchronize_session=False)
        self.db.commit()

    def _soft_delete_seeds_for_files(self, file_ids: Set[UUID]):
        """Soft delete seeds that came from deleted files."""
        now = datetime.now(timezone.utc)
        self.db.query(ProjectSeed).filter(
            ProjectSeed.project_id == self.project_id,
            ProjectSeed.file_id.in_(file_ids),
            ProjectSeed.deleted_at.is_(None)
        ).update({
            ProjectSeed.deleted_at: now
        }, synchronize_session=False)
        self.db.commit()

    def _check_pause_requested(self) -> bool:
        """
        Check if there's a pending pause request.

        Returns True if:
        - There's a pause_requested event for this project
        - AND no corresponding paused event after it
        """
        pause_request = (
            self.db.query(ProjectEvent)
            .filter(
                ProjectEvent.project_id == self.project_id,
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
                ProjectEvent.event_type == "paused",
                ProjectEvent.created_at > pause_request.created_at
            )
            .first()
        )

        return paused_event is None

    # ---- Progress counting methods ----

    def _count_total_files(self) -> int:
        """Count total uploaded (non-deleted) files for this project."""
        return (
            self.db.query(sql_func.count(ProjectFile.id))
            .filter(
                ProjectFile.project_id == self.project_id,
                ProjectFile.deleted_at.is_(None),
                ProjectFile.status == 'uploaded'
            )
            .scalar() or 0
        )

    def _count_processed_files(self) -> int:
        """Count files that have chunks (fully processed)."""
        subq = (
            self.db.query(ProjectRagChunk.file_id)
            .filter(ProjectRagChunk.project_id == self.project_id)
            .distinct()
        )
        return (
            self.db.query(sql_func.count(ProjectFile.id))
            .filter(
                ProjectFile.project_id == self.project_id,
                ProjectFile.deleted_at.is_(None),
                ProjectFile.status == 'uploaded',
                ProjectFile.id.in_(subq)
            )
            .scalar() or 0
        )

    def _count_total_chunks(self) -> int:
        """Count all chunks for this project."""
        return (
            self.db.query(sql_func.count(ProjectRagChunk.id))
            .filter(ProjectRagChunk.project_id == self.project_id)
            .scalar() or 0
        )

    def _count_seeds(self) -> int:
        """Count non-deleted seeds."""
        return (
            self.db.query(sql_func.count(ProjectSeed.id))
            .filter(
                ProjectSeed.project_id == self.project_id,
                ProjectSeed.deleted_at.is_(None)
            )
            .scalar() or 0
        )

    def _count_scored_seeds(self) -> int:
        """Count seeds that have been scored."""
        return (
            self.db.query(sql_func.count(ProjectSeed.id))
            .filter(
                ProjectSeed.project_id == self.project_id,
                ProjectSeed.deleted_at.is_(None),
                ProjectSeed.scored_at.isnot(None)
            )
            .scalar() or 0
        )

    def _count_samples_generated(self) -> int:
        """Count generated samples (for display only)."""
        return (
            self.db.query(sql_func.count(Sample.id))
            .filter(Sample.project_id == self.project_id)
            .scalar() or 0
        )

    # ---- Work availability queries (used by phases for flow control) ----

    def get_unprocessed_files(self, limit: int = 1) -> List[ProjectFile]:
        """Get files that don't have chunks yet."""
        processed_file_ids = (
            self.db.query(ProjectRagChunk.file_id)
            .filter(ProjectRagChunk.project_id == self.project_id)
            .distinct()
        )

        return (
            self.db.query(ProjectFile)
            .filter(
                ProjectFile.project_id == self.project_id,
                ProjectFile.deleted_at.is_(None),
                ProjectFile.status == 'uploaded',
                ~ProjectFile.id.in_(processed_file_ids)
            )
            .limit(limit)
            .all()
        )

    def get_chunks_without_seeds(self, limit: int = 10) -> List[ProjectRagChunk]:
        """Get chunks that haven't been extracted into seeds yet."""
        extracted_chunk_ids = (
            self.db.query(ProjectSeed.chunk_id)
            .filter(
                ProjectSeed.project_id == self.project_id,
                ProjectSeed.deleted_at.is_(None)
            )
            .distinct()
        )

        return (
            self.db.query(ProjectRagChunk)
            .filter(
                ProjectRagChunk.project_id == self.project_id,
                ~ProjectRagChunk.id.in_(extracted_chunk_ids)
            )
            .limit(limit)
            .all()
        )

    def get_unscored_seeds(self, limit: int = 20) -> List[ProjectSeed]:
        """Get seeds that haven't been scored yet."""
        return (
            self.db.query(ProjectSeed)
            .filter(
                ProjectSeed.project_id == self.project_id,
                ProjectSeed.deleted_at.is_(None),
                ProjectSeed.scored_at.is_(None)
            )
            .limit(limit)
            .all()
        )

    def get_scored_seeds(self) -> List[ProjectSeed]:
        """Get all scored (non-deleted) seeds for assignment."""
        return (
            self.db.query(ProjectSeed)
            .filter(
                ProjectSeed.project_id == self.project_id,
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
        """Check if there are chunks without seeds."""
        return len(self.get_chunks_without_seeds(limit=1)) > 0

    def has_unscored_seeds(self) -> bool:
        """Check if there are unscored seeds."""
        return len(self.get_unscored_seeds(limit=1)) > 0