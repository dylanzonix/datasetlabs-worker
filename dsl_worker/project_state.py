"""
Project state tracker that polls database for current status and progress.
"""

import logging
from datetime import datetime, timezone
from typing import Dict, Optional
from uuid import UUID

from sqlalchemy import func as sql_func, desc
from sqlalchemy.orm import Session

from dsl_api.models.project import Project
from dsl_api.models.project_event import ProjectEvent
from dsl_api.models.sample import Sample

logger = logging.getLogger(__name__)


class ProjectState:
    """
    Tracks current state of a project by polling the database.

    Used by phases to make decisions about what work to do next.
    Refreshed periodically by the orchestrator's heartbeat loop.
    """

    def __init__(self, db: Session, project_id: UUID, run_id: UUID):
        self.db = db
        self.project_id = project_id
        self.run_id = run_id

        # State flags
        self.paused = False
        self.preview_mode = False
        self.config_version: Optional[datetime] = None

        # Progress statistics
        self.stats: Dict[str, int] = {}

        # Project metadata
        self.num_samples = 0
        self.generation_prompt = ""
        self.columns = []
        self.diversity_spec = None
        self.use_internet = False

        # Initial refresh
        self.refresh()

    def refresh(self):
        """
        Poll database for latest state.

        Should be called frequently by the orchestrator (every 1-2 seconds).
        """
        # Load project
        project = self.db.query(Project).filter(Project.id == self.project_id).first()
        if not project:
            logger.error(f"Project {self.project_id} not found during state refresh")
            return

        # Update flags
        self.paused = self._check_pause_request()
        self.preview_mode = getattr(project, 'preview_mode', False)  # Add this column to Project model
        self.config_version = project.updated_at

        # Update project metadata
        self.num_samples = project.num_samples
        self.generation_prompt = project.generation_prompt
        self.columns = project.columns or []
        self.diversity_spec = project.diversity_spec
        self.use_internet = project.use_internet

        # Update progress statistics
        self.stats = {
            'files_total': self._count_total_files(),
            'files_processed': self._count_processed_files(),
            'chunks_total': self._count_total_chunks(),
            'chunks_embedded': self._count_embedded_chunks(),
            'seeds_extracted': self._count_seeds(),
            'seeds_scored': self._count_scored_seeds(),
            'seeds_assigned': self._count_assigned_seeds(),
            'recipes_built': self._count_recipes(),
            'samples_generated': self._count_samples_generated(),
            'samples_validated': self._count_samples_validated(),
        }

        logger.debug(f"State refresh: paused={self.paused}, preview={self.preview_mode}, stats={self.stats}")

    def _check_pause_request(self) -> bool:
        """
        Check if there's a pending pause request.

        Returns True if:
        - There's a pause_request event for this run
        - AND no corresponding paused event after it
        """
        pause_request = (
            self.db.query(ProjectEvent)
            .filter(
                ProjectEvent.project_id == self.project_id,
                ProjectEvent.run_id == self.run_id,
                ProjectEvent.event_type == "pause_request"
            )
            .order_by(desc(ProjectEvent.created_at))
            .first()
        )

        if not pause_request:
            return False

        # Check if we've already handled this pause
        paused_event = (
            self.db.query(ProjectEvent)
            .filter(
                ProjectEvent.project_id == self.project_id,
                ProjectEvent.run_id == self.run_id,
                ProjectEvent.event_type == "paused",
                ProjectEvent.created_at > pause_request.created_at
            )
            .first()
        )

        return paused_event is None

    # ---- Progress tracking methods ----
    # TODO: Implement these based on your actual database schema

    def _count_total_files(self) -> int:
        """Count total uploaded files for this project."""
        # TODO: Implement based on your ProjectFile or similar model
        # return self.db.query(ProjectFile).filter(
        #     ProjectFile.project_id == self.project_id
        # ).count()
        return 0

    def _count_processed_files(self) -> int:
        """Count files that have been fully processed (chunked & embedded)."""
        # TODO: Track processing status in ProjectFile model
        # return self.db.query(ProjectFile).filter(
        #     ProjectFile.project_id == self.project_id,
        #     ProjectFile.processing_status == 'completed'
        # ).count()
        return 0

    def _count_total_chunks(self) -> int:
        """Count all text chunks extracted from files."""
        # TODO: Implement based on your Chunk model
        # return self.db.query(Chunk).filter(
        #     Chunk.project_id == self.project_id,
        #     Chunk.run_id == self.run_id
        # ).count()
        return 0

    def _count_embedded_chunks(self) -> int:
        """Count chunks that have been embedded."""
        # TODO: Track embedding status
        # return self.db.query(Chunk).filter(
        #     Chunk.project_id == self.project_id,
        #     Chunk.run_id == self.run_id,
        #     Chunk.embedding.isnot(None)
        # ).count()
        return 0

    def _count_seeds(self) -> int:
        """Count extracted recipe seeds."""
        # TODO: Implement based on your Seed model
        # return self.db.query(Seed).filter(
        #     Seed.project_id == self.project_id,
        #     Seed.run_id == self.run_id
        # ).count()
        return 0

    def _count_scored_seeds(self) -> int:
        """Count seeds that have been scored."""
        # TODO: Track scoring status
        # return self.db.query(Seed).filter(
        #     Seed.project_id == self.project_id,
        #     Seed.run_id == self.run_id,
        #     Seed.score.isnot(None)
        # ).count()
        return 0

    def _count_assigned_seeds(self) -> int:
        """Count seeds that have been assigned to diversity axes."""
        # TODO: Track assignment status
        # return self.db.query(Seed).filter(
        #     Seed.project_id == self.project_id,
        #     Seed.run_id == self.run_id,
        #     Seed.diversity_axis.isnot(None)
        # ).count()
        return 0

    def _count_recipes(self) -> int:
        """Count built recipes (seeds with RAG context)."""
        # TODO: Implement based on your Recipe model
        # return self.db.query(Recipe).filter(
        #     Recipe.project_id == self.project_id,
        #     Recipe.run_id == self.run_id
        # ).count()
        return 0

    def _count_samples_generated(self) -> int:
        """Count generated samples."""
        return (
                self.db.query(sql_func.count(Sample.id))
                .filter(
                    Sample.project_id == self.project_id,
                    Sample.run_id == self.run_id
                )
                .scalar() or 0
        )

    def _count_samples_validated(self) -> int:
        """Count samples that passed validation."""
        # TODO: Track validation status
        # return self.db.query(Sample).filter(
        #     Sample.project_id == self.project_id,
        #     Sample.run_id == self.run_id,
        #     Sample.validation_status == 'passed'
        # ).count()
        return 0

    def needs_rerun(self, phase_name: str, last_run_at: Optional[datetime] = None) -> bool:
        """
        Check if a phase needs to be re-run due to config changes.

        Args:
            phase_name: Name of the phase to check
            last_run_at: When this phase last completed (if ever)

        Returns:
            True if the phase should be re-run
        """
        if last_run_at is None:
            return True  # Never run before

        # If config was updated after the phase last ran, might need to re-run
        if self.config_version and self.config_version > last_run_at:
            # Smart invalidation: only invalidate if relevant config changed
            # For now, just invalidate all phases after config change
            # TODO: Implement field-level tracking to only invalidate affected phases
            return True

        return False