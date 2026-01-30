"""
Project state tracker - minimal version for v3 pipeline.

Only tracks:
- Pause state (polls DB for pause requests)
- Project/version configuration
"""

import logging
from typing import List, Optional
from uuid import UUID

from sqlalchemy import desc
from sqlalchemy.orm import Session

from dsl_api.models.project import Project
from dsl_api.models.project_version import ProjectVersion
from dsl_api.models.project_event import ProjectEvent

logger = logging.getLogger(__name__)


class ProjectState:
    """
    Tracks current state of a project version.
    
    Refreshed periodically by the job processor to check for pause requests
    and load configuration.
    """

    def __init__(self, db: Session, project_id: UUID, version_id: UUID):
        self.db = db
        self.project_id = project_id
        self.version_id = version_id

        # State flags
        self.paused = False

        # Cost controls
        self.spend_limit_cents: Optional[int] = None

        # Project/Version configuration
        self.num_samples = 0
        self.generation_prompt = ""
        self.columns = []
        self.use_internet = False

        # Version snapshot data
        self.files_snapshot: List[dict] = []
        self.examples_snapshot: List[dict] = []

        # Initial refresh
        self.refresh()

    def refresh(self):
        """Poll database for latest state."""
        project = self.db.query(Project).filter(Project.id == self.project_id).first()
        if not project:
            logger.error(f"Project {self.project_id} not found")
            return

        version = self.db.query(ProjectVersion).filter(ProjectVersion.id == self.version_id).first()
        if not version:
            logger.error(f"Version {self.version_id} not found")
            return

        # Update pause state
        self.paused = self._check_pause_requested()

        # Update cost controls
        self.spend_limit_cents = project.spend_limit_cents

        # Update version configuration
        self.num_samples = version.num_samples
        self.generation_prompt = version.generation_prompt
        self.columns = version.columns or []
        self.use_internet = version.use_internet
        self.files_snapshot = version.files_snapshot or []
        self.examples_snapshot = version.examples_snapshot or []

        logger.debug(f"State refresh: paused={self.paused}, num_samples={self.num_samples}")

    def _check_pause_requested(self) -> bool:
        """Check if there's a pending pause request for this version."""
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

        # Check if already handled
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