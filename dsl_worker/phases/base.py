"""
Base class for processing phases in the orchestrator.
"""

import logging
from abc import ABC, abstractmethod
from typing import Optional

from sqlalchemy.orm import Session
from openai import AsyncOpenAI
from azure.storage.blob import BlobServiceClient

from dsl_worker.project_state import ProjectState

logger = logging.getLogger(__name__)


class Phase(ABC):
    """
    Base class for a processing phase.

    Each phase implements:
    - should_run(): Decide if this phase should execute now
    - execute_batch(): Process one batch of work
    - is_complete(): Check if this phase is fully done

    Phases can run sequentially or concurrently depending on their
    should_run() logic and the orchestrator's gather() call.
    """

    def __init__(
            self,
            name: str,
            state: ProjectState,
            db: Session,
            openai_client: Optional[AsyncOpenAI] = None,
            blob_service_client: Optional[BlobServiceClient] = None,
    ):
        self.name = name
        self.state = state
        self.db = db
        self.openai_client = openai_client
        self.blob_service_client = blob_service_client

        logger.debug(f"Initialized phase: {name}")

    @abstractmethod
    def should_run(self) -> bool:
        """
        Decide if this phase should execute in the current iteration.

        Consider:
        - Is there work available for this phase?
        - Are dependencies complete? (for sequential phases)
        - Is preview mode active? (for eager phases)
        - Does config invalidation require re-running?

        Returns:
            True if execute_batch() should be called
        """
        pass

    @abstractmethod
    async def execute_batch(self, batch_size: int = 10) -> int:
        """
        Execute one batch of work for this phase.

        Args:
            batch_size: Maximum number of items to process

        Returns:
            Number of items actually processed (0 if no work available)
        """
        pass

    @abstractmethod
    def is_complete(self) -> bool:
        """
        Check if this phase has finished all its work.

        Returns:
            True if no more work remains for this phase
        """
        pass

    def get_progress(self) -> dict:
        """
        Get progress information for this phase.

        Returns:
            Dictionary with progress metrics
        """
        return {
            "phase": self.name,
            "complete": self.is_complete(),
            "should_run": self.should_run(),
        }