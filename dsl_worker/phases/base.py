"""
Base class for processing phases in the orchestrator.

Key design decisions:
- execute_once() processes one unit of work (phases determine batch size internally)
- Status tracking only matters up to seed_scoring (assignment/generation restart from scratch)
- Phases are responsible for their own resume logic
- All phases return PhaseResult with cost tracking
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

from sqlalchemy.orm import Session
from azure.storage.blob import BlobServiceClient

from dsl_worker.project_state import ProjectState
from dsl_worker.billing import TrackedOpenAIClient

logger = logging.getLogger(__name__)


@dataclass
class PhaseResult:
    """
    Result of a phase execution.

    Attributes:
        did_work: Whether any work was done
        cost_usd: Cost incurred in USD (before margin - margin applied by CostTracker)
    """
    did_work: bool
    cost_usd: float = 0.0

    @staticmethod
    def no_work() -> "PhaseResult":
        """Create a result indicating no work was done."""
        return PhaseResult(did_work=False, cost_usd=0.0)

    @staticmethod
    def work_done(cost_usd: float = 0.0) -> "PhaseResult":
        """Create a result indicating work was done."""
        return PhaseResult(did_work=True, cost_usd=cost_usd)


class Phase(ABC):
    """
    Base class for a processing phase.

    Each phase implements:
    - should_run(): Decide if this phase should execute now
    - execute_once(): Process one unit of work (batch size determined internally)
    - is_complete(): Check if this phase is fully done

    Resume semantics:
    - file_processing, seed_extraction, seed_scoring: Resume from where we left off
    - seed_assignment, generation: Always start fresh (ephemeral state)
    """

    def __init__(
            self,
            name: str,
            state: ProjectState,
            db: Session,
            openai_client: Optional[TrackedOpenAIClient] = None,
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

        Returns:
            True if execute_once() should be called
        """
        pass

    @abstractmethod
    async def execute_once(self) -> PhaseResult:
        """
        Execute one unit of work for this phase.

        The phase determines internally how much work constitutes "one unit":
        - For file processing: one file
        - For seed extraction: one chunk (or a small batch)
        - For seed scoring: a batch of seeds
        - For assignment: all seeds at once (algorithm requirement)
        - For generation: one sample

        Returns:
            PhaseResult with did_work flag and cost_cents
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