"""
Phase base classes.

Provides:
- PhaseResult: Result of a phase execution
- PhaseStatus: Status of a phase for logging
- Phase: Abstract base class for phases
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Any

from sqlalchemy.orm import Session

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


@dataclass
class PhaseStatus:
    """
    Status of a phase for logging/monitoring.

    Attributes:
        phase_name: Name of the phase
        status: One of 'pending', 'active', 'complete', 'idle'
        progress: Human-readable progress string
        detail: Optional additional detail
    """
    phase_name: str
    status: str
    progress: str
    detail: Optional[str] = None

    def __str__(self) -> str:
        if self.detail:
            return f"{self.phase_name}: {self.status} ({self.progress}) - {self.detail}"
        return f"{self.phase_name}: {self.status} ({self.progress})"

    def short(self) -> str:
        """Short format for inline logging."""
        return f"{self.phase_name}={self.progress}"


class Phase(ABC):
    """
    Base class for a processing phase.

    Each phase implements:
    - should_run(): Decide if this phase should execute now
    - execute_once(): Process one unit of work
    - is_complete(): Check if this phase is fully done
    - get_status(): Return current progress/status
    """

    def __init__(
            self,
            name: str,
            state: Any,  # ProjectState
            db: Session,
            openai_client: Optional[Any] = None,
            blob_service_client: Optional[Any] = None,
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

        Returns:
            True if execute_once() should be called
        """
        pass

    @abstractmethod
    async def execute_once(self) -> PhaseResult:
        """
        Execute one unit of work for this phase.

        Returns:
            PhaseResult with did_work flag and cost_usd
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

    def get_status(self) -> PhaseStatus:
        """
        Get current status/progress of this phase.

        Override in subclasses for meaningful progress reporting.
        """
        if self.is_complete():
            status = "complete"
        elif self.should_run():
            status = "active"
        else:
            status = "pending"

        return PhaseStatus(
            phase_name=self.name,
            status=status,
            progress="unknown"
        )