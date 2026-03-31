"""
Candidate data types for the harvester → row generator pipeline.

V11: CandidatePool and StrategyMonitor removed. The orchestrator LLM
directly controls harvesting and processing — no algorithmic sampling.
Candidates live in per-harvester buffers managed by the orchestrator.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class Candidate:
    """A candidate item from a source, ready for row generation."""

    values: Any  # raw candidate: string, dict, or structured data
    source_id: str
    source_context: str = ""  # human-readable scope description
    metadata: Dict[str, Any] = field(default_factory=dict)
