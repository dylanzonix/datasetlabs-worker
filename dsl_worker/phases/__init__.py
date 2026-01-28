"""
Phases for the DSL worker pipeline.

Pipeline:
- ResearchPhaseV2: Explores sources, dispatches extraction
- SeedExtractionPhase: Extracts seeds (source + note)
- SeedAssignmentPhase: Assigns seeds to diversity slots (batch ranking)
- GenerationPhaseV3: Generates rows from assigned seeds
"""

from .base import Phase, PhaseResult, PhaseStatus
from .browser_pool import BrowserPool, BrowseResult
from .sandbox import SandboxExecutor, SandboxResult
from .research_v2 import ResearchPhaseV2
from .seed_extraction_v2 import SeedExtractionPhase
from .seed_assignment import SeedAssignmentPhase
from .generation_v3 import GenerationPhaseV3

__all__ = [
    # Base
    "Phase",
    "PhaseResult",
    "PhaseStatus",
    # Browser
    "BrowserPool",
    "BrowseResult",
    # Sandbox
    "SandboxExecutor",
    "SandboxResult",
    # Phases
    "ResearchPhaseV2",
    "SeedExtractionPhase",
    "SeedAssignmentPhase",
    "GenerationPhaseV3",
]