"""
Web Search agent — research subagent for web-based investigation.

V10: Renamed from ResearchAgent for clarity. Same functionality:
- Brave search for web queries
- open/find/click for page navigation
- interact() for BU-powered site interaction (no submit_seed)
- respond() for explicit answer submission

Used by orchestrator for recon and by row generators for enrichment.
"""

from dsl_worker.agents.research import ResearchAgent

# V10 alias — same agent, clearer name
WebSearchAgent = ResearchAgent

__all__ = ["WebSearchAgent"]
