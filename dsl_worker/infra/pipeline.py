"""
Pipeline configuration and seed processing.

V6: SeedProcessor is created before the orchestrator and configured incrementally.
The orchestrator sets template, variables, filters, and dedup via tools rather than
designing a monolithic PipelineConfig upfront.

PipelineConfig still exists as a data structure — used for:
- Checkpoint serialization
- Constructing per-yielder configs when the orchestrator spawns subagents
- Backward compatibility with V4/V5 checkpoint resume

Seeds flow through: yielder → dedup gate → filters → work item queue → row generators.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import dataclass, field, asdict
from typing import Any, Awaitable, Callable, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


@dataclass
class VariableConfig:
    """Configuration for a single pipeline variable."""
    name: str                           # e.g., "job_url", "topic", "facts"
    description: str                    # what this variable represents
    seed_strategy: str                  # "search" | "iterate" | "synthetic"
    seed_sources: List[str] = field(default_factory=list)   # URLs, file paths, search queries
    seed_context: str = ""              # accumulated research context (for synthetic)
    seed_instructions: str = ""         # specific instructions for seed yielders


@dataclass
class FilterConfig:
    """Configuration for a seed filter."""
    name: str                           # e.g., "active_and_recent"
    description: str                    # what to check
    complexity: str = "simple"          # "simple" | "judgment"


@dataclass
class DedupConfig:
    """Configuration for seed deduplication."""
    strategy: str = "none"              # "exact" | "embedding_similarity" | "none"
    field: str = ""                     # which variable to dedup on
    threshold: float = 0.85             # for similarity-based


@dataclass
class PipelineConfig:
    """
    Full pipeline configuration produced by the orchestrator.

    The template is the row generation instruction with {variable} placeholders.
    Variables define what changes per row and how to yield seeds.
    """
    template: str                       # row generation instructions with {var} placeholders
    variables: List[VariableConfig]     # what changes per row
    filters: List[FilterConfig] = field(default_factory=list)
    dedup: DedupConfig = field(default_factory=DedupConfig)
    distribution: Dict[str, Dict[str, float]] = field(default_factory=dict)
    target_rows: int = 100
    seed_yielder_count: int = 1         # how many parallel seed yielders
    seed_yielder_instructions: str = "" # shared instructions for all seed yielders
    research_context: str = ""          # orchestrator's research findings for inheritance
    preset_seeds: List[Dict[str, Any]] = field(default_factory=list)  # orchestrator-provided seeds

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> PipelineConfig:
        variables = [VariableConfig(**v) for v in data.get("variables", [])]
        filters = [FilterConfig(**f) for f in data.get("filters", [])]
        dedup = DedupConfig(**data.get("dedup", {}))
        return cls(
            template=data["template"],
            variables=variables,
            filters=filters,
            dedup=dedup,
            distribution=data.get("distribution", {}),
            target_rows=data.get("target_rows", 100),
            seed_yielder_count=data.get("seed_yielder_count", 1),
            seed_yielder_instructions=data.get("seed_yielder_instructions", ""),
            research_context=data.get("research_context", ""),
            preset_seeds=data.get("preset_seeds", []),
        )

    def fill_template(self, values: Dict[str, Any]) -> str:
        """Fill the template with variable values.

        Uses simple {var_name} substitution. Unresolved placeholders are
        left as-is so the row generator sees them (should not happen if
        seeds are complete).
        """
        result = self.template
        for name, value in values.items():
            placeholder = "{" + name + "}"
            str_value = str(value) if not isinstance(value, str) else value
            result = result.replace(placeholder, str_value)
        return result


@dataclass
class Seed:
    """A resolved set of variable values that will become one row."""
    values: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    filter_findings: Dict[str, Any] = field(default_factory=dict)


class SeedProcessor:
    """
    Manages the seed processing pipeline: dedup → filter → dispatch to work queue.

    Thread-safe for concurrent seed yielders submitting seeds.

    V6: Can be created with just a target_rows count, then configured incrementally
    via set_template(), set_variables(), set_filters(), set_dedup(). This allows
    the orchestrator to set things up gradually through tool calls.

    Also accepts a full PipelineConfig for V5-compat (checkpoint resume, etc.).
    """

    def __init__(
        self,
        work_queue: asyncio.Queue,
        on_filter: Optional[Callable[[Seed, FilterConfig], Awaitable[Tuple[bool, Dict]]]] = None,
        on_checkpoint: Optional[Callable[[Dict], Awaitable[None]]] = None,
        # V6: configure incrementally
        target_rows: int = 100,
        # V5 compat: pass full config
        config: Optional[PipelineConfig] = None,
    ) -> None:
        self._work_queue = work_queue
        self._on_filter = on_filter
        self._on_checkpoint = on_checkpoint

        # V6 mutable state — set via methods or from PipelineConfig
        if config is not None:
            self._template: Optional[str] = config.template
            self._variables: List[VariableConfig] = list(config.variables)
            self._filters: List[FilterConfig] = list(config.filters)
            self._dedup: DedupConfig = config.dedup
            self._target_rows: int = config.target_rows
            self._research_context: str = config.research_context or ""
            self._distribution_targets: Dict[str, Dict[str, float]] = config.distribution
        else:
            self._template = None
            self._variables = []
            self._filters = []
            self._dedup = DedupConfig()
            self._target_rows = target_rows
            self._research_context = ""
            self._distribution_targets = {}

        # Dedup state
        self._seen: Set[str] = set()
        self._lock = asyncio.Lock()

        # Counters
        self._accepted = 0
        self._processing = 0        # seeds currently being filtered
        self._submitted_total = 0   # total submit_seed calls
        self._rejected_dedup = 0
        self._rejected_filter = 0

        # Distribution tracking (informational, not enforced)
        self._distribution: Dict[str, Dict[str, int]] = {}

    # --- V6 incremental configuration ---

    def set_template(self, template: str) -> None:
        """Set or update the row generation template."""
        self._template = template

    def set_variables(self, variables: List[VariableConfig]) -> None:
        """Set or update the pipeline variables."""
        self._variables = list(variables)

    def set_filters(self, filters: List[FilterConfig]) -> None:
        """Set or update the filter list."""
        self._filters = list(filters)

    def set_dedup(self, dedup: DedupConfig) -> None:
        """Set or update the dedup config."""
        self._dedup = dedup

    def set_research_context(self, context: str) -> None:
        """Set or update the research context passed to row generators."""
        self._research_context = context

    @property
    def accepted_count(self) -> int:
        return self._accepted

    @property
    def stats(self) -> Dict[str, Any]:
        return {
            "accepted": self._accepted,
            "processing": self._processing,
            "rejected_dedup": self._rejected_dedup,
            "rejected_filter": self._rejected_filter,
            "submitted_total": self._submitted_total,
            "target": self._target_rows,
            "remaining": max(0, self._target_rows - self._accepted),
        }

    def _build_status(self, accepted: bool, reason: str = "") -> Dict[str, Any]:
        """Build a rich status dict for the yielder."""
        return {
            "accepted": accepted,
            "reason": reason,
            "stats": self.stats,
        }

    async def submit_seed(self, seed: Seed, yielder_id: str = "") -> Dict[str, Any]:
        """
        Process a seed: dedup → filter → queue for generation.

        Returns a rich status dict: {accepted, reason, stats}.
        Called by seed yielders (possibly concurrently).
        """
        if not self._template:
            return self._build_status(False, "no template set")

        # --- Dedup (under lock) ---
        async with self._lock:
            self._submitted_total += 1
            if not self._check_dedup(seed):
                self._rejected_dedup += 1
                logger.debug(f"[SeedProcessor] Seed rejected by dedup: {seed.values}")
                return self._build_status(False, "duplicate")
            self._processing += 1

        # --- Filters (no lock — can be slow) ---
        for filter_config in self._filters:
            passed, findings = await self._run_filter(seed, filter_config)
            if not passed:
                async with self._lock:
                    self._processing -= 1
                    self._rejected_filter += 1
                logger.info(
                    f"[SeedProcessor] Seed rejected by filter '{filter_config.name}': "
                    f"{findings.get('reason', 'no reason')}"
                )
                return self._build_status(False, f"filter:{filter_config.name}")
            # Merge findings into seed for downstream use
            seed.filter_findings[filter_config.name] = findings

        # --- Accept ---
        work_item = self._build_work_item(seed)

        async with self._lock:
            self._processing -= 1
            self._accepted += 1
            self._track_distribution(seed)

        # Checkpoint the work item
        if self._on_checkpoint:
            await self._on_checkpoint(work_item)

        # Queue for generation
        await self._work_queue.put(work_item)

        logger.info(
            f"[SeedProcessor] Seed accepted ({self._accepted}/{self._target_rows})"
        )
        return self._build_status(True)

    def _check_dedup(self, seed: Seed) -> bool:
        """Check if seed passes dedup. Returns True if unique."""
        dedup = self._dedup
        if dedup.strategy == "none":
            return True

        if dedup.strategy == "exact":
            # Dedup on a specific field, or all values
            if dedup.field and dedup.field in seed.values:
                key = str(seed.values[dedup.field])
            else:
                key = json.dumps(seed.values, sort_keys=True, ensure_ascii=False)

            if key in self._seen:
                return False
            self._seen.add(key)
            return True

        # TODO: embedding_similarity — use embeddings to detect near-duplicates
        if dedup.strategy == "embedding_similarity":
            logger.warning("[SeedProcessor] embedding_similarity dedup not yet implemented, passing")
            return True

        return True

    async def _run_filter(self, seed: Seed, filter_config: FilterConfig) -> Tuple[bool, Dict]:
        """Run a filter on a seed. Returns (passed, findings)."""
        if self._on_filter:
            return await self._on_filter(seed, filter_config)
        # No filter handler configured — pass everything
        return True, {}

    def _build_work_item(self, seed: Seed) -> Dict[str, Any]:
        """Fill template with seed values, build work item for row generator."""
        filled_template = self._fill_template(seed.values)

        # Build filter findings section
        filter_findings_text = ""
        if seed.filter_findings:
            parts = []
            for name, findings in seed.filter_findings.items():
                if isinstance(findings, dict):
                    findings_str = json.dumps(findings, indent=2, ensure_ascii=False)
                else:
                    findings_str = str(findings)
                parts.append(f"### {name}\n{findings_str}")
            filter_findings_text = "\n\n".join(parts)

        return {
            "template": filled_template,
            "seed_values": seed.values,
            "filter_findings": filter_findings_text,
            "research_context": self._research_context,
            "metadata": seed.metadata,
            "tags": seed.metadata.get("tags", {}),
            "status": "pending",
            "row_id": None,
        }

    def _fill_template(self, values: Dict[str, Any]) -> str:
        """Fill the template with variable values."""
        result = self._template or ""
        for name, value in values.items():
            placeholder = "{" + name + "}"
            str_value = str(value) if not isinstance(value, str) else value
            result = result.replace(placeholder, str_value)
        return result

    def _track_distribution(self, seed: Seed) -> None:
        """Track actual distribution of variable values."""
        for var_name, target_dist in self._distribution_targets.items():
            if var_name in seed.values:
                value = str(seed.values[var_name])
                if var_name not in self._distribution:
                    self._distribution[var_name] = {}
                self._distribution[var_name][value] = (
                    self._distribution[var_name].get(value, 0) + 1
                )
