"""
Cost tracker for a project run.

Handles:
- Accumulating costs from API calls (categorized as preprocessing or generation)
- Charging to user balance at intervals
- Checking if user has sufficient balance
- Enforcing project spend limits (dynamically updated)
- THREAD-SAFE for concurrent workers

Key improvements:
- Uses threading.Lock for all state mutations (asyncio-safe)
- Charges happen IMMEDIATELY when thresholds are reached
- Balance/spend limit checks are always up-to-date
"""

import logging
import threading
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Optional, List
from uuid import UUID

from sqlalchemy import func as sql_func
from sqlalchemy.orm import Session

from dsl_api.models.balance_ledger import BalanceLedger

logger = logging.getLogger(__name__)


class CostCategory(str, Enum):
    """Cost categories for tracking."""
    PREPROCESSING = "preprocessing"
    GENERATION = "generation"


@dataclass
class CostEntry:
    """A single cost entry (append-only)."""
    timestamp: datetime
    phase: str
    category: CostCategory
    cost_usd: float
    description: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    model: str = ""


@dataclass
class ChargeRecord:
    """Record of a charge made to the ledger."""
    timestamp: datetime
    amount_cents: int
    ledger_id: UUID


# Mapping of phase names to categories
PHASE_CATEGORIES = {
    "file_processing": CostCategory.PREPROCESSING,
    "seed_extraction": CostCategory.PREPROCESSING,
    "seed_scoring": CostCategory.PREPROCESSING,
    "seed_assignment": CostCategory.PREPROCESSING,
    "generation": CostCategory.GENERATION,
}


class CostTracker:
    """
    Tracks costs for a single project run and handles charging.

    THREAD-SAFE: Uses locks for all state mutations to handle
    concurrent workers safely.

    Design:
    - Costs are append-only (never modified)
    - Charges are also append-only
    - Current uncharged = sum(costs) - sum(charges)
    - Charges occur every interval OR when threshold reached
    - Enforces project spend limits (dynamically fetched from DB)
    - Costs are categorized as preprocessing or generation

    Args:
        db: Database session
        user_id: User being charged
        project_id: Project generating costs
        margin_multiplier: Multiply raw costs by this (e.g., 2.0 for 100% margin)
        charge_threshold_cents: Charge when accumulated costs reach this (e.g., 100 = $1)
        charge_interval_seconds: Charge at least this often (e.g., 60 = 1 minute)
        spend_limit_cents: Initial spend limit (will be refreshed from DB dynamically)
    """

    def __init__(
        self,
        db: Session,
        user_id: UUID,
        project_id: UUID,
        margin_multiplier: float = 2.0,
        charge_threshold_cents: int = 100,
        charge_interval_seconds: int = 60,
        spend_limit_cents: Optional[int] = None,
    ):
        self.db = db
        self.user_id = user_id
        self.project_id = project_id
        self.margin_multiplier = margin_multiplier
        self.charge_threshold_cents = charge_threshold_cents
        self.charge_interval_seconds = charge_interval_seconds

        # Cache for spend limit (refreshed periodically)
        self._spend_limit_cache: Optional[int] = spend_limit_cents
        self._spend_limit_cache_time: datetime = datetime.now(timezone.utc)
        self._spend_limit_cache_ttl_seconds: float = 5.0  # Refresh every 5 seconds

        # Append-only lists
        self._costs: List[CostEntry] = []
        self._charges: List[ChargeRecord] = []

        # Timing
        self._last_charge_time: datetime = datetime.now(timezone.utc)
        self._start_time: datetime = datetime.now(timezone.utc)

        # Cache cumulative spend at init (before this run)
        self._cumulative_spend_at_start = self._query_cumulative_project_spend()

        # Thread-safe lock for all state mutations
        # This is a threading.Lock which is safe to use in asyncio
        # because asyncio runs in a single thread
        self._lock = threading.Lock()

    def _query_cumulative_project_spend(self) -> int:
        """
        Query total spend on this project from all previous runs.
        Returns amount in cents (positive value).
        """
        result = (
            self.db.query(sql_func.sum(BalanceLedger.amount))
            .filter(
                BalanceLedger.project_id == self.project_id,
                BalanceLedger.amount < 0  # debits are negative
            )
            .scalar()
        )
        return abs(result) if result else 0

    def _get_current_spend_limit(self) -> Optional[int]:
        """
        Fetch current spend limit from database with short TTL cache.

        This allows dynamic updates while project is running - user can
        lower the spend limit and the worker will pause accordingly.
        """
        now = datetime.now(timezone.utc)
        age = (now - self._spend_limit_cache_time).total_seconds()

        if age < self._spend_limit_cache_ttl_seconds:
            return self._spend_limit_cache

        # Refresh from DB
        from dsl_api.models.project import Project

        project = self.db.query(Project).filter(Project.id == self.project_id).first()
        if project:
            old_limit = self._spend_limit_cache
            self._spend_limit_cache = project.spend_limit_cents

            # Log if limit changed
            if old_limit != self._spend_limit_cache:
                logger.info(
                    f"Spend limit updated: {old_limit}¢ -> {self._spend_limit_cache}¢"
                )

        self._spend_limit_cache_time = now
        return self._spend_limit_cache

    @property
    def spend_limit_cents(self) -> Optional[int]:
        """Current spend limit (fetched fresh from DB with caching)."""
        return self._get_current_spend_limit()

    @property
    def cumulative_spend_cents(self) -> int:
        """Total spend on this project including current run."""
        with self._lock:
            return self._cumulative_spend_at_start + self._total_charged_cents_unlocked()

    def _total_charged_cents_unlocked(self) -> int:
        """Get total charged cents (must be called with lock held)."""
        return sum(c.amount_cents for c in self._charges)

    @property
    def remaining_budget_cents(self) -> Optional[int]:
        """
        Remaining budget for this project.
        Returns None if no spend limit is set.
        """
        limit = self.spend_limit_cents  # Dynamically fetched
        if limit is None:
            return None
        return max(0, limit - self.cumulative_spend_cents)

    def would_exceed_spend_limit(self, additional_cents: int) -> bool:
        """
        Check if spending additional_cents would exceed the spend limit.
        Returns False if no limit is set.
        """
        limit = self.spend_limit_cents  # Dynamically fetched
        if limit is None:
            return False
        with self._lock:
            projected = (
                self._cumulative_spend_at_start +
                self._total_charged_cents_unlocked() +
                self._uncharged_cents_unlocked() +
                additional_cents
            )
        return projected > limit

    def add_cost(
        self,
        phase: str,
        cost_usd: float,
        description: str = "",
        input_tokens: int = 0,
        output_tokens: int = 0,
        model: str = "",
    ) -> None:
        """
        Add a cost entry.

        THREAD-SAFE: Uses lock for appending.

        Args:
            phase: Phase name that incurred the cost (auto-categorized)
            cost_usd: Raw cost in USD (before margin)
            description: Optional description
            input_tokens: Number of input tokens used
            output_tokens: Number of output tokens used
            model: Model name (e.g., "gpt-4o")
        """
        if cost_usd <= 0:
            return

        # Apply margin
        cost_with_margin = cost_usd * self.margin_multiplier

        # Determine category from phase name
        category = PHASE_CATEGORIES.get(phase, CostCategory.PREPROCESSING)

        entry = CostEntry(
            timestamp=datetime.now(timezone.utc),
            phase=phase,
            category=category,
            cost_usd=cost_with_margin,
            description=description,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            model=model,
        )

        with self._lock:
            self._costs.append(entry)
            total_input = sum(c.input_tokens for c in self._costs)
            total_output = sum(c.output_tokens for c in self._costs)
            total_cost_cents = self._total_costs_cents_unlocked()

        # Verbose logging with token counts
        total_tokens = input_tokens + output_tokens
        if model:
            logger.info(
                f"💰 Cost: {phase} | {model} | "
                f"in={input_tokens:,} out={output_tokens:,} ({total_tokens:,} total) | "
                f"${cost_usd:.4f} raw → ${cost_with_margin:.4f} w/margin | "
                f"running total: {total_cost_cents}¢ ({total_input:,}+{total_output:,} tokens)"
            )
        else:
            logger.debug(
                f"Cost added: phase={phase}, category={category.value}, "
                f"raw=${cost_usd:.6f}, with_margin=${cost_with_margin:.6f}"
            )

    def _total_costs_usd_unlocked(self) -> float:
        """Get total costs USD (must be called with lock held)."""
        return sum(c.cost_usd for c in self._costs)

    def _total_costs_cents_unlocked(self) -> int:
        """Get total costs in cents (must be called with lock held)."""
        return int(self._total_costs_usd_unlocked() * 100)

    def _uncharged_cents_unlocked(self) -> int:
        """Get uncharged cents (must be called with lock held)."""
        return self._total_costs_cents_unlocked() - self._total_charged_cents_unlocked()

    @property
    def total_costs_usd(self) -> float:
        """Total costs accumulated (with margin applied)."""
        with self._lock:
            return self._total_costs_usd_unlocked()

    @property
    def total_costs_cents(self) -> int:
        """Total costs in cents (for display/ledger)."""
        with self._lock:
            return self._total_costs_cents_unlocked()

    @property
    def total_charged_cents(self) -> int:
        """Total amount already charged to ledger."""
        with self._lock:
            return self._total_charged_cents_unlocked()

    @property
    def uncharged_cents(self) -> int:
        """Amount accumulated but not yet charged (in cents)."""
        with self._lock:
            return self._uncharged_cents_unlocked()

    @property
    def uncharged_usd(self) -> float:
        """Amount accumulated but not yet charged (in USD)."""
        with self._lock:
            total_usd = self._total_costs_usd_unlocked()
            charged_cents = self._total_charged_cents_unlocked()
        return total_usd - (charged_cents / 100.0)

    def preprocessing_costs_cents(self) -> int:
        """Total preprocessing costs in cents."""
        with self._lock:
            total = sum(c.cost_usd for c in self._costs if c.category == CostCategory.PREPROCESSING)
        return int(total * 100)

    def generation_costs_cents(self) -> int:
        """Total generation costs in cents."""
        with self._lock:
            total = sum(c.cost_usd for c in self._costs if c.category == CostCategory.GENERATION)
        return int(total * 100)

    def should_charge(self) -> bool:
        """
        Check if we should charge now.

        Returns True if:
        - Uncharged amount >= threshold, OR
        - Time since last charge >= interval
        """
        with self._lock:
            uncharged = self._uncharged_cents_unlocked()
            last_charge = self._last_charge_time

        if uncharged <= 0:
            return False

        # Threshold reached
        if uncharged >= self.charge_threshold_cents:
            return True

        # Interval elapsed
        now = datetime.now(timezone.utc)
        elapsed = (now - last_charge).total_seconds()
        if elapsed >= self.charge_interval_seconds:
            return True

        return False

    def charge_if_needed(self) -> Optional[int]:
        """
        Charge uncharged costs to user balance if needed.

        THREAD-SAFE: Uses lock for the entire operation.

        Returns:
            Amount charged in cents, or None if no charge made
        """
        if not self.should_charge():
            return None

        return self._do_charge()

    def charge_remaining(self) -> Optional[int]:
        """
        Charge any remaining uncharged costs (call at end of run).

        Returns:
            Amount charged in cents, or None if nothing to charge
        """
        with self._lock:
            if self._uncharged_cents_unlocked() <= 0:
                return None

        return self._do_charge()

    def _do_charge(self) -> int:
        """
        Execute a charge to the ledger.

        THREAD-SAFE: Uses lock for the entire operation.
        """
        with self._lock:
            amount = self._uncharged_cents_unlocked()
            if amount <= 0:
                return 0

            # Create ledger entry (negative = debit)
            ledger_entry = BalanceLedger(
                id=uuid.uuid4(),
                user_id=self.user_id,
                amount=-amount,  # Negative for debit
                reason="project_usage",
                project_id=self.project_id,
            )
            self.db.add(ledger_entry)
            self.db.commit()

            # Record the charge
            record = ChargeRecord(
                timestamp=datetime.now(timezone.utc),
                amount_cents=amount,
                ledger_id=ledger_entry.id,
            )
            self._charges.append(record)
            self._last_charge_time = datetime.now(timezone.utc)

        logger.info(
            f"Charged user {self.user_id}: {amount}¢ "
            f"(total charged: {self.total_charged_cents}¢, "
            f"cumulative project spend: {self.cumulative_spend_cents}¢)"
        )

        return amount

    def get_user_balance_cents(self) -> int:
        """
        Get user's current balance from ledger.

        Returns:
            Balance in cents (sum of all ledger entries)
        """
        result = (
            self.db.query(sql_func.sum(BalanceLedger.amount))
            .filter(BalanceLedger.user_id == self.user_id)
            .scalar()
        )
        return result or 0

    def has_sufficient_balance(self) -> bool:
        """
        Check if user has enough balance to continue.

        We consider balance sufficient if:
        - Current balance > 0 (we allow running to zero, but not negative)
        """
        balance = self.get_user_balance_cents()
        return balance > 0

    def is_within_spend_limit(self) -> bool:
        """
        Check if we're still within the project spend limit.

        Returns True if:
        - No spend limit is set, OR
        - Cumulative spend (including uncharged) is within limit

        Note: spend_limit_cents is fetched dynamically from DB.
        """
        limit = self.spend_limit_cents  # Dynamically fetched
        if limit is None:
            return True
        with self._lock:
            projected_spend = (
                self._cumulative_spend_at_start +
                self._total_charged_cents_unlocked() +
                self._uncharged_cents_unlocked()
            )
        return projected_spend <= limit

    def check_balance_and_charge(self) -> tuple[bool, Optional[str]]:
        """
        Combined check: charge if needed, then verify balance and spend limit.

        Returns:
            Tuple of (can_continue, stop_reason)
            - (True, None) if OK to continue
            - (False, reason) if should stop
        """
        # First, charge any accumulated costs
        self.charge_if_needed()

        # Check spend limit (dynamically fetched from DB)
        if not self.is_within_spend_limit():
            return False, "spend_limit_exceeded"

        # Check user balance
        if not self.has_sufficient_balance():
            return False, "insufficient_balance"

        return True, None

    def get_summary(self) -> dict:
        """Get a summary of costs and charges."""
        limit = self.spend_limit_cents  # Dynamically fetched

        with self._lock:
            total_costs_cents = self._total_costs_cents_unlocked()
            total_charged_cents = self._total_charged_cents_unlocked()
            uncharged_cents = self._uncharged_cents_unlocked()
            num_costs = len(self._costs)
            num_charges = len(self._charges)
            cumulative = self._cumulative_spend_at_start + total_charged_cents
            total_input_tokens = sum(c.input_tokens for c in self._costs)
            total_output_tokens = sum(c.output_tokens for c in self._costs)

        summary = {
            "total_costs_cents": total_costs_cents,
            "preprocessing_costs_cents": self.preprocessing_costs_cents(),
            "generation_costs_cents": self.generation_costs_cents(),
            "total_charged_cents": total_charged_cents,
            "uncharged_cents": uncharged_cents,
            "cumulative_project_spend_cents": cumulative,
            "num_cost_entries": num_costs,
            "num_charges": num_charges,
            "user_balance_cents": self.get_user_balance_cents(),
            "total_input_tokens": total_input_tokens,
            "total_output_tokens": total_output_tokens,
            "total_tokens": total_input_tokens + total_output_tokens,
        }

        # Add spend limit info if set
        if limit is not None:
            summary["spend_limit_cents"] = limit
            summary["remaining_budget_cents"] = max(0, limit - cumulative)

        return summary