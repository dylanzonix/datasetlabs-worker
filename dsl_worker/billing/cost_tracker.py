"""
Cost tracker for a project run.

Handles:
- Accumulating costs from API calls (categorized as preprocessing or generation)
- Charging to user balance at intervals
- Checking if user has sufficient balance
- Enforcing project spend limits
"""

import logging
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

    Design:
    - Costs are append-only (never modified)
    - Charges are also append-only
    - Current uncharged = sum(costs) - sum(charges)
    - Charges occur every interval OR when threshold reached
    - Enforces project spend limits (cumulative across all runs)
    - Costs are categorized as preprocessing or generation

    Args:
        db: Database session
        user_id: User being charged
        project_id: Project generating costs
        margin_multiplier: Multiply raw costs by this (e.g., 2.0 for 100% margin)
        charge_threshold_cents: Charge when accumulated costs reach this (e.g., 1000 = $10)
        charge_interval_seconds: Charge at least this often (e.g., 60 = 1 minute)
        spend_limit_cents: Maximum cumulative spend for this project (None = no limit)
    """

    def __init__(
        self,
        db: Session,
        user_id: UUID,
        project_id: UUID,
        margin_multiplier: float = 2.0,
        charge_threshold_cents: int = 1000,
        charge_interval_seconds: int = 60,
        spend_limit_cents: Optional[int] = None,
    ):
        self.db = db
        self.user_id = user_id
        self.project_id = project_id
        self.margin_multiplier = margin_multiplier
        self.charge_threshold_cents = charge_threshold_cents
        self.charge_interval_seconds = charge_interval_seconds
        self.spend_limit_cents = spend_limit_cents

        # Append-only lists
        self._costs: List[CostEntry] = []
        self._charges: List[ChargeRecord] = []

        # Timing
        self._last_charge_time: datetime = datetime.now(timezone.utc)
        self._start_time: datetime = datetime.now(timezone.utc)

        # Cache cumulative spend at init (before this run)
        self._cumulative_spend_at_start = self._query_cumulative_project_spend()

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

    @property
    def cumulative_spend_cents(self) -> int:
        """Total spend on this project including current run."""
        return self._cumulative_spend_at_start + self.total_charged_cents

    @property
    def remaining_budget_cents(self) -> Optional[int]:
        """
        Remaining budget for this project.
        Returns None if no spend limit is set.
        """
        if self.spend_limit_cents is None:
            return None
        return max(0, self.spend_limit_cents - self.cumulative_spend_cents)

    def would_exceed_spend_limit(self, additional_cents: int) -> bool:
        """
        Check if spending additional_cents would exceed the spend limit.
        Returns False if no limit is set.
        """
        if self.spend_limit_cents is None:
            return False
        projected = self.cumulative_spend_cents + self.uncharged_cents + additional_cents
        return projected > self.spend_limit_cents

    def add_cost(self, phase: str, cost_usd: float, description: str = "") -> None:
        """
        Add a cost entry.

        Args:
            phase: Phase name that incurred the cost (auto-categorized)
            cost_usd: Raw cost in USD (before margin)
            description: Optional description
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
        )
        self._costs.append(entry)

        logger.debug(
            f"Cost added: phase={phase}, category={category.value}, "
            f"raw=${cost_usd:.6f}, with_margin=${cost_with_margin:.6f}"
        )

    @property
    def total_costs_usd(self) -> float:
        """Total costs accumulated (with margin applied)."""
        return sum(c.cost_usd for c in self._costs)

    @property
    def total_costs_cents(self) -> int:
        """Total costs in cents (for display/ledger)."""
        return int(self.total_costs_usd * 100)

    @property
    def total_charged_cents(self) -> int:
        """Total amount already charged to ledger."""
        return sum(c.amount_cents for c in self._charges)

    @property
    def uncharged_cents(self) -> int:
        """Amount accumulated but not yet charged (in cents)."""
        return self.total_costs_cents - self.total_charged_cents

    @property
    def uncharged_usd(self) -> float:
        """Amount accumulated but not yet charged (in USD)."""
        return self.total_costs_usd - (self.total_charged_cents / 100.0)

    def preprocessing_costs_cents(self) -> int:
        """Total preprocessing costs in cents."""
        total = sum(c.cost_usd for c in self._costs if c.category == CostCategory.PREPROCESSING)
        return int(total * 100)

    def generation_costs_cents(self) -> int:
        """Total generation costs in cents."""
        total = sum(c.cost_usd for c in self._costs if c.category == CostCategory.GENERATION)
        return int(total * 100)

    def should_charge(self) -> bool:
        """
        Check if we should charge now.

        Returns True if:
        - Uncharged amount >= threshold, OR
        - Time since last charge >= interval
        """
        if self.uncharged_cents <= 0:
            return False

        # Threshold reached
        if self.uncharged_cents >= self.charge_threshold_cents:
            return True

        # Interval elapsed
        now = datetime.now(timezone.utc)
        elapsed = (now - self._last_charge_time).total_seconds()
        if elapsed >= self.charge_interval_seconds:
            return True

        return False

    def charge_if_needed(self) -> Optional[int]:
        """
        Charge uncharged costs to user balance if needed.

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
        if self.uncharged_cents <= 0:
            return None

        return self._do_charge()

    def _do_charge(self) -> int:
        """Execute a charge to the ledger."""
        amount = self.uncharged_cents
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
        """
        if self.spend_limit_cents is None:
            return True
        projected_spend = self.cumulative_spend_cents + self.uncharged_cents
        return projected_spend <= self.spend_limit_cents

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

        # Check spend limit
        if not self.is_within_spend_limit():
            return False, "spend_limit_exceeded"

        # Check user balance
        if not self.has_sufficient_balance():
            return False, "insufficient_balance"

        return True, None

    def get_summary(self) -> dict:
        """Get a summary of costs and charges."""
        summary = {
            "total_costs_cents": self.total_costs_cents,
            "preprocessing_costs_cents": self.preprocessing_costs_cents(),
            "generation_costs_cents": self.generation_costs_cents(),
            "total_charged_cents": self.total_charged_cents,
            "uncharged_cents": self.uncharged_cents,
            "cumulative_project_spend_cents": self.cumulative_spend_cents,
            "num_cost_entries": len(self._costs),
            "num_charges": len(self._charges),
            "user_balance_cents": self.get_user_balance_cents(),
        }

        # Add spend limit info if set
        if self.spend_limit_cents is not None:
            summary["spend_limit_cents"] = self.spend_limit_cents
            summary["remaining_budget_cents"] = self.remaining_budget_cents

        return summary