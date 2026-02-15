"""
Cost tracker for a project run.

Handles:
- Accumulating costs from API calls
- Charging to user balance at intervals
- Checking if user has sufficient balance
- THREAD-SAFE for concurrent workers
"""

import logging
import threading
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional, List
from uuid import UUID

from sqlalchemy import func as sql_func
from sqlalchemy.orm import Session

from dsl_api.models.balance_ledger import BalanceLedger

logger = logging.getLogger(__name__)


@dataclass
class CostEntry:
    """A single cost entry (append-only)."""
    timestamp: datetime
    phase: str
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


class CostTracker:
    """
    Tracks costs for a single project run and handles charging.

    THREAD-SAFE: Uses locks for all state mutations.

    Args:
        db: Database session
        user_id: User being charged
        project_id: Project generating costs
        margin_multiplier: Multiply raw costs by this (e.g., 2.0 for 100% margin)
        charge_threshold_cents: Charge when accumulated costs reach this
        charge_interval_seconds: Charge at least this often
    """

    def __init__(
        self,
        db: Session,
        user_id: UUID,
        project_id: UUID,
        margin_multiplier: float = 2.0,
        charge_threshold_cents: int = 100,
        charge_interval_seconds: int = 60,
    ):
        self.db = db
        self.user_id = user_id
        self.project_id = project_id
        self.margin_multiplier = margin_multiplier
        self.charge_threshold_cents = charge_threshold_cents
        self.charge_interval_seconds = charge_interval_seconds

        # Append-only lists
        self._costs: List[CostEntry] = []
        self._charges: List[ChargeRecord] = []

        # Timing
        self._last_charge_time: datetime = datetime.now(timezone.utc)
        self._start_time: datetime = datetime.now(timezone.utc)

        # Cache cumulative spend at init
        self._cumulative_spend_at_start = self._query_cumulative_project_spend()

        # Thread-safe lock
        self._lock = threading.Lock()

    def _query_cumulative_project_spend(self) -> int:
        """Query total spend on this project from all previous runs."""
        result = (
            self.db.query(sql_func.sum(BalanceLedger.amount))
            .filter(
                BalanceLedger.project_id == self.project_id,
                BalanceLedger.amount < 0
            )
            .scalar()
        )
        return abs(result) if result else 0

    @property
    def cumulative_spend_cents(self) -> int:
        """Total spend on this project including current run."""
        with self._lock:
            return self._cumulative_spend_at_start + self._total_charged_cents_unlocked()

    def _total_charged_cents_unlocked(self) -> int:
        """Get total charged cents (must hold lock)."""
        return sum(c.amount_cents for c in self._charges)

    def add_cost(
        self,
        phase: str,
        cost_usd: float,
        description: str = "",
        input_tokens: int = 0,
        output_tokens: int = 0,
        model: str = "",
    ) -> None:
        """Add a cost entry."""
        if cost_usd <= 0:
            return

        cost_with_margin = cost_usd * self.margin_multiplier

        entry = CostEntry(
            timestamp=datetime.now(timezone.utc),
            phase=phase,
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

        total_tokens = input_tokens + output_tokens
        if model:
            logger.info(
                f"💰 Cost: {phase} | {model} | "
                f"in={input_tokens:,} out={output_tokens:,} ({total_tokens:,} total) | "
                f"${cost_usd:.4f} raw → ${cost_with_margin:.4f} w/margin | "
                f"running total: {total_cost_cents}¢ ({total_input:,}+{total_output:,} tokens)"
            )

    def _total_costs_usd_unlocked(self) -> float:
        return sum(c.cost_usd for c in self._costs)

    def _total_costs_cents_unlocked(self) -> int:
        return int(self._total_costs_usd_unlocked() * 100)

    def _uncharged_cents_unlocked(self) -> int:
        return self._total_costs_cents_unlocked() - self._total_charged_cents_unlocked()

    @property
    def total_costs_usd(self) -> float:
        with self._lock:
            return self._total_costs_usd_unlocked()

    @property
    def total_costs_cents(self) -> int:
        with self._lock:
            return self._total_costs_cents_unlocked()

    @property
    def total_charged_cents(self) -> int:
        with self._lock:
            return self._total_charged_cents_unlocked()

    @property
    def uncharged_cents(self) -> int:
        with self._lock:
            return self._uncharged_cents_unlocked()

    def should_charge(self) -> bool:
        """Check if we should charge now."""
        with self._lock:
            uncharged = self._uncharged_cents_unlocked()
            last_charge = self._last_charge_time

        if uncharged <= 0:
            return False

        if uncharged >= self.charge_threshold_cents:
            return True

        now = datetime.now(timezone.utc)
        elapsed = (now - last_charge).total_seconds()
        if elapsed >= self.charge_interval_seconds:
            return True

        return False

    def charge_if_needed(self) -> Optional[int]:
        """Charge uncharged costs if needed."""
        if not self.should_charge():
            return None
        return self._do_charge()

    def charge_remaining(self) -> Optional[int]:
        """Charge any remaining uncharged costs."""
        with self._lock:
            if self._uncharged_cents_unlocked() <= 0:
                return None
        return self._do_charge()

    def _do_charge(self) -> int:
        """Execute a charge to the ledger."""
        with self._lock:
            amount = self._uncharged_cents_unlocked()
            if amount <= 0:
                return 0

            ledger_entry = BalanceLedger(
                id=uuid.uuid4(),
                user_id=self.user_id,
                amount=-amount,
                reason="project_usage",
                project_id=self.project_id,
            )
            self.db.add(ledger_entry)
            self.db.commit()

            record = ChargeRecord(
                timestamp=datetime.now(timezone.utc),
                amount_cents=amount,
                ledger_id=ledger_entry.id,
            )
            self._charges.append(record)
            self._last_charge_time = datetime.now(timezone.utc)

        logger.info(
            f"Charged user {self.user_id}: {amount}¢ "
            f"(total charged: {self.total_charged_cents}¢)"
        )

        return amount

    def get_user_balance_cents(self) -> int:
        """Get user's current balance."""
        result = (
            self.db.query(sql_func.sum(BalanceLedger.amount))
            .filter(BalanceLedger.user_id == self.user_id)
            .scalar()
        )
        return result or 0

    def has_sufficient_balance(self) -> bool:
        """Check if user has enough balance to continue."""
        return self.get_user_balance_cents() > 0

    def check_balance_and_charge(self) -> tuple[bool, Optional[str]]:
        """Combined check: charge if needed, then verify balance."""
        self.charge_if_needed()

        if not self.has_sufficient_balance():
            return False, "insufficient_balance"

        return True, None

    def get_summary(self) -> dict:
        """Get a summary of costs and charges."""
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

        return summary