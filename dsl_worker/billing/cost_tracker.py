"""
Cost tracker for a project run.

Handles:
- Accumulating costs from API calls
- Charging to user credit balance at intervals
- Checking if user has sufficient balance
- THREAD-SAFE for concurrent workers

Credits: 1 credit = $0.25. Costs are tracked in USD internally,
then converted to credits when charging.
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
from dsl_api.models.account import Account
from dsl_api.credits import consume_credits, get_total_credits

logger = logging.getLogger(__name__)

# 1 credit = $0.25
CREDIT_VALUE_USD = 0.25


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
    """Record of a charge made."""
    timestamp: datetime
    amount_credits: float
    amount_cents: int  # For backward compat with cumulative_spend


class CostTracker:
    """
    Tracks costs for a single project run and handles charging.

    THREAD-SAFE: Uses locks for all state mutations.

    Args:
        db: Database session
        user_id: User being charged
        project_id: Project generating costs
        margin_multiplier: Multiply raw costs by this (e.g., 4.0 for 300% margin)
        charge_threshold_credits: Charge when accumulated costs reach this many credits
        charge_interval_seconds: Charge at least this often
    """

    def __init__(
        self,
        db: Session,
        user_id: UUID,
        project_id: UUID,
        margin_multiplier: float = 4.0,
        charge_threshold_cents: int = 100,  # Keep param name for backward compat
        charge_interval_seconds: int = 60,
    ):
        self.db = db
        self.user_id = user_id
        self.project_id = project_id
        self.margin_multiplier = margin_multiplier
        # Convert cents threshold to credits: 100 cents = 4 credits at $0.25/credit
        self.charge_threshold_credits = charge_threshold_cents / (CREDIT_VALUE_USD * 100)
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
        """Query total spend on this project from all previous runs (in cents)."""
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
        """Total spend on this project including current run (in cents for backward compat)."""
        with self._lock:
            charged_credits = sum(c.amount_credits for c in self._charges)
            charged_cents = int(charged_credits * CREDIT_VALUE_USD * 100)
            return self._cumulative_spend_at_start + charged_cents

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
            total_cost_credits = self._total_costs_credits_unlocked()

        cost_credits = cost_with_margin / CREDIT_VALUE_USD
        total_tokens = input_tokens + output_tokens
        if model:
            logger.info(
                f"💰 Cost: {phase} | {model} | "
                f"in={input_tokens:,} out={output_tokens:,} ({total_tokens:,} total) | "
                f"${cost_usd:.4f} raw → ${cost_with_margin:.4f} w/margin → {cost_credits:.2f} credits | "
                f"running total: {total_cost_credits:.2f} credits ({total_input:,}+{total_output:,} tokens)"
            )

    def _total_costs_usd_unlocked(self) -> float:
        return sum(c.cost_usd for c in self._costs)

    def _total_costs_credits_unlocked(self) -> float:
        return self._total_costs_usd_unlocked() / CREDIT_VALUE_USD

    def _total_charged_credits_unlocked(self) -> float:
        return sum(c.amount_credits for c in self._charges)

    def _uncharged_credits_unlocked(self) -> float:
        return self._total_costs_credits_unlocked() - self._total_charged_credits_unlocked()

    @property
    def total_costs_usd(self) -> float:
        with self._lock:
            return self._total_costs_usd_unlocked()

    @property
    def total_costs_credits(self) -> float:
        with self._lock:
            return self._total_costs_credits_unlocked()

    # Backward compat
    @property
    def total_costs_cents(self) -> int:
        return int(self.total_costs_usd * 100)

    @property
    def total_charged_credits(self) -> float:
        with self._lock:
            return self._total_charged_credits_unlocked()

    @property
    def total_charged_cents(self) -> int:
        return int(self.total_charged_credits * CREDIT_VALUE_USD * 100)

    @property
    def uncharged_credits(self) -> float:
        with self._lock:
            return self._uncharged_credits_unlocked()

    # Backward compat
    @property
    def uncharged_cents(self) -> int:
        return int(self.uncharged_credits * CREDIT_VALUE_USD * 100)

    def should_charge(self) -> bool:
        """Check if we should charge now."""
        with self._lock:
            uncharged = self._uncharged_credits_unlocked()
            last_charge = self._last_charge_time

        if uncharged <= 0:
            return False

        if uncharged >= self.charge_threshold_credits:
            return True

        now = datetime.now(timezone.utc)
        elapsed = (now - last_charge).total_seconds()
        if elapsed >= self.charge_interval_seconds:
            return True

        return False

    def charge_if_needed(self) -> Optional[float]:
        """Charge uncharged costs if needed. Returns credits charged or None."""
        if not self.should_charge():
            return None
        return self._do_charge()

    def charge_remaining(self) -> Optional[float]:
        """Charge any remaining uncharged costs."""
        with self._lock:
            if self._uncharged_credits_unlocked() <= 0:
                return None
        return self._do_charge()

    def _do_charge(self) -> float:
        """Execute a charge by consuming credits from user's balance."""
        with self._lock:
            credits_to_charge = self._uncharged_credits_unlocked()
            if credits_to_charge <= 0:
                return 0

            account = self.db.query(Account).filter(Account.user_id == self.user_id).first()
            if not account:
                logger.error(f"No account found for user {self.user_id}")
                return 0

            success = consume_credits(
                self.db, account, credits_to_charge, project_id=self.project_id
            )
            self.db.commit()

            record = ChargeRecord(
                timestamp=datetime.now(timezone.utc),
                amount_credits=credits_to_charge,
                amount_cents=int(credits_to_charge * CREDIT_VALUE_USD * 100),
            )
            self._charges.append(record)
            self._last_charge_time = datetime.now(timezone.utc)

        logger.info(
            f"Charged user {self.user_id}: {credits_to_charge:.2f} credits "
            f"(total charged: {self.total_charged_credits:.2f} credits)"
        )

        return credits_to_charge

    def get_user_balance_credits(self) -> float:
        """Get user's current total available credits."""
        account = self.db.query(Account).filter(Account.user_id == self.user_id).first()
        if not account:
            return 0
        return get_total_credits(self.db, account)

    # Backward compat
    def get_user_balance_cents(self) -> int:
        return int(self.get_user_balance_credits() * CREDIT_VALUE_USD * 100)

    def has_sufficient_balance(self) -> bool:
        """Check if user has enough balance to continue."""
        return self.get_user_balance_credits() > 0.01

    def check_balance_and_charge(self) -> tuple[bool, Optional[str]]:
        """Combined check: charge if needed, then verify balance."""
        self.charge_if_needed()

        if not self.has_sufficient_balance():
            return False, "insufficient_balance"

        return True, None

    def get_sample_cost_report(self, samples_generated: int, total_target_rows: int) -> dict:
        """
        Get cost report after sample phase for transparency UI.

        Returns credits used for samples and estimated total cost.
        """
        with self._lock:
            total_credits = self._total_costs_credits_unlocked()

        if samples_generated > 0:
            per_row_estimate = total_credits / samples_generated
            estimated_total = per_row_estimate * total_target_rows
        else:
            estimated_total = 0

        return {
            "samples_generated": samples_generated,
            "sample_credits_used": round(total_credits, 2),
            "estimated_total_credits": round(estimated_total, 2),
            "estimated_remaining_credits": round(estimated_total - total_credits, 2),
            "user_balance_credits": round(self.get_user_balance_credits(), 2),
        }

    def get_summary(self) -> dict:
        """Get a summary of costs and charges."""
        with self._lock:
            total_costs_credits = self._total_costs_credits_unlocked()
            total_charged_credits = self._total_charged_credits_unlocked()
            uncharged_credits = self._uncharged_credits_unlocked()
            num_costs = len(self._costs)
            num_charges = len(self._charges)
            total_input_tokens = sum(c.input_tokens for c in self._costs)
            total_output_tokens = sum(c.output_tokens for c in self._costs)

        return {
            "total_costs_credits": round(total_costs_credits, 2),
            "total_charged_credits": round(total_charged_credits, 2),
            "uncharged_credits": round(uncharged_credits, 2),
            # Backward compat in cents
            "total_costs_cents": int(total_costs_credits * CREDIT_VALUE_USD * 100),
            "total_charged_cents": int(total_charged_credits * CREDIT_VALUE_USD * 100),
            "uncharged_cents": int(uncharged_credits * CREDIT_VALUE_USD * 100),
            "cumulative_project_spend_cents": self.cumulative_spend_cents,
            "num_cost_entries": num_costs,
            "num_charges": num_charges,
            "user_balance_credits": round(self.get_user_balance_credits(), 2),
            "total_input_tokens": total_input_tokens,
            "total_output_tokens": total_output_tokens,
            "total_tokens": total_input_tokens + total_output_tokens,
        }
