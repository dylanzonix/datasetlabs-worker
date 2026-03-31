"""
Cost tracker for a project run.

Handles:
- Accumulating costs from API calls
- Charging to user credit balance at intervals
- Checking if user has sufficient balance

Credits are consumed based on raw OpenAI cost divided by a configurable
compute_cost_per_credit rate (e.g. $0.10 means 1 credit = $0.10 of compute).
"""

import logging
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional, List
from uuid import UUID

from sqlalchemy import func as sql_func
from sqlalchemy.orm import Session

from dsl_api.models.balance_ledger import BalanceLedger
from dsl_api.models.account import Account
from dsl_api.credits import consume_credits, get_total_credits
from dsl_api.plans import CENTS_PER_CREDIT

logger = logging.getLogger(__name__)

# Cache balance checks for this many seconds to avoid hammering the DB
BALANCE_CACHE_TTL = 10.0


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
    amount_cents: int


class CostTracker:
    """
    Tracks costs for a single project run and handles charging.

    Thread-safe via threading.Lock (DB calls are synchronous).

    Args:
        db: Database session
        user_id: User being charged
        project_id: Project generating costs
        compute_cost_per_credit: Raw OpenAI USD cost that 1 credit covers
        charge_threshold_cents: Charge when accumulated costs reach this (in ledger cents)
        charge_interval_seconds: Charge at least this often
    """

    def __init__(
        self,
        db: Session,
        user_id: UUID,
        project_id: UUID,
        compute_cost_per_credit: float = 0.10,
        charge_threshold_cents: int = 100,
        charge_interval_seconds: int = 60,
    ):
        self.db = db
        self.user_id = user_id
        self.project_id = project_id
        self.compute_cost_per_credit = compute_cost_per_credit
        self.charge_threshold_credits = charge_threshold_cents / CENTS_PER_CREDIT
        self.charge_interval_seconds = charge_interval_seconds

        # Append-only lists
        self._costs: List[CostEntry] = []
        self._charges: List[ChargeRecord] = []

        # Timing
        self._last_charge_time: datetime = datetime.now(timezone.utc)
        self._start_time: datetime = datetime.now(timezone.utc)

        # Cache cumulative spend at init
        self._cumulative_spend_at_start = self._query_cumulative_project_spend()

        # Balance cache
        self._cached_balance: Optional[float] = None
        self._balance_cache_time: float = 0.0

        # Thread-safe lock
        self._lock = threading.Lock()

    def seed_from_checkpoint(self, total_cost_usd: float) -> None:
        """Seed tracker with costs from a prior run (crash recovery).

        On worker restart, the CostTracker is created fresh with $0.
        This method restores the cost state from the checkpoint so that
        uncharged costs are correctly computed and charged.
        """
        if total_cost_usd <= 0:
            return

        with self._lock:
            self._costs.append(CostEntry(
                timestamp=datetime.now(timezone.utc),
                phase="recovered",
                cost_usd=total_cost_usd,
                description="Recovered from checkpoint after restart",
            ))

        # Figure out how much was already charged by querying the ledger
        already_charged_cents = self._query_charged_since_start()
        if already_charged_cents > 0:
            already_charged_credits = already_charged_cents / CENTS_PER_CREDIT
            with self._lock:
                self._charges.append(ChargeRecord(
                    timestamp=datetime.now(timezone.utc),
                    amount_credits=already_charged_credits,
                    amount_cents=already_charged_cents,
                ))

        uncharged = self.uncharged_credits
        logger.info(
            f"[CostTracker] Seeded from checkpoint: "
            f"${total_cost_usd:.4f} total, "
            f"{already_charged_cents / CENTS_PER_CREDIT:.2f} already charged, "
            f"{uncharged:.2f} uncharged credits"
        )

        # Charge the uncharged remainder now
        self.charge_if_needed()

    def _query_charged_since_start(self) -> int:
        """Query cents already charged for this project since tracker start."""
        result = (
            self.db.query(sql_func.sum(sql_func.abs(BalanceLedger.amount)))
            .filter(
                BalanceLedger.project_id == self.project_id,
                BalanceLedger.amount < 0,
                BalanceLedger.created_at >= self._start_time,
            )
            .scalar()
        )
        return int(result) if result else 0

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
        """Total spend on this project including current run (in cents)."""
        with self._lock:
            charged_credits = sum(c.amount_credits for c in self._charges)
            charged_cents = int(charged_credits * CENTS_PER_CREDIT)
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
        """Add a cost entry. cost_usd is the raw OpenAI cost."""
        if cost_usd <= 0:
            return

        entry = CostEntry(
            timestamp=datetime.now(timezone.utc),
            phase=phase,
            cost_usd=cost_usd,
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

        cost_credits = cost_usd / self.compute_cost_per_credit
        total_tokens = input_tokens + output_tokens
        if model:
            logger.info(
                f"Cost: {phase} | {model} | "
                f"in={input_tokens:,} out={output_tokens:,} ({total_tokens:,} total) | "
                f"${cost_usd:.4f} raw -> {cost_credits:.2f} credits | "
                f"running total: {total_cost_credits:.2f} credits ({total_input:,}+{total_output:,} tokens)"
            )

    def _total_costs_usd_unlocked(self) -> float:
        return sum(c.cost_usd for c in self._costs)

    def _total_costs_credits_unlocked(self) -> float:
        return self._total_costs_usd_unlocked() / self.compute_cost_per_credit

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

    @property
    def total_costs_cents(self) -> int:
        return int(self.total_costs_credits * CENTS_PER_CREDIT)

    @property
    def total_charged_credits(self) -> float:
        with self._lock:
            return self._total_charged_credits_unlocked()

    @property
    def total_charged_cents(self) -> int:
        return int(self.total_charged_credits * CENTS_PER_CREDIT)

    @property
    def uncharged_credits(self) -> float:
        with self._lock:
            return self._uncharged_credits_unlocked()

    @property
    def uncharged_cents(self) -> int:
        return int(self.uncharged_credits * CENTS_PER_CREDIT)

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
        """Charge uncharged costs if threshold or interval is met. Returns credits charged or None."""
        if not self.should_charge():
            return None
        return self._do_charge()

    def charge_remaining(self) -> Optional[float]:
        """Charge any remaining uncharged costs (call at end of run)."""
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

        # DB operations outside lock to minimize hold time
        account = self.db.query(Account).filter(Account.user_id == self.user_id).first()
        if not account:
            logger.error(f"No account found for user {self.user_id}")
            return 0

        success = consume_credits(
            self.db, account, credits_to_charge, project_id=self.project_id
        )

        if not success:
            # Partial balance — consume what's available, record that amount
            self.db.rollback()
            logger.warning(
                f"[CostTracker] Insufficient balance for {credits_to_charge:.2f} credits. "
                f"Charging what's available."
            )
            # Re-query available balance and charge that
            available = get_total_credits(self.db, account)
            if available > 0.01:
                success = consume_credits(
                    self.db, account, available, project_id=self.project_id
                )
                if success:
                    self.db.commit()
                    credits_to_charge = available
                else:
                    self.db.rollback()
                    return 0
            else:
                return 0
        else:
            self.db.commit()

        with self._lock:
            self._charges.append(ChargeRecord(
                timestamp=datetime.now(timezone.utc),
                amount_credits=credits_to_charge,
                amount_cents=int(credits_to_charge * CENTS_PER_CREDIT),
            ))
            self._last_charge_time = datetime.now(timezone.utc)
            # Invalidate balance cache after charging
            self._cached_balance = None

        logger.info(
            f"Charged user {self.user_id}: {credits_to_charge:.2f} credits "
            f"(total charged: {self.total_charged_credits:.2f} credits)"
        )

        return credits_to_charge

    def get_user_balance_credits(self) -> float:
        """Get user's current total available credits (cached)."""
        now = time.monotonic()
        if self._cached_balance is not None and (now - self._balance_cache_time) < BALANCE_CACHE_TTL:
            return self._cached_balance

        account = self.db.query(Account).filter(Account.user_id == self.user_id).first()
        if not account:
            return 0
        balance = get_total_credits(self.db, account)
        self._cached_balance = balance
        self._balance_cache_time = now
        return balance

    def get_user_balance_cents(self) -> int:
        return int(self.get_user_balance_credits() * CENTS_PER_CREDIT)

    def has_sufficient_balance(self) -> bool:
        """Check if user has enough balance to continue."""
        return self.get_user_balance_credits() > 0.01

    def check_balance_and_charge(self) -> tuple[bool, Optional[str]]:
        """Combined check: charge if needed, then verify balance."""
        self.charge_if_needed()

        if not self.has_sufficient_balance():
            return False, "insufficient_balance"

        return True, None

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
            "total_costs_cents": int(total_costs_credits * CENTS_PER_CREDIT),
            "total_charged_cents": int(total_charged_credits * CENTS_PER_CREDIT),
            "uncharged_cents": int(uncharged_credits * CENTS_PER_CREDIT),
            "cumulative_project_spend_cents": self.cumulative_spend_cents,
            "num_cost_entries": num_costs,
            "num_charges": num_charges,
            "user_balance_credits": round(self.get_user_balance_credits(), 2),
            "total_input_tokens": total_input_tokens,
            "total_output_tokens": total_output_tokens,
            "total_tokens": total_input_tokens + total_output_tokens,
        }
