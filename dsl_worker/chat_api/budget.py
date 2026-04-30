"""Budget caps and confirmation chip helpers for chat-mode runs.

The chat agent self-limits via prompt guidance, but unbounded scope
("list all people") and known-expensive enrichment can still blow past
what is reasonable for a single turn. This module:

- Computes a per-turn soft cap from (balance, effort).
- Provides helpers to translate cap-breach signals into structured
  budget_check events the FE renders as approval chips.
- Exposes a balance lookup so other modules don't need to reach into
  the api package's credits helper themselves.

The cap is advisory inside the run (BillingMeter polls it between
tool calls and breaks the loop) and exposed to the agent prompt so
it can self-limit before fanning out expensive work.

Design choice: the cap is **tier-agnostic**. Whether the user is on
free, starter, growth, or pro, the FIRST attempt at any turn should
spend a small amount (<10 credits) before checking in with the user.
Tiers buy you more total credits per month, not bigger unsupervised
turn budgets. If the user wants to authorize more on a specific turn,
they click the chip and the next turn runs with the larger cap.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional
from uuid import UUID

from sqlalchemy.orm import Session

from dsl_api.credits import CENTS_PER_CREDIT, get_credit_balance
from dsl_api.models import Account


log = logging.getLogger(__name__)


# Per-turn soft cap, in cents (of OpenAI/provider compute spend, NOT
# user-payment dollars). 1 credit = $0.10 of compute, so 100 cents =
# 10 credits drained from the user's balance per cap-sized turn.
#
# This is a SOFT cap — the agent sees it in its context message and
# uses it as guidance for when to call confirm_budget. Nothing in
# the system programmatically stops a turn when this is exceeded.
# The only hard stop is out_of_credits (real $0 balance).
#
# One number for everyone, regardless of plan or effort tier. The
# user explicitly asked for tier-agnostic + simple. The agent gets
# the user's actual balance in context so it can be more careful
# when balance is low.
#
# Why 10 credits (down from 25): user principle is "the user should
# never be surprised by pricing — they should know what's happening."
# A lower cap means the agent over-communicates by default — write
# the cost intent for almost any work, and ask permission earlier.
# This is paired with prompt-side guidance to ALWAYS write a 1-line
# estimate before rows_fill, regardless of size.
_BASE_TURN_CAP_CENTS = 100  # 10 credits

# Ceiling on what a single "raise the cap" approval chip can authorize.
# 2000 cents of compute = 200 credits drained — a meaningful spend
# for a single turn, but well short of letting one misclick drain a
# whole month's grant. Anything bigger should require an explicit,
# harder-to-misclick confirmation flow that doesn't exist yet.
_MAX_SINGLE_APPROVAL_CENTS = 2000  # 200 credits

# Max fraction of remaining balance a single approval can authorize.
# Belt-and-suspenders alongside _MAX_SINGLE_APPROVAL_CENTS so a low-
# balance user can't drain themselves on one click.
_MAX_APPROVAL_BALANCE_FRACTION = 0.8


def lookup_balance_cents(db: Session, user_id: UUID) -> int:
    """Return the user's available credit balance, in cents.
    Returns 0 on any lookup failure — callers treat 0 as "very tight"
    and the soft cap floors at $0.50 anyway.
    """
    try:
        account = (
            db.query(Account).filter(Account.user_id == str(user_id)).first()
        )
        if account is None:
            return 0
        bal = get_credit_balance(db, account)
        credits_avail = float(bal.get("total_available") or 0)
        return int(credits_avail * CENTS_PER_CREDIT)
    except Exception:
        log.exception("lookup_balance_cents: failed for user %s", user_id)
        return 0


def compute_soft_cap_cents(
    balance_cents: int,
    effort: Optional[str],
) -> int:
    """Return the per-turn soft cap, in cents.

    One flat number for everyone — see module docstring. Args are
    accepted for backward compatibility (and so the call site reads
    naturally) but currently ignored: the cap doesn't scale by
    balance or effort. Agent sees balance in context separately and
    is expected to use judgment when balance is low.
    """
    del balance_cents, effort  # accepted for caller ergonomics; unused
    return _BASE_TURN_CAP_CENTS


def format_credits(cents: int) -> str:
    """Render a credit count for FE/agent consumption. Credits are the
    user-facing unit; we never surface raw USD anywhere the user can
    see it.

    1 credit covers $0.10 of provider compute (api/config:
    COMPUTE_COST_PER_CREDIT). So credits = cents / 10.
    """
    credits = cents / 10.0
    # Drop trailing .0 for integer-valued credit counts so the string
    # reads "6 credits" not "6.0 credits".
    if credits >= 10:
        return f"{round(credits)} credits"
    if credits >= 1:
        if abs(credits - round(credits)) < 0.05:
            return f"{round(credits)} credits"
        return f"{credits:.1f} credits"
    return f"{credits:.2f} credits"


# Keep the old name as an alias for any caller still using it. Internal
# code should switch to format_credits.
def format_cents(cents: int) -> str:  # noqa: D401
    return format_credits(cents)


def approval_ceiling_cents(balance_cents: int) -> int:
    """Hard ceiling on what one approval chip can authorize. Lower of
    a flat dollar cap and a fraction of remaining balance — neither
    a healthy user nor a low-balance user can be drained on one click.
    20 cents = 2 credits floor so the ceiling is never useless.
    """
    by_balance = max(20, int(balance_cents * _MAX_APPROVAL_BALANCE_FRACTION))
    return min(_MAX_SINGLE_APPROVAL_CENTS, by_balance)


def safe_raised_cap_cents(
    *, current_cap_cents: int, balance_cents: int
) -> int:
    """The cap value to authorize when the user clicks 'keep going'.
    Doubles the current cap, capped at the approval ceiling so a
    runaway click can't authorize an unbounded spend.
    """
    raised = current_cap_cents * 2
    return min(raised, approval_ceiling_cents(balance_cents))


def build_budget_check_payload(
    *,
    summary: str,
    spent_cents: int,
    cap_cents: int,
    options: List[Dict[str, Any]],
    projection_cents: Optional[int] = None,
    reason: str = "soft_cap_hit",
) -> Dict[str, Any]:
    """Shape the structured payload sent over SSE as a `budget_check`
    event. The FE renders this as a chip block with a cost preview
    header.

    options items:
      - label: chip text (~40 chars)
      - message: what to send when clicked (becomes next user message)
      - cap_override_cents (optional): when present, the next run
        starts with this cap instead of recomputing from tier+balance.

    reason is informational — drives the FE icon/copy:
      'scope_ambiguous'        — agent recognized vague scope before working
      'projection_exceeds_cap' — agent estimates the work would blow cap
      'soft_cap_hit'           — legacy/reserved (no longer auto-emitted;
                                  kept so existing chat history with this
                                  reason value still parses)
    """
    return {
        "summary": summary,
        "spent_cents": spent_cents,
        "cap_cents": cap_cents,
        "projection_cents": projection_cents,
        "reason": reason,
        "options": options,
    }


def auto_chips_for_soft_cap(
    *,
    spent_cents: int,
    cap_cents: int,
    balance_cents: int,
) -> List[Dict[str, Any]]:
    """Build the chip set for the BillingMeter passive tripwire — used
    when the cap fires without the agent being involved. Three options:
    continue with raised cap, stop here, or have the agent suggest a
    cheaper plan.
    """
    raised = safe_raised_cap_cents(
        current_cap_cents=cap_cents, balance_cents=balance_cents
    )
    return [
        {
            "label": f"Keep going (up to {format_credits(raised)})",
            "message": (
                f"Keep going on this turn. I approve up to "
                f"{format_credits(raised)} of total spend for this turn."
            ),
            "cap_override_cents": raised,
        },
        {
            "label": "Stop here, commit what you have",
            "message": (
                "Stop here and commit whatever rows or partial data "
                "you've gathered so far. Don't spend more on this turn."
            ),
        },
        {
            "label": "Try a cheaper approach",
            "message": (
                "That's too expensive. Suggest a cheaper way to get "
                "this data — narrower scope, different source, or a "
                "smaller batch first."
            ),
        },
    ]
