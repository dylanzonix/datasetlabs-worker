"""Scrubby auto-verification for emails the LLM agent puts into cells.

Used by both rows_fill (per-cell agent path) and rows_fill_bulk_browser
(batched browser_use path) so every email the agent writes gets the same
treatment: skipped if it came from a paid provider (Apollo / FullEnrich)
since those vendors run their own waterfall, otherwise validated by
Scrubby with the result stamped onto the row's tags for the FE badge.

Concurrency-safe: each call opens its own SessionLocal so cell tasks
running in parallel don't collide on the same row's tags.

The verification task NEVER raises — failures emit status=UNVERIFIED so
the FE shows no badge (same as if the feature were disabled).
"""

from __future__ import annotations

import asyncio
import logging
import re
from typing import Any, Awaitable, Callable, Dict, List, Optional, Set, Tuple

from dsl_api.db import SessionLocal
from dsl_api.models.sample import Sample

from dsl_worker.infra.scrubby_client import ScrubbyStatus, get_scrubby_client


log = logging.getLogger(__name__)

ProgressCallback = Callable[[Dict[str, Any]], Awaitable[None]]


# Tools whose return text we mine for emails. Emails surfaced by these
# providers are treated as already-verified (skip Scrubby) — the user's
# explicit rule.
PROVIDER_ENRICH_TOOLS = {
    "apollo_enrich_person",
    "apollo_bulk_enrich_people",
    "fullenrich_enrich_contacts",
}

_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")


def extract_emails(text: str) -> Set[str]:
    """Lower-cased emails found in a tool result blob."""
    if not text:
        return set()
    return {m.group(0).lower() for m in _EMAIL_RE.finditer(text)}


def is_email_column(col_def: Optional[Dict[str, Any]], col_name: str) -> bool:
    """contact_type=='email' wins; fall back to a name match for columns
    added without the marker so email-ish columns still get verified."""
    if isinstance(col_def, dict) and col_def.get("contact_type") == "email":
        return True
    return bool(re.search(r"e[\W_-]*mail", col_name or "", re.IGNORECASE))


async def verify_and_apply(
    *,
    sample_id: str,
    column: str,
    email: str,
    progress_cb: Optional[ProgressCallback],
) -> None:
    """Validate `email` via Scrubby and persist the result.

    Effects (all best-effort, swallowing errors):
      • Stamps tags["email_verification"][col] = {value, status, source}
      • If Scrubby says Invalid: clears sample.row[col], appends to
        tags["failed_emails"][col], and writes a fill_status entry so
        the existing "couldn't fill" badge appears.
      • Emits email_verified SSE event with the final status.
      • Emits row_merged after value-clearing so the FE drops the cell.
    """
    scrubby = get_scrubby_client()
    if scrubby is None:
        log.info("scrubby: verify_and_apply called but client is None — skipping %s/%s", sample_id, column)
        return

    try:
        sr = await scrubby.validate_email(email)
    except Exception:
        log.exception("scrubby.validate_email raised for %s", email)
        sr = None
    status: ScrubbyStatus = sr.status if sr is not None else "UNVERIFIED"
    log.info("scrubby: %s/%s %s → %s (raw=%s)", sample_id, column, email, status, sr.raw_status if sr else "none")

    cleared = False
    row_snapshot: Optional[Dict[str, Any]] = None
    write_db = SessionLocal()
    try:
        sample = write_db.query(Sample).filter(Sample.id == sample_id).first()
        if sample is None:
            return
        tags = dict(sample.tags or {})
        verifications = dict(tags.get("email_verification") or {})
        verifications[column] = {
            "value": email,
            "status": status,
            "source": "scrubby",
            "raw_status": (sr.raw_status if sr is not None else None),
        }
        tags["email_verification"] = verifications
        if status == "INVALID":
            row = dict(sample.row or {})
            if row.get(column) == email:
                row[column] = None
                sample.row = row
                cleared = True
            failed = dict(tags.get("failed_emails") or {})
            bucket = list(failed.get(column) or [])
            if email.lower() not in {e.lower() for e in bucket}:
                bucket.append(email)
            failed[column] = bucket
            tags["failed_emails"] = failed
            fill_status = dict(tags.get("fill_status") or {})
            fill_status[column] = {
                "status": "null_legitimate",
                "reason": "Email failed verification — Scrubby marked it Invalid.",
                "cost": 0.0,
                "strategy": "scrubby_verify",
            }
            tags["fill_status"] = fill_status
        sample.tags = tags
        write_db.commit()
        write_db.refresh(sample)
        row_snapshot = {
            "_id": str(sample.id),
            "_seq": sample.seq,
            "_tags": sample.tags or {},
            **(sample.row or {}),
        }
    except Exception:
        log.exception("email verification persist failed: row=%s col=%s", sample_id, column)
        try:
            write_db.rollback()
        except Exception:
            pass
        return
    finally:
        write_db.close()

    if progress_cb is None:
        return
    try:
        await progress_cb({
            "type": "email_verified",
            "row_id": str(sample_id),
            "column": column,
            "status": status,
            "value": email if status != "INVALID" else None,
        })
        # Always push the updated row snapshot so the FE gets the new
        # tags.email_verification entry. Without this the badge logic
        # reads from stale client-side tags and never shows the green
        # check / risky icon. (Earlier this was gated on `cleared`
        # which only fires for INVALID — so DELIVERABLE/RISKY emails
        # silently did nothing in the UI.)
        if row_snapshot is not None:
            await progress_cb({"type": "row_merged", "row": row_snapshot})
    except Exception:
        log.exception("progress_cb raised in verify_and_apply (suppressed)")


def schedule_verifications(
    *,
    sample_id: str,
    written_values: Dict[str, Any],
    email_columns: Set[str],
    provider_emails: Set[str],
    progress_cb: Optional[ProgressCallback],
) -> List[asyncio.Task[None]]:
    """Fire verifications for a row's qualifying email values.

    Returns ONE task per row (not per column) — within a row, emails
    are verified sequentially so two columns on the same Sample don't
    race the tags JSON write (each verify reads tags, mutates,
    commits; concurrent writes would last-write-wins one of them).
    Across rows we still parallelize.

    Skips:
      • non-email columns
      • non-string / non-`@` values
      • emails already seen in Apollo / FullEnrich tool results
    """
    if get_scrubby_client() is None:
        log.info("scrubby: client unavailable (no SCRUBBY_API_KEY?) — skipping verifies for sample %s", sample_id)
        return []
    if not email_columns:
        return []
    provider_lower = {e.lower() for e in provider_emails}
    targets: List[Tuple[str, str]] = []
    for col, val in written_values.items():
        if col not in email_columns:
            continue
        if not isinstance(val, str) or "@" not in val:
            continue
        if val.lower() in provider_lower:
            continue
        targets.append((col, val))
    if not targets:
        return []
    log.info(
        "scrubby: scheduling verifies for sample %s — %d email(s) across cols %s",
        sample_id, len(targets), [c for c, _ in targets],
    )

    async def _run_for_row() -> None:
        for col, val in targets:
            if progress_cb is not None:
                try:
                    await progress_cb({
                        "type": "email_verifying",
                        "row_id": str(sample_id),
                        "column": col,
                        "value": val,
                    })
                except Exception:
                    log.exception("progress_cb email_verifying raised; suppressed")
            await verify_and_apply(
                sample_id=sample_id,
                column=col,
                email=val,
                progress_cb=progress_cb,
            )

    return [asyncio.create_task(_run_for_row())]
