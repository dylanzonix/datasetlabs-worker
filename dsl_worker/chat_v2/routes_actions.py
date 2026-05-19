"""User-initiated action endpoints (button-triggered).

Distinct from agent-initiated tool calls — these bypass the LLM and act
directly on tables/enrichments.

  POST /v2/projects/{pid}/tables/{tid}/fetch_more     → table_extend with stored params
  POST /v2/projects/{pid}/enrichments/{eid}/run       → enrichment_run scope
  POST /v2/projects/{pid}/tables/{tid}/filters         → filter_set
  DELETE /v2/projects/{pid}/tables/{tid}/filters/{column} → filter_clear
  GET    /v2/projects/{pid}/approvals                  → list pending
  POST   /v2/projects/{pid}/approvals/{aid}            → resolve {approved}
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional
from uuid import UUID

log = logging.getLogger(__name__)

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import text as sa_text
from sqlalchemy.orm import Session

from dsl_api.auth import CurrentUser, get_current_user
from dsl_api.db import SessionLocal
from dsl_api.models import Project
from dsl_api.models.balance_ledger import BalanceLedger
from dsl_worker.chat_v2.tools import (
    ToolContext,
    table_extend,
    filter_set,
    filter_clear,
)
from dsl_worker.chat_v2.enrichment import enrichment_run
from dsl_worker.chat_v2.approvals import REGISTRY as APPROVALS
from dsl_worker.chat_v2.cancels import REGISTRY as CANCELS


router = APIRouter(prefix="/v2")


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def _verify(project_id: UUID, user_id: UUID, db: Session) -> Project:
    p = db.query(Project).filter(
        Project.id == project_id,
        Project.user_id == user_id,
        Project.deleted_at.is_(None),
    ).first()
    if not p:
        raise HTTPException(404, "Project not found")
    return p


class FetchMoreBody(BaseModel):
    n: int = 100


@router.post("/projects/{project_id}/tables/{table_id}/fetch_more")
async def post_fetch_more(
    project_id: UUID,
    table_id: str,
    body: FetchMoreBody,
    user: CurrentUser = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """User-initiated fetch_more. Calls table_extend with empty query_params
    delta — server uses stored cursor / query_params on the table."""
    _verify(project_id, user.user_id, db)
    ctx = ToolContext(db=db, project_id=str(project_id), user_id=str(user.user_id), run_id=None)
    result, cost = await table_extend(
        {"table_id": str(table_id), "query_params": {}, "n": body.n}, ctx
    )
    return {"result": result, "cost_usd": cost}


class RunEnrichmentBody(BaseModel):
    scope_type: str = "all_unfilled"   # all_unfilled | first_n | row_ids
    first_n: Optional[int] = None
    row_ids: Optional[List[str]] = None
    overwrite: bool = False


class PatchEnrichmentBody(BaseModel):
    # New: research level. Legacy `tier` accepted via alias below.
    research: Optional[str] = None  # fast | smart | expert | standard | deep
    tier: Optional[str] = None      # legacy: classify | lookup | research
    per_row_credit_cap: Optional[float] = None


_RESEARCH_VALUES = {"none", "low", "medium", "high"}
_LEGACY_TIER_TO_RESEARCH = {
    # v3 (classify/lookup/search/investigate)
    "classify":    "none",
    "lookup":      "low",
    "search":      "medium",
    "investigate": "high",
    # v2 (light)
    "light":       "low",
    # v1 (fast/smart/standard/deep/expert)
    "fast":        "none",
    "smart":       "low",
    "expert":      "medium",
    "standard":    "medium",
    "deep":        "high",
    # v0 (research as the highest tier)
    "research":    "high",
}


@router.patch("/projects/{project_id}/enrichments/{enrichment_id}")
def patch_enrichment(
    project_id: UUID,
    enrichment_id: str,
    body: PatchEnrichmentBody,
    user: CurrentUser = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Update research level (writes action.research) and/or per_row_credit_cap.

    Accepts the new `research` field as well as the legacy `tier` field
    (aliased to research). Always writes the new field on the action.
    """
    _verify(project_id, user.user_id, db)
    row = db.execute(
        sa_text(
            "SELECT id::text, action, per_row_credit_cap FROM enrichments "
            "WHERE (short_id = :eid OR id::text = :eid) AND deleted_at IS NULL"
        ),
        {"eid": enrichment_id},
    ).fetchone()
    if not row:
        raise HTTPException(404, "Enrichment not found")
    eid_uuid = row[0]
    action = row[1] if isinstance(row[1], dict) else json.loads(row[1] or "{}")
    cap = row[2]

    # Resolve research target: prefer new field, fall back to legacy tier alias.
    research: Optional[str] = None
    if body.research is not None:
        research = body.research.lower()
    elif body.tier is not None:
        research = _LEGACY_TIER_TO_RESEARCH.get(body.tier.lower(), body.tier.lower())
    if research is not None:
        if research not in _RESEARCH_VALUES:
            raise HTTPException(
                400,
                f"research must be one of {sorted(_RESEARCH_VALUES)} (got {research!r})",
            )
        action["research"] = research
        # Drop legacy `tier` if it lingered on this enrichment.
        action.pop("tier", None)
    if body.per_row_credit_cap is not None:
        cap = body.per_row_credit_cap

    db.execute(
        sa_text(
            "UPDATE enrichments SET action = CAST(:a AS jsonb), "
            "per_row_credit_cap = :cap WHERE id = :id"
        ),
        {"a": json.dumps(action), "cap": cap, "id": eid_uuid},
    )
    db.commit()
    # Cast for JSON serializer — column is numeric(8,2), comes back as
    # Decimal when read from DB. Outgoing body always a plain number.
    return {
        "ok": True,
        "research": action.get("research"),
        "per_row_credit_cap": float(cap) if cap is not None else None,
    }


@router.post("/projects/{project_id}/enrichments/{enrichment_id}/run")
async def post_run_enrichment(
    project_id: UUID,
    enrichment_id: str,
    body: RunEnrichmentBody,
    user: CurrentUser = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    _verify(project_id, user.user_id, db)
    scope: Dict[str, Any] = {"type": body.scope_type}
    if body.first_n is not None:
        scope["first_n"] = body.first_n
    if body.row_ids is not None:
        scope["row_ids"] = body.row_ids
    ctx = ToolContext(db=db, project_id=str(project_id), user_id=str(user.user_id), run_id=None)

    # Resolve eid early so cancel registry uses the canonical UUID (matches
    # what /cancel resolves to). Otherwise short_id vs uuid mismatches
    # would prevent the Stop button from finding the task.
    from dsl_worker.chat_v2.tools import resolve_enrichment_id
    canonical_eid = resolve_enrichment_id(db, str(project_id), enrichment_id) or str(enrichment_id)

    # Wrap the run in an asyncio.Task so the Stop button can cancel it.
    # enrichment.py's cell loop already handles asyncio.CancelledError
    # (cancels in-flight per-cell tasks + flushes partial cost).
    import asyncio as _asyncio
    task: _asyncio.Task = _asyncio.create_task(
        enrichment_run(
            {"enrichment_id": str(enrichment_id), "scope": scope, "overwrite": body.overwrite},
            ctx,
        )
    )
    await CANCELS.register(str(project_id), canonical_eid, task)
    cost = 0.0
    cancelled = False
    try:
        result, cost = await task
    except _asyncio.CancelledError:
        cancelled = True
        # Partial cost may have accrued in ctx.partial_cost_usd before cancel.
        cost = float(getattr(ctx, "partial_cost_usd", 0.0) or 0.0)
        result = {"cancelled": True}
    finally:
        await CANCELS.unregister(str(project_id), canonical_eid, task)

    # Charge the user. Chat-initiated enrichment_run is charged via
    # runs.py's end-of-turn flush, but THIS path (FE-clicked ▶) had no
    # ledger write — runs cost real money on Apollo / FE / BU but
    # nothing got deducted, so the project's spend display stayed flat
    # and the user's balance didn't reflect actual usage.
    spend_cents = int(round(float(cost or 0.0) * 100))
    if spend_cents > 0:
        try:
            db.add(BalanceLedger(
                user_id=user.user_id,
                amount=-spend_cents,
                reason="enrichment_run_rest",
                project_id=project_id,
            ))
            db.commit()
        except Exception:
            try:
                db.rollback()
            except Exception:
                pass

    # Patch-first response: fetch the affected rows fresh from samples
    # and return them so the FE can applyCellEdits without bouncing the
    # whole table. Chat-initiated runs emit cell_filled SSE events for
    # this purpose; the REST path uses run_id=None and never streams,
    # so without this the FE only saw the new value on the next
    # refreshRows() call. The Tier-1 cleanup removed that refresh, so
    # values now flow through this payload instead.
    updated_rows: List[Dict[str, Any]] = []
    if not cancelled and body.row_ids:
        try:
            from sqlalchemy import bindparam
            # IMPORTANT: `IN :ids` (not `ANY(:ids)`) with expanding=True —
            # SQLAlchemy expands the bind into a tuple of placeholders,
            # and Postgres ANY() needs an array, not a tuple. With ANY()
            # the query silently errored and updated_rows came back
            # empty, so the FE cleared the spinner without patching
            # anything (cells stayed blank until manual refresh).
            # `id::text IN :ids` also avoids the uuid-vs-text comparison
            # cast since body.row_ids is a list of plain strings.
            stmt = (
                sa_text(
                    "SELECT id::text, row, tags "
                    "FROM samples WHERE id::text IN :ids AND deleted_at IS NULL"
                ).bindparams(bindparam("ids", expanding=True))
            )
            rows = db.execute(stmt, {"ids": list(body.row_ids)}).fetchall()
            for r in rows:
                payload: Dict[str, Any] = {"id": r[0]}
                row_data = r[1] if isinstance(r[1], dict) else (json.loads(r[1]) if r[1] else {})
                payload.update(row_data)
                if r[2]:
                    payload["tags"] = r[2] if isinstance(r[2], dict) else json.loads(r[2])
                updated_rows.append(payload)
        except Exception:
            log.exception("post_run_enrichment: failed to fetch updated rows for FE patch")

    return {"result": result, "cost_usd": cost, "cancelled": cancelled, "updated_rows": updated_rows}


class FilterBody(BaseModel):
    column: str
    op: str
    value: Any = None


@router.post("/projects/{project_id}/tables/{table_id}/filters")
async def post_filter(
    project_id: UUID,
    table_id: str,
    body: FilterBody,
    user: CurrentUser = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    _verify(project_id, user.user_id, db)
    ctx = ToolContext(db=db, project_id=str(project_id), user_id=str(user.user_id), run_id=None)
    result, _ = await filter_set(
        {"table_id": str(table_id), "column": body.column, "op": body.op, "value": body.value},
        ctx,
    )
    return result


@router.delete("/projects/{project_id}/tables/{table_id}/filters/{column}")
async def delete_filter(
    project_id: UUID,
    table_id: str,
    column: str,
    user: CurrentUser = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    _verify(project_id, user.user_id, db)
    ctx = ToolContext(db=db, project_id=str(project_id), user_id=str(user.user_id), run_id=None)
    result, _ = await filter_clear(
        {"table_id": str(table_id), "column": column}, ctx
    )
    return result


class SortBody(BaseModel):
    column: str
    direction: str = "desc"


@router.post("/projects/{project_id}/tables/{table_id}/sort")
async def post_sort(
    project_id: UUID,
    table_id: str,
    body: SortBody,
    user: CurrentUser = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    _verify(project_id, user.user_id, db)
    from dsl_worker.chat_v2.tools import sort_set as _sort_set
    ctx = ToolContext(db=db, project_id=str(project_id), user_id=str(user.user_id), run_id=None)
    result, _ = await _sort_set(
        {"table_id": str(table_id), "column": body.column, "direction": body.direction},
        ctx,
    )
    return result


@router.delete("/projects/{project_id}/tables/{table_id}/sort")
async def delete_sort(
    project_id: UUID,
    table_id: str,
    user: CurrentUser = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    _verify(project_id, user.user_id, db)
    from dsl_worker.chat_v2.tools import sort_clear as _sort_clear
    ctx = ToolContext(db=db, project_id=str(project_id), user_id=str(user.user_id), run_id=None)
    result, _ = await _sort_clear({"table_id": str(table_id)}, ctx)
    return result


# ---- Cell traces (debug) --------------------------------------------------


@router.get("/projects/{project_id}/enrichments/{enrichment_id}/traces")
def list_cell_traces(
    project_id: UUID,
    enrichment_id: str,
    limit: int = 50,
    user: CurrentUser = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Recent per-cell traces for an enrichment — tier, model, tool calls,
    final values, error, cost, duration. Read-only debug view."""
    _verify(project_id, user.user_id, db)
    # Resolve enrichment_id (accepts short_id like "e1" or uuid)
    eid_row = db.execute(
        sa_text(
            "SELECT e.id::text FROM enrichments e JOIN tables t ON t.id=e.table_id "
            "WHERE t.project_id=:p AND (e.id::text=:eid OR e.short_id=:eid) "
            "AND e.deleted_at IS NULL LIMIT 1"
        ),
        {"p": str(project_id), "eid": enrichment_id},
    ).fetchone()
    if not eid_row:
        raise HTTPException(404, "enrichment not found")
    eid = eid_row[0]
    rows = db.execute(
        sa_text(
            """
            SELECT ct.id::text, ct.sample_id::text, ct.tier, ct.model,
                   ct.tool_calls, ct.final_values, ct.error,
                   ct.cost_credits, ct.duration_ms, ct.created_at
            FROM cell_traces ct
            WHERE ct.enrichment_id=:eid
            ORDER BY ct.created_at DESC
            LIMIT :lim
            """
        ),
        {"eid": eid, "lim": limit},
    ).fetchall()
    return {
        "enrichment_id": enrichment_id,
        "traces": [
            {
                "id": r[0],
                "sample_id": r[1],
                "tier": r[2],
                "model": r[3],
                "tool_calls": r[4],
                "final_values": r[5],
                "error": r[6],
                "cost_credits": r[7],
                "duration_ms": r[8],
                "created_at": r[9].isoformat() if r[9] else None,
            }
            for r in rows
        ],
    }


# ---- Approvals ------------------------------------------------------------
# Tools in APPROVAL_REQUIRED (see approvals.py) pause the agent loop and
# wait for a decision from the FE. These endpoints surface that flow:
#   GET   /approvals               → list pending (used to rehydrate the
#                                    approval card after a reconnect)
#   POST  /approvals/{id}/respond  → resolve a pending approval


@router.get("/projects/{project_id}/approvals")
async def list_approvals(
    project_id: UUID,
    user: CurrentUser = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    _verify(project_id, user.user_id, db)
    return {"approvals": await APPROVALS.list_for_project(str(project_id))}


class ApprovalDecision(BaseModel):
    approved: bool


@router.post("/projects/{project_id}/approvals/{approval_id}/respond")
async def respond_to_approval(
    project_id: UUID,
    approval_id: str,
    body: ApprovalDecision,
    user: CurrentUser = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    _verify(project_id, user.user_id, db)
    found = await APPROVALS.resolve(approval_id, body.approved)
    if not found:
        # Approval already resolved (double-click) or no longer pending
        # (chat run ended) — treat as a no-op rather than 404 so the FE
        # doesn't surface a confusing error.
        return {"ok": True, "approved": body.approved, "found": False}
    return {"ok": True, "approved": body.approved, "found": True}


# ---- Cancel ---------------------------------------------------------------
# Cancels an in-flight REST enrichment run. Chat runs are cancelled via the
# SSE disconnect; this is the parallel path for the column ▶ button etc.


@router.post("/projects/{project_id}/enrichments/{enrichment_id}/cancel")
async def cancel_enrichment(
    project_id: UUID,
    enrichment_id: str,
    user: CurrentUser = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    _verify(project_id, user.user_id, db)
    # Resolve short_id (e1, e2…) to UUID if needed.
    from dsl_worker.chat_v2.tools import resolve_enrichment_id
    eid = resolve_enrichment_id(db, str(project_id), enrichment_id) or enrichment_id
    cancelled = await CANCELS.cancel_enrichment(str(project_id), eid)
    return {"ok": True, "cancelled": cancelled}


@router.get("/projects/{project_id}/enrichments/running")
async def list_running_enrichments(
    project_id: UUID,
    user: CurrentUser = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    _verify(project_id, user.user_id, db)
    return {"enrichment_ids": await CANCELS.list_running(str(project_id))}


@router.get("/projects/{project_id}/cells/running")
async def list_running_cells(
    project_id: UUID,
    user: CurrentUser = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Cells currently being filled by an in-flight enrichment run.

    Used by the FE on page mount + during active runs to rebuild
    pendingCells, so the per-cell spinner survives a refresh. Each
    entry: {enrichment_id (uuid), sample_id, columns: [...]}.
    """
    _verify(project_id, user.user_id, db)
    from dsl_worker.chat_v2.cell_runs import REGISTRY as CELL_RUNS
    return {"cells": await CELL_RUNS.list_for_project(str(project_id))}
