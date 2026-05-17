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
from typing import Any, Dict, List, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import text as sa_text
from sqlalchemy.orm import Session

from dsl_api.auth import CurrentUser, get_current_user
from dsl_api.db import SessionLocal
from dsl_api.models import Project
from dsl_worker.chat_v2.tools import (
    ToolContext,
    table_extend,
    filter_set,
    filter_clear,
)
from dsl_worker.chat_v2.enrichment import enrichment_run


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


_RESEARCH_VALUES = {"classify", "light", "standard", "deep"}
_LEGACY_TIER_TO_RESEARCH = {
    "lookup":   "standard",
    "research": "deep",
    # v1 names that got churned in v2
    "fast":     "classify",
    "smart":    "light",
    "expert":   "standard",
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
    return {"ok": True, "research": action.get("research"), "per_row_credit_cap": cap}


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
    result, cost = await enrichment_run(
        {"enrichment_id": str(enrichment_id), "scope": scope, "overwrite": body.overwrite}, ctx
    )
    return {"result": result, "cost_usd": cost}


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


# ---- Approvals (minimal v1) -----------------------------------------------
# For v1 the approval mechanism is server-pause-then-resume via in-memory
# event. The agent loop is synchronous, so approvals don't come into play
# until we wire streaming. These endpoints exist for FE forward-compat.


@router.get("/projects/{project_id}/approvals")
def list_approvals(
    project_id: UUID,
    user: CurrentUser = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    _verify(project_id, user.user_id, db)
    # Approval table not yet in v1; returns empty for now. When streaming +
    # approval flow is wired, this surfaces pending approvals.
    return {"approvals": []}


class ApprovalDecision(BaseModel):
    approved: bool


@router.post("/projects/{project_id}/approvals/{approval_id}")
def resolve_approval(
    project_id: UUID,
    approval_id: UUID,
    body: ApprovalDecision,
    user: CurrentUser = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    _verify(project_id, user.user_id, db)
    # Stub: when approvals are wired, this signals the paused agent.
    return {"ok": True, "approved": body.approved}
