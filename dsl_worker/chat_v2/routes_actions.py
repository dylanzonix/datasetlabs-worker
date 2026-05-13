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
