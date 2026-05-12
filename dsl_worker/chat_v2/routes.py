"""v-next chat routes (mounted under /v2).

Minimal MVP:
  POST   /v2/projects/{project_id}/chat/turns      — run one turn synchronously
  GET    /v2/projects/{project_id}/tables          — list tables in project
  GET    /v2/projects/{project_id}/tables/{id}/rows — rows of a table
  POST   /v2/projects/{project_id}/approvals/{event_id}  — user approve/deny

The full SSE streaming + run lifecycle (pause/resume/reattach) can layer on
top later. For v1, run a turn synchronously and return the result. FE polls
or just gets the result back.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel
from sqlalchemy import text as sa_text
from sqlalchemy.orm import Session

from dsl_api.auth import CurrentUser, get_current_user
from dsl_api.db import SessionLocal
from dsl_api.models import Project
from dsl_worker.chat_v2.agent import run_turn


router = APIRouter(prefix="/v2")


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ---- Schemas --------------------------------------------------------------


class TurnRequest(BaseModel):
    message: str
    history: Optional[List[Dict[str, Any]]] = None


class TurnResponse(BaseModel):
    final_message: str
    tool_calls_made: List[Dict[str, Any]]
    total_cost_usd: float
    iterations: int


# ---- Helpers --------------------------------------------------------------


def _verify_project(project_id: UUID, user_id: UUID, db: Session) -> Project:
    p = db.query(Project).filter(
        Project.id == project_id,
        Project.user_id == user_id,
        Project.deleted_at.is_(None),
    ).first()
    if not p:
        raise HTTPException(404, "Project not found")
    return p


# ---- Routes ---------------------------------------------------------------


@router.post("/projects/{project_id}/chat/turns")
async def post_turn(
    project_id: UUID,
    body: TurnRequest,
    user: CurrentUser = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> TurnResponse:
    """Run one chat turn synchronously. Returns final message + tool calls."""
    _verify_project(project_id, user.user_id, db)
    result = await run_turn(
        db=db,
        project_id=str(project_id),
        user_id=str(user.user_id),
        run_id=None,
        user_message=body.message,
        history=body.history,
    )
    return TurnResponse(**result)


@router.get("/projects/{project_id}/tables")
def list_tables(
    project_id: UUID,
    user: CurrentUser = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """List tables in a project (multi-table workspace)."""
    _verify_project(project_id, user.user_id, db)
    rows = db.execute(
        sa_text(
            """
            SELECT id::text, name, source, columns, dedup_key_column,
                   last_fetch_returned_rows, last_fetch_cost_credits, last_fetch_at,
                   fetch_status, fetch_error, created_at
            FROM tables
            WHERE project_id = :pid AND deleted_at IS NULL
            ORDER BY created_at
            """
        ),
        {"pid": str(project_id)},
    ).fetchall()
    out = []
    for r in rows:
        row_count = db.execute(
            sa_text("SELECT COUNT(*) FROM samples WHERE table_id = :tid AND deleted_at IS NULL"),
            {"tid": r[0]},
        ).scalar() or 0
        cols = r[3] if isinstance(r[3], list) else json.loads(r[3] or "[]")
        out.append({
            "id": r[0],
            "name": r[1],
            "source": r[2],
            "columns": cols,
            "dedup_key_column": r[4],
            "row_count": row_count,
            "last_fetch_returned_rows": r[5],
            "last_fetch_cost_credits": float(r[6]) if r[6] is not None else None,
            "last_fetch_at": r[7].isoformat() if r[7] else None,
            "fetch_status": r[8],
            "fetch_error": r[9],
            "created_at": r[10].isoformat() if r[10] else None,
        })
    return {"tables": out}


@router.get("/projects/{project_id}/tables/{table_id}/rows")
def list_table_rows(
    project_id: UUID,
    table_id: UUID,
    limit: int = 200,
    offset: int = 0,
    user: CurrentUser = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Read rows of a table. Includes any active filters."""
    _verify_project(project_id, user.user_id, db)
    # Verify table belongs to this project
    owner_pid = db.execute(
        sa_text("SELECT project_id::text FROM tables WHERE id = :tid AND deleted_at IS NULL"),
        {"tid": str(table_id)},
    ).scalar()
    if owner_pid != str(project_id):
        raise HTTPException(404, "Table not found")
    rows = db.execute(
        sa_text(
            "SELECT id::text, row FROM samples "
            "WHERE table_id = :tid AND deleted_at IS NULL "
            "ORDER BY seq LIMIT :lim OFFSET :off"
        ),
        {"tid": str(table_id), "lim": limit, "off": offset},
    ).fetchall()
    total = db.execute(
        sa_text("SELECT COUNT(*) FROM samples WHERE table_id = :tid AND deleted_at IS NULL"),
        {"tid": str(table_id)},
    ).scalar() or 0
    filters = db.execute(
        sa_text("SELECT column_name, op, value FROM table_filters WHERE table_id = :tid"),
        {"tid": str(table_id)},
    ).fetchall()
    return {
        "rows": [{"id": r[0], **(r[1] or {})} for r in rows],
        "total": total,
        "limit": limit,
        "offset": offset,
        "filters": [{"column": f[0], "op": f[1], "value": f[2]} for f in filters],
    }


@router.get("/projects/{project_id}/tables/{table_id}/enrichments")
def list_enrichments(
    project_id: UUID,
    table_id: UUID,
    user: CurrentUser = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    _verify_project(project_id, user.user_id, db)
    rows = db.execute(
        sa_text(
            """
            SELECT id::text, name, columns, action, per_row_credit_cap,
                   last_run_filled_rows, last_run_cost_credits, last_run_at
            FROM enrichments
            WHERE table_id = :tid AND deleted_at IS NULL
            ORDER BY created_at
            """
        ),
        {"tid": str(table_id)},
    ).fetchall()
    return {
        "enrichments": [
            {
                "id": r[0],
                "name": r[1],
                "columns": r[2] if isinstance(r[2], list) else json.loads(r[2] or "[]"),
                "action": r[3] if isinstance(r[3], dict) else json.loads(r[3] or "{}"),
                "per_row_credit_cap": r[4],
                "last_run_filled_rows": r[5],
                "last_run_cost_credits": float(r[6]) if r[6] is not None else None,
                "last_run_at": r[7].isoformat() if r[7] else None,
            }
            for r in rows
        ]
    }
