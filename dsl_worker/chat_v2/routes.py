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
import logging
from typing import Any, Dict, List, Optional
from uuid import UUID

log = logging.getLogger(__name__)

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlalchemy import text as sa_text
from sqlalchemy.orm import Session

from dsl_api.auth import CurrentUser, get_current_user
from dsl_api.db import SessionLocal
from dsl_api.models import Project
from dsl_worker.chat_v2.agent import run_turn, stream_turn


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


@router.post("/projects/{project_id}/chat/turns/stream")
async def post_turn_stream(
    project_id: UUID,
    body: TurnRequest,
    user: CurrentUser = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Run a turn and stream events as SSE.

    Frames are `event: <type>\\ndata: <json>\\n\\n`. The FE renders these
    as live tool-log entries; on `final_message` it commits the assistant
    reply; on `turn_complete` it refreshes the table list. User + assistant
    messages are persisted to chat_messages so refreshing the page restores
    the conversation.
    """
    _verify_project(project_id, user.user_id, db)
    pid = str(project_id)

    # Persist the user message immediately so a mid-turn disconnect
    # still leaves it in history.
    db.execute(
        sa_text(
            "INSERT INTO chat_messages (id, project_id, role, content, created_at) "
            "VALUES (gen_random_uuid(), :pid, 'user', :content, now())"
        ),
        {"pid": pid, "content": body.message},
    )
    db.commit()

    async def event_stream():
        final_text = ""
        tool_log: List[Dict[str, Any]] = []
        total_cost = 0.0
        iterations = 0
        try:
            async for evt in stream_turn(
                db=db,
                project_id=pid,
                user_id=str(user.user_id),
                run_id=None,
                user_message=body.message,
                history=body.history,
            ):
                etype = evt.get("type", "message")
                if etype == "tool_call_start":
                    tool_log.append({
                        "id": evt.get("tool_call_id"),
                        "name": evt.get("name"),
                        "args_preview": json.dumps(evt.get("args") or {}, default=str)[:200],
                    })
                elif etype == "tool_call_result":
                    for t in tool_log:
                        if t.get("id") == evt.get("tool_call_id"):
                            t["summary"] = evt.get("result_preview")
                            t["cost"] = evt.get("cost_usd")
                            break
                elif etype == "final_message":
                    final_text = evt.get("text") or ""
                elif etype == "turn_complete":
                    total_cost = evt.get("total_cost_usd") or 0.0
                    iterations = evt.get("iterations") or 0
                payload = json.dumps(evt, default=str)
                yield f"event: {etype}\ndata: {payload}\n\n"
        except Exception as e:
            yield f"event: error\ndata: {json.dumps({'message': str(e)})}\n\n"

        # Persist the assistant message + tool trace once the turn finishes,
        # and bump the project's cumulative spend so the header chip + usage
        # views reflect what this turn cost.
        try:
            db.execute(
                sa_text(
                    "INSERT INTO chat_messages (id, project_id, role, content, applied_changes, created_at) "
                    "VALUES (gen_random_uuid(), :pid, 'assistant', :content, CAST(:ac AS jsonb), now())"
                ),
                {
                    "pid": pid,
                    "content": final_text,
                    "ac": json.dumps({
                        "tool_log": tool_log,
                        "total_cost_usd": total_cost,
                        "iterations": iterations,
                    }, default=str),
                },
            )
            # cumulative_spend_cents tracks dollars * 100. total_cost is USD;
            # round to nearest cent so a $0.30 turn shows as 30 credits.
            spend_cents = max(0, int(round(float(total_cost) * 100)))
            if spend_cents > 0:
                db.execute(
                    sa_text(
                        "UPDATE projects "
                        "SET cumulative_spend_cents = COALESCE(cumulative_spend_cents, 0) + :c "
                        "WHERE id = :pid"
                    ),
                    {"pid": pid, "c": spend_cents},
                )
            db.commit()
        except Exception:
            log.exception("failed to persist assistant message / spend")

        yield "event: done\ndata: {}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


def _resolve_table_uuid(db: Session, project_id: str, id_or_short: str) -> Optional[str]:
    """Accept either short_id (t1) or UUID at route boundaries; return UUID."""
    if not id_or_short:
        return None
    if "-" in id_or_short:
        row = db.execute(
            sa_text(
                "SELECT id::text FROM tables WHERE id=:x AND project_id=:pid AND deleted_at IS NULL"
            ),
            {"x": id_or_short, "pid": project_id},
        ).fetchone()
    else:
        row = db.execute(
            sa_text(
                "SELECT id::text FROM tables WHERE short_id=:x AND project_id=:pid AND deleted_at IS NULL"
            ),
            {"x": id_or_short, "pid": project_id},
        ).fetchone()
    return row[0] if row else None


@router.get("/projects/{project_id}/tables")
def list_tables(
    project_id: UUID,
    user: CurrentUser = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """List tables in a project (multi-table workspace).

    Response `id` is the short_id (t1, t2, ...) — the FE uses it as the
    table handle in URLs; routes accept either short_id or UUID.
    """
    _verify_project(project_id, user.user_id, db)
    rows = db.execute(
        sa_text(
            """
            SELECT id::text, short_id, name, source, columns, dedup_key_column,
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
        cols = r[4] if isinstance(r[4], list) else json.loads(r[4] or "[]")
        out.append({
            "id": r[1],  # short_id is the primary public id
            "uuid": r[0],  # UUID still available for backend-internal callers
            "name": r[2],
            "source": r[3],
            "columns": cols,
            "dedup_key_column": r[5],
            "row_count": row_count,
            "last_fetch_returned_rows": r[6],
            "last_fetch_cost_credits": float(r[7]) if r[7] is not None else None,
            "last_fetch_at": r[8].isoformat() if r[8] else None,
            "fetch_status": r[9],
            "fetch_error": r[10],
            "created_at": r[11].isoformat() if r[11] else None,
        })
    return {"tables": out}


@router.get("/projects/{project_id}/tables/{table_id}/rows")
def list_table_rows(
    project_id: UUID,
    table_id: str,
    limit: int = 200,
    offset: int = 0,
    user: CurrentUser = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Read rows of a table. Accepts short_id or UUID. Includes filters."""
    _verify_project(project_id, user.user_id, db)
    tid = _resolve_table_uuid(db, str(project_id), table_id)
    if not tid:
        raise HTTPException(404, "Table not found")
    rows = db.execute(
        sa_text(
            "SELECT id::text, row FROM samples "
            "WHERE table_id = :tid AND deleted_at IS NULL "
            "ORDER BY seq LIMIT :lim OFFSET :off"
        ),
        {"tid": tid, "lim": limit, "off": offset},
    ).fetchall()
    total = db.execute(
        sa_text("SELECT COUNT(*) FROM samples WHERE table_id = :tid AND deleted_at IS NULL"),
        {"tid": tid},
    ).scalar() or 0
    filters = db.execute(
        sa_text("SELECT column_name, op, value FROM table_filters WHERE table_id = :tid"),
        {"tid": tid},
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
    table_id: str,
    user: CurrentUser = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    _verify_project(project_id, user.user_id, db)
    tid = _resolve_table_uuid(db, str(project_id), table_id)
    if not tid:
        raise HTTPException(404, "Table not found")
    rows = db.execute(
        sa_text(
            """
            SELECT short_id, name, columns, action, per_row_credit_cap,
                   last_run_filled_rows, last_run_cost_credits, last_run_at
            FROM enrichments
            WHERE table_id = :tid AND deleted_at IS NULL
            ORDER BY created_at
            """
        ),
        {"tid": tid},
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
