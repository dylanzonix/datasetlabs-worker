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

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
from sqlalchemy import text as sa_text
from sqlalchemy.orm import Session

from dsl_api.auth import CurrentUser, get_current_user
from dsl_api.db import SessionLocal
from dsl_api.models import ChatRun, Project
from dsl_worker.chat_api import runs as legacy_runs
from dsl_worker.chat_v2 import runs as v2_runs
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


class StartRunBody(BaseModel):
    content: str


def _run_to_dict(run: ChatRun) -> Dict[str, Any]:
    return {
        "id": str(run.id),
        "project_id": str(run.project_id),
        "status": run.status,
        "current_phase": run.current_phase,
        "triggering_message_id": str(run.triggering_message_id) if run.triggering_message_id else None,
        "assistant_message_id": str(run.assistant_message_id) if run.assistant_message_id else None,
        "error": run.error,
        "next_event_seq": run.next_event_seq,
        "created_at": run.created_at.isoformat() if run.created_at else None,
        "started_at": run.started_at.isoformat() if run.started_at else None,
        "completed_at": run.completed_at.isoformat() if run.completed_at else None,
    }


@router.post("/projects/{project_id}/chat/runs")
async def create_v2_run(
    project_id: UUID,
    body: StartRunBody,
    user: CurrentUser = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> JSONResponse:
    """Start a durable chat_v2 run. The agent executes as a background
    asyncio task; the SSE tail at .../events streams its events. Survives
    client disconnect — refresh + reattach is supported."""
    _verify_project(project_id, user.user_id, db)
    try:
        run = await v2_runs.start_v2_run(
            project_id=project_id,
            user_id=user.user_id,
            user_content=body.content,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return JSONResponse(_run_to_dict(run))


@router.get("/projects/{project_id}/chat/runs/{run_id}/events")
async def stream_v2_run_events(
    project_id: UUID,
    run_id: UUID,
    request: Request,
    cursor: int = Query(0, ge=0),
    user: CurrentUser = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> StreamingResponse:
    """SSE tail of chat_run_events for a chat_v2 run. Supports cursor
    replay so a reconnecting client picks up exactly where it left off.
    Reuses legacy tail_events — chat_v2 writes the same event shape."""
    _verify_project(project_id, user.user_id, db)
    # Verify run belongs to project.
    run = db.query(ChatRun).filter(ChatRun.id == run_id, ChatRun.project_id == project_id).first()
    if run is None:
        raise HTTPException(404, "Run not found")

    async def _gen():
        async for event in legacy_runs.tail_events(
            run_id, cursor=cursor, is_disconnected=request.is_disconnected
        ):
            yield f"data: {json.dumps(event, default=str)}\n\n"

    return StreamingResponse(
        _gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/projects/{project_id}/chat/active-run")
async def get_v2_active_run(
    project_id: UUID,
    user: CurrentUser = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> JSONResponse:
    """If there's an in-flight chat_v2 run on this project, return it so
    the FE can reattach after a refresh."""
    _verify_project(project_id, user.user_id, db)
    run = legacy_runs.get_active_run(db, project_id)
    if run is None:
        return JSONResponse({"run": None})
    return JSONResponse({"run": _run_to_dict(run)})


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
                   fetch_status, fetch_error, created_at, sort_column, sort_direction
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
        sort_obj = {"column": r[12], "direction": r[13] or "desc"} if r[12] else None
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
            "sort": sort_obj,
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
    """Read rows of a table. Accepts short_id or UUID. Applies stored filters
    + sort. Returns the matched slice + total (the filtered total)."""
    _verify_project(project_id, user.user_id, db)
    tid = _resolve_table_uuid(db, str(project_id), table_id)
    if not tid:
        raise HTTPException(404, "Table not found")
    # Read sort + filters
    tinfo = db.execute(
        sa_text("SELECT sort_column, sort_direction FROM tables WHERE id=:tid"),
        {"tid": tid},
    ).fetchone()
    sort_col = tinfo[0] if tinfo else None
    sort_dir = (tinfo[1] if tinfo else None) or "desc"
    filters = db.execute(
        sa_text("SELECT column_name, op, value FROM table_filters WHERE table_id = :tid"),
        {"tid": tid},
    ).fetchall()

    # Build WHERE fragments from active filters. Each filter writes its
    # JSONB-key predicate + adds bound parameters.
    filter_fragments, filter_params = _filters_to_where_sql(filters)
    where_extra = (" AND " + " AND ".join(filter_fragments)) if filter_fragments else ""

    if sort_col:
        # Try numeric sort first via NULLIF + cast; fall back to text. Sorting
        # missing values last regardless of direction.
        sort_col_esc = sort_col.replace("'", "''")
        direction_sql = "ASC" if sort_dir.lower() == "asc" else "DESC"
        order_clause = (
            f"(row->>'{sort_col_esc}' IS NULL) ASC, "
            f"NULLIF(row->>'{sort_col_esc}', '')::numeric {direction_sql} NULLS LAST, "
            f"row->>'{sort_col_esc}' {direction_sql} NULLS LAST, "
            f"seq"
        )
        sql_text = (
            f"SELECT id::text, row FROM samples "
            f"WHERE table_id = :tid AND deleted_at IS NULL{where_extra} "
            f"ORDER BY {order_clause} LIMIT :lim OFFSET :off"
        )
    else:
        sql_text = (
            f"SELECT id::text, row FROM samples "
            f"WHERE table_id = :tid AND deleted_at IS NULL{where_extra} "
            f"ORDER BY seq LIMIT :lim OFFSET :off"
        )
    base_params = {"tid": tid, "lim": limit, "off": offset, **filter_params}
    # Try the typed sort. If the sort column has any non-numeric value the
    # ::numeric cast errors — fall back to plain text sort in that case.
    try:
        rows = db.execute(sa_text(sql_text), base_params).fetchall()
    except Exception:
        if sort_col:
            db.rollback()
            sort_col_esc = sort_col.replace("'", "''")
            direction_sql = "ASC" if sort_dir.lower() == "asc" else "DESC"
            fallback_sql = (
                f"SELECT id::text, row FROM samples "
                f"WHERE table_id = :tid AND deleted_at IS NULL{where_extra} "
                f"ORDER BY (row->>'{sort_col_esc}' IS NULL) ASC, "
                f"row->>'{sort_col_esc}' {direction_sql} NULLS LAST, seq "
                f"LIMIT :lim OFFSET :off"
            )
            rows = db.execute(sa_text(fallback_sql), base_params).fetchall()
        else:
            raise
    # Total = filtered count (so the FE pagination matches what user sees).
    # Unfiltered count exposed separately for context.
    total_sql = (
        f"SELECT COUNT(*) FROM samples "
        f"WHERE table_id = :tid AND deleted_at IS NULL{where_extra}"
    )
    total = db.execute(sa_text(total_sql), {"tid": tid, **filter_params}).scalar() or 0
    unfiltered_total = db.execute(
        sa_text("SELECT COUNT(*) FROM samples WHERE table_id = :tid AND deleted_at IS NULL"),
        {"tid": tid},
    ).scalar() or 0
    return {
        "rows": [{"id": r[0], **(r[1] or {})} for r in rows],
        "total": total,
        "unfiltered_total": unfiltered_total,
        "limit": limit,
        "offset": offset,
        "filters": [{"column": f[0], "op": f[1], "value": f[2]} for f in filters],
        "sort": {"column": sort_col, "direction": sort_dir} if sort_col else None,
    }


def _filters_to_where_sql(filters):
    """Translate stored (column_name, op, value) tuples into SQL fragments
    against the samples.row JSONB column. Returns (list[str], dict).

    Values are always bound via :param; column names are escaped against
    single quotes (JSONB keys are arbitrary strings).
    """
    import re as _re
    fragments = []
    params = {}
    for i, (col, op, val) in enumerate(filters):
        if not col:
            continue
        col_esc = col.replace("'", "''")
        json_text = f"row->>'{col_esc}'"
        p = f"f{i}"
        op_lower = (op or "").lower()
        if op_lower in ("equals", "eq", "="):
            fragments.append(f"{json_text} = :{p}")
            params[p] = str(val) if val is not None else None
        elif op_lower in ("not_equals", "neq", "!="):
            fragments.append(f"({json_text} IS DISTINCT FROM :{p})")
            params[p] = str(val) if val is not None else None
        elif op_lower == "contains":
            fragments.append(f"{json_text} ILIKE :{p}")
            params[p] = f"%{val}%"
        elif op_lower in ("is_any_of", "in", "any_of"):
            arr = val if isinstance(val, list) else [val]
            arr = [str(v) for v in arr]
            fragments.append(f"{json_text} = ANY(:{p})")
            params[p] = arr
        elif op_lower in ("not_in", "is_none_of"):
            arr = val if isinstance(val, list) else [val]
            arr = [str(v) for v in arr]
            fragments.append(f"({json_text} IS NULL OR NOT ({json_text} = ANY(:{p})))")
            params[p] = arr
        elif op_lower in (">=", "gte"):
            try:
                fragments.append(f"NULLIF({json_text},'')::numeric >= :{p}")
                params[p] = float(val)
            except (TypeError, ValueError):
                # Fall back to lexical (works for ISO dates too)
                fragments.append(f"{json_text} >= :{p}")
                params[p] = str(val)
        elif op_lower in ("<=", "lte"):
            try:
                fragments.append(f"NULLIF({json_text},'')::numeric <= :{p}")
                params[p] = float(val)
            except (TypeError, ValueError):
                fragments.append(f"{json_text} <= :{p}")
                params[p] = str(val)
        elif op_lower in (">", "gt"):
            try:
                fragments.append(f"NULLIF({json_text},'')::numeric > :{p}")
                params[p] = float(val)
            except (TypeError, ValueError):
                fragments.append(f"{json_text} > :{p}")
                params[p] = str(val)
        elif op_lower in ("<", "lt"):
            try:
                fragments.append(f"NULLIF({json_text},'')::numeric < :{p}")
                params[p] = float(val)
            except (TypeError, ValueError):
                fragments.append(f"{json_text} < :{p}")
                params[p] = str(val)
        elif op_lower == "between" and isinstance(val, list) and len(val) == 2:
            lo, hi = val
            # If both look numeric → numeric range. Otherwise lexical (handles
            # ISO date strings naturally).
            try:
                lof = float(lo) if lo is not None else None
                hif = float(hi) if hi is not None else None
                p_lo, p_hi = f"{p}_lo", f"{p}_hi"
                if lof is not None and hif is not None:
                    fragments.append(
                        f"NULLIF({json_text},'')::numeric BETWEEN :{p_lo} AND :{p_hi}"
                    )
                    params[p_lo] = lof
                    params[p_hi] = hif
                elif lof is not None:
                    fragments.append(f"NULLIF({json_text},'')::numeric >= :{p_lo}")
                    params[p_lo] = lof
                elif hif is not None:
                    fragments.append(f"NULLIF({json_text},'')::numeric <= :{p_hi}")
                    params[p_hi] = hif
            except (TypeError, ValueError):
                p_lo, p_hi = f"{p}_lo", f"{p}_hi"
                if lo is not None and hi is not None:
                    fragments.append(f"{json_text} BETWEEN :{p_lo} AND :{p_hi}")
                    params[p_lo] = str(lo)
                    params[p_hi] = str(hi)
                elif lo is not None:
                    fragments.append(f"{json_text} >= :{p_lo}")
                    params[p_lo] = str(lo)
                elif hi is not None:
                    fragments.append(f"{json_text} <= :{p_hi}")
                    params[p_hi] = str(hi)
        elif op_lower in ("is_null", "empty"):
            fragments.append(f"({json_text} IS NULL OR {json_text} = '')")
        elif op_lower in ("is_not_null", "not_empty"):
            fragments.append(f"({json_text} IS NOT NULL AND {json_text} != '')")
        else:
            # Unknown op — skip rather than fail the whole query
            continue
    return fragments, params


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
