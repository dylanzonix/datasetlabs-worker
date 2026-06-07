"""Durable enrichment jobs — REST endpoints.

  POST  /v2/projects/{pid}/enrichments/{eid}/jobs   create job + tasks
  POST  /v2/projects/{pid}/jobs/{job_id}/cancel     stop the job
  GET   /v2/projects/{pid}/jobs                     list active jobs
  GET   /v2/jobs/{job_id}                           job status
  GET   /v2/jobs/{job_id}/events                    SSE stream

The coordinator (`enrichment_jobs.Coordinator`) handles the actual
work; these endpoints only manage state + push events.
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from typing import Any, Dict, List, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlalchemy import text as sa_text
from sqlalchemy.orm import Session

from dsl_api.auth import CurrentUser, get_current_user
from dsl_api.db import SessionLocal
from dsl_api.models import Project
from dsl_worker.chat.enrichment_jobs import (
    emit_event,
    get_coordinator,
    notify_new_work,
    subscribe,
    unsubscribe,
    task_id_short,
)

log = logging.getLogger(__name__)

router = APIRouter(prefix="/v2")


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def _verify_project(project_id: UUID, user_id: UUID, db: Session) -> Project:
    p = (
        db.query(Project)
        .filter(
            Project.id == project_id,
            Project.user_id == user_id,
            Project.deleted_at.is_(None),
        )
        .first()
    )
    if not p:
        raise HTTPException(404, "Project not found")
    return p


# ---------------------------------------------------------------------------
# Create job
# ---------------------------------------------------------------------------


class CreateJobBody(BaseModel):
    scope_type: str = "all_unfilled"   # all_unfilled | first_n | row_ids | filtered
    first_n: Optional[int] = None
    row_ids: Optional[List[str]] = None
    filters: Optional[List[Dict[str, Any]]] = None
    overwrite: bool = False
    # Column-scoped run: subset of the enrichment's column NAMES to fill.
    # None/empty = fill all columns (legacy behavior). Used for single-cell
    # retries so we re-research ONLY the missing column, not the whole group.
    columns: Optional[List[str]] = None


@router.post("/projects/{project_id}/enrichments/{enrichment_id}/jobs")
async def create_job(
    project_id: UUID,
    enrichment_id: str,
    body: CreateJobBody,
    user: CurrentUser = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Resolve the scope, insert one job + N task rows, wake the
    coordinator. Returns the job_id; the FE subscribes to the SSE
    stream to follow progress."""
    _verify_project(project_id, user.user_id, db)

    # Reuse the canonical resolvers from the chat-tool path so the new
    # endpoint sees the same enrichment + same scoped rows as the agent
    # would. No duplicated SQL.
    from dsl_worker.chat.tools import resolve_enrichment_id
    from dsl_worker.chat.enrichment import _resolve_scope_rows, _ensure_columns_on_table

    canonical_eid = resolve_enrichment_id(db, str(project_id), enrichment_id)
    if not canonical_eid:
        raise HTTPException(404, f"Enrichment {enrichment_id!r} not found")

    row = db.execute(
        sa_text(
            "SELECT table_id::text, columns FROM enrichments "
            "WHERE id=CAST(:eid AS uuid) AND deleted_at IS NULL"
        ),
        {"eid": canonical_eid},
    ).fetchone()
    if not row:
        raise HTTPException(404, f"Enrichment {enrichment_id!r} not found")
    table_id = row[0]
    columns = row[1] if isinstance(row[1], list) else json.loads(row[1] or "[]")

    # Make sure the enrichment's columns are attached to the table
    # before we resolve scope — otherwise the FE won't see the columns
    # appear once cells start filling.
    _ensure_columns_on_table(db, table_id, columns, enrichment_id=canonical_eid)
    db.commit()

    scope: Dict[str, Any] = {"type": body.scope_type, "overwrite": body.overwrite}
    if body.first_n is not None:
        scope["first_n"] = body.first_n
    if body.row_ids is not None:
        scope["row_ids"] = body.row_ids
    if body.filters is not None:
        scope["filters"] = body.filters
    # Validate the requested column subset against the enrichment's real
    # columns (ignore unknown names). Persist on the job so _execute_task
    # narrows columns_to_fill to exactly these — siblings are left untouched.
    if body.columns:
        _valid = {c["name"] for c in columns if isinstance(c, dict) and c.get("name")}
        _subset = [c for c in body.columns if c in _valid]
        if _subset:
            scope["columns"] = _subset

    sample_rows = _resolve_scope_rows(
        db, table_id, scope, columns, overwrite=body.overwrite,
    )
    if not sample_rows:
        return {
            "job_id": None,
            "total_tasks": 0,
            "message": "No rows to enrich (all targets already filled or scope empty)",
        }

    job_id = str(uuid.uuid4())

    db.execute(
        sa_text(
            "INSERT INTO enrichment_jobs "
            "(id, project_id, enrichment_id, user_id, status, scope, total_tasks) "
            "VALUES (CAST(:id AS uuid), CAST(:pid AS uuid), CAST(:eid AS uuid), "
            "CAST(:uid AS uuid), 'queued', CAST(:scope AS jsonb), :n)"
        ),
        {
            "id": job_id,
            "pid": str(project_id),
            "eid": canonical_eid,
            "uid": str(user.user_id),
            "scope": json.dumps(scope),
            "n": len(sample_rows),
        },
    )
    # Bulk insert tasks. Even at 1000 rows this is a single round trip.
    task_values: List[Dict[str, Any]] = []
    for sid, _, _ in sample_rows:
        task_values.append({
            "id": str(uuid.uuid4()),
            "jid": job_id,
            "pid": str(project_id),
            "eid": canonical_eid,
            "sid": str(sid),
        })
    db.execute(
        sa_text(
            "INSERT INTO enrichment_tasks "
            "(id, job_id, project_id, enrichment_id, sample_id, status) "
            "VALUES (CAST(:id AS uuid), CAST(:jid AS uuid), CAST(:pid AS uuid), "
            "CAST(:eid AS uuid), CAST(:sid AS uuid), 'queued')"
        ),
        task_values,
    )
    db.commit()

    # job_started + per-row queued events so the FE can paint Queued
    # badges immediately without waiting for the coordinator to pick
    # the first task up.
    emit_event(
        db, job_id, "job_started",
        {
            "enrichment_id": canonical_eid,
            "total_tasks": len(sample_rows),
            "row_ids": [str(sid) for sid, _, _ in sample_rows],
            "columns": scope.get("columns") or [c["name"] for c in columns],
        },
    )

    # NOTIFY the coordinator. If it's running in this process the
    # in-process wake also fires (the LISTEN loop catches the NOTIFY
    # too, but the direct wake skips the round-trip).
    notify_new_work(db)
    try:
        get_coordinator().wake()
    except Exception:
        pass

    return {
        "job_id": job_id,
        "total_tasks": len(sample_rows),
        "enrichment_id": canonical_eid,
    }


# ---------------------------------------------------------------------------
# Cancel job
# ---------------------------------------------------------------------------


@router.post("/projects/{project_id}/jobs/{job_id}/cancel")
async def cancel_job(
    project_id: UUID,
    job_id: UUID,
    user: CurrentUser = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    _verify_project(project_id, user.user_id, db)
    # Cancel: mark job + drop queued tasks. Running tasks finish (their
    # cost was already going to be paid). When the last running task
    # ends, the coordinator's _maybe_finalize_job will see done==total
    # and emit job_done; we add a job_cancelled event up front so the
    # FE has an immediate terminal signal.
    res = db.execute(
        sa_text(
            "UPDATE enrichment_jobs "
            "SET status='cancelled', ended_at=now() "
            "WHERE id=CAST(:jid AS uuid) AND project_id=CAST(:pid AS uuid) "
            "  AND status IN ('queued', 'running') "
            "RETURNING id"
        ),
        {"jid": str(job_id), "pid": str(project_id)},
    ).fetchone()
    if not res:
        # Already terminal; idempotent — nothing to do.
        return {"cancelled": False, "message": "Job not active"}
    db.execute(
        sa_text(
            "UPDATE enrichment_tasks SET status='skipped', error='job cancelled', "
            "ended_at=now() WHERE job_id=CAST(:jid AS uuid) AND status='queued'"
        ),
        {"jid": str(job_id)},
    )
    db.commit()
    emit_event(db, str(job_id), "job_cancelled", {})
    return {"cancelled": True}


# ---------------------------------------------------------------------------
# Status / discovery — used by FE on mount + as a polling fallback
# ---------------------------------------------------------------------------


@router.get("/projects/{project_id}/jobs")
def list_active_jobs(
    project_id: UUID,
    user: CurrentUser = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    _verify_project(project_id, user.user_id, db)
    rows = db.execute(
        sa_text(
            "SELECT id::text, enrichment_id::text, status, total_tasks, "
            "done_tasks, failed_tasks, created_at "
            "FROM enrichment_jobs "
            "WHERE project_id=CAST(:pid AS uuid) AND status IN ('queued', 'running') "
            "ORDER BY created_at DESC"
        ),
        {"pid": str(project_id)},
    ).fetchall()
    return {
        "jobs": [
            {
                "id": r[0],
                "enrichment_id": r[1],
                "status": r[2],
                "total_tasks": r[3],
                "done_tasks": r[4],
                "failed_tasks": r[5],
                "created_at": r[6].isoformat() if r[6] else None,
            }
            for r in rows
        ],
    }


@router.get("/jobs/{job_id}")
def get_job(
    job_id: UUID,
    user: CurrentUser = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    row = db.execute(
        sa_text(
            "SELECT j.id::text, j.project_id::text, j.enrichment_id::text, "
            "j.status, j.total_tasks, j.done_tasks, j.failed_tasks, "
            "j.cost_usd, j.error, j.created_at, j.started_at, j.ended_at "
            "FROM enrichment_jobs j "
            "JOIN projects p ON p.id = j.project_id "
            "WHERE j.id=CAST(:jid AS uuid) AND p.user_id=CAST(:uid AS uuid)"
        ),
        {"jid": str(job_id), "uid": str(user.user_id)},
    ).fetchone()
    if not row:
        raise HTTPException(404, "Job not found")
    return {
        "id": row[0],
        "project_id": row[1],
        "enrichment_id": row[2],
        "status": row[3],
        "total_tasks": row[4],
        "done_tasks": row[5],
        "failed_tasks": row[6],
        "cost_usd": float(row[7] or 0),
        "error": row[8],
        "created_at": row[9].isoformat() if row[9] else None,
        "started_at": row[10].isoformat() if row[10] else None,
        "ended_at": row[11].isoformat() if row[11] else None,
    }


# ---------------------------------------------------------------------------
# SSE stream — replay-from-cursor + tail
# ---------------------------------------------------------------------------


@router.get("/jobs/{job_id}/events")
async def stream_job_events(
    job_id: UUID,
    request: Request,
    since: int = 0,
    user: CurrentUser = Depends(get_current_user),
):
    """SSE stream of enrichment_event rows for one job.

    Pass `since` = last seen event id to resume. The endpoint replays
    every event with id > since from the DB, then tails live events
    via the in-process pubsub queue. Closing the connection (browser
    refresh / nav) is harmless: the coordinator keeps writing events,
    and the next subscribe replays from the new cursor.
    """
    # Permission check + early validation in a short DB call.
    db = SessionLocal()
    try:
        row = db.execute(
            sa_text(
                "SELECT j.id FROM enrichment_jobs j "
                "JOIN projects p ON p.id = j.project_id "
                "WHERE j.id=CAST(:jid AS uuid) AND p.user_id=CAST(:uid AS uuid)"
            ),
            {"jid": str(job_id), "uid": str(user.user_id)},
        ).fetchone()
    finally:
        db.close()
    if not row:
        raise HTTPException(404, "Job not found")

    job_id_str = str(job_id)
    q = subscribe(job_id_str)

    async def gen():
        try:
            cursor = since
            # Replay from DB.
            db2 = SessionLocal()
            try:
                while True:
                    rows = db2.execute(
                        sa_text(
                            "SELECT id, kind, payload FROM enrichment_events "
                            "WHERE job_id=CAST(:jid AS uuid) AND id > :since "
                            "ORDER BY id LIMIT 500"
                        ),
                        {"jid": job_id_str, "since": cursor},
                    ).fetchall()
                    if not rows:
                        break
                    for r in rows:
                        payload = r[2] if isinstance(r[2], dict) else json.loads(r[2] or "{}")
                        yield _sse({"id": int(r[0]), "kind": r[1], "payload": payload})
                        cursor = int(r[0])
                    if len(rows) < 500:
                        break
            finally:
                db2.close()

            # Tail live events. Time out periodically with a heartbeat
            # so dead connections get noticed.
            while True:
                if await request.is_disconnected():
                    break
                try:
                    event = await asyncio.wait_for(q.get(), timeout=20)
                except asyncio.TimeoutError:
                    yield ": heartbeat\n\n"
                    continue
                if int(event["id"]) <= cursor:
                    continue
                cursor = int(event["id"])
                yield _sse(event)
                if event["kind"] in ("job_done", "job_failed", "job_cancelled"):
                    break
        finally:
            unsubscribe(job_id_str, q)

    headers = {
        "Cache-Control": "no-cache, no-transform",
        "X-Accel-Buffering": "no",
        "Connection": "keep-alive",
    }
    return StreamingResponse(gen(), media_type="text/event-stream", headers=headers)


def _sse(event: dict) -> str:
    return f"id: {event['id']}\nevent: {event['kind']}\ndata: {json.dumps(event['payload'])}\n\n"
