"""Chat API routes.

Two pipelines coexist during the FE rollover:

  Legacy (deprecate after FE cuts over):
    POST /v1/projects/{project_id}/chat/stream
        SSE generator — tightly coupled to the HTTP request.

  Run-aware (new):
    POST /v1/projects/{project_id}/chat/runs
        Creates a ChatRun, schedules background task, returns run_id.
    GET  /v1/projects/{project_id}/chat/runs/{run_id}/events?cursor=N
        SSE — replay persisted events from cursor, then live-tail.
        Reconnectable; not coupled to the agent run's lifetime.
    POST /v1/projects/{project_id}/chat/runs/{run_id}/pause
    POST /v1/projects/{project_id}/chat/runs/{run_id}/cancel
    POST /v1/projects/{project_id}/chat/runs/{run_id}/resume
    GET  /v1/projects/{project_id}/chat/active-run
        FE reattaches on mount via this lookup.
"""
from __future__ import annotations

import json
from typing import Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session

from dsl_api.auth import CurrentUser, get_current_user
from dsl_api.db import SessionLocal
from dsl_api.models import ChatRun, Project
from dsl_api.schemas.chat import ChatMessageIn

from dsl_worker.chat_api import runs
from dsl_worker.chat_api.streaming import stream_chat_response

router = APIRouter()


# ---- Helpers --------------------------------------------------------------
def _verify_project_access(
    project_id: UUID, user_id: UUID, db: Session
) -> Project:
    project = (
        db.query(Project)
        .filter(
            Project.id == project_id,
            Project.user_id == user_id,
            Project.deleted_at.is_(None),
        )
        .first()
    )
    if project is None:
        raise HTTPException(status_code=404, detail="Project not found")
    return project


def _verify_run_access(
    project_id: UUID, run_id: UUID, user_id: UUID, db: Session
) -> ChatRun:
    _verify_project_access(project_id, user_id, db)
    run = (
        db.query(ChatRun)
        .filter(ChatRun.id == run_id, ChatRun.project_id == project_id)
        .first()
    )
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found")
    return run


def _run_to_dict(run: ChatRun) -> dict:
    return {
        "id": str(run.id),
        "project_id": str(run.project_id),
        "version_id": str(run.version_id) if run.version_id else None,
        "triggering_message_id": (
            str(run.triggering_message_id) if run.triggering_message_id else None
        ),
        "assistant_message_id": (
            str(run.assistant_message_id) if run.assistant_message_id else None
        ),
        "status": run.status,
        "current_phase": run.current_phase,
        "error": run.error,
        "next_event_seq": run.next_event_seq,
        "created_at": run.created_at.isoformat() if run.created_at else None,
        "started_at": run.started_at.isoformat() if run.started_at else None,
        "paused_at": run.paused_at.isoformat() if run.paused_at else None,
        "resumed_at": run.resumed_at.isoformat() if run.resumed_at else None,
        "completed_at": run.completed_at.isoformat() if run.completed_at else None,
    }


# ---- Legacy SSE endpoint --------------------------------------------------
@router.post("/v1/projects/{project_id}/chat/stream")
async def stream_chat_message(
    project_id: UUID,
    payload: ChatMessageIn,
    request: Request,
    user: CurrentUser = Depends(get_current_user),
) -> StreamingResponse:
    return StreamingResponse(
        stream_chat_response(
            project_id,
            user.user_id,
            payload.content,
            request,
            effort=payload.effort,
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ---- New: create a run ----------------------------------------------------
@router.post("/v1/projects/{project_id}/chat/runs")
async def create_run(
    project_id: UUID,
    payload: ChatMessageIn,
    user: CurrentUser = Depends(get_current_user),
) -> JSONResponse:
    db = SessionLocal()
    try:
        _verify_project_access(project_id, user.user_id, db)
    finally:
        db.close()

    try:
        run = await runs.start_run(
            project_id=project_id,
            user_id=user.user_id,
            user_content=payload.content,
            effort=payload.effort,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return JSONResponse(_run_to_dict(run))


# ---- New: tail events (reconnectable SSE) ---------------------------------
@router.get("/v1/projects/{project_id}/chat/runs/{run_id}/events")
async def stream_run_events(
    project_id: UUID,
    run_id: UUID,
    request: Request,
    cursor: int = Query(0, ge=0),
    user: CurrentUser = Depends(get_current_user),
) -> StreamingResponse:
    db = SessionLocal()
    try:
        _verify_run_access(project_id, run_id, user.user_id, db)
    finally:
        db.close()

    async def _gen():
        async for event in runs.tail_events(
            run_id, cursor=cursor, is_disconnected=request.is_disconnected
        ):
            yield f"data: {json.dumps(event)}\n\n"

    return StreamingResponse(
        _gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ---- New: pause / cancel / resume control --------------------------------
class _ResumeBody(BaseModel):
    content: Optional[str] = None
    effort: Optional[str] = None


@router.post("/v1/projects/{project_id}/chat/runs/{run_id}/pause")
async def pause_run_endpoint(
    project_id: UUID,
    run_id: UUID,
    user: CurrentUser = Depends(get_current_user),
) -> JSONResponse:
    db = SessionLocal()
    try:
        _verify_run_access(project_id, run_id, user.user_id, db)
    finally:
        db.close()
    accepted = runs.request_pause(run_id)
    return JSONResponse({"accepted": accepted})


@router.post("/v1/projects/{project_id}/chat/runs/{run_id}/cancel")
async def cancel_run_endpoint(
    project_id: UUID,
    run_id: UUID,
    user: CurrentUser = Depends(get_current_user),
) -> JSONResponse:
    db = SessionLocal()
    try:
        _verify_run_access(project_id, run_id, user.user_id, db)
    finally:
        db.close()
    accepted = runs.request_cancel(run_id)
    return JSONResponse({"accepted": accepted})


@router.post("/v1/projects/{project_id}/chat/runs/{run_id}/resume")
async def resume_run_endpoint(
    project_id: UUID,
    run_id: UUID,
    body: _ResumeBody,
    user: CurrentUser = Depends(get_current_user),
) -> JSONResponse:
    """Resume a paused run by creating a NEW run with the given content
    (defaults to 'Continue.'). Claude Code style: history replay puts
    the model back where it left off."""
    db = SessionLocal()
    try:
        _verify_run_access(project_id, run_id, user.user_id, db)
    finally:
        db.close()

    new_run = await runs.resume_run(
        project_id=project_id,
        user_id=user.user_id,
        paused_run_id=run_id,
        content=body.content or "Continue.",
        effort=body.effort,
    )
    return JSONResponse(_run_to_dict(new_run))


# ---- New: active-run discovery (FE reattach on mount) --------------------
@router.get("/v1/projects/{project_id}/chat/active-run")
async def get_active_run_endpoint(
    project_id: UUID,
    user: CurrentUser = Depends(get_current_user),
) -> JSONResponse:
    db = SessionLocal()
    try:
        _verify_project_access(project_id, user.user_id, db)
        run = runs.get_active_run(db, project_id)
        if run is None:
            return JSONResponse({"run": None})
        return JSONResponse({"run": _run_to_dict(run)})
    finally:
        db.close()
