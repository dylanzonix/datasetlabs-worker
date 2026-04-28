"""POST /v1/projects/{project_id}/chat/stream — SSE chat send-message.

Wire-compatible with the previous API endpoint of the same path so the
frontend can swap base URLs without changing payload/event shapes.
"""
from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse

from dsl_api.auth import CurrentUser, get_current_user
from dsl_api.schemas.chat import ChatMessageIn

from dsl_worker.chat_api.streaming import stream_chat_response

router = APIRouter()


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
