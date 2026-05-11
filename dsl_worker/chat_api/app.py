"""Chat worker FastAPI app.

Hosts the chat send-message SSE endpoint. Imports models, auth, and the
credits ledger from dsl_api so it talks to the same Postgres tables the
main API and the V13 worker already use.

Run locally: `uvicorn dsl_worker.chat_api.app:app --port 8040 --reload`
"""
from __future__ import annotations

import logging
import os
import warnings

from dotenv import load_dotenv

# Load .env before importing anything that reads settings (dsl_api.config).
load_dotenv(".env")

# Silence noisy Pydantic discriminated-union warnings emitted by the OpenAI
# SDK when serializing Response objects that contain `web_search_call`
# items — the SDK's union variants don't always match the live response
# shape (e.g. action `find_in_page` arriving as `ActionSearch`-shaped
# data), so pydantic warns once per non-matching variant on every response.
# Suppressing keeps the request log readable; the responses themselves work.
warnings.filterwarnings(
    "ignore",
    message=r"^Pydantic serializer warnings:",
    category=UserWarning,
)

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from dsl_worker.chat_api import routes_chat, routes_health, runs, tracing

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# Silence noisy third-party SDK INFO logs that flood the worker terminal:
#   • azure.core http policy dumps full request/response headers per blob op
#     (artifacts module reads/writes a LOT of blobs)
#   • openai._base_client logs every retry with timing
#   • httpx logs every outbound request as "HTTP/1.1 200 OK"
# WARN/ERROR still surface — these only mute the chatty INFO traffic.
for noisy in (
    "azure.core.pipeline.policies.http_logging_policy",
    "azure.identity",
    "azure.storage",
    "openai._base_client",
    "httpx",
    "httpcore",
):
    logging.getLogger(noisy).setLevel(logging.WARNING)


def _allowed_origins() -> list[str]:
    raw = os.getenv(
        "CHAT_API_ALLOWED_ORIGINS",
        "http://localhost:8080,http://localhost:5173,http://localhost:3000",
    )
    return [o.strip() for o in raw.split(",") if o.strip()]


app = FastAPI(title="DatasetLabs Chat Worker", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins(),
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type"],
    allow_credentials=False,
    expose_headers=["X-Accel-Buffering"],
)

app.include_router(routes_health.router)
app.include_router(routes_chat.router)


@app.on_event("startup")
async def _on_startup() -> None:
    log.info("chat worker api starting; CORS allowed origins=%s", _allowed_origins())
    log.info(
        "langfuse tracing %s",
        "ENABLED" if tracing.is_enabled() else "disabled (no LANGFUSE_SECRET_KEY)",
    )
    # Worker process restart leaves any in-flight ChatRun rows orphaned
    # (the asyncio.Task that owned them is dead). Mark them failed so
    # subscribers see a terminal event and the FE cleans up its UI.
    try:
        n = runs.recover_orphan_runs()
        if n:
            log.warning("recovered %d orphan chat run(s)", n)
    except Exception:
        log.exception("orphan-run recovery failed")
    # 30-day TTL on chat_run_events. Hourly pass; first pass runs after
    # one interval, so startup isn't slowed by a large initial DELETE.
    import asyncio
    asyncio.create_task(runs.run_ttl_cleanup_loop(), name="chat-events-ttl")
    # Continuous orphan reaper: marks runs failed within ~30s of their
    # heartbeat going stale, instead of waiting for the next worker
    # restart (could be hours).
    asyncio.create_task(runs.orphan_recovery_loop(), name="chat-orphan-reaper")


@app.on_event("shutdown")
async def _on_shutdown() -> None:
    tracing.flush()
