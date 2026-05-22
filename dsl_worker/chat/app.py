"""Chat worker FastAPI app.

Hosts the chat HTTP/SSE endpoints. Imports models, auth, and the credits
ledger from dsl_api so it talks to the same Postgres tables the main API
and the V13 worker already use.

Run locally: `uvicorn dsl_worker.chat.app:app --port 8040 --reload`
"""
from __future__ import annotations

import logging
import os
import warnings

from dotenv import load_dotenv

# override=True so an empty/stale shell env var (e.g. APIFY_API_KEY="" from
# a prior session) doesn't shadow the value in .env. Without this, every
# adapter that reads its key via os.getenv at import sees "" and goes inert.
load_dotenv(".env", override=True)

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

from dsl_worker.chat import routes_health, run_state, tracing
from dsl_worker.chat.routes import router as chat_router
from dsl_worker.chat.routes_actions import router as chat_actions_router
from dsl_worker.chat.routes_table_edit import router as chat_table_edit_router
from dsl_worker.chat.routes_tablepage import router as chat_tablepage_router

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# Bump the default thread executor — Python's default is min(32, cpu+4),
# which on a typical 4-core box gives ~8 threads. Several hot paths use
# asyncio.to_thread for blocking psycopg2 calls (heartbeat, future
# emit_event wrapper). Under 6+ concurrent projects firing events at
# ~5-10/sec, the default executor saturates and to_thread calls queue
# up — that's what feels like "the server is laggy." Bump to 64.
import concurrent.futures as _cf
import asyncio as _asyncio
def _install_bigger_executor() -> None:
    try:
        loop = _asyncio.get_event_loop()
        loop.set_default_executor(_cf.ThreadPoolExecutor(max_workers=64, thread_name_prefix="dsl-worker"))
    except Exception:
        pass
_install_bigger_executor()

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
    allow_methods=["GET", "POST", "PATCH", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type"],
    allow_credentials=False,
    expose_headers=["X-Accel-Buffering"],
)

app.include_router(routes_health.router)
app.include_router(chat_router)
app.include_router(chat_actions_router)
app.include_router(chat_table_edit_router)
app.include_router(chat_tablepage_router)


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
        n = run_state.recover_orphan_runs()
        if n:
            log.warning("recovered %d orphan chat run(s)", n)
    except Exception:
        log.exception("orphan-run recovery failed")
    # 30-day TTL on chat_run_events. Hourly pass; first pass runs after
    # one interval, so startup isn't slowed by a large initial DELETE.
    import asyncio
    asyncio.create_task(run_state.run_ttl_cleanup_loop(), name="chat-events-ttl")
    # Continuous orphan reaper: marks runs failed within ~30s of their
    # heartbeat going stale, instead of waiting for the next worker
    # restart (could be hours).
    asyncio.create_task(run_state.orphan_recovery_loop(), name="chat-orphan-reaper")


@app.on_event("shutdown")
async def _on_shutdown() -> None:
    tracing.flush()
