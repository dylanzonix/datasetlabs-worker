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

from dsl_worker.chat_api import routes_chat, routes_health, tracing

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


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


@app.on_event("shutdown")
async def _on_shutdown() -> None:
    tracing.flush()
