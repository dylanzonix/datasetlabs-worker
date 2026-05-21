#!/bin/bash
set -e

# Chat worker FastAPI entrypoint. Container Apps sets PORT.
PORT="${PORT:-8040}"
WORKERS="${UVICORN_WORKERS:-1}"

exec uvicorn dsl_worker.chat.app:app \
    --host 0.0.0.0 \
    --port "$PORT" \
    --workers "$WORKERS" \
    --proxy-headers \
    --forwarded-allow-ips='*'
