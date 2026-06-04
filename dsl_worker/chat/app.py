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
from dsl_worker.chat.routes_enrichment_jobs import router as enrichment_jobs_router
from dsl_worker.chat.enrichment_jobs import get_coordinator

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
import time as _time
def _install_bigger_executor() -> None:
    try:
        loop = _asyncio.get_event_loop()
        loop.set_default_executor(_cf.ThreadPoolExecutor(max_workers=64, thread_name_prefix="dsl-worker"))
    except Exception:
        pass
_install_bigger_executor()


async def _loop_lag_watchdog(interval: float = 0.5, warn_threshold: float = 0.75) -> None:
    """Logging-only background task: measure event-loop scheduling lag.

    Sleeps `interval`; if the actual elapsed overshoots by more than
    `warn_threshold`, the loop was STARVED — a coroutine hogged it, or the
    thread pool saturated so to_thread callbacks backed up. This is the
    signal behind "a 3s LLM call wall-clocked to 158s": the stream-drain
    coroutine couldn't get scheduled. Logs the lag, the default-executor
    queue depth, and the live task count so a stall is self-diagnosing in
    the file log. Never raises; never touches request handling.
    """
    loop = _asyncio.get_running_loop()
    while True:
        t0 = _time.perf_counter()
        try:
            await _asyncio.sleep(interval)
        except _asyncio.CancelledError:
            return
        lag = _time.perf_counter() - t0 - interval
        if lag < warn_threshold:
            continue
        qdepth: object = "?"
        ntasks: object = "?"
        try:
            ex = getattr(loop, "_default_executor", None)
            wq = getattr(ex, "_work_queue", None)
            if wq is not None:
                qdepth = wq.qsize()
        except Exception:
            pass
        try:
            ntasks = len(_asyncio.all_tasks(loop))
        except Exception:
            pass
        log.warning(
            "[loop-lag] event loop stalled %.2fs — executor_queue=%s pending_tasks=%s "
            "(concurrent chat streams starve here; correlate with nearby [chat timing] ... SLOW lines)",
            lag, qdepth, ntasks,
        )

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

# Persist worker logs to a rotating file so post-hoc diagnosis (slow turns,
# loop stalls) doesn't depend on having caught the live terminal output.
# Everything that goes to stdout (basicConfig) also lands here — the
# [loop-lag] watchdog warnings and [chat timing] ... SLOW lines are the
# ones worth keeping. Override the path with CHAT_WORKER_LOG_FILE.
try:
    from logging.handlers import RotatingFileHandler as _RotatingFileHandler
    _worker_log_path = os.getenv("CHAT_WORKER_LOG_FILE", "logs/chat_worker.log")
    _log_dir = os.path.dirname(_worker_log_path)
    if _log_dir:
        os.makedirs(_log_dir, exist_ok=True)
    _file_handler = _RotatingFileHandler(_worker_log_path, maxBytes=50_000_000, backupCount=5)
    _file_handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    )
    logging.getLogger().addHandler(_file_handler)
    log.info("worker file logging enabled → %s", _worker_log_path)
except Exception:
    log.exception("file logging setup failed; continuing with stdout only")


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
app.include_router(enrichment_jobs_router)


@app.on_event("startup")
async def _on_startup() -> None:
    log.info("chat worker api starting; CORS allowed origins=%s", _allowed_origins())
    # Register the running loop with the run bus so event persistence can
    # run in asyncio.to_thread (off the loop) while live SSE fanout is
    # marshaled safely back onto the loop.
    import asyncio as _asyncio
    _loop = _asyncio.get_running_loop()
    run_state.set_event_loop(_loop)
    # Same for the durable-job event bus — the coordinator persists events
    # in to_thread, so its _publish must marshal fanout back onto the loop.
    from dsl_worker.chat import enrichment_jobs as _ej
    _ej.set_event_loop(_loop)
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
    # Event-loop starvation detector. Logging-only — fires [loop-lag]
    # warnings into the file log when the loop stalls (the root cause of
    # chat LLM calls ballooning from ~3s to ~150s under concurrent load).
    asyncio.create_task(_loop_lag_watchdog(), name="loop-lag-watchdog")
    # Durable enrichment job coordinator: claims queued tasks (FOR
    # UPDATE SKIP LOCKED), runs under Semaphore(25), publishes events.
    # Browser refresh / network drops survive because state lives in PG.
    await get_coordinator().start()


@app.on_event("shutdown")
async def _on_shutdown() -> None:
    try:
        await get_coordinator().stop()
    except Exception:
        log.exception("coordinator shutdown raised; suppressed")
    tracing.flush()
