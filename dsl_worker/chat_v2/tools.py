"""v-next orchestrator tool handlers.

Each tool is registered with a name, JSON schema for OpenAI tool-use, and an
async handler `(args, ctx) -> (result_dict, cost_usd)`. The streaming loop
dispatches by name.

Tool list (15) — see REDESIGN_SPEC.md section 2.1 for the design rationale.

Tools that touch source adapters (table_create, table_extend) update the
table's last_fetch_* columns for the empirical cost preview.
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple

from sqlalchemy.orm import Session
from sqlalchemy import text as sa_text

from dsl_worker.sources_v2 import describe_source, get_adapter, list_sources


log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Context passed to every tool handler
# ---------------------------------------------------------------------------


@dataclass
class ToolContext:
    """Per-call context. Carries DB session + project/run identifiers."""
    db: Session
    project_id: str
    user_id: str
    run_id: Optional[str]
    # Optional async progress emitter: tools that loop / poll (apify
    # actors, web_harvest) call this with a one-line status string so the
    # FE shimmer reflects "Fetched 23/100 items…" instead of going dark.
    # Signature: async (message: str) -> None. None = no-op.
    emit_progress: Optional[Callable[[str], Awaitable[None]]] = None
    # Reserved for richer in-band events (approval cards, etc.) — wire on demand.
    emit_event: Optional[Callable[[str, Dict[str, Any]], None]] = None
    # Cooperative-cancel hook. Set to an asyncio.Event by the agent loop
    # so long-running tools (apify polling, cell-agent rows_fill) can
    # short-circuit gracefully on user cancel and capture their partial
    # cost before raising. task.cancel() still works as a backstop —
    # this just gives well-behaved tools a chance to abort external
    # work (e.g. apify_client.run.abort()) and report the cost so far.
    cancel_event: Optional[asyncio.Event] = None
    # Running tally of cost incurred by the current handler that hasn't
    # been returned yet. Long-running tools update this so the agent
    # loop's CancelledError handler can still attribute the spend to
    # the turn even though the handler never returned its tuple. Reset
    # to 0 by agent.py before each handler invocation.
    partial_cost_usd: float = 0.0


def resolve_table_id(db: Session, project_id: str, id_or_short: str) -> Optional[str]:
    """Look up a table's UUID from either its UUID or its short_id (t1, t2…).

    Returns the UUID as a string, or None if not found. The agent uses short_ids
    in tool args because they're cheap to emit; backend SQL uses UUIDs.
    """
    if not id_or_short:
        return None
    # Short ids never contain dashes; UUIDs always do.
    if "-" in id_or_short:
        row = db.execute(
            sa_text(
                "SELECT id::text FROM tables WHERE id=:id AND project_id=:pid AND deleted_at IS NULL"
            ),
            {"id": id_or_short, "pid": project_id},
        ).fetchone()
    else:
        row = db.execute(
            sa_text(
                "SELECT id::text FROM tables WHERE short_id=:sid AND project_id=:pid AND deleted_at IS NULL"
            ),
            {"sid": id_or_short, "pid": project_id},
        ).fetchone()
    return row[0] if row else None


def resolve_enrichment_id(db: Session, project_id: str, id_or_short: str) -> Optional[str]:
    """Look up an enrichment's UUID from UUID or short_id (e1, e2…)."""
    if not id_or_short:
        return None
    if "-" in id_or_short:
        row = db.execute(
            sa_text(
                "SELECT e.id::text FROM enrichments e JOIN tables t ON t.id=e.table_id "
                "WHERE e.id=:id AND t.project_id=:pid AND e.deleted_at IS NULL"
            ),
            {"id": id_or_short, "pid": project_id},
        ).fetchone()
    else:
        row = db.execute(
            sa_text(
                "SELECT e.id::text FROM enrichments e JOIN tables t ON t.id=e.table_id "
                "WHERE e.short_id=:sid AND t.project_id=:pid AND e.deleted_at IS NULL"
            ),
            {"sid": id_or_short, "pid": project_id},
        ).fetchone()
    return row[0] if row else None


def _next_short_id(db: Session, project_id: str) -> str:
    """Return the next free 't<N>' for this project. Caller holds a write lock
    on the parent (we're inside a transaction that's about to INSERT)."""
    row = db.execute(
        sa_text(
            "SELECT short_id FROM tables WHERE project_id=:pid "
            "ORDER BY (CASE WHEN short_id ~ '^t[0-9]+$' THEN CAST(substring(short_id, 2) AS int) ELSE 0 END) DESC LIMIT 1"
        ),
        {"pid": project_id},
    ).fetchone()
    if not row or not row[0] or not row[0].startswith("t"):
        return "t1"
    try:
        return f"t{int(row[0][1:]) + 1}"
    except (ValueError, IndexError):
        return "t1"


def _next_enrichment_short_id(db: Session, table_id: str) -> str:
    """Return the next free 'e<N>' for this table."""
    row = db.execute(
        sa_text(
            "SELECT short_id FROM enrichments WHERE table_id=:tid "
            "ORDER BY (CASE WHEN short_id ~ '^e[0-9]+$' THEN CAST(substring(short_id, 2) AS int) ELSE 0 END) DESC LIMIT 1"
        ),
        {"tid": table_id},
    ).fetchone()
    if not row or not row[0] or not row[0].startswith("e"):
        return "e1"
    try:
        return f"e{int(row[0][1:]) + 1}"
    except (ValueError, IndexError):
        return "e1"


# ---------------------------------------------------------------------------
# Tool: table_create
# ---------------------------------------------------------------------------


async def table_create(args: Dict[str, Any], ctx: ToolContext) -> Tuple[Dict[str, Any], float]:
    """Create a table from a source. Atomic: fetches rows, commits in one
    step. If the fetch fails or returns 0 rows, nothing is written.

    Args: source, query_params, name (2-5 word label). Optionally `columns`
    ([{name, source_field, type}]) — when omitted, the system commits a
    raw passthrough (every top-level row key becomes a column). The agent
    can then call `column_map_set` to rename / flatten nested fields after
    seeing the actual data.
    """
    name = (args.get("name") or args.get("table_name") or "").strip()
    source = args.get("source")
    query_params = args.get("query_params") or {}
    raw_columns = args.get("columns") or []
    n = int(args.get("n") or 100)

    if not source:
        return {"error": "source is required"}, 0.0
    if source.split(":", 1)[0] not in list_sources():
        return {"error": f"unknown source {source!r}", "available": list_sources()}, 0.0

    if not name:
        src_label = source.split(":", 1)[0].replace("_", " ").title()
        first_q = next(
            (str(v) for v in query_params.values() if isinstance(v, str) and v),
            None,
        )
        if not first_q:
            inp = query_params.get("input") or {}
            if isinstance(inp, dict):
                first_q = next(
                    (str(v) for v in inp.values() if isinstance(v, str) and v),
                    None,
                )
        name = f"{src_label} — {first_q}" if first_q else src_label

    adapter = get_adapter(source)
    val_err = adapter.validate_query_params(query_params)
    if val_err:
        return {"error": val_err}, 0.0

    # File source needs project_id to find files in the candidate store.
    # Underscore-prefixed keys are stripped before storing in table.query_params.
    if source == "file" and ctx.project_id:
        query_params = {**query_params, "_project_id": str(ctx.project_id)}

    # Fetch first batch. For sources that support streaming (apify
    # actors), grab the first batch synchronously and let the rest stream
    # in via a background task — apify can take minutes; the user
    # shouldn't sit at an empty table that long.
    streaming = source.startswith("apify_actor:") and hasattr(adapter, "fetch_stream")
    stream_gen = None
    res_rows: List[Dict[str, Any]] = []
    res_exhausted = True
    res_cost = 0.0
    # Closure that bumps the handler's accumulated cost up to ctx so
    # the agent loop's CancelledError catch can attribute spend that
    # happened AFTER the last completed tool but BEFORE the cancel
    # landed (e.g. an apify actor that was aborted mid-poll — Apify
    # bills for whatever compute units were burned). Cost is in
    # credits at this layer (1 credit = $0.10), so divide by 10 before
    # adding to the USD-denominated partial_cost_usd.
    def _track_partial_cost(cost_credits: float) -> None:
        try:
            ctx.partial_cost_usd += float(cost_credits) / 10.0
        except Exception:
            pass

    if streaming:
        try:
            stream_gen = adapter.fetch_stream(
                query_params, n, source_full=source, on_cost=_track_partial_cost,
            )
            first = await stream_gen.__anext__()
            res_rows = first.get("rows") or []
            res_exhausted = first.get("exhausted", False)
            res_cost = first.get("cost_credits", 0.0)
        except StopAsyncIteration:
            res_exhausted = True
        except asyncio.CancelledError:
            # Adapter already invoked on_cost in its CancelledError
            # path before re-raising — ctx.partial_cost_usd is up to
            # date. Let the cancel propagate so the agent loop's catch
            # flushes the turn ledger with the right amount.
            raise
        except Exception as e:
            log.exception("table_create stream-first-batch failed: %s", e)
            return {"error": f"source fetch failed: {e}"}, 0.0
    else:
        try:
            if source.startswith("apify_actor:"):
                res = await adapter.fetch(
                    query_params, n, prior_cursor=None, source_full=source,
                    on_cost=_track_partial_cost,
                )
            else:
                res = await adapter.fetch(query_params, n, prior_cursor=None)
            res_rows, res_exhausted, res_cost = res.rows, res.exhausted, res.cost_credits
        except asyncio.CancelledError:
            raise
        except Exception as e:
            log.exception("table_create fetch failed: %s", e)
            return {"error": f"source fetch failed: {e}"}, 0.0

    if not res_rows:
        return {"error": "source returned 0 rows; nothing to commit"}, 0.0

    # Columns: agent's choice if provided, otherwise raw passthrough of
    # whatever the rows contain. Agent can refine via column_map_set
    # after seeing the actual committed data.
    columns_for_db: List[Dict[str, str]] = []
    if raw_columns:
        for c in raw_columns:
            if isinstance(c, str):
                return {"error": f"columns must be dicts {{name, source_field, type}}; got bare string {c!r}"}, 0.0
            if not isinstance(c, dict):
                return {"error": f"columns entry must be a dict, got {type(c).__name__}"}, 0.0
            cname = c.get("name") or c.get("column_name") or c.get("key") or c.get("field")
            src_field = c.get("source_field") or c.get("from") or c.get("source")
            if not cname or not src_field:
                return {"error": f"columns entry needs name + source_field; got {c!r}"}, 0.0
            columns_for_db.append({
                "name": cname,
                "type": c.get("type") or "text",
                "source_field": src_field,
            })
    else:
        # Raw passthrough: every top-level key in the rows becomes a column.
        # Names are whatever the source emits (often snake_case); agent can
        # call column_map_set to clean them up after seeing the data.
        seen: Dict[str, None] = {}
        for r in res_rows:
            if isinstance(r, dict):
                for k in r.keys():
                    if k not in seen and not k.startswith("_"):
                        seen[k] = None
        columns_for_db = [
            {"name": k, "type": "text", "source_field": k}
            for k in seen
        ]

    # Ensure the project has a version row before either the sync or
    # background commit paths run — eliminates the project_versions
    # unique-index race between them.
    _ensure_project_version(ctx.db, ctx.project_id)

    table_id = str(uuid.uuid4())
    short_id = _next_short_id(ctx.db, ctx.project_id)
    # "streaming" only when there's actually a background drain task
    # spawning — i.e. apify fetch_stream with more rows coming. For
    # sync fetches (Apollo, FE, gmaps, web_harvest), the sync call IS
    # the whole fetch; status goes straight to 'complete' regardless
    # of whether the adapter happened to report exhausted=False (it
    # may just mean "more pages exist if you table_extend later").
    initial_status = "streaming" if (streaming and not res_exhausted) else "complete"
    ctx.db.execute(
        sa_text(
            """
            INSERT INTO tables (id, project_id, short_id, name, source, query_params, columns,
                                dedup_key_column, fetch_status,
                                last_fetch_returned_rows, last_fetch_cost_credits, last_fetch_at,
                                created_at)
            VALUES (:id, :project_id, :short_id, :name, :source, :query_params, CAST(:cols AS jsonb),
                    :dedup_key_column, :fetch_status,
                    :rows_n, :cost, now(),
                    now())
            """
        ),
        {
            "id": table_id,
            "project_id": ctx.project_id,
            "short_id": short_id,
            "name": name,
            "source": source,
            "query_params": json.dumps(query_params),
            "cols": json.dumps(columns_for_db),
            "dedup_key_column": adapter.default_dedup_key_column,
            "fetch_status": initial_status,
            "rows_n": len(res_rows),
            "cost": res_cost,
        },
    )
    ctx.db.commit()

    _commit_verify_tasks = _commit_rows(
        ctx.db, table_id, res_rows, columns_for_db,
        store_raw=True, run_id=ctx.run_id,
    )

    # Seed the table's comment thread with the agent's "initial description"
    # — what this table represents, rendered from the source adapter. The
    # description is visible in the table detail panel and surfaces in the
    # chat "table created" chip.
    table_desc = None
    try:
        table_desc = describe_source(source, query_params)
        body_parts = [f"**{table_desc.label}** — {table_desc.query_text}"]
        if table_desc.details:
            body_parts.append(table_desc.details)
        from dsl_worker.chat_v2.comments import seed_table_comment
        seed_table_comment(ctx.db, ctx.project_id, table_id, "\n\n".join(body_parts))
    except Exception:
        log.exception("table_create comment seed failed; continuing")

    # Emit a table_card_added SSE event so the chat sidebar can render the
    # "table created" chip inline next to the assistant message. The chip
    # shows favicon + label + query_text and expands to reveal details +
    # enrichments. FE collects these into message.table_cards (parallel to
    # message.sources, already on the message model).
    if ctx.run_id is not None and table_desc is not None:
        try:
            from dsl_worker.chat_api import runs as legacy_runs
            from dsl_api.models import ChatRun
            run_obj = ctx.db.query(ChatRun).filter(ChatRun.id == ctx.run_id).first()
            if run_obj is not None:
                legacy_runs.emit_event(ctx.db, run_obj, "table_card_added", {
                    "table_id": short_id,
                    "table_uuid": table_id,
                    "name": name,
                    "source": source,
                    "kind": table_desc.kind,
                    "label": table_desc.label,
                    "query_text": table_desc.query_text,
                    "details": table_desc.details,
                    "favicon_url": table_desc.favicon_url,
                })
        except Exception:
            log.exception("table_card_added emit failed; continuing")

    # If we're streaming, spawn a background task to drain the remaining
    # rows from the actor. Each new batch reads the table's CURRENT
    # columns at commit time, so if the agent calls column_map_set in
    # between, later rows get mapped through the new column set.
    if streaming and stream_gen is not None and not res_exhausted:
        asyncio.create_task(
            _drain_stream_into_table(
                stream_gen=stream_gen,
                table_id=table_id,
                project_id=ctx.project_id,
                run_id=ctx.run_id,
                first_yielded=len(res_rows),
                n_target=n,
            ),
            name=f"apify-stream-{short_id}",
        )

    # Verifications run in the background — table_create returns
    # immediately so the agent's next tool isn't blocked on dozens of
    # HTTP fetches + Haiku batches. Each verify writes to
    # `tags.url_verification` and emits a `row_merged` SSE event so the
    # FE patches the badge in place as results arrive. The tasks are
    # held by the event loop's registry; the asyncio scheduler keeps
    # them alive until done.
    del _commit_verify_tasks

    # Surface sample rows + the raw field schema so the agent can call
    # column_map_set in the same turn with clean names / nested paths /
    # a dedup key, having seen the actual data.
    return {
        "table_id": short_id,
        "name": name,
        "source": source,
        "rows_committed": len(res_rows),
        "columns": [c["name"] for c in columns_for_db],
        "fetch_status": initial_status,
        "streaming_in_background": initial_status == "streaming",
        "sample_for_mapping": _build_schema_preview(res_rows),
    }, res_cost * 0.10


async def _drain_stream_into_table(
    stream_gen,
    table_id: str,
    project_id: str,
    run_id: Optional[str],
    first_yielded: int,
    n_target: int,
) -> None:
    """Background task: pulls remaining batches from the source stream,
    commits each through the table's CURRENT columns (re-read per batch
    so column_map_set mid-stream is reflected for later rows), updates
    last_fetch_returned_rows incrementally, and flips fetch_status to
    'complete' on exhaustion.
    """
    from dsl_api.db import SessionLocal
    # Late import to avoid circular dep with chat_v2.runs.
    from dsl_worker.chat_api import runs as legacy_runs
    from dsl_api.models import ChatRun

    total_committed = first_yielded
    try:
        async for batch in stream_gen:
            new_rows = batch.get("rows") or []
            if not new_rows:
                if batch.get("exhausted"):
                    break
                continue
            db = SessionLocal()
            try:
                # Re-read columns so a mid-stream column_map_set takes effect
                # for these later rows.
                col_row = db.execute(
                    sa_text("SELECT columns FROM tables WHERE id=:id"),
                    {"id": table_id},
                ).fetchone()
                cols = []
                if col_row and col_row[0]:
                    raw = col_row[0]
                    cols = raw if isinstance(raw, list) else json.loads(raw or "[]")
                if not cols:
                    # Fallback: passthrough the keys we see now.
                    seen: Dict[str, None] = {}
                    for r in new_rows:
                        if isinstance(r, dict):
                            for k in r.keys():
                                if k not in seen and not k.startswith("_"):
                                    seen[k] = None
                    cols = [{"name": k, "type": "text", "source_field": k} for k in seen]

                # Background stream drain — fire-and-forget verify
                # tasks; their own emit adapter opens fresh sessions.
                _commit_rows(db, table_id, new_rows, cols, store_raw=True, run_id=run_id)
                total_committed += len(new_rows)
                db.execute(
                    sa_text(
                        "UPDATE tables SET last_fetch_returned_rows=:n, last_fetch_at=now() WHERE id=:id"
                    ),
                    {"n": total_committed, "id": table_id},
                )
                db.commit()

                # Emit a rows_added event into the chat_run_events stream
                # so the FE refreshes the table view. Best-effort.
                if run_id is not None:
                    try:
                        run_obj = db.query(ChatRun).filter(ChatRun.id == run_id).first()
                        if run_obj is not None:
                            legacy_runs.emit_event(db, run_obj, "rows_added", {
                                "table_id": table_id,
                                "added": len(new_rows),
                                "total": total_committed,
                            })
                    except Exception:
                        log.exception("rows_added emit failed; continuing")
            finally:
                db.close()
            if batch.get("exhausted") or total_committed >= n_target:
                break
    except Exception:
        log.exception("apify stream drain failed for table %s", table_id)
    finally:
        # Mark the table complete regardless of how we exited.
        db = SessionLocal()
        try:
            db.execute(
                sa_text(
                    "UPDATE tables SET fetch_status='complete' WHERE id=:id AND fetch_status='streaming'"
                ),
                {"id": table_id},
            )
            db.commit()
            if run_id is not None:
                try:
                    run_obj = db.query(ChatRun).filter(ChatRun.id == run_id).first()
                    if run_obj is not None:
                        legacy_runs.emit_event(db, run_obj, "table_stream_complete", {
                            "table_id": table_id,
                            "total": total_committed,
                        })
                except Exception:
                    log.exception("table_stream_complete emit failed")
        finally:
            db.close()


def _build_schema_preview(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Return a frequency-ranked field summary with example values.

    Shape:
      {
        row_count_pending: int,
        fields: [
          {name, present_in_rows, example_values: [v1, v2, v3]},
          ...  # ranked by frequency, top 30
        ],
        first_rows: [first 3 raw rows],
      }
    """
    from collections import Counter
    if not rows:
        return {"row_count_pending": 0, "fields": [], "first_rows": []}
    freq: Counter = Counter()
    examples: Dict[str, List[Any]] = {}
    for r in rows:
        if not isinstance(r, dict):
            continue
        for k, v in r.items():
            if v is None or v == "":
                continue
            freq[k] += 1
            if len(examples.setdefault(k, [])) < 3:
                examples[k].append(_truncate_for_preview(v))
    ranked = []
    for name, n in freq.most_common(30):
        ranked.append({
            "name": name,
            "present_in_rows": n,
            "example_values": examples.get(name, []),
        })
    return {
        "row_count_pending": len(rows),
        "fields": ranked,
        "first_rows": [
            {k: _truncate_for_preview(v) for k, v in r.items()}
            for r in rows[:3] if isinstance(r, dict)
        ],
    }


def _truncate_for_preview(v: Any) -> Any:
    """Compact a value for the schema preview so the agent sees the shape
    without us blowing context on long bodies."""
    if isinstance(v, str):
        return v[:120]
    if isinstance(v, list):
        return [_truncate_for_preview(x) for x in v[:3]]
    if isinstance(v, dict):
        return {k: _truncate_for_preview(vv) for k, vv in list(v.items())[:5]}
    return v


# ---------------------------------------------------------------------------
# Tool: table_extend
# ---------------------------------------------------------------------------


async def table_extend(args: Dict[str, Any], ctx: ToolContext) -> Tuple[Dict[str, Any], float]:
    """Fetch more rows into an existing table.

    The LLM constructs the full query_params for each call — including any
    pagination fields (offset, page, etc.) the source requires. The server
    does NOT track cursors or merge with prior params; the most recent
    query_params is stored on the table purely as a historical record the
    LLM can reference via project_state when deciding the next query.
    """
    table_id = resolve_table_id(ctx.db, ctx.project_id, args.get("table_id"))
    new_query_params = args.get("query_params") or {}
    n = int(args.get("n") or 100)

    if not table_id:
        return {"error": "table_id is required"}, 0.0

    row = ctx.db.execute(
        sa_text("SELECT source, columns FROM tables WHERE id=:id AND deleted_at IS NULL"),
        {"id": table_id},
    ).fetchone()
    if not row:
        return {"error": f"table {table_id} not found"}, 0.0
    source, columns = row[0], row[1]

    adapter = get_adapter(source)
    val_err = adapter.validate_query_params(new_query_params)
    if val_err:
        return {"error": val_err}, 0.0

    try:
        if source.startswith("apify_actor:"):
            res = await adapter.fetch(new_query_params, n, prior_cursor=None, source_full=source)
        else:
            res = await adapter.fetch(new_query_params, n, prior_cursor=None)
    except Exception as e:
        log.exception("table_extend fetch failed: %s", e)
        return {"error": f"source fetch failed: {e}"}, 0.0

    # Commit rows using existing column mapping
    cols = json.loads(columns) if isinstance(columns, str) else (columns or [])
    column_map = [{"source_field": c.get("source_field") or c["name"], "column_name": c["name"], "type": c["type"]} for c in cols]
    _extend_verify_tasks = _commit_rows(
        ctx.db, table_id, res.rows, column_map, run_id=ctx.run_id,
    )

    # Overwrite query_params with the LLM's exact params (no merge, no
    # cursor). project_state shows this back to the LLM so the next
    # "more" call sees what it just ran and can construct the appropriate
    # delta (e.g. bump offset, change page).
    ctx.db.execute(
        sa_text(
            """
            UPDATE tables
            SET last_fetch_returned_rows = :rows_n,
                last_fetch_cost_credits = :cost,
                last_fetch_at = now(),
                query_params = CAST(:qp AS jsonb)
            WHERE id = :id
            """
        ),
        {
            "id": table_id,
            "rows_n": len(res.rows),
            "cost": res.cost_credits,
            "qp": json.dumps(new_query_params),
        },
    )
    ctx.db.commit()

    # Verifications run in the background — see table_create for the
    # same fire-and-forget rationale.
    del _extend_verify_tasks

    return {
        "rows_added": len(res.rows),
    }, res.cost_credits * 0.10


# ---------------------------------------------------------------------------
# Tool: column_map_set
# ---------------------------------------------------------------------------


async def column_map_set(args: Dict[str, Any], ctx: ToolContext) -> Tuple[Dict[str, Any], float]:
    table_id = resolve_table_id(ctx.db, ctx.project_id, args.get("table_id"))
    raw_mapping = args.get("mapping")
    if raw_mapping is None:
        raw_mapping = args.get("columns")  # alias the agent reaches for
    dedup_key_column = args.get("dedup_key_column")
    if not table_id:
        return {"error": "table_id is required"}, 0.0
    if not raw_mapping:
        return {
            "error": (
                "mapping is required. Accepts any of: "
                "{source_field: column_name}, "
                "{source_field: {name, type}}, or "
                "[{source_field, name, type}]"
            )
        }, 0.0

    # Normalize the three shapes the agent reaches for into a uniform
    # [{name, type, source_field}, ...] list.
    columns_for_db: List[Dict[str, str]] = []
    if isinstance(raw_mapping, dict):
        for src, v in raw_mapping.items():
            if isinstance(v, str):
                columns_for_db.append({"name": v, "type": "text", "source_field": src})
            elif isinstance(v, dict):
                name = v.get("name") or v.get("column_name") or src
                columns_for_db.append({
                    "name": name,
                    "type": v.get("type") or "text",
                    "source_field": src,
                })
            else:
                return {"error": f"mapping[{src!r}] must be a string or {{name, type}}; got {type(v).__name__}"}, 0.0
    elif isinstance(raw_mapping, list):
        for item in raw_mapping:
            if not isinstance(item, dict):
                return {"error": f"mapping list items must be dicts; got {type(item).__name__}"}, 0.0
            src = item.get("source_field") or item.get("from") or item.get("source")
            name = item.get("name") or item.get("column_name") or src
            if not src or not name:
                return {"error": f"mapping list entry needs source_field + name; got {item!r}"}, 0.0
            columns_for_db.append({
                "name": name,
                "type": item.get("type") or "text",
                "source_field": src,
            })
    else:
        return {"error": f"mapping must be dict or list; got {type(raw_mapping).__name__}"}, 0.0

    # column_map_set rewrites the table's columns AND re-derives every
    # row's mapped cell values from the stored raw_row. The agent can
    # rename / add / drop columns and switch source_field paths (e.g.
    # flatten nested fields) without re-fetching the source.
    table_row = ctx.db.execute(
        sa_text("SELECT 1 FROM tables WHERE id=:id AND deleted_at IS NULL"),
        {"id": table_id},
    ).fetchone()
    if not table_row:
        return {"error": f"table {table_id} not found"}, 0.0

    ctx.db.execute(
        sa_text(
            """
            UPDATE tables
            SET columns = CAST(:cols AS jsonb),
                dedup_key_column = COALESCE(:dedup, dedup_key_column)
            WHERE id = :id
            """
        ),
        {
            "id": table_id,
            "cols": json.dumps(columns_for_db),
            "dedup": dedup_key_column,
        },
    )

    # Re-derive every sample's mapped row from raw_row through the new
    # column set. Pull all in one query to minimize round-trips. Qualify
    # every column — `id` exists on both samples and tables.
    sample_rows = ctx.db.execute(
        sa_text(
            "SELECT s.id::text, s.raw_row, s.tags, t.source FROM samples s "
            "JOIN tables t ON t.id = s.table_id "
            "WHERE s.table_id=:tid AND s.deleted_at IS NULL AND s.raw_row IS NOT NULL"
        ),
        {"tid": table_id},
    ).fetchall()
    # Rebuild per-column citations from the new column_map so renamed /
    # added / dropped columns produce fresh source_record entries.
    new_cell_sources = {
        c["name"]: [
            {
                "type": "source_record",
                "source": None,  # filled per row below from the joined source
                "source_field": c["source_field"],
            }
        ]
        for c in columns_for_db
    }
    rederived = 0
    # Capture each re-derived row so we can fire verifications after the
    # commit. column_map_set is the third row-write site (along with
    # _commit_rows and enrichment) and URLs land here when the agent
    # renames or remaps a column from raw_row to a v2 column.
    rederived_rows: List[Tuple[str, Dict[str, Any]]] = []
    for sid, raw, tags, src in sample_rows:
        if not isinstance(raw, dict):
            continue
        mapped = {
            c["name"]: _extract_source_value(raw, c["source_field"])
            for c in columns_for_db
        }
        cell_sources = {
            name: [{**entry, "source": src} for entry in entries]
            for name, entries in new_cell_sources.items()
        }
        next_tags = dict(tags) if isinstance(tags, dict) else {}
        next_tags["sources"] = cell_sources
        ctx.db.execute(
            sa_text(
                "UPDATE samples SET row=CAST(:row AS jsonb), tags=CAST(:tags AS jsonb) WHERE id=:id"
            ),
            {
                "row": json.dumps(mapped, default=str),
                "tags": json.dumps(next_tags, default=str),
                "id": sid,
            },
        )
        rederived += 1
        rederived_rows.append((sid, mapped))
    ctx.db.commit()

    return {
        "ok": True,
        "columns_committed": len(columns_for_db),
        "rows_rederived": rederived,
    }, 0.0


# ---------------------------------------------------------------------------
# Tool: table_delete
# ---------------------------------------------------------------------------


async def table_delete(args: Dict[str, Any], ctx: ToolContext) -> Tuple[Dict[str, Any], float]:
    table_id = resolve_table_id(ctx.db, ctx.project_id, args.get("table_id"))
    if not table_id:
        return {"error": "table_id is required"}, 0.0
    # Read short_id before delete so the emitted event matches what the
    # FE has cached (it keys tables by short_id, not uuid).
    short_id_row = ctx.db.execute(
        sa_text("SELECT short_id FROM tables WHERE id=:id"),
        {"id": table_id},
    ).fetchone()
    short_id = short_id_row[0] if short_id_row else None
    ctx.db.execute(
        sa_text("UPDATE tables SET deleted_at=now() WHERE id=:id AND project_id=:pid"),
        {"id": table_id, "pid": ctx.project_id},
    )
    ctx.db.commit()

    # Emit a table_card_removed event so the FE drops the deleted table
    # from the tabs bar immediately. Without this, the deleted table
    # lingers in the tabs until SOME other event (rows_added on a new
    # table) triggers a tables refresh — which is the exact "deleted
    # table stayed visible until the new one finished" bug.
    if ctx.run_id is not None:
        try:
            from dsl_worker.chat_api import runs as legacy_runs
            from dsl_api.models import ChatRun
            run_obj = ctx.db.query(ChatRun).filter(ChatRun.id == ctx.run_id).first()
            if run_obj is not None:
                legacy_runs.emit_event(ctx.db, run_obj, "table_card_removed", {
                    "table_id": short_id,
                    "table_uuid": table_id,
                })
        except Exception:
            log.exception("table_card_removed emit failed; continuing")

    return {"ok": True}, 0.0


# ---------------------------------------------------------------------------
# Tool: filter_set / filter_clear
# ---------------------------------------------------------------------------


_FILTER_OP_ALIASES = {
    # Symbolic → canonical
    "=": "equals", "==": "equals", "eq": "equals",
    "!=": "not_equals", "<>": "not_equals", "neq": "not_equals",
    ">": "gt", ">=": "gte",
    "<": "lt", "<=": "lte",
    "does_not_contain": "not_contains",
    "startswith": "starts_with",
    "endswith": "ends_with",
    "is_any_of": "in", "any_of": "in",
    "is_none_of": "not_in",
    "text_include_exclude": "text_inc_exc",
}

# Canonical set of ops we accept (after alias-normalization). Source of
# truth lives in chat_v2/__init__.py:FILTER_OPS; this mirror is kept
# here to avoid an import cycle (__init__ imports tools.HANDLERS).
_CANONICAL_FILTER_OPS = {
    "contains", "not_contains", "starts_with", "ends_with",
    "equals", "not_equals",
    "contains_any", "contains_all", "not_contains_any", "not_contains_all",
    "text_inc_exc",
    "in", "not_in",
    "gt", "gte", "lt", "lte", "between",
    "is_null", "is_not_null",
}


async def filter_set(args: Dict[str, Any], ctx: ToolContext) -> Tuple[Dict[str, Any], float]:
    # Accept friendly aliases the agent reaches for, plus the nested
    # {filter: {op, value}} shape it sometimes emits.
    table_id = resolve_table_id(ctx.db, ctx.project_id, args.get("table_id"))
    nested = args.get("filter")
    if isinstance(nested, dict):
        op = nested.get("op") or nested.get("operator") or nested.get("filter_type")
        value = nested.get("value")
    else:
        op = (
            args.get("op")
            or args.get("operator")
            or args.get("filter_type")
            or args.get("comparison")
        )
        value = args.get("value")
    column = args.get("column") or args.get("column_name") or args.get("field")
    if not (table_id and column and op):
        return {
            "error": "filter_set requires table_id, column, op. "
                     "Example: {table_id, column: 'industry', op: 'contains', value: 'SaaS'}",
            "got_keys": list(args.keys()),
        }, 0.0
    # Normalize op + reject unknown ones so the agent gets corrective
    # feedback instead of silently no-op'ing (which is what was
    # happening before — unknown ops fell through _filters_to_where_sql,
    # the table looked unchanged, and the agent re-tried different
    # guesses without ever knowing why).
    op_raw = str(op).lower().strip()
    op = _FILTER_OP_ALIASES.get(op_raw, op_raw)
    if op not in _CANONICAL_FILTER_OPS:
        return {
            "error": f"filter_set: unsupported op {op_raw!r}. ",
            "supported_ops": sorted(_CANONICAL_FILTER_OPS),
            "hint": "For text use contains/equals/text_inc_exc; for numbers use gt/gte/lt/lte/between/equals; "
                    "use is_null/is_not_null for any column.",
        }, 0.0
    # Upsert
    ctx.db.execute(
        sa_text(
            """
            INSERT INTO table_filters (id, table_id, column_name, op, value, created_at)
            VALUES (gen_random_uuid(), :tid, :col, :op, CAST(:val AS jsonb), now())
            ON CONFLICT (table_id, column_name) DO UPDATE
            SET op = EXCLUDED.op, value = EXCLUDED.value
            """
        ),
        {"tid": table_id, "col": column, "op": op, "val": json.dumps(value)},
    )
    # Return matched count + sample for agent sanity-check
    matched, sample = _apply_filter_count_sample(ctx.db, table_id, column, op, value)
    total = ctx.db.execute(
        sa_text("SELECT COUNT(*) FROM samples WHERE table_id=:tid AND deleted_at IS NULL"),
        {"tid": table_id},
    ).scalar() or 0
    ctx.db.commit()
    return {"matched": matched, "total": total, "sample": sample}, 0.0


async def filter_clear(args: Dict[str, Any], ctx: ToolContext) -> Tuple[Dict[str, Any], float]:
    table_id = resolve_table_id(ctx.db, ctx.project_id, args.get("table_id"))
    column = args.get("column")
    if not (table_id and column):
        return {"error": "table_id, column required"}, 0.0
    ctx.db.execute(
        sa_text("DELETE FROM table_filters WHERE table_id=:tid AND column_name=:col"),
        {"tid": table_id, "col": column},
    )
    ctx.db.commit()
    return {"ok": True}, 0.0


# ---------------------------------------------------------------------------
# Tool: sort_set / sort_clear
# ---------------------------------------------------------------------------


async def sort_set(args: Dict[str, Any], ctx: ToolContext) -> Tuple[Dict[str, Any], float]:
    """Set the active sort on a table. Single sort per table for v1.

    Args:
      table_id: required.
      column: required — the column name to sort by.
      direction: "asc" or "desc". Defaults to "desc".
    """
    table_id = resolve_table_id(ctx.db, ctx.project_id, args.get("table_id"))
    column = args.get("column") or args.get("column_name")
    direction = (args.get("direction") or args.get("dir") or "desc").lower()
    if direction not in ("asc", "desc"):
        return {"error": f"direction must be 'asc' or 'desc'; got {direction!r}"}, 0.0
    if not (table_id and column):
        return {"error": "sort_set requires table_id and column"}, 0.0
    ctx.db.execute(
        sa_text(
            "UPDATE tables SET sort_column = :col, sort_direction = :dir "
            "WHERE id = :tid"
        ),
        {"tid": table_id, "col": column, "dir": direction},
    )
    ctx.db.commit()
    return {"ok": True, "column": column, "direction": direction}, 0.0


async def sort_clear(args: Dict[str, Any], ctx: ToolContext) -> Tuple[Dict[str, Any], float]:
    """Remove the active sort on a table."""
    table_id = resolve_table_id(ctx.db, ctx.project_id, args.get("table_id"))
    if not table_id:
        return {"error": "table_id required"}, 0.0
    ctx.db.execute(
        sa_text(
            "UPDATE tables SET sort_column = NULL, sort_direction = NULL WHERE id = :tid"
        ),
        {"tid": table_id},
    )
    ctx.db.commit()
    return {"ok": True}, 0.0


# ---------------------------------------------------------------------------
# Tool: row_inspect / row_delete
# ---------------------------------------------------------------------------


async def row_inspect(args: Dict[str, Any], ctx: ToolContext) -> Tuple[Dict[str, Any], float]:
    table_id = resolve_table_id(ctx.db, ctx.project_id, args.get("table_id"))
    n = int(args.get("n") or 10)
    if not table_id:
        return {"error": "table_id is required"}, 0.0
    rows = ctx.db.execute(
        sa_text(
            "SELECT row FROM samples WHERE table_id=:tid AND deleted_at IS NULL "
            "ORDER BY created_at DESC LIMIT :n"
        ),
        {"tid": table_id, "n": n},
    ).fetchall()
    return {"rows": [r[0] for r in rows]}, 0.0


async def row_delete(args: Dict[str, Any], ctx: ToolContext) -> Tuple[Dict[str, Any], float]:
    table_id = resolve_table_id(ctx.db, ctx.project_id, args.get("table_id"))
    row_ids = args.get("row_ids") or []
    if not (table_id and row_ids):
        return {"error": "table_id and row_ids required"}, 0.0
    result = ctx.db.execute(
        sa_text(
            "UPDATE samples SET deleted_at=now() WHERE table_id=:tid AND id::text = ANY(:ids)"
        ),
        {"tid": table_id, "ids": row_ids},
    )
    ctx.db.commit()
    return {"rows_deleted": result.rowcount or 0}, 0.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ensure_project_version(db: Session, project_id: str) -> str:
    """Ensure the project has a current_version_id; create one if not.
    Safe to call concurrently — uses ON CONFLICT on the unique
    (project_id, version_number) index, then re-reads. Returns the
    version UUID as a string.
    """
    version_id = db.execute(
        sa_text("SELECT current_version_id::text FROM projects WHERE id=:id"),
        {"id": str(project_id)},
    ).scalar()
    if version_id:
        return version_id
    new_id = str(uuid.uuid4())
    db.execute(
        sa_text(
            """
            INSERT INTO project_versions
              (id, project_id, version_number, generation_prompt,
               num_samples, columns, use_internet, files_snapshot,
               examples_snapshot, status, generated_count, created_at)
            VALUES
              (:id, :pid, 1, '', 0, '[]'::jsonb, false, '[]'::jsonb,
               '[]'::jsonb, 'complete', 0, now())
            ON CONFLICT (project_id, version_number) DO NOTHING
            """
        ),
        {"id": new_id, "pid": str(project_id)},
    )
    db.execute(
        sa_text(
            "UPDATE projects SET current_version_id=COALESCE(current_version_id, "
            "(SELECT id FROM project_versions WHERE project_id=:pid AND version_number=1 LIMIT 1)) "
            "WHERE id=:pid"
        ),
        {"pid": str(project_id)},
    )
    db.commit()
    version_id = db.execute(
        sa_text("SELECT current_version_id::text FROM projects WHERE id=:id"),
        {"id": str(project_id)},
    ).scalar()
    return version_id


def _commit_rows(
    db: Session,
    table_id: str,
    rows: List[Dict[str, Any]],
    column_map: List[Dict[str, str]],
    store_raw: bool = True,
    run_id: Optional[str] = None,
) -> List[asyncio.Task]:
    """Commit fetched rows into the samples table with column_map applied.

    Returns any URL/email verification tasks spawned for the inserted
    rows so async callers can await them within their tool window
    (background drain tasks just discard the return value — fire and
    forget is fine there).
    """
    if not rows:
        return []

    # Pull project_id and source in one shot. source feeds per-cell
    # `tags.sources` citations so the FE can link every mapped cell back
    # to the raw_row payload via SourceRecordDetailPanel.
    tbl_row = db.execute(
        sa_text("SELECT project_id, source FROM tables WHERE id=:id"),
        {"id": table_id},
    ).fetchone()
    if not tbl_row:
        log.error("_commit_rows: table %s not found", table_id)
        return []
    pid, source = tbl_row[0], tbl_row[1]
    version_id = db.execute(
        sa_text("SELECT current_version_id FROM projects WHERE id=:id"),
        {"id": str(pid)},
    ).scalar()
    if not version_id:
        version_id = _ensure_project_version(db, str(pid))
        if not version_id:
            log.error("_commit_rows: could not ensure project_version for %s", pid)
            return []

    # column_map entries: [{name, source_field, type}]. source_field can be:
    #   - a plain key:           "founders"
    #   - a dotted path:         "founder_info.email"
    #   - an array map:          "founders[].name"   → list of values
    # Tolerate the legacy "column_name" key from older adapter default_columns.
    # Preserves `type` (e.g. "url", "email") so the verify hook can pick
    # the right columns to check.
    normalized_map = [
        {
            "name": (c.get("name") or c.get("column_name")),
            "source_field": c["source_field"],
            "type": c.get("type") or "text",
        }
        for c in column_map
        if c.get("source_field") and (c.get("name") or c.get("column_name"))
    ]

    # Pre-build the per-column citation list — same shape for every row in
    # the batch, only differs by sample_id (filled per row below).
    cell_sources_template: Dict[str, List[Dict[str, Any]]] = {}
    if store_raw and source:
        for c in normalized_map:
            cell_sources_template[c["name"]] = [
                {
                    "type": "source_record",
                    "source": source,
                    "source_field": c["source_field"],
                }
            ]

    # Serialize concurrent _commit_rows for the same version (sync first
    # batch + apify background drain + any other parallel commits) so
    # MAX(seq)+1 + INSERT pairs don't race on idx_samples_version_seq_unique.
    # Advisory lock is per-version and released at db.commit().
    db.execute(
        sa_text("SELECT pg_advisory_xact_lock(hashtextextended(:vid, 0))"),
        {"vid": str(version_id)},
    )

    next_seq_row = db.execute(
        sa_text(
            "SELECT COALESCE(MAX(seq), 0) + 1 FROM samples WHERE version_id=:vid"
        ),
        {"vid": str(version_id)},
    ).scalar()
    next_seq = int(next_seq_row or 1)

    # Generate sample_ids client-side so we can hand them to the verify
    # hook after commit (was using `gen_random_uuid()` server-side, which
    # didn't return the new id without an extra round-trip).
    pending_verify: List[Tuple[str, Dict[str, Any]]] = []
    for r in rows:
        mapped: Dict[str, Any] = {}
        for c in normalized_map:
            mapped[c["name"]] = _extract_source_value(r, c["source_field"])
        tags_payload = (
            {"sources": cell_sources_template} if cell_sources_template else None
        )
        sample_id = str(uuid.uuid4())
        db.execute(
            sa_text(
                "INSERT INTO samples (id, project_id, table_id, version_id, seq, row, raw_row, tags, created_at) "
                "VALUES (:sid, :pid, :tid, :vid, :seq, CAST(:row AS jsonb), CAST(:raw AS jsonb), CAST(:tags AS jsonb), now())"
            ),
            {
                "sid": sample_id,
                "pid": str(pid),
                "tid": table_id,
                "vid": str(version_id),
                "seq": next_seq,
                "row": json.dumps(mapped, default=str),
                "raw": json.dumps(r, default=str) if store_raw else None,
                "tags": json.dumps(tags_payload) if tags_payload else None,
            },
        )
        next_seq += 1
        pending_verify.append((sample_id, mapped))
    # Commit to release the advisory lock + flush the batch. Must happen
    # BEFORE we schedule verification — the verify task opens its own
    # SessionLocal and won't see uncommitted rows.
    db.commit()
    return []


def _extract_source_value(row: Dict[str, Any], path: str) -> Any:
    """Resolve a source_field path against a row.

    Supports plain keys (`name`), dotted paths (`founder.email`), and
    array fan-out (`founders[].name` → list of names). Returns None if
    any step is missing.
    """
    if not path:
        return None
    # Plain key fast path.
    if "." not in path and "[]" not in path:
        return row.get(path)

    segments = path.split(".")
    current: Any = row
    for seg in segments:
        if current is None:
            return None
        if seg.endswith("[]"):
            key = seg[:-2]
            if isinstance(current, dict):
                current = current.get(key)
            if not isinstance(current, list):
                return None
            # The remaining segments apply to each element. Recurse.
            remaining = ".".join(segments[segments.index(seg) + 1 :])
            if not remaining:
                return current
            return [_extract_source_value(item, remaining) if isinstance(item, dict) else item for item in current]
        if isinstance(current, dict):
            current = current.get(seg)
        else:
            return None
    return current


def _apply_filter_count_sample(
    db: Session, table_id: str, column: str, op: str, value: Any
) -> Tuple[int, List[Dict[str, Any]]]:
    """Return (matched_count, sample_rows[5]) for the filter without materializing it."""
    # Naive: pull rows, filter in Python. Good enough for v1 with table sizes up to ~1000.
    all_rows = db.execute(
        sa_text("SELECT row FROM samples WHERE table_id=:tid AND deleted_at IS NULL"),
        {"tid": table_id},
    ).fetchall()
    matched: List[Dict[str, Any]] = []
    for (row,) in all_rows:
        if not isinstance(row, dict):
            continue
        v = row.get(column)
        if _match(v, op, value):
            matched.append(row)
    return len(matched), matched[:5]


def _match(cell: Any, op: str, value: Any) -> bool:
    if op == "equals":
        return cell == value
    if op == "not_equals":
        return cell != value
    if op == "contains":
        return value and isinstance(cell, str) and value in cell
    if op == "not_contains":
        return not (value and isinstance(cell, str) and value in cell)
    if op == "starts_with":
        return isinstance(cell, str) and cell.startswith(value)
    if op == "ends_with":
        return isinstance(cell, str) and cell.endswith(value)
    if op == "is_null":
        return cell is None or cell == ""
    if op == "is_not_null":
        return cell is not None and cell != ""
    if op == ">":
        try:
            return float(cell) > float(value)
        except (TypeError, ValueError):
            return False
    if op == "<":
        try:
            return float(cell) < float(value)
        except (TypeError, ValueError):
            return False
    if op == ">=":
        try:
            return float(cell) >= float(value)
        except (TypeError, ValueError):
            return False
    if op == "<=":
        try:
            return float(cell) <= float(value)
        except (TypeError, ValueError):
            return False
    if op == "between":
        if not (isinstance(value, list) and len(value) == 2):
            return False
        try:
            f = float(cell)
            return float(value[0]) <= f <= float(value[1])
        except (TypeError, ValueError):
            return False
    if op == "in":
        return cell in (value or [])
    if op == "not_in":
        return cell not in (value or [])
    return False


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


# Map tool name -> handler. Used by the chat streaming loop.
HANDLERS: Dict[str, Callable[[Dict[str, Any], ToolContext], Awaitable[Tuple[Dict[str, Any], float]]]] = {
    "table_create": table_create,
    "table_extend": table_extend,
    "column_map_set": column_map_set,
    "table_delete": table_delete,
    "filter_set": filter_set,
    "filter_clear": filter_clear,
    "sort_set": sort_set,
    "sort_clear": sort_clear,
    "row_inspect": row_inspect,
    "row_delete": row_delete,
    # enrichment_set, enrichment_run, code_exec, web_search, suggest_replies,
    # apify_search_actors, apify_actor_details — separate modules (see
    # enrichment.py, web_tools.py, apify_discovery.py).
}
