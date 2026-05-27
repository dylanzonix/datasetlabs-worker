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
import re
import uuid
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple

from sqlalchemy.orm import Session
from sqlalchemy import text as sa_text

from dsl_worker.sources import describe_source, get_adapter, list_sources
from dsl_worker.chat.instrumentation import phase_marker, phase_span_async, time_commit


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


# Sources where pulling 1000 rows is effectively free (covered by the
# plan's flat search quota, not the per-record credit pool). Everything
# else pays per row (Apify per-item, FullEnrich credits, Google Maps
# per-call, etc.) so we keep the default safety net at 100 to avoid
# surprise spend when the agent omits `n`.
_FREE_HIGH_N_SOURCES = {"apollo_companies", "apollo_people"}


def _default_n_for_source(source: Optional[str]) -> int:
    if not source:
        return 100
    kind = source.split(":", 1)[0]
    return 1000 if kind in _FREE_HIGH_N_SOURCES else 100


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


_TXEY_RE = re.compile(r"^(t\d+)(e\d+)$")


def resolve_enrichment_id(db: Session, project_id: str, id_or_short: str) -> Optional[str]:
    """Look up an enrichment's UUID from UUID or short_id.

    Three input shapes:
      - UUID (contains '-') → exact match
      - t<N>e<N> (e.g. 't1e2') → composite; disambiguates by table
      - e<N> (legacy) → project-wide. If two enrichments share the
        short_id across tables (the bug fixed by adopting t<X>e<Y>),
        prefer the most recently created — that's almost always what
        the agent meant since it just configured it.
    """
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
        return row[0] if row else None

    # Composite t<X>e<Y> short_id — disambiguate by table.
    m = _TXEY_RE.match(id_or_short)
    if m:
        table_short, enr_short = m.group(1), m.group(2)
        # Match by full short_id (e.g. "t1e2") OR by the enrichment-only
        # part stored against the right table — covers both new rows
        # (full composite stored) and legacy rows where short_id was just
        # the e<N> piece.
        row = db.execute(
            sa_text(
                "SELECT e.id::text FROM enrichments e JOIN tables t ON t.id=e.table_id "
                "WHERE t.project_id=:pid AND t.short_id=:tshort AND e.deleted_at IS NULL "
                "AND (e.short_id=:full OR e.short_id=:eshort) "
                "ORDER BY e.created_at DESC LIMIT 1"
            ),
            {"pid": project_id, "tshort": table_short, "full": id_or_short, "eshort": enr_short},
        ).fetchone()
        return row[0] if row else None

    # Legacy bare e<N>. Tie-break by most recent so an agent that just
    # configured the enrichment lands on its own row, not an older
    # collision from a different table.
    row = db.execute(
        sa_text(
            "SELECT e.id::text FROM enrichments e JOIN tables t ON t.id=e.table_id "
            "WHERE e.short_id=:sid AND t.project_id=:pid AND e.deleted_at IS NULL "
            "ORDER BY e.created_at DESC LIMIT 1"
        ),
        {"sid": id_or_short, "pid": project_id},
    ).fetchone()
    return row[0] if row else None


def _next_short_id(db: Session, project_id: str) -> str:
    """Return the next free 't<N>' for this project.

    Acquires a per-project advisory lock that's held until the caller's
    transaction commits — so a parallel table_create in another branch
    waits here instead of racing on the (project_id, short_id) unique
    index. Salt 1 distinguishes from the version-id lock in _commit_rows
    (salt 0) so the two locks never collide on the same hash slot.
    """
    db.execute(
        sa_text("SELECT pg_advisory_xact_lock(hashtextextended(:pid, 1))"),
        {"pid": str(project_id)},
    )
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
    """Return the next free 't<X>e<N>' for this table.

    Composite IDs (e.g. 't1e2') eliminate the bare-e<N> collision that
    occurred when the same number existed on two tables in one project:
    the project-wide resolver picked whichever row PostgreSQL served
    first, which was almost never the one the agent meant. Per-table
    numbering is preserved — 't1e3' means "the 3rd enrichment on t1".

    Per-table advisory lock prevents parallel enrichment_set on the
    same table from colliding on (table_id, short_id) unique. Salt 2
    keeps this distinct from the project-level table lock (salt 1) and
    the version lock (salt 0).
    """
    db.execute(
        sa_text("SELECT pg_advisory_xact_lock(hashtextextended(:tid, 2))"),
        {"tid": str(table_id)},
    )
    # Find the highest existing enrichment number on this table —
    # accept both legacy "e<N>" and composite "t<X>e<N>" rows so the
    # next free number doesn't collide with either format.
    row = db.execute(
        sa_text(
            "SELECT short_id FROM enrichments WHERE table_id=:tid "
            "ORDER BY ("
            "  CASE "
            "    WHEN short_id ~ '^t[0-9]+e[0-9]+$' THEN CAST(substring(short_id FROM 'e([0-9]+)$') AS int) "
            "    WHEN short_id ~ '^e[0-9]+$' THEN CAST(substring(short_id, 2) AS int) "
            "    ELSE 0 "
            "  END"
            ") DESC LIMIT 1"
        ),
        {"tid": table_id},
    ).fetchone()
    # Look up this table's short_id (t1, t2, …) for the prefix.
    tshort_row = db.execute(
        sa_text("SELECT short_id FROM tables WHERE id=:tid"),
        {"tid": table_id},
    ).fetchone()
    tshort = (tshort_row[0] if tshort_row and tshort_row[0] else "t1")
    next_n = 1
    if row and row[0]:
        s = row[0]
        try:
            # Strip the t<X> prefix if present, then drop the leading "e".
            n_part = s.rsplit("e", 1)[-1]
            next_n = int(n_part) + 1
        except (ValueError, IndexError):
            next_n = 1
    return f"{tshort}e{next_n}"


def _resolve_enrichment_position(
    db: Session,
    table_id: str,
    *,
    insert_before: Optional[str] = None,
) -> int:
    """Pick the position slot for a new enrichment on this table.

    Default: append (max(position) + 1).
    If `insert_before` (an existing enrichment short_id like "t1e2") is
    given, return that enrichment's position and shift it + every later
    one by +1 so the new row slots in at the requested spot. Same
    advisory lock as _next_enrichment_short_id so the shift + the new
    INSERT are atomic.
    """
    db.execute(
        sa_text("SELECT pg_advisory_xact_lock(hashtextextended(:tid, 2))"),
        {"tid": str(table_id)},
    )
    if insert_before:
        target = db.execute(
            sa_text(
                "SELECT position FROM enrichments "
                "WHERE table_id=:tid AND short_id=:sid AND deleted_at IS NULL"
            ),
            {"tid": table_id, "sid": insert_before},
        ).fetchone()
        if target and target[0] is not None:
            pos = int(target[0])
            db.execute(
                sa_text(
                    "UPDATE enrichments SET position = position + 1 "
                    "WHERE table_id=:tid AND position >= :pos AND deleted_at IS NULL"
                ),
                {"tid": table_id, "pos": pos},
            )
            return pos
        # insert_before referenced a missing enrichment — fall through and
        # append rather than failing; the agent gets the new row at the
        # end which is still a valid placement.
    row = db.execute(
        sa_text(
            "SELECT COALESCE(MAX(position), 0) FROM enrichments "
            "WHERE table_id=:tid AND deleted_at IS NULL"
        ),
        {"tid": table_id},
    ).fetchone()
    return int(row[0] or 0) + 1


def _record_query_run(
    db: Session,
    *,
    table_id: str,
    action: str,
    source: str,
    query_params: Dict[str, Any],
    status: str,
    rows_returned: Optional[int] = None,
    rows_added: Optional[int] = None,
    rows_skipped_duplicates: Optional[int] = None,
    cost_credits: Optional[float] = None,
    error: Optional[str] = None,
) -> None:
    """Append a row to table_query_runs for the audit trail the user sees in
    the table detail panel. Best-effort — failures here never block the
    parent fetch from returning to the agent.
    """
    try:
        db.execute(
            sa_text(
                """
                INSERT INTO table_query_runs (
                    id, table_id, action, source, query_params, status,
                    rows_returned, rows_added, rows_skipped_duplicates,
                    cost_credits, error, created_at
                ) VALUES (
                    :id, :table_id, :action, :source, CAST(:qp AS jsonb), :status,
                    :rows_returned, :rows_added, :rows_skipped_duplicates,
                    :cost, :error, now()
                )
                """
            ),
            {
                "id": str(uuid.uuid4()),
                "table_id": table_id,
                "action": action,
                "source": source,
                "qp": json.dumps(query_params or {}),
                "status": status,
                "rows_returned": rows_returned,
                "rows_added": rows_added,
                "rows_skipped_duplicates": rows_skipped_duplicates,
                "cost": cost_credits,
                "error": (error or None) if error is None else error[:2000],
            },
        )
        db.commit()
    except Exception:
        log.exception("table_query_runs insert failed (table_id=%s); continuing", table_id)
        try:
            db.rollback()
        except Exception:
            pass


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

    Optional `wait` (default true). When false, the call returns immediately
    with `{status: "running", task_id: "bt<N>"}` and the fetch + commit runs
    as a tracked background task. Agent monitors via `task_status` /
    `task_wait`. The `wait` arg is popped before recursing so the spawned
    task runs the sync path.
    """
    args = dict(args)
    wait = bool(args.pop("wait", True))
    if not wait:
        from dsl_worker.chat import background_tasks as _bg
        nm = (args.get("name") or args.get("table_name") or "").strip()
        src = args.get("source") or "?"
        summary = f"Creating table {nm!r} from {src}" if nm else f"Creating table from {src}"
        spawn_result = await _bg.spawn(
            handler=table_create,
            args=args,
            ctx=ctx,
            kind="table_create",
            task_key=None,
            summary=summary,
        )
        return spawn_result, 0.0

    name = (args.get("name") or args.get("table_name") or "").strip()
    source = args.get("source")
    query_params = args.get("query_params") or {}
    raw_columns = args.get("columns") or []

    if not source:
        return {"error": "source is required"}, 0.0
    if source.split(":", 1)[0] not in list_sources():
        return {"error": f"unknown source {source!r}", "available": list_sources()}, 0.0

    n = int(args.get("n") or _default_n_for_source(source))

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

    # Hard rail: browser_use and apify_actor:* don't honor a pre-declared
    # schema — they produce whatever JSON keys the page/actor emits.
    # Accepting `columns` here historically caused silent null cells when
    # the agent's idealized source_fields (`url`, `category`) didn't match
    # the source's actual keys (`listing_url`, `category_code`). Force the
    # two-step flow: fetch first, see the preview, then column_map_set.
    if raw_columns and (source == "browser_use" or source.startswith("apify_actor:")):
        return {
            "error": (
                f"{source} can't pre-declare columns — the row shape comes from "
                "the page/actor, not from your schema. Call table_create without "
                "`columns`, inspect the returned sample/schema preview, then call "
                "column_map_set with source_fields that match the actual keys."
            ),
        }, 0.0

    # web_harvest is the LLM-driven research source. By default the LLM
    # picks whatever JSON keys it wants and the agent reconciles
    # afterward via column_map_set. When the agent passes `columns`
    # upfront on table_create, we pipe those source_field paths into
    # query_params.__existing_schema so the adapter prompt locks the
    # LLM to those exact keys. Skipping column_map_set entirely is
    # then safe — the schema matches what the agent asked for.
    if source == "web_harvest" and raw_columns:
        schema_keys = [
            (c.get("source_field") or c.get("name"))
            for c in raw_columns
            if isinstance(c, dict) and (c.get("source_field") or c.get("name"))
        ]
        if schema_keys:
            query_params = {**query_params, "__existing_schema": schema_keys}

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

    res_total_entries: Optional[int] = None
    async with phase_span_async(ctx, "table_create/adapter_fetch", source=source, n=n):
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
                return {"error": f"source fetch failed: {type(e).__name__}: {e}"}, 0.0
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
                res_total_entries = getattr(res, "total_entries", None)
            except asyncio.CancelledError:
                raise
            except Exception as e:
                log.exception("table_create fetch failed: %s", e)
                return {"error": f"source fetch failed: {type(e).__name__}: {e}"}, 0.0
    phase_marker(ctx, "table_create/fetch_returned", rows=len(res_rows), cost=res_cost)

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
            entry: Dict[str, Any] = {
                "name": cname,
                "type": c.get("type") or "text",
                "source_field": src_field,
            }
            fmt = c.get("format")
            if fmt:
                entry["format"] = fmt
            if c.get("pinned"):
                entry["pinned"] = True
            columns_for_db.append(entry)
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

    with time_commit(ctx, "table_create_insert", threshold_ms=100):
        _commit_verify_tasks = _commit_rows(
            ctx.db, table_id, res_rows, columns_for_db,
            store_raw=True, run_id=ctx.run_id,
        )

    # Append to the query-history audit trail. _commit_rows just stashed
    # dedup stats keyed by table_id; pick them up here.
    _create_stats = _LAST_COMMIT_STATS.get(table_id, {})
    _record_query_run(
        ctx.db,
        table_id=table_id,
        action="create",
        source=source,
        query_params=query_params,
        status="success",
        rows_returned=len(res_rows),
        rows_added=int(_create_stats.get("inserted", len(res_rows))),
        rows_skipped_duplicates=int(_create_stats.get("skipped_duplicates", 0)),
        cost_credits=float(res_cost) if res_cost is not None else None,
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
        from dsl_worker.chat.comments import seed_table_comment
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
            from dsl_worker.chat import run_state
            from dsl_api.models import ChatRun
            run_obj = ctx.db.query(ChatRun).filter(ChatRun.id == ctx.run_id).first()
            if run_obj is not None:
                run_state.emit_event(ctx.db, run_obj, "table_card_added", {
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
    result: Dict[str, Any] = {
        "table_id": short_id,
        "name": name,
        "source": source,
        "rows_committed": len(res_rows),
        "columns": [c["name"] for c in columns_for_db],
        "fetch_status": initial_status,
        "streaming_in_background": initial_status == "streaming",
        "sample_for_mapping": _build_schema_preview(res_rows),
    }
    # Total pool size from the source, when exposed (e.g. Apollo's
    # pagination.total_entries). Lets the orchestrator detect over-narrow
    # filters: "got 11 rows but source had 11 total" → broaden BEFORE
    # committing the next step.
    if res_total_entries is not None:
        result["total_matching_in_source"] = res_total_entries
    return result, res_cost * 0.10


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
    # Late import to avoid circular dep with chat.runs.
    from dsl_worker.chat import run_state
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
                # with the actual inserted rows so the FE can applyRowInsert
                # each one instead of refetching the whole table. The
                # _LAST_COMMIT_STATS sidecar populated by _commit_rows
                # carries the FE-canonical payloads (id + mapped data +
                # tags).
                if run_id is not None:
                    try:
                        run_obj = db.query(ChatRun).filter(ChatRun.id == run_id).first()
                        if run_obj is not None:
                            stats = _LAST_COMMIT_STATS.get(table_id) or {}
                            run_state.emit_event(db, run_obj, "rows_added", {
                                "table_id": table_id,
                                "added": int(stats.get("inserted") or 0),
                                "total": total_committed,
                                "rows": stats.get("inserted_rows") or [],
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
                        run_state.emit_event(db, run_obj, "table_stream_complete", {
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

    Optional `wait` (default true). Same semantics as table_create — when
    false, returns immediately with a task_id and the fetch runs in
    background.
    """
    args = dict(args)
    wait = bool(args.pop("wait", True))
    if not wait:
        from dsl_worker.chat import background_tasks as _bg
        # Resolve table_id once so task_key carries the canonical UUID
        # (FE can correlate the running indicator to the table card).
        canonical_tid = resolve_table_id(ctx.db, ctx.project_id, args.get("table_id"))
        spawn_result = await _bg.spawn(
            handler=table_extend,
            args=args,
            ctx=ctx,
            kind="table_extend",
            task_key=canonical_tid,
            summary=f"Extending table {args.get('table_id') or '?'}",
        )
        return spawn_result, 0.0

    table_id = resolve_table_id(ctx.db, ctx.project_id, args.get("table_id"))
    new_query_params = args.get("query_params") or {}

    if not table_id:
        return {"error": "table_id is required"}, 0.0

    row = ctx.db.execute(
        sa_text("SELECT source, columns FROM tables WHERE id=:id AND deleted_at IS NULL"),
        {"id": table_id},
    ).fetchone()
    if not row:
        return {"error": f"table {table_id} not found"}, 0.0
    source, columns = row[0], row[1]

    # Source-aware default: Apollo search is free so default high; paid
    # sources (Apify per-item, FullEnrich credits, Google Maps per-call)
    # keep the safety net at 100. Agent can still pass explicit n.
    n = int(args.get("n") or _default_n_for_source(source))

    adapter = get_adapter(source)
    val_err = adapter.validate_query_params(new_query_params)
    if val_err:
        _record_query_run(
            ctx.db,
            table_id=table_id,
            action="extend",
            source=source,
            query_params=new_query_params,
            status="error",
            error=val_err,
        )
        return {"error": val_err}, 0.0

    # For schema-unpredictable sources (web_harvest, llm) the LLM picks
    # its own row keys. table_extend reuses the table's column_map from
    # the first fetch, so if the LLM picks DIFFERENT keys on the
    # extend, every mapped cell goes empty except the few keys that
    # happen to match. Pipe the existing source_field paths into
    # query_params so the adapter can constrain its prompt to those
    # exact keys.
    if source in ("web_harvest", "llm"):
        cols_for_schema = json.loads(columns) if isinstance(columns, str) else (columns or [])
        existing_schema = [
            c.get("source_field") or c.get("name")
            for c in cols_for_schema
            if (c.get("source_field") or c.get("name"))
        ]
        if existing_schema:
            new_query_params = {**new_query_params, "__existing_schema": existing_schema}

    try:
        async with phase_span_async(ctx, "table_extend/adapter_fetch", source=source, n=n):
            if source.startswith("apify_actor:"):
                res = await adapter.fetch(new_query_params, n, prior_cursor=None, source_full=source)
            else:
                res = await adapter.fetch(new_query_params, n, prior_cursor=None)
        phase_marker(ctx, "table_extend/fetch_returned", rows=len(res.rows), cost=res.cost_credits)
    except Exception as e:
        log.exception("table_extend fetch failed: %s", e)
        _record_query_run(
            ctx.db,
            table_id=table_id,
            action="extend",
            source=source,
            query_params=new_query_params,
            status="error",
            error=str(e),
        )
        return {"error": f"source fetch failed: {type(e).__name__}: {e}"}, 0.0

    # Commit rows using existing column mapping
    cols = json.loads(columns) if isinstance(columns, str) else (columns or [])
    column_map = [{"source_field": c.get("source_field") or c["name"], "column_name": c["name"], "type": c["type"]} for c in cols]
    with time_commit(ctx, "table_extend_insert", threshold_ms=100):
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

    # Read the dedup-aware counts that _commit_rows stashed. rows_added
    # is the count of rows that actually landed (post dedup); the actor
    # may have returned more if it duplicated items.
    stats = _LAST_COMMIT_STATS.get(table_id, {})
    inserted = int(stats.get("inserted", len(res.rows)))
    skipped_dup = int(stats.get("skipped_duplicates", 0))

    _record_query_run(
        ctx.db,
        table_id=table_id,
        action="extend",
        source=source,
        query_params=new_query_params,
        status="empty" if not res.rows else "success",
        rows_returned=len(res.rows),
        rows_added=inserted,
        rows_skipped_duplicates=skipped_dup,
        cost_credits=float(res.cost_credits) if res.cost_credits is not None else None,
    )

    result_te: Dict[str, Any] = {
        "rows_added": inserted,
        "rows_skipped_duplicates": skipped_dup,
        "rows_returned_by_source": len(res.rows),
    }
    te_total = getattr(res, "total_entries", None)
    if te_total is not None:
        result_te["total_matching_in_source"] = te_total
    return result_te, res.cost_credits * 0.10


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
    # [{name, type, source_field, format?}, ...] list. `format` is the
    # number-formatting hint the FE consumes; preserve it whenever the
    # agent set it.
    columns_for_db: List[Dict[str, Any]] = []
    if isinstance(raw_mapping, dict):
        for src, v in raw_mapping.items():
            if isinstance(v, str):
                columns_for_db.append({"name": v, "type": "text", "source_field": src})
            elif isinstance(v, dict):
                name = v.get("name") or v.get("column_name") or src
                entry: Dict[str, Any] = {
                    "name": name,
                    "type": v.get("type") or "text",
                    "source_field": src,
                }
                fmt = v.get("format")
                if fmt:
                    entry["format"] = fmt
                if v.get("pinned"):
                    entry["pinned"] = True
                columns_for_db.append(entry)
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
            entry = {
                "name": name,
                "type": item.get("type") or "text",
                "source_field": src,
            }
            fmt = item.get("format")
            if fmt:
                entry["format"] = fmt
            if item.get("pinned"):
                entry["pinned"] = True
            columns_for_db.append(entry)
    else:
        return {"error": f"mapping must be dict or list; got {type(raw_mapping).__name__}"}, 0.0

    # column_map_set rewrites the table's columns AND re-derives every
    # row's mapped cell values from the stored raw_row. The agent can
    # rename / add / drop columns and switch source_field paths (e.g.
    # flatten nested fields) without re-fetching the source.
    table_row = ctx.db.execute(
        sa_text("SELECT columns FROM tables WHERE id=:id AND deleted_at IS NULL"),
        {"id": table_id},
    ).fetchone()
    if not table_row:
        return {"error": f"table {table_id} not found"}, 0.0

    # Build a source_field -> old_column_name map BEFORE we overwrite
    # tables.columns. Used below to migrate per-column tag buckets
    # (email_verification, fill_status) when the agent renames a column —
    # without this, Scrubby verdicts stay keyed by the pre-rename name
    # and the FE badge lookup (which keys by current column name) misses
    # every row.
    old_cols_raw = table_row[0]
    old_cols = old_cols_raw if isinstance(old_cols_raw, list) else (
        json.loads(old_cols_raw) if old_cols_raw else []
    )
    old_sf_to_name: Dict[str, str] = {}
    for c in old_cols:
        if isinstance(c, dict):
            sf = c.get("source_field")
            nm = c.get("name") or c.get("column_name")
            if sf and nm:
                old_sf_to_name[str(sf)] = str(nm)
    # Reverse the new column list to source_field -> new name.
    new_sf_to_name: Dict[str, str] = {
        str(c["source_field"]): str(c["name"]) for c in columns_for_db
    }
    # rename_map: old_name -> new_name for columns whose source_field is
    # still present. Columns dropped in the new mapping have their tag
    # entries removed; new columns start fresh.
    rename_map: Dict[str, str] = {}
    for sf, old_name in old_sf_to_name.items():
        new_name = new_sf_to_name.get(sf)
        if new_name and new_name != old_name:
            rename_map[old_name] = new_name

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
    # Track per-column null counts so the agent can spot a mapping
    # that silently produced nulls (e.g. a wrong source_field path).
    # Without this signal the agent has no idea its mapping failed
    # until a downstream tool surfaces the empty cells.
    null_counts: Dict[str, int] = {c["name"]: 0 for c in columns_for_db}
    for sid, raw, tags, src in sample_rows:
        if not isinstance(raw, dict):
            continue
        mapped = {
            c["name"]: _extract_source_value(raw, c["source_field"])
            for c in columns_for_db
        }
        for c in columns_for_db:
            v = mapped.get(c["name"])
            if v in (None, "", [], {}):
                null_counts[c["name"]] += 1
        cell_sources = {
            name: [{**entry, "source": src} for entry in entries]
            for name, entries in new_cell_sources.items()
        }
        next_tags = dict(tags) if isinstance(tags, dict) else {}
        next_tags["sources"] = cell_sources
        # Migrate per-column tag buckets through the rename map so
        # Scrubby verdicts (email_verification) + null reasons
        # (fill_status) stay aligned with the new column names. Keys
        # for columns that were dropped in the new mapping are removed;
        # keys for columns that didn't change keep their entry intact.
        new_cols_set = set(new_sf_to_name.values())
        for bucket_key in ("email_verification", "fill_status", "failed_emails"):
            bucket = next_tags.get(bucket_key)
            if not isinstance(bucket, dict) or not bucket:
                continue
            migrated: Dict[str, Any] = {}
            for k, v in bucket.items():
                target = rename_map.get(k, k)
                if target in new_cols_set:
                    migrated[target] = v
            if migrated:
                next_tags[bucket_key] = migrated
            else:
                next_tags.pop(bucket_key, None)
        # Reconcile INVALID emails. When a column is renamed mid-stream
        # (Scrubby verify task submitted under the old name, re-derive
        # writes under the new name), the verdict tag lands here but the
        # value stays in `mapped[new_name]`. Without this sweep, INVALID
        # emails remain visible in cells + exports — the user has to
        # know what Scrubby said, which defeats the verify gate.
        ev_now = next_tags.get("email_verification") or {}
        if isinstance(ev_now, dict):
            for col_name, verdict in ev_now.items():
                if not isinstance(verdict, dict):
                    continue
                if verdict.get("status") != "INVALID":
                    continue
                stale = verdict.get("value")
                if stale and mapped.get(col_name) == stale:
                    mapped[col_name] = None
                    # Make the empty-cell reason discoverable in the FE.
                    fs = dict(next_tags.get("fill_status") or {})
                    fs.setdefault(col_name, {
                        "status": "null_legitimate",
                        "reason": "Email failed verification — Scrubby marked it Invalid.",
                        "cost": 0.0,
                        "strategy": "scrubby_verify",
                    })
                    next_tags["fill_status"] = fs
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

    # Return enough that the agent can verify the mapping worked without a
    # follow-up row_inspect: the committed schema, one sample mapped row,
    # and per-column null counts (a column that's 100% null after re-derive
    # is almost certainly a bad source_field — easy to catch and fix).
    sample_mapped = rederived_rows[0][1] if rederived_rows else None
    fully_null_columns = [
        name for name, n in null_counts.items()
        if rederived > 0 and n == rederived
    ]
    return {
        "ok": True,
        "columns_committed": len(columns_for_db),
        "columns": columns_for_db,
        "rows_rederived": rederived,
        "sample_rederived_row": sample_mapped,
        "null_counts_per_column": null_counts,
        "fully_null_columns": fully_null_columns,
    }, 0.0


# ---------------------------------------------------------------------------
# Tool: table_delete
# ---------------------------------------------------------------------------


async def table_delete(args: Dict[str, Any], ctx: ToolContext) -> Tuple[Dict[str, Any], float]:
    table_id = resolve_table_id(ctx.db, ctx.project_id, args.get("table_id"))
    if not table_id:
        return {"error": "table_id is required"}, 0.0
    # Read short_id + name + row count before delete so the result echoes
    # exactly what disappeared (FE keys tables by short_id, agent reasons
    # about deletions by name and scale).
    pre = ctx.db.execute(
        sa_text(
            "SELECT t.short_id, t.name, "
            "(SELECT COUNT(*) FROM samples WHERE table_id=t.id AND deleted_at IS NULL) "
            "FROM tables t WHERE t.id=:id"
        ),
        {"id": table_id},
    ).fetchone()
    short_id = pre[0] if pre else None
    table_name = pre[1] if pre else None
    rows_deleted = int(pre[2]) if pre and pre[2] is not None else 0
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
            from dsl_worker.chat import run_state
            from dsl_api.models import ChatRun
            run_obj = ctx.db.query(ChatRun).filter(ChatRun.id == ctx.run_id).first()
            if run_obj is not None:
                run_state.emit_event(ctx.db, run_obj, "table_card_removed", {
                    "table_id": short_id,
                    "table_uuid": table_id,
                })
        except Exception:
            log.exception("table_card_removed emit failed; continuing")

    return {
        "ok": True,
        "table_id": short_id,
        "name": table_name,
        "rows_deleted": rows_deleted,
    }, 0.0


# ---------------------------------------------------------------------------
# Tool: filter_set / filter_clear
# ---------------------------------------------------------------------------


# Canonical filter ops — must match chat/__init__.py:FILTER_OPS.
# Mirrored here to avoid an import cycle.
# Per-table stats from the last _commit_rows call. Read by table_extend
# (and table_create's apify streaming drain) so the public tool result
# can report rows_added vs rows_skipped_duplicates honestly. Last-write-
# wins; small leak per table is fine.
_LAST_COMMIT_STATS: Dict[str, Dict[str, int]] = {}


# Canonical set mirrors chat/__init__.py:FILTER_OPS. `is_null` was
# removed from the AI's vocabulary (see comment there); the normalizer
# below still accepts is_null aliases and passes them through so legacy
# DB rows continue to apply, but `filter_set` schema validation gates
# new writes to this set.
_CANONICAL_FILTER_OPS = {
    "text_inc_exc", "is_any_of", "between", "gte", "lte",
    "is_not_null",
}


def _normalize_filter(op_raw: Any, value: Any) -> Optional[Tuple[str, Any]]:
    """Map any op + value the agent (or legacy FE) emits into the canonical
    (op, value) pair. Returns None if the op has no clean mapping — caller
    should surface an error with the canonical set.

    The 7 canonical ops are intentionally small (FE filter UI has 1:1
    coverage). Older / wider ops collapse:
      - `contains` / `starts_with` / `ends_with` / `equals` (text)
            → text_inc_exc {include: [value], exclude: []}
      - `not_contains`     → text_inc_exc {include: [], exclude: [value]}
      - `contains_any`     → text_inc_exc {include: list,  exclude: []}
      - `not_contains_any` → text_inc_exc {include: [],    exclude: list}
      - `in` / `is_any_of` → is_any_of   list
      - `equals` (number)  → between [v, v]
      - `>=` / `gte` / `>` → gte   (note: strict `>` rounds to inclusive)
      - `<=` / `lte` / `<` → lte
    No-clean-mapping (returns None): `not_equals`, `contains_all`,
    `not_contains_all`, `not_in`, `is_none_of` — these can't be expressed in
    the FE filter UI, so we reject rather than silently doing the wrong
    thing.
    """
    if op_raw is None:
        return None
    op = str(op_raw).strip().lower()

    # Already canonical — passthrough.
    if op in _CANONICAL_FILTER_OPS:
        return op, value

    # Numeric / date ranges.
    if op in (">", ">=", "gt", "gte"):
        return "gte", value
    if op in ("<", "<=", "lt", "lte"):
        return "lte", value

    # Multi-value membership.
    if op in ("in", "any_of"):
        items = value if isinstance(value, list) else [value]
        return "is_any_of", items

    # Equality. Without column type info, default to is_any_of which works
    # for both text and number columns; agent can override by writing
    # `between [v, v]` for numeric exact-match if needed.
    if op in ("=", "==", "eq", "equals"):
        items = value if isinstance(value, list) else [value]
        return "is_any_of", items

    # Substring family — collapse to text_inc_exc include side.
    if op in ("contains", "starts_with", "ends_with", "startswith", "endswith", "icontains"):
        return "text_inc_exc", {"include": [str(value)] if value is not None else [], "exclude": []}
    if op in ("not_contains", "does_not_contain"):
        return "text_inc_exc", {"include": [], "exclude": [str(value)] if value is not None else []}
    if op == "contains_any":
        items = value if isinstance(value, list) else [value]
        return "text_inc_exc", {"include": [str(v) for v in items if v is not None], "exclude": []}
    if op == "not_contains_any":
        items = value if isinstance(value, list) else [value]
        return "text_inc_exc", {"include": [], "exclude": [str(v) for v in items if v is not None]}
    if op == "text_include_exclude":
        return "text_inc_exc", value

    # Null checks — accept several shapes.
    if op in ("is_null", "isnull", "is_empty"):
        return "is_null", None
    if op in ("is_not_null", "is_not_empty", "exists"):
        return "is_not_null", None

    # Ops that can't be expressed in the FE filter UI — refuse.
    # (not_equals, contains_all, not_contains_all, not_in, is_none_of)
    return None


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
    # Normalize op + value into the canonical 7-op set. Legacy ops
    # (`contains`, `>=`, `in`, etc.) collapse into canonical ops with
    # adjusted values. Ops with no FE-renderable equivalent (`not_equals`,
    # `contains_all`, etc.) return an error — picking one of those silently
    # would store a filter the user can't see/edit in the filter panel.
    op_raw = str(op).lower().strip()
    normalized = _normalize_filter(op_raw, value)
    if normalized is None:
        return {
            "error": (
                f"filter_set: op {op_raw!r} has no equivalent in the filter UI. "
                "Pick one of the 7 canonical ops below — they map 1:1 to filter "
                "controls the user can see and edit."
            ),
            "supported_ops": sorted(_CANONICAL_FILTER_OPS),
            "hint": (
                "text/url/email → text_inc_exc {include, exclude}. "
                "enum → is_any_of [strings]. "
                "number/date → between [min,max] | gte n | lte n. "
                "any column → is_null | is_not_null."
            ),
        }, 0.0
    op, value = normalized
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
    # Return matched count + both samples + canonical op so the agent can
    # (a) confirm it filtered the right things AND excluded the right things,
    # (b) verify the op it asked for is the op that got stored (we
    # normalize aliases like `>` → `gt`, `startswith` → `starts_with`;
    # echoing the canonical form lets the agent tell at a glance).
    matched, sample_kept, sample_excluded = _apply_filter_count_sample(
        ctx.db, table_id, column, op, value,
    )
    total = ctx.db.execute(
        sa_text("SELECT COUNT(*) FROM samples WHERE table_id=:tid AND deleted_at IS NULL"),
        {"tid": table_id},
    ).scalar() or 0
    ctx.db.commit()
    return {
        "matched": matched,
        "total": total,
        "op": op,
        "sample_kept": sample_kept,
        "sample_excluded": sample_excluded,
    }, 0.0


async def filter_clear(args: Dict[str, Any], ctx: ToolContext) -> Tuple[Dict[str, Any], float]:
    table_id = resolve_table_id(ctx.db, ctx.project_id, args.get("table_id"))
    column = args.get("column")
    if not (table_id and column):
        return {"error": "table_id, column required"}, 0.0
    # Read existing filter (if any) so the result echoes what was cleared.
    # Lets the agent tell apart "you removed the filter I wanted to remove"
    # from "there was no filter on that column to begin with".
    pre = ctx.db.execute(
        sa_text(
            "SELECT op, value FROM table_filters "
            "WHERE table_id=:tid AND column_name=:col"
        ),
        {"tid": table_id, "col": column},
    ).fetchone()
    ctx.db.execute(
        sa_text("DELETE FROM table_filters WHERE table_id=:tid AND column_name=:col"),
        {"tid": table_id, "col": column},
    )
    # Count remaining filters on this table so agent has the post-state.
    remaining = ctx.db.execute(
        sa_text("SELECT COUNT(*) FROM table_filters WHERE table_id=:tid"),
        {"tid": table_id},
    ).scalar() or 0
    ctx.db.commit()
    return {
        "ok": True,
        "filter_existed": pre is not None,
        "cleared": {"column": column, "op": pre[0], "value": pre[1]} if pre else None,
        "remaining_filters_on_table": int(remaining),
    }, 0.0


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
    # Read prior sort so the agent can see what was cleared.
    pre = ctx.db.execute(
        sa_text("SELECT sort_column, sort_direction FROM tables WHERE id = :tid"),
        {"tid": table_id},
    ).fetchone()
    was_sorted = bool(pre and pre[0])
    ctx.db.execute(
        sa_text(
            "UPDATE tables SET sort_column = NULL, sort_direction = NULL WHERE id = :tid"
        ),
        {"tid": table_id},
    )
    ctx.db.commit()
    return {
        "ok": True,
        "was_sorted": was_sorted,
        "cleared": {"column": pre[0], "direction": pre[1]} if was_sorted else None,
    }, 0.0


# ---------------------------------------------------------------------------
# Tool: row_inspect / row_delete
# ---------------------------------------------------------------------------


async def row_inspect(args: Dict[str, Any], ctx: ToolContext) -> Tuple[Dict[str, Any], float]:
    """Peek at rows. When include_raw=true, also returns the raw_row payload
    from the source so the agent can see fields that exist in the source
    but aren't in the current column_map — useful when the user wants a
    column "unhidden" (just call column_map_set with the new source_field).
    """
    table_id = resolve_table_id(ctx.db, ctx.project_id, args.get("table_id"))
    n = int(args.get("n") or args.get("limit") or 10)
    include_raw = bool(args.get("include_raw") or args.get("with_raw"))
    if not table_id:
        return {"error": "table_id is required"}, 0.0
    select_cols = "row, raw_row" if include_raw else "row"
    rows = ctx.db.execute(
        sa_text(
            f"SELECT {select_cols} FROM samples WHERE table_id=:tid AND deleted_at IS NULL "
            "ORDER BY created_at DESC LIMIT :n"
        ),
        {"tid": table_id, "n": n},
    ).fetchall()
    if include_raw:
        # Also surface the set of source_field keys NOT currently mapped —
        # that's the signal the agent needs to decide "should I column_map_set
        # this field in?" without guessing what raw fields exist.
        cols_row = ctx.db.execute(
            sa_text("SELECT columns FROM tables WHERE id=:tid"),
            {"tid": table_id},
        ).fetchone()
        mapped_source_fields: set[str] = set()
        if cols_row and cols_row[0]:
            cols_list = cols_row[0] if isinstance(cols_row[0], list) else json.loads(cols_row[0])
            for c in cols_list:
                sf = c.get("source_field") if isinstance(c, dict) else None
                if sf:
                    # Keep just the top-level key for the "unmapped" comparison
                    mapped_source_fields.add(sf.split(".")[0].split("[]")[0])
        unmapped_raw_fields: set[str] = set()
        for r in rows:
            raw = r[1] if isinstance(r[1], dict) else {}
            if isinstance(raw, dict):
                for k in raw.keys():
                    if k not in mapped_source_fields and not k.startswith("_"):
                        unmapped_raw_fields.add(k)
        return {
            "rows": [{"row": r[0], "raw_row": r[1]} for r in rows],
            "unmapped_raw_fields": sorted(unmapped_raw_fields),
        }, 0.0
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

    # Dedup against existing rows when the table has a dedup_key_column.
    # Without this, table_extend (and apify's background streaming drain)
    # cheerfully inserts every row the adapter returns — including rows
    # that match existing ones, since most actors don't paginate via the
    # `startUrls` config and re-return the same items on each call. The
    # dedup_key was set on the table but never enforced; this is the
    # missing enforcement.
    dedup_key_row = db.execute(
        sa_text("SELECT dedup_key_column FROM tables WHERE id=:tid"),
        {"tid": table_id},
    ).fetchone()
    dedup_key = dedup_key_row[0] if dedup_key_row else None
    existing_dedup_values: set[str] = set()
    if dedup_key:
        existing = db.execute(
            sa_text(
                "SELECT DISTINCT row->>:k FROM samples "
                "WHERE table_id=:tid AND deleted_at IS NULL AND row->>:k IS NOT NULL"
            ),
            {"k": dedup_key, "tid": table_id},
        ).fetchall()
        existing_dedup_values = {x[0] for x in existing if x[0] is not None and x[0] != ""}

    # Generate sample_ids client-side so we can hand them to the verify
    # hook after commit (was using `gen_random_uuid()` server-side, which
    # didn't return the new id without an extra round-trip).
    #
    # Build the full insert payload in memory, then issue ONE multi-row
    # INSERT. The previous loop ran one INSERT per row → N network
    # round-trips to Supabase. With ~50ms RTT and 1000 rows, table_create
    # took 50s+ just on commit. A single multi-row INSERT is one
    # round-trip regardless of batch size (Postgres caps params per query
    # at 65,535 — at 5 params per row we can pack ~13k rows; we chunk
    # at 1000 to stay well under).
    pending_verify: List[Tuple[str, Dict[str, Any]]] = []
    insert_payloads: List[Dict[str, Any]] = []
    skipped_dup_count = 0
    for r in rows:
        mapped: Dict[str, Any] = {}
        for c in normalized_map:
            mapped[c["name"]] = _extract_source_value(r, c["source_field"])

        if dedup_key:
            key_val = mapped.get(dedup_key)
            if key_val not in (None, "") and str(key_val) in existing_dedup_values:
                skipped_dup_count += 1
                continue
            if key_val not in (None, ""):
                existing_dedup_values.add(str(key_val))

        tags_payload = (
            {"sources": cell_sources_template} if cell_sources_template else None
        )
        sample_id = str(uuid.uuid4())
        insert_payloads.append({
            "sid": sample_id,
            "seq": next_seq,
            "row": json.dumps(mapped, default=str),
            "raw": json.dumps(r, default=str) if store_raw else None,
            "tags": json.dumps(tags_payload) if tags_payload else None,
        })
        next_seq += 1
        pending_verify.append((sample_id, mapped))

    # Chunked multi-row INSERT. project_id / table_id / version_id are
    # identical for every row in this commit so we hoist them out of the
    # per-row params; only sid/seq/row/raw/tags vary.
    CHUNK_SIZE = 500
    for chunk_start in range(0, len(insert_payloads), CHUNK_SIZE):
        chunk = insert_payloads[chunk_start:chunk_start + CHUNK_SIZE]
        if not chunk:
            continue
        values_sql_parts: List[str] = []
        params: Dict[str, Any] = {
            "pid": str(pid),
            "tid": table_id,
            "vid": str(version_id),
        }
        for i, row_params in enumerate(chunk):
            values_sql_parts.append(
                f"(:sid{i}, :pid, :tid, :vid, :seq{i}, "
                f"CAST(:row{i} AS jsonb), CAST(:raw{i} AS jsonb), CAST(:tags{i} AS jsonb), now())"
            )
            params[f"sid{i}"] = row_params["sid"]
            params[f"seq{i}"] = row_params["seq"]
            params[f"row{i}"] = row_params["row"]
            params[f"raw{i}"] = row_params["raw"]
            params[f"tags{i}"] = row_params["tags"]
        db.execute(
            sa_text(
                "INSERT INTO samples (id, project_id, table_id, version_id, seq, row, raw_row, tags, created_at) "
                "VALUES " + ", ".join(values_sql_parts)
            ),
            params,
        )
    # Commit to release the advisory lock + flush the batch. Must happen
    # BEFORE we schedule verification — the verify task opens its own
    # SessionLocal and won't see uncommitted rows.
    db.commit()
    if skipped_dup_count:
        log.info(
            "_commit_rows: skipped %d duplicate rows on dedup_key=%s for table %s",
            skipped_dup_count, dedup_key, table_id,
        )
    # Stash the dedup counts on a module-level dict keyed by table_id so
    # table_extend (and friends) can read them without changing every
    # caller's signature. Last-write-wins; only the most recent commit
    # for a table is what callers care about.
    # Stash the inserted rows in FE-canonical shape (id + mapped data)
    # so emitters of the `rows_added` SSE event can carry the payloads
    # — that lets the FE applyRowInsert() each one surgically instead
    # of refetching the whole table. Without this, the FE bounced a
    # full /tables/{id}/rows on every batch which was the post-fetch
    # jitter source on streaming tables.
    inserted_rows_payload: List[Dict[str, Any]] = []
    for sid, mapped in pending_verify:
        payload: Dict[str, Any] = {"id": sid}
        payload.update(mapped)
        # tags_payload was the same for every row in this batch — attach
        # so the FE has cell-side citations from frame 1.
        if cell_sources_template:
            payload["tags"] = {"sources": cell_sources_template}
        inserted_rows_payload.append(payload)
    _LAST_COMMIT_STATS[table_id] = {
        "inserted": len(pending_verify),
        "skipped_duplicates": skipped_dup_count,
        "inserted_rows": inserted_rows_payload,
    }

    # Email verification for any email cells just written. Imported
    # lazily so this module stays import-cycle-free. Fire-and-forget:
    # tasks pin themselves in the hook's _BACKGROUND_TASKS set to
    # survive past this return.
    # Bulk path — one /validate_bulk_emails submit for all rows in the
    # batch. ~5–10× faster than N single-email calls when a connector
    # drops 50+ emails at once. Single-mode (schedule_for_row) is still
    # the right call for one-cell enrichment hits — those go through
    # enrichment.py.
    if run_id and pending_verify:
        from dsl_worker.chat import email_verify_hook, url_verify_hook
        col_defs = [
            {"name": c["name"], "type": c["type"]}
            for c in normalized_map
        ]
        try:
            email_verify_hook.schedule_bulk_for_rows(
                run_id=run_id,
                rows=pending_verify,
                columns=col_defs,
            )
        except Exception:
            log.exception("email_verify_hook.schedule_bulk_for_rows raised in _commit_rows; suppressed")
        # URL verification: per-URL firecrawl + LLM tool-use judge.
        # The hook short-circuits on `source` for upstream-trusted
        # sources (apify_actor:* etc.) so we don't burn firecrawl
        # credits on URLs the scraper already validated.
        try:
            url_verify_hook.schedule_bulk_for_rows(
                run_id=run_id,
                rows=pending_verify,
                source=source,
            )
        except Exception:
            log.exception("url_verify_hook.schedule_bulk_for_rows raised in _commit_rows; suppressed")
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
) -> Tuple[int, List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Return (matched_count, sample_kept[5], sample_excluded[5]) for the filter
    without materializing it. Both samples let the agent sanity-check that
    the filter kept what it wanted AND excluded what it wanted, in one shot.
    """
    # Naive: pull rows, filter in Python. Good enough for v1 with table sizes up to ~1000.
    all_rows = db.execute(
        sa_text("SELECT row FROM samples WHERE table_id=:tid AND deleted_at IS NULL"),
        {"tid": table_id},
    ).fetchall()
    matched: List[Dict[str, Any]] = []
    excluded: List[Dict[str, Any]] = []
    for (row,) in all_rows:
        if not isinstance(row, dict):
            continue
        v = row.get(column)
        if _match(v, op, value):
            matched.append(row)
        elif len(excluded) < 5:
            excluded.append(row)
    return len(matched), matched[:5], excluded


def _match(cell: Any, op: str, value: Any) -> bool:
    """Python-side predicate matching the SQL semantics in
    routes.py:_filters_to_where_sql. Canonical ops are the 7-op set; legacy
    ops (`contains`, `>=`, `in`, etc.) are also recognized here so old DB
    rows from before the vocabulary shrink keep filtering correctly.
    """
    # Null checks — any column.
    if op in ("is_null", "isnull"):
        return cell is None or cell == ""
    if op in ("is_not_null", "is_not_empty", "exists"):
        return cell is not None and cell != ""

    # Membership / equality.
    if op in ("is_any_of", "in", "any_of"):
        return cell in (value or [])
    if op == "equals":
        return cell == value
    if op == "not_equals":
        return cell != value

    # Numeric / date ranges. Word + symbol forms both recognized.
    if op in (">", "gt"):
        try:
            return float(cell) > float(value)
        except (TypeError, ValueError):
            return False
    if op in ("<", "lt"):
        try:
            return float(cell) < float(value)
        except (TypeError, ValueError):
            return False
    if op in (">=", "gte"):
        try:
            return float(cell) >= float(value)
        except (TypeError, ValueError):
            return False
    if op in ("<=", "lte"):
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

    # Text — canonical text_inc_exc plus legacy single-term ops.
    if op in ("text_inc_exc", "text_include_exclude"):
        if not isinstance(value, dict):
            return False
        include = value.get("include") or []
        exclude = value.get("exclude") or []
        if isinstance(include, str): include = [include]
        if isinstance(exclude, str): exclude = [exclude]
        cell_str = "" if cell is None else str(cell).lower()
        if include:
            if not any(str(t).lower() in cell_str for t in include if t):
                return False
        if exclude:
            if any(str(t).lower() in cell_str for t in exclude if t):
                return False
        return True
    if op in ("contains", "icontains"):
        return bool(value) and isinstance(cell, str) and str(value).lower() in cell.lower()
    if op in ("not_contains", "does_not_contain"):
        return not (bool(value) and isinstance(cell, str) and str(value).lower() in cell.lower())
    if op in ("starts_with", "startswith"):
        return isinstance(cell, str) and cell.lower().startswith(str(value).lower())
    if op in ("ends_with", "endswith"):
        return isinstance(cell, str) and cell.lower().endswith(str(value).lower())

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
