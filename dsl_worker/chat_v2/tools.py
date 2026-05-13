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

from dsl_worker.sources_v2 import get_adapter, list_sources


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
    # Emitter for in-band events (approval cards, progress, etc.). Optional —
    # tools that don't emit can pass None.
    emit_event: Optional[Callable[[str, Dict[str, Any]], None]] = None


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
    """Create a new table backed by one source query and fetch the first batch.

    For predictable sources: applies default column map, commits all rows.
    For unpredictable sources: fetches first ~10 rows as preview, leaves
    table in `pending_mapping` status. Agent must call column_map_set.
    """
    name = args.get("name") or "Untitled"
    source = args.get("source")
    query_params = args.get("query_params") or {}
    n = int(args.get("n") or 100)

    if not source:
        return {"error": "source is required"}, 0.0
    if source.split(":", 1)[0] not in list_sources():
        return {
            "error": f"unknown source {source!r}",
            "available": list_sources(),
        }, 0.0

    adapter = get_adapter(source)
    val_err = adapter.validate_query_params(query_params)
    if val_err:
        return {"error": val_err}, 0.0

    # Insert table row immediately so the agent has a table_id to reference,
    # even if the fetch is slow or fails. Assign a short_id (t1, t2, ...)
    # for LLM-friendly handles.
    table_id = str(uuid.uuid4())
    short_id = _next_short_id(ctx.db, ctx.project_id)
    columns_for_db = []
    if adapter.predictable and adapter.default_columns:
        columns_for_db = [
            {"name": c["column_name"], "type": c["type"], "source_field": c["source_field"]}
            for c in adapter.default_columns
        ]

    ctx.db.execute(
        sa_text(
            """
            INSERT INTO tables (id, project_id, short_id, name, source, query_params, columns,
                                dedup_key_column, fetch_status, created_at)
            VALUES (:id, :project_id, :short_id, :name, :source, :query_params, :columns,
                    :dedup_key_column, :fetch_status, now())
            """
        ),
        {
            "id": table_id,
            "project_id": ctx.project_id,
            "short_id": short_id,
            "name": name,
            "source": source,
            "query_params": json.dumps(query_params),
            "columns": json.dumps(columns_for_db),
            "dedup_key_column": adapter.default_dedup_key_column,
            "fetch_status": "fetching",
        },
    )
    ctx.db.commit()

    # Synchronous first fetch
    fetch_n = 10 if not adapter.predictable else n
    try:
        if source.startswith("apify_actor:"):
            res = await adapter.fetch(
                query_params, fetch_n, prior_cursor=None, source_full=source
            )
        else:
            res = await adapter.fetch(query_params, fetch_n, prior_cursor=None)
    except Exception as e:
        log.exception("table_create fetch failed: %s", e)
        ctx.db.execute(
            sa_text("UPDATE tables SET fetch_status='failed', fetch_error=:err WHERE id=:id"),
            {"id": table_id, "err": str(e)[:500]},
        )
        ctx.db.commit()
        return {"error": f"source fetch failed: {e}"}, 0.0

    # Commit rows (predictable) OR stash in pending state (unpredictable)
    if adapter.predictable:
        _commit_rows(ctx.db, table_id, res.rows, adapter.default_columns)
        new_status = "complete" if res.exhausted else "idle"
    else:
        # Unpredictable: stash raw rows under a holding key in the table state;
        # they'll be committed after column_map_set with the agreed mapping.
        ctx.db.execute(
            sa_text(
                "UPDATE tables SET query_params = jsonb_set(query_params::jsonb, '{_pending_rows}', CAST(:pending AS jsonb)) WHERE id=:id"
            ),
            {"id": table_id, "pending": json.dumps(res.rows)},
        )
        new_status = "pending_mapping"

    ctx.db.execute(
        sa_text(
            """
            UPDATE tables
            SET fetch_status = :status,
                last_fetch_returned_rows = :rows_n,
                last_fetch_cost_credits = :cost,
                last_fetch_at = now(),
                fetch_error = NULL
            WHERE id = :id
            """
        ),
        {
            "id": table_id,
            "status": new_status,
            "rows_n": len(res.rows),
            "cost": res.cost_credits,
        },
    )
    ctx.db.commit()

    # For predictable: kick off background continuation if more rows to fetch
    if adapter.predictable and not res.exhausted and len(res.rows) < n:
        # In a real worker, this would enqueue an async job. For v1, we keep
        # it synchronous-simple — if the agent asked for n=100 and we got 10,
        # the agent can just call table_extend on the next turn. Predictable
        # adapters typically return n on the first call anyway (fast APIs).
        pass

    result = {
        "table_id": short_id,  # LLM-friendly handle; resolved to UUID at tool boundaries
        "name": name,
        "source": source,
        "rows_initial": len(res.rows) if adapter.predictable else 0,
        "fetch_status": new_status,
        "exhausted_first_batch": res.exhausted,
    }
    if not adapter.predictable:
        # Preview for agent to inspect before column_map_set. Old version
        # surfaced an alphabetized field union which biased the agent toward
        # leading-underscore meta keys and made it hallucinate column names
        # that weren't in the rows. New version: frequency-ranked fields
        # with example values, plus a few full sample rows.
        result["source_schema_preview"] = _build_schema_preview(res.rows)
    return result, res.cost_credits * 0.10  # credits -> dollars for cost return


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
    table_id = resolve_table_id(ctx.db, ctx.project_id, args.get("table_id"))
    new_query_params = args.get("query_params") or {}
    n = int(args.get("n") or 100)

    if not table_id:
        return {"error": "table_id is required"}, 0.0

    row = ctx.db.execute(
        sa_text("SELECT source, query_params, columns FROM tables WHERE id=:id AND deleted_at IS NULL"),
        {"id": table_id},
    ).fetchone()
    if not row:
        return {"error": f"table {table_id} not found"}, 0.0
    source, base_params, columns = row[0], row[1], row[2]

    adapter = get_adapter(source)
    # Merge: new_query_params overrides base table query_params field-by-field
    merged = {**(base_params or {}), **(new_query_params or {})}
    val_err = adapter.validate_query_params(merged)
    if val_err:
        return {"error": val_err}, 0.0

    # Read prior cursor from table state if any
    cursor_row = ctx.db.execute(
        sa_text("SELECT (query_params::jsonb)->'_cursor' AS cursor FROM tables WHERE id=:id"),
        {"id": table_id},
    ).fetchone()
    prior_cursor = cursor_row[0] if cursor_row else None

    try:
        if source.startswith("apify_actor:"):
            res = await adapter.fetch(merged, n, prior_cursor=prior_cursor, source_full=source)
        else:
            res = await adapter.fetch(merged, n, prior_cursor=prior_cursor)
    except Exception as e:
        log.exception("table_extend fetch failed: %s", e)
        return {"error": f"source fetch failed: {e}"}, 0.0

    # Commit rows using existing column mapping
    cols = json.loads(columns) if isinstance(columns, str) else (columns or [])
    column_map = [{"source_field": c.get("source_field") or c["name"], "column_name": c["name"], "type": c["type"]} for c in cols]
    _commit_rows(ctx.db, table_id, res.rows, column_map)

    # Update last_fetch_* + cursor
    ctx.db.execute(
        sa_text(
            """
            UPDATE tables
            SET last_fetch_returned_rows = :rows_n,
                last_fetch_cost_credits = :cost,
                last_fetch_at = now(),
                query_params = jsonb_set(query_params::jsonb, '{_cursor}', CAST(:cursor AS jsonb))
            WHERE id = :id
            """
        ),
        {
            "id": table_id,
            "rows_n": len(res.rows),
            "cost": res.cost_credits,
            "cursor": json.dumps(res.cursor) if res.cursor else "null",
        },
    )
    ctx.db.commit()

    return {
        "rows_added": len(res.rows),
        "exhausted": res.exhausted,
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

    # Apply mapping retroactively. If table is in pending_mapping state, commit
    # the stashed rows now under the new mapping.
    table_row = ctx.db.execute(
        sa_text("SELECT fetch_status, query_params FROM tables WHERE id=:id AND deleted_at IS NULL"),
        {"id": table_id},
    ).fetchone()
    if not table_row:
        return {"error": f"table {table_id} not found"}, 0.0

    fetch_status, qp = table_row
    qp = qp or {}
    if isinstance(qp, str):
        qp = json.loads(qp)
    pending_rows = qp.get("_pending_rows") or []

    if fetch_status == "pending_mapping" and pending_rows:
        _commit_rows(ctx.db, table_id, pending_rows, columns_for_db)
        # Strip _pending_rows from query_params
        qp.pop("_pending_rows", None)
        ctx.db.execute(
            sa_text("UPDATE tables SET query_params=CAST(:qp AS jsonb) WHERE id=:id"),
            {"id": table_id, "qp": json.dumps(qp)},
        )

    ctx.db.execute(
        sa_text(
            """
            UPDATE tables
            SET columns = CAST(:cols AS jsonb),
                dedup_key_column = COALESCE(:dedup, dedup_key_column),
                fetch_status = CASE WHEN fetch_status = 'pending_mapping' THEN 'complete' ELSE fetch_status END
            WHERE id = :id
            """
        ),
        {
            "id": table_id,
            "cols": json.dumps(columns_for_db),
            "dedup": dedup_key_column,
        },
    )
    ctx.db.commit()
    return {"ok": True, "columns_committed": len(columns_for_db), "rows_committed_from_pending": len(pending_rows)}, 0.0


# ---------------------------------------------------------------------------
# Tool: table_delete
# ---------------------------------------------------------------------------


async def table_delete(args: Dict[str, Any], ctx: ToolContext) -> Tuple[Dict[str, Any], float]:
    table_id = resolve_table_id(ctx.db, ctx.project_id, args.get("table_id"))
    if not table_id:
        return {"error": "table_id is required"}, 0.0
    ctx.db.execute(
        sa_text("UPDATE tables SET deleted_at=now() WHERE id=:id AND project_id=:pid"),
        {"id": table_id, "pid": ctx.project_id},
    )
    ctx.db.commit()
    return {"ok": True}, 0.0


# ---------------------------------------------------------------------------
# Tool: filter_set / filter_clear
# ---------------------------------------------------------------------------


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


def _commit_rows(db: Session, table_id: str, rows: List[Dict[str, Any]], column_map: List[Dict[str, str]]) -> None:
    """Commit fetched rows into the samples table with column_map applied."""
    if not rows:
        return

    # Pull the current_version for this project to satisfy samples.version_id NOT NULL.
    pid = db.execute(
        sa_text("SELECT project_id FROM tables WHERE id=:id"), {"id": table_id}
    ).scalar()
    version_id = db.execute(
        sa_text("SELECT current_version_id FROM projects WHERE id=:id"),
        {"id": str(pid)},
    ).scalar()
    if not version_id:
        # Create a version row if the project has none yet. Need to fill all
        # legacy NOT NULL columns (use_internet, files_snapshot, examples_snapshot,
        # status, generated_count) so we don't trip schema constraints from the
        # V13-era project_versions table.
        version_id = str(uuid.uuid4())
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
                """
            ),
            {"id": version_id, "pid": str(pid)},
        )
        db.execute(
            sa_text("UPDATE projects SET current_version_id=:vid WHERE id=:pid"),
            {"vid": version_id, "pid": str(pid)},
        )

    # Map source_field → column name. Tolerate either {column_name, source_field}
    # (the adapter default_columns shape) or {name, source_field} (the normalized
    # shape coming out of column_map_set).
    field_to_col = {
        c["source_field"]: (c.get("name") or c.get("column_name"))
        for c in column_map
        if c.get("source_field") and (c.get("name") or c.get("column_name"))
    }

    next_seq_row = db.execute(
        sa_text(
            "SELECT COALESCE(MAX(seq), 0) + 1 FROM samples WHERE version_id=:vid"
        ),
        {"vid": str(version_id)},
    ).scalar()
    next_seq = int(next_seq_row or 1)

    for r in rows:
        mapped = {field_to_col.get(k, k): v for k, v in r.items() if k in field_to_col}
        # For unmapped fields, keep the raw value too — agent can decide later.
        # Actually no — keep clean by ONLY storing mapped fields. Agent can call
        # column_map_set to add more.
        db.execute(
            sa_text(
                "INSERT INTO samples (id, project_id, table_id, version_id, seq, row, created_at) "
                "VALUES (gen_random_uuid(), :pid, :tid, :vid, :seq, CAST(:row AS jsonb), now())"
            ),
            {
                "pid": str(pid),
                "tid": table_id,
                "vid": str(version_id),
                "seq": next_seq,
                "row": json.dumps(mapped),
            },
        )
        next_seq += 1


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
    "row_inspect": row_inspect,
    "row_delete": row_delete,
    # enrichment_set, enrichment_run, code_exec, web_search, suggest_replies,
    # apify_search_actors, apify_actor_details — separate modules (see
    # enrichment.py, web_tools.py, apify_discovery.py).
}
