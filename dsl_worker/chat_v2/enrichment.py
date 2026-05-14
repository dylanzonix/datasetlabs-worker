"""Enrichment tools: enrichment_set + enrichment_run.

enrichment_set defines or updates an enrichment AND runs it on the first 10
unfilled rows. Refining = call again with same enrichment_id and revised action.

enrichment_run extends to more rows. Default skips already-filled cells
(Clay-style). Approval-gated for scope > 10 rows in the orchestrator's
streaming loop.

Two action shapes (stored in enrichments.action JSONB):

  Deterministic: {type: "tool", tool: <name>, args_template: {...}, output_map: {...}}
    Server calls the integration tool directly per row, maps result fields to
    columns. No LLM-per-cell — predictable cost.

  Cell agent: {type: "cell_agent", prompt: "...", columns_to_fill: [...],
               per_row_credit_cap: int}
    Spawns a mini-LLM per row with the cell-agent toolset (see cell_agent.py).
    Bounded by per_row_credit_cap.
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy import text as sa_text
from sqlalchemy.orm import Session

from dsl_worker.chat_v2.tools import (
    ToolContext,
    resolve_enrichment_id,
    resolve_table_id,
    _next_enrichment_short_id,
)


log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# enrichment_set
# ---------------------------------------------------------------------------


async def enrichment_set(args: Dict[str, Any], ctx: ToolContext) -> Tuple[Dict[str, Any], float]:
    table_id = resolve_table_id(ctx.db, ctx.project_id, args.get("table_id"))
    raw_enrichment = args.get("enrichment_id")
    enrichment_id = resolve_enrichment_id(ctx.db, ctx.project_id, raw_enrichment) if raw_enrichment else None
    name = args.get("name") or "Enrichment"
    columns = args.get("columns") or []
    action = args.get("action") or {}
    per_row_credit_cap = int(args.get("per_row_credit_cap") or 5)

    if not table_id:
        return {"error": "table_id is required"}, 0.0
    if not action or "type" not in action:
        return {"error": "action with type=tool|cell_agent is required"}, 0.0
    if not columns:
        return {"error": "columns is required (at least one column to fill)"}, 0.0

    # Normalize columns: accept the various shapes the agent reaches for.
    # Supported: bare strings, {name}, {name, type}, {key, label}, {column, type}.
    norm_cols: List[Dict[str, str]] = []
    for c in columns:
        if isinstance(c, str):
            norm_cols.append({"name": c, "type": "text"})
        elif isinstance(c, dict):
            n = c.get("name") or c.get("column") or c.get("column_name") or c.get("key") or c.get("field")
            if not n:
                return {"error": f"columns entry needs a name (or key/column/field): got {c!r}"}, 0.0
            norm_cols.append({"name": n, "type": c.get("type") or "text"})
        else:
            return {"error": f"columns entries must be 'col_name' or {{name, type}}; got: {c!r}"}, 0.0
    columns = norm_cols

    db = ctx.db
    is_refinement = enrichment_id is not None

    if is_refinement:
        existing = db.execute(
            sa_text("SELECT id FROM enrichments WHERE id=:eid AND deleted_at IS NULL"),
            {"eid": enrichment_id},
        ).fetchone()
        if not existing:
            return {"error": f"enrichment {enrichment_id} not found"}, 0.0
        db.execute(
            sa_text(
                """
                UPDATE enrichments
                SET name = :name,
                    columns = CAST(:cols AS jsonb),
                    action = CAST(:action AS jsonb),
                    per_row_credit_cap = :cap
                WHERE id = :eid
                """
            ),
            {
                "eid": enrichment_id,
                "name": name,
                "cols": json.dumps(columns),
                "action": json.dumps(action),
                "cap": per_row_credit_cap,
            },
        )
    else:
        enrichment_id = str(uuid.uuid4())
        short_id = _next_enrichment_short_id(db, table_id)
        db.execute(
            sa_text(
                """
                INSERT INTO enrichments (id, table_id, short_id, name, columns, action, per_row_credit_cap, created_at)
                VALUES (:eid, :tid, :sid, :name, CAST(:cols AS jsonb), CAST(:action AS jsonb), :cap, now())
                """
            ),
            {
                "eid": enrichment_id,
                "tid": table_id,
                "sid": short_id,
                "name": name,
                "cols": json.dumps(columns),
                "action": json.dumps(action),
                "cap": per_row_credit_cap,
            },
        )
        # Add the enrichment columns to the table's column list if not already present.
        _ensure_columns_on_table(db, table_id, columns)
    db.commit()

    # Run on the first 10 unfilled rows
    rows_filled, cost = await _run_enrichment_on_rows(
        ctx, table_id, enrichment_id, scope={"type": "first_n", "first_n": 10}, overwrite=is_refinement
    )

    # Update last_run_* on the enrichment
    db.execute(
        sa_text(
            """
            UPDATE enrichments
            SET last_run_filled_rows = :n,
                last_run_cost_credits = :cost,
                last_run_at = now()
            WHERE id = :eid
            """
        ),
        {"eid": enrichment_id, "n": rows_filled, "cost": cost},
    )
    db.commit()

    # Return preview rows so agent can inspect
    preview = db.execute(
        sa_text(
            "SELECT row FROM samples WHERE table_id=:tid AND deleted_at IS NULL ORDER BY seq LIMIT 10"
        ),
        {"tid": table_id},
    ).fetchall()

    # Return short_id so the agent can reference this enrichment in subsequent calls.
    public_eid = ctx.db.execute(
        sa_text("SELECT short_id FROM enrichments WHERE id=:eid"),
        {"eid": enrichment_id},
    ).scalar() or enrichment_id

    return {
        "enrichment_id": public_eid,
        "rows_filled": rows_filled,
        "results_preview": [r[0] for r in preview],
    }, cost * 0.10


# ---------------------------------------------------------------------------
# enrichment_run
# ---------------------------------------------------------------------------


async def enrichment_run(args: Dict[str, Any], ctx: ToolContext) -> Tuple[Dict[str, Any], float]:
    enrichment_id = resolve_enrichment_id(ctx.db, ctx.project_id, args.get("enrichment_id"))
    scope = args.get("scope") or {"type": "all_unfilled"}
    overwrite = bool(args.get("overwrite", False))
    if not enrichment_id:
        return {"error": "enrichment_id is required"}, 0.0

    row = ctx.db.execute(
        sa_text("SELECT table_id FROM enrichments WHERE id=:eid AND deleted_at IS NULL"),
        {"eid": enrichment_id},
    ).fetchone()
    if not row:
        return {"error": f"enrichment {enrichment_id} not found"}, 0.0
    table_id = str(row[0])

    rows_filled, cost = await _run_enrichment_on_rows(ctx, table_id, enrichment_id, scope, overwrite)

    ctx.db.execute(
        sa_text(
            """
            UPDATE enrichments
            SET last_run_filled_rows = :n,
                last_run_cost_credits = :cost,
                last_run_at = now()
            WHERE id = :eid
            """
        ),
        {"eid": enrichment_id, "n": rows_filled, "cost": cost},
    )
    ctx.db.commit()

    return {
        "rows_queued": rows_filled,
        "rows_filled": rows_filled,
    }, cost * 0.10


# ---------------------------------------------------------------------------
# Internal — execute the enrichment over a scope
# ---------------------------------------------------------------------------


async def _run_enrichment_on_rows(
    ctx: ToolContext,
    table_id: str,
    enrichment_id: str,
    scope: Dict[str, Any],
    overwrite: bool = False,
) -> Tuple[int, float]:
    """Resolve scope → list of sample rows → run the enrichment action on each."""
    enrichment = ctx.db.execute(
        sa_text(
            "SELECT columns, action, per_row_credit_cap FROM enrichments WHERE id=:eid"
        ),
        {"eid": enrichment_id},
    ).fetchone()
    if not enrichment:
        return 0, 0.0

    columns, action, per_row_cap = enrichment[0], enrichment[1], enrichment[2]
    if isinstance(columns, str):
        columns = json.loads(columns)
    if isinstance(action, str):
        action = json.loads(action)

    # Resolve scope
    rows = _resolve_scope_rows(ctx.db, table_id, scope, columns, overwrite)

    total_cost = 0.0
    filled_count = 0

    # Concurrency cap for parallel cell ops
    sem = asyncio.Semaphore(4)

    async def run_one(sample_id: str, row_data: Dict[str, Any]) -> Tuple[Dict[str, Any], float]:
        async with sem:
            return await _execute_action(action, row_data, per_row_cap, columns, ctx)

    if not rows:
        return 0, 0.0

    results = await asyncio.gather(
        *[run_one(sid, rd) for sid, rd in rows], return_exceptions=True
    )

    # Apply outputs back to samples.row
    for (sample_id, original_row), result in zip(rows, results):
        if isinstance(result, Exception):
            log.warning("cell op raised: %s", result)
            continue
        new_fields, cost = result
        total_cost += cost
        if not new_fields:
            continue
        # Defensive: psycopg2 occasionally returns a JSONB column as a
        # JSON string instead of a dict (postgres adapter quirk on
        # certain Python versions / connection states). And new_fields
        # ought to be a dict but some cell-agent paths have returned a
        # bare string. Coerce both so a single misbehaving row doesn't
        # crash the whole batch.
        if isinstance(original_row, str):
            try:
                original_row = json.loads(original_row)
            except Exception:
                original_row = {}
        if not isinstance(original_row, dict):
            original_row = {}
        if not isinstance(new_fields, dict):
            log.warning("cell op returned non-dict new_fields: %r", new_fields)
            continue
        merged = {**original_row, **new_fields}
        ctx.db.execute(
            sa_text("UPDATE samples SET row=CAST(:row AS jsonb) WHERE id=:sid"),
            {"row": json.dumps(merged), "sid": sample_id},
        )
        filled_count += 1

    ctx.db.commit()
    return filled_count, total_cost


def _resolve_scope_rows(
    db: Session,
    table_id: str,
    scope: Dict[str, Any],
    enrichment_columns: List[Dict[str, str]],
    overwrite: bool,
) -> List[Tuple[str, Dict[str, Any]]]:
    """Return [(sample_id, row_dict), ...] matching the scope."""
    base_sql = "SELECT id::text, row FROM samples WHERE table_id=:tid AND deleted_at IS NULL"
    params: Dict[str, Any] = {"tid": table_id}
    scope_type = scope.get("type", "all_unfilled")

    if scope_type == "row_ids":
        ids = scope.get("row_ids") or []
        if not ids:
            return []
        rows = db.execute(
            sa_text(base_sql + " AND id::text = ANY(:ids)"),
            {**params, "ids": ids},
        ).fetchall()
    elif scope_type == "first_n":
        n = int(scope.get("first_n") or 10)
        rows = db.execute(
            sa_text(base_sql + " ORDER BY seq LIMIT :n"),
            {**params, "n": n},
        ).fetchall()
    else:  # all_unfilled
        rows = db.execute(
            sa_text(base_sql + " ORDER BY seq"),
            params,
        ).fetchall()

    # Filter to unfilled (unless overwrite) — a row is "unfilled" if any of the
    # enrichment's target columns has no value.
    target_cols = [c["name"] for c in enrichment_columns]
    out: List[Tuple[str, Dict[str, Any]]] = []
    for sid, row_data in rows:
        if not isinstance(row_data, dict):
            row_data = {}
        if overwrite:
            out.append((sid, row_data))
        else:
            any_unfilled = any(not row_data.get(c) for c in target_cols)
            if any_unfilled:
                out.append((sid, row_data))
    return out


async def _execute_action(
    action: Dict[str, Any],
    row_data: Dict[str, Any],
    per_row_cap: int,
    columns: List[Dict[str, str]],
    ctx: ToolContext,
) -> Tuple[Dict[str, Any], float]:
    """Run one enrichment action against one row. Returns (new_fields_dict, cost_credits)."""
    action_type = action.get("type")
    if action_type == "tool":
        return await _execute_tool_action(action, row_data, columns, ctx)
    if action_type == "cell_agent":
        from dsl_worker.chat_v2.cell_agent import run_cell_agent
        return await run_cell_agent(action, row_data, per_row_cap, columns, ctx)
    return {}, 0.0


async def _execute_tool_action(
    action: Dict[str, Any],
    row_data: Dict[str, Any],
    columns: List[Dict[str, str]],
    ctx: ToolContext,
) -> Tuple[Dict[str, Any], float]:
    """Deterministic single-tool-call per row. Template args from row data,
    map result fields to columns."""
    tool_name = action.get("tool")
    args_template = action.get("args_template") or {}
    output_map = action.get("output_map") or {}

    # Template substitution: replace {row.field} with row_data.field
    args = _template_args(args_template, row_data)

    # Dispatch through the cell agent's tool registry (same toolset)
    from dsl_worker.chat_v2.cell_agent import CELL_TOOL_HANDLERS
    handler = CELL_TOOL_HANDLERS.get(tool_name)
    if not handler:
        return {}, 0.0

    try:
        result, cost = await handler(args, ctx)
    except Exception as e:
        log.exception("deterministic tool action %s failed: %s", tool_name, e)
        return {}, 0.0

    # Map result fields → column names
    new_fields: Dict[str, Any] = {}
    for src_field, col_name in output_map.items():
        if src_field in result:
            new_fields[col_name] = result[src_field]
    return new_fields, cost


def _template_args(template: Dict[str, Any], row: Dict[str, Any]) -> Dict[str, Any]:
    """Replace `{row.field}` in any string template value with row[field]."""
    out: Dict[str, Any] = {}
    for k, v in template.items():
        if isinstance(v, str) and "{row." in v:
            # Naive substitution — replace each {row.foo} token
            new_v = v
            for field, val in row.items():
                token = "{row." + field + "}"
                if token in new_v:
                    new_v = new_v.replace(token, str(val) if val is not None else "")
            out[k] = new_v
        else:
            out[k] = v
    return out


def _ensure_columns_on_table(db: Session, table_id: str, enrichment_columns: List[Dict[str, str]]) -> None:
    """Append enrichment columns to the table's columns array if not present."""
    row = db.execute(
        sa_text("SELECT columns FROM tables WHERE id=:tid"),
        {"tid": table_id},
    ).fetchone()
    if not row:
        return
    existing = row[0] if isinstance(row[0], list) else json.loads(row[0] or "[]")
    existing_names = {c["name"] for c in existing}
    to_add = [c for c in enrichment_columns if c["name"] not in existing_names]
    if not to_add:
        return
    new_cols = existing + to_add
    db.execute(
        sa_text("UPDATE tables SET columns=CAST(:c AS jsonb) WHERE id=:tid"),
        {"tid": table_id, "c": json.dumps(new_cols)},
    )


HANDLERS = {
    "enrichment_set": enrichment_set,
    "enrichment_run": enrichment_run,
}
