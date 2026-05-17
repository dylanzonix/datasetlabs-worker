"""Enrichment tools: enrichment_set + enrichment_run.

enrichment_set defines or updates an enrichment. Does NOT auto-run anything;
the caller must invoke enrichment_run separately. Refining = call again
with same enrichment_id and revised action.

enrichment_run runs the enrichment over a scope of rows. Approval-gated
in the orchestrator's streaming loop (the agent's tool call pauses until
the user approves via the chat-input approval card).

Action shape (stored in enrichments.action JSONB):

  {
    research: "fast" | "smart" | "expert" | "standard" | "deep",
    prompt: "...",
    columns_to_fill: [...],          // optional; defaults to all enrichment columns
    per_row_credit_cap: float        // optional; defaults from research level
  }

Per-row execution always spawns a cell agent (see cell_agent.py). The old
deterministic `type: "tool"` shape was removed — small models are cheap
enough that we always wrap a tool call in an agent loop, which is more
resilient to source variance.
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
from dsl_worker.chat_v2 import verify_hook


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
    # Top-level per_row_credit_cap is the canonical place; action.per_row_credit_cap
    # works too. None means "let the research level pick its default" — do NOT
    # coerce to a fixed number here; the per-tier defaults handle that downstream.
    cap_in = args.get("per_row_credit_cap")
    if cap_in is None and isinstance(action, dict):
        cap_in = action.get("per_row_credit_cap")
    per_row_credit_cap: Optional[float] = float(cap_in) if cap_in not in (None, "") else None

    if not table_id:
        return {"error": "table_id is required"}, 0.0
    if not isinstance(action, dict) or not action:
        return {"error": "action is required (with at least `research` and `prompt`)"}, 0.0
    if not columns:
        return {"error": "columns is required (at least one column to fill)"}, 0.0
    # Warn (don't block) on the dropped deterministic shape and on the
    # renamed `tier` field — back-compat aliasing happens downstream.
    if action.get("type") == "tool":
        log.warning(
            "enrichment_set: deprecated action.type='tool' (deterministic shape removed) — coercing to cell_agent")
    if action.get("type") and action.get("type") not in ("cell_agent", "tool"):
        log.warning("enrichment_set: unknown action.type=%r — ignoring", action.get("type"))
    if "tier" in action and "research" not in action:
        log.info("enrichment_set: legacy 'tier' field used — auto-aliasing to 'research'")

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
        _ensure_columns_on_table(db, table_id, columns, enrichment_id=enrichment_id)
    db.commit()

    # Seed an agent comment per column the enrichment fills. Only on first
    # creation — refinements append commentary explicitly via comment_on_column.
    if not is_refinement:
        from dsl_worker.chat_v2.comments import seed_column_comment
        body = _format_enrichment_seed_body(name, action)
        for col in columns:
            cname = col.get("name") if isinstance(col, dict) else None
            if cname:
                seed_column_comment(db, ctx.project_id, table_id, cname, body)

    # NOTE: enrichment_set NO LONGER auto-runs. Caller must invoke
    # enrichment_run separately (approval-gated). This change was made to
    # keep cost predictable and put a confirmation step in front of any
    # spend.

    public_eid = ctx.db.execute(
        sa_text("SELECT short_id FROM enrichments WHERE id=:eid"),
        {"eid": enrichment_id},
    ).scalar() or enrichment_id

    return {
        "enrichment_id": public_eid,
        "status": "configured",
        "note": "Enrichment defined but not run. Call enrichment_run to fill rows.",
    }, 0.0


def _format_enrichment_seed_body(name: str, action: Dict[str, Any]) -> str:
    """One-paragraph description of an enrichment for the column comment thread."""
    prompt = (action or {}).get("prompt", "").strip()
    research = (action or {}).get("research") or (action or {}).get("tier") or ""
    if len(prompt) > 400:
        prompt = prompt[:400] + "…"
    header = f"**Enrichment:** {name}"
    if research:
        header += f" _(research: {research})_"
    if prompt:
        return f"{header}\n\n> {prompt}"
    return header


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

    if not rows:
        return 0, 0.0

    target_cols = [c["name"] for c in columns]
    total = len(rows)
    completed = 0

    # Helper to emit events to the chat run (best-effort, never raises).
    run_id = getattr(ctx, "run_id", None)
    def _emit(event_type: str, payload: Dict[str, Any]) -> None:
        if not run_id:
            return
        try:
            from dsl_worker.chat_api import runs as legacy_runs
            from dsl_api.models import ChatRun
            run_obj = ctx.db.query(ChatRun).filter(ChatRun.id == run_id).first()
            if run_obj is not None:
                legacy_runs.emit_event(ctx.db, run_obj, event_type, payload)
        except Exception:
            log.debug("emit %s failed; continuing", event_type, exc_info=True)

    # Up-front: tell FE all the cells about to be processed. Renders as
    # "Queued" badges until cell_start fires per row.
    _emit("fill_start", {
        "tool_call_id": enrichment_id,
        "enrichment_id": enrichment_id,
        "total": total,
        "columns": target_cols,
        "row_ids": [sid for sid, _, _ in rows],
    })

    # Concurrency cap for parallel cell ops. Cells beyond this wait at the
    # semaphore — FE shows them as "Queued" until cell_start fires.
    sem = asyncio.Semaphore(25)

    # Index assigned by start order — useful for the toolLog summary.
    start_seq = {"i": 0}

    async def run_one(sample_id: str, row_data: Dict[str, Any], raw_row: Dict[str, Any]):
        async with sem:
            start_seq["i"] += 1
            idx = start_seq["i"]
            _emit("cell_start", {
                "tool_call_id": enrichment_id,
                "enrichment_id": enrichment_id,
                "row_id": sample_id,
                "index": idx,
                "total": total,
                "columns": target_cols,
            })
            new_fields, cost, status = await _execute_action(
                action, row_data, per_row_cap, columns, ctx,
                enrichment_id=enrichment_id, sample_id=sample_id,
                raw_row=raw_row,
            )
            return sample_id, row_data, new_fields, cost, status, idx

    tasks = [asyncio.create_task(run_one(sid, rd, raw)) for sid, rd, raw in rows]
    # Verify tasks spawned on row commits — pinned in
    # verify_hook._BACKGROUND_TASKS so they survive past this function
    # without being GC'd. We collect handles here only so the
    # CancelledError path can drop the reference cleanly.
    pending_verifications: List[asyncio.Task] = []

    try:
      for fut in asyncio.as_completed(tasks):
        try:
            sample_id, original_row, new_fields, cost, status, idx = await fut
        except asyncio.CancelledError:
            raise
        except Exception as e:
            log.warning("cell op raised: %s", e)
            completed += 1
            continue

        total_cost += cost
        completed += 1

        if isinstance(original_row, str):
            try:
                original_row = json.loads(original_row)
            except Exception:
                original_row = {}
        if not isinstance(original_row, dict):
            original_row = {}

        # Merge new values + cell_status sidecar in one write. We treat
        # `__cell_status__` as a reserved row key: a dict of column → status.
        # Statuses we write: "hit_budget" only (filled is implicit when the
        # value is present; error leaves the cell untouched, no status).
        merged = dict(original_row)
        if isinstance(new_fields, dict) and new_fields:
            merged.update(new_fields)
            filled_count += 1

        existing_status = merged.get("__cell_status__") or {}
        if not isinstance(existing_status, dict):
            existing_status = {}
        if status == "hit_budget":
            # Mark every target column that didn't get a value — that's
            # what hit the cap.
            for cn in target_cols:
                if not (isinstance(new_fields, dict) and new_fields.get(cn)):
                    existing_status[cn] = "hit_budget"
        elif status == "filled" and isinstance(new_fields, dict):
            # Clear stale status for any column we just filled.
            for cn, v in new_fields.items():
                if v not in (None, "") and cn in existing_status:
                    existing_status.pop(cn, None)
        if existing_status:
            merged["__cell_status__"] = existing_status
        elif "__cell_status__" in merged:
            merged.pop("__cell_status__")

        if merged != original_row:
            try:
                ctx.db.execute(
                    sa_text("UPDATE samples SET row=CAST(:row AS jsonb) WHERE id=:sid"),
                    {"row": json.dumps(merged), "sid": sample_id},
                )
                ctx.db.commit()
                # Fire email + URL verifications for the just-written
                # values. Tasks are pinned in verify_hook so they outlive
                # this function — they emit url_verifying / url_verified
                # / row_merged via fresh sessions as each verdict lands.
                if isinstance(new_fields, dict) and new_fields:
                    try:
                        pending_verifications.extend(
                            verify_hook.schedule_for_row(
                                run_id=getattr(ctx, "run_id", None),
                                sample_id=sample_id,
                                written_values=new_fields,
                                columns=columns,
                                row_snapshot=merged,
                            )
                        )
                    except Exception:
                        log.exception("verify_hook.schedule_for_row raised; suppressed")
            except Exception as e:
                log.warning("enrichment row commit failed for %s: %s", sample_id, e)
                ctx.db.rollback()

        # Terminal cell event: carries the new value + status + cell_status
        # sidecar. FE uses this to (a) merge the value into the row in
        # place, and (b) clear the queued/processing badge.
        _emit("cell_filled", {
            "tool_call_id": enrichment_id,
            "enrichment_id": enrichment_id,
            "sample_id": sample_id,
            "row_id": sample_id,
            "index": idx,
            "completed": completed,
            "total": total,
            "new_fields": new_fields if isinstance(new_fields, dict) else None,
            "status": status,
            "cell_status": existing_status or None,
            "cost": cost,
        })
    except asyncio.CancelledError:
        # User cancelled mid-fill. Cancel any still-running cell
        # tasks so they don't keep racking up external API spend
        # (apollo / fullenrich / browser_use calls per cell). The
        # cost already accumulated in total_cost (cells that finished
        # before the cancel landed) gets surfaced via ctx.partial_cost_usd
        # so agent.py's CancelledError handler attributes it to the
        # turn ledger. asyncio.shield around the gather prevents
        # the cleanup itself from being cancelled mid-shutdown.
        for t in tasks:
            if not t.done():
                t.cancel()
        try:
            await asyncio.shield(asyncio.gather(*tasks, return_exceptions=True))
        except Exception:
            pass
        try:
            ctx.partial_cost_usd += float(total_cost)
        except Exception:
            pass
        raise

    # Verify tasks are intentionally NOT awaited — they're pinned in
    # verify_hook._BACKGROUND_TASKS so they survive past this function
    # without being GC'd, and they emit url_verified / row_merged events
    # via fresh DB sessions as each verdict lands. Awaiting here would
    # block the enrichment tool's return on ~3s of HTTP fetches per row
    # plus the Haiku batch round, defeating the point of streaming the
    # cell_filled events.
    del pending_verifications

    return filled_count, total_cost


def _resolve_scope_rows(
    db: Session,
    table_id: str,
    scope: Dict[str, Any],
    enrichment_columns: List[Dict[str, str]],
    overwrite: bool,
) -> List[Tuple[str, Dict[str, Any], Dict[str, Any]]]:
    """Return [(sample_id, row_dict, raw_row_dict), ...] matching the scope.

    raw_row is the unmapped source payload — cell agent sees both so it
    has the same context the orchestrator had at column_map_set time.
    """
    base_sql = "SELECT id::text, row, raw_row FROM samples WHERE table_id=:tid AND deleted_at IS NULL"
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
    out: List[Tuple[str, Dict[str, Any], Dict[str, Any]]] = []
    for sid, row_data, raw_row in rows:
        if not isinstance(row_data, dict):
            row_data = {}
        if not isinstance(raw_row, dict):
            raw_row = {}
        if overwrite:
            out.append((sid, row_data, raw_row))
        else:
            any_unfilled = any(not row_data.get(c) for c in target_cols)
            if any_unfilled:
                out.append((sid, row_data, raw_row))
    return out


async def _execute_action(
    action: Dict[str, Any],
    row_data: Dict[str, Any],
    per_row_cap: Optional[float],
    columns: List[Dict[str, str]],
    ctx: ToolContext,
    *,
    enrichment_id: Optional[str] = None,
    sample_id: Optional[str] = None,
    raw_row: Optional[Dict[str, Any]] = None,
) -> Tuple[Dict[str, Any], float, str]:
    """Run one enrichment action against one row.
    Returns (new_fields_dict, cost_credits, status).
    """
    from dsl_worker.chat_v2.cell_agent import run_cell_agent
    return await run_cell_agent(
        action, row_data, per_row_cap, columns, ctx,
        enrichment_id=enrichment_id, sample_id=sample_id,
        raw_row=raw_row,
    )


def _ensure_columns_on_table(
    db: Session,
    table_id: str,
    enrichment_columns: List[Dict[str, str]],
    enrichment_id: Optional[str] = None,
) -> None:
    """Append enrichment columns to the table's columns array if not present.

    Each appended column carries `enrichment_id` so the FE can render
    grouped headers + per-cell rerun buttons for enrichment columns.
    """
    row = db.execute(
        sa_text("SELECT columns FROM tables WHERE id=:tid"),
        {"tid": table_id},
    ).fetchone()
    if not row:
        return
    existing = row[0] if isinstance(row[0], list) else json.loads(row[0] or "[]")
    existing_names = {c["name"] for c in existing}
    to_add = []
    for c in enrichment_columns:
        if c["name"] in existing_names:
            continue
        cnew = dict(c)
        if enrichment_id:
            cnew["enrichment_id"] = enrichment_id
        to_add.append(cnew)
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
