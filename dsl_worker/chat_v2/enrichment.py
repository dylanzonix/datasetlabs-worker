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
from dsl_worker.chat_v2 import email_verify_hook
from dsl_worker.chat_v2.instrumentation import phase_marker, phase_span, time_commit


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
    # `format` (percent / currency / currency_compact) is passed through when set.
    norm_cols: List[Dict[str, Any]] = []
    for c in columns:
        if isinstance(c, str):
            norm_cols.append({"name": c, "type": "text"})
        elif isinstance(c, dict):
            n = c.get("name") or c.get("column") or c.get("column_name") or c.get("key") or c.get("field")
            if not n:
                return {"error": f"columns entry needs a name (or key/column/field): got {c!r}"}, 0.0
            entry: Dict[str, Any] = {"name": n, "type": c.get("type") or "text"}
            fmt = c.get("format")
            if fmt:
                entry["format"] = fmt
            norm_cols.append(entry)
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

    phase_marker(ctx, "enrichment_run/start", enrichment_id=enrichment_id, scope_type=scope.get("type"))
    stats = await _run_enrichment_on_rows(ctx, table_id, enrichment_id, scope, overwrite)
    phase_marker(ctx, "enrichment_run/done", rows_filled=stats.get("rows_filled"), rows_attempted=stats.get("rows_attempted"))
    cost = stats["total_cost_credits"]

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
        {"eid": enrichment_id, "n": stats["rows_filled"], "cost": cost},
    )
    ctx.db.commit()

    # Rich return: rows_attempted vs rows_filled lets the agent see drop-off;
    # sample_filled_row + sample_not_found_row let it eyeball whether values
    # look right or the prompt is broken (no re-inspect round-trip needed).
    # rows_skipped_missing_deps tells the agent how many rows were skipped
    # because depends_on inputs were empty — usually a sign to run the
    # upstream enrichment first OR expand the filter.
    return {
        "rows_attempted": stats["rows_attempted"],
        "rows_filled": stats["rows_filled"],
        "rows_not_found": stats["rows_not_found"],
        "rows_hit_budget": stats["rows_hit_budget"],
        "rows_errored": stats["rows_errored"],
        "rows_skipped_missing_deps": stats["rows_skipped_missing_deps"],
        "sample_filled_row": stats["sample_filled_row"],
        "sample_not_found_row": stats["sample_not_found_row"],
        "sample_missing_deps_row": stats["sample_missing_deps_row"],
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
) -> Dict[str, Any]:
    """Resolve scope → list of sample rows → run the enrichment action on each.

    Returns a stats dict the orchestrator uses to build its rich tool result:
      rows_attempted, rows_filled, rows_not_found, rows_hit_budget, rows_errored,
      total_cost_credits, sample_filled_row, sample_not_found_row.
    """
    empty_stats = {
        "rows_attempted": 0,
        "rows_filled": 0,
        "rows_not_found": 0,
        "rows_hit_budget": 0,
        "rows_errored": 0,
        "rows_skipped_missing_deps": 0,
        "total_cost_credits": 0.0,
        "sample_filled_row": None,
        "sample_not_found_row": None,
        "sample_missing_deps_row": None,
    }

    enrichment = ctx.db.execute(
        sa_text(
            "SELECT columns, action, per_row_credit_cap FROM enrichments WHERE id=:eid"
        ),
        {"eid": enrichment_id},
    ).fetchone()
    if not enrichment:
        return empty_stats

    columns, action, per_row_cap = enrichment[0], enrichment[1], enrichment[2]
    if isinstance(columns, str):
        columns = json.loads(columns)
    if isinstance(action, str):
        action = json.loads(action)

    # Resolve scope
    with phase_span(ctx, "enrichment_run/resolve_scope"):
        rows = _resolve_scope_rows(ctx.db, table_id, scope, columns, overwrite)
    phase_marker(ctx, "enrichment_run/scope_resolved", rows=len(rows))

    total_cost = 0.0
    filled_count = 0
    not_found_count = 0
    hit_budget_count = 0
    errored_count = 0
    skipped_missing_deps_count = 0
    sample_filled_row: Dict[str, Any] | None = None
    sample_not_found_row: Dict[str, Any] | None = None
    sample_missing_deps_row: Dict[str, Any] | None = None

    if not rows:
        return empty_stats

    target_cols = [c["name"] for c in columns]
    # depends_on: pre-filter rows where any listed input column is empty.
    # These rows skip the cell agent entirely (no credits spent) and get
    # __cell_status__: "missing_dependency" written so the FE shows the
    # right badge. The agent's tool result echoes the count + a sample so
    # it can decide whether to expand the filter or fill the upstream
    # column first.
    depends_on_raw = action.get("depends_on") or []
    depends_on = [c for c in depends_on_raw if isinstance(c, str) and c]

    # Build a map: upstream column name → list of downstream target columns
    # that should have their stale "missing_dependency" status cleared when
    # this upstream column gets filled. Without this, a row that was once
    # blocked on (e.g.) Founder Name keeps showing "Missing inputs" on the
    # Founder Email cell forever, even after Founder Name fills, because
    # the Email cell agent never re-runs on its own. With this, the moment
    # Founder Name lands, the downstream missing_dependency badge clears —
    # the cell goes back to "blank, never tried" and the user can choose
    # to re-run the Email enrichment to actually fill it.
    other_enrichments = ctx.db.execute(
        sa_text(
            "SELECT columns, action FROM enrichments "
            "WHERE table_id=:tid AND id != :eid AND deleted_at IS NULL"
        ),
        {"tid": table_id, "eid": enrichment_id},
    ).fetchall()
    downstream_unblocks: Dict[str, List[str]] = {}
    for other_cols_blob, other_action_blob in other_enrichments:
        try:
            other_cols = other_cols_blob if isinstance(other_cols_blob, list) else json.loads(other_cols_blob or "[]")
            other_action = other_action_blob if isinstance(other_action_blob, dict) else json.loads(other_action_blob or "{}")
        except Exception:
            continue
        other_deps = other_action.get("depends_on") or []
        if not isinstance(other_deps, list):
            continue
        other_target_cols = [c.get("name") for c in other_cols if isinstance(c, dict) and c.get("name")]
        for dep in other_deps:
            if not isinstance(dep, str) or not dep:
                continue
            arr = downstream_unblocks.setdefault(dep, [])
            arr.extend(other_target_cols)

    def _row_missing_deps(row_data: Dict[str, Any]) -> List[str]:
        if not depends_on:
            return []
        return [c for c in depends_on if row_data.get(c) in (None, "", [], {})]

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

    def _emit_many(events: List[Tuple[str, Dict[str, Any]]]) -> None:
        """Persist many events in a single commit. Same best-effort
        semantics as `_emit`."""
        if not run_id or not events:
            return
        try:
            from dsl_worker.chat_api import runs as legacy_runs
            from dsl_api.models import ChatRun
            run_obj = ctx.db.query(ChatRun).filter(ChatRun.id == run_id).first()
            if run_obj is not None:
                legacy_runs.emit_events_batch(ctx.db, run_obj, events)
        except Exception:
            log.debug("emit_many (%d events) failed; continuing", len(events), exc_info=True)

    # Partition rows into ready (will run cell agent) vs missing-deps
    # (will be marked "missing_dependency" without spawning a cell agent).
    # No credits are spent on missing-deps rows.
    with phase_span(ctx, "enrichment_run/deps_partition"):
        ready_rows: List[Tuple[str, Dict[str, Any], Dict[str, Any]]] = []
        deps_missing_rows: List[Tuple[str, Dict[str, Any], List[str]]] = []
        for sid, rd, raw in rows:
            missing = _row_missing_deps(rd)
            if missing:
                deps_missing_rows.append((sid, rd, missing))
            else:
                ready_rows.append((sid, rd, raw))
    phase_marker(
        ctx, "enrichment_run/deps_partitioned",
        ready=len(ready_rows), missing=len(deps_missing_rows),
    )

    # Up-front: tell FE all the cells about to be processed. Renders as
    # "Queued" badges until cell_start fires per row. Only ready rows
    # are queued; missing-deps rows get their terminal state directly.
    _emit("fill_start", {
        "tool_call_id": enrichment_id,
        "enrichment_id": enrichment_id,
        "total": len(ready_rows),
        "columns": target_cols,
        "row_ids": [sid for sid, _, _ in ready_rows],
    })

    # Write the missing_dependency cell status for every row that's
    # missing inputs, and fire cell_filled with status so the FE shows
    # the correct badge. We do this BEFORE any cell agents spawn so the
    # FE renders the skip immediately.
    #
    # This used to be a per-row UPDATE+commit+_emit loop — N rows = N×2
    # DB round-trips, each commit blocking the event loop. With N=100+
    # missing-deps rows that translates into seconds of dead time
    # between fill_start and the first cell_start. Now we do one
    # batched UPDATE + one batched event insert. _commit_rows and the
    # cell pool below stay row-at-a-time because each row has its own
    # cost and real-time progress signalling.
    if deps_missing_rows:
      with phase_span(ctx, "enrichment_run/missing_deps_batch", n=len(deps_missing_rows)):
        ready_total = len(ready_rows)
        update_payload = []
        events_to_emit: List[Tuple[str, Dict[str, Any]]] = []
        for sid, row_data, missing in deps_missing_rows:
            merged = dict(row_data) if isinstance(row_data, dict) else {}
            existing_status = merged.get("__cell_status__") or {}
            if not isinstance(existing_status, dict):
                existing_status = {}
            for cn in target_cols:
                existing_status[cn] = "missing_dependency"
            merged["__cell_status__"] = existing_status
            update_payload.append({"sid": sid, "row": json.dumps(merged)})
            if sample_missing_deps_row is None:
                sample_missing_deps_row = {
                    "missing_columns": missing,
                    "row_preview": {
                        k: v for k, v in (row_data or {}).items()
                        if not k.startswith("__") and k not in target_cols
                    },
                }
            events_to_emit.append((
                "cell_filled",
                {
                    "tool_call_id": enrichment_id,
                    "enrichment_id": enrichment_id,
                    "sample_id": sid,
                    "row_id": sid,
                    "index": 0,
                    "completed": 0,
                    "total": ready_total,
                    "new_fields": None,
                    "status": "missing_dependency",
                    "cell_status": existing_status,
                    "missing_columns": missing,
                    "cost": 0,
                },
            ))

        # One batched UPDATE for every skipped row: a single statement
        # using a VALUES-join, so it goes over the wire once and commits
        # once. (Per-row execute under one commit would still cost N
        # network round-trips inside the transaction.)
        values_sql_parts = []
        params: Dict[str, Any] = {}
        for i, p in enumerate(update_payload):
            values_sql_parts.append(f"(CAST(:sid_{i} AS uuid), :row_{i})")
            params[f"sid_{i}"] = p["sid"]
            params[f"row_{i}"] = p["row"]
        batched_sql = (
            "UPDATE samples AS s "
            "SET row = CAST(u.row AS jsonb) "
            f"FROM (VALUES {', '.join(values_sql_parts)}) AS u(sid, row) "
            "WHERE s.id = u.sid"
        )
        try:
            with time_commit(ctx, "missing_deps_update", threshold_ms=50):
                ctx.db.execute(sa_text(batched_sql), params)
                ctx.db.commit()
        except Exception as e:
            log.warning(
                "dependency-skip batched commit failed (%d rows): %s",
                len(update_payload), e,
            )
            try:
                ctx.db.rollback()
            except Exception:
                pass
        skipped_missing_deps_count += len(deps_missing_rows)
        with phase_span(ctx, "enrichment_run/missing_deps_emit", n=len(events_to_emit)):
            _emit_many(events_to_emit)

    # Concurrency cap for parallel cell ops. Cells beyond this wait at the
    # semaphore — FE shows them as "Queued" until cell_start fires.
    sem = asyncio.Semaphore(25)

    # Index assigned by start order — useful for the toolLog summary.
    start_seq = {"i": 0}
    # The "total" the cell_start events report reflects ready rows only —
    # missing-deps rows are already terminal at this point.
    ready_total = len(ready_rows)

    async def run_one(sample_id: str, row_data: Dict[str, Any], raw_row: Dict[str, Any]):
        async with sem:
            start_seq["i"] += 1
            idx = start_seq["i"]
            _emit("cell_start", {
                "tool_call_id": enrichment_id,
                "enrichment_id": enrichment_id,
                "row_id": sample_id,
                "index": idx,
                "total": ready_total,
                "columns": target_cols,
            })
            new_fields, cost, status = await _execute_action(
                action, row_data, per_row_cap, columns, ctx,
                enrichment_id=enrichment_id, sample_id=sample_id,
                raw_row=raw_row,
            )
            return sample_id, row_data, new_fields, cost, status, idx

    phase_marker(ctx, "enrichment_run/tasks_spawning", n=len(ready_rows))
    tasks = [asyncio.create_task(run_one(sid, rd, raw)) for sid, rd, raw in ready_rows]
    phase_marker(ctx, "enrichment_run/tasks_spawned", n=len(tasks))

    try:
      for fut in asyncio.as_completed(tasks):
        try:
            sample_id, original_row, new_fields, cost, status, idx = await fut
        except asyncio.CancelledError:
            raise
        except Exception as e:
            log.warning("cell op raised: %s", e)
            errored_count += 1
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
        has_real_value = bool(
            isinstance(new_fields, dict)
            and any(new_fields.get(cn) not in (None, "") for cn in target_cols)
        )
        if isinstance(new_fields, dict) and new_fields:
            merged.update(new_fields)
            if has_real_value:
                filled_count += 1
                if sample_filled_row is None:
                    # Sample what the agent actually wrote — only the
                    # enrichment's target columns, not the whole row,
                    # to keep the tool result tight.
                    sample_filled_row = {cn: new_fields.get(cn) for cn in target_cols}
        if status == "hit_budget":
            hit_budget_count += 1
        elif status == "filled" and not has_real_value:
            not_found_count += 1
            if sample_not_found_row is None:
                # For not_found, give the agent a hint about what was on
                # the row so it can judge whether the prompt was wrong or
                # the data genuinely doesn't exist for this kind of input.
                sample_not_found_row = {
                    k: v for k, v in original_row.items()
                    if not k.startswith("__") and k not in target_cols
                }

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
            # Differentiate "filled with a value" from "ran and the answer
            # genuinely doesn't exist". The latter (cell agent returned
            # null because it couldn't find anything) is its own status
            # so the FE can show a "Not found" badge — otherwise the cell
            # looks identical to "haven't tried yet", and the user
            # re-clicks ▶ wondering if anything happened.
            for cn in target_cols:
                v = new_fields.get(cn) if isinstance(new_fields, dict) else None
                if v not in (None, ""):
                    existing_status.pop(cn, None)
                else:
                    existing_status[cn] = "not_found"
        # Auto-clear stale missing_dependency on DOWNSTREAM enrichments.
        # When this run just filled an upstream column, any other
        # enrichment in this table that depends_on it had a stale
        # "missing_dependency" status sitting on its target cells.
        # Clear those so the dependent cell goes back to "blank, retry-able"
        # instead of showing "Missing inputs" indefinitely.
        if isinstance(new_fields, dict) and new_fields and downstream_unblocks:
            for cn, v in new_fields.items():
                if v in (None, ""):
                    continue
                for downstream_cn in downstream_unblocks.get(cn, []):
                    if existing_status.get(downstream_cn) == "missing_dependency":
                        existing_status.pop(downstream_cn, None)
        if existing_status:
            merged["__cell_status__"] = existing_status
        elif "__cell_status__" in merged:
            merged.pop("__cell_status__")

        # Per-cell cost sidecar. Tagged onto each column the cell agent
        # just filled so the FE can render a small "$X" badge under the
        # value (visible at all times, survives reload — cell_filled
        # events only fire live). Same pattern as __cell_status__.
        if isinstance(new_fields, dict) and new_fields:
            existing_cost = merged.get("__cell_cost__") or {}
            if not isinstance(existing_cost, dict):
                existing_cost = {}
            for cn in new_fields.keys():
                existing_cost[cn] = float(cost)
            merged["__cell_cost__"] = existing_cost

        if merged != original_row:
            try:
                ctx.db.execute(
                    sa_text("UPDATE samples SET row=CAST(:row AS jsonb) WHERE id=:sid"),
                    {"row": json.dumps(merged), "sid": sample_id},
                )
                ctx.db.commit()
                # Schedule Scrubby verify for any email values just
                # written. Tasks are pinned in email_verify_hook so they
                # outlive this function; they emit email_verifying /
                # email_verified / row_merged via fresh DB sessions as
                # each verdict lands. Fire-and-forget — awaiting here
                # would block cell_filled on Scrubby's 15-60s round trip.
                if isinstance(new_fields, dict) and new_fields:
                    try:
                        email_verify_hook.schedule_for_row(
                            run_id=getattr(ctx, "run_id", None),
                            sample_id=sample_id,
                            written_values=new_fields,
                            columns=columns,
                        )
                    except Exception:
                        log.exception("email_verify_hook.schedule_for_row raised; suppressed")
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
            "total": ready_total,
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

    return {
        # rows_attempted = rows the cell agent actually tried (excludes skipped)
        "rows_attempted": ready_total,
        "rows_filled": filled_count,
        "rows_not_found": not_found_count,
        "rows_hit_budget": hit_budget_count,
        "rows_errored": errored_count,
        "rows_skipped_missing_deps": skipped_missing_deps_count,
        "total_cost_credits": total_cost,
        "sample_filled_row": sample_filled_row,
        "sample_not_found_row": sample_not_found_row,
        "sample_missing_deps_row": sample_missing_deps_row,
    }


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

    Scope types:
      row_ids:      {type, row_ids: [...]}
      first_n:      {type, first_n: 10}
      filtered:     {type, filters: [{column, op, value}, ...]} — same
                    {column, op, value} shape as filter_set. Filters are
                    AND'd together, evaluated in Python via the same _match
                    semantics used everywhere else (no hidden state — the
                    scope's filters are self-contained, NOT a reference to
                    whatever table_filters happens to have at exec time).
      all_unfilled: {type} — every row missing at least one target column.
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
    elif scope_type == "filtered":
        # Explicit-filter scope. Read the table, apply each {column, op,
        # value} in Python via _match. We could push this to SQL via
        # _filters_to_where_sql but Python is fast enough at v1 table
        # sizes and keeps the canonical predicate (used for filter_set's
        # preview) as the single source of truth for match semantics.
        from dsl_worker.chat_v2.tools import _match, _normalize_filter
        raw_filters = scope.get("filters") or []
        if not isinstance(raw_filters, list) or not raw_filters:
            # Empty filters → match every row (caller intended "filtered
            # but no constraints"); still better than silently falling
            # through to all_unfilled.
            rows = db.execute(sa_text(base_sql + " ORDER BY seq"), params).fetchall()
        else:
            # Normalize each filter into the canonical (op, value) pair
            # so we evaluate the SAME way filter_set's preview does.
            norm_filters: List[Tuple[str, str, Any]] = []
            for f in raw_filters:
                if not isinstance(f, dict):
                    continue
                col = f.get("column") or f.get("column_name") or f.get("field")
                if not col:
                    continue
                normalized = _normalize_filter(f.get("op") or f.get("operator"), f.get("value"))
                if normalized is None:
                    continue
                op, value = normalized
                norm_filters.append((col, op, value))
            all_rows = db.execute(sa_text(base_sql + " ORDER BY seq"), params).fetchall()
            rows = []
            for sid, row_data, raw_row in all_rows:
                if not isinstance(row_data, dict):
                    continue
                keep = all(
                    _match(row_data.get(c), op, v) for c, op, v in norm_filters
                )
                if keep:
                    rows.append((sid, row_data, raw_row))
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
