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

from dsl_worker.chat.tools import (
    ToolContext,
    resolve_enrichment_id,
    resolve_table_id,
    _next_enrichment_short_id,
    _resolve_enrichment_position,
)
from dsl_worker.chat import email_verify_hook
from dsl_worker.chat.instrumentation import phase_marker, phase_span, time_commit


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
        # Resolve insert_before to a short_id (agent may have used 'e1',
        # 't1e1', or a uuid — normalize through resolve_enrichment_id).
        insert_before_raw = args.get("insert_before")
        insert_before_sid: Optional[str] = None
        if insert_before_raw:
            ib_uuid = resolve_enrichment_id(db, ctx.project_id, insert_before_raw)
            if ib_uuid:
                ib_row = db.execute(
                    sa_text("SELECT short_id FROM enrichments WHERE id=:eid"),
                    {"eid": ib_uuid},
                ).fetchone()
                insert_before_sid = ib_row[0] if ib_row else None
        position = _resolve_enrichment_position(
            db, table_id, insert_before=insert_before_sid,
        )
        db.execute(
            sa_text(
                """
                INSERT INTO enrichments (id, table_id, short_id, name, columns, action, per_row_credit_cap, position, created_at)
                VALUES (:eid, :tid, :sid, :name, CAST(:cols AS jsonb), CAST(:action AS jsonb), :cap, :pos, now())
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
                "pos": position,
            },
        )
        # Add the enrichment columns to the table's column list if not already present.
        _ensure_columns_on_table(db, table_id, columns, enrichment_id=enrichment_id)
    db.commit()

    # Seed an agent comment per column the enrichment fills. Only on first
    # creation — refinements append commentary explicitly via comment_on_column.
    if not is_refinement:
        from dsl_worker.chat.comments import seed_column_comment
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

    # Emit an enrichment_card_added SSE event so the chat sidebar can
    # render an "Enrichment created" chip inline next to the assistant
    # message — same pattern as table_card_added. Only on NEW creation;
    # refinements would spam the chip list every time the agent tweaked
    # the prompt or cap.
    if ctx.run_id is not None and not is_refinement:
        try:
            from dsl_worker.chat import run_state
            from dsl_api.models import ChatRun
            run_obj = ctx.db.query(ChatRun).filter(ChatRun.id == ctx.run_id).first()
            if run_obj is not None:
                _trow = ctx.db.execute(
                    sa_text("SELECT short_id, name FROM tables WHERE id=:tid"),
                    {"tid": table_id},
                ).first()
                table_short = (_trow[0] if _trow else None) or table_id
                table_name_val = (_trow[1] if _trow else None) or ""
                research_tier = action.get("research") or action.get("tier") or None
                prompt_text = (action.get("prompt") or "").strip()
                column_names = [c.get("name") for c in columns if isinstance(c, dict) and c.get("name")]
                run_state.emit_event(ctx.db, run_obj, "enrichment_card_added", {
                    "enrichment_id": public_eid,
                    "enrichment_uuid": enrichment_id,
                    "table_id": table_short,
                    "table_uuid": table_id,
                    "table_name": table_name_val,
                    "name": name,
                    "columns": column_names,
                    "research_tier": research_tier,
                    "prompt": prompt_text,
                })
        except Exception:
            log.exception("enrichment_card_added emit failed; continuing")

    # Classify-tier enrichments auto-run on creation. They're nano,
    # tool-less, ~$0.0005/row — cheaper than fetching the rows in the
    # first place. The approval card just adds friction to what is
    # functionally a "filter the fetch noise" step. Skip the gate, spawn
    # a background run over all unfilled rows, and surface task_id so
    # the agent can mention it.
    research_tier = (action.get("research") or action.get("tier") or "").lower()
    if research_tier == "classify" and not is_refinement:
        try:
            run_args = {
                "enrichment_id": public_eid,
                "scope": {"type": "all_unfilled"},
                "wait": False,
            }
            spawn_result, _spawn_cost = await enrichment_run(run_args, ctx)
            return {
                "enrichment_id": public_eid,
                "status": "configured_and_running",
                "auto_run": True,
                "background_task": spawn_result,
                "note": (
                    "Classify-tier enrichment auto-runs on creation (no "
                    "approval needed — nano, no tools, dirt cheap). The "
                    "cells are filling now in the background."
                ),
            }, 0.0
        except Exception:
            log.exception("classify auto-run failed for %s; user can run manually", public_eid)

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
    """Run an enrichment over a scope of rows.

    Optional `wait` (default true). When false, returns immediately with
    `{status: "running", task_id: "bt<N>"}` after the approval gate has
    already resolved — approval happens upstream in the agent loop, so
    background mode doesn't bypass cost confirmation. The cell loop runs
    as a tracked background asyncio.Task; agent monitors via
    `task_status` / `task_wait`.
    """
    args = dict(args)
    wait = bool(args.pop("wait", True))
    if not wait:
        from dsl_worker.chat import background_tasks as _bg
        canonical_eid = resolve_enrichment_id(
            ctx.db, ctx.project_id, args.get("enrichment_id")
        )
        spawn_result = await _bg.spawn(
            handler=enrichment_run,
            args=args,
            ctx=ctx,
            kind="enrichment_run",
            task_key=canonical_eid,
            summary=f"Running enrichment {args.get('enrichment_id') or '?'}",
        )
        return spawn_result, 0.0

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
    # total_cost_usd = full raw spend (backend tracking); total_charge_usd
    # = what we actually bill (full on success, 10% subsidy on failure).
    charge_usd = stats["total_charge_usd"]
    cost_credits_full = stats["total_cost_usd"] * 10.0

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
        {"eid": enrichment_id, "n": stats["rows_filled"], "cost": cost_credits_full},
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
    }, charge_usd


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
      total_cost_usd, total_charge_usd, sample_filled_row, sample_not_found_row.
    """
    empty_stats = {
        "rows_attempted": 0,
        "rows_filled": 0,
        "rows_not_found": 0,
        "rows_hit_budget": 0,
        "rows_errored": 0,
        "rows_skipped_missing_deps": 0,
        "total_cost_usd": 0.0,
        "total_charge_usd": 0.0,
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

    total_cost = 0.0          # actual OpenAI/tool USD across all cells (for backend tracking)
    total_charge_usd = 0.0    # what gets billed: full USD on success, 10% subsidy on failure
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
            from dsl_worker.chat import run_state
            from dsl_api.models import ChatRun
            run_obj = ctx.db.query(ChatRun).filter(ChatRun.id == run_id).first()
            if run_obj is not None:
                run_state.emit_event(ctx.db, run_obj, event_type, payload)
        except Exception:
            log.debug("emit %s failed; continuing", event_type, exc_info=True)

    def _emit_many(events: List[Tuple[str, Dict[str, Any]]]) -> None:
        """Persist many events in a single commit. Same best-effort
        semantics as `_emit`."""
        if not run_id or not events:
            return
        try:
            from dsl_worker.chat import run_state
            from dsl_api.models import ChatRun
            run_obj = ctx.db.query(ChatRun).filter(ChatRun.id == run_id).first()
            if run_obj is not None:
                run_state.emit_events_batch(ctx.db, run_obj, events)
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
        sids_to_update: List[str] = []
        events_to_emit: List[Tuple[str, Dict[str, Any]]] = []
        # Status delta is the same for every row in this batch: every
        # target column flips to missing_dependency. We deep-merge into
        # __cell_status__ in SQL so we don't clobber other enrichments'
        # values or status entries.
        status_delta = {cn: "missing_dependency" for cn in target_cols}
        for sid, row_data, missing in deps_missing_rows:
            sids_to_update.append(sid)
            # Build a cell_status preview for the SSE event so the FE can
            # render the badge immediately. Best-effort against the
            # snapshot; the SQL below is authoritative.
            existing_status_preview = (row_data.get("__cell_status__") if isinstance(row_data, dict) else None) or {}
            if not isinstance(existing_status_preview, dict):
                existing_status_preview = {}
            preview = dict(existing_status_preview)
            preview.update(status_delta)
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
                    "cell_status": preview,
                    "missing_columns": missing,
                    "cost": 0,
                },
            ))

        # One batched UPDATE for every skipped row. Uses jsonb_set so we
        # only touch __cell_status__ (deep-merged with whatever's already
        # there) — concurrent writers can still update other columns on
        # the same row without us clobbering them. Until 2026-05-19 this
        # was a full-row replace which raced disastrously with parallel
        # enrichment runs.
        from sqlalchemy import bindparam
        batched_sql = (
            "UPDATE samples "
            "SET row = jsonb_set("
            "  COALESCE(row, '{}'::jsonb), "
            "  '{__cell_status__}', "
            "  COALESCE(row->'__cell_status__', '{}'::jsonb) || CAST(:status_delta AS jsonb)"
            ") "
            "WHERE id::text IN :sids"
        )
        params: Dict[str, Any] = {
            "status_delta": json.dumps(status_delta),
            "sids": tuple(sids_to_update),
        }
        try:
            with time_commit(ctx, "missing_deps_update", threshold_ms=50):
                ctx.db.execute(
                    sa_text(batched_sql).bindparams(bindparam("sids", expanding=True)),
                    params,
                )
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
    # Classify-tier runs nano + no tools (~3s/cell, ~$0.0005), so it
    # gets a much higher cap matching CLASSIFY_CONCURRENCY in the
    # durable-jobs coordinator. Research/deep stay at 25 — they hit
    # FullEnrich/Apollo/BU and the rate limits / latencies make higher
    # concurrency pointless there.
    research_tier_local = (action.get("research") or action.get("tier") or "").lower()
    sem_cap = 100 if research_tier_local == "classify" else 25
    sem = asyncio.Semaphore(sem_cap)

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
            # Register this row as in-progress so the /cells/running
            # endpoint surfaces it to the FE for refresh-resilient
            # pending state. Remove on completion or error.
            from dsl_worker.chat.cell_runs import REGISTRY as CELL_RUNS
            await CELL_RUNS.add(
                ctx.project_id, enrichment_id, sample_id, target_cols,
            )
            try:
                new_fields, new_sources, cost, status = await _execute_action(
                    action, row_data, per_row_cap, columns, ctx,
                    enrichment_id=enrichment_id, sample_id=sample_id,
                    raw_row=raw_row,
                )
            finally:
                await CELL_RUNS.remove(
                    ctx.project_id, enrichment_id, sample_id, target_cols,
                )
            return sample_id, row_data, new_fields, new_sources, cost, status, idx

    phase_marker(ctx, "enrichment_run/tasks_spawning", n=len(ready_rows))
    # Track (sample_id, original_row) per task so the supervisor can
    # write an error status to the correct row when a task raises.
    # Until 2026-05-19 the supervisor just `log.warning`'d and dropped
    # the failure on the floor — the cell stayed blank with no badge,
    # no trace, no FE signal. User saw 1/10 cells silently missing
    # after a bulk run.
    task_meta: Dict[asyncio.Task, Tuple[str, Dict[str, Any]]] = {}
    tasks: List[asyncio.Task] = []
    for sid, rd, raw in ready_rows:
        t = asyncio.create_task(run_one(sid, rd, raw))
        task_meta[t] = (sid, rd if isinstance(rd, dict) else {})
        tasks.append(t)
    phase_marker(ctx, "enrichment_run/tasks_spawned", n=len(tasks))

    try:
      for fut in asyncio.as_completed(tasks):
        try:
            sample_id, original_row, new_fields, new_sources, cost, status, idx = await fut
        except asyncio.CancelledError:
            raise
        except Exception as e:
            log.warning("cell op raised: %s", e)
            errored_count += 1
            completed += 1
            # Recover (sample_id, original_row) from the task so we can
            # write an error sentinel onto the row + emit a cell_filled
            # event with status='error'. The FE renders this as an
            # "Error — retry" badge so the user knows the cell ran but
            # failed, instead of staring at a perpetually-blank cell.
            failed_sid: Optional[str] = None
            failed_row: Dict[str, Any] = {}
            for t, (msid, mrow) in task_meta.items():
                if t is fut:
                    failed_sid = msid
                    failed_row = mrow
                    break
            if failed_sid:
                err_status_delta = {cn: "error" for cn in target_cols}
                try:
                    ctx.db.execute(
                        sa_text(
                            "SELECT pg_advisory_xact_lock(hashtextextended(:sid, 0))"
                        ),
                        {"sid": str(failed_sid)},
                    )
                    ctx.db.execute(
                        sa_text(
                            "UPDATE samples "
                            "SET row = jsonb_set("
                            "  COALESCE(row, '{}'::jsonb), "
                            "  '{__cell_status__}', "
                            "  COALESCE(row->'__cell_status__', '{}'::jsonb) || CAST(:s AS jsonb)"
                            ") "
                            "WHERE id=:sid"
                        ),
                        {"s": json.dumps(err_status_delta), "sid": failed_sid},
                    )
                    ctx.db.commit()
                except Exception:
                    log.exception(
                        "failed to write error status for sample %s; suppressing",
                        failed_sid,
                    )
                    try:
                        ctx.db.rollback()
                    except Exception:
                        pass
                # Cell_filled event with the error status so the FE
                # clears the spinner and shows the error badge live.
                _emit(
                    "cell_filled",
                    {
                        "tool_call_id": enrichment_id,
                        "enrichment_id": enrichment_id,
                        "sample_id": failed_sid,
                        "row_id": failed_sid,
                        "index": 0,
                        "completed": completed,
                        "total": ready_total,
                        "new_fields": None,
                        "status": "error",
                        "cell_status": err_status_delta,
                        "cost": 0,
                    },
                )
            continue

        total_cost += cost
        # Subsidy gate: full charge only when the cell actually produced
        # something the user can use. status=="filled" with all-null values
        # means the agent ran but couldn't find an answer — bill 10%, same
        # as hit_budget / soft error. Keeps the user from paying full price
        # for a row they got nothing for.
        produced_value = (
            status == "filled"
            and isinstance(new_fields, dict)
            and any(v not in (None, "") for v in new_fields.values())
        )
        total_charge_usd += cost if produced_value else cost * 0.10
        completed += 1

        if isinstance(original_row, str):
            try:
                original_row = json.loads(original_row)
            except Exception:
                original_row = {}
        if not isinstance(original_row, dict):
            original_row = {}

        # Compute DELTAS this enrichment will apply (only keys we own:
        # the enrichment's target_cols + sidecars for those cols, plus
        # downstream missing_dependency clears).
        #
        # CRITICAL: until 2026-05-19 this code copied original_row,
        # mutated it, and wrote the whole JSONB back. Two enrichments
        # touching the same sample concurrently raced on that write —
        # last writer wins, intermediate values silently dropped. On
        # Captain (project beae87a4, table t2) 4/10 Universities cells
        # and 7/11 Past Employers cells were wiped this way. Fix: lock
        # the sample, re-read fresh state, apply ONLY our deltas, write.
        has_real_value = bool(
            isinstance(new_fields, dict)
            and any(new_fields.get(cn) not in (None, "") for cn in target_cols)
        )
        if isinstance(new_fields, dict) and new_fields and has_real_value:
            filled_count += 1
            if sample_filled_row is None:
                sample_filled_row = {cn: new_fields.get(cn) for cn in target_cols}
        if status == "hit_budget":
            hit_budget_count += 1
        elif status == "filled" and not has_real_value:
            not_found_count += 1
            if sample_not_found_row is None:
                sample_not_found_row = {
                    k: v for k, v in original_row.items()
                    if not k.startswith("__") and k not in target_cols
                }

        # Value delta: only the columns this enrichment is responsible for.
        value_delta: Dict[str, Any] = {}
        if isinstance(new_fields, dict):
            for cn in target_cols:
                if cn in new_fields:
                    value_delta[cn] = new_fields[cn]

        # Status delta: keys to SET vs keys to CLEAR (within target_cols
        # only — plus downstream missing_dependency clears that this run
        # legitimately unblocks).
        status_set: Dict[str, str] = {}
        status_clear: List[str] = []
        if status == "hit_budget":
            for cn in target_cols:
                if not (isinstance(new_fields, dict) and new_fields.get(cn)):
                    status_set[cn] = "hit_budget"
        elif status == "filled" and isinstance(new_fields, dict):
            for cn in target_cols:
                v = new_fields.get(cn)
                if v not in (None, ""):
                    status_clear.append(cn)
                else:
                    status_set[cn] = "not_found"
        if isinstance(new_fields, dict) and new_fields and downstream_unblocks:
            for cn, v in new_fields.items():
                if v in (None, ""):
                    continue
                for downstream_cn in downstream_unblocks.get(cn, []):
                    status_clear.append(downstream_cn)

        # Cost delta: attribute the run's cost to every target column
        # the cell agent worked on, whether or not it produced a value.
        # `cost` is in USD (cell agent's total_cost). The FE expects
        # __cell_cost__ in CREDITS (1 cr = $0.10), so convert here.
        # Without the *10, a 5.5-credit run shows up in the detail panel
        # as "0.55" and a hit_budget cell with cap=5 looks like it
        # tripped at 0.55 credits — completely misleading.
        cost_delta: Dict[str, float] = {}
        if cost > 0:
            keys = list(new_fields.keys()) if isinstance(new_fields, dict) and new_fields else list(target_cols)
            for cn in keys:
                cost_delta[cn] = float(cost) * 10.0

        # Sources delta: only persist sources for columns that actually
        # got a value this run.
        sources_to_persist: Dict[str, List[Dict[str, Any]]] = {}
        if isinstance(new_sources, dict):
            for cn, citations in new_sources.items():
                if (
                    isinstance(new_fields, dict)
                    and new_fields.get(cn) not in (None, "")
                    and isinstance(citations, list)
                    and citations
                ):
                    sources_to_persist[cn] = citations

        # Default existing_status for the SSE event in the no-write path
        # (best-effort: original_row's status; next refresh corrects).
        existing_status: Dict[str, str] = {}
        _orig_status = original_row.get("__cell_status__") if isinstance(original_row, dict) else None
        if isinstance(_orig_status, dict):
            existing_status = dict(_orig_status)

        # Fresh tags after the per-row update — read once after commit and
        # pass through to cell_filled so the FE has the merged sources +
        # status + verification state without a refetch.
        fresh_tags: Optional[Dict[str, Any]] = None

        has_any_change = bool(value_delta or status_set or status_clear or cost_delta or sources_to_persist)
        if has_any_change:
            try:
                # Serialize concurrent writes on this sample. xact lock is
                # released at commit; other writers wait. Without this,
                # the read+modify+write below races at the row-JSONB level
                # and clobbers fields the other writer just committed.
                ctx.db.execute(
                    sa_text("SELECT pg_advisory_xact_lock(hashtextextended(:sid, 0))"),
                    {"sid": str(sample_id)},
                )
                # Re-read inside the lock so we merge with anything the
                # other writer just committed.
                fresh = ctx.db.execute(
                    sa_text("SELECT row FROM samples WHERE id=:sid"),
                    {"sid": sample_id},
                ).fetchone()
                if fresh:
                    fresh_row = fresh[0] if isinstance(fresh[0], dict) else json.loads(fresh[0] or "{}")
                else:
                    fresh_row = {}
                if not isinstance(fresh_row, dict):
                    fresh_row = {}

                # Build the row we'll write: start from FRESH state,
                # layer our deltas on top. Sidecars deep-merge.
                #
                # When overwrite=false, also skip cells that already hold a
                # value in fresh_row. Row-level filtering already drops
                # fully-filled rows, but a partially-filled row's filled
                # cells were still being clobbered by the agent's restated
                # values (e.g. an adopted query column where URL was scraped
                # and Starting Bid was missing — agent emits both and the
                # original URL got rewritten). With this skip, "overwrite:
                # false" means "fill missing" at the cell level too.
                final = dict(fresh_row)
                for k, v in value_delta.items():
                    if not overwrite and fresh_row.get(k) not in (None, ""):
                        continue
                    final[k] = v

                final_status = final.get("__cell_status__") if isinstance(final.get("__cell_status__"), dict) else {}
                if not isinstance(final_status, dict):
                    final_status = {}
                else:
                    final_status = dict(final_status)
                for cn, s in status_set.items():
                    final_status[cn] = s
                for cn in status_clear:
                    final_status.pop(cn, None)
                if final_status:
                    final["__cell_status__"] = final_status
                elif "__cell_status__" in final:
                    final.pop("__cell_status__")

                final_cost = final.get("__cell_cost__") if isinstance(final.get("__cell_cost__"), dict) else {}
                if not isinstance(final_cost, dict):
                    final_cost = {}
                else:
                    final_cost = dict(final_cost)
                for cn, c in cost_delta.items():
                    final_cost[cn] = c
                if final_cost:
                    final["__cell_cost__"] = final_cost

                if sources_to_persist:
                    ctx.db.execute(
                        sa_text(
                            "UPDATE samples "
                            "SET row=CAST(:row AS jsonb), "
                            "    tags=jsonb_set("
                            "      COALESCE(tags, '{}'::jsonb), "
                            "      '{sources}', "
                            "      COALESCE(tags->'sources', '{}'::jsonb) || CAST(:srcs AS jsonb)"
                            "    ) "
                            "WHERE id=:sid"
                        ),
                        {
                            "row": json.dumps(final),
                            "srcs": json.dumps(sources_to_persist),
                            "sid": sample_id,
                        },
                    )
                else:
                    ctx.db.execute(
                        sa_text("UPDATE samples SET row=CAST(:row AS jsonb) WHERE id=:sid"),
                        {"row": json.dumps(final), "sid": sample_id},
                    )
                # Materialize-on-write: maintain table_column_value counts
                # so the filter dropdowns stay accurate as agent-triggered
                # cells fill. Same column set the new coordinator uses.
                try:
                    from dsl_worker.chat.enrichment_jobs import (
                        _column_diff,
                        _apply_column_value_deltas,
                    )
                    deltas = _column_diff(fresh_row, final, target_cols)
                    if deltas:
                        _apply_column_value_deltas(ctx.db, table_id, deltas)
                except Exception:
                    log.warning(
                        "materialize column-value deltas failed for %s; suppressed",
                        sample_id,
                        exc_info=True,
                    )
                # Keep `merged` + `existing_status` for downstream code
                # (cell_filled SSE payload, email verify hook). `final`
                # is value_delta layered over a fresh snapshot — what
                # the FE should see in the event.
                merged = final
                existing_status = final_status
                ctx.db.commit()
                # Re-read tags AFTER commit so the SSE event carries
                # the freshly-merged tags.sources for this row. The FE
                # wholesale-replaces row.tags on cell_filled, so we
                # need to send the FULL tags state, not just the delta —
                # otherwise sibling tag subkeys (fill_status,
                # email_verification, prior columns' sources) get
                # clobbered or the new sources never paint until refresh.
                try:
                    fresh_tags_row = ctx.db.execute(
                        sa_text("SELECT tags FROM samples WHERE id=:sid"),
                        {"sid": sample_id},
                    ).fetchone()
                    if fresh_tags_row and fresh_tags_row[0] is not None:
                        fresh_tags = fresh_tags_row[0]
                        if isinstance(fresh_tags, str):
                            fresh_tags = json.loads(fresh_tags or "{}")
                    else:
                        fresh_tags = None
                except Exception:
                    fresh_tags = None
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
            # Convert USD → credits to match __cell_cost__ persistence.
            # FE merges this straight onto the row's sidecar.
            "cost": float(cost) * 10.0,
            # Fresh row.tags snapshot after commit so the FE replaces
            # row.tags wholesale and sees the just-merged sources +
            # any other tag subkeys (fill_status, email_verification)
            # without a refresh. Omitted when nothing changed in this
            # call (no write happened, no fresh read needed).
            "_tags": fresh_tags,
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
            ctx.partial_cost_usd += float(total_charge_usd)
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
        "total_cost_usd": total_cost,
        "total_charge_usd": total_charge_usd,
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
      filtered:     {type, filters: [{column, op, value}, ...], first_n?: N}
                    same {column, op, value} shape as filter_set. Filters
                    are AND'd together, evaluated in Python via the same
                    _match semantics used everywhere else. Optional
                    `first_n` caps the result to the first N rows (after
                    seq-ordering) — used for "do 10 more empty rows" style
                    asks where the agent expresses BOTH a filter AND a
                    batch size.
      all_unfilled: {type, first_n?: N} — every row missing at least one
                    target column. Optional `first_n` caps the result so
                    "run 10 more unfilled" is expressible directly.
    """
    # `tags` carries failed_urls / failed_emails — values previously
    # verified as broken. We strip those from raw_row before handing
    # to the cell agent so a re-run can't just regurgitate the same
    # bad URL it pulled from hidden_source_fields last time.
    base_sql = "SELECT id::text, row, raw_row, tags FROM samples WHERE table_id=:tid AND deleted_at IS NULL"
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
        from dsl_worker.chat.tools import _match, _normalize_filter
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
            for sid, row_data, raw_row, tags in all_rows:
                if not isinstance(row_data, dict):
                    continue
                keep = all(
                    _match(row_data.get(c), op, v) for c, op, v in norm_filters
                )
                if keep:
                    rows.append((sid, row_data, raw_row, tags))
    else:  # all_unfilled
        rows = db.execute(
            sa_text(base_sql + " ORDER BY seq"),
            params,
        ).fetchall()

    # Filter to unfilled (unless overwrite) — a row is "unfilled" if any of the
    # enrichment's target columns has no value.
    target_cols = [c["name"] for c in enrichment_columns]
    out: List[Tuple[str, Dict[str, Any], Dict[str, Any]]] = []
    for sid, row_data, raw_row, tags in rows:
        if not isinstance(row_data, dict):
            row_data = {}
        if not isinstance(raw_row, dict):
            raw_row = {}
        raw_row = _scrub_failed_values(raw_row, tags)
        if overwrite:
            out.append((sid, row_data, raw_row))
        else:
            any_unfilled = any(not row_data.get(c) for c in target_cols)
            if any_unfilled:
                out.append((sid, row_data, raw_row))

    # Apply first_n cap if the scope carried one. Honored on filtered +
    # all_unfilled so the agent can express "10 more empty rows" by
    # combining a filter (or all_unfilled) with first_n. Until 2026-05-20
    # this was silently dropped: the agent passed first_n=10 with a
    # filtered scope and got the full 86-row match instead. Skip for
    # row_ids (caller named the rows explicitly) and the bare first_n
    # branch (already applied a SQL LIMIT).
    if scope_type in ("filtered", "all_unfilled"):
        cap_raw = scope.get("first_n")
        if cap_raw is not None:
            try:
                cap = int(cap_raw)
                if cap > 0:
                    out = out[:cap]
            except (TypeError, ValueError):
                pass
    return out


def _collect_failed_values(tags: Any) -> set:
    """Collect every value previously verified as broken on this row.

    Reads ``tags.failed_urls`` and ``tags.failed_emails`` (both shaped
    ``{column: [value, ...]}``) and flattens them into a single set of
    case-insensitive strings. The set is used by ``_scrub_failed_values``
    to remove these values from ``raw_row`` before the cell agent sees
    it — otherwise the agent reads the broken URL out of
    ``row_hidden_source_fields`` and silently returns it without
    actually researching, defeating the point of a re-run.
    """
    if not isinstance(tags, dict):
        return set()
    out: set = set()
    for bucket_key in ("failed_urls", "failed_emails"):
        bucket = tags.get(bucket_key)
        if not isinstance(bucket, dict):
            continue
        for vals in bucket.values():
            if not isinstance(vals, list):
                continue
            for v in vals:
                if isinstance(v, str) and v:
                    out.add(v.casefold())
    return out


def _scrub_failed_values(raw_row: Dict[str, Any], tags: Any) -> Dict[str, Any]:
    """Return a shallow copy of ``raw_row`` with previously-failed values
    removed. Top-level strings whose case-insensitive form is in the
    banned set are dropped entirely (key removed); list values get
    those entries filtered out, and the key is dropped if the list ends
    up empty. Other types pass through untouched.
    """
    banned = _collect_failed_values(tags)
    if not banned or not isinstance(raw_row, dict):
        return raw_row
    cleaned: Dict[str, Any] = {}
    for k, v in raw_row.items():
        if isinstance(v, str):
            if v.casefold() in banned:
                continue
            cleaned[k] = v
        elif isinstance(v, list):
            kept = [
                item
                for item in v
                if not (isinstance(item, str) and item.casefold() in banned)
            ]
            if kept:
                cleaned[k] = kept
        else:
            cleaned[k] = v
    return cleaned


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
) -> Tuple[Dict[str, Any], Dict[str, List[Dict[str, Any]]], float, str]:
    """Run one enrichment action against one row.
    Returns (new_fields_dict, sources_per_column, cost_credits, status).

    sources_per_column matches the shape of fetch-side citations stored in
    samples.tags.sources: {col_name → [{type: "source_record", source, source_field?}]}.
    """
    from dsl_worker.chat.cell_agent import run_cell_agent
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
    """Attach enrichment columns to the table's columns array.

    New columns are appended. Existing query columns (no enrichment_id) get
    adopted — the enrichment_id is stamped on them so a partially-filled
    query column can be backfilled by enrichment_run without redefining or
    overwriting good data. Existing columns already owned by a different
    enrichment are left alone.
    """
    row = db.execute(
        sa_text("SELECT columns FROM tables WHERE id=:tid"),
        {"tid": table_id},
    ).fetchone()
    if not row:
        return
    existing = row[0] if isinstance(row[0], list) else json.loads(row[0] or "[]")
    by_name = {c["name"]: c for c in existing}
    to_add: List[Dict[str, Any]] = []
    adopted = False
    for c in enrichment_columns:
        name = c["name"]
        if name in by_name:
            current = by_name[name]
            if enrichment_id and not current.get("enrichment_id"):
                current["enrichment_id"] = enrichment_id
                adopted = True
            continue
        cnew = dict(c)
        if enrichment_id:
            cnew["enrichment_id"] = enrichment_id
        to_add.append(cnew)
    if not to_add and not adopted:
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
