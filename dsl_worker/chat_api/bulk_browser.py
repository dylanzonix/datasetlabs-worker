"""rows_fill_bulk_browser — batched browser_use enrichment.

Higher-cost / higher-yield alternative to per-cell rows_fill: groups
rows into batches (default 5/batch) and dispatches ONE browser_use
session per batch with a structured task ("find {columns} for these
N people, return one item per person"). browser_use sees patterns
across people in the same query, so it finds matches that per-cell
web_search misses.

Tradeoff: ~3-5 credits/row vs ~0.3-1 for rows_fill. Batches do NOT
retry hard on misses — if browser_use returns null for someone in
the batch, that's it. The chat agent decides whether to call this
tool again with the still-null rows. That matches the empirical
observation: 2-3 light passes >> 1 pass with maxed-out turns.
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple

from sqlalchemy import text as sql_text

from dsl_api.db import SessionLocal
from dsl_api.models import Project
from dsl_api.models.sample import Sample

from dsl_worker import skills as skills_loader
from dsl_worker.chat_api import cell_traces, sources

ProgressCallback = Callable[[Dict[str, Any]], Awaitable[None]]
log = logging.getLogger(__name__)


_BATCH_SIZE_DEFAULT = 5
# Two parallel batches by default. browser_use sessions are heavy
# (cloud browser, anti-bot, residential proxies), so we don't fan out
# wide. The chat agent can re-call to chip away at remaining nulls.
_BATCH_CONCURRENCY = 2
_BATCH_TIMEOUT_SECS = 300
# Per-row context value cap. Larger values bloat the task without
# typically helping identification.
_ROW_VALUE_TRUNCATE = 200


def _row_context(row_idx: int, row_data: Dict[str, Any]) -> str:
    """Render one row as a compact line for the task."""
    parts = [f"  {row_idx}."]
    for k, v in row_data.items():
        if v is None or v == "":
            continue
        if isinstance(k, str) and k.startswith("_"):
            continue
        s = json.dumps(v, default=str, ensure_ascii=False)
        if len(s) > _ROW_VALUE_TRUNCATE:
            s = s[:_ROW_VALUE_TRUNCATE] + "..."
        parts.append(f"{k}={s}")
    return " ".join(parts) if len(parts) > 1 else parts[0] + " (no fields)"


def _build_task(
    rows_in_batch: List[Tuple[Any, Dict[str, Any]]],
    target_columns: List[str],
    target_specs: Dict[str, Dict[str, str]],
    skills_extra: Optional[str],
) -> str:
    """Compose the browser_use task for a batch.

    Asks for one item per person, keyed by `row_index` (1..N) so two
    same-named people in the same batch don't collide. Each item must
    cite specific evidence (matches the find_x_handles skill rule).
    """
    n = len(rows_in_batch)
    lines: List[str] = []
    lines.append(
        f"You are filling specific data fields for {n} different people. "
        f"Use the web (X/Twitter, LinkedIn, company sites) to find the "
        f"requested fields for each person. Return ONE item per person."
    )
    lines.append("")
    lines.append("Fields to fill for each person:")
    for col in target_columns:
        spec = target_specs.get(col, {})
        line = f"  - {col}"
        if spec.get("format"):
            line += f" — format: {spec['format']}"
        if spec.get("description"):
            line += f" — {spec['description']}"
        lines.append(line)
    lines.append("")
    lines.append(f"People in this batch ({n}):")
    for i, (_row_id, row_data) in enumerate(rows_in_batch, start=1):
        lines.append(_row_context(i, row_data))
    lines.append("")
    lines.append("Output format:")
    lines.append(
        f"  Return EXACTLY {n} items, one per person, matching the "
        f"people above by `row_index`. Each item MUST have:"
    )
    lines.append(f"    - row_index: integer 1..{n} matching the input order")
    for col in target_columns:
        lines.append(f"    - {col!r}: the found value, or null if no confident match")
    lines.append(
        "    - evidence: one short sentence quoting the specific bio/"
        "profile text that confirmed identity (or 'no match' if null)."
    )
    lines.append("")
    lines.append(
        "Critical: a confident null beats a wrong value. If you cannot "
        "verify identity via concrete bio/profile evidence, return null "
        "for that person. Don't guess based on name alone — collisions "
        "with same-named strangers are common."
    )
    lines.append(
        "Don't ballroom on people with no public presence: spend ~1 minute "
        "per person max. If two angles miss, mark null and move on. The "
        "caller may re-run on the still-null rows with a different "
        "approach — that's cheaper than burning the whole session here."
    )
    if skills_extra:
        lines.append("")
        lines.append(skills_extra.strip())
    return "\n".join(lines)


def _coerce_row_index(item: Dict[str, Any]) -> Optional[int]:
    v = item.get("row_index")
    if v is None:
        return None
    try:
        return int(v)
    except (ValueError, TypeError):
        return None


async def _process_batch(
    *,
    batch_idx: int,
    rows_in_batch: List[Tuple[Any, Dict[str, Any]]],
    target_columns: List[str],
    target_specs: Dict[str, Dict[str, str]],
    skills_extra: Optional[str],
    skills_applied: List[str],
    timeout_secs: int,
    progress_cb: Optional[ProgressCallback],
) -> Tuple[List[cell_traces.CellTrace], List[Tuple[Any, Dict[str, Any]]], float]:
    """Run one batch through browser_use and produce (traces, writes, cost)."""
    task = _build_task(rows_in_batch, target_columns, target_specs, skills_extra)

    bu = sources._bu()
    if bu is None:
        traces = []
        for row_id, _row_data in rows_in_batch:
            tr = cell_traces.new_trace(row_id=str(row_id), columns=list(target_columns))
            tr.skills_applied = list(skills_applied)
            tr.status = "error"
            tr.reason = "BROWSER_USE_API_KEY not configured"
            tr.ended_at = datetime.now(timezone.utc).isoformat()
            traces.append(tr)
        return traces, [], 0.0

    try:
        items, cost, _session_id, _summary = await bu.extract(
            task=task, timeout=timeout_secs,
        )
    except Exception as e:
        log.exception("bulk_browser batch %d failed", batch_idx)
        traces = []
        for row_id, _row_data in rows_in_batch:
            tr = cell_traces.new_trace(row_id=str(row_id), columns=list(target_columns))
            tr.skills_applied = list(skills_applied)
            tr.status = "error"
            tr.reason = f"browser_use failed: {type(e).__name__}: {e}"
            tr.ended_at = datetime.now(timezone.utc).isoformat()
            traces.append(tr)
        return traces, [], 0.0

    cost = float(cost or 0)
    items = items or []
    per_row_cost = cost / max(1, len(rows_in_batch))

    items_by_idx: Dict[int, Dict[str, Any]] = {}
    for it in items:
        if not isinstance(it, dict):
            continue
        idx = _coerce_row_index(it)
        if idx is None or idx < 1 or idx > len(rows_in_batch):
            continue
        if idx not in items_by_idx:
            items_by_idx[idx] = it

    traces: List[cell_traces.CellTrace] = []
    row_writes: List[Tuple[Any, Dict[str, Any]]] = []

    for i, (row_id, _row_data) in enumerate(rows_in_batch, start=1):
        tr = cell_traces.new_trace(row_id=str(row_id), columns=list(target_columns))
        tr.skills_applied = list(skills_applied)
        tr.cost_usd = per_row_cost
        tr.turns_used = 1

        item = items_by_idx.get(i)
        if item is None:
            tr.status = "null_legitimate"
            tr.reason = (
                f"browser_use returned no item for row_index={i} in this "
                f"batch of {len(rows_in_batch)}. The session likely couldn't "
                f"verify identity for this person."
            )
            tr.turn_log.append(cell_traces.CellTraceTurn(
                turn=1,
                kind="tool_call",
                name="browser_use",
                args={"batch_size": len(rows_in_batch), "row_index": i},
                result="no item returned for this row_index",
            ))
        else:
            evidence = item.get("evidence") or ""
            values = {col: item.get(col) for col in target_columns}
            non_null = {k: v for k, v in values.items() if v is not None and v != ""}
            if non_null:
                tr.status = "filled"
                tr.values = values
                tr.reason = evidence or "matched via bulk browser_use"
                row_writes.append((row_id, values))
            else:
                tr.status = "null_legitimate"
                tr.reason = evidence or "browser_use returned null for all target columns"
            tr.turn_log.append(cell_traces.CellTraceTurn(
                turn=1,
                kind="tool_call",
                name="browser_use",
                args={"batch_size": len(rows_in_batch), "row_index": i},
                result=item,
                cost_usd=per_row_cost,
            ))

        tr.ended_at = datetime.now(timezone.utc).isoformat()
        traces.append(tr)

    if progress_cb:
        try:
            await progress_cb({
                "type": "bulk_batch_done",
                "batch": batch_idx,
                "rows": len(rows_in_batch),
                "matched_items": len(items_by_idx),
                "cost": round(cost, 4),
            })
        except Exception:
            log.exception("progress_cb in bulk batch raised; suppressing")

    return traces, row_writes, cost


async def bulk_fill_rows(
    *,
    project: Project,
    target_columns: List[str],
    where_sql: str,
    where_params: Dict[str, Any],
    limit: Optional[int],
    batch_size: int = _BATCH_SIZE_DEFAULT,
    concurrency: int = _BATCH_CONCURRENCY,
    timeout_secs: int = _BATCH_TIMEOUT_SECS,
    progress_cb: Optional[ProgressCallback] = None,
) -> Tuple[Dict[str, Any], float]:
    """Run browser_use in batches over matching rows.

    Returns (summary_dict, total_cost_usd). Same summary shape as
    fill.fill_rows so the chat agent surface stays consistent.
    """
    run_id = uuid.uuid4().hex[:12]

    project_columns = {
        c.get("name"): c for c in (project.columns or []) if isinstance(c, dict)
    }
    missing = [c for c in target_columns if c not in project_columns]
    if missing:
        return (
            {"error": f"columns not found: {missing}. Add them with columns_add first."},
            0.0,
        )
    target_specs: Dict[str, Dict[str, str]] = {}
    for col in target_columns:
        spec = project_columns.get(col, {}) or {}
        target_specs[col] = {
            "format": spec.get("format") or "",
            "description": spec.get("description") or "",
        }

    skill_columns_for_match = [
        {
            "name": col,
            "description": target_specs[col]["description"],
            "format": target_specs[col]["format"],
        }
        for col in target_columns
    ]
    try:
        matched_skills = skills_loader.match_skills("cell_agent", skill_columns_for_match)
        skills_extra = skills_loader.render_skills(matched_skills) or None
        skills_applied_names = [s.name for s in matched_skills]
    except Exception:
        log.exception("skills loader failed (continuing without skills)")
        skills_extra = None
        skills_applied_names = []

    db = SessionLocal()
    try:
        version_id = project.current_version_id
        if not version_id:
            return ({"error": "project has no version yet"}, 0.0)
        sql = (
            f"SELECT id, row FROM samples WHERE version_id = :vid "
            f"AND deleted_at IS NULL AND ({where_sql}) "
            f"ORDER BY seq"
        )
        if limit is not None:
            sql += f" LIMIT {int(limit)}"
        rows = db.execute(sql_text(sql), {"vid": version_id, **where_params}).all()
    finally:
        db.close()

    if not rows:
        return ({"matched_rows": 0, "filled": 0, "summary": "no rows match"}, 0.0)

    def _is_empty(v: Any) -> bool:
        if v is None:
            return True
        if isinstance(v, str) and v.strip() == "":
            return True
        return False

    work_items: List[Tuple[Any, Dict[str, Any]]] = []
    rows_skipped_already_filled = 0
    for row_id, row_data in rows:
        existing = dict(row_data or {})
        unfilled = any(_is_empty(existing.get(c)) for c in target_columns)
        if unfilled:
            work_items.append((row_id, existing))
        else:
            rows_skipped_already_filled += 1

    if not work_items:
        return ({
            "matched_rows": len(rows),
            "rows_skipped_already_filled": rows_skipped_already_filled,
            "processed": 0,
            "cells_filled": 0,
            "by_status": {},
            "note": "All matched rows already have values in the target columns.",
        }, 0.0)

    bs = max(1, int(batch_size))
    batches: List[List[Tuple[Any, Dict[str, Any]]]] = [
        work_items[i:i + bs] for i in range(0, len(work_items), bs)
    ]

    if progress_cb:
        try:
            await progress_cb({
                "type": "bulk_fill_start",
                "total_rows": len(work_items),
                "batches": len(batches),
                "batch_size": bs,
                "columns": list(target_columns),
            })
        except Exception:
            log.exception("progress_cb raised; suppressing")

    sem = asyncio.Semaphore(max(1, int(concurrency)))

    async def _run_one(idx: int, batch):
        async with sem:
            return await _process_batch(
                batch_idx=idx,
                rows_in_batch=batch,
                target_columns=target_columns,
                target_specs=target_specs,
                skills_extra=skills_extra,
                skills_applied=skills_applied_names,
                timeout_secs=timeout_secs,
                progress_cb=progress_cb,
            )

    results = await asyncio.gather(
        *[_run_one(i + 1, b) for i, b in enumerate(batches)],
        return_exceptions=True,
    )

    all_traces: List[cell_traces.CellTrace] = []
    all_writes: List[Tuple[Any, Dict[str, Any]]] = []
    total_cost = 0.0
    batches_failed = 0
    for r in results:
        if isinstance(r, Exception):
            batches_failed += 1
            log.exception("bulk batch raised: %s", r)
            continue
        traces, writes, cost = r
        all_traces.extend(traces)
        all_writes.extend(writes)
        total_cost += cost

    failed_cols_by_row: Dict[Any, Dict[str, Dict[str, Any]]] = {}
    for tr in all_traces:
        if tr.status == "filled":
            continue
        per_col: Dict[str, Dict[str, Any]] = {}
        for col in tr.columns:
            per_col[col] = {
                "status": tr.status,
                "reason": tr.reason or None,
                "cost": round(tr.cost_usd, 4),
            }
        failed_cols_by_row[tr.row_id] = per_col

    write_db = SessionLocal()
    merged_rows_to_emit: List[Dict[str, Any]] = []
    try:
        for row_id, values in all_writes:
            sample = write_db.query(Sample).filter(Sample.id == row_id).first()
            if sample is None:
                continue
            d = dict(sample.row or {})
            for k, v in values.items():
                # Don't clobber an existing non-null cell with a null
                # the bulk task returned. The pre-filter lets a row
                # through if ANY target column is empty, so BU sees
                # all target columns regardless of which were already
                # filled per row. If BU returns null for a column that
                # was already filled (e.g. from an earlier rows_fill
                # pass), preserve the prior value instead of wiping
                # it. Without this guard we silently lost cells when
                # bulk ran a second time over a partially-filled set.
                existing_v = d.get(k)
                is_new_null = v is None or v == ""
                is_existing_filled = existing_v is not None and existing_v != ""
                if is_new_null and is_existing_filled:
                    continue
                d[k] = v
            sample.row = d
            tags = dict(sample.tags or {})
            existing_status = dict(tags.get("fill_status") or {})
            for col, v in values.items():
                if v is not None and v != "" and col in existing_status:
                    del existing_status[col]
            if existing_status:
                tags["fill_status"] = existing_status
            else:
                tags.pop("fill_status", None)
            sample.tags = tags
        for row_id, per_col in failed_cols_by_row.items():
            sample = write_db.query(Sample).filter(Sample.id == row_id).first()
            if sample is None:
                continue
            tags = dict(sample.tags or {})
            existing_status = dict(tags.get("fill_status") or {})
            existing_status.update(per_col)
            tags["fill_status"] = existing_status
            sample.tags = tags
        write_db.commit()

        if progress_cb:
            for row_id, _values in all_writes:
                sample = write_db.query(Sample).filter(Sample.id == row_id).first()
                if sample is None:
                    continue
                merged_rows_to_emit.append({
                    "_id": str(sample.id),
                    "_seq": sample.seq,
                    "_tags": sample.tags or {},
                    **(sample.row or {}),
                })
    except Exception:
        log.exception("bulk_browser persist failed")
        try:
            write_db.rollback()
        except Exception:
            pass
    finally:
        write_db.close()

    if progress_cb:
        for merged in merged_rows_to_emit:
            try:
                await progress_cb({"type": "row_merged", "row": merged})
            except Exception:
                log.exception("progress_cb row_merged raised; suppressing")

    trace_persist_info: Optional[Dict[str, Any]] = None
    if all_traces:
        try:
            trace_persist_info = cell_traces.write_traces(
                project.id, run_id, all_traces, target_columns=list(target_columns),
            )
        except Exception:
            log.exception("cell_traces persist failed (continuing)")

    by_status: Dict[str, int] = {}
    for tr in all_traces:
        by_status[tr.status] = by_status.get(tr.status, 0) + 1
    filled_total = sum(1 for tr in all_traces if tr.status == "filled")

    failure_buckets: Dict[str, int] = {}
    for tr in all_traces:
        if tr.status == "filled":
            continue
        key = (tr.reason or f"({tr.status})").strip().lower()[:100]
        failure_buckets[key] = failure_buckets.get(key, 0) + 1
    top_failures = sorted(
        ({"reason": k, "count": v} for k, v in failure_buckets.items()),
        key=lambda x: x["count"],
        reverse=True,
    )[:5]

    summary: Dict[str, Any] = {
        "matched_rows": len(rows),
        "rows_skipped_already_filled": rows_skipped_already_filled,
        "processed": len(all_traces),
        "cells_filled": filled_total,
        "batches_run": len(batches),
        "batches_failed": batches_failed,
        "by_status": by_status,
        "avg_cost_per_row": round(total_cost / max(1, len(all_traces)), 4),
        "samples": [
            {
                "row_id": tr.row_id,
                "values": tr.values,
                "status": tr.status,
                "reason": tr.reason,
                "cost": round(tr.cost_usd, 4),
            }
            for tr in all_traces[:5]
        ],
        "run_id": run_id,
        "top_failure_reasons": top_failures,
    }
    if trace_persist_info and trace_persist_info.get("persisted"):
        summary["trace_file"] = trace_persist_info.get("file")
    if skills_applied_names:
        summary["skills_applied"] = list(skills_applied_names)

    return (summary, total_cost)
