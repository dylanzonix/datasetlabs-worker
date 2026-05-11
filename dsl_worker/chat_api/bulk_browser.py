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

from dsl_api.config import settings as _api_settings
from dsl_api.db import SessionLocal
from dsl_api.models import Project
from dsl_api.models.sample import Sample

from dsl_worker import skills as skills_loader
from dsl_worker.chat_api import cell_traces, email_verify, sources

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
    lines.append(
        "    - sources: object mapping each filled column name to a list "
        "of URL(s) you actually visited that justify the value. Skip the "
        "key for columns you set to null. The frontend renders these as "
        "per-cell citations. Don't fabricate URLs — only cite pages you "
        "really opened."
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
) -> Tuple[
    List[cell_traces.CellTrace],
    # (row_id, values, per-cell sources dict)
    List[Tuple[Any, Dict[str, Any], Dict[str, List[Dict[str, str]]]]],
    float,
]:
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
    row_writes: List[Tuple[Any, Dict[str, Any], Dict[str, List[Dict[str, str]]]]] = []

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
            # Per-cell sources: BU returns {col: [url, ...]} in `sources`.
            # Normalize to the rows_add wire shape so the persist path
            # below + the FE render can stay uniform with rows_fill.
            raw_sources = item.get("sources") or {}
            cell_sources: Dict[str, List[Dict[str, str]]] = {}
            if isinstance(raw_sources, dict):
                for col, urls in raw_sources.items():
                    if col not in target_columns:
                        continue
                    if values.get(col) is None or values.get(col) == "":
                        continue
                    if isinstance(urls, str):
                        urls = [urls]
                    if not isinstance(urls, list):
                        continue
                    normed = [
                        {"type": "url", "value": str(u).strip()}
                        for u in urls
                        if isinstance(u, str) and u.strip()
                    ]
                    if normed:
                        cell_sources[col] = normed
            if non_null:
                tr.status = "filled"
                tr.values = values
                tr.reason = evidence or "matched via bulk browser_use"
                row_writes.append((row_id, values, cell_sources))
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
    retry_failed: bool = False,
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

    email_columns: set = {
        c for c in target_columns
        if email_verify.is_email_column(project_columns.get(c) or {}, c)
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
            f"SELECT id, row, tags FROM samples WHERE version_id = :vid "
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

    # Pre-filter: skip rows whose target columns are already filled OR
    # were already attempted via this strategy. The skip-prior-fail
    # branch fires when retry_failed=False (default) and the row's
    # tags.fill_status[col].status == 'null_legitimate' — re-running
    # bulk_browser on rows that already came back null from a prior
    # bulk pass produces the same null again. The project f34982fd
    # regression burned $1.03 on exactly this pattern.
    work_items: List[Tuple[Any, Dict[str, Any]]] = []
    rows_skipped_already_filled = 0
    rows_skipped_prior_fail = 0
    for row_id, row_data, tags in rows:
        existing = dict(row_data or {})
        fill_status = (tags or {}).get("fill_status") or {}
        all_filled = True
        any_unfilled_skippable = False
        any_unfilled_attemptable = False
        for c in target_columns:
            if _is_empty(existing.get(c)):
                all_filled = False
                if not retry_failed:
                    prior = fill_status.get(c) or {}
                    if isinstance(prior, dict) and prior.get("status") == "null_legitimate":
                        any_unfilled_skippable = True
                        continue
                any_unfilled_attemptable = True
        if all_filled:
            rows_skipped_already_filled += 1
        elif any_unfilled_attemptable:
            work_items.append((row_id, existing))
        elif any_unfilled_skippable:
            rows_skipped_prior_fail += 1

    if not work_items:
        if rows_skipped_prior_fail and not rows_skipped_already_filled:
            note = (
                "All matched rows have a prior null_legitimate "
                "fill_status on the target column(s) — already "
                "attempted via bulk_browser. Skipping retries by "
                "default. Pass retry_failed=true to retry, or "
                "start_seq/end_seq to target a fresh row window."
            )
        elif rows_skipped_prior_fail:
            note = (
                f"{rows_skipped_already_filled} rows already filled; "
                f"{rows_skipped_prior_fail} skipped due to prior "
                f"null_legitimate fill_status. Pass retry_failed=true "
                f"to retry."
            )
        else:
            note = "All matched rows already have values in the target columns."
        return ({
            "matched_rows": len(rows),
            "rows_skipped_already_filled": rows_skipped_already_filled,
            "rows_skipped_prior_fail": rows_skipped_prior_fail,
            "processed": 0,
            "cells_filled": 0,
            "by_status": {},
            "note": note,
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
    all_writes: List[Tuple[Any, Dict[str, Any], Dict[str, List[Dict[str, str]]]]] = []
    total_cost = 0.0
    voided_cost = 0.0
    batches_failed = 0
    for r in results:
        if isinstance(r, Exception):
            batches_failed += 1
            log.exception("bulk batch raised: %s", r)
            continue
        traces, writes, cost = r
        all_traces.extend(traces)
        all_writes.extend(writes)
        # Billing gate: non-filled cells contribute only `rate * cost`
        # to the returned total (default rate=0.1 — enough to deter
        # retry-spam, soft enough to feel like a refund on genuine
        # misses). Bulk batches share one BU session cost evenly across
        # rows, so per-row attribution on tr.cost_usd is already correct.
        rate = _api_settings.FAILED_FILL_CHARGE_RATE
        for tr in traces:
            if tr.status == "filled":
                total_cost += tr.cost_usd
            else:
                charged = rate * tr.cost_usd
                total_cost += charged
                voided_cost += tr.cost_usd - charged

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
                # Strategy tag drives the skip-prior-fail pre-filter on
                # subsequent calls. fill.py writes "per_cell"; this is
                # "bulk_browser".
                "strategy": "bulk_browser",
            }
        failed_cols_by_row[tr.row_id] = per_col

    write_db = SessionLocal()
    merged_rows_to_emit: List[Dict[str, Any]] = []
    try:
        for row_id, values, cell_sources in all_writes:
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
            # Persist per-cell sources (BU-cited URLs) under tags.sources
            # in the same wire shape rows_add and the new rows_fill path
            # use, so the FE renders citations uniformly.
            if cell_sources:
                existing_sources = dict(tags.get("sources") or {})
                for col, srcs in cell_sources.items():
                    if srcs and (values.get(col) is not None and values.get(col) != ""):
                        existing_sources[col] = srcs
                if existing_sources:
                    tags["sources"] = existing_sources
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
            for row_id, _values, _srcs in all_writes:
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

    # Auto-verify any newly written emails via Scrubby. browser_use is
    # never a paid-provider source, so provider_emails is empty here —
    # every email goes through Scrubby. Pending tasks are awaited at the
    # end of this function (worker may scale to zero shortly after).
    pending_verifications: List[asyncio.Task] = []
    if email_columns:
        for row_id, values, _srcs in all_writes:
            written = {col: values.get(col) for col in values if col in email_columns}
            if not written:
                continue
            pending_verifications.extend(
                email_verify.schedule_verifications(
                    sample_id=str(row_id),
                    written_values=written,
                    email_columns=email_columns,
                    provider_emails=set(),
                    progress_cb=progress_cb,
                )
            )

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
        "rows_skipped_prior_fail": rows_skipped_prior_fail,
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
    if voided_cost > 0:
        # Internal-audit field — compute we ate (didn't bill the user)
        # via the (1 - FAILED_FILL_CHARGE_RATE) discount on non-filled
        # cells. Same field name as fill.fill_rows so summary readers
        # handle both uniformly.
        summary["voided_cost_usd"] = round(voided_cost, 4)

    # Drain Scrubby verifications. Same rationale as fill.fill_rows — the
    # worker may scale to zero shortly after this returns; in-flight
    # verifies must complete first or the FE never sees the final badge.
    if pending_verifications:
        await asyncio.gather(*pending_verifications, return_exceptions=True)

    return (summary, total_cost)
