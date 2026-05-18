"""Project state banner — auto-injected before each user message in a chat run.

Surfaces the agent's situational awareness: tables, columns, row counts,
filters, enrichments configured, fetch status (so agent knows what's mid-flight),
last fetch cost info for empirical estimates.

Cost: ~200-1000 tokens per turn depending on project size. Cached as part of
prompt context if the agent's framework supports it.
"""

from __future__ import annotations

import datetime as dt
import json
from collections import Counter
from typing import Any, Dict, List

from sqlalchemy import text as sa_text
from sqlalchemy.orm import Session


def build_project_state(db: Session, project_id: str, max_tables: int = 10) -> str:
    """Return the project state XML block as a string."""
    today = dt.date.today()
    parts: List[str] = []
    parts.append(f"Today is {today.isoformat()} ({today.strftime('%A')}).\n")

    # Tables
    tables = db.execute(
        sa_text(
            """
            SELECT id::text, short_id, name, source, query_params, columns, dedup_key_column,
                   last_fetch_returned_rows, last_fetch_cost_credits, last_fetch_at,
                   fetch_status, fetch_error, sort_column, sort_direction
            FROM tables
            WHERE project_id = :pid AND deleted_at IS NULL
            ORDER BY created_at
            LIMIT :limit
            """
        ),
        {"pid": project_id, "limit": max_tables},
    ).fetchall()

    parts.append("Tables (already exist on this project — extend these instead of duplicating):")
    if not tables:
        parts.append("  (none yet)")
    for t in tables:
        (tid, short_id, name, source, qp, columns, dedup_col,
         last_rows, last_cost, last_at, fetch_status, fetch_err,
         sort_column, sort_direction) = t

        row_count = db.execute(
            sa_text("SELECT COUNT(*) FROM samples WHERE table_id = :tid AND deleted_at IS NULL"),
            {"tid": tid},
        ).scalar() or 0

        parts.append(f"  - [{short_id}] {name} ({row_count} rows, source: {source})")
        if isinstance(qp, str):
            qp = json.loads(qp)
        # Strip internal _cursor/_pending_rows from display
        qp_display = {k: v for k, v in (qp or {}).items() if not k.startswith("_")}
        if qp_display:
            parts.append(f"      Query: {json.dumps(qp_display, default=str)[:200]}")

        cols = json.loads(columns) if isinstance(columns, str) else (columns or [])
        if cols:
            # Per-column fill rate so the agent knows which columns are
            # actually empty vs already populated. Caps at 500 rows scanned.
            fill_rates: Dict[str, int] = {}
            if row_count > 0:
                row_iter = db.execute(
                    sa_text(
                        "SELECT row FROM samples WHERE table_id=:tid AND deleted_at IS NULL LIMIT 500"
                    ),
                    {"tid": tid},
                ).fetchall()
                sample_n = len(row_iter)
                for c in cols:
                    cname = c.get("name")
                    if not cname:
                        continue
                    filled = sum(
                        1 for (r,) in row_iter
                        if isinstance(r, dict) and r.get(cname) not in (None, "", [], {})
                    )
                    if sample_n > 0:
                        fill_rates[cname] = int(round(100 * filled / sample_n))

            def _col_label(c: Dict[str, Any]) -> str:
                pct = fill_rates.get(c["name"])
                if pct is None:
                    return f"{c['name']} ({c['type']})"
                return f"{c['name']} ({c['type']}, {pct}% filled)"

            col_summary = ", ".join(_col_label(c) for c in cols[:12])
            if len(cols) > 12:
                col_summary += f", +{len(cols) - 12} more"
            parts.append(f"      Columns: {col_summary}")

        if dedup_col:
            parts.append(f"      dedup_key_column: {dedup_col}")

        # Class breakdown for low-cardinality columns (enum-shaped)
        for col in cols[:8]:
            if col.get("type") != "enum":
                continue
            breakdown = _column_value_breakdown(db, tid, col["name"], limit=10)
            if breakdown:
                pretty = ", ".join(f"{v} ({n})" for v, n in breakdown.items())
                parts.append(f"      Class breakdown — {col['name']}: {pretty}")

        # Filters. Display the canonical op form even if the row was
        # written with a legacy symbol (`>=` → `gte`, etc.) so the agent's
        # own state log uses one vocabulary — matches the schema enum it
        # sees on filter_set and prevents the agent from learning multiple
        # forms of the same op from its own state.
        filters = db.execute(
            sa_text(
                "SELECT column_name, op, value FROM table_filters WHERE table_id = :tid"
            ),
            {"tid": tid},
        ).fetchall()
        if filters:
            for col, op, val in filters:
                canonical_op = _canonical_op_for_display(op)
                parts.append(f"      Filter: {col} {canonical_op} {json.dumps(val)}")

        # Active sort — surfaces so the agent knows "more rows" or
        # "first N" requests will land in the sorted order, and so
        # it doesn't propose a sort that's already applied.
        if sort_column:
            parts.append(f"      Sort: {sort_column} {sort_direction or 'desc'}")

        # Fetch status + last fetch info
        if fetch_status not in ("idle", "complete"):
            status_line = f"      ⚠ fetch_status: {fetch_status}"
            if fetch_status == "failed" and fetch_err:
                status_line += f" — {fetch_err[:120]}"
            parts.append(status_line)
        if last_rows is not None:
            credits = f"{float(last_cost):.1f}" if last_cost is not None else "?"
            timing = ""
            if last_at:
                age_s = (dt.datetime.now(dt.timezone.utc) - last_at).total_seconds()
                timing = _human_ago(age_s)
            parts.append(f"      Last fetch: {last_rows} rows, {credits} credits, {timing}")
            if last_rows == 0:
                parts.append("        (returned 0 — source may be tapped out for this query)")

    # Uploaded files — agent uses these via source="file" + file_id.
    # Without surfacing them here the agent has no idea any file exists,
    # which reads as "files aren't hooked up" even when the user
    # successfully uploaded.
    files = db.execute(
        sa_text(
            """
            SELECT id::text, filename, content_type, size_bytes, status
            FROM project_files
            WHERE project_id = :pid AND deleted_at IS NULL
            ORDER BY created_at
            LIMIT 50
            """
        ),
        {"pid": project_id},
    ).fetchall()
    uploaded = [f for f in files if (f[4] or "").lower() == "uploaded"]
    if uploaded:
        parts.append("\nUploaded files (reference by file_id with source=\"file\"):")
        for (fid, fname, ctype, size, _status) in uploaded:
            size_str = _human_size(size)
            type_str = f" {ctype}" if ctype else ""
            parts.append(f"  - {fname} ({size_str}{type_str}) — file_id: {fid}")

    # Enrichments configured
    enrichments = db.execute(
        sa_text(
            """
            SELECT e.short_id, e.name, t.short_id AS table_short, t.name AS table_name, e.columns,
                   e.per_row_credit_cap, e.last_run_filled_rows, e.last_run_cost_credits
            FROM enrichments e
            JOIN tables t ON t.id = e.table_id
            WHERE t.project_id = :pid AND e.deleted_at IS NULL
            ORDER BY e.created_at
            LIMIT 20
            """
        ),
        {"pid": project_id},
    ).fetchall()
    if enrichments:
        parts.append("\nEnrichments configured:")
        for (esid, ename, tsid, tname, cols, cap, last_filled, last_cost) in enrichments:
            cols_list = json.loads(cols) if isinstance(cols, str) else (cols or [])
            col_names = ", ".join(c["name"] for c in cols_list[:6])
            parts.append(f"  - [{esid}] {ename} on table [{tsid}] {tname}")
            parts.append(f"      Fills: {col_names}; cap: {cap} cr/row")
            if last_filled is not None:
                cost_s = f", {float(last_cost):.1f} cr" if last_cost is not None else ""
                parts.append(f"      Last run: {last_filled} rows{cost_s}")

    body = "\n".join(parts)
    return f"<project_state>\n{body}\n</project_state>"


def _human_size(n: int | None) -> str:
    if not n:
        return "?"
    if n < 1024:
        return f"{n} B"
    if n < 1024 * 1024:
        return f"{n / 1024:.1f} KB"
    if n < 1024 * 1024 * 1024:
        return f"{n / (1024 * 1024):.1f} MB"
    return f"{n / (1024 * 1024 * 1024):.1f} GB"


def _human_ago(seconds: float) -> str:
    if seconds < 60:
        return f"{int(seconds)}s ago"
    if seconds < 3600:
        return f"{int(seconds // 60)} min ago"
    if seconds < 86400:
        return f"{int(seconds // 3600)} hr ago"
    return f"{int(seconds // 86400)} days ago"


_OP_DISPLAY_ALIASES = {
    ">": "gte", ">=": "gte", "gt": "gte",
    "<": "lte", "<=": "lte", "lt": "lte",
    "in": "is_any_of", "any_of": "is_any_of",
    "text_include_exclude": "text_inc_exc",
    "isnull": "is_null", "is_empty": "is_null",
    "is_not_empty": "is_not_null", "exists": "is_not_null",
}


def _canonical_op_for_display(op: str) -> str:
    """Map any stored filter op to the canonical 7-op form for the project
    state banner. Display-only; the SQL serving path still tolerates both.
    """
    return _OP_DISPLAY_ALIASES.get((op or "").strip().lower(), op or "")


def _column_value_breakdown(db: Session, table_id: str, column_name: str, limit: int = 10) -> Dict[str, int]:
    """Top-N value frequencies for a column. Returns {} if cardinality is high."""
    rows = db.execute(
        sa_text(
            "SELECT row FROM samples WHERE table_id = :tid AND deleted_at IS NULL LIMIT 500"
        ),
        {"tid": table_id},
    ).fetchall()
    counter: Counter = Counter()
    for (row,) in rows:
        if not isinstance(row, dict):
            continue
        v = row.get(column_name)
        if v is None or v == "":
            continue
        counter[str(v)[:60]] += 1
    if len(counter) > 20 or not counter:
        return {}
    return dict(counter.most_common(limit))
