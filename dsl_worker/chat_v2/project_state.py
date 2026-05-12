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
            SELECT id::text, name, source, query_params, columns, dedup_key_column,
                   last_fetch_returned_rows, last_fetch_cost_credits, last_fetch_at,
                   fetch_status, fetch_error
            FROM tables
            WHERE project_id = :pid AND deleted_at IS NULL
            ORDER BY created_at
            LIMIT :limit
            """
        ),
        {"pid": project_id, "limit": max_tables},
    ).fetchall()

    parts.append("Tables:")
    if not tables:
        parts.append("  (none yet)")
    for t in tables:
        (tid, name, source, qp, columns, dedup_col,
         last_rows, last_cost, last_at, fetch_status, fetch_err) = t

        row_count = db.execute(
            sa_text("SELECT COUNT(*) FROM samples WHERE table_id = :tid AND deleted_at IS NULL"),
            {"tid": tid},
        ).scalar() or 0

        parts.append(f"  - {name} ({row_count} rows, source: {source})")
        if isinstance(qp, str):
            qp = json.loads(qp)
        # Strip internal _cursor/_pending_rows from display
        qp_display = {k: v for k, v in (qp or {}).items() if not k.startswith("_")}
        if qp_display:
            parts.append(f"      Query: {json.dumps(qp_display, default=str)[:200]}")

        cols = json.loads(columns) if isinstance(columns, str) else (columns or [])
        if cols:
            col_summary = ", ".join(f"{c['name']} ({c['type']})" for c in cols[:12])
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

        # Filters
        filters = db.execute(
            sa_text(
                "SELECT column_name, op, value FROM table_filters WHERE table_id = :tid"
            ),
            {"tid": tid},
        ).fetchall()
        if filters:
            for col, op, val in filters:
                parts.append(f"      Filter: {col} {op} {json.dumps(val)}")

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

    # Enrichments configured
    enrichments = db.execute(
        sa_text(
            """
            SELECT e.id::text, e.name, e.table_id::text, t.name AS table_name, e.columns,
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
        for (eid, ename, tid, tname, cols, cap, last_filled, last_cost) in enrichments:
            cols_list = json.loads(cols) if isinstance(cols, str) else (cols or [])
            col_names = ", ".join(c["name"] for c in cols_list[:6])
            parts.append(f"  - {ename} (id={eid[:8]}) on table {tname}")
            parts.append(f"      Fills: {col_names}; cap: {cap} cr/row")
            if last_filled is not None:
                cost_s = f", {float(last_cost):.1f} cr" if last_cost is not None else ""
                parts.append(f"      Last run: {last_filled} rows{cost_s}")

    body = "\n".join(parts)
    return f"<project_state>\n{body}\n</project_state>"


def _human_ago(seconds: float) -> str:
    if seconds < 60:
        return f"{int(seconds)}s ago"
    if seconds < 3600:
        return f"{int(seconds // 60)} min ago"
    if seconds < 86400:
        return f"{int(seconds // 3600)} hr ago"
    return f"{int(seconds // 86400)} days ago"


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
