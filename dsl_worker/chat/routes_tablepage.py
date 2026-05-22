"""TablePage.ai integration routes.

Two endpoints — both mounted under /v2 via this module's `router`:

  POST /v2/projects/{pid}/tables/{tid}/tablepage/push
    Dumps the table's current rows + visible columns as CSV, uploads to
    TablePage via /api/bot-upload-json, stores the returned slug on the
    table row. Returns { slug, url, uploaded_at }.

  GET  /v2/projects/{pid}/tables/{tid}/tablepage
    Returns the persisted slug + last-uploaded timestamp + a live TP
    `ready` probe so the UI knows when the iframe will render insights.

Why this layer exists at all:
- TablePage's API has no update-by-slug or delete-by-key. Every push
  creates a brand new dataset. We persist the most recent slug here so
  the UI can keep showing a stable embed across reloads without
  re-uploading every time.
- The TP API key is stored server-side (TABLEPAGE_API_KEY env). We
  never expose it to the browser.
- TP rate-limits uploads to 10/day per bot; the UI only triggers this
  on explicit user click ("Push to TablePage") so the limit is hard to
  burn through during normal use.
"""

from __future__ import annotations

import csv
import io
import logging
import re
from typing import Any, Dict, Optional
from uuid import UUID

import httpx
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import text as sa_text
from sqlalchemy.orm import Session

from dsl_api.auth import CurrentUser, get_current_user
from dsl_api.db import get_db
from dsl_worker.config import settings


log = logging.getLogger(__name__)

router = APIRouter(prefix="/v2")

TABLEPAGE_BASE = "https://data.tablepage.ai"
UPLOAD_PATH = "/api/bot-upload-json"
STATUS_PATH_TMPL = "/api/d/{slug}/status"

# Row-level sidecars we add to samples.row internally — strip from CSV.
# (__cell_status__, __cell_cost__ are the two reserved keys; anything
# starting with __ is excluded as a guard against future additions.)
_RESERVED_ROW_KEYS_PREFIX = "__"

# Hard ceiling. TablePage's own limit is 25 MB; we cut earlier to leave
# headroom for CSV escaping overhead and to avoid sending a giant payload
# only to be rejected at the edge.
_MAX_CSV_BYTES = 20 * 1024 * 1024


def _stringify_cell(v: Any) -> str:
    """Render any JSONB cell value into a CSV-friendly string.

    None → empty string. Numbers / bools render via str(). Lists/dicts
    serialize to JSON so structured cells survive the round trip
    (TablePage will treat them as text, but at least the data is
    preserved instead of dropped)."""
    if v is None:
        return ""
    if isinstance(v, bool):
        return "Yes" if v else "No"
    if isinstance(v, (str, int, float)):
        return str(v)
    # list / dict → compact JSON. TP treats as text; user can still see it.
    import json
    return json.dumps(v, ensure_ascii=False, separators=(",", ":"), default=str)


def _slugify_filename(name: str) -> str:
    """Make a CSV filename TP-safe. TP slugifies internally but the
    filename also flows into the public URL prefix, so reduce to a
    reasonable shape upfront."""
    if not name:
        return "table.csv"
    s = re.sub(r"[^A-Za-z0-9._-]+", "-", name).strip("-")
    if not s:
        s = "table"
    if not s.lower().endswith(".csv"):
        s = s + ".csv"
    return s[:120]


def _build_csv(columns: list[Dict[str, Any]], rows: list[Dict[str, Any]]) -> str:
    """Build a CSV from the table's column list + row dicts. Column
    order follows the table's columns array (which already encodes the
    user-visible order including any reorder). Hidden columns are
    omitted — TablePage shouldn't render a column the user has hidden
    in our UI."""
    visible_cols = [
        c for c in columns
        if isinstance(c, dict) and c.get("name") and not c.get("hidden")
    ]
    if not visible_cols:
        # Fall back to whatever keys the first row has, minus reserved.
        if rows:
            keys = sorted(
                k for k in rows[0].keys()
                if not k.startswith(_RESERVED_ROW_KEYS_PREFIX)
            )
            visible_cols = [{"name": k} for k in keys]
        else:
            visible_cols = [{"name": "value"}]

    header = [c["name"] for c in visible_cols]
    buf = io.StringIO()
    writer = csv.writer(buf, quoting=csv.QUOTE_MINIMAL)
    writer.writerow(header)
    for row in rows:
        if not isinstance(row, dict):
            continue
        writer.writerow([_stringify_cell(row.get(c["name"])) for c in visible_cols])
    return buf.getvalue()


def _verify_project_owns_user(db: Session, project_id: UUID, user_id: UUID) -> None:
    """Subset of routes.py:_verify_project — minimal ownership check.

    We keep a local copy to avoid importing the larger routes.py module
    here (and the circular-import risk that comes with it)."""
    row = db.execute(
        sa_text(
            "SELECT 1 FROM projects "
            "WHERE id=:pid AND user_id=:uid AND deleted_at IS NULL"
        ),
        {"pid": str(project_id), "uid": str(user_id)},
    ).fetchone()
    if not row:
        raise HTTPException(404, "Project not found")


def _resolve_table_uuid(db: Session, project_id: str, id_or_short: str) -> Optional[str]:
    """Accept either a UUID or short_id (e.g. 't1'); return the UUID."""
    # Cheap UUID heuristic — full validation happens via the query.
    if len(id_or_short) == 36 and id_or_short.count("-") == 4:
        row = db.execute(
            sa_text(
                "SELECT id::text FROM tables "
                "WHERE id=:tid AND project_id=:pid AND deleted_at IS NULL"
            ),
            {"tid": id_or_short, "pid": project_id},
        ).fetchone()
    else:
        row = db.execute(
            sa_text(
                "SELECT id::text FROM tables "
                "WHERE short_id=:sid AND project_id=:pid AND deleted_at IS NULL"
            ),
            {"sid": id_or_short, "pid": project_id},
        ).fetchone()
    return row[0] if row else None


@router.post("/projects/{project_id}/tables/{table_id}/tablepage/push")
def push_to_tablepage(
    project_id: UUID,
    table_id: str,
    user: CurrentUser = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Push current table state to TablePage.ai. Manual-only trigger.

    Always uploads ALL rows (no filter applied) — the visualization is
    of the dataset, not a slice. Hidden columns are excluded.

    Returns the new slug + embed URL. On TP failure, returns a 502 with
    the upstream error so the FE can show a clear message instead of a
    generic 500.
    """
    if not settings.tablepage_api_key:
        raise HTTPException(503, "TablePage integration not configured")

    _verify_project_owns_user(db, project_id, user.user_id)
    tid = _resolve_table_uuid(db, str(project_id), table_id)
    if not tid:
        raise HTTPException(404, "Table not found")

    meta_row = db.execute(
        sa_text("SELECT name, columns FROM tables WHERE id=:tid"),
        {"tid": tid},
    ).fetchone()
    if not meta_row:
        raise HTTPException(404, "Table not found")
    table_name, columns = meta_row[0], (meta_row[1] or [])

    sample_rows = db.execute(
        sa_text(
            "SELECT row FROM samples "
            "WHERE table_id=:tid AND deleted_at IS NULL "
            "ORDER BY seq ASC"
        ),
        {"tid": tid},
    ).fetchall()
    rows = [r[0] for r in sample_rows if isinstance(r[0], dict)]
    if not rows:
        raise HTTPException(
            400,
            "No rows to push — fetch or fill the table first.",
        )

    csv_text = _build_csv(columns, rows)
    csv_bytes = csv_text.encode("utf-8")
    if len(csv_bytes) > _MAX_CSV_BYTES:
        raise HTTPException(
            413,
            f"CSV exceeds TablePage size limit ({len(csv_bytes)} bytes, "
            f"cap {_MAX_CSV_BYTES}). Reduce rows or columns and retry.",
        )

    filename = _slugify_filename(table_name)
    try:
        r = httpx.post(
            f"{TABLEPAGE_BASE}{UPLOAD_PATH}",
            headers={
                "Authorization": f"Bearer {settings.tablepage_api_key}",
                "Content-Type": "application/json",
            },
            json={"filename": filename, "content": csv_text},
            timeout=60.0,
        )
    except httpx.HTTPError as e:
        log.warning("tablepage upload network error: %s", e)
        raise HTTPException(502, f"TablePage unreachable: {e}")

    if r.status_code != 200:
        log.warning(
            "tablepage upload non-200: status=%s body=%s",
            r.status_code, r.text[:300],
        )
        raise HTTPException(502, f"TablePage upload failed ({r.status_code})")

    data = r.json()
    slug = data.get("slug")
    if not slug:
        raise HTTPException(502, "TablePage response missing slug")

    db.execute(
        sa_text(
            "UPDATE tables SET tablepage_slug=:s, tablepage_uploaded_at=now() "
            "WHERE id=:tid"
        ),
        {"s": slug, "tid": tid},
    )
    db.commit()

    return {
        "slug": slug,
        "url": data.get("url") or f"{TABLEPAGE_BASE}/d/{slug}",
        "embed_url": f"{TABLEPAGE_BASE}/d/{slug}?embed=1",
        "rows_uploaded": len(rows),
    }


@router.get("/projects/{project_id}/tables/{table_id}/tablepage")
def get_tablepage_state(
    project_id: UUID,
    table_id: str,
    user: CurrentUser = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Return persisted slug + a live TP `ready` probe.

    `ready=true` means TablePage has finished its automatic analysis +
    chart generation, so the iframe will render insights instead of a
    spinner. The FE polls this endpoint every few seconds after a push
    until ready=true.
    """
    _verify_project_owns_user(db, project_id, user.user_id)
    tid = _resolve_table_uuid(db, str(project_id), table_id)
    if not tid:
        raise HTTPException(404, "Table not found")

    row = db.execute(
        sa_text(
            "SELECT tablepage_slug, tablepage_uploaded_at FROM tables "
            "WHERE id=:tid"
        ),
        {"tid": tid},
    ).fetchone()
    slug = row[0] if row else None
    uploaded_at = row[1] if row else None

    if not slug:
        return {
            "slug": None,
            "uploaded_at": None,
            "url": None,
            "embed_url": None,
            "ready": False,
        }

    ready = False
    try:
        s = httpx.get(
            f"{TABLEPAGE_BASE}{STATUS_PATH_TMPL.format(slug=slug)}",
            timeout=5.0,
        )
        if s.status_code == 200:
            ready = bool((s.json() or {}).get("ready"))
    except httpx.HTTPError:
        # Surface the persisted slug even when TP is unreachable —
        # the embed iframe handles its own loading state, so the FE
        # can still render. ready=false just means "we don't know."
        pass

    return {
        "slug": slug,
        "uploaded_at": uploaded_at.isoformat() if uploaded_at else None,
        "url": f"{TABLEPAGE_BASE}/d/{slug}",
        "embed_url": f"{TABLEPAGE_BASE}/d/{slug}?embed=1",
        "ready": ready,
    }
