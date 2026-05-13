"""User-initiated table edit endpoints.

Distinct from the agent-driven tool calls (which go through the LLM)
and from routes_actions.py (button-triggered fetch_more / enrichment_run).
These are the Sheets-style direct edits:

  PATCH  /v2/projects/{pid}/tables/{tid}/rows/{row_id}        → update cells
  POST   /v2/projects/{pid}/tables/{tid}/rows                 → insert row
  POST   /v2/projects/{pid}/tables/{tid}/rows/delete          → bulk soft-delete
  POST   /v2/projects/{pid}/tables/{tid}/columns              → add column
  PATCH  /v2/projects/{pid}/tables/{tid}/columns/{column}     → patch column meta
"""

from __future__ import annotations

import json
import uuid
from typing import Any, Dict, List, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import text as sa_text
from sqlalchemy.orm import Session

from dsl_api.auth import CurrentUser, get_current_user
from dsl_api.db import SessionLocal
from dsl_api.models import Project


router = APIRouter(prefix="/v2")


def _next_table_short_id(db: Session, project_id: str) -> str:
    """Allocate the next free 't<N>' short id for this project. Mirrors
    the helper in chat_v2/tools.py (kept here to avoid an inter-module
    import cycle)."""
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


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def _verify(project_id: UUID, user_id: UUID, db: Session) -> Project:
    p = db.query(Project).filter(
        Project.id == project_id,
        Project.user_id == user_id,
        Project.deleted_at.is_(None),
    ).first()
    if not p:
        raise HTTPException(404, "Project not found")
    return p


def _resolve_table_uuid(db: Session, project_id: str, id_or_short: str) -> Optional[str]:
    """Accept either short_id (t1) or UUID at route boundaries; return UUID."""
    if not id_or_short:
        return None
    if "-" in id_or_short:
        row = db.execute(
            sa_text(
                "SELECT id::text FROM tables WHERE id=:x AND project_id=:pid AND deleted_at IS NULL"
            ),
            {"x": id_or_short, "pid": project_id},
        ).fetchone()
    else:
        row = db.execute(
            sa_text(
                "SELECT id::text FROM tables WHERE short_id=:x AND project_id=:pid AND deleted_at IS NULL"
            ),
            {"x": id_or_short, "pid": project_id},
        ).fetchone()
    return row[0] if row else None


def _row_to_dict(seq: int, row_data: Any, row_id: str) -> Dict[str, Any]:
    """Mirror list_table_rows' row shape so FE state can merge cleanly."""
    rd = row_data if isinstance(row_data, dict) else (json.loads(row_data) if row_data else {})
    return {"id": row_id, "seq": seq, **rd}


# ---------------------------------------------------------------------------
# Create empty table (source="manual")
# ---------------------------------------------------------------------------


class CreateTableBody(BaseModel):
    name: Optional[str] = None


@router.post("/projects/{project_id}/tables")
def create_table(
    project_id: UUID,
    body: CreateTableBody,
    user: CurrentUser = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Create an empty user-driven table (no source fetch). FE creates
    one of these via the "+ New table" button; the user then adds
    columns and rows directly with the cell/row edit endpoints."""
    _verify(project_id, user.user_id, db)
    name = (body.name or "").strip() or "New table"
    short_id = _next_table_short_id(db, str(project_id))
    new_id = str(uuid.uuid4())
    db.execute(
        sa_text(
            """
            INSERT INTO tables
              (id, project_id, short_id, name, source, query_params, columns,
               fetch_status, created_at)
            VALUES
              (:id, :pid, :sid, :name, 'manual',
               CAST('{}' AS jsonb), CAST('[]' AS jsonb),
               'complete', now())
            """
        ),
        {"id": new_id, "pid": str(project_id), "sid": short_id, "name": name},
    )
    db.commit()
    return {
        "id": short_id,
        "uuid": new_id,
        "name": name,
        "source": "manual",
        "columns": [],
        "row_count": 0,
    }


# ---------------------------------------------------------------------------
# Cell edit
# ---------------------------------------------------------------------------


class EditCellsBody(BaseModel):
    # Partial map of column_name → new value. Pass `null` to clear a cell.
    cells: Dict[str, Any]


@router.patch("/projects/{project_id}/tables/{table_id}/rows/{row_id}")
def patch_row_cells(
    project_id: UUID,
    table_id: str,
    row_id: str,
    body: EditCellsBody,
    user: CurrentUser = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Merge `body.cells` into the row's JSONB. Any column with a null
    value is set to null (not removed) so the agent's downstream
    fill-status logic still has something to attach to."""
    _verify(project_id, user.user_id, db)
    tid = _resolve_table_uuid(db, str(project_id), table_id)
    if not tid:
        raise HTTPException(404, "Table not found")
    existing = db.execute(
        sa_text(
            "SELECT row, seq FROM samples WHERE id=:rid AND table_id=:tid AND deleted_at IS NULL"
        ),
        {"rid": row_id, "tid": tid},
    ).fetchone()
    if not existing:
        raise HTTPException(404, "Row not found")
    current = existing[0] if isinstance(existing[0], dict) else json.loads(existing[0] or "{}")
    current.update(body.cells or {})
    # Clear stale fill_status entries for cells the user just touched —
    # otherwise a "couldn't fill" badge sticks next to a value the user
    # has now provided manually.
    tags_row = db.execute(
        sa_text("SELECT tags FROM samples WHERE id=:rid"),
        {"rid": row_id},
    ).scalar()
    tags = tags_row if isinstance(tags_row, dict) else (json.loads(tags_row) if tags_row else {})
    fs = tags.get("fill_status") if isinstance(tags, dict) else None
    if isinstance(fs, dict):
        changed = False
        for col in list(body.cells.keys()):
            if col in fs:
                del fs[col]
                changed = True
        if changed:
            tags["fill_status"] = fs
    db.execute(
        sa_text(
            "UPDATE samples SET row=CAST(:row AS jsonb), tags=CAST(:tags AS jsonb) "
            "WHERE id=:rid AND table_id=:tid"
        ),
        {
            "row": json.dumps(current),
            "tags": json.dumps(tags) if tags else None,
            "rid": row_id,
            "tid": tid,
        },
    )
    db.commit()
    return {"row": _row_to_dict(int(existing[1]), current, row_id)}


# ---------------------------------------------------------------------------
# Insert row
# ---------------------------------------------------------------------------


class InsertRowBody(BaseModel):
    # Optional starting cell values; columns the row lacks just stay missing.
    data: Optional[Dict[str, Any]] = None
    # If set, the new row is inserted after this row's seq. All later seqs
    # are shifted by +1. If null, the row is appended at the end.
    after_row_id: Optional[str] = None


@router.post("/projects/{project_id}/tables/{table_id}/rows")
def insert_row(
    project_id: UUID,
    table_id: str,
    body: InsertRowBody,
    user: CurrentUser = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    _verify(project_id, user.user_id, db)
    tid = _resolve_table_uuid(db, str(project_id), table_id)
    if not tid:
        raise HTTPException(404, "Table not found")

    # Resolve target version_id — samples are scoped to a version. Pick the
    # most recent version that has samples for this table; fall back to the
    # project's current_version_id when the table is fresh.
    version_row = db.execute(
        sa_text(
            "SELECT version_id::text FROM samples "
            "WHERE table_id=:tid AND deleted_at IS NULL "
            "ORDER BY created_at DESC LIMIT 1"
        ),
        {"tid": tid},
    ).fetchone()
    if version_row:
        version_id = version_row[0]
    else:
        version_id = db.execute(
            sa_text("SELECT current_version_id::text FROM projects WHERE id=:pid"),
            {"pid": str(project_id)},
        ).scalar()
        if not version_id:
            raise HTTPException(400, "Project has no version to attach rows to")

    # Compute the new row's seq. Insert-in-middle requires shifting later
    # seqs by +1 — handled inside a serializable-ish transaction (lock the
    # samples for this table). For append (after_row_id=None or last row)
    # we just take MAX(seq)+1.
    if body.after_row_id:
        ref = db.execute(
            sa_text(
                "SELECT seq FROM samples WHERE id=:rid AND table_id=:tid AND deleted_at IS NULL"
            ),
            {"rid": body.after_row_id, "tid": tid},
        ).fetchone()
        if not ref:
            raise HTTPException(404, "after_row_id not found")
        ref_seq = int(ref[0])
        # Shift later seqs. Two-step to avoid hitting the unique index
        # during update: bump to negative range first, then renumber.
        db.execute(
            sa_text(
                "UPDATE samples SET seq = -(seq + 1) "
                "WHERE version_id=:vid AND seq > :ref"
            ),
            {"vid": version_id, "ref": ref_seq},
        )
        db.execute(
            sa_text(
                "UPDATE samples SET seq = -seq "
                "WHERE version_id=:vid AND seq < 0"
            ),
            {"vid": version_id},
        )
        new_seq = ref_seq + 1
    else:
        max_seq = db.execute(
            sa_text("SELECT COALESCE(MAX(seq), 0) FROM samples WHERE version_id=:vid"),
            {"vid": version_id},
        ).scalar() or 0
        new_seq = int(max_seq) + 1

    new_id = str(uuid.uuid4())
    db.execute(
        sa_text(
            "INSERT INTO samples (id, project_id, table_id, version_id, seq, row, tags, created_at) "
            "VALUES (:id, :pid, :tid, :vid, :seq, CAST(:row AS jsonb), CAST(:tags AS jsonb), now())"
        ),
        {
            "id": new_id,
            "pid": str(project_id),
            "tid": tid,
            "vid": version_id,
            "seq": new_seq,
            "row": json.dumps(body.data or {}),
            "tags": json.dumps({}),
        },
    )
    db.commit()
    return {"row": _row_to_dict(new_seq, body.data or {}, new_id)}


# ---------------------------------------------------------------------------
# Delete rows (bulk soft-delete)
# ---------------------------------------------------------------------------


class DeleteRowsBody(BaseModel):
    row_ids: List[str]


@router.post("/projects/{project_id}/tables/{table_id}/rows/delete")
def delete_rows(
    project_id: UUID,
    table_id: str,
    body: DeleteRowsBody,
    user: CurrentUser = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    _verify(project_id, user.user_id, db)
    tid = _resolve_table_uuid(db, str(project_id), table_id)
    if not tid:
        raise HTTPException(404, "Table not found")
    if not body.row_ids:
        return {"rows_deleted": 0}
    result = db.execute(
        sa_text(
            "UPDATE samples SET deleted_at=now() "
            "WHERE table_id=:tid AND id::text = ANY(:ids) AND deleted_at IS NULL"
        ),
        {"tid": tid, "ids": body.row_ids},
    )
    db.commit()
    return {"rows_deleted": result.rowcount or 0}


# ---------------------------------------------------------------------------
# Add column
# ---------------------------------------------------------------------------


class AddColumnBody(BaseModel):
    name: str
    type: Optional[str] = "text"
    description: Optional[str] = None


@router.post("/projects/{project_id}/tables/{table_id}/columns")
def add_column(
    project_id: UUID,
    table_id: str,
    body: AddColumnBody,
    user: CurrentUser = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    _verify(project_id, user.user_id, db)
    tid = _resolve_table_uuid(db, str(project_id), table_id)
    if not tid:
        raise HTTPException(404, "Table not found")
    name = (body.name or "").strip()
    if not name:
        raise HTTPException(400, "name is required")
    cols_raw = db.execute(
        sa_text("SELECT columns FROM tables WHERE id=:tid"),
        {"tid": tid},
    ).scalar()
    cols: List[Dict[str, Any]] = cols_raw if isinstance(cols_raw, list) else (json.loads(cols_raw) if cols_raw else [])
    if any(isinstance(c, dict) and c.get("name") == name for c in cols):
        raise HTTPException(409, "Column with that name already exists")
    new_col: Dict[str, Any] = {"name": name, "type": body.type or "text"}
    if body.description:
        new_col["description"] = body.description
    cols.append(new_col)
    db.execute(
        sa_text("UPDATE tables SET columns=CAST(:cols AS jsonb) WHERE id=:tid"),
        {"cols": json.dumps(cols), "tid": tid},
    )
    db.commit()
    return {"column": new_col, "columns": cols}


# ---------------------------------------------------------------------------
# Patch column meta (hidden, width, order, description, type, rename)
# ---------------------------------------------------------------------------


class PatchColumnBody(BaseModel):
    hidden: Optional[bool] = None
    width: Optional[int] = None
    order: Optional[int] = None
    description: Optional[str] = None
    type: Optional[str] = None
    new_name: Optional[str] = None


@router.delete("/projects/{project_id}/tables/{table_id}/columns/{column_name}")
def delete_column(
    project_id: UUID,
    table_id: str,
    column_name: str,
    user: CurrentUser = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Remove a column from the table's column list AND strip its key
    from every sample's row / tags JSONB. Soft data loss — there's no
    undo on this side; the agent's row-history view (if/when wired) is
    the recovery path."""
    _verify(project_id, user.user_id, db)
    tid = _resolve_table_uuid(db, str(project_id), table_id)
    if not tid:
        raise HTTPException(404, "Table not found")
    cols_raw = db.execute(
        sa_text("SELECT columns FROM tables WHERE id=:tid"),
        {"tid": tid},
    ).scalar()
    cols: List[Dict[str, Any]] = cols_raw if isinstance(cols_raw, list) else (json.loads(cols_raw) if cols_raw else [])
    idx = next((i for i, c in enumerate(cols) if isinstance(c, dict) and c.get("name") == column_name), -1)
    if idx == -1:
        raise HTTPException(404, "Column not found")
    cols.pop(idx)
    db.execute(
        sa_text("UPDATE tables SET columns=CAST(:cols AS jsonb) WHERE id=:tid"),
        {"cols": json.dumps(cols), "tid": tid},
    )
    # Strip the column key from every sample row + tags subkeys. Cheap
    # for small tables; for huge tables this should move to a job.
    rows = db.execute(
        sa_text(
            "SELECT id::text, row, tags FROM samples "
            "WHERE table_id=:tid AND deleted_at IS NULL"
        ),
        {"tid": tid},
    ).fetchall()
    for rid, row_data, tags_data in rows:
        rd = row_data if isinstance(row_data, dict) else (json.loads(row_data) if row_data else {})
        if column_name in rd:
            del rd[column_name]
            db.execute(
                sa_text("UPDATE samples SET row=CAST(:row AS jsonb) WHERE id=:id"),
                {"row": json.dumps(rd), "id": rid},
            )
        if tags_data:
            td = tags_data if isinstance(tags_data, dict) else json.loads(tags_data)
            changed = False
            for key in ("sources", "fill_status", "email_verification"):
                bucket = td.get(key) if isinstance(td, dict) else None
                if isinstance(bucket, dict) and column_name in bucket:
                    del bucket[column_name]
                    changed = True
            if changed:
                db.execute(
                    sa_text("UPDATE samples SET tags=CAST(:tags AS jsonb) WHERE id=:id"),
                    {"tags": json.dumps(td), "id": rid},
                )
    db.commit()
    return {"columns": cols}


@router.post("/projects/{project_id}/tables/{table_id}/columns/{column_name}/duplicate")
def duplicate_column(
    project_id: UUID,
    table_id: str,
    column_name: str,
    user: CurrentUser = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Append a copy of `column_name` (named `column_name (copy)`) with
    every existing cell's value carried over. New columns inserted via
    add_column don't backfill cells; duplicate explicitly does."""
    _verify(project_id, user.user_id, db)
    tid = _resolve_table_uuid(db, str(project_id), table_id)
    if not tid:
        raise HTTPException(404, "Table not found")
    cols_raw = db.execute(
        sa_text("SELECT columns FROM tables WHERE id=:tid"),
        {"tid": tid},
    ).scalar()
    cols: List[Dict[str, Any]] = cols_raw if isinstance(cols_raw, list) else (json.loads(cols_raw) if cols_raw else [])
    src_idx = next((i for i, c in enumerate(cols) if isinstance(c, dict) and c.get("name") == column_name), -1)
    if src_idx == -1:
        raise HTTPException(404, "Column not found")
    base_name = f"{column_name} (copy)"
    new_name = base_name
    suffix = 2
    existing = {c.get("name") for c in cols if isinstance(c, dict)}
    while new_name in existing:
        new_name = f"{base_name} {suffix}"
        suffix += 1
    new_col = dict(cols[src_idx])
    new_col["name"] = new_name
    cols.insert(src_idx + 1, new_col)
    db.execute(
        sa_text("UPDATE tables SET columns=CAST(:cols AS jsonb) WHERE id=:tid"),
        {"cols": json.dumps(cols), "tid": tid},
    )
    # Backfill cell values from the source column.
    rows = db.execute(
        sa_text(
            "SELECT id::text, row FROM samples WHERE table_id=:tid AND deleted_at IS NULL"
        ),
        {"tid": tid},
    ).fetchall()
    for rid, row_data in rows:
        rd = row_data if isinstance(row_data, dict) else (json.loads(row_data) if row_data else {})
        if column_name in rd:
            rd[new_name] = rd[column_name]
            db.execute(
                sa_text("UPDATE samples SET row=CAST(:row AS jsonb) WHERE id=:id"),
                {"row": json.dumps(rd), "id": rid},
            )
    db.commit()
    return {"column": new_col, "columns": cols}


@router.patch("/projects/{project_id}/tables/{table_id}/columns/{column_name}")
def patch_column(
    project_id: UUID,
    table_id: str,
    column_name: str,
    body: PatchColumnBody,
    user: CurrentUser = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    _verify(project_id, user.user_id, db)
    tid = _resolve_table_uuid(db, str(project_id), table_id)
    if not tid:
        raise HTTPException(404, "Table not found")
    cols_raw = db.execute(
        sa_text("SELECT columns FROM tables WHERE id=:tid"),
        {"tid": tid},
    ).scalar()
    cols: List[Dict[str, Any]] = cols_raw if isinstance(cols_raw, list) else (json.loads(cols_raw) if cols_raw else [])
    idx = next((i for i, c in enumerate(cols) if isinstance(c, dict) and c.get("name") == column_name), -1)
    if idx == -1:
        raise HTTPException(404, "Column not found")
    col = dict(cols[idx])
    if body.hidden is not None:
        col["hidden"] = bool(body.hidden)
    if body.width is not None:
        col["width"] = max(40, int(body.width))
    if body.description is not None:
        col["description"] = body.description
    if body.type is not None:
        col["type"] = body.type
    if body.new_name and body.new_name != column_name:
        if any(c.get("name") == body.new_name for j, c in enumerate(cols) if j != idx and isinstance(c, dict)):
            raise HTTPException(409, "Another column with that name already exists")
        # Rename across all samples' JSONB rows + tags. JSONB keys aren't
        # directly renameable so we do it client-side via Python.
        rows = db.execute(
            sa_text(
                "SELECT id::text, row, tags FROM samples "
                "WHERE table_id=:tid AND deleted_at IS NULL"
            ),
            {"tid": tid},
        ).fetchall()
        for rid, row_data, tags_data in rows:
            rd = row_data if isinstance(row_data, dict) else (json.loads(row_data) if row_data else {})
            if column_name in rd:
                rd[body.new_name] = rd.pop(column_name)
                db.execute(
                    sa_text("UPDATE samples SET row=CAST(:row AS jsonb) WHERE id=:id"),
                    {"row": json.dumps(rd), "id": rid},
                )
            if tags_data:
                td = tags_data if isinstance(tags_data, dict) else json.loads(tags_data)
                changed = False
                for key in ("sources", "fill_status", "email_verification"):
                    bucket = td.get(key) if isinstance(td, dict) else None
                    if isinstance(bucket, dict) and column_name in bucket:
                        bucket[body.new_name] = bucket.pop(column_name)
                        changed = True
                if changed:
                    db.execute(
                        sa_text("UPDATE samples SET tags=CAST(:tags AS jsonb) WHERE id=:id"),
                        {"tags": json.dumps(td), "id": rid},
                    )
        col["name"] = body.new_name
    cols[idx] = col

    # Reorder columns when an `order` index is supplied.
    if body.order is not None:
        new_idx = max(0, min(int(body.order), len(cols) - 1))
        item = cols.pop(idx)
        cols.insert(new_idx, item)

    db.execute(
        sa_text("UPDATE tables SET columns=CAST(:cols AS jsonb) WHERE id=:tid"),
        {"cols": json.dumps(cols), "tid": tid},
    )
    db.commit()
    return {"columns": cols}
