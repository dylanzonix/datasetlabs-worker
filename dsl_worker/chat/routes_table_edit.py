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
from dsl_api.models.entity_comment import EntityComment


router = APIRouter(prefix="/v2")


def _next_table_short_id(db: Session, project_id: str) -> str:
    """Allocate the next free 't<N>' short id for this project. Mirrors
    the helper in chat/tools.py (kept here to avoid an inter-module
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


class CreateTableFromFileBody(BaseModel):
    file_id: str
    name: Optional[str] = None


@router.post("/projects/{project_id}/tables/from-file")
async def create_table_from_file(
    project_id: UUID,
    body: CreateTableFromFileBody,
    user: CurrentUser = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Create a new table directly from an already-uploaded file
    (project_files.id). Skips the LLM — calls the same `table_create`
    helper the agent does, but driven from the FE for a one-click
    "New table from file" flow.

    Body: { file_id, name? }
    """
    _verify(project_id, user.user_id, db)

    # Confirm the file belongs to the project and is uploaded.
    row = db.execute(
        sa_text(
            """
            SELECT filename FROM project_files
            WHERE id=:fid AND project_id=:pid AND deleted_at IS NULL
              AND status='uploaded'
            """
        ),
        {"fid": body.file_id, "pid": str(project_id)},
    ).fetchone()
    if not row:
        raise HTTPException(404, "File not found")
    filename = row[0]

    # Default the table name to the file's base name (without extension).
    name = (body.name or "").strip()
    if not name:
        stem = filename
        if "." in stem:
            stem = stem.rsplit(".", 1)[0]
        name = stem or "New table"

    from dsl_worker.chat.tools import table_create, ToolContext
    ctx = ToolContext(
        db=db,
        project_id=str(project_id),
        user_id=str(user.user_id),
        run_id=None,
    )
    # File uploads should pull every row the user gave us. table_create
    # defaults n=100 (sensible for Apollo / FE / web search / etc.) but
    # for a CSV/XLSX the user already chose the row count by what's in
    # the file — truncating at 100 silently drops the rest. Pass an
    # effectively-unlimited n so the file adapter reads to EOF.
    result, _cost = await table_create(
        {
            "source": "file",
            "query_params": {"file_id": body.file_id},
            "name": name,
            "n": 10_000_000,
        },
        ctx,
    )
    if "error" in result:
        raise HTTPException(400, result["error"])
    return result


@router.delete("/projects/{project_id}/tables/{table_id}")
def delete_table(
    project_id: UUID,
    table_id: str,
    user: CurrentUser = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Soft-delete a table (sets deleted_at). The samples are kept
    intact (each has its own deleted_at logic in the sample query),
    but the table no longer surfaces via list_tables / project_state.
    Idempotent — returns ok=true even if the table was already gone."""
    _verify(project_id, user.user_id, db)
    tid = _resolve_table_uuid(db, str(project_id), table_id)
    if not tid:
        raise HTTPException(404, "Table not found")
    db.execute(
        sa_text(
            "UPDATE tables SET deleted_at = now() "
            "WHERE id=:tid AND project_id=:pid AND deleted_at IS NULL"
        ),
        {"tid": tid, "pid": str(project_id)},
    )
    db.commit()
    return {"ok": True, "id": table_id, "uuid": tid}


class ReorderTablesBody(BaseModel):
    # Ordered list of table short_ids (or UUIDs). Tables not in the list
    # are appended to the end in their original order.
    table_ids: list[str]


@router.post("/projects/{project_id}/tables/reorder")
def reorder_tables(
    project_id: UUID,
    body: ReorderTablesBody,
    user: CurrentUser = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Reorder tables by rewriting their created_at timestamps in the
    given order. We sort tables by created_at on read, so spacing them
    by 1ms in the requested sequence gives us the new tab order.
    Cheap, no migration needed for a sort_order column."""
    _verify(project_id, user.user_id, db)
    if not body.table_ids:
        return {"ok": True}
    # Resolve every id to a uuid, in order.
    uuids: list[str] = []
    for raw in body.table_ids:
        u = _resolve_table_uuid(db, str(project_id), raw)
        if u:
            uuids.append(u)
    if not uuids:
        return {"ok": True}
    # Anchor 1 hour ago so a concurrent insert (server clock = now)
    # doesn't end up earlier than our reordered set.
    db.execute(
        sa_text(
            """
            WITH new_order AS (
              SELECT unnest(CAST(:ids AS uuid[])) AS id,
                     generate_series(0, cardinality(CAST(:ids AS uuid[])) - 1) AS pos
            )
            UPDATE tables t
            SET created_at = (now() - INTERVAL '1 hour') + (new_order.pos * INTERVAL '1 millisecond')
            FROM new_order
            WHERE t.id = new_order.id
              AND t.project_id = :pid
              AND t.deleted_at IS NULL
            """
        ),
        {"ids": uuids, "pid": str(project_id)},
    )
    db.commit()
    return {"ok": True, "order": uuids}


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
# Patch table (rename) / delete / duplicate
# ---------------------------------------------------------------------------


class PatchTableBody(BaseModel):
    name: Optional[str] = None


@router.patch("/projects/{project_id}/tables/{table_id}")
def patch_table(
    project_id: UUID,
    table_id: str,
    body: PatchTableBody,
    user: CurrentUser = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Rename the table. Currently only `name` is patchable from the UI;
    other fields (source/query_params/columns) are agent-managed."""
    _verify(project_id, user.user_id, db)
    tid = _resolve_table_uuid(db, str(project_id), table_id)
    if not tid:
        raise HTTPException(404, "Table not found")
    name = (body.name or "").strip()
    if not name:
        raise HTTPException(400, "name is required")
    db.execute(
        sa_text("UPDATE tables SET name = :name WHERE id = :id"),
        {"name": name, "id": tid},
    )
    db.commit()
    return {"id": table_id, "uuid": tid, "name": name}


@router.delete("/projects/{project_id}/tables/{table_id}")
def delete_table(
    project_id: UUID,
    table_id: str,
    user: CurrentUser = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Soft-delete a table. Rows / enrichments / filters remain in the DB
    via the CASCADE-on-tables-deletion FKs but are filtered out of FE
    queries by the `deleted_at IS NULL` predicate."""
    _verify(project_id, user.user_id, db)
    tid = _resolve_table_uuid(db, str(project_id), table_id)
    if not tid:
        raise HTTPException(404, "Table not found")
    db.execute(
        sa_text("UPDATE tables SET deleted_at = now() WHERE id = :id"),
        {"id": tid},
    )
    db.commit()
    return {"ok": True, "id": table_id, "uuid": tid}


@router.post("/projects/{project_id}/tables/{table_id}/duplicate")
def duplicate_table(
    project_id: UUID,
    table_id: str,
    user: CurrentUser = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Clone a table — copies the Table row (name + " (copy)"), all
    samples (rows), and any enrichments. Filters and the active
    sort/cursor are intentionally NOT copied (they're view state, not
    intrinsic to the dataset). Re-running fetches against the copy is
    a separate user action."""
    _verify(project_id, user.user_id, db)
    src_tid = _resolve_table_uuid(db, str(project_id), table_id)
    if not src_tid:
        raise HTTPException(404, "Table not found")
    src = db.execute(
        sa_text(
            """
            SELECT name, source, query_params, columns, dedup_key_column,
                   fetch_status
            FROM tables WHERE id = :id
            """
        ),
        {"id": src_tid},
    ).fetchone()
    if not src:
        raise HTTPException(404, "Table not found")
    new_id = str(uuid.uuid4())
    new_short = _next_table_short_id(db, str(project_id))
    new_name = f"{src[0]} (copy)"
    # `fetch_status` defaults to 'complete' on the copy — no in-flight
    # background job was inherited.
    db.execute(
        sa_text(
            """
            INSERT INTO tables
              (id, project_id, short_id, name, source, query_params, columns,
               dedup_key_column, fetch_status, created_at)
            VALUES
              (:id, :pid, :sid, :name, :source,
               CAST(:qp AS jsonb), CAST(:cols AS jsonb),
               :dedup, 'complete', now())
            """
        ),
        {
            "id": new_id,
            "pid": str(project_id),
            "sid": new_short,
            "name": new_name,
            "source": src[1],
            "qp": json.dumps(src[2] if isinstance(src[2], (dict, list)) else (src[2] or {})),
            "cols": json.dumps(src[3] if isinstance(src[3], (dict, list)) else (src[3] or [])),
            "dedup": src[4],
        },
    )
    # Copy samples in one SQL statement — preserve seq order so the
    # copied table renders identically.
    db.execute(
        sa_text(
            """
            INSERT INTO samples
              (id, project_id, table_id, seq, row, raw_row, tags,
               enrichment_data, created_at, version_id)
            SELECT
              gen_random_uuid(), project_id, :new_tid, seq, row, raw_row,
              tags, enrichment_data, now(), version_id
            FROM samples
            WHERE table_id = :src_tid AND deleted_at IS NULL
            """
        ),
        {"new_tid": new_id, "src_tid": src_tid},
    )
    # Copy enrichments (each gets a fresh id + short_id; per-row run
    # state is left empty so the user can re-run on the copy).
    enrichments = db.execute(
        sa_text(
            "SELECT name, columns, action, per_row_credit_cap FROM enrichments "
            "WHERE table_id = :tid AND deleted_at IS NULL ORDER BY created_at"
        ),
        {"tid": src_tid},
    ).fetchall()
    for i, e in enumerate(enrichments, start=1):
        db.execute(
            sa_text(
                """
                INSERT INTO enrichments
                  (id, table_id, short_id, name, columns, action,
                   per_row_credit_cap, position, created_at)
                VALUES
                  (gen_random_uuid(), :tid, :sid, :name,
                   CAST(:cols AS jsonb), CAST(:action AS jsonb),
                   :cap, :pos, now())
                """
            ),
            {
                "tid": new_id,
                "sid": f"e{i}",
                "name": e[0],
                "cols": json.dumps(e[1] if isinstance(e[1], (dict, list)) else (e[1] or [])),
                "action": json.dumps(e[2] if isinstance(e[2], (dict, list)) else (e[2] or {})),
                "cap": e[3],
                "pos": i,
            },
        )
    db.commit()
    return {
        "id": new_short,
        "uuid": new_id,
        "name": new_name,
        "source": src[1],
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
    pinned: Optional[bool] = None
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
    if body.pinned is not None:
        col["pinned"] = bool(body.pinned)
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


# ---------------------------------------------------------------------------
# Entity comments — description thread per table / column
# ---------------------------------------------------------------------------


class EntityCommentCreate(BaseModel):
    body: str


def _comment_to_dict(c: EntityComment) -> Dict[str, Any]:
    return {
        "id": str(c.id),
        "project_id": str(c.project_id),
        "table_id": str(c.table_id),
        "column_name": c.column_name,
        "author": c.author,
        "body": c.body,
        "created_by": str(c.created_by) if c.created_by else None,
        "created_at": c.created_at.isoformat() if c.created_at else None,
    }


@router.get("/projects/{project_id}/entity_comments")
def list_entity_comments(
    project_id: UUID,
    table_id: str,
    column_name: Optional[str] = None,
    user: CurrentUser = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """List the comment thread on a table (column_name omitted) or one
    of its columns (column_name set). Oldest first."""
    _verify(project_id, user.user_id, db)
    tid = _resolve_table_uuid(db, str(project_id), table_id)
    if not tid:
        raise HTTPException(404, "Table not found")
    q = db.query(EntityComment).filter(
        EntityComment.project_id == project_id,
        EntityComment.table_id == tid,
    )
    if column_name is None:
        q = q.filter(EntityComment.column_name.is_(None))
    else:
        q = q.filter(EntityComment.column_name == column_name)
    return [_comment_to_dict(c) for c in q.order_by(EntityComment.created_at.asc()).all()]


@router.post("/projects/{project_id}/entity_comments")
def create_entity_comment(
    project_id: UUID,
    body: EntityCommentCreate,
    table_id: str,
    column_name: Optional[str] = None,
    user: CurrentUser = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Append a user-authored comment to a table or column thread."""
    _verify(project_id, user.user_id, db)
    tid = _resolve_table_uuid(db, str(project_id), table_id)
    if not tid:
        raise HTTPException(404, "Table not found")
    text = (body.body or "").strip()
    if not text:
        raise HTTPException(400, "body is required")
    if column_name is not None:
        row = db.execute(
            sa_text("SELECT columns FROM tables WHERE id=:id"),
            {"id": tid},
        ).fetchone()
        cols = row[0] if (row and row[0]) else []
        if isinstance(cols, str):
            cols = json.loads(cols)
        names = {c.get("name") for c in cols if isinstance(c, dict)}
        if column_name not in names:
            raise HTTPException(404, "Column not found on table")
    c = EntityComment(
        project_id=project_id,
        table_id=tid,
        column_name=column_name,
        author="user",
        body=text,
        created_by=user.user_id,
    )
    db.add(c)
    db.commit()
    db.refresh(c)
    return _comment_to_dict(c)
