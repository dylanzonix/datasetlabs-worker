"""SQLite schema and ORM for a V-next project.

One SQLite file per project. Schema is tiny and intentionally generic:

  columns         — column definitions (name, format, description, direct_call)
  rows            — one row of project data; columns stored as JSON dict
  cell_meta       — sidecar per (row_id, column_name) tracking fill status
  snapshots       — per-snapshot blobs of the SQLite file (stored elsewhere)
  turns           — chat turn audit log (links to snapshots)

Row data lives in `rows.data` as a JSON dict so column adds/removes don't
require schema migrations. Indexed columns can be added later if perf matters.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
from uuid import uuid4


SCHEMA = """
CREATE TABLE IF NOT EXISTS columns (
    name           TEXT PRIMARY KEY,
    format         TEXT NOT NULL DEFAULT '',
    description    TEXT NOT NULL DEFAULT '',
    direct_call    TEXT,                            -- JSON; nullable
    max_cost       REAL NOT NULL DEFAULT 0.15,
    position       INTEGER NOT NULL,                -- display order
    created_at     TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS rows (
    id             TEXT PRIMARY KEY,                -- uuid
    data           TEXT NOT NULL DEFAULT '{}',      -- JSON object
    created_at     TEXT NOT NULL,
    updated_at     TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS cell_meta (
    row_id         TEXT NOT NULL,
    column_name    TEXT NOT NULL,
    status         TEXT NOT NULL,                   -- filled|null_legitimate|budget_exhausted|error
    budget_used    REAL NOT NULL DEFAULT 0,
    last_error     TEXT,
    last_attempt_at TEXT NOT NULL,
    PRIMARY KEY (row_id, column_name),
    FOREIGN KEY (row_id) REFERENCES rows(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS snapshots (
    id             TEXT PRIMARY KEY,                -- uuid
    description    TEXT NOT NULL,                   -- human-readable label
    created_at     TEXT NOT NULL,
    blob_path      TEXT                             -- where the snapshot lives; null if inline
);

CREATE TABLE IF NOT EXISTS turns (
    id             TEXT PRIMARY KEY,                -- uuid
    role           TEXT NOT NULL,                   -- user|assistant|tool
    content        TEXT NOT NULL,
    tool_calls     TEXT,                            -- JSON list, nullable
    snapshot_id    TEXT,                            -- nullable; non-null if turn modified DB
    cost_usd       REAL NOT NULL DEFAULT 0,
    created_at     TEXT NOT NULL,
    FOREIGN KEY (snapshot_id) REFERENCES snapshots(id)
);

CREATE INDEX IF NOT EXISTS ix_rows_updated_at ON rows(updated_at);
CREATE INDEX IF NOT EXISTS ix_turns_created_at ON turns(created_at);
"""


# ---------------------------------------------------------------------------
# Connection helpers
# ---------------------------------------------------------------------------


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def connect(path: str | Path) -> sqlite3.Connection:
    """Open (or create) a project SQLite file with our schema applied."""
    conn = sqlite3.connect(str(path), isolation_level=None)  # autocommit; we open transactions explicitly
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    conn.executescript(SCHEMA)
    return conn


@contextmanager
def transaction(conn: sqlite3.Connection):
    """Wrap a unit of writes so they commit or roll back atomically."""
    conn.execute("BEGIN")
    try:
        yield
        conn.execute("COMMIT")
    except Exception:
        conn.execute("ROLLBACK")
        raise


# ---------------------------------------------------------------------------
# Column ops
# ---------------------------------------------------------------------------


@dataclass
class Column:
    name: str
    format: str = ""
    description: str = ""
    direct_call: Optional[Dict[str, Any]] = None
    max_cost: float = 0.15
    position: int = 0
    created_at: str = ""

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> "Column":
        return cls(
            name=row["name"],
            format=row["format"],
            description=row["description"],
            direct_call=json.loads(row["direct_call"]) if row["direct_call"] else None,
            max_cost=row["max_cost"],
            position=row["position"],
            created_at=row["created_at"],
        )


def list_columns(conn: sqlite3.Connection) -> List[Column]:
    rows = conn.execute("SELECT * FROM columns ORDER BY position, created_at").fetchall()
    return [Column.from_row(r) for r in rows]


def get_column(conn: sqlite3.Connection, name: str) -> Optional[Column]:
    row = conn.execute("SELECT * FROM columns WHERE name = ?", (name,)).fetchone()
    return Column.from_row(row) if row else None


def add_column(
    conn: sqlite3.Connection,
    name: str,
    *,
    format: str = "",
    description: str = "",
    direct_call: Optional[Dict[str, Any]] = None,
    max_cost: float = 0.15,
) -> Column:
    if get_column(conn, name) is not None:
        raise ValueError(f"Column {name!r} already exists")
    next_pos = conn.execute("SELECT COALESCE(MAX(position), -1) + 1 FROM columns").fetchone()[0]
    ts = now()
    conn.execute(
        "INSERT INTO columns (name, format, description, direct_call, max_cost, position, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (
            name,
            format,
            description,
            json.dumps(direct_call) if direct_call else None,
            max_cost,
            next_pos,
            ts,
        ),
    )
    return Column(
        name=name,
        format=format,
        description=description,
        direct_call=direct_call,
        max_cost=max_cost,
        position=next_pos,
        created_at=ts,
    )


def modify_column(conn: sqlite3.Connection, name: str, **fields: Any) -> Column:
    col = get_column(conn, name)
    if col is None:
        raise ValueError(f"Column {name!r} does not exist")
    allowed = {"format", "description", "direct_call", "max_cost"}
    sets: List[str] = []
    vals: List[Any] = []
    for k, v in fields.items():
        if k not in allowed:
            raise ValueError(f"Cannot modify field {k!r}")
        if k == "direct_call":
            v = json.dumps(v) if v else None
        sets.append(f"{k} = ?")
        vals.append(v)
    if sets:
        conn.execute(f"UPDATE columns SET {', '.join(sets)} WHERE name = ?", (*vals, name))
    updated = get_column(conn, name)
    assert updated is not None
    return updated


def delete_column(conn: sqlite3.Connection, name: str) -> int:
    """Remove a column and strip its data from every row. Returns rows affected."""
    if get_column(conn, name) is None:
        raise ValueError(f"Column {name!r} does not exist")
    # Drop the field from each row's JSON
    affected = 0
    for r in conn.execute("SELECT id, data FROM rows").fetchall():
        d = json.loads(r["data"])
        if name in d:
            del d[name]
            conn.execute(
                "UPDATE rows SET data = ?, updated_at = ? WHERE id = ?",
                (json.dumps(d), now(), r["id"]),
            )
            affected += 1
    conn.execute("DELETE FROM cell_meta WHERE column_name = ?", (name,))
    conn.execute("DELETE FROM columns WHERE name = ?", (name,))
    return affected


# ---------------------------------------------------------------------------
# Row ops
# ---------------------------------------------------------------------------


def _row_to_dict(row: sqlite3.Row) -> Dict[str, Any]:
    out = json.loads(row["data"])
    out["_id"] = row["id"]
    return out


def add_rows(
    conn: sqlite3.Connection,
    items: Iterable[Dict[str, Any]],
    merge_key: Optional[str] = None,
) -> Tuple[int, int]:
    """Insert or merge rows. Returns (inserted, merged)."""
    inserted = 0
    merged = 0
    for item in items:
        if merge_key:
            mv = item.get(merge_key)
            if mv is not None:
                # Look for an existing row with the same merge key value
                existing = conn.execute(
                    "SELECT id, data FROM rows WHERE json_extract(data, ?) = ?",
                    (f"$.\"{merge_key}\"", str(mv)),
                ).fetchone()
                if existing:
                    existing_data = json.loads(existing["data"])
                    # Merge: new fields fill empty cells, don't overwrite non-null
                    for k, v in item.items():
                        if v is not None and existing_data.get(k) in (None, ""):
                            existing_data[k] = v
                    conn.execute(
                        "UPDATE rows SET data = ?, updated_at = ? WHERE id = ?",
                        (json.dumps(existing_data), now(), existing["id"]),
                    )
                    merged += 1
                    continue
        # No merge — insert fresh
        rid = str(uuid4())
        ts = now()
        conn.execute(
            "INSERT INTO rows (id, data, created_at, updated_at) VALUES (?, ?, ?, ?)",
            (rid, json.dumps(item), ts, ts),
        )
        inserted += 1
    return inserted, merged


def count_rows(conn: sqlite3.Connection, where: Optional[Dict[str, Any]] = None) -> int:
    sql, args = _where_to_sql(where or {})
    cursor = conn.execute(f"SELECT COUNT(*) FROM rows WHERE {sql}", args)
    return cursor.fetchone()[0]


def get_rows(
    conn: sqlite3.Connection,
    where: Optional[Dict[str, Any]] = None,
    limit: Optional[int] = None,
    columns: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    sql, args = _where_to_sql(where or {})
    q = f"SELECT * FROM rows WHERE {sql} ORDER BY created_at"
    if limit is not None:
        q += f" LIMIT {int(limit)}"
    out: List[Dict[str, Any]] = []
    for r in conn.execute(q, args).fetchall():
        d = _row_to_dict(r)
        if columns is not None:
            d = {"_id": d["_id"], **{k: d.get(k) for k in columns}}
        out.append(d)
    return out


def update_rows(
    conn: sqlite3.Connection,
    where: Dict[str, Any],
    values: Dict[str, Any],
) -> int:
    sql, args = _where_to_sql(where)
    matched = conn.execute(f"SELECT id, data FROM rows WHERE {sql}", args).fetchall()
    for r in matched:
        d = json.loads(r["data"])
        d.update(values)
        conn.execute(
            "UPDATE rows SET data = ?, updated_at = ? WHERE id = ?",
            (json.dumps(d), now(), r["id"]),
        )
    return len(matched)


def delete_rows(conn: sqlite3.Connection, where: Dict[str, Any]) -> int:
    sql, args = _where_to_sql(where)
    cur = conn.execute(f"DELETE FROM rows WHERE {sql}", args)
    return cur.rowcount


def set_cell(
    conn: sqlite3.Connection,
    row_id: str,
    column: str,
    value: Any,
    *,
    status: str = "filled",
    budget_used: float = 0,
    last_error: Optional[str] = None,
) -> None:
    row = conn.execute("SELECT data FROM rows WHERE id = ?", (row_id,)).fetchone()
    if row is None:
        raise ValueError(f"Row {row_id} not found")
    d = json.loads(row["data"])
    d[column] = value
    conn.execute(
        "UPDATE rows SET data = ?, updated_at = ? WHERE id = ?",
        (json.dumps(d), now(), row_id),
    )
    conn.execute(
        "INSERT INTO cell_meta (row_id, column_name, status, budget_used, last_error, last_attempt_at) "
        "VALUES (?, ?, ?, ?, ?, ?) "
        "ON CONFLICT(row_id, column_name) DO UPDATE SET "
        "status = excluded.status, "
        "budget_used = excluded.budget_used, "
        "last_error = excluded.last_error, "
        "last_attempt_at = excluded.last_attempt_at",
        (row_id, column, status, budget_used, last_error, now()),
    )


# ---------------------------------------------------------------------------
# WHERE clause translation
# ---------------------------------------------------------------------------
#
# The LLM sees `where` as a dict like:
#
#   {"Verified Email": None, "headcount__lt": 10, "title__contains": "Founder"}
#
# We translate to SQL using JSON1 path lookups. Operators are encoded by
# suffix on the key name (Django-ish):
#
#   col            → col = ?
#   col__ne        → col != ?
#   col__lt/gt/lte/gte
#   col__contains  → col LIKE '%?%'
#   col__in        → col IN (...)
#   col            with value None → col IS NULL
#
# Dotted lookups for cell_meta sidecar:
#
#   "_meta.<column>.status"        → join cell_meta on column_name = <column>
#                                    and check status field
#   "_meta.<column>.budget_used"   → numeric compare
#

_OP_SUFFIXES = {
    "__ne": "!=",
    "__lt": "<",
    "__gt": ">",
    "__lte": "<=",
    "__gte": ">=",
}


def _column_expr(field: str) -> str:
    """Render a SQL expression that selects field's value from rows.data."""
    # Use JSON1 ->> which yields a TEXT value (or NULL).
    return f"json_extract(data, '$.\"{field}\"')"


def _where_to_sql(where: Dict[str, Any]) -> Tuple[str, List[Any]]:
    """Translate the dict-where into a SQL fragment + bound args.

    Returns ("1=1", []) for an empty filter so callers can always inline it
    after WHERE.
    """
    if not where:
        return "1=1", []
    clauses: List[str] = []
    args: List[Any] = []
    for raw_key, value in where.items():
        # Operator suffix?
        op = "="
        field = raw_key
        for suffix, sym in _OP_SUFFIXES.items():
            if raw_key.endswith(suffix):
                op = sym
                field = raw_key[: -len(suffix)]
                break
        else:
            if raw_key.endswith("__contains"):
                field = raw_key[: -len("__contains")]
                clauses.append(f"{_column_expr(field)} LIKE ?")
                args.append(f"%{value}%")
                continue
            if raw_key.endswith("__in"):
                field = raw_key[: -len("__in")]
                if not isinstance(value, list) or not value:
                    clauses.append("0=1")
                    continue
                placeholders = ",".join("?" * len(value))
                clauses.append(f"{_column_expr(field)} IN ({placeholders})")
                args.extend(value)
                continue
            if raw_key.endswith("__isnull"):
                field = raw_key[: -len("__isnull")]
                if value:
                    clauses.append(f"{_column_expr(field)} IS NULL")
                else:
                    clauses.append(f"{_column_expr(field)} IS NOT NULL")
                continue

        if value is None:
            if op == "=":
                clauses.append(f"{_column_expr(field)} IS NULL")
            elif op == "!=":
                clauses.append(f"{_column_expr(field)} IS NOT NULL")
            else:
                raise ValueError(f"Cannot compare {raw_key!r} to NULL with {op}")
            continue

        clauses.append(f"{_column_expr(field)} {op} ?")
        args.append(value)
    return " AND ".join(clauses), args


# ---------------------------------------------------------------------------
# Snapshots (file-level)
# ---------------------------------------------------------------------------
#
# A snapshot is a copy of the SQLite file taken before a destructive turn.
# Triggers are decided by the agent loop; this module just stores the bytes
# and returns an id you can checkout later.
#

def take_snapshot(
    conn: sqlite3.Connection,
    project_path: Path,
    description: str,
    snapshot_dir: Path,
) -> str:
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    sid = str(uuid4())
    target = snapshot_dir / f"{sid}.sqlite"
    # SQLite's online backup API gives us a consistent copy without locking writes.
    backup_conn = sqlite3.connect(str(target))
    with backup_conn:
        conn.backup(backup_conn)
    backup_conn.close()
    conn.execute(
        "INSERT INTO snapshots (id, description, created_at, blob_path) VALUES (?, ?, ?, ?)",
        (sid, description, now(), str(target)),
    )
    return sid


def list_snapshots(conn: sqlite3.Connection) -> List[Dict[str, Any]]:
    return [dict(r) for r in conn.execute(
        "SELECT id, description, created_at FROM snapshots ORDER BY created_at"
    ).fetchall()]


def checkout_snapshot(
    conn: sqlite3.Connection,
    project_path: Path,
    snapshot_id: str,
    snapshot_dir: Path,
) -> None:
    """Replace the live project SQLite with a snapshot's contents.

    Caller is responsible for closing `conn` first; this function reopens.
    """
    target_blob = snapshot_dir / f"{snapshot_id}.sqlite"
    if not target_blob.exists():
        raise ValueError(f"Snapshot {snapshot_id} blob missing at {target_blob}")
    conn.close()
    project_path.unlink(missing_ok=True)
    # File-level copy
    project_path.write_bytes(target_blob.read_bytes())
