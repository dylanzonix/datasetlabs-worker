"""Flat tool surface for the V-next chat agent.

Each tool is a Python function that takes JSON-friendly args (a plain dict
already validated by the LLM SDK against `schema()`) and returns a dict
result the agent can read in subsequent turns. There's no chained ORM at
this layer — that's a stylistic choice forced by how function-calling APIs
work. Internally each tool calls into `db.py`.

A single dispatch helper (`call_tool`) is exposed so the chat agent can
turn an SDK `tool_call` object into a result without knowing about each
tool individually.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from . import db


# ---------------------------------------------------------------------------
# Tool registry
# ---------------------------------------------------------------------------


@dataclass
class ToolSpec:
    name: str
    description: str
    parameters: Dict[str, Any]   # JSON schema
    handler: Callable[..., Dict[str, Any]]
    modifies_db: bool = False    # if True, agent loop snapshots before calling


_REGISTRY: Dict[str, ToolSpec] = {}


def register(spec: ToolSpec) -> None:
    _REGISTRY[spec.name] = spec


def all_tool_specs() -> List[ToolSpec]:
    return list(_REGISTRY.values())


def to_openai_tools() -> List[Dict[str, Any]]:
    """Render the registry as the function-tools array the OpenAI Responses API expects."""
    return [
        {
            "type": "function",
            "name": spec.name,
            "description": spec.description,
            "parameters": spec.parameters,
        }
        for spec in _REGISTRY.values()
    ]


def call_tool(
    name: str,
    args: Dict[str, Any],
    *,
    conn: sqlite3.Connection,
    project_path: Path,
    snapshot_dir: Path,
) -> Dict[str, Any]:
    spec = _REGISTRY.get(name)
    if spec is None:
        return {"error": f"unknown tool: {name}"}
    try:
        return spec.handler(
            conn=conn,
            project_path=project_path,
            snapshot_dir=snapshot_dir,
            **args,
        )
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _row_summary(row: Dict[str, Any], column_order: List[str]) -> Dict[str, Any]:
    """Return row data ordered the way the user expects (columns in declared order)."""
    out = {"_id": row.get("_id")}
    for col in column_order:
        out[col] = row.get(col)
    # Include any extras (dropped columns or stray fields)
    for k, v in row.items():
        if k not in out and not k.startswith("_"):
            out[k] = v
    return out


# ---------------------------------------------------------------------------
# Column tools
# ---------------------------------------------------------------------------


def _columns_add(
    *,
    conn: sqlite3.Connection,
    name: str,
    format: str = "",
    description: str = "",
    direct_call: Optional[Dict[str, Any]] = None,
    max_cost: float = 0.15,
    **_,
) -> Dict[str, Any]:
    col = db.add_column(
        conn,
        name=name,
        format=format,
        description=description,
        direct_call=direct_call,
        max_cost=max_cost,
    )
    return {
        "ok": True,
        "column": {
            "name": col.name,
            "format": col.format,
            "description": col.description,
            "direct_call": col.direct_call,
            "max_cost": col.max_cost,
        },
    }


register(ToolSpec(
    name="columns_add",
    description=(
        "Define a new column. Required: name. Optional: format (e.g. 'lowercase email or null', "
        "'range string like 10-15'), description (what the column is), direct_call (skip the cell "
        "agent — see schema), max_cost (per-cell budget cap; default 0.15)."
    ),
    parameters={
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "format": {"type": "string"},
            "description": {"type": "string"},
            "direct_call": {
                "type": "object",
                "description": (
                    "Optional fast path that skips the cell LLM. Schema: "
                    "{tool: 'fullenrich.enrich_email', args: {first_name: '{Founder Name}', ...}, "
                    "extract: 'result.email'}. Templates substitute row column values."
                ),
            },
            "max_cost": {"type": "number"},
        },
        "required": ["name"],
    },
    handler=_columns_add,
    modifies_db=True,
))


def _columns_list(*, conn: sqlite3.Connection, **_) -> Dict[str, Any]:
    cols = db.list_columns(conn)
    return {
        "columns": [
            {
                "name": c.name,
                "format": c.format,
                "description": c.description,
                "direct_call": c.direct_call,
                "max_cost": c.max_cost,
            }
            for c in cols
        ]
    }


register(ToolSpec(
    name="columns_list",
    description="List all columns defined on the project, in display order.",
    parameters={"type": "object", "properties": {}},
    handler=_columns_list,
))


def _columns_modify(
    *,
    conn: sqlite3.Connection,
    name: str,
    format: Optional[str] = None,
    description: Optional[str] = None,
    direct_call: Optional[Dict[str, Any]] = None,
    max_cost: Optional[float] = None,
    **_,
) -> Dict[str, Any]:
    fields: Dict[str, Any] = {}
    if format is not None:
        fields["format"] = format
    if description is not None:
        fields["description"] = description
    if direct_call is not None:
        fields["direct_call"] = direct_call
    if max_cost is not None:
        fields["max_cost"] = max_cost
    col = db.modify_column(conn, name, **fields)
    return {"ok": True, "column": {"name": col.name, "format": col.format,
                                   "description": col.description,
                                   "direct_call": col.direct_call,
                                   "max_cost": col.max_cost}}


register(ToolSpec(
    name="columns_modify",
    description="Update a column's metadata. Pass only the fields you want to change.",
    parameters={
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "format": {"type": "string"},
            "description": {"type": "string"},
            "direct_call": {"type": "object"},
            "max_cost": {"type": "number"},
        },
        "required": ["name"],
    },
    handler=_columns_modify,
    modifies_db=True,
))


def _columns_delete(*, conn: sqlite3.Connection, name: str, **_) -> Dict[str, Any]:
    affected = db.delete_column(conn, name)
    return {"ok": True, "rows_with_data_dropped": affected}


register(ToolSpec(
    name="columns_delete",
    description="Drop a column definition and remove its data from every row. Snapshots before running.",
    parameters={
        "type": "object",
        "properties": {"name": {"type": "string"}},
        "required": ["name"],
    },
    handler=_columns_delete,
    modifies_db=True,
))


# ---------------------------------------------------------------------------
# Row tools
# ---------------------------------------------------------------------------


def _rows_add(
    *,
    conn: sqlite3.Connection,
    items: List[Dict[str, Any]],
    merge_key: Optional[str] = None,
    **_,
) -> Dict[str, Any]:
    if not isinstance(items, list):
        return {"error": "items must be a list of objects"}
    inserted, merged = db.add_rows(conn, items, merge_key=merge_key)
    return {"ok": True, "inserted": inserted, "merged": merged, "total": db.count_rows(conn)}


register(ToolSpec(
    name="rows_add",
    description=(
        "Insert (or merge by `merge_key`) a small batch of rows. Each item is a dict of "
        "column-name → value. Column names should match the project's declared columns; "
        "values for columns that don't exist yet are stored anyway and become visible once "
        "the column is added. For nested-source commits with hundreds of rows, prefer "
        "code_exec to map the JSONL file."
    ),
    parameters={
        "type": "object",
        "properties": {
            "items": {
                "type": "array",
                "items": {"type": "object"},
                "description": "List of row dicts to insert/merge.",
            },
            "merge_key": {
                "type": "string",
                "description": (
                    "Column name; if a row with the same value already exists, merge non-null "
                    "fields from the incoming row into the existing one (no overwrite of "
                    "non-null cells)."
                ),
            },
        },
        "required": ["items"],
    },
    handler=_rows_add,
    modifies_db=True,
))


def _rows_count(
    *,
    conn: sqlite3.Connection,
    where: Optional[Dict[str, Any]] = None,
    **_,
) -> Dict[str, Any]:
    return {"count": db.count_rows(conn, where)}


register(ToolSpec(
    name="rows_count",
    description=(
        "Count rows matching `where`. Where syntax: dict of {column: value} for equality, "
        "or {column__lt: n} / __gt / __lte / __gte for ordering, {column__contains: s} for "
        "substring, {column__in: [...]} for set membership, {column__isnull: true|false}, "
        "{column: null} for IS NULL. Multiple keys AND together. No `where` = all rows."
    ),
    parameters={
        "type": "object",
        "properties": {"where": {"type": "object"}},
    },
    handler=_rows_count,
))


def _rows_get(
    *,
    conn: sqlite3.Connection,
    where: Optional[Dict[str, Any]] = None,
    limit: Optional[int] = None,
    columns: Optional[List[str]] = None,
    **_,
) -> Dict[str, Any]:
    col_order = [c.name for c in db.list_columns(conn)]
    rows = db.get_rows(conn, where=where, limit=limit, columns=columns)
    return {
        "rows": [_row_summary(r, columns or col_order) for r in rows],
        "count": len(rows),
    }


register(ToolSpec(
    name="rows_get",
    description=(
        "Fetch rows matching `where`. Returns a list of dicts (each with `_id`). "
        "Use `limit` for pagination; use `columns` to project specific columns."
    ),
    parameters={
        "type": "object",
        "properties": {
            "where": {"type": "object"},
            "limit": {"type": "integer", "minimum": 1},
            "columns": {"type": "array", "items": {"type": "string"}},
        },
    },
    handler=_rows_get,
))


def _rows_sample(*, conn: sqlite3.Connection, n: int = 3, **_) -> Dict[str, Any]:
    col_order = [c.name for c in db.list_columns(conn)]
    rows = conn.execute("SELECT * FROM rows ORDER BY RANDOM() LIMIT ?", (n,)).fetchall()
    return {
        "rows": [_row_summary({**json.loads(r["data"]), "_id": r["id"]}, col_order) for r in rows]
    }


register(ToolSpec(
    name="rows_sample",
    description="Return up to N random rows (default 3) for spot-checking the table state.",
    parameters={
        "type": "object",
        "properties": {"n": {"type": "integer", "minimum": 1}},
    },
    handler=_rows_sample,
))


def _rows_update(
    *,
    conn: sqlite3.Connection,
    where: Dict[str, Any],
    values: Dict[str, Any],
    **_,
) -> Dict[str, Any]:
    affected = db.update_rows(conn, where, values)
    return {"ok": True, "affected": affected}


register(ToolSpec(
    name="rows_update",
    description="Set the given column values on every row matching `where`. Snapshots before running.",
    parameters={
        "type": "object",
        "properties": {
            "where": {"type": "object"},
            "values": {"type": "object"},
        },
        "required": ["where", "values"],
    },
    handler=_rows_update,
    modifies_db=True,
))


def _rows_delete(*, conn: sqlite3.Connection, where: Dict[str, Any], **_) -> Dict[str, Any]:
    affected = db.delete_rows(conn, where)
    return {"ok": True, "deleted": affected}


register(ToolSpec(
    name="rows_delete",
    description=(
        "Delete every row matching `where`. Always snapshots first so the rows can be "
        "restored via versions_checkout."
    ),
    parameters={
        "type": "object",
        "properties": {"where": {"type": "object"}},
        "required": ["where"],
    },
    handler=_rows_delete,
    modifies_db=True,
))


# ---------------------------------------------------------------------------
# Versions tools
# ---------------------------------------------------------------------------


def _versions_list(*, conn: sqlite3.Connection, **_) -> Dict[str, Any]:
    return {"snapshots": db.list_snapshots(conn)}


register(ToolSpec(
    name="versions_list",
    description="List all snapshots taken so far (id, description, timestamp).",
    parameters={"type": "object", "properties": {}},
    handler=_versions_list,
))


def _versions_checkout(
    *,
    conn: sqlite3.Connection,
    project_path: Path,
    snapshot_dir: Path,
    snapshot_id: str,
    **_,
) -> Dict[str, Any]:
    db.checkout_snapshot(conn, project_path, snapshot_id, snapshot_dir)
    return {
        "ok": True,
        "checked_out": snapshot_id,
        "note": "DB has been replaced. Reopen the project file to continue.",
    }


register(ToolSpec(
    name="versions_checkout",
    description=(
        "Replace the live project file with a snapshot's contents. Linear history — any "
        "changes since that snapshot are dropped. Caller (or chat loop) must reopen the "
        "DB connection after this returns."
    ),
    parameters={
        "type": "object",
        "properties": {"snapshot_id": {"type": "string"}},
        "required": ["snapshot_id"],
    },
    handler=_versions_checkout,
    modifies_db=True,
))
