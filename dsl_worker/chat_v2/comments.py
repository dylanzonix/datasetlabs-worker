"""Entity comments — the chat-style thread that backs the table/column
detail panels' "Description" section.

Two surfaces:

  * `seed_table_comment` / `seed_column_comment` — internal SQL helpers
    called by tools.py / enrichment.py after they commit. These write
    agent-authored seeds (the "initial description" the user sees first
    when they open the detail panel).

  * `comment_on_table` / `comment_on_column` — chat tools the agent can
    call mid-conversation to append further commentary (e.g. "Updated
    email enrichment to verified-only"). Wired into HANDLERS / TOOL_DEFS.

User-authored comments are written via the API router
(`/v1/projects/{pid}/entity_comments`), not this module — the agent only
writes agent rows.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple

from sqlalchemy import text as sa_text
from sqlalchemy.orm import Session

from dsl_worker.chat_v2.tools import ToolContext, resolve_table_id


log = logging.getLogger(__name__)


def seed_comment(
    db: Session,
    project_id: str,
    table_id: str,
    column_name: Optional[str],
    body: str,
) -> None:
    """Insert an agent-authored comment row. Best-effort — never raises
    out (a failed comment seed should not abort the caller's commit).
    """
    if not body:
        return
    try:
        db.execute(
            sa_text(
                """
                INSERT INTO entity_comments
                  (id, project_id, table_id, column_name, author, body, created_by, created_at)
                VALUES
                  (gen_random_uuid(), :pid, :tid, :col, 'agent', :body, NULL, now())
                """
            ),
            {
                "pid": project_id,
                "tid": table_id,
                "col": column_name,
                "body": body,
            },
        )
        db.commit()
    except Exception:
        log.exception("seed_comment failed for table_id=%s col=%s", table_id, column_name)
        try:
            db.rollback()
        except Exception:
            pass


def seed_table_comment(
    db: Session, project_id: str, table_id: str, body: str
) -> None:
    seed_comment(db, project_id, table_id, None, body)


def seed_column_comment(
    db: Session, project_id: str, table_id: str, column_name: str, body: str
) -> None:
    seed_comment(db, project_id, table_id, column_name, body)


# ---------------------------------------------------------------------------
# Chat tools — agent-callable
# ---------------------------------------------------------------------------


async def comment_on_table(
    args: Dict[str, Any], ctx: ToolContext
) -> Tuple[Dict[str, Any], float]:
    """Append an agent-authored comment to a table's thread.

    Used when the agent wants to leave a note explaining a non-obvious
    decision or summarizing what just changed (e.g. after a parameter
    tweak). Args: table_id, body.
    """
    table_id = resolve_table_id(ctx.db, ctx.project_id, args.get("table_id"))
    body = (args.get("body") or "").strip()
    if not table_id:
        return {"error": "table_id is required"}, 0.0
    if not body:
        return {"error": "body is required"}, 0.0
    seed_table_comment(ctx.db, ctx.project_id, table_id, body)
    return {"ok": True, "table_id": args.get("table_id"), "wrote_chars": len(body)}, 0.0


async def comment_on_column(
    args: Dict[str, Any], ctx: ToolContext
) -> Tuple[Dict[str, Any], float]:
    """Append an agent-authored comment to a column's thread.

    Args: table_id, column (name as it appears on the table), body.
    """
    table_id = resolve_table_id(ctx.db, ctx.project_id, args.get("table_id"))
    column = (args.get("column") or args.get("column_name") or "").strip()
    body = (args.get("body") or "").strip()
    if not table_id:
        return {"error": "table_id is required"}, 0.0
    if not column:
        return {"error": "column is required"}, 0.0
    if not body:
        return {"error": "body is required"}, 0.0

    # Validate the column exists to avoid orphan threads.
    row = ctx.db.execute(
        sa_text("SELECT columns FROM tables WHERE id=:id"),
        {"id": table_id},
    ).fetchone()
    cols = []
    if row and row[0]:
        raw = row[0]
        if isinstance(raw, list):
            cols = raw
        else:
            import json as _json
            try:
                cols = _json.loads(raw)
            except Exception:
                cols = []
    names = {c.get("name") for c in cols if isinstance(c, dict)}
    if column not in names:
        return {"error": f"column {column!r} not found on this table"}, 0.0

    seed_column_comment(ctx.db, ctx.project_id, table_id, column, body)
    return {"ok": True, "table_id": args.get("table_id"), "column": column, "wrote_chars": len(body)}, 0.0


HANDLERS = {
    "comment_on_table": comment_on_table,
    "comment_on_column": comment_on_column,
}
