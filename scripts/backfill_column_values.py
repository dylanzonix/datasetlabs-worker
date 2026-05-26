"""Backfill table_column_value from existing samples.

After the alembic migration creates the table, this populates it from
historic sample rows so the distinct endpoint can read from the
materialized counts without first waiting for a cell write.

Usage:
  cd /home/user/datasetlabs/worker
  .venv/bin/python scripts/backfill_column_values.py            # dev
  .venv/bin/python scripts/backfill_column_values.py --prod     # prod

The script is idempotent: it truncates per-table before reinserting,
so re-running rebuilds without doubling counts.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from collections import defaultdict
from typing import Dict, Tuple

# Allow imports from the worker package without installing.
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..")))

from sqlalchemy import text as sa_text  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("backfill")


MAX_DISTINCT_VALUE_LEN = 500


def _stringify(v):
    if v is None:
        return None
    if isinstance(v, str):
        if not v:
            return None
        return v if len(v) <= MAX_DISTINCT_VALUE_LEN else None
    if isinstance(v, (int, float, bool)):
        return str(v)
    try:
        s = json.dumps(v, sort_keys=True, separators=(",", ":"))
    except Exception:
        s = str(v)
    return s if len(s) <= MAX_DISTINCT_VALUE_LEN else None


def backfill_table(db, table_id: str) -> int:
    """Rebuild table_column_value for one table. Returns rows inserted."""
    cols_row = db.execute(
        sa_text("SELECT columns FROM tables WHERE id = CAST(:tid AS uuid)"),
        {"tid": table_id},
    ).fetchone()
    if not cols_row:
        return 0
    columns = cols_row[0] if isinstance(cols_row[0], list) else json.loads(cols_row[0] or "[]")
    col_names = [c.get("name") for c in columns if isinstance(c, dict) and c.get("name")]
    if not col_names:
        return 0

    counts: Dict[Tuple[str, str], int] = defaultdict(int)
    sample_rows = db.execute(
        sa_text(
            "SELECT row FROM samples "
            "WHERE table_id = CAST(:tid AS uuid) AND deleted_at IS NULL"
        ),
        {"tid": table_id},
    )
    for (row_data,) in sample_rows:
        if not isinstance(row_data, dict):
            try:
                row_data = json.loads(row_data) if row_data else {}
            except Exception:
                continue
        for col in col_names:
            v = _stringify(row_data.get(col))
            if v is None:
                continue
            counts[(col, v)] += 1

    db.execute(
        sa_text(
            "DELETE FROM table_column_value WHERE table_id = CAST(:tid AS uuid)"
        ),
        {"tid": table_id},
    )
    if counts:
        rows = [
            {"tid": table_id, "col": col, "val": val, "cnt": cnt}
            for (col, val), cnt in counts.items()
        ]
        db.execute(
            sa_text(
                "INSERT INTO table_column_value "
                "(table_id, column_name, value, count, updated_at) "
                "VALUES (CAST(:tid AS uuid), :col, :val, :cnt, now())"
            ),
            rows,
        )
    db.commit()
    return len(counts)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prod", action="store_true", help="Use .env.prod for DATABASE_URL")
    parser.add_argument("--table", help="Limit to a single table id")
    args = parser.parse_args()

    # Load the right .env BEFORE importing dsl_api.db so SessionLocal
    # picks up the prod DSN.
    from dotenv import load_dotenv
    if args.prod:
        load_dotenv(".env.prod", override=True)
        log.info("loaded .env.prod")
    else:
        load_dotenv(".env", override=True)

    from dsl_api.db import SessionLocal

    db = SessionLocal()
    try:
        if args.table:
            table_ids = [args.table]
        else:
            rows = db.execute(
                sa_text(
                    "SELECT id::text FROM tables WHERE deleted_at IS NULL ORDER BY created_at"
                )
            ).fetchall()
            table_ids = [r[0] for r in rows]
        log.info("backfilling %d table(s)", len(table_ids))
        total = 0
        for i, tid in enumerate(table_ids, 1):
            t0 = time.time()
            n = backfill_table(db, tid)
            log.info(
                "[%d/%d] %s → %d value rows (%.1fs)",
                i, len(table_ids), tid, n, time.time() - t0,
            )
            total += n
        log.info("done. wrote %d total value rows.", total)
    finally:
        db.close()


if __name__ == "__main__":
    main()
