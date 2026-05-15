"""Backfill `enrichment_id` on table columns for enrichments created before
the FE-grouping change (commit a9bb85f on worker/redesign, 2026-05-15).

After that commit, `_ensure_columns_on_table` stamps each new enrichment
column with the parent enrichment_id so the FE can:
  - render a colored top-border across the enrichment's columns
  - show the per-cell ▶ hover button on those cells
  - show the Run-first-N / Run-empty options in the column header popover

This script walks every table, builds a (column_name → enrichment_id) map
from the existing enrichments, and patches the table's columns array.
Idempotent — safe to re-run.

Usage:
  cd worker && python scripts/backfill_enrichment_column_tags.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path


def main() -> None:
    for f in [Path(".env.prod"), Path("../api/.env.prod")]:
        if f.exists():
            for line in f.read_text().splitlines():
                if "=" in line and not line.startswith("#"):
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip())

    sys.path.insert(0, ".")
    sys.path.insert(0, "../api")
    from dsl_api.db import SessionLocal
    from sqlalchemy import text

    db = SessionLocal()
    updated_tables = 0
    total_cols_tagged = 0

    rows = db.execute(
        text("SELECT id::text, columns FROM tables WHERE deleted_at IS NULL")
    ).fetchall()
    for tid, tcols in rows:
        if isinstance(tcols, str):
            tcols = json.loads(tcols)
        if not tcols:
            continue
        enrichments = db.execute(
            text(
                "SELECT id::text, columns FROM enrichments "
                "WHERE table_id=:tid AND deleted_at IS NULL"
            ),
            {"tid": tid},
        ).fetchall()
        if not enrichments:
            continue
        name_to_eid: dict[str, str] = {}
        for eid, ecols in enrichments:
            if isinstance(ecols, str):
                ecols = json.loads(ecols)
            for c in ecols or []:
                n = c.get("name")
                if n:
                    name_to_eid[n] = eid
        changed = False
        for c in tcols:
            n = c.get("name")
            eid = name_to_eid.get(n)
            if eid and c.get("enrichment_id") != eid:
                c["enrichment_id"] = eid
                changed = True
                total_cols_tagged += 1
        if changed:
            db.execute(
                text("UPDATE tables SET columns=CAST(:c AS jsonb) WHERE id=:tid"),
                {"tid": tid, "c": json.dumps(tcols)},
            )
            updated_tables += 1

    db.commit()
    print(f"Backfilled: {total_cols_tagged} columns across {updated_tables} tables")


if __name__ == "__main__":
    main()
