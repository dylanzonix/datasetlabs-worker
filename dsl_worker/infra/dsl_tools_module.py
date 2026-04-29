"""
dsl_tools module — uploaded to sandbox as /workspace/dsl_tools.py on session init.

Provides utility functions for code execution in the sandbox environment.

Two families:

1. **Workspace utilities** (`list_files`, `read_jsonl`, `write_jsonl`,
   `read_csv`, `preview`, etc.) — pure local file-IO inside the sandbox.

2. **Project ops** (`add_columns`, `add_rows`, `update_rows`,
   `delete_rows`, `add_candidates`) — record an intent to mutate the
   project. The sandbox is offline; these helpers DO NOT call the
   database. They append a JSON line to `/workspace/_dsl_ops.jsonl`.
   After `code_exec` returns, the worker reads that file, applies each
   op through the canonical chat-mode tool handlers, and persists a
   transcript (stdout + stderr + op results) to blob. The agent gets a
   small summary back; the data never round-trips through the LLM.

   Constraints (the helpers enforce locally; the worker re-validates):
   - items lists are capped at 10,000 per op call
   - destructive ops (`update_rows`, `delete_rows`) require `confirm=True`

NOTE: submit_candidates is NOT here — that's a v13 orchestrator concept.
The chat agent's bulk-write path is `add_rows` / `add_candidates`.
"""

# This is the source code that gets uploaded to the sandbox.
# It runs inside the sandbox's Python environment (nsjail container).

DSL_TOOLS_SOURCE = '''"""
dsl_tools — workspace utilities for code execution.

Usage:
    from dsl_tools import list_files, file_info, read_jsonl, write_jsonl
"""

import json
import os
import csv
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


WORKSPACE = "/workspace"


def list_files(directory: str = "all") -> List[str]:
    """List files in the workspace.

    Args:
        directory: "uploads", "candidates", "downloads", or "all"

    Returns:
        List of file paths.
    """
    files = []
    dirs_to_scan = []

    if directory in ("uploads", "all"):
        dirs_to_scan.append(Path(WORKSPACE) / "uploads")
    if directory in ("candidates", "all"):
        dirs_to_scan.append(Path(WORKSPACE) / "candidates")
    if directory in ("downloads", "all"):
        dirs_to_scan.append(Path(WORKSPACE) / "downloads")
    if directory == "all":
        # Also scan workspace root for any loose files
        dirs_to_scan.append(Path(WORKSPACE))

    for d in dirs_to_scan:
        if d.exists():
            for f in sorted(d.iterdir()):
                if f.is_file() and not f.name.startswith("."):
                    files.append(str(f))

    return files


def file_info(path: str) -> Dict[str, Any]:
    """Get metadata about a file.

    Returns:
        Dict with: exists, name, size_bytes, extension, num_lines (for text files)
    """
    p = Path(path)
    if not p.exists():
        return {"exists": False, "name": p.name, "size_bytes": 0, "extension": p.suffix}

    info = {
        "exists": True,
        "name": p.name,
        "size_bytes": p.stat().st_size,
        "extension": p.suffix,
    }

    # Count lines for text files
    if p.suffix in (".jsonl", ".csv", ".tsv", ".txt", ".json", ".md"):
        try:
            with open(p, "r", encoding="utf-8", errors="replace") as f:
                info["num_lines"] = sum(1 for _ in f)
        except Exception:
            pass

    return info


def read_jsonl(path: str) -> List[Dict]:
    """Read a JSONL file (one JSON object per line).

    Returns:
        List of dicts.
    """
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    items.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return items


def write_jsonl(path: str, data: List[Dict]) -> int:
    """Write a list of dicts to a JSONL file.

    Args:
        path: File path to write to.
        data: List of dicts to write.

    Returns:
        Number of lines written.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    count = 0
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\\n")
            count += 1
    return count


def append_jsonl(path: str, data: Union[Dict, List[Dict]]) -> int:
    """Append one or more dicts to a JSONL file.

    Returns:
        Number of lines appended.
    """
    if isinstance(data, dict):
        data = [data]
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    count = 0
    with open(path, "a", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\\n")
            count += 1
    return count


def read_csv(path: str, max_rows: int = 0) -> List[Dict]:
    """Read a CSV file into a list of dicts.

    Args:
        path: CSV file path.
        max_rows: Maximum rows to read (0 = all).

    Returns:
        List of dicts (one per row, keys from header).
    """
    rows = []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if max_rows and i >= max_rows:
                break
            rows.append(dict(row))
    return rows


_OPS_LOG = "/workspace/_dsl_ops.jsonl"
_MAX_ITEMS_PER_OP = 10000


def _emit_op(op: Dict[str, Any]) -> None:
    """Append one op to the ops log. Local file write only — no network.
    The worker reads this file after exec_python returns and applies
    each op through the canonical chat-mode tool handlers.
    """
    os.makedirs(os.path.dirname(_OPS_LOG) or ".", exist_ok=True)
    with open(_OPS_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(op, ensure_ascii=False, default=str) + "\\n")


def add_columns(specs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Add one or more columns to the project schema.

    Each spec: {"name": str, "format"?: str, "description"?: str}.

    Records the op locally; the worker creates each column post-exec.
    Returns immediately with {"queued_columns": N} — no DB round-trip.
    """
    if not isinstance(specs, list):
        raise TypeError("add_columns: specs must be a list of dicts")
    if not specs:
        return {"queued_columns": 0}
    if len(specs) > _MAX_ITEMS_PER_OP:
        raise ValueError(
            f"add_columns: {len(specs)} specs > cap of {_MAX_ITEMS_PER_OP}"
        )
    for s in specs:
        if not isinstance(s, dict) or not s.get("name"):
            raise ValueError("add_columns: each spec must be a dict with a 'name' field")
    _emit_op({"op": "add_columns", "specs": specs})
    return {"queued_columns": len(specs)}


def add_rows(items: List[Dict[str, Any]], merge_key: Optional[str] = None) -> Dict[str, Any]:
    """Insert (or upsert by merge_key) a batch of rows.

    Each item is a dict of column-name -> value. Pass the FULL list in
    one call — items has no practical size limit up to the 10,000 cap;
    chunking adds nothing. Records the op locally; the worker commits
    server-side post-exec.
    """
    if not isinstance(items, list):
        raise TypeError("add_rows: items must be a list of dicts")
    if not items:
        return {"queued_rows": 0}
    if len(items) > _MAX_ITEMS_PER_OP:
        raise ValueError(
            f"add_rows: {len(items)} items > cap of {_MAX_ITEMS_PER_OP}. "
            "Split across multiple add_rows() calls."
        )
    if not all(isinstance(it, dict) for it in items):
        raise ValueError("add_rows: every item must be a dict")
    op: Dict[str, Any] = {"op": "add_rows", "items": items}
    if merge_key:
        op["merge_key"] = merge_key
    _emit_op(op)
    return {"queued_rows": len(items)}


def update_rows(
    where: Dict[str, Any], values: Dict[str, Any], confirm: bool = False
) -> Dict[str, Any]:
    """Set column values on every row matching `where`. Destructive —
    requires confirm=True (without it the worker rejects the op).

    `where` uses the standard filter dialect: {col: v}, {col__lt: n},
    {col__contains: s}, {col__in: [...]}, {col__isnull: bool}, etc.
    """
    if not isinstance(where, dict):
        raise TypeError("update_rows: where must be a dict")
    if not isinstance(values, dict) or not values:
        raise ValueError("update_rows: values must be a non-empty dict")
    if not confirm:
        raise ValueError(
            "update_rows: refused — pass confirm=True. Destructive ops "
            "are gated to prevent accidental mass updates."
        )
    _emit_op({
        "op": "update_rows",
        "where": where,
        "values": values,
        "confirm": True,
    })
    return {"queued": "update_rows"}


def delete_rows(where: Dict[str, Any], confirm: bool = False) -> Dict[str, Any]:
    """Soft-delete every row matching `where`. Destructive — requires
    confirm=True. Same filter dialect as update_rows."""
    if not isinstance(where, dict):
        raise TypeError("delete_rows: where must be a dict")
    if not confirm:
        raise ValueError(
            "delete_rows: refused — pass confirm=True. Destructive ops "
            "are gated to prevent accidental mass deletes."
        )
    _emit_op({"op": "delete_rows", "where": where, "confirm": True})
    return {"queued": "delete_rows"}


def add_candidates(items: List[Dict[str, Any]], name: Optional[str] = None) -> Dict[str, Any]:
    """Stage `items` as a candidates JSONL file on the project. Use this
    when you've computed a list of records via Python (e.g. flattened
    from an upload, joined across files) and want to inspect or
    bulk-commit them with `candidates_to_rows` later.

    `name` is a short slug for the resulting file (defaults to "code_exec").
    Returns immediately; the worker creates the candidates blob post-exec.
    """
    if not isinstance(items, list):
        raise TypeError("add_candidates: items must be a list of dicts")
    if not items:
        return {"queued_candidates": 0}
    if len(items) > _MAX_ITEMS_PER_OP:
        raise ValueError(
            f"add_candidates: {len(items)} items > cap of {_MAX_ITEMS_PER_OP}"
        )
    if not all(isinstance(it, dict) for it in items):
        raise ValueError("add_candidates: every item must be a dict")
    op: Dict[str, Any] = {"op": "add_candidates", "items": items}
    if name:
        op["name"] = str(name)
    _emit_op(op)
    return {"queued_candidates": len(items)}


def preview(path: str, n: int = 5) -> str:
    """Preview first N lines/rows of a file.

    Returns:
        Formatted string preview.
    """
    p = Path(path)
    if not p.exists():
        return f"File not found: {path}"

    ext = p.suffix.lower()

    if ext == ".jsonl":
        items = read_jsonl(path)[:n]
        total = file_info(path).get("num_lines", "?")
        lines = [f"JSONL: {total} lines, showing first {len(items)}:"]
        for i, item in enumerate(items):
            lines.append(f"  [{i}] {json.dumps(item, ensure_ascii=False)[:200]}")
        return "\\n".join(lines)

    elif ext == ".csv":
        rows = read_csv(path, max_rows=n)
        total = file_info(path).get("num_lines", "?")
        if not rows:
            return f"CSV: empty or unreadable"
        lines = [f"CSV: ~{total} lines, columns: {list(rows[0].keys())}"]
        for i, row in enumerate(rows):
            lines.append(f"  [{i}] {row}")
        return "\\n".join(lines)

    elif ext == ".json":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return f"JSON array: {len(data)} items. First {min(n, len(data))}:\\n" + \\
                json.dumps(data[:n], indent=2, ensure_ascii=False)[:1000]
        elif isinstance(data, dict):
            keys = list(data.keys())
            return f"JSON object with keys: {keys}\\n" + \\
                json.dumps(data, indent=2, ensure_ascii=False)[:1000]
        return str(data)[:500]

    else:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            lines = [f.readline().rstrip() for _ in range(n)]
        return f"Text file, first {len(lines)} lines:\\n" + "\\n".join(lines)
'''
