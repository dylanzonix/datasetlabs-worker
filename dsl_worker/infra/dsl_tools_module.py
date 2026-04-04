"""
dsl_tools module — uploaded to sandbox as /workspace/dsl_tools.py on session init.

Provides utility functions for code execution in the sandbox environment.
The LLM imports these with: from dsl_tools import list_files, file_info, read_jsonl, write_jsonl

NOTE: submit_candidates is NOT here — it's a native orchestrator tool call.
The sandbox does data manipulation, tools do actions.
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
