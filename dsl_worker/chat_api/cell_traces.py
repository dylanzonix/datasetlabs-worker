"""Cell traces: per-fill blob-backed transcripts of cell-agent runs.

Each `rows_fill` call produces one JSONL file at:

    projects/{project_id}/cell_traces/{run_id}.jsonl

Each line is one cell — the row id, target columns, final status, cost,
turn count, and a turn-by-turn record of every tool call (name, args
summary, result snippet, cost). Mirrors the candidates / exec_logs
machinery in candidates.py — the agent inspects via cell_traces_inspect
without ever holding the full content in context.
"""
from __future__ import annotations

import io
import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, Iterator, List, Optional

from azure.core.exceptions import ResourceNotFoundError
from azure.storage.blob import ContainerClient

from dsl_api.azure.blob import get_blob_client
from dsl_api.config import settings


log = logging.getLogger(__name__)


# Hard cap on per-tool-call result snippet stored in the trace. The cell
# agent currently sees up to 6000 chars (fill.py); we save a much smaller
# slice for forensics — enough to see the shape of what came back without
# blowing trace files up.
TOOL_RESULT_SNIPPET_BYTES = 800
TOOL_ARGS_SNIPPET_BYTES = 400


def _safe_slug(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]+", "_", s)[:80] or "trace"


def _traces_prefix(project_id) -> str:
    return f"projects/{project_id}/cell_traces"


def _trace_blob_path(project_id, file_name: str) -> str:
    return f"{_traces_prefix(project_id)}/{file_name}"


def _truncate(s: Any, limit: int) -> Any:
    """Stringify and truncate; preserve dicts/lists if small enough."""
    if s is None:
        return None
    if isinstance(s, (dict, list)):
        try:
            j = json.dumps(s, default=str, ensure_ascii=False)
        except Exception:
            j = str(s)
        if len(j) <= limit:
            return s
        return j[:limit] + "..."
    text = s if isinstance(s, str) else str(s)
    if len(text) <= limit:
        return text
    return text[:limit] + "..."


@dataclass
class CellTraceTurn:
    turn: int
    kind: str  # "tool_call" | "web_search" | "no_op"
    name: Optional[str] = None
    args: Any = None
    result: Any = None
    cost_usd: float = 0.0
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {"turn": self.turn, "kind": self.kind}
        if self.name is not None:
            d["name"] = self.name
        if self.args is not None:
            d["args"] = self.args
        if self.result is not None:
            d["result"] = self.result
        if self.cost_usd:
            d["cost_usd"] = round(self.cost_usd, 5)
        if self.error:
            d["error"] = self.error
        return d


@dataclass
class CellTrace:
    """In-memory trace built up during _run_cell_agent. Serialized to one
    JSONL line at the end of fill_rows. Keep it append-only — each cell
    agent owns exactly one CellTrace and writes to it as it progresses.
    """
    row_id: str
    columns: List[str]
    started_at: str
    status: str = "pending"
    reason: Optional[str] = None
    values: Dict[str, Any] = field(default_factory=dict)
    cost_usd: float = 0.0
    turns_used: int = 0
    skills_applied: List[str] = field(default_factory=list)
    turn_log: List[CellTraceTurn] = field(default_factory=list)
    ended_at: Optional[str] = None

    def add_tool_call(
        self,
        turn: int,
        name: str,
        args: Any,
        result: Any,
        cost_usd: float = 0.0,
        error: Optional[str] = None,
    ) -> None:
        self.turn_log.append(CellTraceTurn(
            turn=turn,
            kind="tool_call",
            name=name,
            args=_truncate(args, TOOL_ARGS_SNIPPET_BYTES),
            result=_truncate(result, TOOL_RESULT_SNIPPET_BYTES),
            cost_usd=cost_usd,
            error=error,
        ))

    def add_web_search(self, turn: int, cost_usd: float) -> None:
        self.turn_log.append(CellTraceTurn(
            turn=turn,
            kind="web_search",
            cost_usd=cost_usd,
        ))

    def add_no_op(self, turn: int, note: str) -> None:
        self.turn_log.append(CellTraceTurn(
            turn=turn,
            kind="no_op",
            result=note,
        ))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "row_id": self.row_id,
            "columns": self.columns,
            "status": self.status,
            "reason": self.reason,
            "values": self.values,
            "cost_usd": round(self.cost_usd, 5),
            "turns_used": self.turns_used,
            "skills_applied": self.skills_applied,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "turns": [t.to_dict() for t in self.turn_log],
        }


def new_trace(row_id: str, columns: List[str]) -> CellTrace:
    return CellTrace(
        row_id=str(row_id),
        columns=list(columns),
        started_at=datetime.now(timezone.utc).isoformat(),
    )


def write_traces(
    project_id,
    run_id: str,
    traces: Iterable[CellTrace],
    *,
    target_columns: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Write a fill batch's cell traces as JSONL to blob.

    Returns {"file": file_name, "count": N}. Best-effort: any blob error
    is logged and swallowed (we don't want trace persistence failures to
    break the fill). Caller can write multiple times for one run only if
    they want overwrite — we always upload with overwrite=True.
    """
    file_name = f"{_safe_slug(run_id)}.jsonl"
    blob_path = _trace_blob_path(project_id, file_name)
    buf = io.BytesIO()
    n = 0
    for tr in traces:
        line = json.dumps(tr.to_dict(), default=str, ensure_ascii=False) + "\n"
        buf.write(line.encode("utf-8"))
        n += 1
    buf.seek(0)
    md = {
        "kind": "cell_trace",
        "cells": str(n),
        "run_id": _safe_slug(run_id),
    }
    if target_columns:
        md["columns"] = _safe_slug(",".join(target_columns))[:64]
    try:
        client = get_blob_client(blob_path)
        client.upload_blob(buf, overwrite=True, metadata=md)
    except Exception:
        log.exception("cell_traces.write_traces failed for %s", file_name)
        return {"file": file_name, "count": n, "persisted": False}
    return {"file": file_name, "count": n, "persisted": True}


def list_trace_files(project_id) -> List[Dict[str, Any]]:
    """List all cell trace files for a project. Most recent first."""
    container = ContainerClient(
        account_url=f"https://{settings.AZURE_STORAGE_ACCOUNT_NAME}.blob.core.windows.net",
        container_name=settings.AZURE_STORAGE_CONTAINER_NAME,
        credential=settings.AZURE_STORAGE_ACCOUNT_KEY,
    )
    prefix = _traces_prefix(project_id) + "/"
    out: List[Dict[str, Any]] = []
    try:
        for blob in container.list_blobs(name_starts_with=prefix, include=["metadata"]):
            md = blob.metadata or {}
            out.append({
                "file": blob.name.split("/")[-1],
                "cells": int(md.get("cells", 0) or 0),
                "columns": md.get("columns", ""),
                "size_bytes": blob.size or 0,
                "created_at": (
                    blob.creation_time.isoformat() if blob.creation_time else None
                ),
            })
    except Exception:
        log.exception("cell_traces.list_trace_files failed")
        return []
    out.sort(key=lambda x: x["created_at"] or "", reverse=True)
    return out


def stream_trace(project_id, file_name: str) -> Iterator[Dict[str, Any]]:
    """Stream cells from a trace file as a sync iterator of dicts."""
    blob_path = _trace_blob_path(project_id, file_name)
    client = get_blob_client(blob_path)
    try:
        downloader = client.download_blob()
    except ResourceNotFoundError as e:
        raise FileNotFoundError(f"Cell trace not found: {file_name}") from e

    pending = b""
    for chunk in downloader.chunks():
        pending += chunk
        while b"\n" in pending:
            line, pending = pending.split(b"\n", 1)
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line.decode("utf-8"))
            except json.JSONDecodeError:
                continue
    if pending.strip():
        try:
            yield json.loads(pending.decode("utf-8"))
        except json.JSONDecodeError:
            pass


def latest_trace_file(project_id) -> Optional[str]:
    """Return the file name of the most recent trace file, or None."""
    files = list_trace_files(project_id)
    return files[0]["file"] if files else None
