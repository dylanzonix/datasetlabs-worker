"""Candidates: per-project blob-backed staging files for source-tool fetches.

Source tools write fetched items as JSONL to blob:

    projects/{project_id}/candidates/{tool}_{run_id}.jsonl

Files are listed/inspected/processed by the candidates_* chat tools without
the LLM ever holding the full dataset in context. The agent sees a path,
metadata, and a small preview; it uses candidates_inspect to look around,
candidates_to_rows to bulk-commit a subset, and code_exec for custom
transforms.
"""
from __future__ import annotations

import io
import json
import logging
import re
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, Iterator, List, Optional

from azure.core.exceptions import ResourceNotFoundError
from azure.storage.blob import ContainerClient

from dsl_api.azure.blob import get_blob_client
from dsl_api.config import settings


log = logging.getLogger(__name__)


PREVIEW_ITEMS = 5
TOOL_RESULT_PREVIEW_BYTES = 4000  # cap preview JSON size hard


def _safe_slug(s: str) -> str:
    """Make a filesystem-safe slug from a tool name."""
    return re.sub(r"[^a-zA-Z0-9_-]+", "_", s)[:80] or "tool"


def _candidates_prefix(project_id) -> str:
    return f"projects/{project_id}/candidates"


def _candidate_blob_path(project_id, file_name: str) -> str:
    return f"{_candidates_prefix(project_id)}/{file_name}"


# ---- Exec logs --------------------------------------------------------
# Sibling to candidates/. Holds per-`code_exec`-call transcripts —
# stdout/stderr lines, op results, and structured errors. The agent
# inspects them via the same `candidates_inspect` machinery (the inspect
# handler resolves `exec_*` filenames to this prefix). Apply a 7-day
# Azure blob lifecycle rule on the `exec_logs/` prefix so they don't
# accumulate forever.

def _exec_logs_prefix(project_id) -> str:
    return f"projects/{project_id}/exec_logs"


def _exec_log_blob_path(project_id, file_name: str) -> str:
    return f"{_exec_logs_prefix(project_id)}/{file_name}"


def is_exec_log_filename(file_name: str) -> bool:
    """`exec_*.jsonl` files live under exec_logs/, not candidates/."""
    return isinstance(file_name, str) and file_name.startswith("exec_")


def write_exec_log(
    project_id, file_name: str, lines: Iterable[Dict[str, Any]]
) -> Dict[str, Any]:
    """Write a per-`code_exec` transcript as JSONL to exec_logs/.

    Each line is a dict — typically {"stream": "stdout"|"stderr"|"op"|
    "error", ...}. The agent never sees the full content; it gets a small
    envelope back from `code_exec` with the filename + counts, and uses
    `candidates_inspect(file=<exec_*.jsonl>, filter=..., limit=...)` to
    fetch slices on demand.
    """
    blob_path = _exec_log_blob_path(project_id, file_name)
    buf = io.BytesIO()
    n = 0
    for it in lines:
        if not isinstance(it, dict):
            continue
        buf.write((json.dumps(it, default=str, ensure_ascii=False) + "\n").encode("utf-8"))
        n += 1
    buf.seek(0)
    client = get_blob_client(blob_path)
    md = {"kind": "exec_log", "lines": str(n)}
    client.upload_blob(buf, overwrite=True, metadata=md)
    return {"file": file_name, "lines": n}


def stream_exec_log(project_id, file_name: str) -> Iterator[Dict[str, Any]]:
    """Stream transcript lines from an exec log. Same shape as
    stream_candidates but resolves to exec_logs/."""
    blob_path = _exec_log_blob_path(project_id, file_name)
    client = get_blob_client(blob_path)
    try:
        downloader = client.download_blob()
    except ResourceNotFoundError as e:
        raise FileNotFoundError(f"Exec log not found: {file_name}") from e

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


@dataclass
class CandidatesFileMeta:
    file: str
    tool: str
    items_count: int
    fields: List[str]
    cost_usd: float
    created_at: str
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def write_candidates(
    project_id,
    tool: str,
    items: Iterable[Dict[str, Any]],
    *,
    cost_usd: float = 0.0,
    extra: Optional[Dict[str, Any]] = None,
) -> CandidatesFileMeta:
    """Write items as JSONL to blob. Returns metadata.

    Sync — Azure storage SDK is sync; results are typically <50MB. If we
    need true async we'd switch to azure.storage.blob.aio.
    """
    run_id = uuid.uuid4().hex[:8]
    file_name = f"{_safe_slug(tool)}_{run_id}.jsonl"
    blob_path = _candidate_blob_path(project_id, file_name)

    buf = io.BytesIO()
    fields_set: set[str] = set()
    items_count = 0

    for it in items:
        if not isinstance(it, dict):
            continue
        fields_set.update(it.keys())
        line = json.dumps(it, default=str, ensure_ascii=False) + "\n"
        buf.write(line.encode("utf-8"))
        items_count += 1

    buf.seek(0)
    client = get_blob_client(blob_path)
    # Blob metadata values must be ASCII-safe and short. Tool name + counts
    # only; fields list goes in our own response, not blob metadata.
    md = {
        "tool": _safe_slug(tool)[:64],
        "items_count": str(items_count),
        "cost_usd": f"{cost_usd:.6f}",
    }
    client.upload_blob(buf, overwrite=True, metadata=md)

    return CandidatesFileMeta(
        file=file_name,
        tool=tool,
        items_count=items_count,
        fields=sorted(fields_set),
        cost_usd=cost_usd,
        created_at=datetime.now(timezone.utc).isoformat(),
        extra=extra or {},
    )


def list_candidate_files(project_id) -> List[Dict[str, Any]]:
    """List all candidate files for a project. Metadata only, no content."""
    container = ContainerClient(
        account_url=f"https://{settings.AZURE_STORAGE_ACCOUNT_NAME}.blob.core.windows.net",
        container_name=settings.AZURE_STORAGE_CONTAINER_NAME,
        credential=settings.AZURE_STORAGE_ACCOUNT_KEY,
    )
    prefix = _candidates_prefix(project_id) + "/"
    out: List[Dict[str, Any]] = []
    for blob in container.list_blobs(name_starts_with=prefix, include=["metadata"]):
        md = blob.metadata or {}
        out.append({
            "file": blob.name.split("/")[-1],
            "tool": md.get("tool", "?"),
            "items_count": int(md.get("items_count", 0) or 0),
            "cost_usd": float(md.get("cost_usd", 0.0) or 0.0),
            "size_bytes": blob.size or 0,
            "created_at": (
                blob.creation_time.isoformat() if blob.creation_time else None
            ),
        })
    out.sort(key=lambda x: x["created_at"] or "", reverse=True)
    return out


def stream_candidates(project_id, file_name: str) -> Iterator[Dict[str, Any]]:
    """Stream items from a candidates file as a sync iterator of dicts.

    Raises FileNotFoundError if the blob doesn't exist.
    """
    blob_path = _candidate_blob_path(project_id, file_name)
    client = get_blob_client(blob_path)
    try:
        downloader = client.download_blob()
    except ResourceNotFoundError as e:
        raise FileNotFoundError(f"Candidates file not found: {file_name}") from e

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


def read_candidates_bytes(project_id, file_name: str) -> bytes:
    """Read the full file as bytes — used for code_exec injection."""
    blob_path = _candidate_blob_path(project_id, file_name)
    client = get_blob_client(blob_path)
    try:
        return client.download_blob().readall()
    except ResourceNotFoundError as e:
        raise FileNotFoundError(f"Candidates file not found: {file_name}") from e


# ---------------------------------------------------------------------------
# Filter dialect (mirrors agent._where_to_sql for rows tools)
# ---------------------------------------------------------------------------


def apply_filter(item: Dict[str, Any], filter_dict: Optional[Dict[str, Any]]) -> bool:
    """Match the rows filter dialect: {col: v} eq, {col__lt/gt/lte/gte: n},
    {col__contains: s}, {col__in: [...]}, {col__isnull: bool}, {col: null}
    for IS NULL. Multiple keys AND together.
    """
    if not filter_dict:
        return True
    for key, val in filter_dict.items():
        col, op = (key.rsplit("__", 1) if "__" in key else (key, "eq"))
        item_val = item.get(col)
        try:
            if op == "eq":
                # {col: null} -> IS NULL
                if val is None:
                    if item_val is not None:
                        return False
                else:
                    if item_val != val:
                        return False
            elif op == "lt":
                if item_val is None or not (item_val < val):
                    return False
            elif op == "lte":
                if item_val is None or not (item_val <= val):
                    return False
            elif op == "gt":
                if item_val is None or not (item_val > val):
                    return False
            elif op == "gte":
                if item_val is None or not (item_val >= val):
                    return False
            elif op == "contains":
                if item_val is None or str(val) not in str(item_val):
                    return False
            elif op == "in":
                if item_val not in (val or []):
                    return False
            elif op == "isnull":
                if (item_val is None) != bool(val):
                    return False
        except TypeError:
            # Type mismatch (e.g. comparing str to int) — treat as no-match
            return False
    return True


def project_fields(
    item: Dict[str, Any], fields: Optional[List[str]]
) -> Dict[str, Any]:
    if not fields:
        return item
    return {k: item.get(k) for k in fields}


# ---------------------------------------------------------------------------
# Canonical tool result shape
# ---------------------------------------------------------------------------


def make_tool_result(
    *,
    meta: CandidatesFileMeta,
    preview_items: List[Dict[str, Any]],
    extra_top_level: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Standard shape returned by every source tool that writes candidates."""
    out: Dict[str, Any] = {
        "candidates_file": meta.file,
        "tool": meta.tool,
        "items_count": meta.items_count,
        "fields": meta.fields,
        "cost_usd": round(meta.cost_usd, 4),
        "preview": preview_items[:PREVIEW_ITEMS],
        "created_at": meta.created_at,
    }
    if meta.extra:
        out["run_metadata"] = meta.extra
    if extra_top_level:
        out.update(extra_top_level)
    return out
