"""In-memory approval registry for tool calls that need user consent.

Pattern (Claude Code-style):
  1. Agent calls an approval-gated tool (e.g. enrichment_run).
  2. agent.py registers a pending approval and emits an `approval_required`
     SSE event with the approval_id + tool args + estimated cost.
  3. agent.py awaits the registered Future. The turn is paused here.
  4. FE shows an approval card. User clicks Approve or Deny.
  5. API endpoint POST /approvals/{id}/respond resolves the Future.
  6. agent.py either runs the handler (approved) or skips it with a
     "User denied this action" tool result (denied), then continues
     the turn.

Storage is in-process — approvals do NOT survive a worker restart or
SSE reconnect. If the user reloads the tab mid-approval, the chat run
will hang until cleanup_stale runs (or staleness sweeper kills the run).
This is acceptable for v1; persisting approvals to a DB table would
let us survive reconnects but adds plumbing we can defer.
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


log = logging.getLogger(__name__)


# Tools that require explicit user approval before running. The agent
# loop checks this set BEFORE dispatching the handler.
APPROVAL_REQUIRED: frozenset[str] = frozenset({
    "enrichment_run",
    "table_delete",
    "row_delete",
})


@dataclass
class PendingApproval:
    """One in-flight approval: the Future agent.py is awaiting + metadata
    captured at request time so the FE can render the card."""
    id: str
    project_id: str
    tool: str
    args: Dict[str, Any]
    # Server-computed estimate; FE shows it on the card so the user can
    # decide. Approximate — handler may end up spending less.
    estimated_cost_credits: float
    # Human description ("Run enrichment 'Verified Email' on 47 rows").
    summary: str
    future: asyncio.Future = field(default_factory=asyncio.Future)


class ApprovalRegistry:
    """Single per-process registry of pending approvals."""

    def __init__(self) -> None:
        self._pending: Dict[str, PendingApproval] = {}
        self._lock = asyncio.Lock()

    async def request(
        self,
        project_id: str,
        tool: str,
        args: Dict[str, Any],
        *,
        estimated_cost_credits: float,
        summary: str,
    ) -> PendingApproval:
        """Register a pending approval and return it. Caller is expected
        to emit an SSE event with the approval_id, then await pending.future
        to block on the user's decision."""
        loop = asyncio.get_running_loop()
        pending = PendingApproval(
            id=str(uuid.uuid4()),
            project_id=project_id,
            tool=tool,
            args=args,
            estimated_cost_credits=estimated_cost_credits,
            summary=summary,
            future=loop.create_future(),
        )
        async with self._lock:
            self._pending[pending.id] = pending
        log.info("approval requested: %s tool=%s cost~%.2f", pending.id, tool, estimated_cost_credits)
        return pending

    async def peek(self, approval_id: str) -> Optional[PendingApproval]:
        """Look up a pending approval without resolving it. Caller needs
        the original tool args (e.g. to fire the deferred enrichment_run
        when the user approves the end-of-turn chip)."""
        async with self._lock:
            return self._pending.get(approval_id)

    async def resolve(self, approval_id: str, approved: bool) -> bool:
        """Resolve a pending approval. Returns True if the approval was
        found (and resolved), False if not. Idempotent — second call to
        resolve the same id is a no-op."""
        async with self._lock:
            pending = self._pending.get(approval_id)
        if pending is None:
            return False
        if pending.future.done():
            log.debug("approval %s already resolved; skipping", approval_id)
            return True
        pending.future.set_result(approved)
        log.info("approval resolved: %s approved=%s", approval_id, approved)
        async with self._lock:
            self._pending.pop(approval_id, None)
        return True

    async def list_for_project(self, project_id: str) -> list[dict[str, Any]]:
        """Snapshot of pending approvals for a project. Used to rehydrate
        the FE after a reconnect."""
        async with self._lock:
            return [
                {
                    "id": p.id,
                    "tool": p.tool,
                    "args": p.args,
                    "estimated_cost_credits": p.estimated_cost_credits,
                    "summary": p.summary,
                }
                for p in self._pending.values()
                if p.project_id == project_id
            ]

    async def cleanup_chat_run(self, project_id: str) -> None:
        """Cancel any pending approvals for a project whose chat run is
        ending. Resolving as denied so the awaiting agent loop unwinds
        with a "denied" tool result instead of hanging forever."""
        async with self._lock:
            stale_ids = [p.id for p in self._pending.values() if p.project_id == project_id]
            for pid in stale_ids:
                p = self._pending.get(pid)
                if p and not p.future.done():
                    p.future.set_result(False)
                self._pending.pop(pid, None)
        if stale_ids:
            log.info("cleared %d pending approvals on chat-run end (project=%s)", len(stale_ids), project_id)


# Singleton — agent.py and routes_actions.py both import this instance.
REGISTRY = ApprovalRegistry()


def estimate_enrichment_run_cost(args: Dict[str, Any], db, project_id: str) -> tuple[float, str]:
    """Compute estimated cost + human summary for an enrichment_run call.

    Looks up the enrichment's per_row_credit_cap and the scope row count
    so the approval card can show "Run 'X' on N rows, up to Y cr".
    Returns (estimated_cost, summary). Both are best-effort — if anything
    fails, returns (0, generic summary) and lets the user decide.
    """
    from sqlalchemy import text as sa_text
    try:
        from dsl_worker.chat.tools import resolve_enrichment_id
        eid = resolve_enrichment_id(db, project_id, args.get("enrichment_id"))
        if not eid:
            return 0.0, "Run enrichment"

        row = db.execute(
            sa_text(
                "SELECT e.name, e.per_row_credit_cap, e.table_id::text, "
                "       e.columns, t.name AS table_name "
                "FROM enrichments e JOIN tables t ON t.id=e.table_id "
                "WHERE e.id=:eid AND e.deleted_at IS NULL"
            ),
            {"eid": eid},
        ).fetchone()
        if not row:
            return 0.0, "Run enrichment"
        name, cap, table_id, columns_raw, table_name = row[0], float(row[1] or 1.0), row[2], row[3], row[4]
        try:
            cols = columns_raw if isinstance(columns_raw, list) else json.loads(columns_raw or "[]")
        except Exception:
            cols = []
        col_names = [c.get("name") for c in cols if isinstance(c, dict) and c.get("name")]

        # Resolve scope → row count. Mirrors the scope handling in
        # enrichment._resolve_scope_rows so the approval card's "Run on N
        # rows" matches what will actually run.
        scope = args.get("scope") or {"type": "all_unfilled"}
        scope_type = scope.get("type", "all_unfilled")
        if scope_type == "row_ids":
            n_rows = len(scope.get("row_ids") or [])
        elif scope_type == "first_n":
            n_rows = int(scope.get("first_n") or 10)
        elif scope_type == "filtered":
            # Apply explicit scope.filters[] in Python (same predicate
            # used in _resolve_scope_rows and filter_set's preview).
            from dsl_worker.chat.tools import _match, _normalize_filter
            raw_filters = scope.get("filters") or []
            norm_filters = []
            for f in raw_filters:
                if not isinstance(f, dict):
                    continue
                col = f.get("column") or f.get("column_name") or f.get("field")
                if not col:
                    continue
                normalized = _normalize_filter(f.get("op") or f.get("operator"), f.get("value"))
                if normalized is None:
                    continue
                op_norm, value_norm = normalized
                norm_filters.append((col, op_norm, value_norm))
            if not norm_filters:
                count_row = db.execute(
                    sa_text("SELECT count(*) FROM samples WHERE table_id=:tid AND deleted_at IS NULL"),
                    {"tid": table_id},
                ).fetchone()
                n_rows = int(count_row[0] or 0) if count_row else 0
            else:
                all_rows = db.execute(
                    sa_text("SELECT row FROM samples WHERE table_id=:tid AND deleted_at IS NULL"),
                    {"tid": table_id},
                ).fetchall()
                n_rows = sum(
                    1 for (rd,) in all_rows
                    if isinstance(rd, dict) and all(_match(rd.get(c), op, v) for c, op, v in norm_filters)
                )
        else:
            # all_unfilled — count samples missing any of the enrichment's columns
            count_row = db.execute(
                sa_text("SELECT count(*) FROM samples WHERE table_id=:tid AND deleted_at IS NULL"),
                {"tid": table_id},
            ).fetchone()
            n_rows = int(count_row[0] or 0) if count_row else 0

        # Apply first_n cap on filtered + all_unfilled so the approval
        # card matches what _resolve_scope_rows will actually run. Mirror
        # of the cap applied in enrichment.py.
        if scope_type in ("filtered", "all_unfilled"):
            cap_raw = scope.get("first_n")
            if cap_raw is not None:
                try:
                    cap_n = int(cap_raw)
                    if cap_n > 0:
                        n_rows = min(n_rows, cap_n)
                except (TypeError, ValueError):
                    pass

        est = cap * n_rows
        # Build a rich summary: table name, columns, filter (if any) so the
        # user can see exactly what's about to run rather than a bare
        # "Run X on N rows" label.
        table_part = f" on **{table_name}**" if table_name else ""
        col_part = ""
        if col_names:
            if len(col_names) == 1:
                col_part = f" — fills **{col_names[0]}**"
            elif len(col_names) <= 3:
                col_part = f" — fills **{', '.join(col_names)}**"
            else:
                col_part = f" — fills **{', '.join(col_names[:2])} +{len(col_names) - 2} more**"
        filter_part = ""
        if scope_type == "filtered":
            raw_filters = scope.get("filters") or []
            terms: list[str] = []
            for f in raw_filters:
                if not isinstance(f, dict):
                    continue
                col = f.get("column") or f.get("column_name") or f.get("field")
                op = (f.get("op") or f.get("operator") or "").lower()
                value = f.get("value")
                if not col:
                    continue
                value_text = ""
                if isinstance(value, (list, tuple)):
                    value_text = ", ".join(str(x) for x in value)
                elif isinstance(value, dict):
                    inc = value.get("include") or []
                    if inc:
                        value_text = ", ".join(str(x) for x in inc)
                elif value is not None:
                    value_text = str(value)
                if value_text:
                    terms.append(f"{col} = {value_text}")
                else:
                    terms.append(col)
            if terms:
                filter_part = f" (filter: {'; '.join(terms[:3])})"
        scope_part = (
            f"{n_rows} row{'s' if n_rows != 1 else ''}"
            if scope_type != "row_ids" or n_rows != 1
            else "1 row"
        )
        summary = (
            f"Run “{name}”{table_part}, {scope_part}{filter_part}"
            f"{col_part} — up to {est:.1f} credits"
        )
        return est, summary
    except Exception as e:
        log.warning("estimate_enrichment_run_cost failed: %s", e)
        return 0.0, "Run enrichment"
