"""User-initiated action endpoints (button-triggered).

Distinct from agent-initiated tool calls — these bypass the LLM and act
directly on tables/enrichments.

  POST /v2/projects/{pid}/tables/{tid}/fetch_more     → table_extend with stored params
  POST /v2/projects/{pid}/enrichments/{eid}/run       → enrichment_run scope
  POST /v2/projects/{pid}/tables/{tid}/filters         → filter_set
  DELETE /v2/projects/{pid}/tables/{tid}/filters/{column} → filter_clear
  GET    /v2/projects/{pid}/approvals                  → list pending
  POST   /v2/projects/{pid}/approvals/{aid}            → resolve {approved}
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional
from uuid import UUID

log = logging.getLogger(__name__)

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import text as sa_text
from sqlalchemy.orm import Session

from dsl_api.auth import CurrentUser, get_current_user
from dsl_api.credits import consume_credits
from dsl_api.db import SessionLocal
from dsl_api.models import Account, Project
from dsl_api.plans import CENTS_PER_CREDIT
from dsl_worker.chat.tools import (
    ToolContext,
    table_extend,
    filter_set,
    filter_clear,
)
from dsl_worker.chat.enrichment import enrichment_run
from dsl_worker.chat.approvals import REGISTRY as APPROVALS
from dsl_worker.chat.cancels import REGISTRY as CANCELS
from dsl_worker.chat.option_picks import REGISTRY as OPTION_PICKS
from dsl_worker.chat.routes import _enforce_balance


router = APIRouter(prefix="/v2")


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def _verify(project_id: UUID, user_id: UUID, db: Session) -> Project:
    p = db.query(Project).filter(
        Project.id == project_id,
        Project.user_id == user_id,
        Project.deleted_at.is_(None),
    ).first()
    if not p:
        raise HTTPException(404, "Project not found")
    return p


class FetchMoreBody(BaseModel):
    n: int = 100


@router.post("/projects/{project_id}/tables/{table_id}/fetch_more")
async def post_fetch_more(
    project_id: UUID,
    table_id: str,
    body: FetchMoreBody,
    user: CurrentUser = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """User-initiated fetch_more. Calls table_extend with empty query_params
    delta — server uses stored cursor / query_params on the table."""
    _verify(project_id, user.user_id, db)
    _enforce_balance(db, user.user_id)
    ctx = ToolContext(db=db, project_id=str(project_id), user_id=str(user.user_id), run_id=None)
    result, cost = await table_extend(
        {"table_id": str(table_id), "query_params": {}, "n": body.n}, ctx
    )
    return {"result": result, "cost_usd": cost}


class RunEnrichmentBody(BaseModel):
    scope_type: str = "all_unfilled"   # all_unfilled | first_n | row_ids
    first_n: Optional[int] = None
    row_ids: Optional[List[str]] = None
    overwrite: bool = False


class PatchEnrichmentBody(BaseModel):
    # Canonical research levels — must match cell_agent.RESEARCH_CONFIG keys.
    # Legacy values (high/medium/low/none and older tier names) are
    # normalized via _LEGACY_TIER_TO_RESEARCH so old clients keep working.
    research: Optional[str] = None  # classify | research | deep
    tier: Optional[str] = None      # legacy alias for research
    per_row_credit_cap: Optional[float] = None


_RESEARCH_VALUES = {"classify", "research", "deep"}
# Maps every old name across every rename pass to the current canonical
# triple. Mirrors cell_agent.LEGACY_ALIASES so the read path and the
# write path agree.
_LEGACY_TIER_TO_RESEARCH = {
    # v4 (none/low/medium/high)
    "none":        "classify",
    "low":         "research",
    "medium":      "research",
    "high":        "deep",
    # v3 (classify/lookup/search/investigate)
    "lookup":      "research",
    "search":      "research",
    "investigate": "deep",
    # v2 (light)
    "light":       "research",
    # v1 (fast/smart/standard/deep/expert)
    "fast":        "classify",
    "smart":       "research",
    "expert":      "deep",
    "standard":    "research",
}


@router.patch("/projects/{project_id}/enrichments/{enrichment_id}")
def patch_enrichment(
    project_id: UUID,
    enrichment_id: str,
    body: PatchEnrichmentBody,
    user: CurrentUser = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Update research level (writes action.research) and/or per_row_credit_cap.

    Accepts the new `research` field as well as the legacy `tier` field
    (aliased to research). Always writes the new field on the action.
    """
    _verify(project_id, user.user_id, db)
    # Use the central resolver so a short_id like 'e1' that exists on
    # multiple projects can't cross-project leak. Also handles the new
    # composite t<X>e<Y> format.
    from dsl_worker.chat.tools import resolve_enrichment_id
    eid_uuid = resolve_enrichment_id(db, str(project_id), enrichment_id)
    if not eid_uuid:
        raise HTTPException(404, "Enrichment not found")
    row = db.execute(
        sa_text("SELECT action, per_row_credit_cap FROM enrichments WHERE id=:id"),
        {"id": eid_uuid},
    ).fetchone()
    if not row:
        raise HTTPException(404, "Enrichment not found")
    action = row[0] if isinstance(row[0], dict) else json.loads(row[0] or "{}")
    cap = row[1]

    # Resolve research target: prefer new field, fall back to legacy tier alias.
    # Run BOTH fields through the alias map so a client that sends a stale
    # value (e.g. "high" or "expert") still hits a canonical level.
    research: Optional[str] = None
    if body.research is not None:
        raw = body.research.lower()
        research = _LEGACY_TIER_TO_RESEARCH.get(raw, raw)
    elif body.tier is not None:
        raw = body.tier.lower()
        research = _LEGACY_TIER_TO_RESEARCH.get(raw, raw)
    if research is not None:
        if research not in _RESEARCH_VALUES:
            raise HTTPException(
                400,
                f"research must be one of {sorted(_RESEARCH_VALUES)} (got {research!r})",
            )
        action["research"] = research
        # Drop legacy `tier` if it lingered on this enrichment.
        action.pop("tier", None)
    if body.per_row_credit_cap is not None:
        cap = body.per_row_credit_cap
        # Mirror to action JSON so the two stay in sync. Runtime reads
        # the DB column, but downstream diagnostics + future readers
        # expect action.per_row_credit_cap to match what's in effect.
        action["per_row_credit_cap"] = float(cap)

    db.execute(
        sa_text(
            "UPDATE enrichments SET action = CAST(:a AS jsonb), "
            "per_row_credit_cap = :cap WHERE id = :id"
        ),
        {"a": json.dumps(action), "cap": cap, "id": eid_uuid},
    )
    db.commit()
    # Cast for JSON serializer — column is numeric(8,2), comes back as
    # Decimal when read from DB. Outgoing body always a plain number.
    return {
        "ok": True,
        "research": action.get("research"),
        "per_row_credit_cap": float(cap) if cap is not None else None,
    }


@router.post("/projects/{project_id}/enrichments/{enrichment_id}/run")
async def post_run_enrichment(
    project_id: UUID,
    enrichment_id: str,
    body: RunEnrichmentBody,
    user: CurrentUser = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    _verify(project_id, user.user_id, db)
    _enforce_balance(db, user.user_id)
    scope: Dict[str, Any] = {"type": body.scope_type}
    if body.first_n is not None:
        scope["first_n"] = body.first_n
    if body.row_ids is not None:
        scope["row_ids"] = body.row_ids
    ctx = ToolContext(db=db, project_id=str(project_id), user_id=str(user.user_id), run_id=None)

    # Resolve eid early so cancel registry uses the canonical UUID (matches
    # what /cancel resolves to). Otherwise short_id vs uuid mismatches
    # would prevent the Stop button from finding the task.
    from dsl_worker.chat.tools import resolve_enrichment_id
    canonical_eid = resolve_enrichment_id(db, str(project_id), enrichment_id) or str(enrichment_id)

    # Wrap the run in an asyncio.Task so the Stop button can cancel it.
    # enrichment.py's cell loop already handles asyncio.CancelledError
    # (cancels in-flight per-cell tasks + flushes partial cost).
    import asyncio as _asyncio
    task: _asyncio.Task = _asyncio.create_task(
        enrichment_run(
            {"enrichment_id": str(enrichment_id), "scope": scope, "overwrite": body.overwrite},
            ctx,
        )
    )
    await CANCELS.register(str(project_id), canonical_eid, task)
    cost = 0.0
    cancelled = False
    try:
        result, cost = await task
    except _asyncio.CancelledError:
        cancelled = True
        # Partial cost may have accrued in ctx.partial_cost_usd before cancel.
        cost = float(getattr(ctx, "partial_cost_usd", 0.0) or 0.0)
        result = {"cancelled": True}
    finally:
        await CANCELS.unregister(str(project_id), canonical_eid, task)

    # Charge the user via consume_credits — that decrements Account
    # pools (subscription / rollover / daily) AND writes the BalanceLedger
    # entries in one go. A bare ledger insert leaves the Account pools
    # untouched, so balance displays go stale while audit shows spend.
    spend_cents = int(round(float(cost or 0.0) * 100))
    if spend_cents > 0:
        try:
            account = (
                db.query(Account).filter(Account.user_id == user.user_id).first()
            )
            if account:
                consume_credits(
                    db,
                    account,
                    spend_cents / CENTS_PER_CREDIT,
                    project_id=project_id,
                    reason="enrichment_run_rest",
                )
                db.commit()
            else:
                log.warning(
                    "enrichment_run_rest: no account for user %s", user.user_id
                )
        except Exception:
            try:
                db.rollback()
            except Exception:
                pass

    # Patch-first response: fetch the affected rows fresh from samples
    # and return them so the FE can applyCellEdits without bouncing the
    # whole table. Chat-initiated runs emit cell_filled SSE events for
    # this purpose; the REST path uses run_id=None and never streams,
    # so without this the FE only saw the new value on the next
    # refreshRows() call. The Tier-1 cleanup removed that refresh, so
    # values now flow through this payload instead.
    updated_rows: List[Dict[str, Any]] = []
    if not cancelled and body.row_ids:
        try:
            from sqlalchemy import bindparam
            # IMPORTANT: `IN :ids` (not `ANY(:ids)`) with expanding=True —
            # SQLAlchemy expands the bind into a tuple of placeholders,
            # and Postgres ANY() needs an array, not a tuple. With ANY()
            # the query silently errored and updated_rows came back
            # empty, so the FE cleared the spinner without patching
            # anything (cells stayed blank until manual refresh).
            # `id::text IN :ids` also avoids the uuid-vs-text comparison
            # cast since body.row_ids is a list of plain strings.
            stmt = (
                sa_text(
                    "SELECT id::text, row, tags "
                    "FROM samples WHERE id::text IN :ids AND deleted_at IS NULL"
                ).bindparams(bindparam("ids", expanding=True))
            )
            rows = db.execute(stmt, {"ids": list(body.row_ids)}).fetchall()
            for r in rows:
                payload: Dict[str, Any] = {"id": r[0]}
                row_data = r[1] if isinstance(r[1], dict) else (json.loads(r[1]) if r[1] else {})
                payload.update(row_data)
                if r[2]:
                    payload["tags"] = r[2] if isinstance(r[2], dict) else json.loads(r[2])
                updated_rows.append(payload)
        except Exception:
            log.exception("post_run_enrichment: failed to fetch updated rows for FE patch")

    return {"result": result, "cost_usd": cost, "cancelled": cancelled, "updated_rows": updated_rows}


class PromoteQueryColumnsBody(BaseModel):
    # When set, adopt ONLY this column into the ghost — narrow scope so
    # the cell agent only refetches one field per row. Omit to adopt
    # every query column at once (one ghost, fills everything per row).
    # Each scope produces its own ghost: a single-column ghost and the
    # all-columns ghost coexist; lookup matches on adopted column-set.
    column: Optional[str] = None
    # Multi-column scope (block "Turn into enrichment & run"): adopt
    # EXACTLY this set of query columns into one ghost. Lenient — names
    # that are already promoted or absent are silently skipped, so the
    # caller can pass a whole selection without pre-filtering. Takes
    # precedence over `column` when both are present.
    columns: Optional[List[str]] = None


@router.post("/projects/{project_id}/tables/{table_id}/promote_query_columns")
async def post_promote_query_columns(
    project_id: UUID,
    table_id: str,
    body: PromoteQueryColumnsBody,
    user: CurrentUser = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Idempotent: ensure a ghost enrichment exists adopting all query
    columns on the table. Returns its id/short_id + the adopted columns.

    A "query column" is a column with no enrichment_id — i.e. it was
    populated by the original scrape (table_create / table_extend) rather
    than by a prior enrichment. After this call the column has an
    enrichment_id; from then on it behaves like any enrichment column and
    the existing /enrichments/{eid}/run endpoint backfills missing cells.

    The auto-derived prompt is intentionally generic ("use available tools
    to fill these fields for each row"). The cell agent's research tools
    figure out the retrieval from the row's existing values (URL, domain,
    name, etc.).
    """
    _verify(project_id, user.user_id, db)

    from dsl_worker.chat.tools import resolve_table_id, _next_enrichment_short_id, _resolve_enrichment_position
    from dsl_worker.chat.enrichment import _ensure_columns_on_table

    tid = resolve_table_id(db, str(project_id), table_id)
    if not tid:
        raise HTTPException(404, "Table not found")

    row = db.execute(
        sa_text("SELECT name, source, columns FROM tables WHERE id=:tid AND deleted_at IS NULL"),
        {"tid": tid},
    ).fetchone()
    if not row:
        raise HTTPException(404, "Table not found")
    table_name = row[0] or "table"
    source = row[1] or ""
    cols = row[2] if isinstance(row[2], list) else (json.loads(row[2]) if row[2] else [])

    all_query_cols = [c for c in cols if isinstance(c, dict) and not c.get("enrichment_id")]

    # Resolve the adopt scope from the request body.
    if body.columns is not None:
        # Multi-column (block run): adopt exactly the query columns named
        # in the list. Lenient + idempotent — names that are already
        # promoted or no longer query columns are silently skipped, so the
        # caller can pass a raw selection without pre-filtering. Nothing
        # left to adopt → empty result (the caller's other columns may
        # already be enrichments), NOT an error.
        requested = set(body.columns)
        target_query_cols = [c for c in all_query_cols if c.get("name") in requested]
        if not target_query_cols:
            return {"enrichment_id": None, "short_id": None, "columns": []}
    elif not all_query_cols:
        # Nothing to adopt — caller should just use the existing enrichment.
        raise HTTPException(400, "No query columns on this table to promote")
    elif body.column:
        # Single-column scope: narrow the adopt set to just the requested
        # column. Avoids the "click Run on URL → enrichment scopes in Name
        # + Price too" surprise. The caller still gets a usable enrichment
        # focused on the one field they're trying to backfill.
        target_query_cols = [
            c for c in all_query_cols if c.get("name") == body.column
        ]
        if not target_query_cols:
            raise HTTPException(
                400,
                f"Column {body.column!r} is not a query column on this table",
            )
    else:
        target_query_cols = all_query_cols
    target_col_set = {c["name"] for c in target_query_cols if c.get("name")}

    # Find an existing ghost whose adopted columns match this scope
    # exactly. A single-column ghost and an all-columns ghost can
    # coexist on the same table — they serve different intents.
    all_ghosts = db.execute(
        sa_text(
            "SELECT id::text, short_id, columns FROM enrichments "
            "WHERE table_id=:tid AND deleted_at IS NULL "
            "AND COALESCE(action->>'ghost', '') = '1' "
            "ORDER BY created_at ASC"
        ),
        {"tid": tid},
    ).fetchall()
    existing_ghost = None
    for gid, gsid, gcols_raw in all_ghosts:
        gcols = gcols_raw if isinstance(gcols_raw, list) else (
            json.loads(gcols_raw) if gcols_raw else []
        )
        ghost_set = {c.get("name") for c in gcols if isinstance(c, dict) and c.get("name")}
        if ghost_set == target_col_set:
            existing_ghost = (gid, gsid, gcols)
            break

    if existing_ghost:
        eid, short_id, ghost_cols = existing_ghost
        # Defensive: re-stamp enrichment_id on the table's column defs
        # in case it was lost (e.g. someone edited columns directly).
        _ensure_columns_on_table(db, tid, target_query_cols, enrichment_id=eid)
        db.commit()
        return {"enrichment_id": eid, "short_id": short_id, "columns": ghost_cols}

    # Fresh ghost. Synthesize an action prompt from the table's source +
    # column list. Steered toward a focused, budget-conscious lookup —
    # these auto-promoted columns are almost always simple factual
    # backfills (the user clicked "Turn into enrichment & run" on scraped
    # columns), so the agent should resolve them in a search or two, not
    # an open-ended research crawl that burns the whole per-row budget.
    col_names = [c["name"] for c in target_query_cols if c.get("name")]
    col_list_md = "\n".join(
        f"- **{c['name']}** ({c.get('type', 'text')})"
        for c in target_query_cols if c.get("name")
    )
    source_hint = f" The table was originally populated from `{source}`." if source else ""
    prompt = (
        f"For each row in '{table_name}', use the row's most identifying "
        f"existing field (its name/title, or a URL/domain/identifier) to look "
        f"up and fill these columns:\n\n"
        f"{col_list_md}\n\n"
        f"These are straightforward factual lookups. Do ONE focused web search "
        f"keyed on the row's identifier — for well-known subjects the answer is "
        f"usually right there in the top results. Don't browse exhaustively or "
        f"chase obscure sources: if a value isn't found within a search or two, "
        f"leave it null rather than spending the whole budget. Only fill what's "
        f"missing; never fabricate."
        f"{source_hint}"
    )

    eid = str(__import__("uuid").uuid4())
    short_id = _next_enrichment_short_id(db, tid)
    position = _resolve_enrichment_position(db, tid)
    action = {
        "research": "research",
        "prompt": prompt,
        "ghost": "1",
    }
    enrichment_columns = [
        {"name": c["name"], "type": c.get("type", "text")}
        for c in target_query_cols if c.get("name")
    ]
    db.execute(
        sa_text(
            "INSERT INTO enrichments (id, table_id, short_id, name, columns, action, per_row_credit_cap, position, created_at) "
            "VALUES (:eid, :tid, :sid, :name, CAST(:cols AS jsonb), CAST(:action AS jsonb), :cap, :pos, now())"
        ),
        {
            "eid": eid,
            "tid": tid,
            "sid": short_id,
            "name": "Backfill missing",
            "cols": json.dumps(enrichment_columns),
            "action": json.dumps(action),
            "cap": 2.0,
            "pos": position,
        },
    )
    _ensure_columns_on_table(db, tid, target_query_cols, enrichment_id=eid)
    db.commit()

    return {"enrichment_id": eid, "short_id": short_id, "columns": enrichment_columns}


class FilterBody(BaseModel):
    column: str
    op: str
    value: Any = None


@router.post("/projects/{project_id}/tables/{table_id}/filters")
async def post_filter(
    project_id: UUID,
    table_id: str,
    body: FilterBody,
    user: CurrentUser = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    _verify(project_id, user.user_id, db)
    ctx = ToolContext(db=db, project_id=str(project_id), user_id=str(user.user_id), run_id=None)
    result, _ = await filter_set(
        {"table_id": str(table_id), "column": body.column, "op": body.op, "value": body.value},
        ctx,
    )
    return result


@router.delete("/projects/{project_id}/tables/{table_id}/filters/{column:path}")
async def delete_filter(
    project_id: UUID,
    table_id: str,
    column: str,
    user: CurrentUser = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    _verify(project_id, user.user_id, db)
    ctx = ToolContext(db=db, project_id=str(project_id), user_id=str(user.user_id), run_id=None)
    result, _ = await filter_clear(
        {"table_id": str(table_id), "column": column}, ctx
    )
    return result


class SortBody(BaseModel):
    column: str
    direction: str = "desc"


@router.post("/projects/{project_id}/tables/{table_id}/sort")
async def post_sort(
    project_id: UUID,
    table_id: str,
    body: SortBody,
    user: CurrentUser = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    _verify(project_id, user.user_id, db)
    from dsl_worker.chat.tools import sort_set as _sort_set
    ctx = ToolContext(db=db, project_id=str(project_id), user_id=str(user.user_id), run_id=None)
    result, _ = await _sort_set(
        {"table_id": str(table_id), "column": body.column, "direction": body.direction},
        ctx,
    )
    return result


@router.delete("/projects/{project_id}/tables/{table_id}/sort")
async def delete_sort(
    project_id: UUID,
    table_id: str,
    user: CurrentUser = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    _verify(project_id, user.user_id, db)
    from dsl_worker.chat.tools import sort_clear as _sort_clear
    ctx = ToolContext(db=db, project_id=str(project_id), user_id=str(user.user_id), run_id=None)
    result, _ = await _sort_clear({"table_id": str(table_id)}, ctx)
    return result


# ---- Cell traces (debug) --------------------------------------------------


@router.get("/projects/{project_id}/enrichments/{enrichment_id}/traces")
def list_cell_traces(
    project_id: UUID,
    enrichment_id: str,
    limit: int = 50,
    user: CurrentUser = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Recent per-cell traces for an enrichment — tier, model, tool calls,
    final values, error, cost, duration. Read-only debug view."""
    _verify(project_id, user.user_id, db)
    # Use the central resolver so t1e2 composite ids work and bare e<N>
    # collisions tie-break consistently.
    from dsl_worker.chat.tools import resolve_enrichment_id
    eid = resolve_enrichment_id(db, str(project_id), enrichment_id)
    if not eid:
        raise HTTPException(404, "enrichment not found")
    rows = db.execute(
        sa_text(
            """
            SELECT ct.id::text, ct.sample_id::text, ct.tier, ct.model,
                   ct.tool_calls, ct.final_values, ct.error,
                   ct.cost_credits, ct.duration_ms, ct.created_at
            FROM cell_traces ct
            WHERE ct.enrichment_id=:eid
            ORDER BY ct.created_at DESC
            LIMIT :lim
            """
        ),
        {"eid": eid, "lim": limit},
    ).fetchall()
    return {
        "enrichment_id": enrichment_id,
        "traces": [
            {
                "id": r[0],
                "sample_id": r[1],
                "tier": r[2],
                "model": r[3],
                "tool_calls": r[4],
                "final_values": r[5],
                "error": r[6],
                "cost_credits": r[7],
                "duration_ms": r[8],
                "created_at": r[9].isoformat() if r[9] else None,
            }
            for r in rows
        ],
    }


# ---- Approvals ------------------------------------------------------------
# Tools in APPROVAL_REQUIRED (see approvals.py) pause the agent loop and
# wait for a decision from the FE. These endpoints surface that flow:
#   GET   /approvals               → list pending (used to rehydrate the
#                                    approval card after a reconnect)
#   POST  /approvals/{id}/respond  → resolve a pending approval


@router.get("/projects/{project_id}/approvals")
async def list_approvals(
    project_id: UUID,
    user: CurrentUser = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    _verify(project_id, user.user_id, db)
    return {"approvals": await APPROVALS.list_for_project(str(project_id))}


class ApprovalDecision(BaseModel):
    approved: bool


@router.post("/projects/{project_id}/approvals/{approval_id}/respond")
async def respond_to_approval(
    project_id: UUID,
    approval_id: str,
    body: ApprovalDecision,
    user: CurrentUser = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Resolve a pending approval.

    Two flavors:
    - Blocking (table_delete / row_delete): the agent loop is awaiting
      `pending.future`. We just `resolve()` and let the loop continue.
    - Non-blocking (enrichment_run): the agent loop already moved on
      with a `{scheduled: true}` stub. The actual enrichment hasn't run
      yet — if approved, we fire it here (synchronously, like the
      `/enrichments/{eid}/run` REST endpoint does) and return the
      fresh `updated_rows` so the FE can patch cells without a refetch.
      Denied → we just resolve and return; agent already shaped its
      reply around "queued, awaiting your call."
    """
    _verify(project_id, user.user_id, db)
    pending = await APPROVALS.peek(approval_id)
    if pending is None:
        # Double-click or chat run ended — no-op, FE shouldn't surface
        # a confusing error.
        return {"ok": True, "approved": body.approved, "found": False}

    if pending.tool == "enrichment_run":
        # Always resolve (clears the registry entry) regardless of
        # approval outcome.
        await APPROVALS.resolve(approval_id, body.approved)
        if not body.approved:
            return {"ok": True, "approved": False, "found": True}

        # Fire the enrichment now. Mirrors post_run_enrichment's flow
        # (REST path, run_id=None, returns updated_rows for FE patching).
        args = dict(pending.args or {})
        scope = args.get("scope") or {"type": "all_unfilled"}
        eid = args.get("enrichment_id")
        if not eid:
            return {"ok": False, "error": "enrichment_id missing on approval"}

        ctx = ToolContext(
            db=db, project_id=str(project_id), user_id=str(user.user_id), run_id=None
        )
        canonical_eid = (
            __import__("dsl_worker.chat.tools", fromlist=["resolve_enrichment_id"]).resolve_enrichment_id(
                db, str(project_id), eid
            )
            or str(eid)
        )

        # Durable-jobs path (opt-in): create a background job + tasks and return
        # immediately. The coordinator streams cell_start/cell_done over the
        # enrichment_events SSE → live spinners + progress widget + per-cell
        # charging + refresh-survival, instead of blocking this request for the
        # whole run with zero events (the inline path below). Default off.
        from dsl_worker.chat.enrichment_jobs import (
            _approval_via_jobs_enabled,
            create_job_for_enrichment,
        )
        if _approval_via_jobs_enabled():
            job_scope = dict(scope) if isinstance(scope, dict) else {"type": "all_unfilled"}
            job_scope["overwrite"] = bool(args.get("overwrite", False))
            created = create_job_for_enrichment(
                db,
                project_id=str(project_id),
                enrichment_id=canonical_eid,
                scope=job_scope,
                user_id=str(user.user_id),
            )
            return {"ok": True, "approved": True, "found": True, **created}

        import asyncio as _asyncio
        task: _asyncio.Task = _asyncio.create_task(
            enrichment_run(
                {
                    "enrichment_id": str(eid),
                    "scope": scope,
                    "overwrite": bool(args.get("overwrite", False)),
                },
                ctx,
            )
        )
        await CANCELS.register(str(project_id), canonical_eid, task)
        cost = 0.0
        cancelled = False
        try:
            result, cost = await task
        except _asyncio.CancelledError:
            cancelled = True
            cost = float(getattr(ctx, "partial_cost_usd", 0.0) or 0.0)
            result = {"cancelled": True}
        finally:
            await CANCELS.unregister(str(project_id), canonical_eid, task)

        # Charge credits — same path post_run_enrichment uses.
        spend_cents = int(round(float(cost or 0.0) * 100))
        if spend_cents > 0:
            try:
                account = (
                    db.query(Account).filter(Account.user_id == user.user_id).first()
                )
                if account:
                    consume_credits(
                        db,
                        account,
                        spend_cents / CENTS_PER_CREDIT,
                        project_id=project_id,
                        reason="enrichment_run_approval",
                    )
                    db.commit()
                else:
                    log.warning(
                        "enrichment_run_approval: no account for user %s", user.user_id
                    )
            except Exception:
                try:
                    db.rollback()
                except Exception:
                    pass

        # Pull the affected rows so the FE can applyCellEdits without
        # a refetch. Same logic as post_run_enrichment.
        updated_rows: List[Dict[str, Any]] = []
        if not cancelled and isinstance(scope, dict) and scope.get("type") == "row_ids":
            row_ids = scope.get("row_ids") or []
            if row_ids:
                try:
                    from sqlalchemy import bindparam
                    stmt = (
                        sa_text(
                            "SELECT id::text, row, tags "
                            "FROM samples WHERE id::text IN :ids AND deleted_at IS NULL"
                        ).bindparams(bindparam("ids", expanding=True))
                    )
                    rows = db.execute(stmt, {"ids": list(row_ids)}).fetchall()
                    for r in rows:
                        payload: Dict[str, Any] = {"id": r[0]}
                        row_data = r[1] if isinstance(r[1], dict) else (json.loads(r[1]) if r[1] else {})
                        payload.update(row_data)
                        if r[2]:
                            payload["tags"] = r[2] if isinstance(r[2], dict) else json.loads(r[2])
                        updated_rows.append(payload)
                except Exception:
                    log.exception("respond_to_approval: failed to fetch updated rows for FE patch")

        return {
            "ok": True,
            "approved": True,
            "found": True,
            "result": result,
            "cost_usd": cost,
            "cancelled": cancelled,
            "updated_rows": updated_rows,
        }

    # Blocking path (table_delete / row_delete): old behavior.
    found = await APPROVALS.resolve(approval_id, body.approved)
    if not found:
        return {"ok": True, "approved": body.approved, "found": False}
    return {"ok": True, "approved": body.approved, "found": True}


# ---- Plan option picks ----------------------------------------------------
# `plan_options` is a blocking-style tool the agent calls to ask the user
# to pick between 2-4 options before continuing. Same shape as approvals
# except the resolution is the chosen option key, not a bool.


@router.get("/projects/{project_id}/plan_option_picks")
async def list_plan_option_picks(
    project_id: UUID,
    user: CurrentUser = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    _verify(project_id, user.user_id, db)
    return {"picks": await OPTION_PICKS.list_for_project(str(project_id))}


class PlanOptionResponse(BaseModel):
    chosen: str


@router.post("/projects/{project_id}/plan_option_picks/{pick_id}/respond")
async def respond_to_plan_option(
    project_id: UUID,
    pick_id: str,
    body: PlanOptionResponse,
    user: CurrentUser = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Resolve a pending plan_options pick. The chosen key flows back
    through the awaiting future inside the plan_options tool handler;
    the agent's next iteration sees `{chosen: '<key>'}` as the tool
    result and proceeds with that choice."""
    _verify(project_id, user.user_id, db)
    pending = await OPTION_PICKS.peek(pick_id)
    if pending is None:
        return {"ok": True, "chosen": body.chosen, "found": False}
    # Validate against the registered option keys so a stale FE can't
    # smuggle in a key the agent doesn't expect.
    valid_keys = {o.get("key") for o in pending.options if isinstance(o, dict)}
    if body.chosen not in valid_keys:
        raise HTTPException(
            400,
            f"chosen={body.chosen!r} is not one of {sorted(k for k in valid_keys if k)}",
        )
    await OPTION_PICKS.resolve(pick_id, body.chosen)
    return {"ok": True, "chosen": body.chosen, "found": True}


# ---- Cancel ---------------------------------------------------------------
# Cancels an in-flight REST enrichment run. Chat runs are cancelled via the
# SSE disconnect; this is the parallel path for the column ▶ button etc.


@router.post("/projects/{project_id}/enrichments/{enrichment_id}/cancel")
async def cancel_enrichment(
    project_id: UUID,
    enrichment_id: str,
    user: CurrentUser = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    _verify(project_id, user.user_id, db)
    # Resolve short_id (e1, e2…) to UUID if needed.
    from dsl_worker.chat.tools import resolve_enrichment_id
    eid = resolve_enrichment_id(db, str(project_id), enrichment_id) or enrichment_id
    cancelled = await CANCELS.cancel_enrichment(str(project_id), eid)
    return {"ok": True, "cancelled": cancelled}


@router.get("/projects/{project_id}/enrichments/running")
async def list_running_enrichments(
    project_id: UUID,
    user: CurrentUser = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    _verify(project_id, user.user_id, db)
    return {"enrichment_ids": await CANCELS.list_running(str(project_id))}


@router.get("/projects/{project_id}/cells/running")
async def list_running_cells(
    project_id: UUID,
    user: CurrentUser = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Cells currently being filled by an in-flight enrichment run.

    Used by the FE on page mount + during active runs to rebuild
    pendingCells, so the per-cell spinner survives a refresh. Each
    entry: {enrichment_id (uuid), sample_id, columns: [...]}.
    """
    _verify(project_id, user.user_id, db)
    from dsl_worker.chat.cell_runs import REGISTRY as CELL_RUNS
    return {"cells": await CELL_RUNS.list_for_project(str(project_id))}


@router.get("/projects/{project_id}/background-tasks")
async def list_background_tasks(
    project_id: UUID,
    user: CurrentUser = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Running background tasks for this project — the wait=false tools
    spawned via the chat agent loop. Lets the FE rehydrate running
    indicators on page refresh + show a "tasks in flight" chip while
    bg work is still going. Mirrors /enrichments/running for the
    new background_tasks subsystem.

    Returns {tasks: [{task_id, kind, task_key, started_at,
    partial_cost_credits, run_id}, ...]} — same shape the agent sees
    via task_status for status='running' rows.
    """
    _verify(project_id, user.user_id, db)
    from dsl_worker.chat.background_tasks import list_running_rows
    return {"tasks": list_running_rows(db, str(project_id))}


@router.post("/projects/{project_id}/background-tasks/{task_id}/cancel")
async def cancel_background_task(
    project_id: UUID,
    task_id: str,
    user: CurrentUser = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Cancel a single background task by short_id or uuid. The bg
    task's CancelledError handler captures partial cost + updates the
    chat_background_tasks row to status='cancelled'."""
    _verify(project_id, user.user_id, db)
    from dsl_worker.chat.background_tasks import REGISTRY as BG_REGISTRY
    bg = await BG_REGISTRY.lookup_by_short_or_uuid(str(project_id), task_id)
    if bg is None or bg.task.done():
        return {"ok": True, "cancelled": False}
    bg.task.cancel()
    return {"ok": True, "cancelled": True}
