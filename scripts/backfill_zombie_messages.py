"""Reconstruct chat_messages for runs that ended terminally without one.

A "zombie" run is one with `status in (failed, cancelled, completed)` AND
`assistant_message_id IS NULL`. The events ARE in chat_run_events — we
just never wrote a chat_messages row, so the FE shows nothing in chat
history for that turn.

This script walks every zombie, reconstructs:
  content     ← latest text_checkpoint payload.full_content
  tool_log    ← walked from tool_call + tool_result events
  applied_changes.interrupted = True
  resume_input ← latest text_checkpoint may not have it, but if any
                 ChatMessage on the same project carries one we can
                 cross-link (skipped for now — keep it simple)

…then inserts a chat_messages row + sets run.assistant_message_id.

Idempotent: skips runs that already have an assistant_message_id.

Usage:
    python -m scripts.backfill_zombie_messages              # dry-run
    python -m scripts.backfill_zombie_messages --apply      # actually write
    python -m scripts.backfill_zombie_messages --project <id>   # one project
"""
from __future__ import annotations

import sys
from typing import Any, Dict, List, Optional
from uuid import UUID

from dotenv import load_dotenv

load_dotenv(".env")

from dsl_api.db import SessionLocal  # noqa: E402
from dsl_api.models import ChatMessage, ChatRun, ChatRunEvent  # noqa: E402
from dsl_api.models.chat_run import RUN_TERMINAL_STATUSES, RUN_STATUS_PAUSED  # noqa: E402


def reconstruct_message(db, run: ChatRun) -> Optional[ChatMessage]:
    evs = (
        db.query(ChatRunEvent)
        .filter(ChatRunEvent.run_id == run.id)
        .order_by(ChatRunEvent.seq.asc())
        .all()
    )
    if not evs:
        return None
    content = ""
    tool_log: List[Dict[str, Any]] = []
    tool_log_idx: Dict[str, int] = {}
    sources: List[Dict[str, Any]] = []
    sources_seen: set = set()
    total_cost: float = 0.0
    for e in evs:
        pl = e.payload or {}
        if e.type == "text_checkpoint":
            content = pl.get("full_content") or content
        elif e.type == "tool_call":
            cid = pl.get("id") or f"_evt_{e.seq}"
            tool_log_idx[cid] = len(tool_log)
            tool_log.append({
                "id": cid,
                "name": pl.get("name", "?"),
                "args_preview": pl.get("args_preview", ""),
            })
        elif e.type == "tool_result":
            cid = pl.get("id")
            cost = pl.get("cost") or 0
            if isinstance(cost, (int, float)):
                total_cost += float(cost)
            if cid and cid in tool_log_idx:
                tool_log[tool_log_idx[cid]].update({
                    "summary": pl.get("summary"),
                    "cost": cost,
                })
        elif e.type == "source_added":
            url = pl.get("url")
            if url and url not in sources_seen:
                sources_seen.add(url)
                sources.append({
                    "n": pl.get("n", len(sources) + 1),
                    "url": url,
                    "title": pl.get("title") or url,
                })
    ac: Dict[str, Any] = {}
    # Only mark interrupted if the run actually failed; completed-but-
    # zombie runs (rare, but possible) still get a recovered message
    # with no scary flag.
    if run.error or run.status not in (RUN_STATUS_PAUSED,):
        if run.error:
            ac["error"] = (run.error or "")[:500]
            ac["interrupted"] = True
    if tool_log:
        ac["tool_log"] = tool_log
    if sources:
        ac["sources"] = sources
    if total_cost > 0:
        ac["total_cost_usd"] = round(total_cost, 4)
    msg = ChatMessage(
        project_id=run.project_id,
        role="assistant",
        content=content,
        applied_changes=ac if ac else None,
        version_id=run.version_id,
        run_id=run.id,
    )
    return msg


def main(argv: List[str]) -> int:
    apply = "--apply" in argv
    project_filter: Optional[UUID] = None
    if "--project" in argv:
        try:
            project_filter = UUID(argv[argv.index("--project") + 1])
        except (IndexError, ValueError):
            print("--project requires a UUID arg", file=sys.stderr)
            return 2

    db = SessionLocal()
    try:
        q = db.query(ChatRun).filter(
            ChatRun.assistant_message_id.is_(None),
            ChatRun.status.in_(list(RUN_TERMINAL_STATUSES) + [RUN_STATUS_PAUSED]),
        )
        if project_filter is not None:
            q = q.filter(ChatRun.project_id == project_filter)
        zombies = q.order_by(ChatRun.created_at.asc()).all()
        print(f"Found {len(zombies)} zombie run(s){'  (DRY RUN — pass --apply to write)' if not apply else ''}")
        recovered = 0
        empty = 0
        for run in zombies:
            msg = reconstruct_message(db, run)
            if msg is None:
                empty += 1
                print(f"  {str(run.id)[:8]}  status={run.status}  no events — skipped")
                continue
            tool_count = len((msg.applied_changes or {}).get("tool_log") or [])
            print(
                f"  {str(run.id)[:8]}  status={run.status}  "
                f"content={len(msg.content or '')}ch  tools={tool_count}  "
                f"project={str(run.project_id)[:8]}"
            )
            if apply:
                db.add(msg)
                db.flush()
                run.assistant_message_id = msg.id
                recovered += 1
        if apply:
            db.commit()
            print(f"\nRecovered {recovered} message(s); {empty} zombie(s) had no events to recover.")
        else:
            print(f"\nDry run only. {len(zombies) - empty} would be recovered; {empty} have no events.")
    finally:
        db.close()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
