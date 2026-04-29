"""Dump the full trace for a chat run from Supabase.

Usage:
    python -m scripts.diagnose_run <project_id_or_run_id>

Shows every tool call's full args + full result_text, every reasoning
delta concatenated by round, every assistant token concatenated by
round, and run timing. The 140-char `summary` truncation in the FE
event log is bypassed — this dumps everything the model actually saw.

Persisted from streaming.py:run_agent_loop:
  tool_call.args_full      — full JSON dict the model passed
  tool_result.result_text  — full string the model received
  thinking deltas + token deltas — concatenable per-round
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from typing import Any, Dict, List
from uuid import UUID

from dotenv import load_dotenv

load_dotenv(".env")

from dsl_api.db import SessionLocal  # noqa: E402
from dsl_api.models import ChatMessage, ChatRun, ChatRunEvent, Project  # noqa: E402


def _pretty(obj: Any, max_chars: int = 0) -> str:
    """JSON-format obj. max_chars=0 means no cap."""
    s = json.dumps(obj, indent=2, default=str, ensure_ascii=False)
    if max_chars and len(s) > max_chars:
        s = s[:max_chars] + f"\n... (+{len(s) - max_chars} chars elided)"
    return s


def _dump_run(db, run: ChatRun, project: Project) -> None:
    print(f"\n{'=' * 78}")
    print(f"RUN {run.id}")
    print(f"  project: {project.name} ({project.id})")
    print(f"  status:  {run.status}   phase: {run.current_phase or '-'}")
    print(f"  started: {run.started_at}   completed: {run.completed_at}")
    if run.error:
        print(f"  ERROR:   {run.error}")
    print(f"{'=' * 78}\n")

    evs = (
        db.query(ChatRunEvent)
        .filter(ChatRunEvent.run_id == run.id)
        .order_by(ChatRunEvent.seq.asc())
        .all()
    )
    print(f"{len(evs)} events\n")

    # Group thinking + token deltas by "round" (we don't have a
    # round-boundary marker, but tool_call breaks usually correspond
    # to round transitions — concatenate between them).
    thinking_buf: List[str] = []
    token_buf: List[str] = []
    round_idx = 0

    def flush_text():
        nonlocal thinking_buf, token_buf
        if thinking_buf:
            blob = "".join(thinking_buf)
            print(f"  [thinking] ({len(blob)} chars)")
            print("    " + blob.replace("\n", "\n    "))
            print()
            thinking_buf = []
        if token_buf:
            blob = "".join(token_buf)
            print(f"  [assistant text] ({len(blob)} chars)")
            print("    " + blob.replace("\n", "\n    "))
            print()
            token_buf = []

    for e in evs:
        pl = e.payload or {}
        ts = e.created_at.strftime("%H:%M:%S")

        if e.type == "thinking":
            thinking_buf.append(pl.get("content", ""))
            continue
        if e.type == "token":
            token_buf.append(pl.get("content", ""))
            continue
        if e.type == "heartbeat":
            continue
        if e.type == "row_count":
            continue

        # Non-text event — flush buffered text first.
        flush_text()

        if e.type == "tool_call":
            round_idx += 1
            print(f"--- [{ts}] seq={e.seq} TOOL_CALL: {pl.get('name')} ---")
            args_full = pl.get("args_full")
            args_preview = pl.get("args_preview")
            if args_full is not None:
                print(f"  args_full:")
                print("    " + _pretty(args_full).replace("\n", "\n    "))
            elif args_preview:
                print(f"  args_preview (legacy event, no args_full): {args_preview}")
            print()
        elif e.type == "tool_result":
            print(f"--- [{ts}] seq={e.seq} TOOL_RESULT: {pl.get('name')}  cost=${pl.get('cost') or 0} ---")
            result_text = pl.get("result_text")
            if result_text is not None:
                print(f"  result_text ({len(result_text)} chars):")
                print("    " + result_text.replace("\n", "\n    "))
            else:
                summary = pl.get("summary") or ""
                print(f"  summary (legacy, truncated): {summary}")
            print()
        elif e.type == "thinking_checkpoint":
            content = pl.get("content") or ""
            round_num = pl.get("round")
            tag = f"round {round_num}" if round_num is not None else "—"
            print(f"--- [{ts}] seq={e.seq} THINKING_CHECKPOINT ({tag}) "
                  f"({len(content)} chars) ---")
            print("    " + content.replace("\n", "\n    "))
            print()
        elif e.type == "text_checkpoint":
            content = pl.get("full_content") or ""
            print(f"--- [{ts}] seq={e.seq} TEXT_CHECKPOINT "
                  f"(cumulative content, {len(content)} chars) ---")
            print("    " + content.replace("\n", "\n    "))
            print()
        elif e.type in ("done", "paused", "cancelled", "error"):
            print(f"=== [{ts}] seq={e.seq} {e.type.upper()}: {_pretty(pl, 800)}")
            print()
        elif e.type == "tool_start":
            print(f"  [{ts}] seq={e.seq} tool_start: {pl.get('tools')}")
        elif e.type == "version":
            print(f"  [{ts}] seq={e.seq} version: v{pl.get('version_number')} ({pl.get('label') or '-'})")
        elif e.type == "change":
            print(f"  [{ts}] seq={e.seq} change.{pl.get('field')}: {pl.get('description')}")
        elif e.type == "source_added":
            print(f"  [{ts}] seq={e.seq} source[{pl.get('n')}] {pl.get('url')}")
        elif e.type == "questions":
            print(f"  [{ts}] seq={e.seq} ASK_QUESTIONS:")
            for q in pl.get("questions") or []:
                print(f"    - {q.get('label') or q.get('question')}")
        elif e.type == "suggestions":
            print(f"  [{ts}] seq={e.seq} SUGGEST_REPLIES:")
            for s in pl.get("items") or []:
                print(f"    - {s.get('label')!r} → {s.get('message')!r}")
        else:
            print(f"  [{ts}] seq={e.seq} {e.type}: {_pretty(pl, 200)}")

    flush_text()


def main(argv: List[str]) -> int:
    if len(argv) != 2:
        print(__doc__, file=sys.stderr)
        return 2

    target = argv[1].strip()
    try:
        target_uuid = UUID(target)
    except ValueError:
        print(f"Not a UUID: {target}", file=sys.stderr)
        return 2

    db = SessionLocal()
    try:
        # Try project-id first, then run-id.
        project = db.query(Project).filter(Project.id == target_uuid).first()
        if project:
            print(f"\n### Project: {project.name} ({project.id})")
            print(f"### Mode: {project.mode}    Status: {project.status}")
            msgs = (
                db.query(ChatMessage)
                .filter(ChatMessage.project_id == project.id)
                .order_by(ChatMessage.created_at.asc())
                .all()
            )
            for m in msgs:
                print(f"\n[{m.role} @ {m.created_at}] (msg {m.id})")
                print(m.content or "")
                ac = m.applied_changes or {}
                if ac:
                    print(f"  applied_changes keys: {list(ac.keys())}")
            runs = (
                db.query(ChatRun)
                .filter(ChatRun.project_id == project.id)
                .order_by(ChatRun.created_at.asc())
                .all()
            )
            print(f"\n### {len(runs)} run(s)")
            for r in runs:
                _dump_run(db, r, project)
        else:
            run = db.query(ChatRun).filter(ChatRun.id == target_uuid).first()
            if not run:
                print(f"No project or run with id {target_uuid}", file=sys.stderr)
                return 1
            project = db.query(Project).filter(Project.id == run.project_id).first()
            _dump_run(db, run, project)
    finally:
        db.close()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
