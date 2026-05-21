"""Per-phase timing breakdown for a chat run.

Usage:
    python -m scripts.analyze_run <run_id>

Reads chat_run_events sorted by mono_ns (trustworthy) with created_at
fallback (for pre-instrumentation events that have NULL mono_ns). Prints:

  - per-event-type aggregate: count, total_ms, avg_ms, max_ms
  - per-phase-path aggregate from `phase_start`/`phase_end` pairs
  - top-N largest gaps between successive events with their surrounding
    types (so we can see where time leaks between instrumented work)

mono_ns is from `time.perf_counter_ns()` at the writer process. It's
monotonic within a single process — comparing two events from the same
worker process gives an exact elapsed duration. Across worker restarts
the reference point resets, but a single run executes on one process so
that's not an issue in practice.
"""

from __future__ import annotations

import os
import sys
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from dotenv import load_dotenv

load_dotenv(".env")

from sqlalchemy import text as sa_text  # noqa: E402

from dsl_api.db import SessionLocal  # noqa: E402


def _fmt_ms(ms: float) -> str:
    if ms < 1:
        return f"{ms*1000:>6.0f} us"
    if ms < 1000:
        return f"{ms:>6.1f} ms"
    return f"{ms/1000:>6.2f} s"


def _row_label(row: Dict) -> str:
    """One-line label for an event row in gap listings."""
    t = row["type"]
    payload = row["payload"] or {}
    if t in ("phase", "phase_start", "phase_end"):
        return f"{t}({payload.get('phase', '?')})"
    if t == "tool_call":
        return f"tool_call({payload.get('name', '?')})"
    if t == "tool_result":
        return f"tool_result({payload.get('name', '?')})"
    if t == "cell_filled":
        return f"cell_filled({payload.get('status', '?')})"
    if t == "slow_commit":
        return f"slow_commit({payload.get('label', '?')})"
    return t


def analyze(run_id: str) -> int:
    db = SessionLocal()
    try:
        rows = db.execute(
            sa_text(
                """
                SELECT seq, type, payload, mono_ns,
                       EXTRACT(EPOCH FROM created_at) * 1e9 AS created_ns
                FROM chat_run_events
                WHERE run_id = :rid
                ORDER BY seq
                """
            ),
            {"rid": run_id},
        ).fetchall()
    finally:
        db.close()

    if not rows:
        print(f"No events found for run {run_id}")
        return 1

    # Use mono_ns where present; fall back to created_at-derived ns for old rows.
    points: List[Dict] = []
    for r in rows:
        seq, etype, payload, mono_ns, created_ns = r
        ts_ns = int(mono_ns) if mono_ns is not None else int(created_ns or 0)
        points.append({
            "seq": int(seq),
            "type": etype,
            "payload": payload or {},
            "ns": ts_ns,
            "mono": mono_ns is not None,
        })

    has_mono = any(p["mono"] for p in points)
    if not has_mono:
        print(
            "WARNING: no mono_ns timestamps in this run. Falling back to "
            "created_at (may be out-of-order under concurrent inserts)."
        )

    # ---- per-event-type aggregate -----------------------------------------
    print("\n=== Per-event-type ===")
    type_stats: Dict[str, List[float]] = defaultdict(list)
    for i in range(1, len(points)):
        gap_ns = points[i]["ns"] - points[i - 1]["ns"]
        type_stats[points[i]["type"]].append(gap_ns / 1e6)
    print(f"{'type':30s} {'count':>6s} {'avg_ms':>10s}")
    for t in sorted(type_stats.keys()):
        gaps = type_stats[t]
        avg = sum(gaps) / len(gaps)
        print(f"{t:30s} {len(gaps):>6d} {_fmt_ms(avg):>10s}")

    # ---- phase_start/phase_end pairs --------------------------------------
    print("\n=== Phases ===")
    pair_durs: Dict[str, List[float]] = defaultdict(list)
    open_phases: Dict[str, int] = {}  # phase -> seq of last open start
    for p in points:
        t = p["type"]
        payload = p["payload"]
        phase_name = payload.get("phase")
        if t == "phase_start" and phase_name:
            open_phases[phase_name] = p["seq"]
        elif t == "phase_end" and phase_name:
            dur_ms = payload.get("dur_ms")
            if dur_ms is None and phase_name in open_phases:
                start_seq = open_phases[phase_name]
                start = next((x for x in points if x["seq"] == start_seq), None)
                if start:
                    dur_ms = (p["ns"] - start["ns"]) / 1e6
            if dur_ms is not None:
                pair_durs[phase_name].append(float(dur_ms))
            open_phases.pop(phase_name, None)
        elif t == "phase" and phase_name:
            # Single-shot marker — record one zero-duration entry so we
            # at least surface that it fired.
            pair_durs.setdefault(phase_name, [])
    if pair_durs:
        print(f"{'phase':45s} {'count':>6s} {'total':>10s} {'avg':>10s} {'max':>10s}")
        for ph in sorted(pair_durs.keys()):
            durs = pair_durs[ph]
            if not durs:
                print(f"{ph:45s} {'-':>6s} {'(marker)':>10s} {'-':>10s} {'-':>10s}")
                continue
            tot = sum(durs)
            print(
                f"{ph:45s} {len(durs):>6d} "
                f"{_fmt_ms(tot):>10s} {_fmt_ms(tot/len(durs)):>10s} {_fmt_ms(max(durs)):>10s}"
            )
    else:
        print("(no phase_start/phase_end pairs in this run — instrumentation not yet active?)")

    # ---- top-N biggest gaps -----------------------------------------------
    print("\n=== Top 15 biggest event gaps ===")
    gaps: List[Tuple[float, Dict, Dict]] = []
    for i in range(1, len(points)):
        gap_ms = (points[i]["ns"] - points[i - 1]["ns"]) / 1e6
        gaps.append((gap_ms, points[i - 1], points[i]))
    gaps.sort(key=lambda x: -x[0])
    print(f"{'gap':>10s}  {'seq':>5s}  prev → next")
    for gap_ms, prev, curr in gaps[:15]:
        print(
            f"{_fmt_ms(gap_ms):>10s}  #{curr['seq']:>4d}  "
            f"{_row_label(prev)} → {_row_label(curr)}"
        )

    # ---- slow_commits -----------------------------------------------------
    slow = [p for p in points if p["type"] == "slow_commit"]
    if slow:
        print("\n=== Slow commits ===")
        for p in slow:
            payload = p["payload"]
            print(f"  #{p['seq']:>4d}  {payload.get('label', '?'):30s}  {_fmt_ms(float(payload.get('dur_ms') or 0))}")

    return 0


def main(argv: List[str]) -> int:
    if len(argv) < 2:
        print(__doc__)
        return 2
    return analyze(argv[1])


if __name__ == "__main__":
    sys.exit(main(sys.argv))
