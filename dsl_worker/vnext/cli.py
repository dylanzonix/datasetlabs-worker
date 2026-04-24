"""CLI REPL for the V-next chat agent.

Usage:
  python -m dsl_worker.vnext.cli ./projects/test.sqlite

Each project is one SQLite file plus a sibling `_snapshots/` directory
for version history. The OPENAI_API_KEY env var is required.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from .agent import ChatAgent


GREEN = "\033[92m"
GRAY = "\033[90m"
CYAN = "\033[96m"
YELLOW = "\033[93m"
RESET = "\033[0m"


def _format_arg_preview(args: dict) -> str:
    """Render a one-line preview of tool args, truncating long values."""
    parts = []
    for k, v in args.items():
        s = json.dumps(v, default=str)
        if len(s) > 80:
            s = s[:77] + "..."
        parts.append(f"{k}={s}")
    return ", ".join(parts)


def _format_result_preview(result: dict) -> str:
    """Render a one-line preview of a tool result."""
    if "error" in result:
        return f"ERROR: {result['error']}"
    s = json.dumps(result, default=str)
    if len(s) > 200:
        s = s[:197] + "..."
    return s


def main() -> int:
    parser = argparse.ArgumentParser(description="V-next chat REPL")
    parser.add_argument("project", help="path to the project SQLite file (created if missing)")
    parser.add_argument("--model", default="gpt-5.4")
    parser.add_argument("--quiet-tools", action="store_true",
                        help="suppress tool call / result lines")
    args = parser.parse_args()

    load_dotenv()
    if not os.environ.get("OPENAI_API_KEY"):
        print("OPENAI_API_KEY is not set", file=sys.stderr)
        return 1

    project_path = Path(args.project).expanduser().resolve()
    project_path.parent.mkdir(parents=True, exist_ok=True)
    snapshot_dir = project_path.parent / f"{project_path.stem}_snapshots"

    is_new = not project_path.exists()
    agent = ChatAgent(
        project_path=project_path,
        snapshot_dir=snapshot_dir,
        model=args.model,
    )
    # Open the connection up front so the schema is applied
    agent.conn()

    if not args.quiet_tools:
        agent.on_tool_call = lambda name, args_: print(
            f"{GRAY}> {name}({_format_arg_preview(args_)}){RESET}"
        )
        agent.on_tool_result = lambda name, result: print(
            f"{GRAY}  → {_format_result_preview(result)}{RESET}"
        )

    print(f"{CYAN}Project: {project_path}{RESET}")
    print(f"{CYAN}Snapshots: {snapshot_dir}{RESET}")
    print(f"{CYAN}Model: {args.model}{RESET}")
    if is_new:
        print(f"{YELLOW}(new project — empty){RESET}")
    print(f"{GRAY}Type your message. Ctrl-D or 'exit' to quit.{RESET}")
    print()

    while True:
        try:
            user = input(f"{GREEN}you ▸ {RESET}").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user:
            continue
        if user.lower() in ("exit", "quit", ":q"):
            break

        try:
            reply = agent.send(user)
        except Exception as e:
            print(f"{YELLOW}error: {type(e).__name__}: {e}{RESET}")
            continue

        if reply:
            print(f"\n{CYAN}agent ▸{RESET} {reply}\n")
        else:
            print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
