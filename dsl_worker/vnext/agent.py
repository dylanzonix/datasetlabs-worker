"""V-next chat agent.

Wraps the OpenAI Responses API with our flat tool surface. One
`ChatAgent` instance owns one project (one SQLite + one snapshot dir +
one chat history). Calling `send(user_text)` runs the agent until it
emits a final assistant message; tool calls are executed inline against
the project and their outputs fed back to the model.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from openai import OpenAI

from . import db, tools


logger = logging.getLogger(__name__)


SYSTEM_PROMPT = """\
You are a chat agent that builds data tables collaboratively with the user.
The user describes what they want; you act in small batches and report back
before scaling up. The conversation is the control surface — there is no
separate setup screen, no plan to approve upfront. Just chat, do the next
obvious thing, and ask before bulk operations.

# Project state

- The project is one SQLite file with a `rows` table (one row of data each)
  and a `columns` table (column definitions). Both grow over time.
- Each row is a dict of column-name → value. Columns can have a `format`
  (consistency hint shown to anything filling the column) and a
  `description` (what the column means).
- Snapshots are taken automatically before destructive operations, so
  deletes can be rewound via versions_checkout.

# Tools

You have flat function tools. They fall into four families:

- **rows_***  — add/get/count/sample/update/delete rows. Filters are dicts:
  `{col: value}` for equality, `{col__lt: n}` / __gt / __lte / __gte,
  `{col__contains: s}`, `{col__in: [...]}`, `{col__isnull: true}`,
  `{col: null}` for IS NULL. Multiple keys AND together.
- **columns_*** — add/list/modify/delete column definitions.
- **versions_*** — list snapshots, checkout to roll back.
- (Sources like FullEnrich, Apify, etc. will be added; for now you only
  have the table-management tools and rows_add for direct inserts.)

# Behavior rules

1. **Do then ask.** For small operations (≤10 rows, ≤$0.10), just do it
   and report. For anything larger, do a small batch first (say ~10),
   show the result, ask before scaling up.

2. **Always sample first** before bulk filling/committing. The user wants
   to see how it looks on a few rows before committing 100s of API
   credits.

3. **Ask in plain language.** No structured questions. "Want me to grab
   80 more?" or "Got 10 — looks good, run on the rest?" are perfect.
   The user can answer freely.

4. **Pick reasonable defaults.** When the user says "find me 100 X" you
   start with 10 to validate the source/filter, not 100. They have to
   give the green light for the rest.

5. **Filters before destructive ops.** Before `rows_delete`, run
   `rows_count` with the same `where` and tell the user how many will
   be deleted, then confirm.

6. **Keep schemas consistent.** When adding a column whose values come
   in many formats (phone, money, ranges), set a clear `format` so all
   future cells render uniformly.

7. **One or two tool calls per turn.** Don't chain ten things hoping
   nothing breaks. Do one meaningful action, observe, then proceed.

# Output style

Be concise. After a tool call, say what happened in one or two sentences
and (when relevant) suggest the next obvious move. No headers, no lists
unless they're genuinely shorter that way. The user reads everything in
chat.
"""


@dataclass
class Turn:
    role: str
    content: Optional[str] = None
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    tool_call_id: Optional[str] = None  # only on tool-result turns


def _to_input_items(turns: List[Turn]) -> List[Dict[str, Any]]:
    """Convert our history into Responses API input items."""
    items: List[Dict[str, Any]] = []
    for t in turns:
        if t.role == "user":
            items.append({"role": "user", "content": t.content or ""})
        elif t.role == "assistant":
            if t.content:
                items.append({"role": "assistant", "content": t.content})
            for tc in t.tool_calls:
                items.append({
                    "type": "function_call",
                    "call_id": tc["call_id"],
                    "name": tc["name"],
                    "arguments": json.dumps(tc.get("arguments") or {}),
                })
        elif t.role == "tool":
            items.append({
                "type": "function_call_output",
                "call_id": t.tool_call_id,
                "output": t.content or "",
            })
    return items


@dataclass
class ChatAgent:
    project_path: Path
    snapshot_dir: Path
    model: str = "gpt-5.4"
    history: List[Turn] = field(default_factory=list)
    client: OpenAI = field(default_factory=OpenAI)
    on_tool_call: Optional[Callable[[str, Dict[str, Any]], None]] = None
    on_tool_result: Optional[Callable[[str, Dict[str, Any]], None]] = None
    on_assistant_text: Optional[Callable[[str], None]] = None
    max_turns: int = 12

    # Connection is lazy so versions_checkout can rebuild the file
    _conn: Optional[sqlite3.Connection] = None

    def conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = db.connect(self.project_path)
        return self._conn

    def reopen(self) -> None:
        if self._conn is not None:
            self._conn.close()
        self._conn = db.connect(self.project_path)

    def send(self, user_text: str) -> str:
        """Run the agent until it produces a final assistant message. Returns that text."""
        self.history.append(Turn(role="user", content=user_text))

        for turn_idx in range(self.max_turns):
            response = self.client.responses.create(
                model=self.model,
                instructions=SYSTEM_PROMPT,
                input=_to_input_items(self.history),
                tools=tools.to_openai_tools(),
                tool_choice="auto",
            )

            # Walk the response items
            assistant_text_parts: List[str] = []
            tool_calls: List[Dict[str, Any]] = []
            for item in response.output:
                if item.type == "message":
                    for content in item.content:
                        if content.type == "output_text":
                            assistant_text_parts.append(content.text)
                elif item.type == "function_call":
                    tool_calls.append({
                        "call_id": item.call_id,
                        "name": item.name,
                        "arguments": json.loads(item.arguments) if item.arguments else {},
                    })

            assistant_text = "".join(assistant_text_parts).strip() or None

            # Persist the assistant turn
            self.history.append(Turn(
                role="assistant",
                content=assistant_text,
                tool_calls=tool_calls,
            ))

            if assistant_text and self.on_assistant_text:
                self.on_assistant_text(assistant_text)

            if not tool_calls:
                return assistant_text or ""

            # Execute each tool call in order
            needs_reopen = False
            for tc in tool_calls:
                if self.on_tool_call:
                    self.on_tool_call(tc["name"], tc["arguments"])

                spec = tools._REGISTRY.get(tc["name"])
                if spec and spec.modifies_db:
                    db.take_snapshot(
                        self.conn(),
                        self.project_path,
                        f"before {tc['name']}",
                        self.snapshot_dir,
                    )

                result = tools.call_tool(
                    tc["name"],
                    tc["arguments"],
                    conn=self.conn(),
                    project_path=self.project_path,
                    snapshot_dir=self.snapshot_dir,
                )

                if self.on_tool_result:
                    self.on_tool_result(tc["name"], result)

                if tc["name"] == "versions_checkout" and result.get("ok"):
                    needs_reopen = True

                self.history.append(Turn(
                    role="tool",
                    content=json.dumps(result, default=str),
                    tool_call_id=tc["call_id"],
                ))

            if needs_reopen:
                self.reopen()

        # Hit max turns without a final message
        return "(agent reached max turns without finishing)"
