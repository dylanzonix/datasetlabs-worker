"""
Row generator agent — generates a single dataset row from a candidate.

V10: Row generator receives:
- Raw user conversation (same as orchestrator sees)
- Schema
- The candidate (string or dict from harvester)
- Source context (what source the candidate came from)

No explicit "instructions" — the row generator reads the conversation and
understands what to do. This avoids the "telephone game" where orchestrator
instructions lose or distort the user's original intent.

set_column() checks ALL columns for similar values (token Jaccard) against
both submitted rows and in-flight rows from concurrent generators.
The LLM judges whether matches are true duplicates based on context.
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import jsonschema

from dsl_worker.agents.base import AgentConversation
from dsl_worker.agents.tools import ToolRegistry
from dsl_worker.billing.tracked_client import TrackedOpenAIClient
from dsl_worker.infra.bu_client import BUClient

logger = logging.getLogger(__name__)

MAX_GENERATION_TURNS = 30
READ_FILE_LIMIT = 30_000
SIMILARITY_THRESHOLD = 0.5
MAX_SIMILAR_MATCHES = 5

# ── Token Jaccard helpers ──────────────────────────────────────────

_SPLIT_RE = re.compile(r"[^a-z0-9]+")


def _tokenize(value: Any) -> Set[str]:
    """Split a value into a set of lowercase alpha-numeric tokens."""
    text = str(value).lower()
    return {t for t in _SPLIT_RE.split(text) if t}


def _token_jaccard(a: Set[str], b: Set[str]) -> float:
    """Jaccard similarity between two token sets. Returns 0.0-1.0."""
    if not a or not b:
        return 0.0
    intersection = len(a & b)
    union = len(a | b)
    return intersection / union if union else 0.0


# ── Dedup store — shared between concurrent row generators ─────────

class DedupStore:
    """Thread-safe store for row-level dedup checking.

    Only submitted (finalized) rows are used for dedup decisions.
    In-flight tracking exists for bookkeeping but is NOT used in find_similar().

    Dedup flow:
    1. set_column() checks submitted rows → warns LLM of similar values
    2. LLM judges and may skip_row() if it's clearly the same entity
    3. submit_row() does a final high-confidence check → rejects if a
       concurrent generator submitted the same row while we were working
    """

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        # row_id → {col_name: value}
        self._submitted: Dict[str, Dict[str, Any]] = {}
        self._in_flight: Dict[str, Dict[str, Any]] = {}
        # Pre-computed token sets: (row_id, col_name) → token set
        self._token_cache: Dict[Tuple[str, str], Set[str]] = {}

    async def register_in_flight(self, row_id: str, col_name: str, value: Any) -> None:
        """Register a column value from an in-flight row generator."""
        async with self._lock:
            if row_id not in self._in_flight:
                self._in_flight[row_id] = {}
            self._in_flight[row_id][col_name] = value
            self._token_cache[(row_id, col_name)] = _tokenize(value)

    async def promote_to_submitted(self, row_id: str, row: Dict[str, Any]) -> None:
        """Move an in-flight row to submitted (called on submit_row)."""
        async with self._lock:
            self._in_flight.pop(row_id, None)
            self._submitted[row_id] = row
            for col_name, value in row.items():
                self._token_cache[(row_id, col_name)] = _tokenize(value)

    async def remove_in_flight(self, row_id: str) -> None:
        """Remove an in-flight row (called on skip/error)."""
        async with self._lock:
            cols = self._in_flight.pop(row_id, {})
            for col_name in cols:
                self._token_cache.pop((row_id, col_name), None)

    async def find_similar(
        self,
        col_name: str,
        value: Any,
        exclude_row_id: str,
    ) -> List[Tuple[float, str, Dict[str, Any]]]:
        """Find rows with similar values in the given column.

        Returns list of (similarity, row_id, full_row) sorted by similarity desc.
        Only checks submitted rows. In-flight is not used for dedup.
        Excludes the requesting row's own ID.
        """
        query_tokens = _tokenize(value)
        if not query_tokens:
            return []

        matches: List[Tuple[float, str, Dict[str, Any]]] = []

        async with self._lock:
            all_rows = list(self._submitted.items())

        for rid, row in all_rows:
            if rid == exclude_row_id:
                continue
            existing_value = row.get(col_name)
            if existing_value is None:
                continue

            cached_key = (rid, col_name)
            existing_tokens = self._token_cache.get(cached_key)
            if existing_tokens is None:
                existing_tokens = _tokenize(existing_value)

            sim = _token_jaccard(query_tokens, existing_tokens)
            if sim >= SIMILARITY_THRESHOLD:
                matches.append((sim, rid, row))

        matches.sort(key=lambda x: -x[0])
        return matches[:MAX_SIMILAR_MATCHES]


@dataclass
class GeneratedRow:
    """Result of row generation."""
    success: bool
    row: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    cost_usd: float = 0.0
    skipped: bool = False
    skip_reason: str = ""


ROW_GENERATOR_SYSTEM_PROMPT = """\
# Row Generator

You are a row generator in a dataset generation pipeline. A harvester upstream \
collected this candidate — your job is to validate it, research any missing \
details, and produce a complete dataset row. You are the quality gate.

## What the user wants

<conversation>
{conversation}
</conversation>

## Schema

<schema>
{schema_str}
</schema>

Today's date: {current_date}

## How to work

1. Read the candidate and user conversation to understand what's needed.
2. Fill in what you already know from the candidate using set_column(). \
If the candidate already has all the data you need, just fill columns and submit — \
no browsing required.
   - When you set a column, the system will warn you if similar values exist. \
If the match is close on something that should be unique (name, URL, email), \
this is likely a duplicate. Call skip_row(reason="duplicate: ...").
   - Use judgment: many rows with country="USA" is normal, but 2 rows with \
the same email is suspicious.
3. Only browse if columns are missing from the candidate:
   - Use browse(url, task) to visit a page and extract specific information.
   - You can also search: browse(task="Search for Acme Corp founding year").
4. Fill remaining columns with set_column().
5. Call submit_row() when all columns are filled.
6. If the candidate doesn't qualify based on the user's criteria, \
call skip_row(reason="...") with a clear explanation.

## Rules

- Output via tool calls ONLY. Text responses are ignored.
- If the candidate is a dead end (broken URL, doesn't exist, doesn't qualify), \
call skip_row(reason="...").
- If you can't find information after 2-3 attempts, note it in the column \
rather than making something up.
"""


class RowGeneratorAgent:
    """
    Generates a single dataset row from a candidate.

    Usage:
        dedup = DedupStore()
        agent = RowGeneratorAgent(
            openai_client=tracked_client,
            model="gpt-5.2",
            workspace_dir=Path("/workspace"),
            chat_history=[...],
            dedup_store=dedup,
            ...
        )
        result = await agent.generate(
            instructions="You will be given a podcast...",
            candidate={"name": "XYZ Podcast", "url": "https://..."},
            schema=[{"name": "podcast_name", "type": "string"}, ...],
        )
    """

    def __init__(
        self,
        openai_client: TrackedOpenAIClient,
        model: str,
        workspace_dir: Path,
        chat_history: Optional[List[Dict[str, str]]] = None,
        dedup_store: Optional[DedupStore] = None,
        bu_client: Optional[BUClient] = None,
        sandbox: Optional[Any] = None,
        stop_checker: Optional[Callable[[], bool]] = None,
        stop_event: Optional[asyncio.Event] = None,
        blob_service_client: Optional[Any] = None,
        project_id: Optional[Any] = None,
        uploaded_file_urls: Optional[Dict[str, str]] = None,
        mcp_tools: Optional[List[Dict[str, Any]]] = None,
        on_cost: Optional[Callable] = None,
        langfuse_parent: Optional[Any] = None,
        # Legacy kwargs (ignored)
        brave_api_key: Optional[str] = None,
        on_browser_started: Optional[Callable] = None,
        on_browser_stopped: Optional[Callable] = None,
    ) -> None:
        self.openai_client = openai_client
        self.model = model
        self.workspace_dir = Path(workspace_dir)
        self.chat_history = chat_history or []
        self.dedup_store = dedup_store or DedupStore()
        self.bu_client = bu_client
        self.stop_checker = stop_checker
        self.stop_event = stop_event
        self.mcp_tools = mcp_tools or []
        self.on_cost = on_cost
        self.langfuse_parent = langfuse_parent
        self._row_id = str(id(self))

        # State — reset per generate() call
        self._current_row: Dict[str, Any] = {}
        self._submitted: bool = False
        self._skipped: bool = False
        self._skip_reason: str = ""
        self._schema: List[Dict] = []

        # Sandbox for code_exec only (minimal ResearchTools)
        self._sandbox_impl: Optional[Any] = None
        if sandbox:
            from dsl_worker.infra.research_tools import ResearchTools, ResearchScope
            self._sandbox_impl = ResearchTools(
                workspace_dir=workspace_dir,
                schema=[],
                brave_api_key=None,
                openai_client=openai_client,
                model=model,
                sandbox=sandbox,
                stop_checker=stop_checker,
                blob_service_client=blob_service_client,
                project_id=project_id,
                uploaded_file_urls=uploaded_file_urls,
            )
            self._sandbox_impl.set_scope(ResearchScope(id="row_gen", description="", quota=0))

        self._registry = ToolRegistry()
        self._register_tools(self._registry)

    def _get_col_def(self, name: str) -> Optional[Dict]:
        for col in self._schema:
            if col.get("name") == name:
                return col
        return None

    def _validate_value(self, col_def: Dict, value: Any) -> Optional[str]:
        col_type = col_def.get("type", "string")
        if col_type == "string":
            if not isinstance(value, str):
                return f"expects string, got {type(value).__name__}"
        elif col_type == "int":
            if not isinstance(value, int) or isinstance(value, bool):
                return f"expects int, got {type(value).__name__}"
        elif col_type == "float":
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                return f"expects float, got {type(value).__name__}"
        elif col_type == "bool":
            if not isinstance(value, bool):
                return f"expects bool, got {type(value).__name__}"
        elif col_type == "enum":
            allowed = col_def.get("enum_values", [])
            if value not in allowed:
                return f"expects one of {allowed}, got {value!r}"
        elif col_type == "json":
            schema = col_def.get("json_schema")
            if schema:
                try:
                    jsonschema.validate(value, schema)
                except jsonschema.ValidationError as e:
                    return f"json_schema validation failed: {e.message}"
        return None

    def _register_tools(self, registry: ToolRegistry) -> None:

        async def set_column(args: Dict) -> tuple[str, float]:
            name = args.get("name", "")
            value = args.get("value")

            col_def = self._get_col_def(name)
            if col_def:
                error = self._validate_value(col_def, value)
                if error:
                    return f"Error: column '{name}' {error}", 0.0

            self._current_row[name] = value

            # Register in dedup store so concurrent generators can see it
            await self.dedup_store.register_in_flight(self._row_id, name, value)

            # Check ALL columns for similar values (token Jaccard)
            # Only matches against submitted (finalized) rows — no in-flight.
            similar = await self.dedup_store.find_similar(name, value, self._row_id)
            if similar:
                lines = []
                for sim_score, _rid, row in similar:
                    row_preview = {k: v for k, v in row.items() if v is not None}
                    for k, v in row_preview.items():
                        s = str(v)
                        if len(s) > 120:
                            row_preview[k] = s[:120] + "..."
                    lines.append(
                        f"  - similarity {sim_score:.0%}: {json.dumps(row_preview, ensure_ascii=False)}"
                    )
                return (
                    f"Set {name} = {value!r}\n\n"
                    f"⚠ {len(similar)} existing row(s) in the dataset have similar '{name}' values:\n"
                    + "\n".join(lines) + "\n\n"
                    f"If any of these is the same entity, call skip_row(reason=\"duplicate: ...\")."
                ), 0.0

            return f"Set {name}", 0.0

        registry.add(
            name="set_column",
            description=(
                "Set a column value. Returns warnings if similar values "
                "exist in other rows — check them for duplicates."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Column name"},
                    "value": {"description": "Column value"},
                },
                "required": ["name", "value"],
            },
            handler=set_column,
        )

        async def append_to_column(args: Dict) -> tuple[str, float]:
            name = args.get("name", "")
            value = args.get("value")
            col_def = self._get_col_def(name)
            if not col_def:
                return f"Error: unknown column '{name}'", 0.0
            col_type = col_def.get("type", "string")
            if col_type == "json":
                if name not in self._current_row:
                    self._current_row[name] = []
                if not isinstance(self._current_row[name], list):
                    return f"Error: column '{name}' is not a list", 0.0
                self._current_row[name].append(value)
                return f"Appended to {name} ({len(self._current_row[name])} items)", 0.0
            elif col_type == "string":
                if name not in self._current_row:
                    self._current_row[name] = ""
                if not isinstance(value, str):
                    return "Error: append value must be string for string column", 0.0
                sep = "\n" if self._current_row[name] else ""
                self._current_row[name] += sep + value
                return f"Appended to {name}", 0.0
            else:
                return f"Error: append not supported for type '{col_type}'", 0.0

        registry.add(
            name="append_to_column",
            description="Append to a column. For json columns, appends as list element. For strings, concatenates with newline.",
            parameters={
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "value": {"description": "Value to append"},
                },
                "required": ["name", "value"],
            },
            handler=append_to_column,
        )

        async def clear_column(args: Dict) -> tuple[str, float]:
            name = args.get("name", "")
            self._current_row.pop(name, None)
            return f"Cleared {name}", 0.0

        registry.add(
            name="clear_column",
            description="Clear a column value to start over.",
            parameters={
                "type": "object",
                "properties": {"name": {"type": "string"}},
                "required": ["name"],
            },
            handler=clear_column,
        )

        async def submit_row(args: Dict) -> tuple[str, float]:
            missing = [
                col.get("name") for col in self._schema
                if col.get("name") and col.get("name") not in self._current_row
            ]
            if missing:
                return f"Error: missing columns {missing}. Set them before submitting.", 0.0

            for col in self._schema:
                col_name = col.get("name", "")
                if col_name in self._current_row:
                    error = self._validate_value(col, self._current_row[col_name])
                    if error:
                        return f"Error: column '{col_name}' {error}.", 0.0

            # Final dedup check — holistic comparison against submitted rows.
            # Catches the race condition where a concurrent generator submitted
            # the same entity while we were working. Combines ALL column values
            # into one blob and checks full-row similarity at a very high bar (95%).
            my_blob = " ".join(str(v) for v in self._current_row.values() if v)
            my_tokens = _tokenize(my_blob)
            if my_tokens:
                async with self.dedup_store._lock:
                    for rid, existing_row in self.dedup_store._submitted.items():
                        if rid == self._row_id:
                            continue
                        existing_blob = " ".join(str(v) for v in existing_row.values() if v)
                        existing_tokens = _tokenize(existing_blob)
                        sim = _token_jaccard(my_tokens, existing_tokens)
                        if sim >= 0.95:
                            self._skipped = True
                            self._skip_reason = (
                                f"duplicate detected at submit: {sim:.0%} overall "
                                f"similarity with existing row"
                            )
                            return (
                                f"Row rejected — a nearly identical row was submitted "
                                f"by another generator ({sim:.0%} overall similarity). "
                                f"Skipping as duplicate."
                            ), 0.0

            self._submitted = True
            return "Row submitted.", 0.0

        registry.add(
            name="submit_row",
            description="Submit the completed row. Call when all columns are filled.",
            parameters={"type": "object", "properties": {}},
            handler=submit_row,
        )

        async def skip_row(args: Dict) -> tuple[str, float]:
            reason = args.get("reason", "")
            self._skipped = True
            self._skip_reason = reason
            return f"Row skipped: {reason}", 0.0

        registry.add(
            name="skip_row",
            description=(
                "Skip this candidate. Use when it's a duplicate, dead end, "
                "broken URL, or doesn't qualify."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "reason": {"type": "string", "description": "Why this row is being skipped"},
                },
                "required": ["reason"],
            },
            handler=skip_row,
        )

        async def rng(args: Dict) -> tuple[str, float]:
            options = args.get("options", [])
            weights = args.get("weights")
            if not options:
                return "Error: no options provided", 0.0
            if weights and len(weights) == len(options):
                selected = random.choices(options, weights=weights, k=1)[0]
            else:
                selected = random.choice(options)
            return f"Selected: {selected}", 0.0

        registry.add(
            name="rng",
            description="Pick a random option for controlled randomization.",
            parameters={
                "type": "object",
                "properties": {
                    "options": {"type": "array", "items": {"type": "string"}},
                    "weights": {"type": "array", "items": {"type": "number"}},
                },
                "required": ["options"],
            },
            handler=rng,
        )

        async def read_file(args: Dict) -> tuple[str, float]:
            path_str = args.get("path", "")
            try:
                path = Path(path_str)
                if not path.is_absolute():
                    candidate = self.workspace_dir / path
                    if not candidate.exists():
                        candidate = self.workspace_dir / "sources" / path
                    path = candidate
                if not path.exists():
                    return f"File not found: {path_str}", 0.0
                content = path.read_text(encoding="utf-8")
                if len(content) > READ_FILE_LIMIT:
                    content = content[:READ_FILE_LIMIT] + f"\n\n[Truncated at {READ_FILE_LIMIT} chars]"
                return content, 0.0
            except Exception as e:
                return f"Error reading file: {e}", 0.0

        registry.add(
            name="read_file",
            description="Read a file from the workspace.",
            parameters={
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
            handler=read_file,
        )

        # code_exec from sandbox (if available)
        if self._sandbox_impl:
            self._sandbox_impl.register_on(
                registry,
                exclude=[
                    "brave_search", "open", "find", "click",
                    "interact", "shell_exec",
                ],
                include_builtins=False,
            )

        # --- browse: BU V3 SDK ---
        async def browse(args: Dict) -> tuple[str, float]:
            url = args.get("url", "")
            task = args.get("task", "")
            if not url and not task:
                return "Error: url or task is required", 0.0
            if url and not task:
                task = "Extract relevant information from this page."
            if not url:
                # Task-only: web search
                return await self._browse("", task)
            return await self._browse(url, task)

        registry.add(
            name="browse",
            description=(
                "Browse the web — navigate to a URL, search, or extract information. "
                "Provide a URL to visit a specific page, or just a task to search the web. "
                "Returns extracted text."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "URL to visit (optional if task is a search query)",
                    },
                    "task": {
                        "type": "string",
                        "description": (
                            "What to do. E.g.: 'Find the CEO name and founding year' "
                            "or 'Search for Acme Corp leadership team'."
                        ),
                    },
                },
            },
            handler=browse,
        )

    # ── BU V3 SDK web access ────────────────────────────────────────

    async def _browse(self, url: str, task: str) -> Tuple[str, float]:
        """Navigate to a URL and extract information via BU V3 SDK."""
        if not self.bu_client:
            return "Error: BU client not configured", 0.0

        bu_task = f"Navigate to: {url}\n\n{task}" if url else task

        try:
            text, bu_cost, _sid = await self.bu_client.research(bu_task)
            if len(text) > 4000:
                text = text[:4000] + "\n\n[Truncated to 4K chars]"
            return text, bu_cost
        except Exception as e:
            logger.warning(f"[row_gen] browse error: {e}")
            return f"Browse error: {e}", 0.0

    def _format_conversation(self) -> str:
        if not self.chat_history:
            return "(no conversation history)"
        parts = []
        for msg in self.chat_history:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            parts.append(f"**{role}**: {content}")
        return "\n\n".join(parts)

    def _format_candidate(self, candidate: Any) -> str:
        if candidate is None:
            return "(no candidate)"
        if isinstance(candidate, dict):
            return json.dumps(candidate, indent=2, ensure_ascii=False)
        return str(candidate)

    async def generate(
        self,
        candidate: Any,
        schema: Optional[List[Dict]] = None,
        source_context: str = "",
        source_url: Optional[str] = None,
        # Backward compat: old callers may pass instructions kwarg
        instructions: Optional[str] = None,
        source_content: Optional[str] = None,
    ) -> GeneratedRow:
        """Generate a single row from a candidate."""
        if schema is None:
            schema = []

        # Backward compat: use instructions as source_context fallback
        if not source_context and instructions:
            source_context = instructions

        self._current_row = {}
        self._submitted = False
        self._skipped = False
        self._skip_reason = ""
        self._schema = schema

        from datetime import date
        system_prompt = ROW_GENERATOR_SYSTEM_PROMPT.format(
            conversation=self._format_conversation(),
            schema_str=json.dumps(schema, indent=2),
            current_date=date.today().isoformat(),
        )

        conversation = AgentConversation(
            openai_client=self.openai_client,
            model=self.model,
            system_prompt=system_prompt,
            tools=self._registry,
            stop_checker=self.stop_checker,
            stop_event=self.stop_event,
            max_turns=MAX_GENERATION_TURNS,
            reasoning={"effort": "low", "summary": "auto"},
            label="row_generator",
            on_cost=self.on_cost,
            extra_tools=self.mcp_tools,
            langfuse_parent=self.langfuse_parent,
        )

        # Candidate goes in the user message (not system prompt) so the system
        # prompt prefix is identical across all row generators → prompt caching.
        candidate_text = self._format_candidate(candidate)
        extra_sections = ""
        if source_context:
            extra_sections += f"\n\n## Source\n\n{source_context}"
        if source_url:
            extra_sections += f"\n\n## Source URL\n\n{source_url}"
        user_msg = (
            f"## Candidate\n\n{candidate_text}"
            f"{extra_sections}\n\n"
            f"Process this candidate and generate a dataset row."
        )

        await conversation.send(
            user_msg,
            exit_condition=lambda: self._submitted or self._skipped,
        )

        cost = conversation.total_cost

        if self._skipped:
            await self.dedup_store.remove_in_flight(self._row_id)
            return GeneratedRow(
                success=False, skipped=True, skip_reason=self._skip_reason, cost_usd=cost
            )

        if self._submitted:
            await self.dedup_store.promote_to_submitted(self._row_id, self._current_row)
            return GeneratedRow(success=True, row=self._current_row, cost_usd=cost)

        await self.dedup_store.remove_in_flight(self._row_id)
        return GeneratedRow(
            success=False,
            error=f"Row generation did not complete ({conversation.total_turns} turns)",
            row=self._current_row if self._current_row else None,
            cost_usd=cost,
        )

    @property
    def cost_usd(self) -> float:
        return 0.0

    async def cleanup(self) -> None:
        if self._sandbox_impl:
            try:
                await self._sandbox_impl.cleanup()
            except Exception as e:
                logger.warning(f"RowGeneratorAgent cleanup error: {e}")
