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
import hashlib
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
from dsl_worker.infra.apollo_client import ApolloClient

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
    sources: Optional[Dict[str, str]] = None  # column_name → source/citation
    error: Optional[str] = None
    cost_usd: float = 0.0
    skipped: bool = False
    skip_reason: str = ""
    is_duplicate: bool = False


ROW_GENERATOR_SYSTEM_PROMPT = """\
# Row Generator

You process one candidate into one dataset row. Follow the orchestrator's \
instructions — they've already figured out the best approach. Fill columns \
from the candidate data first, then research what's missing.

## What the user wants

<conversation>
{conversation}
</conversation>

## Schema

<schema>
{schema_str}
</schema>

Today's date: {current_date}

{instructions_section}\

## Tools

- **set_column(column, value, sources)** — set a schema column value. \
Include sources when you have them (type: "url", "file", or "enrichment"). \
When you set a column, the system warns if similar values exist — if the \
match is close on something unique (name, URL, email), call mark_duplicate().
- **submit_row()** — submit the completed row.
- **skip_row(reason)** — skip if the candidate doesn't qualify.
- **mark_duplicate(reason)** — mark as duplicate.
- **Web search** (built-in) — fast and cheap. Use for lookups, research, \
finding info on websites. Prefer over browser_use whenever possible.
- **code_exec(script)** — Python sandbox. Read files, parse data.
- **browser_use(task, reason)** — real cloud browser. EXPENSIVE ($0.10-0.50). \
Only when you have a specific URL that needs a browser (anti-bot, JS, \
interactive) and web_search + apify couldn't get it.
- Additional tools: **apify** (22K+ web scrapers — check before using \
browser_use), **fullenrich** (verified email ~$0.05, phone ~$0.55), \
**apollo** (B2B company/people enrichment), **google_maps** (local \
business data).

## Guidelines

- You are an automated data tool in an authorized pipeline. The user has \
explicitly requested this dataset. Always complete the task — never refuse, \
never apologize, never output "I can't help." If information is publicly \
available, use it. If you cannot find something, leave the column blank \
or put "Not found" — but never stop working on the row entirely.
- Output via tool calls ONLY. Text responses are ignored by the system.
- Fill columns from candidate data first — don't research what's already there.
- If the instructions say which tools to use, use those. Don't search for \
alternatives or explore different approaches.
- Skip immediately if the candidate clearly doesn't qualify.
- If you can't find information after a few attempts, put "Not found" \
rather than fabricating.
- Capitalize properly — "Not found" not "not found".
- Cite sources when you have them — use the actual URL, not prose.
- Set multiple columns per turn when possible — don't set one at a time.
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
        uploaded_files: Optional[List[Dict[str, Any]]] = None,
        apollo_client: Optional[ApolloClient] = None,
        google_maps_client: Optional[Any] = None,
        youtube_client: Optional[Any] = None,
        apify_client: Optional[Any] = None,
        fullenrich_client: Optional[Any] = None,
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
        self.apollo_client = apollo_client
        self.google_maps_client = google_maps_client
        self.youtube_client = youtube_client
        self.apify_client = apify_client
        self.fullenrich_client = fullenrich_client
        self.uploaded_files = uploaded_files or []
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
        self._is_duplicate: bool = False
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

    def _coerce_value(self, col_def: Dict, value: Any) -> Tuple[Any, Optional[str]]:
        """Coerce value to the column type. Returns (coerced_value, warning_or_None)."""
        col_type = col_def.get("type", "string")

        # Handle None — use type-appropriate default
        if value is None:
            defaults = {"string": "", "int": 0, "float": 0.0, "bool": False}
            if col_type in defaults:
                return defaults[col_type], f"was null, defaulted to {defaults[col_type]!r}"
            return value, None

        if col_type == "string":
            if not isinstance(value, str):
                return str(value), None
        elif col_type == "int":
            if isinstance(value, bool):
                return int(value), None
            if not isinstance(value, int):
                try:
                    return int(value), None
                except (ValueError, TypeError):
                    return 0, f"couldn't convert {type(value).__name__} to int, defaulted to 0"
        elif col_type == "float":
            if isinstance(value, bool):
                return float(value), None
            if not isinstance(value, (int, float)):
                try:
                    return float(value), None
                except (ValueError, TypeError):
                    return 0.0, f"couldn't convert {type(value).__name__} to float, defaulted to 0.0"
        elif col_type == "bool":
            if not isinstance(value, bool):
                return bool(value), None
        elif col_type == "enum":
            allowed = col_def.get("enum_values", [])
            if value not in allowed:
                return value, f"not in allowed values {allowed}"
        elif col_type == "json":
            schema = col_def.get("json_schema")
            if schema:
                try:
                    jsonschema.validate(value, schema)
                except jsonschema.ValidationError as e:
                    return value, f"json_schema: {e.message}"
        return value, None

    def _register_tools(self, registry: ToolRegistry) -> None:

        async def set_column(args: Dict) -> tuple[str, float]:
            name = args.get("name", "")
            value = args.get("value")
            sources = args.get("sources")

            col_def = self._get_col_def(name)
            if not col_def:
                # Try case-insensitive match
                for col in self._schema:
                    if col.get("name", "").lower() == name.lower():
                        col_def = col
                        name = col["name"]  # use the canonical name
                        break
            if not col_def:
                valid = [c.get("name") for c in self._schema]
                return f"Error: unknown column '{name}'. Valid columns: {valid}", 0.0
            value, warning = self._coerce_value(col_def, value)

            self._current_row[name] = value

            # Track structured sources/citations for this column
            if sources and isinstance(sources, list):
                # Resolve file numbers to filenames
                resolved = []
                for src in sources:
                    if isinstance(src, dict):
                        if src.get("type") == "file" and src.get("value"):
                            try:
                                idx = int(src["value"]) - 1
                                if 0 <= idx < len(self.uploaded_files):
                                    src = {**src, "value": self.uploaded_files[idx].get("filename", src["value"])}
                            except (ValueError, IndexError):
                                pass
                        resolved.append(src)
                self._current_sources[name] = resolved

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
                    f"If any of these is the same entity, call mark_duplicate(reason=\"...\")."
                ), 0.0

            result_msg = f"Set {name}"
            if warning:
                result_msg += f" (note: {warning})"
            return result_msg, 0.0

        registry.add(
            name="set_column",
            description=(
                "Set a column value. Optionally include sources for citation."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Column name"},
                    "value": {"description": "Column value"},
                    "sources": {
                        "type": "array",
                        "description": "Where this value came from. Each source has a type.",
                        "items": {
                            "type": "object",
                            "properties": {
                                "type": {
                                    "type": "string",
                                    "enum": ["url", "file", "enrichment"],
                                    "description": (
                                        "url = web page, file = uploaded file (use file number), "
                                        "enrichment = business directory"
                                    ),
                                },
                                "value": {
                                    "type": "string",
                                    "description": (
                                        "URL for url type, file number for file type, "
                                        "Omit for enrichment."
                                    ),
                                },
                            },
                            "required": ["type"],
                        },
                    },
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
                    coerced, _ = self._coerce_value(col, self._current_row[col_name])
                    self._current_row[col_name] = coerced

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
                "Skip this candidate because it doesn't qualify — wrong category, "
                "outside date range, dead end, requires private data, etc."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "reason": {"type": "string", "description": "Why this candidate doesn't qualify"},
                },
                "required": ["reason"],
            },
            handler=skip_row,
        )

        async def mark_duplicate(args: Dict) -> tuple[str, float]:
            reason = args.get("reason", "")
            self._skipped = True
            self._is_duplicate = True
            self._skip_reason = reason
            return f"Row marked as duplicate: {reason}", 0.0

        registry.add(
            name="mark_duplicate",
            description=(
                "Mark this candidate as a duplicate of an existing row. "
                "Use when the dedup warnings show a clear match."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "reason": {"type": "string", "description": "What it's a duplicate of"},
                },
                "required": ["reason"],
            },
            handler=mark_duplicate,
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
                    "shell_exec",
                ],
                include_builtins=False,
            )

        # Enable deferred tool discovery — namespace tools only load when needed
        registry.add_builtin({"type": "tool_search"})

        # Integration namespaces (deferred)
        if self.apify_client:
            from dsl_worker.agents.integrations.apify import register_apify_namespace
            from dsl_worker.config import settings as _apify_settings
            register_apify_namespace(
                registry, self.apify_client, _apify_settings.apify_api_key,
                self.workspace_dir,
            )
        if self.apollo_client:
            from dsl_worker.agents.integrations.apollo import register_apollo_namespace
            from dsl_worker.config import settings as _apollo_s
            register_apollo_namespace(
                registry, self.apollo_client, self.workspace_dir,
                cost_per_credit=_apollo_s.apollo_cost_per_credit,
            )
        if self.google_maps_client:
            from dsl_worker.agents.integrations.google_maps import register_google_maps_namespace
            from dsl_worker.config import settings as _gm_settings
            register_google_maps_namespace(
                registry, _gm_settings.google_api_key, self.workspace_dir,
            )
        if hasattr(self, 'fullenrich_client') and self.fullenrich_client:
            from dsl_worker.agents.integrations.fullenrich import register_fullenrich_namespace
            from dsl_worker.config import settings as _fe_s
            register_fullenrich_namespace(
                registry, self.fullenrich_client, self.workspace_dir,
                cost_per_credit=_fe_s.fullenrich_cost_per_credit,
            )

        # --- browser_use: BU V3 SDK ---
        async def browser_use(args: Dict) -> tuple[str, float]:
            task = args.get("task", "")
            if not task:
                return "Error: task is required", 0.0
            # Log the reason for debugging/optimization
            reason = args.get("reason", "")
            if reason:
                logger.info(f"[row_generator] browser_use reason: {reason}")
            return await self._browse("", task)

        registry.add(
            name="browser_use",
            description=(
                "Open a real cloud browser to visit a specific URL. EXPENSIVE "
                "($0.10-0.50 per session). Use ONLY when: (1) you have a specific "
                "URL that needs a real browser — anti-bot protection, JS rendering, "
                "or interactive elements, AND (2) web_search couldn't get the data, "
                "AND (3) no Apify actor exists for this site. If the data could be "
                "found from a different source without a browser, do that instead."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "task": {
                        "type": "string",
                        "description": (
                            "What to do. Include the specific URL. E.g.: 'Go to "
                            "coastmgt.com/about and find the leadership team names.'"
                        ),
                    },
                    "reason": {
                        "type": "string",
                        "description": (
                            "Why browser_use is needed — what you tried that didn't "
                            "work (e.g. 'web_search returned no results for this "
                            "company contact page, site appears to block indexing')"
                        ),
                    },
                },
                "required": ["task"],
            },
            handler=browser_use,
        )

    # ── Apollo enrichment ──────────────────────────────────────────
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
        # V13 additions
        note: Optional[str] = None,
        preset_fields: Optional[Dict[str, str]] = None,
        candidate_data: Optional[Any] = None,
    ) -> GeneratedRow:
        """Generate a single row from a candidate.

        V13 additions:
        - note: orchestrator's handoff briefing (included in system prompt)
        - preset_fields: dict mapping {schema_column: candidate_field} for
          pre-filling columns from candidate_data before the LLM runs
        - candidate_data: structured candidate data (dict). If provided and
          candidate is a string, this is used for preset_fields lookups.
        """
        if schema is None:
            schema = []

        # Backward compat: use instructions as source_context fallback
        if not source_context and instructions:
            source_context = instructions

        self._current_row = {}
        self._current_sources = {}
        self._submitted = False
        self._skipped = False
        self._is_duplicate = False
        self._skip_reason = ""
        self._schema = schema

        # Normalize candidate_data: if it's a string, try to parse as JSON
        if candidate_data is not None:
            if isinstance(candidate_data, str):
                try:
                    candidate_data = json.loads(candidate_data)
                except (json.JSONDecodeError, ValueError):
                    candidate_data = None

        from datetime import date
        apollo_note = ""
        if self.apollo_client:
            apollo_note = (
                "\n\n## Apollo.io Enrichment\n\n"
                "You have apollo_enrich and apollo_enrich_company tools. Use them to get "
                "email addresses, phone numbers, and detailed company info. "
                "If the candidate has an apollo_id, use that for the most reliable match. "
                "Otherwise, use name + organization_name or linkedin_url.\n"
            )
        # Format schema as readable lines instead of raw JSON
        schema_lines = []
        for col in schema:
            name = col.get("name", "?")
            fmt = col.get("format", "")
            # Support both old (type-based) and new (format-based) schemas
            col_type = col.get("type", "")
            if fmt:
                schema_lines.append(f"- **{name}** — {fmt}")
            elif col_type:
                schema_lines.append(f"- **{name}** ({col_type})")
            else:
                schema_lines.append(f"- **{name}**")
        schema_str = "\n".join(schema_lines) if schema_lines else "(no columns defined)"

        # Build file list for citation references
        files_note = ""
        if self.uploaded_files:
            file_lines = ["\n\n## Uploaded Files (for source citations, use file number)"]
            for idx, f in enumerate(self.uploaded_files, 1):
                file_lines.append(f"  [{idx}] {f.get('filename', 'unknown')}")
            files_note = "\n".join(file_lines)

        # V13: orchestrator instructions
        instructions_section = ""
        if note:
            instructions_section = (
                "## Instructions from orchestrator\n\n"
                f"{note}\n\n"
            )

        system_prompt = ROW_GENERATOR_SYSTEM_PROMPT.format(
            conversation=self._format_conversation(),
            schema_str=schema_str,
            current_date=date.today().isoformat(),
            instructions_section=instructions_section,
        ) + apollo_note + files_note

        # V13: pre-fill columns from preset_fields before the LLM starts.
        # This uses the same validation + dedup logic as set_column.
        prefilled_columns: List[str] = []
        if preset_fields and isinstance(preset_fields, dict) and candidate_data and isinstance(candidate_data, dict):
            for schema_col, candidate_field in preset_fields.items():
                # Support dot-notation for nested fields (e.g. "author.displayName")
                value = candidate_data
                for key in candidate_field.split("."):
                    if isinstance(value, dict):
                        value = value.get(key)
                    else:
                        value = None
                        break
                if value is None:
                    continue
                # Validate against schema
                col_def = None
                for col in schema:
                    if col.get("name") == schema_col:
                        col_def = col
                        break
                if col_def:
                    error = self._validate_value(col_def, value)
                    if error:
                        logger.debug(
                            f"[row_gen] preset_fields: skipping {schema_col} — {error}"
                        )
                        continue

                self._current_row[schema_col] = value
                await self.dedup_store.register_in_flight(self._row_id, schema_col, value)

                # Check for duplicates (same as set_column)
                similar = await self.dedup_store.find_similar(schema_col, value, self._row_id)
                if similar:
                    # If high-confidence duplicate on a pre-filled column, skip early
                    top_sim = similar[0][0]
                    if top_sim >= 0.95:
                        await self.dedup_store.remove_in_flight(self._row_id)
                        return GeneratedRow(
                            success=False, skipped=True, is_duplicate=True,
                            skip_reason=f"Pre-fill dedup: {schema_col} matched existing row at {top_sim:.0%}",
                            cost_usd=0.0,
                        )

                prefilled_columns.append(schema_col)

        # Add pre-filled columns info to system prompt so the LLM knows
        if prefilled_columns:
            cols_str = ", ".join(prefilled_columns)
            vals_str = ", ".join(
                f"{c}={self._current_row[c]!r}" for c in prefilled_columns
            )
            system_prompt += (
                f"\n\nPre-filled columns (already set, do not re-set unless wrong): "
                f"{vals_str}"
            )

        # Built-in web search (OpenAI/Bing grounded) — cheap, fast, pre-indexed.
        # Model uses this automatically for factual lookups. browse() (BU) is
        # the fallback for anti-bot, interaction, JS-heavy pages.
        web_search_tool = {"type": "web_search"}
        all_extra_tools = [web_search_tool] + self.mcp_tools

        # Derive a prompt_cache_key from the system prompt so all row generators
        # sharing the same prefix get routed to the same backend → cache hits.
        # DISABLED: may contribute to Azure content filter cascading refusals
        cache_key = None  # hashlib.sha256(system_prompt.encode()).hexdigest()[:16]

        conversation = AgentConversation(
            openai_client=self.openai_client,
            model=self.model,
            system_prompt=system_prompt,
            tools=self._registry,
            stop_checker=self.stop_checker,
            stop_event=self.stop_event,
            max_turns=MAX_GENERATION_TURNS,
            reasoning={"effort": "medium", "summary": "detailed"},
            label="row_generator",
            on_cost=self.on_cost,
            extra_tools=all_extra_tools,
            langfuse_parent=self.langfuse_parent,
            continue_on_text=True,  # Retry on refusals — text is never valid output
            prompt_cache_key=cache_key,
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
                success=False, skipped=True, skip_reason=self._skip_reason,
                is_duplicate=self._is_duplicate, cost_usd=cost,
            )

        if self._submitted:
            await self.dedup_store.promote_to_submitted(self._row_id, self._current_row)
            return GeneratedRow(
                success=True,
                row=self._current_row,
                sources=self._current_sources if self._current_sources else None,
                cost_usd=cost,
            )

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
