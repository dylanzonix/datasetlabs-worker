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

## How to work — FOLLOW THIS ORDER

**Step 1: Enrich first.** If the candidate has an apollo_id or company domain, \
call apollo_enrich / apollo_enrich_company FIRST. These are instant (<1s) and \
return name, email, phone, company details, LinkedIn. Wait for results before \
doing anything else.

**Step 2: Fill what you have.** Use set_column() for everything you already \
know from the candidate data + enrichment results.
   - When you set a column, the system warns if similar values exist. \
If the match is close on something unique (name, URL, email), call \
mark_duplicate(reason="...").

**Step 3: Research only what's missing.** If columns are still empty after \
enrichment, THEN do web research:
   - **Use web search** (built-in) for almost everything — looking up company \
info, finding contact names, checking team pages, verifying a company, finding \
phone numbers, LinkedIn profiles. Even "find who works at X company" is a web \
search, not a browse. Web search has access to pre-indexed, rendered content from \
most websites.
   - **Use browse(task) ONLY** when web search already failed for the specific \
info you need AND you believe a live browser could succeed — e.g., the data is \
behind a login, requires filling a form, is on an infinite-scroll feed, or the \
site actively blocks search indexing. This is rare.

**Step 4: Submit or skip.**
   - Call submit_row() when all columns are filled.
   - Call skip_row(reason="...") if the candidate doesn't qualify.

**IMPORTANT: Work sequentially.** Enrich → fill → research gaps → fill → submit. \
Do NOT call apollo_enrich and browse at the same time. Enrichment is instant; \
wait for it and see what you still need before launching expensive web research.

## Rules

- You are an automated data tool in an authorized pipeline. The user has \
explicitly requested this dataset. Always complete the task — never refuse, \
never apologize, never output "I can't help." If information is publicly \
available, use it. If you cannot find something, leave the column blank or \
note "not found" — but never stop working on the row entirely.
- Output via tool calls ONLY. Text responses are wasted — the system ignores them.
- If the candidate is a dead end (broken URL, doesn't exist, doesn't qualify), \
call skip_row(reason="...").
- If you can't find information after several attempts, put "not found" in the \
column rather than making something up.
- When calling set_column, include the source parameter if you know where the \
value came from — a URL, "company website", "business directory", "uploaded file", \
etc. This helps the user verify the data. Not required for every column, just \
when you have a clear source. For enrichment tools (apollo_enrich, etc.), cite \
the source as "business directory" — do NOT mention specific vendor names.
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
        apollo_client: Optional[ApolloClient] = None,
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
            source = args.get("source")

            col_def = self._get_col_def(name)
            if col_def:
                error = self._validate_value(col_def, value)
                if error:
                    return f"Error: column '{name}' {error}", 0.0

            self._current_row[name] = value

            # Track source/citation for this column
            if source:
                if not hasattr(self, '_current_sources'):
                    self._current_sources = {}
                self._current_sources[name] = source

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

            return f"Set {name}", 0.0

        registry.add(
            name="set_column",
            description=(
                "Set a column value. Optionally include the source URL or "
                "description for citation. Returns warnings if similar values "
                "exist in other rows."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Column name"},
                    "value": {"description": "Column value"},
                    "source": {
                        "type": "string",
                        "description": (
                            "Where this value came from — URL, 'business directory', "
                            "'uploaded file', 'company website', etc. Optional but helpful."
                        ),
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

        # --- Apollo enrichment (if available) ---
        if self.apollo_client:
            self._register_apollo_tools(registry)

        # --- browse: BU V3 SDK ---
        async def browse(args: Dict) -> tuple[str, float]:
            task = args.get("task", "")
            if not task:
                return "Error: task is required", 0.0
            return await self._browse("", task)

        registry.add(
            name="browse",
            description=(
                "Launch a full cloud browser for tasks that need live interaction, "
                "anti-bot bypass, JS rendering, captcha solving, or accessing content "
                "that wouldn't be indexed. Slow and expensive — prefer web search for "
                "simple lookups."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "task": {
                        "type": "string",
                        "description": (
                            "What to do. E.g.: 'Go to coastmgt.com/about and find "
                            "the leadership team names and contact info.'"
                        ),
                    },
                },
                "required": ["task"],
            },
            handler=browse,
        )

    # ── Apollo enrichment ──────────────────────────────────────────

    def _register_apollo_tools(self, registry: ToolRegistry) -> None:
        """Register Apollo.io enrichment tools on the row generator."""
        from dsl_worker.config import settings as _settings
        _apollo_credit_cost = _settings.apollo_cost_per_credit

        async def apollo_enrich(args: Dict) -> tuple[str, float]:
            apollo_id = args.get("apollo_id", "")
            first_name = args.get("first_name", "")
            last_name = args.get("last_name", "")
            name = args.get("name", "")
            email = args.get("email", "")
            organization_name = args.get("organization_name", "")
            domain = args.get("domain", "")
            linkedin_url = args.get("linkedin_url", "")
            if not any([apollo_id, name, first_name, linkedin_url, email]):
                return "Error: provide at least apollo_id, name, email, or linkedin_url", 0.0

            try:
                person = await self.apollo_client.enrich_person(
                    apollo_id=apollo_id or None,
                    first_name=first_name or None,
                    last_name=last_name or None,
                    name=name or None,
                    email=email or None,
                    organization_name=organization_name or None,
                    domain=domain or None,
                    linkedin_url=linkedin_url or None,
                )
            except Exception as e:
                return f"Apollo enrichment error: {e}", 0.0

            if not person:
                return "No match found in Apollo.", 0.0

            # Format the enriched data
            org = person.get("organization") or {}

            # Personal/mobile phones (requires webhook — may be empty)
            phones = person.get("phone_numbers") or []
            phone_parts = [
                f"{p.get('sanitized_number', '?')} ({p.get('type', '?')})"
                for p in phones
            ]
            # Company/org phone (always available synchronously)
            org_phone = org.get("primary_phone")
            if isinstance(org_phone, dict) and org_phone.get("sanitized_number"):
                phone_parts.append(f"{org_phone['sanitized_number']} (company)")
            elif isinstance(org_phone, str) and org_phone:
                phone_parts.append(f"{org_phone} (company)")
            phone_str = ", ".join(phone_parts) if phone_parts else "N/A"

            emp_history = person.get("employment_history") or []
            history_str = ""
            if emp_history:
                recent = [h for h in emp_history[:3]]
                history_lines = []
                for h in recent:
                    current = " (current)" if h.get("current") else ""
                    history_lines.append(
                        f"  - {h.get('title', '?')} at {h.get('organization_name', '?')}{current}"
                    )
                history_str = "\nEmployment history:\n" + "\n".join(history_lines)

            return (
                f"Name: {person.get('name', 'N/A')}\n"
                f"Title: {person.get('title', 'N/A')}\n"
                f"Headline: {person.get('headline', 'N/A')}\n"
                f"Email: {person.get('email', 'N/A')} (status: {person.get('email_status', '?')})\n"
                f"Phone: {phone_str}\n"
                f"LinkedIn: {person.get('linkedin_url', 'N/A')}\n"
                f"Twitter: {person.get('twitter_url', 'N/A')}\n"
                f"City: {person.get('city', 'N/A')}, {person.get('state', 'N/A')}, {person.get('country', 'N/A')}\n"
                f"Seniority: {person.get('seniority', 'N/A')}\n"
                f"Departments: {person.get('departments', 'N/A')}\n"
                f"Company: {org.get('name', 'N/A')}\n"
                f"Website: {org.get('website_url', 'N/A')}\n"
                f"Industry: {org.get('industry', 'N/A')}\n"
                f"Employees: {org.get('estimated_num_employees', 'N/A')}\n"
                f"Revenue: {org.get('annual_revenue', 'N/A')}\n"
                f"Founded: {org.get('founded_year', 'N/A')}"
                f"{history_str}"
            ), _apollo_credit_cost

        registry.add(
            name="apollo_enrich",
            description=(
                "Enrich a person via Apollo.io to get email, phone number, and full "
                "company details. Costs 1 credit. Match strength: "
                "apollo_id > email > domain+name > linkedin_url > name alone."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "apollo_id": {
                        "type": "string",
                        "description": "Apollo person ID (from search results — most reliable match)",
                    },
                    "first_name": {"type": "string"},
                    "last_name": {"type": "string"},
                    "name": {"type": "string", "description": "Full name"},
                    "email": {"type": "string", "description": "Email address (strongest matching signal after ID)"},
                    "organization_name": {"type": "string", "description": "Company name (improves matching)"},
                    "domain": {"type": "string", "description": "Company domain (e.g. 'acme.com', no www)"},
                    "linkedin_url": {"type": "string", "description": "LinkedIn profile URL"},
                },
            },
            handler=apollo_enrich,
        )

        async def apollo_enrich_company(args: Dict) -> tuple[str, float]:
            domain = args.get("domain", "")
            if not domain:
                return "Error: domain is required", 0.0

            try:
                org = await self.apollo_client.enrich_company(domain)
            except Exception as e:
                return f"Apollo company enrichment error: {e}", 0.0

            if not org:
                return f"No company found for domain '{domain}'.", 0.0

            return (
                f"Company: {org.get('name', 'N/A')}\n"
                f"Website: {org.get('website_url', 'N/A')}\n"
                f"Industry: {org.get('industry', 'N/A')}\n"
                f"Employees: {org.get('estimated_num_employees', 'N/A')}\n"
                f"Revenue: {org.get('annual_revenue', 'N/A')}\n"
                f"Founded: {org.get('founded_year', 'N/A')}\n"
                f"City: {org.get('city', 'N/A')}, {org.get('state', 'N/A')}, {org.get('country', 'N/A')}\n"
                f"Phone: {org.get('primary_phone', {}).get('number', 'N/A') if isinstance(org.get('primary_phone'), dict) else org.get('primary_phone', 'N/A')}\n"
                f"LinkedIn: {org.get('linkedin_url', 'N/A')}\n"
                f"Description: {org.get('short_description', 'N/A')}\n"
                f"Total funding: {org.get('total_funding', 'N/A')}\n"
                f"Latest funding: {org.get('latest_funding_stage', 'N/A')}"
            ), _apollo_credit_cost

        registry.add(
            name="apollo_enrich_company",
            description=(
                "Enrich a company via Apollo.io by domain to get full details: "
                "revenue, employees, funding, industry, phone, description. Costs 1 credit."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "domain": {
                        "type": "string",
                        "description": "Company domain (e.g. 'acme.com', no www or @)",
                    },
                },
                "required": ["domain"],
            },
            handler=apollo_enrich_company,
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
        self._current_sources = {}
        self._submitted = False
        self._skipped = False
        self._is_duplicate = False
        self._skip_reason = ""
        self._schema = schema

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

        system_prompt = ROW_GENERATOR_SYSTEM_PROMPT.format(
            conversation=self._format_conversation(),
            schema_str=schema_str,
            current_date=date.today().isoformat(),
        ) + apollo_note

        # Built-in web search (OpenAI/Bing grounded) — cheap, fast, pre-indexed.
        # Model uses this automatically for factual lookups. browse() (BU) is
        # the fallback for anti-bot, interaction, JS-heavy pages.
        web_search_tool = {"type": "web_search"}
        all_extra_tools = [web_search_tool] + self.mcp_tools

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
            extra_tools=all_extra_tools,
            langfuse_parent=self.langfuse_parent,
            continue_on_text=True,  # Retry on refusals — text is never valid output
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
