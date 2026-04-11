"""
Orchestrator agent — V13 architecture.

Key changes from V12:
- No harvesters. Orchestrator directly controls candidate generation.
- Blocking tool calls. No timer-based check-in loop.
- File-based candidate flow. All tools write JSONL to /workspace/candidates/.
- submit_candidates / continue_processing for row generation with feedback.
- web_harvest subagent for multi-page web investigation.
- bu_extract for BU cloud sessions with file download.
- finish() only for genuinely impossible tasks.
- Target reached / credits exhausted handled by the system, not the LLM.

The orchestrator's run() is simple: build prompt, create conversation,
call conversation.run(), handle result. The LLM loop in base.py does
the rest — tool calls block and return results, the LLM decides what
to do next between each call.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import time
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple

from dsl_worker.agents.base import AgentConversation, AgentResult
from dsl_worker.agents.tools import ToolRegistry
from dsl_worker.billing.tracked_client import TrackedOpenAIClient
from dsl_worker.infra.candidate_pool import Candidate

logger = logging.getLogger(__name__)


# ── System Prompt ─────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
# Dataset Builder — {num_samples} rows

## Objective

Produce the dataset at the lowest cost per row. Every tool call costs \
money — yours and every row generator you spawn. Optimize ruthlessly.

## How it works

You have two modes:

**Workshop mode** — Process candidates yourself using process_candidate, \
set_column, submit_row, skip_row, and your research tools. This is how \
you learn what works: which fields are in the data, which need research, \
what gets skipped, what each row costs. Start here.

**Delegate mode** — Hand off remaining candidates to row generators via \
submit_candidates. They see your full conversation history including \
your workshop, so they know what works. Only delegate once you've \
figured out the optimal strategy.

## Tools

{integration_tools_section}\
**process_candidate(file, index)** — Load a candidate from a JSONL \
file to process yourself. Then use set_column/submit_row/skip_row.

**set_column(name, value)** — Set a column value on the loaded candidate.

**submit_row()** — Submit the completed row.

**skip_row(reason)** — Skip the current candidate.

**submit_candidates(file, instructions, checkin_after)** — Delegate \
remaining candidates to row generators. Include a brief note about \
what you learned: what works, what to skip, target cost per row.

**continue_processing(checkin_after)** — Resume delegated processing.

**code_exec(script)** — Python sandbox. /workspace/uploads/ (read-only), \
/workspace/candidates/ (your output). pandas, json, csv available. \
Use for filtering candidates, extracting fields, data manipulation.

**web_harvest(query, candidate_description)** — Spawns an agent that \
googles, visits pages, extracts candidates to a file. ~$0.10-0.20/call.

**browser_use(task)** — Cloud browser. $0.10-0.50/call. Last resort — \
only when a specific URL needs JS rendering or anti-bot bypass AND no \
Apify actor exists for that site.

**reprocess_rows(instructions)** — Modify existing rows in place (add \
column, filter, update). Preferred — salvage rows when possible.

**clear_rows(reason)** — Delete all rows and start over. Last resort \
only — when rows are genuinely unsalvageable.

**finish(reason)** — Only for genuinely impossible tasks.

### Source hierarchy
1. Uploaded files — free
2. Integrations (Apify, FullEnrich, Apollo, Google Maps) — cheap, structured
3. web_search — cheap, good for per-row lookups
4. web_harvest — ~$0.15/call, only when no integration exists
5. browser_use — $0.10-0.50/call, absolute last resort

Apify has scrapers for most sites. Always check apify_search first.

### Cost reference
- Apify actors: <$0.01/result
- FullEnrich emails: ~$0.05 each
- FullEnrich phones: ~$0.55 each (try web_search first)
- browser_use: $0.10-0.50/session

## How to work

### 1. Find candidates

Pick the most obvious source. Don't overthink it — for most projects \
the source is clear. A quick web_search to understand the landscape is \
fine. Keep total harvesting cost under ~$1.

For cheap sources (Apify <$0.01/result), harvest generously. For \
expensive sources, start small. Stick with what works — if an Apify \
actor is producing results, keep using it with adjusted parameters.

### 2. Workshop — process candidates yourself

Call process_candidate to load a candidate, then fill schema columns \
using set_column. Research missing fields with web_search or \
integrations. Submit or skip.

This is where you learn:
- Which fields are already in the candidate data (don't research those)
- Which tools work for missing fields
- What patterns cause skips (wrong category, out of range, etc.)
- What a row costs

If you see a pattern in skips (e.g. most candidates have wrong size), \
use code_exec to filter remaining candidates before processing more.

For small projects (≤20 rows), you might just do them all yourself. \
For larger ones, process enough to know the strategy (usually 3-8 rows).

### 3. Delegate

Once you know the pattern, call submit_candidates with a note \
summarizing what you learned. Row generators see your full conversation \
so they already know what worked — the note is just a quick summary.

If the feedback report shows problems (high skips, high cost), adjust \
your approach: filter candidates, change instructions, or try a \
different source parameter.

### Live user messages

User messages appear as [User message] in status updates:
- **Behavioral change**: adapt going forward
- **Schema change**: call reprocess_rows to update existing rows
- **Filtering**: call reprocess_rows with filter criteria
- **Start over**: call clear_rows

### Principles

- **Optimize cost per produced row.** That's the metric.
- **Use real data.** Don't fabricate.
- **Filter obvious non-matches programmatically.** If a JSON field \
clearly disqualifies a candidate (wrong size, wrong category), filter \
with code_exec. Don't send obvious skips to row generators.
- **Stick with what works.** Don't switch sources unless exhausted or \
clearly wrong.

<scenarios>
Scenario 1 — Structured integration data:
Task: 100 Airbnb listings in Barcelona, 2+ bedrooms.
1. apify_search("airbnb") → found scraper
2. call_actor(maxItems=20) → 20 listings
3. process_candidate(file, 0) → inspect data, all fields present
   set_column("Title", ...), set_column("Price", ...), submit_row()
4. Process 2 more — pattern clear, all columns from data, no research
5. submit_candidates(file, "All fields in data. Skip if bedrooms < 2.")
6. Good conversion → call_actor(maxItems=200), submit_candidates

Scenario 2 — Need per-row research:
Task: 100 Shopify stores with owner email.
1. call_actor("shopify scraper", maxItems=20) → 20 stores
2. process_candidate(0) — name/URL in data, need owner email
   web_search("store name owner email") → found on About page
   set_column("Owner Email", ...) → submit_row()
3. Process 3 more — web_search works ~80% for email
4. submit_candidates("Stores from Apify. For email: web_search \
   '{{store_name}} owner email'. Skip if not pet products.")

Scenario 3 — Uploaded file:
Task: Enrich 200 companies from CSV.
1. code_exec to check structure → name, website, industry columns
2. process_candidate(0) — need HQ address + phone
   apollo.enrich_company(domain) → got both
3. Process 2 more — apollo works consistently
4. submit_candidates("For HQ + phone: apollo.enrich_company with domain.")

Scenario 4 — Low fertility, needs filtering:
Task: 100 industrial land listings, 2000-15000 sqm.
1. call_actor("immobiliare scraper", maxItems=50) → 50 listings
2. process_candidate(0) → residential land, 500 sqm → skip
3. process_candidate(1) → agricultural, 64000 sqm → skip
4. process_candidate(2) → industrial, 3200 sqm → submit_row()
5. Pattern: most candidates are wrong type/size. Surface area is in JSON.
   code_exec to filter: keep only 2000-15000 sqm → 8 out of 50 remain
6. Process remaining 7, then harvest more with adjusted query
</scenarios>

Today's date: {current_date}

<conversation>
{conversation_history}
</conversation>

<schema>
{columns_description}
</schema>

<resources>
{resources_section}
</resources>
{resume_section}
"""


# ── Per-file processing state ─────────────────────────────────────────

@dataclass
class _FileProcessingState:
    """Tracks processing state for a submitted candidate file."""
    file_path: str
    note: str
    preset_fields: Dict[str, str]
    total_lines: int = 0
    next_line: int = 0            # next unprocessed line index
    processed: int = 0
    in_flight: int = 0
    rows: int = 0
    skipped: int = 0
    duplicates: int = 0
    errors: int = 0
    process_cost: float = 0.0
    skip_reasons: List[str] = field(default_factory=list)


# ── Orchestrator ──────────────────────────────────────────────────────

class OrchestratorV13:
    """
    V13 Orchestrator. Blocking tool calls, file-based candidate flow,
    no harvesters. The LLM conversation loop in base.py handles the
    main loop naturally.
    """

    def __init__(
        self,
        chat_history: List[Dict[str, str]],
        columns: List[Dict[str, Any]],
        num_samples: int,
        openai_client: TrackedOpenAIClient,
        model: str,
        workspace_dir: Path,
        generation_stats: Dict[str, Any],
        dedup_store: Any,
        save_row: Callable[..., Awaitable[Optional[str]]],
        generate_row_fn: Callable,
        uploaded_files: Optional[List[Dict[str, Any]]] = None,
        bu_client: Optional[Any] = None,
        sandbox: Optional[Any] = None,
        stop_checker: Optional[Callable[[], bool]] = None,
        stop_event: Optional[asyncio.Event] = None,
        blob_service_client: Optional[Any] = None,
        project_id: Optional[Any] = None,
        on_tool_call: Optional[Callable[[str, str], None]] = None,
        on_cost: Optional[Callable[[float, str], Awaitable[None]]] = None,
        on_checkpoint: Optional[Callable] = None,
        uploaded_file_urls: Optional[Dict[str, str]] = None,
        mcp_tools: Optional[List[Dict[str, Any]]] = None,
        apollo_client: Optional[Any] = None,
        google_maps_client: Optional[Any] = None,
        youtube_client: Optional[Any] = None,
        apify_client: Optional[Any] = None,
        fullenrich_client: Optional[Any] = None,
        feedback_context: Optional[Dict[str, Any]] = None,
        resume_context: Optional[Dict[str, Any]] = None,
        check_messages: Optional[Callable] = None,
        state: Optional[Any] = None,
        # Sample CRUD for row reprocessing
        read_samples: Optional[Callable] = None,
        update_sample: Optional[Callable] = None,
        delete_sample: Optional[Callable] = None,
    ) -> None:
        self.chat_history = chat_history
        self.columns = columns
        self.num_samples = num_samples
        self.workspace_dir = Path(workspace_dir)
        self.openai_client = openai_client
        self.model = model
        self.bu_client = bu_client
        self.sandbox = sandbox
        self.stop_checker = stop_checker
        self.stop_event = stop_event
        self.blob_service_client = blob_service_client
        self.project_id = project_id
        self.on_tool_call = on_tool_call
        self.on_cost = on_cost
        self._on_checkpoint = on_checkpoint
        self.uploaded_file_urls = uploaded_file_urls
        self.uploaded_files = uploaded_files
        self.mcp_tools = mcp_tools or []
        self.apollo_client = apollo_client
        self.google_maps_client = google_maps_client
        self.apify_client = apify_client
        self.fullenrich_client = fullenrich_client
        self.youtube_client = youtube_client
        self.feedback_context = feedback_context
        self.resume_context = resume_context
        self._check_messages = check_messages
        self._state = state
        self._read_samples = read_samples
        self._update_sample = update_sample
        self._delete_sample = delete_sample

        self._activity_log: List[Dict[str, Any]] = []

        self._generation_stats = generation_stats
        self._save_row = save_row
        self._generate_row_fn = generate_row_fn
        self._dedup_store = dedup_store
        self._save_lock = asyncio.Lock()

        # Processing state
        self._current_file: Optional[_FileProcessingState] = None
        self._processing_semaphore = asyncio.Semaphore(10)
        self._active_tasks: set = set()
        self._finish_requested: bool = False
        self._start_time: float = time.time()
        self._last_checkpoint_time: float = 0.0

        # Counters for unique file naming
        self._web_harvest_counter: int = 0
        self._bu_extract_counter: int = 0
        self._apify_run_counter: int = 0

        # Workshop mode — orchestrator processes rows directly
        self._workshop_row: Dict[str, Any] = {}
        self._workshop_candidate: Optional[str] = None

        # Candidate tracking for status line
        self._candidates_harvested: int = 0
        self._candidates_submitted: int = 0

        # Build tools and conversation
        registry = ToolRegistry()
        self._register_tools(registry)

        system_prompt = self._build_system_prompt()

        from dsl_worker.config import settings
        max_turns = getattr(settings, 'orchestrator_max_turns', 40)

        web_search_tool = {"type": "web_search"}
        all_extra_tools = [web_search_tool] + self.mcp_tools

        self._conversation = AgentConversation(
            openai_client=openai_client,
            model=model,
            system_prompt=system_prompt,
            tools=registry,
            stop_checker=stop_checker,
            stop_event=stop_event,
            max_turns=max_turns,
            soft_turn_limit=max_turns - 5,
            reasoning={"effort": "medium", "summary": "detailed"},
            label="orchestrator",
            continue_on_text=True,
            on_tool_call=on_tool_call,
            on_cost=on_cost,
            extra_tools=all_extra_tools,
        )
        self._conversation.after_turn = self._build_status_line

    def _build_status_line(self) -> Optional[str]:
        """Build a status line injected after every orchestrator turn.

        Also checks for live user messages and injects them before the
        status line so the orchestrator sees them on its next turn.
        """
        parts = []

        # Check for new user messages from the chat
        if self._check_messages:
            try:
                new_messages = self._check_messages()
                for msg in new_messages:
                    content = msg.get("content", "")
                    changes = msg.get("applied_changes") or {}
                    change_details = changes.get("changes", {})

                    msg_parts = [f"[User message]: {content}"]

                    # Summarize schema changes if any
                    if "columns" in change_details:
                        new_cols = change_details["columns"]
                        col_names = [c.get("name", "?") for c in new_cols if isinstance(c, dict)]
                        msg_parts.append(f"[Schema updated: columns are now {', '.join(col_names)}]")
                        # Update our local schema
                        self.columns = new_cols
                    if "num_samples" in change_details:
                        new_target = change_details["num_samples"]
                        msg_parts.append(f"[Target updated: now {new_target} rows]")
                        self.num_samples = new_target

                    parts.append("\n".join(msg_parts))
                    logger.info(f"[orchestrator] Injected live user message: {content[:100]}")
            except Exception as e:
                logger.warning(f"[orchestrator] check_messages error: {e}")

        # Also sync schema from state if available (catches changes we missed)
        if self._state:
            try:
                self._state.refresh()
                if self._state.columns and self._state.columns != self.columns:
                    self.columns = self._state.columns
                if self._state.num_samples and self._state.num_samples != self.num_samples:
                    self.num_samples = self._state.num_samples
            except Exception:
                pass

        # Build status line
        rows = self._generation_stats.get("rows_generated", 0)
        cost = self._generation_stats.get("total_cost", 0.0)
        harvested = self._candidates_harvested
        submitted = self._candidates_submitted

        status = (
            f"[Status] {rows}/{self.num_samples} rows | "
            f"{harvested} candidates harvested | "
            f"{submitted} submitted | "
            f"${cost:.2f} spent"
        )

        # Nudge if candidates exist but none submitted
        if harvested > 0 and submitted == 0:
            status += (
                "\n→ You have candidates but haven't submitted any yet. "
                "Call submit_candidates to start producing rows."
            )

        parts.append(status)
        return "\n\n".join(parts)

    # ── System prompt construction ────────────────────────────────────

    def _build_system_prompt(self) -> str:
        columns_desc = self._format_columns(self.columns)
        convo = self._format_conversation()
        resources = self._format_resources(self.uploaded_files)
        integration_section = self._format_integration_tools()
        resume_section = self._format_resume_section()

        return SYSTEM_PROMPT.format(
            num_samples=self.num_samples,
            columns_description=columns_desc,
            conversation_history=convo,
            resources_section=resources,
            integration_tools_section=integration_section,
            current_date=date.today().isoformat(),
            resume_section=resume_section,
        )

    def _format_columns(self, columns: List[Dict[str, Any]]) -> str:
        if not columns:
            return "(no columns defined)"
        col_lines = []
        for col in columns:
            name = col.get("name", "?")
            fmt = col.get("format", "")
            col_type = col.get("type", "")
            if fmt:
                col_lines.append(f"- {name} — {fmt}")
            elif col_type:
                col_lines.append(f"- {name} ({col_type})")
            else:
                col_lines.append(f"- {name}")
        return "\n".join(col_lines)

    def _format_conversation(self) -> str:
        if not self.chat_history:
            return "(no conversation history)"
        parts = []
        for msg in self.chat_history:
            role = msg.get("role", "unknown")
            content = self._clean_conversation_content(msg.get("content", ""))
            ts = msg.get("created_at", "")
            if ts:
                parts.append(f"[{ts}] **{role}**: {content}")
            else:
                parts.append(f"**{role}**: {content}")
        return "\n\n".join(parts)

    @staticmethod
    def _clean_conversation_content(content: str) -> str:
        """Strip client-side noise from conversation messages."""
        import re
        # Remove sort_regex and sort_multiplier from embedded JSON
        content = re.sub(r',?\s*"sort_regex"\s*:\s*"[^"]*"', '', content)
        content = re.sub(r',?\s*"sort_multiplier"\s*:\s*\d+', '', content)
        return content

    def _format_resources(self, uploaded_files: Optional[List[Dict[str, Any]]]) -> str:
        if not uploaded_files:
            return "No uploaded files."
        lines = ["Uploaded files at /workspace/uploads/ (read-only):"]
        for idx, f in enumerate(uploaded_files, 1):
            name = f.get("filename", "unknown")
            size = f.get("size_bytes", 0)
            ctype = f.get("content_type", "")
            if size > 1_000_000:
                size_str = f"{size / 1_000_000:.1f} MB"
            elif size > 1_000:
                size_str = f"{size / 1_000:.0f} KB"
            else:
                size_str = f"{size} bytes"
            lines.append(f"  [{idx}] /workspace/uploads/{name} ({ctype}, {size_str})")
            inspection = f.get("inspection")
            if inspection:
                row_count = (
                    inspection.get("row_count")
                    or inspection.get("item_count")
                    or inspection.get("line_count")
                )
                cols = inspection.get("columns") or inspection.get("keys") or []
                if row_count is not None:
                    lines.append(f"      {row_count} rows, {len(cols)} columns")
                if cols:
                    cols_str = ", ".join(cols[:20])
                    if len(cols) > 20:
                        cols_str += f" ... ({len(cols)} total)"
                    lines.append(f"      Columns: {cols_str}")
                preview = inspection.get("preview")
                if preview:
                    lines.append(
                        f"      Sample: {json.dumps(preview[0], default=str)[:300]}"
                    )
        return "\n".join(lines)

    def _format_integration_tools(self) -> str:
        """Build the integration tools section of the system prompt.

        Describes available namespaces — individual tools are deferred
        and discovered via tool_search when needed.
        """
        sections = []

        if self.apify_client:
            sections.append(
                "**apify** — 22,000+ pre-built web scrapers. Best for any specific "
                "website. Search for actors, check schemas, run scrapers. Cheap and fast."
            )

        if hasattr(self, 'fullenrich_client') and self.fullenrich_client:
            sections.append(
                "**fullenrich** — Search for people (by title, location, company, "
                "seniority, skills) and companies (by industry, location, size). "
                "Enrich contacts with verified emails and phones via 20+ providers."
            )

        if self.apollo_client:
            sections.append(
                "**apollo** — B2B company search and enrichment. Search companies "
                "by industry, location, size, tech stack. Enrich people and companies."
            )

        if self.google_maps_client:
            sections.append(
                "**google_maps** — Local business search, place details (phone, "
                "website, hours, reviews), geocoding, directions, distance matrix."
            )

        if not sections:
            return ""
        return (
            "Available integrations:\n"
            + "\n".join(f"- {s}" for s in sections)
            + "\n"
        )

    def _format_resume_section(self) -> str:
        """Format resume context if this is a resumed job."""
        if self.feedback_context:
            prev = self.feedback_context.get("previous_config", {})
            feedback = self.feedback_context.get("user_feedback", "")
            return (
                f"\n\n**Previous pipeline config:**\n"
                f"```json\n{json.dumps(prev, indent=2)}\n```\n\n"
                f"**User feedback:** \"{feedback}\"\n\n"
                f"The previous results were discarded. Design a new approach "
                f"based on this feedback."
            )

        if self.resume_context:
            rc = self.resume_context
            return (
                f"\n\nRESUMING: {rc['rows_generated']}/{rc['target']} rows "
                f"already done ({rc['remaining']} remaining). "
                f"Prior cost: ${rc['prior_cost_usd']:.4f}."
            )

        return ""

    # ── Activity log ──────────────────────────────────────────────────

    def _log_activity(self, text: str) -> None:
        """Append an entry to the activity log (surfaced to frontend)."""
        self._activity_log.append({
            "ts": time.time(),
            "text": text,
        })
        # Keep last 50 entries
        if len(self._activity_log) > 50:
            self._activity_log = self._activity_log[-50:]

    def get_activity_log(self) -> List[Dict[str, Any]]:
        return list(self._activity_log)

    # ── File helpers ───────────────────────────────────────────────────

    def _to_workspace_path(self, local_path) -> str:
        """Convert a local file path to /workspace/ path for the LLM."""
        local_str = str(local_path)
        workspace_str = str(self.workspace_dir)
        if local_str.startswith(workspace_str):
            relative = local_str[len(workspace_str):].lstrip("/")
            return f"/workspace/{relative}"
        return local_str

    async def _write_candidates_file(self, filename: str, content: str) -> str:
        """Write a candidate file locally and return the /workspace/ path.

        Files are written to workspace_dir/candidates/ locally. They get
        synced to the sandbox automatically before each code_exec call.
        The LLM sees /workspace/candidates/ paths consistently.
        """
        sandbox_path = f"candidates/{filename}"
        workspace_path = f"/workspace/{sandbox_path}"

        local_path = self.workspace_dir / sandbox_path
        local_path.parent.mkdir(parents=True, exist_ok=True)
        local_path.write_text(content, encoding="utf-8")

        # Persist to blob for pause/resume (also counts candidates)
        self._upload_candidate_to_blob(local_path)

        return workspace_path

    def _upload_candidate_to_blob(self, local_path: Path) -> None:
        """Upload a candidate file to blob storage for pause/resume durability.

        Also counts lines to track _candidates_harvested — this is how
        integration tools (apify, fullenrich, etc.) report their output
        since they bypass _write_candidates_file().
        """
        # Count candidates in the file
        line_count = 0
        try:
            with open(local_path, "r", encoding="utf-8", errors="replace") as f:
                line_count = sum(1 for line in f if line.strip())
            self._candidates_harvested += line_count
        except Exception:
            pass

        # Log activity — infer source from filename
        if line_count > 0:
            name = local_path.stem
            if "apify" in name:
                self._log_activity(f"Harvested {line_count} candidates from Apify")
            elif "fullenrich" in name:
                self._log_activity(f"Found {line_count} results from FullEnrich")
            elif "apollo" in name:
                self._log_activity(f"Found {line_count} results from Apollo")
            elif "gmaps" in name:
                self._log_activity(f"Found {line_count} places from Google Maps")
            else:
                self._log_activity(f"Harvested {line_count} candidates")

        if not self.blob_service_client:
            return
        blob_path = f"projects/{self.project_id}/candidates/{local_path.name}"
        try:
            from dsl_worker.config import settings
            blob = self.blob_service_client.get_blob_client(
                container=settings.azure_storage_container_name,
                blob=blob_path,
            )
            with open(local_path, "rb") as f:
                blob.upload_blob(f, overwrite=True)
            logger.info(f"[orchestrator] Uploaded candidate file to blob: {local_path.name}")
        except Exception as e:
            logger.warning(f"[orchestrator] Failed to upload candidate to blob: {e}")

    # ── Tool registration ─────────────────────────────────────────────

    def _register_tools(self, registry: ToolRegistry) -> None:
        # Core tools (always loaded)
        self._register_code_exec(registry)
        self._register_web_harvest(registry)
        self._register_bu_extract(registry)
        self._register_workshop_tools(registry)
        self._register_submit_candidates(registry)
        self._register_continue_processing(registry)
        self._register_reprocess_rows(registry)
        self._register_clear_rows(registry)
        self._register_finish(registry)

        # Enable deferred tool discovery — namespace tools only load when needed
        registry.add_builtin({"type": "tool_search"})

        # Shared file counter for integrations — start from existing file count
        # so resume doesn't collide with files from before pause
        candidates_dir = self.workspace_dir / "candidates"
        existing_files = len(list(candidates_dir.glob("*"))) if candidates_dir.exists() else 0
        file_counter = [existing_files]

        # Integration namespaces
        if self.apify_client:
            from dsl_worker.agents.integrations.apify import register_apify_namespace
            from dsl_worker.config import settings
            register_apify_namespace(
                registry, self.apify_client, settings.apify_api_key,
                self.workspace_dir,
                file_counter=file_counter,
                on_file_written=self._upload_candidate_to_blob,
            )
        if self.apollo_client:
            from dsl_worker.agents.integrations.apollo import register_apollo_namespace
            from dsl_worker.config import settings as _apollo_settings
            register_apollo_namespace(
                registry, self.apollo_client, self.workspace_dir,
                file_counter=file_counter,
                cost_per_credit=_apollo_settings.apollo_cost_per_credit,
                on_file_written=self._upload_candidate_to_blob,
            )
        if self.google_maps_client:
            from dsl_worker.agents.integrations.google_maps import register_google_maps_namespace
            from dsl_worker.config import settings as _settings
            register_google_maps_namespace(
                registry, _settings.google_api_key, self.workspace_dir,
                file_counter=file_counter,
                on_file_written=self._upload_candidate_to_blob,
            )
        if hasattr(self, 'fullenrich_client') and self.fullenrich_client:
            from dsl_worker.agents.integrations.fullenrich import register_fullenrich_namespace
            from dsl_worker.config import settings as _fe_settings
            register_fullenrich_namespace(
                registry, self.fullenrich_client, self.workspace_dir,
                file_counter=file_counter,
                cost_per_credit=_fe_settings.fullenrich_cost_per_credit,
                on_file_written=self._upload_candidate_to_blob,
            )

    # --- code_exec ---

    def _register_code_exec(self, registry: ToolRegistry) -> None:
        self._sandbox_impl = None
        if not self.sandbox:
            # Register a stub that tells the LLM code_exec isn't available
            async def no_sandbox(args: Dict) -> Tuple[str, float]:
                return "Code execution not available in this environment.", 0.0

            registry.add(
                name="code_exec",
                description="Run Python in a sandbox. Not available in this environment.",
                parameters={
                    "type": "object",
                    "properties": {
                        "script": {"type": "string", "description": "Python script to execute"},
                    },
                    "required": ["script"],
                },
                handler=no_sandbox,
            )
            return

        from dsl_worker.infra.research_tools import ResearchTools, ResearchScope

        self._sandbox_impl = ResearchTools(
            workspace_dir=self.workspace_dir,
            schema=[],
            brave_api_key=None,
            openai_client=self.openai_client,
            model=self.model,
            sandbox=self.sandbox,
            stop_checker=self.stop_checker,
            blob_service_client=self.blob_service_client,
            project_id=self.project_id,
            uploaded_file_urls=self.uploaded_file_urls,
        )
        self._sandbox_impl.set_scope(ResearchScope(
            id="orchestrator",
            description="",
            quota=0,
        ))
        # Wrap code_exec to sync candidate files to sandbox before each call.
        # Integration tools write candidates locally; we need them visible
        # in the sandbox for the LLM's Python scripts.
        original_code_exec = self._sandbox_impl.code_exec

        async def synced_code_exec(script: str, description: str = "") -> Tuple[str, float]:
            # Sync local candidates/ to sandbox before running script
            candidates_dir = self.workspace_dir / "candidates"
            if candidates_dir.exists():
                try:
                    session = await self._sandbox_impl._get_sandbox_session()
                    for f in candidates_dir.iterdir():
                        if f.is_file() and not f.name.startswith("."):
                            content = f.read_text(encoding="utf-8", errors="replace")
                            await session.write_file(f"candidates/{f.name}", content)
                except Exception as e:
                    logger.warning(f"[orchestrator] Failed to sync candidates to sandbox: {e}")

            result = await original_code_exec(script, description)

            # After code_exec: sync any new/modified files FROM sandbox back to local.
            # This ensures local is always the source of truth — sandbox can die
            # and we lose nothing.
            try:
                if self._sandbox_impl and self._sandbox_impl._sandbox_session:
                    session = self._sandbox_impl._sandbox_session
                    # List files in sandbox candidates/ and download any we don't have locally
                    try:
                        listing = await session.exec_shell(
                            "ls -1 /workspace/candidates/ 2>/dev/null || true", timeout=5
                        )
                        if listing.success and listing.stdout.strip():
                            sandbox_files = set(listing.stdout.strip().split("\n"))
                            local_files = set(
                                f.name for f in candidates_dir.iterdir()
                                if f.is_file()
                            ) if candidates_dir.exists() else set()

                            new_files = sandbox_files - local_files
                            for fname in new_files:
                                if fname.startswith("."):
                                    continue
                                try:
                                    content = await session.read_file(f"candidates/{fname}")
                                    local_file = candidates_dir / fname
                                    local_file.write_text(content, encoding="utf-8")
                                    # Persist to blob for pause/resume
                                    self._upload_candidate_to_blob(local_file)
                                except Exception:
                                    pass
                    except Exception:
                        pass
            except Exception as e:
                logger.warning(f"[orchestrator] Post-exec sync error: {e}")

            return result

        registry.add(
            name="code_exec",
            description=(
                "Run Python in a sandbox. Files at /workspace/uploads/ "
                "(user files) and /workspace/candidates/ (your output). "
                "pandas, json, csv, openpyxl, pdfplumber available. "
                "Use from dsl_tools import read_jsonl, write_jsonl, preview "
                "for common operations."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "script": {
                        "type": "string",
                        "description": "Python script to execute.",
                    },
                },
                "required": ["script"],
            },
            handler=lambda args: synced_code_exec(args.get("script", ""), args.get("description", "")),
        )

    # --- web_harvest ---

    def _register_web_harvest(self, registry: ToolRegistry) -> None:

        async def web_harvest(args: Dict) -> Tuple[str, float]:
            query = args.get("query", "")
            candidate_description = args.get("candidate_description", "")

            if not query:
                return "Error: query is required.", 0.0

            idx = self._web_harvest_counter
            self._web_harvest_counter += 1
            timestamp = int(time.time())
            filename = f"web_harvest_{idx}_{timestamp}.jsonl"

            total_cost = 0.0
            candidates_found = 0
            collected_lines: List[str] = []

            try:
                # Build the subagent
                sub_registry = ToolRegistry()
                sub_system_prompt = self._build_web_harvest_prompt(
                    query, candidate_description, f"/workspace/candidates/{filename}",
                )

                # yield_candidate tool — accumulates lines in memory
                async def yield_candidate(yc_args: Dict) -> Tuple[str, float]:
                    nonlocal candidates_found
                    data = yc_args.get("data", {})
                    if not data:
                        return "Error: data is required.", 0.0
                    try:
                        line = json.dumps(data, ensure_ascii=False)
                        collected_lines.append(line)
                        candidates_found += 1
                        return f"Candidate #{candidates_found} saved.", 0.0
                    except Exception as e:
                        return f"Error writing candidate: {e}", 0.0

                sub_registry.add(
                    name="yield_candidate",
                    description=(
                        "Save a candidate to the output file. Pass a JSON object "
                        "with whatever fields you found for this entity."
                    ),
                    parameters={
                        "type": "object",
                        "properties": {
                            "data": {
                                "type": "object",
                                "description": "Candidate data as a JSON object.",
                                "additionalProperties": True,
                            },
                        },
                        "required": ["data"],
                    },
                    handler=yield_candidate,
                )

                # Web search is built-in via extra_tools
                web_search_tool = {"type": "web_search"}

                subagent = AgentConversation(
                    openai_client=self.openai_client,
                    model=self.model,
                    system_prompt=sub_system_prompt,
                    tools=sub_registry,
                    stop_checker=self.stop_checker,
                    stop_event=self.stop_event,
                    max_turns=30,
                    soft_turn_limit=20,
                    reasoning={"effort": "low", "summary": "concise"},
                    label=f"web_harvest:{idx}",
                    continue_on_text=False,
                    on_cost=self.on_cost,
                    extra_tools=[web_search_tool],
                )

                result = await subagent.send(
                    f"Research: {query}\n\n"
                    f"Find candidates matching: {candidate_description}\n\n"
                    f"Search the web, open promising pages, and yield_candidate "
                    f"for each entity you find."
                )
                total_cost = subagent.total_cost

            except asyncio.CancelledError:
                logger.info(f"[web_harvest:{idx}] cancelled")
            except Exception as e:
                logger.error(f"[web_harvest:{idx}] error: {e}", exc_info=True)
                return f"Web research failed: {e}", total_cost

            if candidates_found == 0:
                return (
                    f"Web research completed but found 0 candidates. "
                    f"Cost: ${total_cost:.4f}. "
                    f"The query may be too specific or the data isn't "
                    f"available on the open web."
                ), total_cost

            # Write all collected lines to sandbox + local
            output_path = await self._write_candidates_file(
                filename, "\n".join(collected_lines) + "\n"
            )

            # Build sample from collected lines
            sample_lines = []
            for raw_line in collected_lines[:3]:
                try:
                    sample_lines.append(json.loads(raw_line))
                except json.JSONDecodeError:
                    pass

            sample_str = ""
            if sample_lines:
                sample_str = (
                    "\nSample:\n"
                    + "\n".join(
                        json.dumps(s, ensure_ascii=False)[:200]
                        for s in sample_lines
                    )
                )

            self._log_activity(f"Web search: {candidates_found} candidates found")

            return (
                f"Web research complete: {candidates_found} candidates found.\n"
                f"File: {output_path}\n"
                f"Cost: ${total_cost:.4f}{sample_str}\n\n"
                f"Next: call submit_candidates with this file to start "
                f"producing rows."
            ), total_cost

        registry.add(
            name="web_harvest",
            description=(
                "Harvest candidates from the web. Spawns an agent that googles "
                "the query, visits pages, and extracts entities to a JSONL file. "
                "~$0.10-0.20/call. For harvesting entities only — not for research."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "What to search for.",
                    },
                    "candidate_description": {
                        "type": "string",
                        "description": (
                            "What a good candidate looks like — the entity type, "
                            "what fields to extract, any filters."
                        ),
                    },
                },
                "required": ["query", "candidate_description"],
            },
            handler=web_harvest,
        )

    def _build_web_harvest_prompt(
        self, query: str, candidate_description: str, output_file: str,
    ) -> str:
        return (
            "# Web Research Agent\n\n"
            "You search the web and yield candidate entities for a dataset.\n\n"
            "## Tools\n\n"
            "- **Web search** (built-in) — search for information.\n"
            "- **yield_candidate(data)** — save a candidate to the output file. "
            "Call this for each entity you find. Pass a JSON object with all "
            "relevant fields.\n\n"
            "## How to work\n\n"
            "1. Search for the query.\n"
            "2. Open promising results (aim for 5-15 pages depending on density).\n"
            "3. For each relevant entity found, call yield_candidate with its data.\n"
            "4. Be thorough but efficient. Don't visit obviously irrelevant pages.\n"
            "5. When you've exhausted the useful results, stop.\n\n"
            f"## Output\n\nCandidates are written to: {output_file}\n\n"
            f"## Query\n\n{query}\n\n"
            f"## Candidate Description\n\n{candidate_description}"
        )

    # --- bu_extract ---

    def _register_bu_extract(self, registry: ToolRegistry) -> None:

        async def bu_extract(args: Dict) -> Tuple[str, float]:
            task = args.get("task", "")
            if not task:
                return "Error: task is required.", 0.0

            if not self.bu_client:
                return "BU browser not available in this environment.", 0.0

            idx = self._bu_extract_counter
            self._bu_extract_counter += 1
            timestamp = int(time.time())
            filename = f"bu_extract_{idx}_{timestamp}.jsonl"

            try:
                items, cost, session_id, summary = await self.bu_client.extract(
                    task=task,
                    timeout=300,
                )

                if items:
                    # Build JSONL content
                    lines = []
                    for item in items:
                        lines.append(json.dumps(item, ensure_ascii=False))

                    output_path = await self._write_candidates_file(
                        filename, "\n".join(lines) + "\n"
                    )

                    # Build sample for summary
                    sample_str = ""
                    if items[:3]:
                        sample_str = (
                            "\nSample:\n"
                            + "\n".join(
                                json.dumps(s, ensure_ascii=False)[:200]
                                for s in items[:3]
                            )
                        )

                    return (
                        f"BU extraction complete: {len(items)} items extracted.\n"
                        f"File: {output_path}\n"
                        f"Cost: ${cost:.4f}\n"
                        f"BU summary: {summary[:300]}{sample_str}"
                    ), cost
                else:
                    return (
                        f"BU extraction completed but found 0 items.\n"
                        f"Cost: ${cost:.4f}\n"
                        f"BU summary: {summary[:300]}"
                    ), cost

            except asyncio.CancelledError:
                logger.info(f"[bu_extract:{idx}] cancelled")
                return "BU extraction cancelled (job paused).", 0.0
            except Exception as e:
                logger.error(f"[bu_extract:{idx}] error: {e}", exc_info=True)
                return f"BU extraction failed: {e}", 0.0

        registry.add(
            name="browser_use",
            description=(
                "Open a cloud browser to extract data from a specific URL. "
                "$0.10-0.50/session. Use ONLY when the site requires a real "
                "browser (anti-bot, JS interaction, captcha) AND no Apify actor "
                "exists for it. One URL, one task, tightly scoped."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "task": {
                        "type": "string",
                        "description": (
                            "ONE specific URL + ONE specific extraction task. "
                            "E.g. 'Go to https://example.com/listings and extract "
                            "the business names and phone numbers from the table.'"
                        ),
                    },
                },
                "required": ["task"],
            },
            handler=bu_extract,
        )

    # --- workshop tools (process rows directly) ---

    def _register_workshop_tools(self, registry: ToolRegistry) -> None:

        async def process_candidate(args: Dict) -> Tuple[str, float]:
            """Load a candidate from a file for the orchestrator to process directly."""
            file_path = args.get("file", "")
            index = args.get("index", 0)

            if not file_path:
                return "Error: file is required.", 0.0

            # Resolve workspace path
            if file_path.startswith("/workspace/"):
                local_path = self.workspace_dir / file_path[len("/workspace/"):]
            else:
                local_path = Path(file_path)
                if not local_path.is_absolute():
                    local_path = self.workspace_dir / file_path

            if not local_path.exists():
                return f"Error: file not found: {file_path}", 0.0

            # Read the specific line
            try:
                with open(local_path, "r", encoding="utf-8") as f:
                    for i, line in enumerate(f):
                        if i == index:
                            line = line.strip()
                            if not line:
                                return f"Error: line {index} is empty.", 0.0
                            candidate_data = json.loads(line)
                            self._workshop_row = {}
                            self._workshop_candidate = json.dumps(
                                candidate_data, indent=2, ensure_ascii=False
                            )
                            return (
                                f"Candidate {index} loaded. Data:\n\n"
                                f"```json\n{self._workshop_candidate[:3000]}\n```\n\n"
                                f"Now fill the schema columns with set_column, "
                                f"then call submit_row or skip_row."
                            ), 0.0
                    return f"Error: file has fewer than {index + 1} lines.", 0.0
            except json.JSONDecodeError:
                return f"Error: invalid JSON at line {index}.", 0.0
            except Exception as e:
                return f"Error reading file: {e}", 0.0

        registry.add(
            name="process_candidate",
            description=(
                "Load a candidate from a JSONL file to process directly. "
                "Use this in workshop mode — process a few candidates yourself "
                "before delegating to row generators."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "file": {
                        "type": "string",
                        "description": "Path to JSONL file.",
                    },
                    "index": {
                        "type": "integer",
                        "description": "Line index (0-based) of the candidate to load.",
                    },
                },
                "required": ["file", "index"],
            },
            handler=process_candidate,
        )

        async def set_column(args: Dict) -> Tuple[str, float]:
            name = args.get("name", "")
            value = args.get("value")

            if self._workshop_candidate is None:
                return "Error: no candidate loaded. Call process_candidate first.", 0.0

            # Case-insensitive column match
            matched_name = None
            for col in self.columns:
                if col.get("name", "").lower() == name.lower():
                    matched_name = col["name"]
                    break
            if not matched_name:
                valid = [c.get("name") for c in self.columns]
                return f"Error: unknown column '{name}'. Valid: {valid}", 0.0

            self._workshop_row[matched_name] = value
            return f"Set {matched_name}", 0.0

        registry.add(
            name="set_column",
            description="Set a column value on the current candidate.",
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

        async def submit_row(args: Dict) -> Tuple[str, float]:
            if self._workshop_candidate is None:
                return "Error: no candidate loaded.", 0.0

            missing = [
                col.get("name") for col in self.columns
                if col.get("name") and col.get("name") not in self._workshop_row
            ]
            if missing:
                return f"Error: missing columns {missing}. Set them before submitting.", 0.0

            # Save to DB
            row_id = await self._save_row(self._workshop_row)
            if row_id:
                self._generation_stats["rows_generated"] = (
                    self._generation_stats.get("rows_generated", 0) + 1
                )
                rows_done = self._generation_stats["rows_generated"]
                self._log_activity(f"Row {rows_done} submitted (workshop)")
                self._workshop_row = {}
                self._workshop_candidate = None
                return f"Row submitted ({rows_done}/{self.num_samples} done).", 0.0
            else:
                self._workshop_row = {}
                self._workshop_candidate = None
                return "Target reached — row discarded.", 0.0

        registry.add(
            name="submit_row",
            description="Submit the current row. Call when all columns are filled.",
            parameters={"type": "object", "properties": {}},
            handler=submit_row,
        )

        async def skip_row(args: Dict) -> Tuple[str, float]:
            reason = args.get("reason", "")
            if self._workshop_candidate is None:
                return "Error: no candidate loaded.", 0.0

            self._generation_stats["skipped"] = (
                self._generation_stats.get("skipped", 0) + 1
            )
            self._workshop_row = {}
            self._workshop_candidate = None
            return f"Skipped: {reason}", 0.0

        registry.add(
            name="skip_row",
            description=(
                "Skip this candidate — wrong category, doesn't match criteria, etc."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "reason": {"type": "string", "description": "Why"},
                },
                "required": ["reason"],
            },
            handler=skip_row,
        )

    # --- submit_candidates ---

    def _extract_workshop_transcript(self) -> str:
        """Extract workshop tool calls from conversation as a readable transcript.

        Returns a condensed view of process_candidate/set_column/submit_row/skip_row
        calls so row generators can see exactly what the orchestrator did.
        """
        workshop_tools = {"process_candidate", "set_column", "submit_row", "skip_row"}
        lines = []
        in_workshop = False

        for msg in self._conversation.messages:
            if not isinstance(msg, dict):
                continue

            if msg.get("type") == "function_call":
                name = msg.get("name", "")
                if name in workshop_tools:
                    in_workshop = True
                    args = msg.get("arguments", "")
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except (json.JSONDecodeError, TypeError):
                            pass
                    if name == "process_candidate":
                        idx = args.get("index", "?") if isinstance(args, dict) else "?"
                        lines.append(f"\n--- Candidate {idx} ---")
                    elif name == "set_column":
                        col = args.get("name", "?") if isinstance(args, dict) else "?"
                        val = args.get("value", "") if isinstance(args, dict) else ""
                        val_str = str(val)[:100]
                        lines.append(f"  {col} = {val_str}")
                    elif name == "submit_row":
                        lines.append(f"  → submitted")
                    elif name == "skip_row":
                        reason = args.get("reason", "") if isinstance(args, dict) else ""
                        lines.append(f"  → skipped: {reason[:100]}")

                # Also capture web_search calls during workshop
                elif in_workshop and name == "web_search":
                    lines.append(f"  [web_search]")

        if not lines:
            return ""

        return "## Workshop — what I did for sample rows\n\n" + "\n".join(lines) + "\n\n"

    def _register_submit_candidates(self, registry: ToolRegistry) -> None:

        async def submit_candidates(args: Dict) -> Tuple[str, float]:
            file_path = args.get("file", "")
            note = args.get("instructions", "") or args.get("note", "")
            preset_fields = args.get("preset_fields", {})
            checkin_after = args.get("checkin_after", 10)

            # Prepend workshop transcript so row gens see what the orchestrator did
            workshop = self._extract_workshop_transcript()
            if workshop:
                note = workshop + note

            if not file_path:
                return "Error: file is required.", 0.0

            # Resolve file to local path. Files are written both locally
            # and to sandbox by _write_candidates_file, so local should
            # always exist. /workspace/X maps to workspace_dir/X.
            if file_path.startswith("/workspace/"):
                local_path = self.workspace_dir / file_path[len("/workspace/"):]
            else:
                local_path = Path(file_path)
                if not local_path.is_absolute():
                    local_path = self.workspace_dir / file_path

            # If not found locally, try reading from sandbox (code_exec case)
            if not local_path.exists():
                sandbox_path = file_path
                if sandbox_path.startswith("/workspace/"):
                    sandbox_path = sandbox_path[len("/workspace/"):]
                try:
                    if self._sandbox_impl:
                        session = await self._sandbox_impl._get_sandbox_session()
                        content = await session.read_file(sandbox_path)
                        local_path.parent.mkdir(parents=True, exist_ok=True)
                        local_path.write_text(content, encoding="utf-8")
                        logger.info(f"[orchestrator] Synced {sandbox_path} from sandbox to local")
                    else:
                        return f"Error: file not found: {file_path}", 0.0
                except Exception as e:
                    return f"Error: file not found locally or in sandbox: {file_path} ({e})", 0.0

            # Count lines
            total_lines = 0
            try:
                with open(local_path, "r", encoding="utf-8") as f:
                    for _ in f:
                        total_lines += 1
            except Exception as e:
                return f"Error reading file: {e}", 0.0

            if total_lines == 0:
                return "Error: file is empty.", 0.0

            self._candidates_submitted += total_lines
            self._log_activity(f"Submitted {total_lines} candidates for processing")

            # Set up file processing state
            self._current_file = _FileProcessingState(
                file_path=str(local_path),
                note=note,
                preset_fields=preset_fields if isinstance(preset_fields, dict) else {},
                total_lines=total_lines,
                next_line=0,
            )

            return await self._process_batch(checkin_after)

        registry.add(
            name="submit_candidates",
            description=(
                "Send candidates to row generators. Each follows your instructions "
                "to process the candidate into a row. This is how you produce rows "
                "— nothing happens until you call this."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "file": {
                        "type": "string",
                        "description": (
                            "Path to JSONL file (e.g. the file returned by "
                            "apify_run or web_harvest)."
                        ),
                    },
                    "instructions": {
                        "type": "string",
                        "description": (
                            "Instructions for row generators: what the data looks "
                            "like, which fields map to schema columns, what to "
                            "research for gaps, what makes a valid row vs skip, "
                            "and any tips to save cost."
                        ),
                    },
                    "checkin_after": {
                        "type": "integer",
                        "description": "How many candidates to process before returning feedback.",
                    },
                    "preset_fields": {
                        "type": "object",
                        "description": (
                            "Optional. Map schema columns to candidate field names "
                            "to pre-fill. Supports dot-notation for nested fields."
                        ),
                        "additionalProperties": {"type": "string"},
                    },
                },
                "required": ["file", "instructions"],
            },
            handler=submit_candidates,
        )

    # --- continue_processing ---

    def _register_continue_processing(self, registry: ToolRegistry) -> None:

        async def continue_processing(args: Dict) -> Tuple[str, float]:
            checkin_after = args.get("checkin_after", 10)

            if not self._current_file:
                return (
                    "Error: no file is being processed. "
                    "Call submit_candidates first."
                ), 0.0

            state = self._current_file
            remaining = state.total_lines - state.next_line

            if remaining <= 0 and state.in_flight == 0:
                return self._build_feedback_report(
                    state, final=True,
                    note="All candidates from this file have been processed.",
                ), 0.0

            return await self._process_batch(checkin_after)

        registry.add(
            name="continue_processing",
            description=(
                "Resume processing remaining candidates from the last submitted "
                "file. Blocks until checkin_after more candidates are processed."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "checkin_after": {
                        "type": "integer",
                        "description": "How many more processed candidates before next feedback.",
                    },
                },
            },
            handler=continue_processing,
        )

    # --- reprocess_rows ---

    def _register_reprocess_rows(self, registry: ToolRegistry) -> None:
        if not self._read_samples:
            return

        async def reprocess_rows(args: Dict) -> Tuple[str, float]:
            instructions = args.get("instructions", "")
            if not instructions:
                return "Error: instructions are required.", 0.0

            samples = await self._read_samples()
            if not samples:
                return "No rows to reprocess.", 0.0

            total = len(samples)
            updated = 0
            skipped = 0
            errors = 0
            total_cost = 0.0

            self._log_activity(f"Reprocessing {total} rows")

            for sample in samples:
                if self.stop_checker and self.stop_checker():
                    break

                try:
                    candidate = type("Candidate", (), {
                        "values": json.dumps(sample["row"], ensure_ascii=False),
                        "source_id": "reprocess",
                        "source_context": instructions,
                        "metadata": {
                            "preset_fields": {},
                            "candidate_data": sample["row"],
                            "reprocess_sample_id": sample["id"],
                        },
                    })()

                    result, cost, saved = await self._generate_row_fn(
                        candidate, "reprocess"
                    )
                    total_cost += cost

                    if result.success and result.row:
                        await self._update_sample(sample["id"], result.row)
                        updated += 1
                    elif result.skipped:
                        await self._delete_sample(sample["id"])
                        skipped += 1
                    else:
                        errors += 1

                except Exception as e:
                    logger.error(f"[orchestrator] Reprocess error: {e}", exc_info=True)
                    errors += 1

            self._log_activity(f"Reprocessed: {updated} updated, {skipped} removed")

            return (
                f"Reprocessed {total} rows:\n"
                f"  Updated: {updated}\n"
                f"  Filtered out: {skipped}\n"
                f"  Errors: {errors}\n"
                f"  Cost: ${total_cost:.4f}"
            ), total_cost

        registry.add(
            name="reprocess_rows",
            description=(
                "Re-evaluate ALL existing rows through row generators. Use when "
                "user asks to: add/fill a column, filter rows, or modify existing "
                "data. Rows are modified in place. Prefer this over clear_rows — "
                "most rows can be updated rather than discarded."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "instructions": {
                        "type": "string",
                        "description": (
                            "Instructions for row generators on what to do with each row. "
                            "Examples: 'Fill in the Email column using web search', "
                            "'Skip rows where karma < 500', "
                            "'Add a Website column by searching for each company'."
                        ),
                    },
                },
                "required": ["instructions"],
            },
            handler=reprocess_rows,
        )

    # --- clear_rows ---

    def _register_clear_rows(self, registry: ToolRegistry) -> None:
        if not self._read_samples:
            return

        async def clear_rows(args: Dict) -> Tuple[str, float]:
            reason = args.get("reason", "User requested fresh start")

            rows_done = self._generation_stats.get("rows_generated", 0)
            if rows_done == 0:
                return "No rows to clear.", 0.0

            samples = await self._read_samples()
            for s in samples:
                await self._delete_sample(s["id"])
            self._generation_stats["rows_generated"] = 0
            self._log_activity(f"Cleared {rows_done} rows: {reason}")

            return f"Cleared {rows_done} rows. Starting fresh.", 0.0

        registry.add(
            name="clear_rows",
            description=(
                "Delete ALL existing rows and start over. Last resort — only when "
                "existing rows are genuinely unsalvageable and a completely different "
                "approach is needed. Prefer reprocess_rows when possible."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "reason": {
                        "type": "string",
                        "description": "Why rows can't be salvaged.",
                    },
                },
                "required": ["reason"],
            },
            handler=clear_rows,
        )

    # --- finish ---

    def _register_finish(self, registry: ToolRegistry) -> None:

        async def finish(args: Dict) -> Tuple[str, float]:
            reason = args.get("reason", "No reason provided")
            self._finish_requested = True
            logger.info(f"[orchestrator] finish() called: {reason}")
            return f"Job aborted: {reason}", 0.0

        registry.add(
            name="finish",
            description=(
                "Abort the job. ONLY for genuinely impossible tasks — all feasible "
                "approaches exhausted, 100% certainty there's no viable path. "
                "Target reached and credits exhausted are handled automatically."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "reason": {
                        "type": "string",
                        "description": "Why the task is impossible.",
                    },
                },
                "required": ["reason"],
            },
            handler=finish,
        )

    # ── Candidate processing engine ───────────────────────────────────

    async def _process_batch(self, checkin_after: int) -> Tuple[str, float]:
        """Process a batch of candidates from the current file.

        Reads up to checkin_after + floor(checkin_after * 0.5) candidates
        from the file, processes them concurrently (10 max), blocks until
        checkin_after are done, returns feedback report.
        """
        state = self._current_file
        if not state:
            return "Error: no file loaded.", 0.0

        checkin_after = max(1, checkin_after)
        # Optimism buffer: process extra candidates speculatively
        buffer_size = checkin_after + math.floor(checkin_after * 0.5)
        remaining = state.total_lines - state.next_line

        if remaining <= 0 and state.in_flight == 0:
            return self._build_feedback_report(
                state, final=True,
                note="All candidates have been processed.",
            ), 0.0

        to_dispatch = min(buffer_size, remaining)

        # Read candidates from file
        candidates_to_process: List[Tuple[int, Dict]] = []
        try:
            with open(state.file_path, "r", encoding="utf-8") as f:
                for i, line in enumerate(f):
                    if i < state.next_line:
                        continue
                    if len(candidates_to_process) >= to_dispatch:
                        break
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        candidates_to_process.append((i, data))
                    except json.JSONDecodeError:
                        logger.warning(
                            f"[orchestrator] Skipping invalid JSON at line {i} "
                            f"in {state.file_path}"
                        )
        except Exception as e:
            return f"Error reading candidate file: {e}", 0.0

        if not candidates_to_process:
            return self._build_feedback_report(
                state, final=True,
                note="No more valid candidates in file.",
            ), 0.0

        state.next_line = candidates_to_process[-1][0] + 1

        # Track completion for this batch
        batch_done = asyncio.Event()
        batch_completed = [0]
        batch_total_cost = [0.0]

        async def _process_one(line_idx: int, candidate_data: Dict) -> None:
            """Process a single candidate through the row generator."""
            async with self._processing_semaphore:
                # Check if we should stop
                if self.stop_checker and self.stop_checker():
                    return
                rows_done = self._generation_stats.get("rows_generated", 0)
                if rows_done >= self.num_samples:
                    return

                state.in_flight += 1
                try:
                    # Apply preset_fields to build candidate values
                    candidate_values = json.dumps(candidate_data, ensure_ascii=False)

                    # Build a Candidate object for the row generator
                    candidate = Candidate(
                        values=candidate_values,
                        source_id=f"file:{Path(state.file_path).stem}",
                        source_context=state.note,
                        metadata={
                            "preset_fields": state.preset_fields,
                            "candidate_data": candidate_data,
                        },
                    )

                    result, cost, saved = await self._generate_row_fn(
                        candidate,
                        f"file:{Path(state.file_path).stem}",
                    )

                    batch_total_cost[0] += cost
                    state.process_cost += cost
                    state.processed += 1

                    if result.success:
                        state.rows += 1
                    elif result.skipped:
                        if getattr(result, "is_duplicate", False):
                            state.duplicates += 1
                        else:
                            state.skipped += 1
                            reason = getattr(result, "skip_reason", "")
                            if reason:
                                state.skip_reasons.append(reason[:150])
                                if len(state.skip_reasons) > 20:
                                    state.skip_reasons = state.skip_reasons[-10:]
                    else:
                        state.errors += 1

                except asyncio.CancelledError:
                    pass
                except Exception as e:
                    logger.error(
                        f"[orchestrator] Row gen error for line {line_idx}: {e}",
                        exc_info=True,
                    )
                    state.errors += 1
                    state.processed += 1
                finally:
                    state.in_flight = max(0, state.in_flight - 1)
                    batch_completed[0] += 1
                    # Signal when we've hit the checkin threshold
                    if batch_completed[0] >= checkin_after:
                        batch_done.set()
                    # Also signal if this was the last one
                    if batch_completed[0] >= len(candidates_to_process):
                        batch_done.set()

        # Launch all candidates as tasks
        tasks = []
        for line_idx, candidate_data in candidates_to_process:
            task = asyncio.create_task(_process_one(line_idx, candidate_data))
            tasks.append(task)
            self._active_tasks.add(task)
            task.add_done_callback(lambda t: self._active_tasks.discard(t))

        # Block until checkin_after candidates are done (or all done, or stopped)
        try:
            # Use a timeout to periodically check stop conditions
            while not batch_done.is_set():
                if self.stop_checker and self.stop_checker():
                    break
                rows_done = self._generation_stats.get("rows_generated", 0)
                if rows_done >= self.num_samples:
                    break
                try:
                    await asyncio.wait_for(batch_done.wait(), timeout=5.0)
                except asyncio.TimeoutError:
                    continue
        except asyncio.CancelledError:
            # Cancel remaining tasks
            for task in tasks:
                if not task.done():
                    task.cancel()
            raise

        self._maybe_checkpoint()

        # Determine if processing was stopped early
        rows_done = self._generation_stats.get("rows_generated", 0)
        is_final = (
            rows_done >= self.num_samples
            or (state.next_line >= state.total_lines and state.in_flight == 0)
        )

        extra_note = ""
        if rows_done >= self.num_samples:
            extra_note = "TARGET REACHED."
        elif self.stop_checker and self.stop_checker():
            extra_note = "Processing paused."

        return self._build_feedback_report(
            state, final=is_final, note=extra_note,
        ), batch_total_cost[0]

    def _build_feedback_report(
        self,
        state: _FileProcessingState,
        final: bool = False,
        note: str = "",
    ) -> str:
        """Build a feedback report for the orchestrator."""
        # Log activity for the batch
        self._log_activity(
            f"{state.rows} rows generated, {state.skipped} skipped"
            + (f", {state.duplicates} duplicates" if state.duplicates else "")
        )

        rows_done = self._generation_stats.get("rows_generated", 0)
        total_cost = self._generation_stats.get("total_cost", 0.0)
        total_cost += self._conversation.total_cost

        remaining_in_file = state.total_lines - state.next_line
        in_flight_str = f" ({state.in_flight} in flight)" if state.in_flight > 0 else ""

        lines = [
            f"Processed: {state.processed}/{state.total_lines}{in_flight_str}",
            f"Rows: {state.rows} | Skipped: {state.skipped} | "
            f"Dupes: {state.duplicates} | Errors: {state.errors}",
        ]

        if state.skip_reasons:
            recent = state.skip_reasons[-3:]
            lines.append(
                "Skip reasons: " + "; ".join(f'"{r}"' for r in recent)
            )

        avg_cost = (
            f"${state.process_cost / state.rows:.3f}"
            if state.rows > 0
            else "N/A"
        )
        lines.append(f"Avg cost: {avg_cost}/row")
        lines.append(
            f"Total project: ${total_cost:.2f} spent, "
            f"{rows_done}/{self.num_samples} rows"
        )

        if remaining_in_file > 0 and not final:
            lines.append(
                f"Remaining in file: {remaining_in_file} candidates"
            )
        elif remaining_in_file == 0 and state.in_flight == 0:
            lines.append("File fully processed.")

        if note:
            lines.append(note)

        return "\n".join(lines)

    # ── Checkpointing ─────────────────────────────────────────────────

    def _maybe_checkpoint(self, force: bool = False) -> None:
        if not self._on_checkpoint:
            return
        now = time.time()
        if not force and now - self._last_checkpoint_time < 15.0:
            return
        self._last_checkpoint_time = now
        try:
            self._on_checkpoint(self)
        except Exception as e:
            logger.warning(f"[orchestrator] checkpoint callback error: {e}")

    # ── Run ───────────────────────────────────────────────────────────

    async def run(self) -> AgentResult:
        """Run the V13 orchestrator.

        Simple flow: send initial message, let the conversation loop handle
        everything. Tool calls block and return results. The LLM decides
        what to do next between each call. When it has nothing left to do,
        it responds with text and the loop exits.
        """
        # Ensure workspace directories exist
        candidates_dir = self.workspace_dir / "candidates"
        candidates_dir.mkdir(parents=True, exist_ok=True)

        # Determine initial message
        if self._conversation.messages:
            # Resuming from checkpoint — pick up where we left off
            rows_done = self._generation_stats.get("rows_generated", 0)
            logger.info(
                f"[orchestrator] Resuming: {len(self._conversation.messages)} messages, "
                f"{rows_done}/{self.num_samples} rows"
            )
            resume_parts = [
                f"Resumed. {rows_done}/{self.num_samples} rows done so far."
            ]
            if self._current_file:
                fs = self._current_file
                remaining = fs.total_lines - fs.next_line
                resume_parts.append(
                    f"You were processing {Path(fs.file_path).name}: "
                    f"{fs.processed}/{fs.total_lines} done, "
                    f"{remaining} remaining. Call continue_processing to resume."
                )
            else:
                resume_parts.append("Continue where you left off.")
            initial_msg = " ".join(resume_parts)
        elif self.feedback_context:
            initial_msg = (
                "Begin. The user reviewed previous results and gave feedback "
                "(shown in system prompt). Research as needed and design a "
                "new approach."
            )
        elif self.resume_context:
            initial_msg = (
                "Begin. This is a resumed job — some rows already exist. "
                "Pick up where the previous run left off."
            )
        else:
            initial_msg = (
                "Begin. Read the conversation and schema, figure out what the "
                "user wants, then find and submit candidates."
            )

        def _exit_condition() -> bool:
            if self._finish_requested:
                return True
            rows_done = self._generation_stats.get("rows_generated", 0)
            if rows_done >= self.num_samples:
                return True
            return False

        try:
            result = await self._conversation.send(
                initial_msg,
                exit_condition=_exit_condition,
            )
            logger.info(
                f"[orchestrator] Finished: "
                f"{self._generation_stats.get('rows_generated', 0)}/{self.num_samples} rows, "
                f"${self._conversation.total_cost:.4f} orchestrator cost"
            )
            return result or AgentResult(text="Orchestrator finished.")

        except asyncio.CancelledError:
            logger.info("[orchestrator] Cancelled")
            return AgentResult(
                text="Orchestrator cancelled.",
                cost_usd=self._conversation.total_cost,
                stopped=True,
            )
        finally:
            # Wait for any in-flight row generation tasks
            await self._drain_active_tasks()
            self._maybe_checkpoint(force=True)

    async def _drain_active_tasks(self) -> None:
        """Wait for in-flight row generation tasks to complete."""
        if not self._active_tasks:
            return
        logger.info(
            f"[orchestrator] Draining {len(self._active_tasks)} in-flight tasks"
        )
        pending = list(self._active_tasks)
        try:
            done, still_pending = await asyncio.wait(pending, timeout=30.0)
        except Exception:
            still_pending = pending
        for task in still_pending:
            task.cancel()
        if still_pending:
            await asyncio.gather(*still_pending, return_exceptions=True)

    # ── State export/restore ──────────────────────────────────────────

    @property
    def cost_usd(self) -> float:
        return self._conversation.total_cost

    def export_state(self) -> Dict[str, Any]:
        """Export full state for checkpointing.

        For file processing: rolls next_line back by in_flight count so
        candidates that were mid-generation get re-dispatched on resume.
        Dedup catches any that actually completed before the kill.
        """
        file_state = None
        if self._current_file:
            fs = self._current_file
            # Roll back next_line so in-flight candidates are re-processed
            safe_next_line = max(0, fs.next_line - fs.in_flight)
            file_state = {
                "file_path": fs.file_path,
                "note": fs.note,
                "preset_fields": fs.preset_fields,
                "total_lines": fs.total_lines,
                "next_line": safe_next_line,
                "processed": fs.processed,
                "rows": fs.rows,
                "skipped": fs.skipped,
                "duplicates": fs.duplicates,
                "errors": fs.errors,
                "process_cost": fs.process_cost,
                "skip_reasons": fs.skip_reasons[-10:],
            }

        return {
            "orchestrator_conversation": {
                "messages": list(self._conversation.messages),
                "total_cost": self._conversation.total_cost,
                "total_turns": self._conversation.total_turns,
            },
            "generation_stats": dict(self._generation_stats),
            "web_research_counter": self._web_harvest_counter,
            "bu_extract_counter": self._bu_extract_counter,
            "apify_run_counter": self._apify_run_counter,
            "candidates_harvested": self._candidates_harvested,
            "candidates_submitted": self._candidates_submitted,
            "current_file": file_state,
            "activity_log": self._activity_log[-50:],
        }

    def restore_state(self, state: Dict[str, Any]) -> None:
        """Restore state from checkpoint. Call BEFORE run()."""
        conv = state.get("orchestrator_conversation")
        if conv:
            self._conversation.messages = conv["messages"]
            self._conversation.total_cost = conv.get("total_cost", 0.0)
            self._conversation.total_turns = conv.get("total_turns", 0)
            logger.info(
                f"[orchestrator] Restored conversation: "
                f"{len(conv['messages'])} messages"
            )

        self._web_harvest_counter = state.get("web_research_counter", 0)
        self._bu_extract_counter = state.get("bu_extract_counter", 0)
        self._apify_run_counter = state.get("apify_run_counter", 0)
        self._candidates_harvested = state.get("candidates_harvested", 0)
        self._candidates_submitted = state.get("candidates_submitted", 0)
        self._activity_log = state.get("activity_log", [])

        saved_stats = state.get("generation_stats", {})
        for key in ("skipped", "errors", "total_cost"):
            if key in saved_stats:
                self._generation_stats[key] = saved_stats[key]

        # Restore file processing state
        file_state = state.get("current_file")
        if file_state:
            self._current_file = _FileProcessingState(
                file_path=file_state["file_path"],
                note=file_state["note"],
                preset_fields=file_state.get("preset_fields", {}),
                total_lines=file_state.get("total_lines", 0),
                next_line=file_state.get("next_line", 0),
                processed=file_state.get("processed", 0),
                rows=file_state.get("rows", 0),
                skipped=file_state.get("skipped", 0),
                duplicates=file_state.get("duplicates", 0),
                errors=file_state.get("errors", 0),
                process_cost=file_state.get("process_cost", 0.0),
                skip_reasons=file_state.get("skip_reasons", []),
            )
            logger.info(
                f"[orchestrator] Restored file state: "
                f"{self._current_file.processed}/{self._current_file.total_lines} "
                f"processed from {self._current_file.file_path}"
            )

    async def cleanup(self) -> None:
        """Cancel active tasks and close resources."""
        await self._drain_active_tasks()
        if self._sandbox_impl:
            try:
                await self._sandbox_impl.cleanup()
            except Exception:
                pass
