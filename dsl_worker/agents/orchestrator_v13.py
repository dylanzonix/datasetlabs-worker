"""
Orchestrator agent — V13 architecture.

Key changes from V12:
- No harvesters. Orchestrator directly controls candidate generation.
- Blocking tool calls. No timer-based check-in loop.
- File-based candidate flow. All tools write JSONL to /workspace/candidates/.
- submit_candidates / continue_processing for row generation with feedback.
- web_research subagent for multi-page web investigation.
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
# Dataset Builder

You build structured datasets by finding candidates from various sources, \
then processing them into rows. Your goal: produce {num_samples} high-quality \
rows at the lowest possible cost.

## How It Works

1. **Find candidates.** Use APIs, web research, code execution, or BU browser \
to discover entities that match the user's request. All candidate-producing tools \
write results to files — you never see raw data in context, just summaries and \
file paths.

2. **Prep candidates.** Use code_exec to inspect, filter, dedupe, merge, or \
transform candidate files as needed. Candidates are messy — clean them before \
submitting.

3. **Submit for processing.** Call submit_candidates with the file, a note for \
the row generator, and a preset_fields mapping. The system processes candidates \
into rows (10 concurrent), checking for duplicates. You get a feedback report \
after a batch completes.

4. **Iterate.** Based on feedback (skip rates, dupe rates, cost), adjust your \
approach. Continue processing the same file, find new sources, or refine your \
candidate prep.

## Your Tools

**code_exec(script)** — Run Python in a sandbox. Files at /workspace/uploads/ \
(user files, read-only) and /workspace/candidates/ (your output). pandas, json, \
csv, openpyxl, pdfplumber available. Use to inspect files, transform data, \
merge sources, filter candidates.

**web_research(query, candidate_description)** — Spawn a research agent that \
searches the web, opens pages, and yields candidates to a file. Returns \
summary + file path. ~30s, moderate cost. Good for finding entities on the \
open web.

**bu_extract(task)** — Delegate to a BU cloud browser agent. The agent \
navigates a real browser, handles CAPTCHAs and anti-bot. Returns summary + \
file path. 1-5 min, EXPENSIVE. Rules: \
(1) LAST RESORT — only after APIs and web_research failed. Never use BU for \
sites you have an API for (Google Maps, Apollo, YouTube). \
(2) ONE specific URL, ONE specific extraction task. Not "search 10 queries." \
(3) Keep the task tightly scoped — extract data from a single page or a \
single paginated list. If you need data from 5 sites, make 5 separate calls. \
(4) Include the exact URL in the task.

**submit_candidates(file, note, preset_fields, checkin_after)** — Submit a \
JSONL file for row generation. Each line is a JSON object (candidate). \
preset_fields maps schema columns to candidate fields for pre-filling. \
note tells the row generator where data came from and what to watch for. \
Blocks until checkin_after candidates are processed, then returns a feedback \
report.

**continue_processing(checkin_after)** — Resume processing remaining \
candidates from the last submitted file. Same blocking behavior and \
feedback report.

{integration_tools_section}\

**finish(reason)** — Abort the job. ONLY for genuinely impossible tasks — all \
feasible approaches exhausted, 100% certainty there's no viable path forward. \
Do NOT call this when the target is reached or credits run out — the system \
handles those automatically.

## Strategy

**Start small, then scale.** Find a small set of candidates (~10-20), submit \
them as a test batch with checkin_after=5. Read the feedback. If the skip rate \
is high, fix your candidate selection or preset_fields mapping before scaling \
up. If cost per row is reasonable, scale up aggressively.

**Understand the request.** Read the conversation carefully. The user might \
want specific entities (companies, people, restaurants) or broad categories. \
Match your sourcing strategy to what they need.

**Use the cheapest source first:**
- Uploaded files are free — if the user uploaded data, that IS your candidate \
list. Inspect with code_exec, prep, and submit.
- Apollo is free for search (B2B contacts/companies). Use it when the request \
is business-oriented.
- Google Maps is cheap and fast for local businesses.
- web_research is moderate — good for open web entities.
- bu_extract is expensive — only when nothing else works.

**Submit early, don't hoard.** Don't wait to gather everything before \
submitting. The feedback from processed rows is more valuable than a \
perfect candidate set. Submit what you have, read the feedback report, \
then gather more if needed. You can always submit additional batches.

**Candidates don't need to be complete.** A candidate just identifies the \
entity — a company name, a URL, a person's name. It does NOT need all \
schema columns filled. Row generators handle enrichment: finding contact \
info, verifying details, filling missing fields. Your job is to find \
entities, not to build complete rows. Don't try to enrich candidates \
before submitting — that's the row generator's job.

**Prep matters, but don't over-prep.** Use code_exec to filter obvious \
junk before submitting, but don't spend multiple turns perfecting \
candidates. A quick filter and submit beats a thorough analysis that \
delays row generation.

**preset_fields saves money.** If your candidates already have data that maps \
directly to schema columns, set those in preset_fields. The row generator \
won't need to re-research that information, cutting cost per row significantly. \
But don't delay submission to pre-fill — submit what you have.

**Use real data.** Default to finding real data through research, APIs, and \
web sources — not generating content from your own knowledge. Exceptions: \
tasks where LLM judgment is the point (scoring, classification, translation) \
or where the data is common knowledge that doesn't need sourcing.

**Cost is in dollars.** Every tool reports its cost. Track it.

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

Target: {num_samples} rows.{resume_section}
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
        feedback_context: Optional[Dict[str, Any]] = None,
        resume_context: Optional[Dict[str, Any]] = None,
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
        self.youtube_client = youtube_client
        self.feedback_context = feedback_context
        self.resume_context = resume_context

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
        self._web_research_counter: int = 0
        self._bu_extract_counter: int = 0

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
            content = msg.get("content", "")
            ts = msg.get("created_at", "")
            if ts:
                parts.append(f"[{ts}] **{role}**: {content}")
            else:
                parts.append(f"**{role}**: {content}")
        return "\n\n".join(parts)

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
        """Build the integration tools section of the system prompt."""
        sections = []

        if self.apollo_client:
            sections.append(
                "**apollo_search(...)** — Search Apollo.io's 210M+ contact database. "
                "FREE — no credits. Returns summary + file path. Great for B2B "
                "contacts and company discovery.\n\n"
                "**apollo_search_companies(...)** — Search Apollo.io's 30M+ company "
                "database. Returns summary + file path.\n"
            )

        if self.google_maps_client:
            sections.append(
                "**google_maps_search(query, page_token)** — Search Google Maps for "
                "local businesses. Returns summary + file path. Fast and cheap "
                "(~$0.003/search). Better than web research for local business data.\n\n"
                "**google_maps_details(place_id)** — Full details for a place (phone, "
                "website, hours). Use after google_maps_search to enrich.\n"
            )

        if self.apify_client:
            sections.append(
                "**apify_search(query)** — Search 22,000+ pre-built web scrapers on "
                "Apify for specific sites (Upwork, LinkedIn, Reddit, Yelp, etc).\n\n"
                "**apify_actor_details(actor_id)** — Get full details including "
                "description, readme, and input schema. Always check this before "
                "running an actor to know what input to pass.\n\n"
                "**apify_run(actor_id, input)** — Run a scraper. Returns structured "
                "data written to file. Faster and cheaper than BU.\n\n"
                "Apify workflow: apify_search → apify_actor_details → apify_run.\n"
            )

        if not sections:
            return ""
        return "\n".join(sections) + "\n"

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

        return workspace_path

    # ── Tool registration ─────────────────────────────────────────────

    def _register_tools(self, registry: ToolRegistry) -> None:
        self._register_code_exec(registry)
        self._register_web_research(registry)
        self._register_bu_extract(registry)
        self._register_submit_candidates(registry)
        self._register_continue_processing(registry)
        self._register_finish(registry)

        if self.apollo_client:
            self._register_apollo_tools(registry)
        if self.google_maps_client:
            self._register_google_maps(registry)
        if self.apify_client:
            self._register_apify_tools(registry)

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
                                    (candidates_dir / fname).write_text(content, encoding="utf-8")
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

    # --- web_research ---

    def _register_web_research(self, registry: ToolRegistry) -> None:

        async def web_research(args: Dict) -> Tuple[str, float]:
            query = args.get("query", "")
            candidate_description = args.get("candidate_description", "")

            if not query:
                return "Error: query is required.", 0.0

            idx = self._web_research_counter
            self._web_research_counter += 1
            timestamp = int(time.time())
            filename = f"web_research_{idx}_{timestamp}.jsonl"

            total_cost = 0.0
            candidates_found = 0
            collected_lines: List[str] = []

            try:
                # Build the subagent
                sub_registry = ToolRegistry()
                sub_system_prompt = self._build_web_research_prompt(
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
                    label=f"web_research:{idx}",
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
                logger.info(f"[web_research:{idx}] cancelled")
            except Exception as e:
                logger.error(f"[web_research:{idx}] error: {e}", exc_info=True)
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

            return (
                f"Web research complete: {candidates_found} candidates found.\n"
                f"File: {output_path}\n"
                f"Cost: ${total_cost:.4f}{sample_str}"
            ), total_cost

        registry.add(
            name="web_research",
            description=(
                "Spawn a research agent that searches the web for a query, "
                "opens pages, and yields candidates to a file. Returns "
                "summary + file path. ~30s, moderate cost."
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
            handler=web_research,
        )

    def _build_web_research_prompt(
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
            name="bu_extract",
            description=(
                "Delegate to a BU cloud browser agent. LAST RESORT — never "
                "use for sites you have an API for. One URL, one task, tightly "
                "scoped. Handles CAPTCHAs, JavaScript, anti-bot. 1-5 min, expensive."
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

    # --- submit_candidates ---

    def _register_submit_candidates(self, registry: ToolRegistry) -> None:

        async def submit_candidates(args: Dict) -> Tuple[str, float]:
            file_path = args.get("file", "")
            note = args.get("note", "")
            preset_fields = args.get("preset_fields", {})
            checkin_after = args.get("checkin_after", 10)

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
                "Submit a JSONL file for row generation. Each line is a candidate. "
                "Blocks until checkin_after candidates are processed, then returns "
                "a feedback report with outcomes."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "file": {
                        "type": "string",
                        "description": "Path to JSONL file in /workspace/candidates/.",
                    },
                    "note": {
                        "type": "string",
                        "description": (
                            "Handoff note for the row generator: where the data "
                            "came from, what's trustworthy, what to look for, "
                            "any heads up."
                        ),
                    },
                    "preset_fields": {
                        "type": "object",
                        "description": (
                            "Mapping of {schema_column: candidate_field} to pre-fill "
                            "row columns from candidate data. E.g. "
                            '{"Company Name": "company_name", "Website": "website"}'
                        ),
                        "additionalProperties": {"type": "string"},
                    },
                    "checkin_after": {
                        "type": "integer",
                        "description": (
                            "How many processed candidates before returning a "
                            "feedback report. Start with 5-10 for test batches."
                        ),
                    },
                },
                "required": ["file", "note"],
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

    # --- Apollo tools ---

    def _register_apollo_tools(self, registry: ToolRegistry) -> None:

        async def apollo_search(args: Dict) -> Tuple[str, float]:
            page = args.get("page", 1)
            timestamp = int(time.time())
            filename = f"apollo_people_{timestamp}.jsonl"

            try:
                people, total = await self.apollo_client.search_people(
                    person_titles=args.get("person_titles") or None,
                    person_seniorities=args.get("person_seniorities") or None,
                    person_locations=args.get("person_locations") or None,
                    person_names=args.get("person_names") or None,
                    contact_email_status=args.get("contact_email_status") or None,
                    department_ids=args.get("department_ids") or None,
                    include_similar_titles=args.get("include_similar_titles"),
                    organization_keywords=args.get("organization_keywords") or None,
                    organization_name=args.get("organization_name") or None,
                    organization_locations=args.get("organization_locations") or None,
                    organization_not_locations=args.get("organization_not_locations") or None,
                    organization_num_employees_ranges=args.get("employee_ranges") or None,
                    organization_ids=args.get("organization_ids") or None,
                    organization_domains=args.get("organization_domains") or None,
                    organization_revenue_ranges=args.get("revenue_ranges") or None,
                    industry_tag_ids=args.get("industry_tag_ids") or None,
                    technology_uids=args.get("technology_uids") or None,
                    q_keywords=args.get("q_keywords") or None,
                    per_page=100,
                    page=page,
                )
            except Exception as e:
                return f"Apollo search error: {e}", 0.0

            if not people:
                return (
                    f"Apollo search returned 0 results. "
                    f"Try broader filters or different keywords."
                ), 0.0

            # Build JSONL content
            lines = []
            for person in people:
                org = person.get("organization") or {}
                record = {
                    "apollo_id": person.get("id"),
                    "name": person.get("name"),
                    "first_name": person.get("first_name"),
                    "last_name": person.get("last_name"),
                    "title": person.get("title"),
                    "headline": person.get("headline"),
                    "linkedin_url": person.get("linkedin_url"),
                    "city": person.get("city"),
                    "state": person.get("state"),
                    "country": person.get("country"),
                    "seniority": person.get("seniority"),
                    "departments": person.get("departments"),
                    "organization_name": org.get("name"),
                    "organization_id": person.get("organization_id"),
                }
                lines.append(json.dumps(record, ensure_ascii=False))

            output_path = await self._write_candidates_file(
                filename, "\n".join(lines) + "\n"
            )

            # Pagination info
            if total > 0:
                total_pages = min((total + 99) // 100, 500)
                pagination = f"page {page}/{total_pages} ({total:,} total matches)"
            else:
                pagination = f"page {page} ({len(people)} returned)"

            # Sample
            sample = people[0] if people else {}
            sample_str = ""
            if sample:
                org = sample.get("organization") or {}
                sample_str = (
                    f"\nSample: {sample.get('name', '?')} — "
                    f"{sample.get('title', '?')} at "
                    f"{org.get('name', '?')}"
                )

            return (
                f"Apollo people search: {len(people)} results, {pagination}.\n"
                f"File: {output_path}\n"
                f"For more pages: apollo_search(..., page={page + 1}){sample_str}"
            ), 0.0

        registry.add(
            name="apollo_search",
            description=(
                "Search Apollo.io's 210M+ contact database. FREE — no credits. "
                "Results written to file."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "person_titles": {
                        "type": "array", "items": {"type": "string"},
                        "description": "Job titles (e.g. ['CEO', 'VP Marketing'])",
                    },
                    "person_seniorities": {
                        "type": "array", "items": {"type": "string"},
                        "description": "Seniority: c_suite, founder, vp, director, manager, senior, entry",
                    },
                    "person_locations": {
                        "type": "array", "items": {"type": "string"},
                        "description": "Person locations (e.g. ['California, US'])",
                    },
                    "person_names": {
                        "type": "array", "items": {"type": "string"},
                        "description": "Search by individual names",
                    },
                    "contact_email_status": {
                        "type": "array", "items": {"type": "string"},
                        "description": "Email availability: 'verified', 'guessed', 'unavailable'",
                    },
                    "department_ids": {
                        "type": "array", "items": {"type": "string"},
                        "description": "Department classification IDs",
                    },
                    "include_similar_titles": {
                        "type": "boolean",
                        "description": "Include similar/related job titles",
                    },
                    "organization_keywords": {
                        "type": "array", "items": {"type": "string"},
                        "description": "Industry/keyword tags (e.g. ['healthcare', 'fintech'])",
                    },
                    "organization_name": {
                        "type": "string",
                        "description": "Company name search (partial match)",
                    },
                    "organization_locations": {
                        "type": "array", "items": {"type": "string"},
                        "description": "Company HQ locations",
                    },
                    "organization_not_locations": {
                        "type": "array", "items": {"type": "string"},
                        "description": "Exclude companies in these locations",
                    },
                    "employee_ranges": {
                        "type": "array", "items": {"type": "string"},
                        "description": "Employee count: '1-10', '11-50', '51-200', '201-500', '501-1000', etc.",
                    },
                    "organization_ids": {
                        "type": "array", "items": {"type": "string"},
                        "description": "Apollo organization IDs",
                    },
                    "organization_domains": {
                        "type": "array", "items": {"type": "string"},
                        "description": "Company domains (e.g. ['apollo.io'])",
                    },
                    "revenue_ranges": {
                        "type": "array", "items": {"type": "string"},
                        "description": "Annual revenue ranges",
                    },
                    "industry_tag_ids": {
                        "type": "array", "items": {"type": "string"},
                        "description": "Industry category IDs",
                    },
                    "technology_uids": {
                        "type": "array", "items": {"type": "string"},
                        "description": "Technology stack UIDs",
                    },
                    "q_keywords": {
                        "type": "string",
                        "description": "Free text keyword search",
                    },
                    "page": {
                        "type": "integer",
                        "description": "Page number (1-500). 100 results per page.",
                    },
                },
            },
            handler=apollo_search,
        )

        async def apollo_search_companies(args: Dict) -> Tuple[str, float]:
            page = args.get("page", 1)
            timestamp = int(time.time())
            filename = f"apollo_companies_{timestamp}.jsonl"

            try:
                orgs, total = await self.apollo_client.search_companies(
                    organization_keywords=args.get("keywords") or None,
                    organization_name=args.get("name") or None,
                    organization_locations=args.get("locations") or None,
                    organization_not_locations=args.get("not_locations") or None,
                    organization_num_employees_ranges=args.get("employee_ranges") or None,
                    organization_revenue_ranges=args.get("revenue_ranges") or None,
                    organization_latest_funding_stage_cd=args.get("funding_stages") or None,
                    technology_uids=args.get("technology_uids") or None,
                    website_urls=args.get("website_urls") or None,
                    industry_tag_ids=args.get("industry_tag_ids") or None,
                    founded_year_min=args.get("founded_year_min"),
                    founded_year_max=args.get("founded_year_max"),
                    publicly_traded=args.get("publicly_traded"),
                    per_page=100,
                    page=page,
                )
            except Exception as e:
                return f"Apollo company search error: {e}", 0.0

            if not orgs:
                return "Apollo company search returned 0 results.", 0.0

            # Build JSONL content
            lines = []
            for org in orgs:
                record = {
                    "apollo_org_id": org.get("id"),
                    "company_name": org.get("name"),
                    "website": org.get("website_url"),
                    "industry": org.get("industry"),
                    "keywords": org.get("keywords"),
                    "estimated_employees": org.get("estimated_num_employees"),
                    "city": org.get("city"),
                    "state": org.get("state"),
                    "country": org.get("country"),
                    "linkedin_url": org.get("linkedin_url"),
                    "short_description": org.get("short_description"),
                    "founded_year": org.get("founded_year"),
                    "annual_revenue": org.get("annual_revenue"),
                    "total_funding": org.get("total_funding"),
                    "latest_funding_stage": org.get("latest_funding_stage"),
                }
                lines.append(json.dumps(record, ensure_ascii=False))

            output_path = await self._write_candidates_file(
                filename, "\n".join(lines) + "\n"
            )

            if total > 0:
                total_pages = min((total + 99) // 100, 500)
                pagination = f"page {page}/{total_pages} ({total:,} total matches)"
            else:
                pagination = f"page {page} ({len(orgs)} returned)"

            sample = orgs[0] if orgs else {}
            sample_str = ""
            if sample:
                sample_str = (
                    f"\nSample: {sample.get('name', '?')} — "
                    f"{sample.get('industry', '?')}, "
                    f"{sample.get('estimated_num_employees', '?')} employees"
                )

            return (
                f"Apollo company search: {len(orgs)} companies, {pagination}.\n"
                f"File: {output_path}\n"
                f"For more pages: apollo_search_companies(..., page={page + 1})"
                f"{sample_str}"
            ), 0.0

        registry.add(
            name="apollo_search_companies",
            description=(
                "Search Apollo.io's 30M+ company database. "
                "Results written to file."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "keywords": {"type": "array", "items": {"type": "string"}, "description": "Industry/keyword tags"},
                    "name": {"type": "string", "description": "Company name search"},
                    "locations": {"type": "array", "items": {"type": "string"}, "description": "Company HQ locations"},
                    "not_locations": {"type": "array", "items": {"type": "string"}, "description": "Exclude locations"},
                    "employee_ranges": {"type": "array", "items": {"type": "string"}, "description": "Employee count ranges"},
                    "revenue_ranges": {"type": "array", "items": {"type": "string"}, "description": "Revenue ranges"},
                    "funding_stages": {"type": "array", "items": {"type": "string"}, "description": "Funding stage codes"},
                    "technology_uids": {"type": "array", "items": {"type": "string"}, "description": "Tech stack UIDs"},
                    "website_urls": {"type": "array", "items": {"type": "string"}, "description": "Filter by website URLs"},
                    "industry_tag_ids": {"type": "array", "items": {"type": "string"}, "description": "Industry IDs"},
                    "founded_year_min": {"type": "integer", "description": "Earliest founding year"},
                    "founded_year_max": {"type": "integer", "description": "Latest founding year"},
                    "publicly_traded": {"type": "boolean", "description": "Publicly traded only"},
                    "page": {"type": "integer", "description": "Page number (1-500)"},
                },
            },
            handler=apollo_search_companies,
        )

    # --- Google Maps ---

    def _register_google_maps(self, registry: ToolRegistry) -> None:

        async def google_maps_search(args: Dict) -> Tuple[str, float]:
            query = args.get("query", "")
            page_token = args.get("page_token")

            if not query and not page_token:
                return "Error: query is required.", 0.0

            timestamp = int(time.time())
            filename = f"google_maps_{timestamp}.jsonl"

            try:
                result = await self.google_maps_client.text_search(
                    query=query, page_token=page_token,
                )
            except Exception as e:
                return f"Google Maps search error: {e}", 0.0

            places = result.get("results", [])
            next_token = result.get("next_page_token")

            if not places:
                return "Google Maps search returned 0 results.", 0.0

            # Write raw API response — let orchestrator use whatever fields exist
            lines = []
            for place in places:
                lines.append(json.dumps(place, ensure_ascii=False, default=str))

            output_path = await self._write_candidates_file(
                filename, "\n".join(lines) + "\n"
            )

            sample = places[0] if places else {}
            sample_str = ""
            if sample:
                sample_str = (
                    f"\nSample: {sample.get('name', '?')} — "
                    f"{sample.get('formatted_address', '?')}"
                )
            keys_str = f"\nFields: {', '.join(sorted(sample.keys()))}" if sample else ""

            pagination_str = ""
            if next_token:
                pagination_str = (
                    f"\nMore results available: "
                    f"google_maps_search(query=\"{query}\", "
                    f"page_token=\"{next_token}\")"
                )

            return (
                f"Google Maps: {len(places)} businesses found.\n"
                f"File: {output_path}{keys_str}{sample_str}{pagination_str}"
            ), 0.0

        async def google_maps_details(args: Dict) -> Tuple[str, float]:
            place_id = args.get("place_id", "")

            if not place_id:
                return "Error: place_id is required.", 0.0

            try:
                details = await self.google_maps_client.place_details(place_id)
            except Exception as e:
                return f"Google Maps details error: {e}", 0.0

            if not details:
                return f"No details found for place_id: {place_id}", 0.0

            return json.dumps(details, ensure_ascii=False, default=str), 0.0

        registry.add(
            name="google_maps_search",
            description=(
                "Search Google Maps for local businesses. Returns basic info "
                "(name, address, place_id, rating, types). ~$0.003/search. "
                "Use google_maps_details for full info (phone, website, hours)."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query (e.g. 'coffee shops in Seattle')",
                    },
                    "page_token": {
                        "type": "string",
                        "description": "Pagination token from a previous search (for next page).",
                    },
                },
            },
            handler=google_maps_search,
        )

        registry.add(
            name="google_maps_details",
            description=(
                "Get full details for a Google Maps place by place_id. Returns "
                "phone, website, hours, URL, etc. ~$0.003/call. Use after "
                "google_maps_search to enrich specific places."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "place_id": {
                        "type": "string",
                        "description": "Google Maps place_id from a search result.",
                    },
                },
                "required": ["place_id"],
            },
            handler=google_maps_details,
        )

    # --- Apify tools ---

    def _register_apify_tools(self, registry: ToolRegistry) -> None:

        async def apify_search(args: Dict) -> Tuple[str, float]:
            query = args.get("query", "")
            if not query:
                return "Error: query is required.", 0.0

            try:
                results = await self.apify_client.search_actors(query, limit=5)
            except Exception as e:
                return f"Apify search error: {e}", 0.0

            if not results:
                return f"No Apify actors found for: {query}", 0.0

            lines = []
            for r in results:
                lines.append(
                    f"- **{r['actor_id']}** — {r['title']}\n"
                    f"  {r['description']}\n"
                    f"  Runs: {r['total_runs']:,} | Users: {r['total_users']:,}"
                )

            return (
                f"Found {len(results)} Apify actors for \"{query}\":\n\n"
                + "\n\n".join(lines)
                + "\n\nUse apify_actor_details(actor_id) to see what input an actor "
                + "expects, then apify_run(actor_id, input) to run it."
            ), 0.0

        async def apify_run(args: Dict) -> Tuple[str, float]:
            actor_id = args.get("actor_id", "")
            run_input = args.get("input", {})
            max_items = args.get("max_items")

            if not actor_id:
                return "Error: actor_id is required.", 0.0

            logger.info(f"[orchestrator] Running Apify actor: {actor_id}")

            try:
                result = await self.apify_client.run_actor(
                    actor_id,
                    run_input,
                    timeout=300,
                    max_items=max_items,
                )
            except Exception as e:
                return f"Apify run error: {e}", 0.0

            if result["status"] != "SUCCEEDED":
                return (
                    f"Apify actor {actor_id} failed: {result.get('error', result['status'])}"
                ), 0.0

            items = result["items"]
            cost = result.get("cost_usd", 0.0)

            if not items:
                return f"Apify actor {actor_id} returned 0 items.", cost

            # Write raw results to file
            timestamp = int(time.time())
            filename = f"apify_{actor_id.replace('/', '_')}_{timestamp}.jsonl"
            lines = [json.dumps(item, ensure_ascii=False, default=str) for item in items]
            output_path = await self._write_candidates_file(
                filename, "\n".join(lines) + "\n"
            )

            # Build summary
            sample = items[0]
            keys_str = ", ".join(sorted(sample.keys())[:10])
            sample_str = ""
            for key in ("title", "name", "Title", "Name"):
                if key in sample:
                    sample_str = f"\nSample: {sample[key]}"
                    break

            return (
                f"Apify {actor_id}: {len(items)} items returned.\n"
                f"File: {output_path}\n"
                f"Fields: {keys_str}\n"
                f"Cost: ${cost:.4f}{sample_str}"
            ), cost

        registry.add(
            name="apify_search",
            description=(
                "Search 22,000+ pre-built web scrapers on Apify. Returns actor "
                "names and descriptions. Use to discover scrapers for specific "
                "sites before running them."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query (e.g. 'upwork jobs', 'yelp reviews', 'linkedin profiles')",
                    },
                },
                "required": ["query"],
            },
            handler=apify_search,
        )

        async def apify_actor_details(args: Dict) -> Tuple[str, float]:
            actor_id = args.get("actor_id", "")
            if not actor_id:
                return "Error: actor_id is required.", 0.0

            try:
                details = await self.apify_client.get_actor_details(actor_id)
            except Exception as e:
                return f"Error getting actor details: {e}", 0.0

            if not details:
                return f"Actor not found: {actor_id}", 0.0

            parts = [
                f"**{details['title']}**",
                f"URL: {details['url']}",
                "",
                details["description"],
            ]
            if details.get("readme_summary"):
                parts.append("")
                parts.append(details["readme_summary"][:1000])
            if details.get("example_input"):
                parts.append("")
                parts.append(f"Example input: {details['example_input']}")

            return "\n".join(parts), 0.0

        registry.add(
            name="apify_actor_details",
            description=(
                "Get full details for an Apify actor — description, readme, "
                "and example input. Use after apify_search to understand what "
                "input parameters an actor expects before running it."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "actor_id": {
                        "type": "string",
                        "description": "Actor ID (e.g. 'neatrat/upwork-job-scraper')",
                    },
                },
                "required": ["actor_id"],
            },
            handler=apify_actor_details,
        )

        registry.add(
            name="apify_run",
            description=(
                "Run an Apify scraper. Find the right actor first with "
                "apify_search. Returns structured data written to file. "
                "Faster and cheaper than BU for sites with a pre-built scraper."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "actor_id": {
                        "type": "string",
                        "description": "Actor ID from apify_search (e.g. 'neatrat/upwork-job-scraper')",
                    },
                    "input": {
                        "type": "object",
                        "description": (
                            "Input JSON for the actor. Use apify_actor_details to "
                            "find what parameters the actor expects. Common patterns: "
                            "search URL scrapers take {\"searchUrl\": \"...\"}, "
                            "keyword scrapers take {\"searchQuery\": \"...\", \"maxItems\": N}."
                        ),
                    },
                    "max_items": {
                        "type": "integer",
                        "description": "Max items to return (optional).",
                    },
                },
                "required": ["actor_id", "input"],
            },
            handler=apify_run,
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
            initial_msg = (
                f"Resumed. {rows_done}/{self.num_samples} rows done so far. "
                f"Continue where you left off."
            )
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
        """Export full state for checkpointing."""
        file_state = None
        if self._current_file:
            fs = self._current_file
            file_state = {
                "file_path": fs.file_path,
                "note": fs.note,
                "preset_fields": fs.preset_fields,
                "total_lines": fs.total_lines,
                "next_line": fs.next_line,
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
            "web_research_counter": self._web_research_counter,
            "bu_extract_counter": self._bu_extract_counter,
            "current_file": file_state,
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

        self._web_research_counter = state.get("web_research_counter", 0)
        self._bu_extract_counter = state.get("bu_extract_counter", 0)

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
