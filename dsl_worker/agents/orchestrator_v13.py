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

## Your role

You are the **orchestrator**. You find the best sources of candidates, \
harvest them, and submit them for processing by row generators.

When you call submit_candidates, each candidate goes to a separate AI \
agent (row generator) that inspects it, researches missing info, fills \
schema columns, and decides row or skip. You get a feedback report back.

Row generators have: **web_search**, **apify**, **fullenrich** (people/ \
company search + verified email/phone), **apollo** (B2B enrichment), \
**google_maps**, **code_exec**, and **browser_use** (expensive last \
resort). Your instructions guide what they do — share what you've \
learned so they don't each independently rediscover the same things.

## Tools

{integration_tools_section}\
**web_harvest(query, candidate_description)** — Spawns an agent that \
googles, visits pages, extracts candidates to a file. ~$0.10-0.20/call. \
For harvesting entities only — use your built-in web_search for lookups.

**browser_use(task)** — Full cloud browser with anti-bot bypass, JS \
rendering, captcha solving. $0.10-0.50/call. Use ONLY when you have a \
specific URL that requires a real browser — the site blocks scraping, \
needs JS interaction, or has anti-bot protection, AND no Apify actor \
exists for it. If the data could be found from a different source that \
doesn't need a browser, use that instead.

**code_exec(script)** — Python sandbox. /workspace/uploads/ (read-only), \
/workspace/candidates/ (your output). pandas, json, csv available.

### Source priority
1. Uploaded files — free, inspect with code_exec
2. Integrations (Apify, FullEnrich, Apollo, Google Maps) — cheap, structured
3. web_search — cheap but not free (uses tokens), good for lookups
4. web_harvest — $0.10-0.20/call, for scraping when no integration exists
5. browser_use — $0.10-0.50/call, last resort for anti-bot/JS-heavy sites

Apify has 22,000+ scrapers covering social media, e-commerce, real estate, \
job boards, news, and more. Always check apify_search before resorting to \
web_harvest or browser_use.

### Cost awareness
- FullEnrich emails: ~$0.05 each (worth it — high find rate)
- FullEnrich phones: ~$0.55 each (expensive — try web_search first)
- Apify actors: usually <$0.01/result
- browser_use: $0.10-0.50/session (avoid unless truly necessary)

## Submission

**submit_candidates(file, instructions, checkin_after)** — Send a JSONL \
file to row generators. Returns a feedback report.

**continue_processing(checkin_after)** — Resume the current file.

**finish(reason)** — Only for genuinely impossible tasks.

## How to work

### Step 1: Identify your sources (~2-3 turns)

Quickly figure out the best source for candidates. Check uploaded files \
first, then use your integration tools (apify_search, fullenrich, apollo, \
google_maps) to see what's available. A couple of searches is enough — \
don't spend turns googling for how tools work, just call them.

For larger targets (100+), prefer scalable live sources (integrations) \
over static articles. For smaller targets, a single dense source is fine.

### Step 2: Harvest, prep instructions, submit

Harvest a test batch and submit. If row generators will need to do \
non-trivial research per candidate, try a couple yourself first — test \
the integrations, see what comes back, find what works. Then share what \
you learned in the instructions so row generators don't each rediscover \
the same things. If processing is straightforward, just describe what's \
in the data and what needs filling.

### Step 3: Scale

Read feedback. Good conversion → scale up. High skips → adjust source \
or filtering. Source runs dry → pivot.

**You produce rows by submitting candidates.** Don't over-explore.

### Principles

- **Filter programmatically when you can.** code_exec to remove obvious \
non-matches before submitting saves wasted row generators.
- **Use real data.** Don't fabricate from your own knowledge.
- **Costs add up.** FullEnrich phones $0.55 each. browser_use $0.10-0.50. \
Use intentionally.

<scenarios>
Scenario 1 — Candidates from an integration, straightforward processing:
Task: 300 Airbnb listings in Barcelona with 2+ bedrooms.
- apify_search("airbnb") → found scraper
- call_actor(startUrl="https://airbnb.com/s/Barcelona/homes?adults=1&bedrooms=2", maxItems=30) → 30 listings
- All schema fields are in the data
- submit_candidates(instructions="Airbnb listings from Apify. All fields present — title, price, location, host, bedrooms, URL. No research needed. Skip if bedrooms < 2.")
- Good conversion → scale up

Scenario 2 — Row generators need research, orchestrator figures out workflow first:
Task: 100 Shopify stores selling pet products with owner email.
- apify_search("shopify store scraper") → found actor
- call_actor with maxItems=5 → got store name, URL, products
- Schema needs owner email — not in data
- Try one: web_search for store owner → found on the About page
- Try fullenrich.enrich_contacts → got email
- submit_candidates(instructions="Shopify stores from Apify. Store name/URL/products are in the data. For owner email: check About page via web search for owner name, then fullenrich.enrich_contacts. Skip if not primarily pet products.")

Scenario 3 — Uploaded file:
Task: Enrich 200 companies from user's CSV with HQ address and phone.
- code_exec to inspect file → 200 rows with name, website, industry
- submit_candidates(instructions="User's company list, all fields trustworthy. For HQ address + phone: try apollo.enrich_company with the domain. If no result, web search the company name.")

Scenario 4 — No integration, web-based sourcing:
Task: 40 coworking spaces in Tokyo with pricing info.
- google_maps_search("coworking space Tokyo") → 20 results
- submit_candidates(instructions="Google Maps results. Name, address, rating, phone are in the data. For pricing: check their website via web search. Skip if not actually a coworking space.")
- Search more areas → submit → keep going

Scenario 5 — Large scale, need scalable source:
Task: 1000 SaaS companies in the US with 50-200 employees.
- fullenrich.search_companies(industries=["Software Development"], locations=["United States"], headcount_min=50, headcount_max=200, max_results=50) → 50 companies
- submit_candidates(instructions="Companies from FullEnrich. Name, domain, industry, headcount, location all present. For contact info: use apollo.enrich_company with the domain.")
- Good conversion → scale to max_results=500, keep going
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
        """Build a status line injected after every orchestrator turn."""
        rows = self._generation_stats.get("rows_generated", 0)
        cost = self._generation_stats.get("total_cost", 0.0)
        harvested = self._candidates_harvested
        submitted = self._candidates_submitted

        line = (
            f"[Status] {rows}/{self.num_samples} rows | "
            f"{harvested} candidates harvested | "
            f"{submitted} submitted | "
            f"${cost:.2f} spent"
        )

        if harvested > 0 and submitted == 0 and rows == 0:
            line += (
                "\n→ You have candidates but haven't submitted any. "
                "Call submit_candidates to start producing rows."
            )

        return line

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

        # Track candidates harvested (count non-empty lines)
        self._candidates_harvested += sum(
            1 for line in content.splitlines() if line.strip()
        )

        return workspace_path

    # ── Tool registration ─────────────────────────────────────────────

    def _register_tools(self, registry: ToolRegistry) -> None:
        # Core tools (always loaded)
        self._register_code_exec(registry)
        self._register_web_harvest(registry)
        self._register_bu_extract(registry)
        self._register_submit_candidates(registry)
        self._register_continue_processing(registry)
        self._register_finish(registry)

        # tool_search not supported on Azure OpenAI (returns 400 even with gpt-5.4).
        # Namespace tools load upfront instead of deferred.

        # Integration namespaces
        if self.apify_client:
            from dsl_worker.agents.integrations.apify import register_apify_namespace
            from dsl_worker.config import settings
            register_apify_namespace(
                registry, self.apify_client, settings.apify_api_key,
                self.workspace_dir,
            )
        if self.apollo_client:
            from dsl_worker.agents.integrations.apollo import register_apollo_namespace
            from dsl_worker.config import settings as _apollo_settings
            register_apollo_namespace(
                registry, self.apollo_client, self.workspace_dir,
                cost_per_credit=_apollo_settings.apollo_cost_per_credit,
            )
        if self.google_maps_client:
            from dsl_worker.agents.integrations.google_maps import register_google_maps_namespace
            from dsl_worker.config import settings as _settings
            register_google_maps_namespace(
                registry, _settings.google_api_key, self.workspace_dir,
            )
        if hasattr(self, 'fullenrich_client') and self.fullenrich_client:
            from dsl_worker.agents.integrations.fullenrich import register_fullenrich_namespace
            from dsl_worker.config import settings as _fe_settings
            register_fullenrich_namespace(
                registry, self.fullenrich_client, self.workspace_dir,
                cost_per_credit=_fe_settings.fullenrich_cost_per_credit,
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

    # --- submit_candidates ---

    def _register_submit_candidates(self, registry: ToolRegistry) -> None:

        async def submit_candidates(args: Dict) -> Tuple[str, float]:
            file_path = args.get("file", "")
            note = args.get("instructions", "") or args.get("note", "")
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

            self._candidates_submitted += total_lines

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
            "web_research_counter": self._web_harvest_counter,
            "bu_extract_counter": self._bu_extract_counter,
            "apify_run_counter": self._apify_run_counter,
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

        self._web_harvest_counter = state.get("web_research_counter", 0)
        self._bu_extract_counter = state.get("bu_extract_counter", 0)
        self._apify_run_counter = state.get("apify_run_counter", 0)

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
