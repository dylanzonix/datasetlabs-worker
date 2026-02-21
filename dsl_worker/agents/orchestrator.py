"""
Orchestrator agent — the brain that coordinates dataset generation.

V3: Loop-based orchestrator with direct research tools, source management,
work item creation, and human-in-the-loop via ask_user().

No more seeds, generators, or forced pipeline phases. The orchestrator:
1. Researches the domain (directly or via subagents)
2. Accumulates sources via save_source
3. Creates work items (instruction + schema per row)
4. Monitors generation progress
5. Talks to the user mid-generation via ask_user
6. Loops until target reached, then done()
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional

from dsl_worker.agents.base import AgentConversation, AgentResult
from dsl_worker.agents.tools import ToolRegistry
from dsl_worker.agents.research import ResearchAgent
from dsl_worker.billing.tracked_client import TrackedOpenAIClient
from dsl_worker.phases.source_manager import SourceManager

logger = logging.getLogger(__name__)


# Max chars for read_file results
READ_FILE_LIMIT = 30_000


ORCHESTRATOR_SYSTEM_PROMPT = """\
You are the orchestrator for a dataset generation system.

## Your mission

A user described a dataset through a consultation chat (below). Your job is to
build it by: researching the domain, accumulating sources, creating work items
(one per row), and monitoring generation quality.

## Source system

Sources are files saved to /workspace/sources/ with metadata in a manifest.
Use save_source() to accumulate research material. Subagents (via run_subagent)
can also save sources. Row generators read the manifest to find relevant sources.

Save broadly — it's better to have extra sources than to miss useful material.
Good sources include: wiki pages, expert articles, reference data, code samples,
curated lists, structured datasets. Authority scores: wiki/docs=0.9, expert=0.7,
forum/community=0.5, random=0.3.

## How to work (loop, not pipeline)

1. **Understand the task.** Read the conversation history and any uploaded files.
2. **Research.** Use your direct tools (brave_search, open) for domain understanding.
   Delegate to subagents (run_subagent) for breadth — e.g. "research X and save
   relevant sources." Subagents can save_source too.
3. **Create work items.** Call create_work_items() with instructions for the row
   generator. Each work item produces one row. The instruction tells the row
   generator exactly what to do — reference source files by path/tag, specify
   research tasks, define the transformation.
4. **Monitor.** Call check_progress() to see how generation is going. If error
   rates are high, adjust instructions. If more rows are needed, create more
   work items.
5. **Sample and consult.** After ~20 rows are generated, use ask_user() to show
   sample rows and get feedback. Adjust instructions or sources based on feedback.
6. **Repeat.** Steps 2-5 can overlap. Start generating while still researching
   other areas. Create more work items as you discover new sources.
7. **Complete.** Call done() when the target is reached.

## Work items

Each work item = {{instruction, schema (optional)}}

The instruction tells the row generator exactly what to do. Reference source
files and tags in the instruction. The row generator has tools: read_file,
brave_search, open, code_exec, set_column, submit_row, skip, rng.

Examples of good instructions:
- "Read sources/combat/wiki_lean_peek.md. Write a gameplay tip about the lean/
  peek mechanic. Use a casual, experienced tone."
- "Search for a recent news article about {{topic}}. Summarize the key points
  as a structured entry with title, summary, and key_facts."
- "Read the manifest for sources tagged 'recipes'. Pick one recipe source,
  extract the ingredients and steps, and format as a structured row."

One work item → one agent → one row. The instruction determines how much
effort the row generator puts in (2 turns for simple extraction, 15 for
synthesis with research).

## Synthetic diversity risk

| Output type | Risk | Strategy |
|---|---|---|
| Code, math, formal reasoning | LOW | Correctness constraints suffice |
| Structured outputs (JSON, SQL) | VERY LOW | Schema validation dominates |
| Information extraction | LOW | Deterministic validation possible |
| Instruction following | MEDIUM | Vary instruction styles |
| Technical explanations | MEDIUM | Vary tone/audience in instructions |
| Creative / dialogue / humor | HIGH-EXTREME | Source variety essential |

For high-risk areas, diversity comes from varied sources and varied instructions.

## Resources

{resources_section}
- Web search and browsing (direct tools + subagents)
- Code execution (Python)

## Column schema
{columns_description}

## Conversation history
{conversation_summary}

## Current source manifest
{manifest_summary}

## Tools

**Direct research tools:**
- brave_search(query): Search the web
- open(ref_id_or_url, start_line): View a page or file
- find(ref_id, pattern): Search within a loaded page
- click(ref_id, link_id): Follow a link
- code_exec(script, description): Execute Python
- interact(url_or_ref_id, task): Browser agent for complex interactions

**Source management:**
- save_source(content, path, description, tags, authority, source_type):
  Save research material as a persistent source file
- read_file(path): Read any file in the workspace (sources, uploads, etc.)
- list_files(directory): List files in a directory

**Delegation:**
- run_subagent(task): Spawn a research subagent. It can search, browse,
  and save_source. Returns its text response.

**Generation:**
- create_work_items(items): Create work items for row generation. Each item
  has an instruction string and optional schema override. Does NOT block —
  generation runs in background while you continue working.
- check_progress(): Check generation stats (rows generated, errors, skipped)

**Human-in-the-loop:**
- ask_user(message): Send a message to the user and wait for their response.
  Use to show samples, ask for feedback, clarify requirements.

**Completion:**
- done(reason): Signal orchestration is complete

## Principles

- **Quality first.** Seek source material that exceeds what you'd synthesize.
- **Sources are for grounding.** Save research material so row generators can
  reference it. Real sources over synthetic generation.
- **Work items are instructions.** Each one tells the row generator exactly what
  to do. Be specific about sources to read and output to produce.
- **Loop, don't pipeline.** Start generating early. Monitor. Adjust. Create more
  work items as you learn. Don't over-research before starting.
- **No licensing concerns.** All content is free to use. Focus on quality.
- **The row generator is capable.** It has browsing, search, code execution, and
  rng tools. If per-row research is needed, say so in the instruction.
- Target: {num_samples} rows.
"""


class OrchestratorAgent:
    """
    V3 Orchestrator. Researches, accumulates sources, creates work items,
    monitors generation, and talks to the user.

    Usage:
        orchestrator = OrchestratorAgent(
            chat_history=[...],
            columns=[...],
            num_samples=1000,
            source_manager=source_manager,
            on_create_work_items=my_callback,
            on_ask_user=my_ask_callback,
            on_check_progress=my_progress_callback,
            openai_client=tracked_client,
            ...
        )
        await orchestrator.run()
    """

    def __init__(
        self,
        chat_history: List[Dict[str, str]],
        columns: List[Dict[str, Any]],
        num_samples: int,
        openai_client: TrackedOpenAIClient,
        model: str,
        workspace_dir: Path,
        source_manager: SourceManager,
        on_create_work_items: Callable[[List[Dict]], Awaitable[int]],
        on_ask_user: Callable[[str], Awaitable[str]],
        on_check_progress: Callable[[], Dict],
        uploaded_files: Optional[List[Dict[str, Any]]] = None,
        brave_api_key: Optional[str] = None,
        sandbox: Optional[Any] = None,
        stop_checker: Optional[Callable[[], bool]] = None,
        cost_checker: Optional[Callable[[], tuple[bool, Optional[str]]]] = None,
        blob_service_client: Optional[Any] = None,
        project_id: Optional[Any] = None,
        on_tool_call: Optional[Callable[[str, str], None]] = None,
        mcp_tools: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        self.chat_history = chat_history
        self.columns = columns
        self.num_samples = num_samples
        self.workspace_dir = Path(workspace_dir)
        self.openai_client = openai_client
        self.model = model
        self.source_manager = source_manager
        self.on_create_work_items = on_create_work_items
        self.on_ask_user = on_ask_user
        self.on_check_progress = on_check_progress
        self.brave_api_key = brave_api_key
        self.sandbox = sandbox
        self.stop_checker = stop_checker
        self.cost_checker = cost_checker
        self.blob_service_client = blob_service_client
        self.project_id = project_id
        self.on_tool_call = on_tool_call
        self.mcp_tools = mcp_tools or []

        # State
        self._subagents: Dict[str, ResearchAgent] = {}
        self._is_done = False
        self._next_id = 0
        self._total_work_items_created = 0

        # Build tools — direct research tools + orchestrator-specific tools
        from dsl_worker.phases.research_tools import ResearchTools, ResearchScope

        self._impl = ResearchTools(
            workspace_dir=workspace_dir,
            schema=[],
            brave_api_key=brave_api_key,
            openai_client=openai_client,
            model=model,
            sandbox=sandbox,
            stop_checker=stop_checker,
            blob_service_client=blob_service_client,
            project_id=project_id,
        )
        self._impl.set_scope(ResearchScope(
            id="orchestrator",
            description="",
            quota=0,
        ))

        registry = ToolRegistry()
        self._register_tools(registry)

        # Build system prompt
        columns_desc = self._format_columns()
        convo_summary = self._format_conversation()
        resources_section = self._format_resources(uploaded_files)
        manifest_summary = source_manager.get_manifest_summary()

        system_prompt = ORCHESTRATOR_SYSTEM_PROMPT.format(
            num_samples=num_samples,
            columns_description=columns_desc,
            conversation_summary=convo_summary,
            resources_section=resources_section,
            manifest_summary=manifest_summary,
        )

        self._conversation = AgentConversation(
            openai_client=openai_client,
            model=model,
            system_prompt=system_prompt,
            tools=registry,
            stop_checker=stop_checker,
            max_turns=300,
            reasoning={"effort": "high", "summary": "detailed"},
            label="orchestrator",
            continue_on_text=True,
            on_tool_call=on_tool_call,
            extra_tools=self.mcp_tools,
        )

    def _new_id(self) -> str:
        self._next_id += 1
        return f"agent_{self._next_id}"

    def _format_columns(self) -> str:
        if not self.columns:
            return "(no columns defined)"

        lines = ["| Name | Type | Details |", "|------|------|---------|"]
        for col in self.columns:
            name = col.get("name", "?")
            ctype = col.get("type", "?")
            details = ""
            if ctype == "enum" and col.get("enum_values"):
                details = f"values: {col['enum_values']}"
            elif ctype == "json" and col.get("json_schema"):
                details = "json_schema defined"
            lines.append(f"| {name} | {ctype} | {details} |")
        return "\n".join(lines)

    def _format_conversation(self) -> str:
        if not self.chat_history:
            return "(no conversation history)"

        parts = []
        for msg in self.chat_history:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            parts.append(f"**{role}**: {content}")
        return "\n\n".join(parts)

    def _format_resources(self, uploaded_files: Optional[List[Dict[str, Any]]]) -> str:
        lines = []
        if uploaded_files:
            lines.append("Uploaded files:")
            for f in uploaded_files:
                name = f.get("filename", "unknown")
                size = f.get("size_bytes", 0)
                ctype = f.get("content_type", "")
                if size > 1_000_000:
                    size_str = f"{size / 1_000_000:.1f} MB"
                elif size > 1_000:
                    size_str = f"{size / 1_000:.0f} KB"
                else:
                    size_str = f"{size} bytes"
                lines.append(f"  - {name} ({ctype}, {size_str})")
        else:
            lines.append("No uploaded files.")
        return "\n".join(lines)

    def _register_tools(self, registry: ToolRegistry) -> None:
        """Register all orchestrator tools."""

        # --- Direct research tools (brave_search, open, find, click, etc.) ---
        self._impl.register_on(registry)

        # --- save_source ---
        source_mgr = self.source_manager

        async def save_source(args: Dict) -> tuple[str, float]:
            try:
                result = await source_mgr.save_source(
                    content=args.get("content", ""),
                    path=args.get("path", ""),
                    description=args.get("description", ""),
                    tags=args.get("tags", []),
                    authority=args.get("authority", 0.5),
                    source_type=args.get("source_type", "article"),
                    url=args.get("url"),
                )
                count = source_mgr.source_count
                return f"Source saved: {result} (manifest now has {count} sources)", 0.0
            except Exception as e:
                return f"Error saving source: {e}", 0.0

        registry.add(
            name="save_source",
            description=(
                "Save research material as a persistent source file. "
                "Row generators will read these sources when producing rows."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": "The content to save",
                    },
                    "path": {
                        "type": "string",
                        "description": "Relative path within sources/ (e.g., 'combat/wiki_peek.md')",
                    },
                    "description": {
                        "type": "string",
                        "description": "Brief description of what this source contains",
                    },
                    "tags": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Tags for filtering",
                    },
                    "authority": {
                        "type": "number",
                        "description": "Authority score 0-1",
                    },
                    "source_type": {
                        "type": "string",
                        "description": "Source category: wiki, forum, article, code, data, upload",
                    },
                    "url": {
                        "type": "string",
                        "description": "Original URL if web content (optional)",
                    },
                },
                "required": ["content", "path", "description", "tags", "authority", "source_type"],
            },
            handler=save_source,
        )

        # --- read_file ---
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
            description="Read a file from the workspace (sources, uploads, etc.).",
            parameters={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "File path (relative to workspace or sources directory)",
                    },
                },
                "required": ["path"],
            },
            handler=read_file,
        )

        # --- run_subagent ---
        async def run_subagent(args: Dict) -> tuple[str, float]:
            task = args.get("task", "")
            agent_id = self._new_id()

            agent = ResearchAgent(
                openai_client=self.openai_client,
                model=self.model,
                workspace_dir=self.workspace_dir,
                brave_api_key=self.brave_api_key,
                sandbox=self.sandbox,
                stop_checker=self.stop_checker,
                blob_service_client=self.blob_service_client,
                project_id=self.project_id,
                on_tool_call=self.on_tool_call,
                mcp_tools=self.mcp_tools,
                source_manager=self.source_manager,
            )
            agent._conversation.label = f"subagent:{agent_id}"
            self._subagents[agent_id] = agent

            result = await agent.ask_full(task)
            return (
                f"[Subagent {agent_id}]\n{result.text}\n\n"
                f"(cost: ${result.cost_usd:.4f}, {result.turns_taken} turns)"
            ), result.cost_usd

        registry.add(
            name="run_subagent",
            description=(
                "Spawn a research subagent. It can search, browse, save sources, "
                "and return a text response. Use for delegating research tasks."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "task": {
                        "type": "string",
                        "description": "What the subagent should research",
                    },
                },
                "required": ["task"],
            },
            handler=run_subagent,
        )

        # --- create_work_items ---
        async def create_work_items(args: Dict) -> tuple[str, float]:
            items = args.get("items", [])
            if not items:
                return "Error: no items provided", 0.0

            # Validate items
            errors = []
            for i, item in enumerate(items):
                if not isinstance(item, dict):
                    errors.append(f"Item {i}: must be an object")
                elif not item.get("instruction"):
                    errors.append(f"Item {i}: 'instruction' is required")
            if errors:
                return "Validation errors:\n" + "\n".join(f"- {e}" for e in errors), 0.0

            count = await self.on_create_work_items(items)
            self._total_work_items_created += count

            # Refresh manifest summary in case subagents added sources
            manifest_summary = self.source_manager.get_manifest_summary()
            return (
                f"Created {count} work items. "
                f"Total work items created: {self._total_work_items_created}. "
                f"Target: {self.num_samples} rows.\n"
                f"Generation is running in the background. Use check_progress() to monitor.\n"
                f"\nCurrent manifest:\n{manifest_summary}"
            ), 0.0

        registry.add(
            name="create_work_items",
            description=(
                "Create work items for row generation. Each item has an instruction "
                "that tells the row generator what to do. Generation runs in the "
                "background — you can continue researching and creating more items."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "items": {
                        "type": "array",
                        "description": "Work items to create",
                        "items": {
                            "type": "object",
                            "properties": {
                                "instruction": {
                                    "type": "string",
                                    "description": (
                                        "What the row generator should do. Reference source "
                                        "files, specify research tasks, define the transformation."
                                    ),
                                },
                                "schema": {
                                    "type": "array",
                                    "description": "Optional schema override for this item",
                                    "items": {"type": "object"},
                                },
                                "tags": {
                                    "type": "object",
                                    "description": "Optional metadata tags for the generated row",
                                },
                            },
                            "required": ["instruction"],
                        },
                    },
                },
                "required": ["items"],
            },
            handler=create_work_items,
        )

        # --- check_progress ---
        async def check_progress(args: Dict) -> tuple[str, float]:
            stats = self.on_check_progress()
            rows = stats.get("rows_generated", 0)
            errors = stats.get("errors", 0)
            skipped = stats.get("skipped", 0)
            in_progress = stats.get("in_progress", 0)
            total_cost = stats.get("total_cost", 0)

            target = self.num_samples
            remaining = max(0, target - rows)

            lines = [
                f"## Generation Progress",
                f"- Rows generated: {rows} / {target} ({rows/target*100:.0f}%)" if target else f"- Rows generated: {rows}",
                f"- Errors: {errors}",
                f"- Skipped: {skipped}",
                f"- In progress: {in_progress}",
                f"- Work items created: {self._total_work_items_created}",
                f"- Generation cost: ${total_cost:.4f}",
                f"- Remaining to target: {remaining}",
            ]
            return "\n".join(lines), 0.0

        registry.add(
            name="check_progress",
            description="Check generation progress: rows generated, errors, skipped, etc.",
            parameters={"type": "object", "properties": {}},
            handler=check_progress,
        )

        # --- ask_user ---
        async def ask_user(args: Dict) -> tuple[str, float]:
            message = args.get("message", "")
            logger.info(f"[Orchestrator] ask_user() called ({len(message)} chars)")
            response = await self.on_ask_user(message)
            return f"User response: {response}", 0.0

        registry.add(
            name="ask_user",
            description=(
                "Send a message to the user and wait for their response. "
                "Use to show sample rows, ask for feedback, or clarify requirements. "
                "The message is posted as a chat message — the user sees it in the UI."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "message": {
                        "type": "string",
                        "description": "Message to send to the user",
                    },
                },
                "required": ["message"],
            },
            handler=ask_user,
        )

        # --- done ---
        async def done(args: Dict) -> tuple[str, float]:
            reason = args.get("reason", "complete")
            self._is_done = True
            return f"Orchestrator done: {reason}", 0.0

        registry.add(
            name="done",
            description="Signal that orchestration is complete.",
            parameters={
                "type": "object",
                "properties": {
                    "reason": {
                        "type": "string",
                        "description": "Why orchestration is done",
                    },
                },
            },
            handler=done,
        )

    async def run(self) -> AgentResult:
        """
        Run the orchestrator. This is the main entry point.

        The orchestrator will loop: research → create work items → monitor →
        adjust → done. It continues until it calls done() or hits max_turns.
        """
        result = await self._conversation.send(
            "Begin. Read the conversation history and uploaded files, then start "
            "researching and building the dataset.",
            exit_condition=lambda: self._is_done,
        )
        return result

    @property
    def cost_usd(self) -> float:
        """Total cost across orchestrator + all sub-agents."""
        total = self._conversation.total_cost
        for agent in self._subagents.values():
            total += agent.cost_usd
        return total

    async def cleanup(self) -> None:
        """Clean up all sub-agents and resources."""
        for agent in self._subagents.values():
            try:
                await agent.cleanup()
            except Exception as e:
                logger.warning(f"Subagent cleanup error: {e}")

        try:
            await self._impl.cleanup()
        except Exception as e:
            logger.warning(f"Orchestrator research tools cleanup error: {e}")
