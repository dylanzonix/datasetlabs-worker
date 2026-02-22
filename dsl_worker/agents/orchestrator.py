"""
Orchestrator agent — thin manager that delegates everything.

V4: The orchestrator finishes in ~5 turns:
1. Delegates initial research to a subagent
2. Uses code_exec for bulk operations (clone repos, process data)
3. Creates the instruction template
4. Breaks dataset into topics
5. Delegates all topics in one delegate_topics() call
6. Done

The orchestrator has NO browse tools — all research is delegated.
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

logger = logging.getLogger(__name__)

READ_FILE_LIMIT = 30_000


ORCHESTRATOR_SYSTEM_PROMPT = """\
You are the orchestrator for a dataset generation system.

## Your mission

A user described a dataset through a consultation chat (below). Your job is to
understand the request, delegate research, create an instruction template, break
the dataset into topics, and delegate everything in one call.

## How to work (~5 turns)

1. **Understand the task.** Read the conversation history and any uploaded files.

2. **Get the lay of the land.** If you need a high-level overview to figure out what
   topic areas exist, delegate a quick research question to a subagent or use code_exec
   to explore uploaded files. You're NOT doing deep research — just enough to know what
   categories/topics to break the dataset into. Topic agents do the real research.

3. **Create the instruction template.** Write an instruction FOR the row generator.
   It should have variable slots like {{topic}}, {{difficulty}} that seeds will fill.
   The schema is presented separately to row generators — do NOT paraphrase it.

4. **Break into topics.** Identify logical topic areas that cover the dataset.
   Write a briefing for each — what the topic agent should research and focus on.

5. **Delegate.** Call delegate_topics() with the instruction template, seed
   variables, shared context, and all topics. This spawns topic agents in parallel.

6. **Done.** Call done() immediately after delegating.

## Instruction Template

The instruction is written for the row generator — it's the assignment they execute.
It has variable slots that topic agents fill with specific values (seeds).

Example:
```
Generate a single-turn Q&A about the browser-use Python library.
A {{difficulty}} developer asks a {{question_style}} question about
{{topic}}. An expert answers with code examples grounded in the
actual library documentation. Research the real docs to verify.
```

The row generator sees this with variables filled in:
"A intermediate developer asks a 'how do I' question about proxy configuration..."

Rules:
- Write as a direct task for the row generator
- Include variable slots for things that should vary per row
- Do NOT describe the schema — it's shown separately
- Do NOT include meta-instructions about the system — just the assignment
- The row generator has browsing, search, code_exec, and file reading tools

## Resources

{resources_section}

## Column Schema
{columns_description}

## Conversation History
{conversation_summary}

## Tools

**Research delegation:**
- run_subagent(question): Spawn a research subagent. It can search and browse the
  web. Returns a text summary of its findings. Use sparingly — just to get a
  high-level overview for topic categorization, not deep research.

**Bulk operations:**
- code_exec(script, description): Execute Python. Use for cloning repos,
  processing data, listing files — anything mechanical.

**File reading:**
- read_file(path): Read any file in the workspace (uploads, downloads, etc.)

**Topic delegation:**
- delegate_topics(instruction, seed_variables, shared_context, topics): Spawn
  all topic agents in one call. Each topic gets a name, target seed count, and
  briefing. The system handles everything from here.

**Completion:**
- done(reason): Signal orchestration is complete.

## What Happens After You Delegate

After you call delegate_topics(), the system runs a **sample phase**: each topic agent
produces 1 row, then the user reviews the samples. If the user has feedback,
the topic agents adjust. Then full generation proceeds. You don't participate
after delegation — topic agents handle everything from there.

## Principles

- **You are a CEO, not a researcher.** Your job is to understand the request,
  figure out what topic areas to break it into, and delegate. Topic agents do
  the real research and produce the actual row assignments.
- **~5 turns.** Understand, categorize, delegate, done. Don't linger.
- **One instruction template.** It should capture what every row in the dataset
  looks like. Topic agents handle variation through seeds.
- **Briefings guide topic agents.** Write enough context that topic agents know
  where to look and what to focus on. They do their own deep research.
- **Don't over-research.** You only need to know enough to identify the right
  topic areas. A quick subagent call or reading uploaded files is usually enough.
- Target: {num_samples} rows total.
"""


class OrchestratorAgent:
    """
    V4 Orchestrator. Thin manager — delegates research and generation.

    Usage:
        orchestrator = OrchestratorAgent(
            chat_history=[...],
            columns=[...],
            num_samples=100,
            on_delegate_topics=my_callback,
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
        on_delegate_topics: Callable[[Dict], Awaitable[Dict]],
        source_manager: Optional[Any] = None,  # deprecated, unused
        uploaded_files: Optional[List[Dict[str, Any]]] = None,
        brave_api_key: Optional[str] = None,
        sandbox: Optional[Any] = None,
        stop_checker: Optional[Callable[[], bool]] = None,
        cost_checker: Optional[Callable[[], tuple[bool, Optional[str]]]] = None,
        blob_service_client: Optional[Any] = None,
        project_id: Optional[Any] = None,
        on_tool_call: Optional[Callable[[str, str], None]] = None,
        uploaded_file_urls: Optional[Dict[str, str]] = None,
        mcp_tools: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        self.chat_history = chat_history
        self.columns = columns
        self.num_samples = num_samples
        self.workspace_dir = Path(workspace_dir)
        self.openai_client = openai_client
        self.model = model
        self.on_delegate_topics = on_delegate_topics
        self.brave_api_key = brave_api_key
        self.sandbox = sandbox
        self.stop_checker = stop_checker
        self.cost_checker = cost_checker
        self.blob_service_client = blob_service_client
        self.project_id = project_id
        self.on_tool_call = on_tool_call
        self.uploaded_file_urls = uploaded_file_urls
        self.mcp_tools = mcp_tools or []

        # State
        self._subagents: Dict[str, ResearchAgent] = {}
        self._is_done = False
        self._next_id = 0

        # Build tools
        registry = ToolRegistry()
        self._register_tools(registry)

        # Build system prompt
        columns_desc = self._format_columns()
        convo_summary = self._format_conversation()
        resources_section = self._format_resources(uploaded_files)

        system_prompt = ORCHESTRATOR_SYSTEM_PROMPT.format(
            num_samples=num_samples,
            columns_description=columns_desc,
            conversation_summary=convo_summary,
            resources_section=resources_section,
        )

        self._conversation = AgentConversation(
            openai_client=openai_client,
            model=model,
            system_prompt=system_prompt,
            tools=registry,
            stop_checker=stop_checker,
            max_turns=30,
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
        """Register orchestrator tools. No browse tools — delegation only."""

        # --- run_subagent ---
        async def run_subagent(args: Dict) -> tuple[str, float]:
            question = args.get("question", "")
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
                uploaded_file_urls=self.uploaded_file_urls,
                mcp_tools=self.mcp_tools,
            )
            agent._conversation.label = f"subagent:{agent_id}"
            self._subagents[agent_id] = agent

            result = await agent.ask_full(question)
            return (
                f"[Subagent {agent_id}]\n{result.text}\n\n"
                f"(cost: ${result.cost_usd:.4f}, {result.turns_taken} turns)"
            ), result.cost_usd

        registry.add(
            name="run_subagent",
            description=(
                "Spawn a research subagent to investigate a question. It can search "
                "the web, browse pages, and save sources. Returns a text summary."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "What the subagent should research",
                    },
                },
                "required": ["question"],
            },
            handler=run_subagent,
        )

        # --- code_exec ---
        # Re-use ResearchTools just for code_exec
        from dsl_worker.phases.research_tools import ResearchTools, ResearchScope

        self._impl = ResearchTools(
            workspace_dir=self.workspace_dir,
            schema=[],
            brave_api_key=self.brave_api_key,
            openai_client=self.openai_client,
            model=self.model,
            sandbox=self.sandbox,
            stop_checker=self.stop_checker,
            blob_service_client=self.blob_service_client,
            project_id=self.project_id,
            uploaded_file_urls=self.uploaded_file_urls,
        )
        self._impl.set_scope(ResearchScope(
            id="orchestrator",
            description="",
            quota=0,
        ))

        async def code_exec(args: Dict) -> tuple[str, float]:
            return await self._impl.code_exec(
                script=args.get("script", ""),
                description=args.get("description", ""),
            )

        registry.add(
            name="code_exec",
            description=(
                "Execute Python. Use for bulk operations: clone repos, "
                "process data files, list directory contents, parse uploads."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "script": {
                        "type": "string",
                        "description": "Python script to execute",
                    },
                    "description": {
                        "type": "string",
                        "description": "Brief description of what this script does",
                    },
                },
                "required": ["script"],
            },
            handler=code_exec,
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
            description="Read a file from the workspace (uploads, downloads, etc.).",
            parameters={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "File path (relative to workspace)",
                    },
                },
                "required": ["path"],
            },
            handler=read_file,
        )

        # --- delegate_topics ---
        async def delegate_topics(args: Dict) -> tuple[str, float]:
            instruction = args.get("instruction", "")
            seed_variables = args.get("seed_variables", [])
            shared_context = args.get("shared_context", "")
            topics = args.get("topics", [])

            if not instruction:
                return "Error: instruction is required", 0.0
            if not seed_variables:
                return "Error: seed_variables is required (list of variable names)", 0.0
            if not topics:
                return "Error: at least one topic is required", 0.0

            # Validate topics
            errors = []
            total_target = 0
            for i, topic in enumerate(topics):
                if not isinstance(topic, dict):
                    errors.append(f"Topic {i}: must be an object")
                    continue
                if not topic.get("name"):
                    errors.append(f"Topic {i}: 'name' is required")
                if not topic.get("briefing"):
                    errors.append(f"Topic {i}: 'briefing' is required")
                target = topic.get("target", 10)
                total_target += target
            if errors:
                return "Validation errors:\n" + "\n".join(f"- {e}" for e in errors), 0.0

            # Normalize targets to match num_samples
            if total_target != self.num_samples and total_target > 0:
                ratio = self.num_samples / total_target
                for topic in topics:
                    topic["target"] = max(1, round(topic.get("target", 10) * ratio))
                # Adjust the last topic to hit exact target
                adjusted_total = sum(t.get("target", 10) for t in topics)
                if adjusted_total != self.num_samples:
                    topics[-1]["target"] += self.num_samples - adjusted_total

            # Dispatch via callback — the job processor handles spawning topic agents
            config = {
                "instruction": instruction,
                "seed_variables": seed_variables,
                "shared_context": shared_context,
                "topics": topics,
            }

            await self.on_delegate_topics(config)
            topic_count = len(topics)
            total = sum(t.get("target", 10) for t in topics)

            return (
                f"Delegated {topic_count} topics ({total} total target rows). "
                f"Topic agents will research, produce seeds, and dispatch row generators. "
                f"Your job is done — call done() now."
            ), 0.0

        registry.add(
            name="delegate_topics",
            description=(
                "Delegate all topics to topic agents in one call. "
                "Specify the instruction template, seed variables, shared context, "
                "and a list of topics with names, targets, and briefings. "
                "Topic agents run in parallel and handle everything from here."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "instruction": {
                        "type": "string",
                        "description": (
                            "The instruction template for row generators. "
                            "Use {variable_name} for slots that seeds fill. "
                            "Written as a direct task for the row generator."
                        ),
                    },
                    "seed_variables": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Variable names used in the instruction template (e.g., ['topic', 'difficulty'])",
                    },
                    "shared_context": {
                        "type": "string",
                        "description": "Context shared across all topics (e.g., 'Repo cloned at /workspace/repo')",
                    },
                    "topics": {
                        "type": "array",
                        "description": "Topics to delegate",
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": {
                                    "type": "string",
                                    "description": "Topic name",
                                },
                                "target": {
                                    "type": "integer",
                                    "description": "Target number of seeds (rows) for this topic",
                                },
                                "briefing": {
                                    "type": "string",
                                    "description": (
                                        "What this topic covers. Guide the topic agent — "
                                        "what to research, what subtopics to cover, key files to look at."
                                    ),
                                },
                            },
                            "required": ["name", "target", "briefing"],
                        },
                    },
                },
                "required": ["instruction", "seed_variables", "topics"],
            },
            handler=delegate_topics,
        )

        # --- done ---
        async def done(args: Dict) -> tuple[str, float]:
            reason = args.get("reason", "complete")
            self._is_done = True
            return f"Orchestrator done: {reason}", 0.0

        registry.add(
            name="done",
            description="Signal that orchestration is complete. Call after delegate_topics.",
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
        """Run the orchestrator (~5 turns)."""
        result = await self._conversation.send(
            "Begin. Read the conversation history and uploaded files, then research, "
            "create an instruction template, break into topics, and delegate.",
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
            logger.warning(f"Orchestrator cleanup error: {e}")
