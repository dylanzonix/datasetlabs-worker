"""
Topic agent — delegation layer that maps a topic area and produces row assignments.

V4: Topic agents are spawned by the orchestrator via delegate_topics.
Each topic agent:
1. Receives the dataset brief, topic briefing, and target count
2. Does light research to understand the landscape of its topic
3. Plans sub-areas for diversity
4. Produces specific natural language row assignments
5. Dispatches assignments to row generators
6. Runs independently — doesn't report back to orchestrator
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional

from dsl_worker.agents.base import AgentConversation, AgentResult
from dsl_worker.agents.tools import ToolRegistry
from dsl_worker.billing.tracked_client import TrackedOpenAIClient

logger = logging.getLogger(__name__)

READ_FILE_LIMIT = 30_000


TOPIC_AGENT_SYSTEM_PROMPT = """\
You are a topic agent in a dataset generation system. You manage one topic area —
your job is to produce row assignments that row generators will execute.

## Your Topic

**Name:** {topic_name}
**Briefing:** {topic_briefing}
**Target rows:** {target_count}

## Dataset Brief

This is what every row generator sees as context for what kind of row to produce:

<dataset_brief>
{dataset_brief}
</dataset_brief>

## Column Schema

{columns_description}

## How You Work

You work in two phases. You'll start in the first phase.

### Phase 1: Produce 1 row assignment
Do a quick search if needed to understand the landscape of your topic. Pick one good,
representative assignment. Dispatch it and call done().

### Phase 2: Produce remaining assignments
You'll be asked to continue. Now produce all remaining assignments ({target_count}
total minus what you already dispatched).

In both phases:
1. **Map the space.** Do light research to see what's out there in your topic area.
   What subtopics exist? What variations are possible?
2. **Plan internally.** Break your topic into sub-categories or dimensions, then produce
   assignments that cover the space systematically for diversity.
3. **Write assignments.** Each assignment is a natural language briefing that tells a row
   generator exactly what row to produce. Be specific — include the particular angle,
   entity, scenario, or data point. Include research context when you have it (see below).
4. **Dispatch.** Call dispatch_rows with your assignments.
5. **Done.** Call done() when finished.

## Adapting to Dataset Type

Read the dataset brief carefully — it usually signals what type of work this is.
Adjust your approach:

**If you're discovering real entities** (extractive): Your main job is to find out what
exists. Search directories, registries, indexes, and lists. Browse to confirm they're real.
Then write assignments naming specific entities and where you found them.
- Research pattern: search broadly → collect a list → dispatch all at once
- Example: search PyPI, find 15 real libraries, dispatch 15 assignments naming each one

**If you're designing synthetic content** (synthetic): Your main job is to ensure variety.
Map the space of interesting variations — different styles, difficulty levels, personas,
or scenarios. Then design assignments that systematically cover the space.
- Research pattern: quick research to understand the landscape → plan dimensions → dispatch
- Example: for expert DayZ Q&A, map what topic areas exist (combat, survival, medical,
  base building), then design specific questions across areas and difficulty levels

**If you're batching data for evaluation** (judgment): Your main job is to identify the
data row generators will evaluate. Pull items from uploaded files or sources, then dispatch
assignments presenting specific items to evaluate.
- Research pattern: explore the data source → identify items → dispatch in batches

Your topic briefing may include specific source recommendations — use them as your
starting point rather than searching blind.

## Writing Good Assignments

Each assignment is a specific, natural language instruction for one row. It builds on the
dataset brief (which the row generator also sees).

Good assignments are specific and include research context:
- "Write a Q&A about configuring Playwright's proxy settings for residential proxies.
  The official docs cover this at https://playwright.dev/docs/network — start there."
- "Profile the Polars library (polars-rs on GitHub, available on PyPI). It's a DataFrame
  library written in Rust. Check the latest version and benchmark claims."
- "Generate a coding problem about implementing a binary search tree deletion operation,
  intermediate difficulty. The tricky part is handling nodes with two children."
- "Create a customer support conversation about a delayed international shipment with
  customs issues. The customer is frustrated but polite."

Bad assignments are vague or miss opportunities to pass context:
- "Write a Q&A about Playwright" (too broad, no direction)
- "Generate a coding problem" (no topic, difficulty, or angle)
- "Profile a Python library" (which one? row generator has to start from scratch)

When you've found useful information during your research, pass it forward:
- **URLs you found**: "Check https://... for the official docs"
- **File paths**: "The data is in uploads/products.csv, rows 50-75"
- **Key facts**: "Note: this library was deprecated in 2024, mention alternatives"
- **Source tips**: "Best source is the official API reference, not blog posts (many outdated)"
- **Gotchas**: "Many web sources confuse this with the similarly-named X"

You don't have to deeply research every entity — that's the row generator's job. But if
you stumbled across a useful URL or fact while mapping the space, include it. It saves
the row generator from repeating your work.

## Tools

- brave_search(query): Search the web
- open(ref_id_or_url, start_line): View a page or file
- find(ref_id, pattern): Search within a loaded page
- click(ref_id, link_id): Follow a link
- code_exec(script, description): Execute Python
- read_file(path): Read workspace files
- dispatch_rows(assignments): Send assignments to row generators
- done(): Signal current phase is complete

## Principles

- **You are a delegation layer.** You figure out WHAT rows to produce. Row generators
  figure out HOW — they do the deep research, truth-finding, and content creation.
- **Research to delegate.** A quick search to see what exists in your topic area is
  great. Reading 10 pages on one subtopic is overkill.
- **Diversity through planning.** Break your topic into sub-areas, then systematically
  produce assignments that cover each. Don't cluster on one subtopic.
- **Specific assignments.** Each assignment should name the particular angle, entity, or
  scenario. The row generator uses this as its starting point for research.
- **Pass your research forward.** If you found a useful URL, doc page, or fact while
  mapping the space, include it in the assignment. Row generators work better when
  they have a head start.
"""


class TopicAgent:
    """
    Topic agent — maps a topic area, produces row assignments, dispatches row generators.

    Spawned by the orchestrator via delegate_topics. Runs independently.

    Usage:
        agent = TopicAgent(
            topic_name="Browser Configuration",
            topic_briefing="Covers proxy setup, headless mode, ...",
            dataset_brief="Generate a Q&A about browser-use...",
            target_count=12,
            columns=[...],
            on_dispatch_rows=my_dispatch_callback,
            openai_client=tracked_client,
            ...
        )
        result = await agent.run()
    """

    def __init__(
        self,
        topic_name: str,
        topic_briefing: str,
        dataset_brief: str,
        target_count: int,
        columns: List[Dict[str, Any]],
        openai_client: TrackedOpenAIClient,
        model: str,
        workspace_dir: Path,
        on_dispatch_rows: Callable[..., Awaitable[int]],
        brave_api_key: Optional[str] = None,
        sandbox: Optional[Any] = None,
        stop_checker: Optional[Callable[[], bool]] = None,
        blob_service_client: Optional[Any] = None,
        project_id: Optional[Any] = None,
        on_tool_call: Optional[Callable[[str, str], None]] = None,
        on_cost: Optional[Callable] = None,
        mcp_tools: Optional[List[Dict[str, Any]]] = None,
        langfuse_parent: Optional[Any] = None,
        on_browser_started: Optional[Callable] = None,
    ) -> None:
        self.topic_name = topic_name
        self.dataset_brief = dataset_brief
        self.target_count = target_count
        self.columns = columns
        self.workspace_dir = Path(workspace_dir)
        self.openai_client = openai_client
        self.model = model
        self.on_dispatch_rows = on_dispatch_rows
        self.stop_checker = stop_checker
        self.on_tool_call = on_tool_call
        self.on_cost = on_cost
        self.mcp_tools = mcp_tools or []

        # State
        self._is_done = False
        self._total_dispatched = 0

        # Build research tools
        from dsl_worker.infra.research_tools import ResearchTools, ResearchScope

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
            on_browser_started=on_browser_started,
        )
        self._impl.set_scope(ResearchScope(
            id=f"topic:{topic_name}",
            description=topic_briefing,
            quota=0,
        ))

        registry = ToolRegistry()
        self._register_tools(registry)

        # Build system prompt
        columns_desc = self._format_columns()

        system_prompt = TOPIC_AGENT_SYSTEM_PROMPT.format(
            topic_name=topic_name,
            topic_briefing=topic_briefing,
            target_count=target_count,
            dataset_brief=dataset_brief,
            columns_description=columns_desc,
        )

        self._conversation = AgentConversation(
            openai_client=openai_client,
            model=model,
            system_prompt=system_prompt,
            tools=registry,
            stop_checker=stop_checker,
            max_turns=50,
            reasoning={"effort": "medium", "summary": "detailed"},
            label=f"topic:{topic_name}",
            on_tool_call=on_tool_call,
            on_cost=on_cost,
            extra_tools=self.mcp_tools,
            langfuse_parent=langfuse_parent,
        )

    def _format_columns(self) -> str:
        if not self.columns:
            return "(no columns defined)"
        return json.dumps(self.columns, indent=2)

    def _register_tools(self, registry: ToolRegistry) -> None:
        """Register topic agent tools."""

        # --- Research/browsing tools ---
        self._impl.register_on(registry)

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
            description="Read a file from the workspace (sources, uploads, repo, etc.).",
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

        # --- dispatch_rows ---
        async def dispatch_rows(args: Dict) -> tuple[str, float]:
            assignments = args.get("assignments", [])

            if not assignments:
                return "Error: no assignments provided", 0.0

            # Validate assignments are non-empty strings
            valid = []
            for i, assignment in enumerate(assignments):
                if isinstance(assignment, str) and assignment.strip():
                    valid.append(assignment)
            if not valid:
                return "Error: all assignments were empty strings", 0.0

            # Cap to remaining target so topics never over-dispatch.
            remaining = self.target_count - self._total_dispatched
            if remaining <= 0:
                return (
                    f"Already at target ({self._total_dispatched}/{self.target_count}). "
                    f"No more assignments needed — call done()."
                ), 0.0
            if len(valid) > remaining:
                valid = valid[:remaining]

            # Dispatch via callback — pass the topic's langfuse span so
            # row_generator traces nest under this topic, not the job root.
            topic_span = getattr(self._conversation, "_current_langfuse_span", None)
            count = await self.on_dispatch_rows(
                valid,
                self.dataset_brief,
                self.columns,
                self.topic_name,
                langfuse_parent=topic_span,
            )
            self._total_dispatched += count

            remaining = self.target_count - self._total_dispatched
            if remaining > 0:
                return (
                    f"Dispatched {count} row assignments. "
                    f"Total dispatched: {self._total_dispatched}/{self.target_count}. "
                    f"Still need {remaining} more — keep going."
                ), 0.0
            return (
                f"Dispatched {count} row assignments. "
                f"Target reached ({self._total_dispatched}/{self.target_count}). "
                f"Call done() now."
            ), 0.0

        registry.add(
            name="dispatch_rows",
            description=(
                "Send row assignments to row generators. Each assignment is a "
                "specific, natural language instruction for one row."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "assignments": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "List of row assignment strings. Each is a specific, "
                            "natural language instruction for one row."
                        ),
                    },
                },
                "required": ["assignments"],
            },
            handler=dispatch_rows,
        )

        # --- done ---
        async def done(args: Dict) -> tuple[str, float]:
            remaining = self.target_count - self._total_dispatched
            if remaining > 0:
                return (
                    f"Cannot finish yet — still need {remaining} more assignments "
                    f"({self._total_dispatched}/{self.target_count} dispatched). "
                    f"Please dispatch the remaining assignments first."
                ), 0.0
            self._is_done = True
            return (
                f"Topic '{self.topic_name}' complete. "
                f"Dispatched {self._total_dispatched} row assignments."
            ), 0.0

        registry.add(
            name="done",
            description="Signal this topic is complete.",
            parameters={"type": "object", "properties": {}},
            handler=done,
        )

    async def run(self) -> AgentResult:
        """Run the topic agent — first phase (1 assignment).

        Produces 1 representative assignment with light research, dispatches it,
        and calls done(). The exit condition is count-based: the conversation
        ends when at least 1 assignment has been dispatched.
        """
        result = await self._conversation.send(
            f"Begin. Research your topic area briefly, pick 1 good representative "
            f"assignment, dispatch it, and call done().",
            exit_condition=lambda: self._total_dispatched >= 1,
        )
        return result

    async def resume(self, feedback: Optional[str] = None) -> AgentResult:
        """Resume the topic agent — produce remaining assignments.

        Continues the same conversation (all research context preserved).
        The exit condition is count-based: the conversation keeps running
        until ``target_count`` assignments have been dispatched.  The
        ``done()`` tool also enforces this — it rejects early exits.

        Args:
            feedback: Optional user feedback from sample review.
        """
        self._is_done = False
        remaining = self.target_count - self._total_dispatched

        if remaining <= 0:
            logger.info(f"[topic:{self.topic_name}] Already at target, nothing to resume")
            return AgentResult(text="Already at target", turns_taken=0)

        if feedback:
            message = (
                f"Continue. The user reviewed samples and said: \"{feedback}\"\n\n"
                f"Adjust your approach based on this feedback. "
                f"Produce the remaining {remaining} assignments with good variety. "
                f"Dispatch them and call done()."
            )
        else:
            message = (
                f"Continue — samples approved. "
                f"Produce the remaining {remaining} assignments with good variety. "
                f"Dispatch them and call done()."
            )

        result = await self._conversation.send(
            message,
            exit_condition=lambda: self._total_dispatched >= self.target_count,
        )

        if self._total_dispatched < self.target_count:
            logger.warning(
                f"[topic:{self.topic_name}] hit max_turns with only "
                f"{self._total_dispatched}/{self.target_count} dispatched"
            )

        return result

    @property
    def cost_usd(self) -> float:
        return self._conversation.total_cost

    async def cleanup(self) -> None:
        """Clean up browser and other resources."""
        try:
            await self._impl.cleanup()
        except Exception as e:
            logger.warning(f"Topic agent '{self.topic_name}' cleanup error: {e}")
