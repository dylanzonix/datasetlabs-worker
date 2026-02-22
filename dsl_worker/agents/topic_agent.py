"""
Topic agent — middle manager that researches a topic area, produces seeds,
and dispatches row generators directly.

V4: Topic agents are spawned by the orchestrator via delegate_topics.
Each topic agent:
1. Receives the instruction template, seed variable names, topic briefing, shared context
2. Researches its topic area using browse/search/code tools
3. Produces seeds (variable values that fill the instruction template)
4. Optionally adds context notes for row generators
5. Dispatches row generators with filled assignments
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
You are a topic agent in a dataset generation system. Your mission is to produce
diverse, specific seeds and dispatch them as row assignments. Each seed fills
the instruction template to create one row assignment for a row generator.

## Your Topic

**Name:** {topic_name}
**Briefing:** {topic_briefing}
**Total target seeds:** {target_count}

## Instruction Template

This is the instruction each row generator will receive (with seed values filled in):

```
{instruction_template}
```

The seed variables you need to produce values for: {seed_variables}

## Shared Context

{shared_context}

## Column Schema

{columns_description}

## How You Work

You work in two phases. You'll start in the SAMPLE phase.

### SAMPLE phase
Produce exactly **1 good, representative seed**. Do a quick search if needed to
understand the landscape of your topic, but don't go deep — just enough to pick
a solid first seed. Dispatch it and call done(). This sample row goes to the user
for review.

### FULL phase
After the user approves (or gives feedback), you'll be asked to continue. Now
produce all remaining seeds ({target_count} total minus however many you already
dispatched). Use research to discover the breadth of your topic — what subtopics
exist, what variations are possible — so you can produce diverse seeds. If the user
gave feedback, adjust your approach accordingly. Dispatch them and call done().

### In both phases
1. **Figure out what seeds to produce.** Use search/browse if needed to understand
   what's out there in your topic area. But keep it focused — you're mapping the
   space, not becoming an expert.
2. **Produce seeds** — each is a unique set of variable values for the instruction template.
3. **Add context** (optional) — brief notes for row generators (key facts, gotchas).
4. **Dispatch rows** — call dispatch_rows with your seeds and optional context.
5. **Done** — call done() when finished with the current phase.

## Seed Format

Each seed is an object with keys matching the seed variables. Example:
```json
{{"topic": "proxy configuration", "difficulty": "intermediate", "question_style": "how do I"}}
```

Seeds should have good variety. Don't repeat similar combinations.

## Tools

- brave_search(query): Search the web
- open(ref_id_or_url, start_line): View a page or file
- find(ref_id, pattern): Search within a loaded page
- click(ref_id, link_id): Follow a link
- code_exec(script, description): Execute Python
- read_file(path): Read workspace files
- dispatch_rows(seeds, context): Send assignments to row generators
- done(): Signal the current phase is complete

## Principles

- **Your output is seeds, not research.** Research is just a means to produce better
  seeds. Don't go deep — row generators have their own search/browse tools and will
  do their own research when executing the assignment.
- **Map the space, don't master it.** A quick search to see what subtopics exist is
  great. Reading 10 pages on one subtopic is overkill. You just need to know enough
  to produce diverse, specific seed values.
- **Seeds handle variation.** Each seed should produce a meaningfully different row.
  Vary all seed variables, not just one.
- **Context is brief and supplementary.** A sentence or two of tips for row generators.
  Don't dump research findings — row generators will research on their own.
- **Be efficient.** Sample phase: 1-3 turns. Full phase: 5-10 turns. If you're
  spending more than a couple turns researching before dispatching, you're over-doing it.
"""


class TopicAgent:
    """
    Topic agent — researches a topic area, produces seeds, dispatches row generators.

    Spawned by the orchestrator via delegate_topics. Runs independently.

    Usage:
        agent = TopicAgent(
            topic_name="Browser Configuration",
            topic_briefing="Covers proxy setup, headless mode, ...",
            instruction_template="A {difficulty} developer asks ...",
            seed_variables=["topic", "difficulty", "question_style"],
            target_count=12,
            shared_context="Repo at /workspace/repo",
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
        instruction_template: str,
        seed_variables: List[str],
        target_count: int,
        shared_context: str,
        columns: List[Dict[str, Any]],
        openai_client: TrackedOpenAIClient,
        model: str,
        workspace_dir: Path,
        on_dispatch_rows: Callable[[str, List[Dict], str, List[Dict], str], Awaitable[int]],
        source_manager: Optional[Any] = None,  # deprecated, unused
        brave_api_key: Optional[str] = None,
        sandbox: Optional[Any] = None,
        stop_checker: Optional[Callable[[], bool]] = None,
        blob_service_client: Optional[Any] = None,
        project_id: Optional[Any] = None,
        on_tool_call: Optional[Callable[[str, str], None]] = None,
        mcp_tools: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        self.topic_name = topic_name
        self.instruction_template = instruction_template
        self.seed_variables = seed_variables
        self.target_count = target_count
        self.columns = columns
        self.workspace_dir = Path(workspace_dir)
        self.openai_client = openai_client
        self.model = model
        self.on_dispatch_rows = on_dispatch_rows
        self.stop_checker = stop_checker
        self.on_tool_call = on_tool_call
        self.mcp_tools = mcp_tools or []

        # State
        self._is_done = False
        self._total_seeds_dispatched = 0

        # Build research tools
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
            instruction_template=instruction_template,
            seed_variables=", ".join(seed_variables),
            shared_context=shared_context or "(none)",
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
            extra_tools=self.mcp_tools,
        )

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
            seeds = args.get("seeds", [])
            context = args.get("context", "")

            if not seeds:
                return "Error: no seeds provided", 0.0

            # Validate seeds have the right variables
            errors = []
            for i, seed in enumerate(seeds):
                if not isinstance(seed, dict):
                    errors.append(f"Seed {i}: must be an object")
                    continue
                missing = [v for v in self.seed_variables if v not in seed]
                if missing:
                    errors.append(f"Seed {i}: missing variables {missing}")
            if errors:
                return "Validation errors:\n" + "\n".join(f"- {e}" for e in errors), 0.0

            # Dispatch via callback
            count = await self.on_dispatch_rows(
                self.instruction_template,
                seeds,
                context,
                self.columns,
                self.topic_name,
            )
            self._total_seeds_dispatched += count

            return (
                f"Dispatched {count} row assignments. "
                f"Total dispatched for this topic: {self._total_seeds_dispatched}/{self.target_count}."
            ), 0.0

        registry.add(
            name="dispatch_rows",
            description=(
                "Send row assignments to row generators. Each seed fills the "
                "instruction template to create an assignment. Context is optional "
                "notes that all row generators in this topic will see."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "seeds": {
                        "type": "array",
                        "description": "Seed objects, each with keys matching the seed variables",
                        "items": {"type": "object"},
                    },
                    "context": {
                        "type": "string",
                        "description": (
                            "Optional context notes for row generators. "
                            "Info you discovered that helps them do their job — "
                            "file paths, gotchas, key facts. Not the instruction itself."
                        ),
                    },
                },
                "required": ["seeds"],
            },
            handler=dispatch_rows,
        )

        # --- done ---
        async def done(args: Dict) -> tuple[str, float]:
            self._is_done = True
            return (
                f"Topic '{self.topic_name}' complete. "
                f"Dispatched {self._total_seeds_dispatched} row assignments."
            ), 0.0

        registry.add(
            name="done",
            description="Signal this topic is complete.",
            parameters={"type": "object", "properties": {}},
            handler=done,
        )

    async def run(self) -> AgentResult:
        """Run the topic agent — SAMPLE phase.

        Produces 1 representative seed with thin research, dispatches it,
        and calls done(). The agent's conversation state is fully preserved
        so resume() can continue with all prior context.
        """
        result = await self._conversation.send(
            f"Begin SAMPLE phase for topic '{self.topic_name}'. "
            f"Do quick research, produce exactly 1 good representative seed, "
            f"dispatch it, and call done().",
            exit_condition=lambda: self._is_done,
        )
        return result

    async def resume(self, feedback: Optional[str] = None) -> AgentResult:
        """Resume the topic agent — FULL phase.

        Continues the same conversation (all research context preserved).
        The agent produces remaining seeds at full depth.

        Args:
            feedback: Optional user feedback from sample review.
        """
        self._is_done = False
        remaining = self.target_count - self._total_seeds_dispatched

        if remaining <= 0:
            logger.info(f"[topic:{self.topic_name}] Already at target, nothing to resume")
            return AgentResult(text="Already at target", turns_taken=0)

        if feedback:
            message = (
                f"FULL phase. The user reviewed the sample and said: \"{feedback}\"\n\n"
                f"Adjust your approach based on this feedback. "
                f"Produce the remaining {remaining} seeds with good variety. "
                f"Dispatch them and call done()."
            )
        else:
            message = (
                f"FULL phase — sample approved. "
                f"Produce the remaining {remaining} seeds with good variety. "
                f"Dispatch them and call done()."
            )

        result = await self._conversation.send(
            message,
            exit_condition=lambda: self._is_done,
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
