"""
Orchestrator agent — the brain that coordinates dataset generation.

Reads the conversation history, does research, spawns generators to produce
seeds, writes a recipe (generation prompt), and triggers row generation.
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from dsl_worker.agents.base import AgentConversation, AgentResult
from dsl_worker.agents.tools import ToolRegistry
from dsl_worker.agents.research import ResearchAgent
from dsl_worker.agents.generator import GeneratorAgent
from dsl_worker.billing.tracked_client import TrackedOpenAIClient

logger = logging.getLogger(__name__)


ORCHESTRATOR_SYSTEM_PROMPT = """\
You are the orchestrator for a synthetic dataset generation system.

## Your context

The user has described a dataset they want through a chat conversation.
You have access to the conversation history and the column schema.

## Your job

1. **Understand** — Read the conversation to understand the dataset requirements
2. **Research** — Use research() to deeply understand the domain, find sources,
   and gather the knowledge needed to generate high-quality data
3. **Generate seeds** — Use start_generator() to spawn generators that iterate
   through specific scopes and yield individual seeds (items to generate rows from)
4. **Write recipe** — Use write_recipe() to write detailed generation instructions
   that tell the row generator exactly how to produce each row from a seed
5. **Begin generation** — Use begin_generation() to start producing rows
6. **Monitor** — Check progress with check_row_count(), stop generators when enough
   seeds are collected
7. **Complete** — Call done() when generation is complete

## Guidelines

- Research THOROUGHLY before writing the recipe. Quality comes from understanding.
- The recipe should be specific, detailed, and reference what you learned from research.
- Generators yield seeds concurrently — you can run multiple for different scopes.
- Each seed becomes one row in the dataset.
- Aim for {num_samples} total rows.
- Check balance/costs periodically.

## Column schema
{columns_description}

## Conversation history
{conversation_summary}
"""


class OrchestratorAgent:
    """
    Main orchestrator. Reads conversation, researches, spawns generators,
    writes recipe, and triggers generation.

    Usage:
        orchestrator = OrchestratorAgent(
            chat_history=[...],
            columns=[...],
            num_samples=1000,
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
        brave_api_key: Optional[str] = None,
        sandbox: Optional[Any] = None,
        stop_checker: Optional[Callable[[], bool]] = None,
        cost_checker: Optional[Callable[[], tuple[bool, Optional[str]]]] = None,
        on_recipe_ready: Optional[Callable[[str, asyncio.Queue], Any]] = None,
        on_done: Optional[Callable[[], Any]] = None,
    ) -> None:
        self.chat_history = chat_history
        self.columns = columns
        self.num_samples = num_samples
        self.workspace_dir = Path(workspace_dir)
        self.openai_client = openai_client
        self.model = model
        self.brave_api_key = brave_api_key
        self.sandbox = sandbox
        self.stop_checker = stop_checker
        self.cost_checker = cost_checker
        self.on_recipe_ready = on_recipe_ready
        self.on_done = on_done

        # State
        self.recipe: Optional[str] = None
        self.seed_queue: asyncio.Queue = asyncio.Queue()
        self._research_agents: Dict[str, ResearchAgent] = {}
        self._generators: Dict[str, GeneratorAgent] = {}
        self._generator_tasks: Dict[str, asyncio.Task] = {}
        self._generation_started = False
        self._is_done = False
        self._next_id = 0

        # Build tools
        registry = ToolRegistry()
        self._register_tools(registry)

        # Build system prompt
        columns_desc = self._format_columns()
        convo_summary = self._format_conversation()

        system_prompt = ORCHESTRATOR_SYSTEM_PROMPT.format(
            num_samples=num_samples,
            columns_description=columns_desc,
            conversation_summary=convo_summary,
        )

        self._conversation = AgentConversation(
            openai_client=openai_client,
            model=model,
            system_prompt=system_prompt,
            tools=registry,
            stop_checker=stop_checker,
            max_turns=300,
        )

    def _new_id(self) -> str:
        self._next_id += 1
        return f"agent_{self._next_id}"

    def _format_columns(self) -> str:
        if not self.columns:
            return "(no columns defined)"

        lines = ["| Name | Type | Description |", "|------|------|-------------|"]
        for col in self.columns:
            name = col.get("name", "?")
            ctype = col.get("type", "?")
            desc = col.get("description", "")
            lines.append(f"| {name} | {ctype} | {desc} |")
        return "\n".join(lines)

    def _format_conversation(self) -> str:
        if not self.chat_history:
            return "(no conversation history)"

        parts = []
        for msg in self.chat_history:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            # Truncate long messages
            if len(content) > 500:
                content = content[:500] + "..."
            parts.append(f"**{role}**: {content}")
        return "\n\n".join(parts)

    def _register_tools(self, registry: ToolRegistry) -> None:
        """Register orchestrator tools."""

        # --- research ---
        async def research(args: Dict) -> tuple[str, float]:
            task = args.get("task", "")
            agent_id = self._new_id()

            agent = ResearchAgent(
                openai_client=self.openai_client,
                model=self.model,
                workspace_dir=self.workspace_dir,
                brave_api_key=self.brave_api_key,
                sandbox=self.sandbox,
                stop_checker=self.stop_checker,
            )
            self._research_agents[agent_id] = agent

            result = await agent.ask_full(task)
            return (
                f"[Research agent {agent_id}]\n{result.text}\n\n"
                f"(cost: ${result.cost_usd:.4f}, {result.turns_taken} turns)"
            ), result.cost_usd

        registry.add(
            name="research",
            description="Spawn a research agent to investigate a topic. Returns findings.",
            parameters={
                "type": "object",
                "properties": {
                    "task": {
                        "type": "string",
                        "description": "What to research (be specific)",
                    },
                },
                "required": ["task"],
            },
            handler=research,
        )

        # --- ask_research ---
        async def ask_research(args: Dict) -> tuple[str, float]:
            agent_id = args.get("agent_id", "")
            question = args.get("question", "")

            agent = self._research_agents.get(agent_id)
            if not agent:
                return f"Research agent '{agent_id}' not found.", 0.0

            result = await agent.ask_full(question)
            return result.text, result.cost_usd

        registry.add(
            name="ask_research",
            description="Ask a follow-up question to an existing research agent.",
            parameters={
                "type": "object",
                "properties": {
                    "agent_id": {"type": "string", "description": "Research agent ID"},
                    "question": {"type": "string", "description": "Follow-up question"},
                },
                "required": ["agent_id", "question"],
            },
            handler=ask_research,
        )

        # --- start_generator ---
        async def start_generator(args: Dict) -> tuple[str, float]:
            scope = args.get("scope", "")
            seed_description = args.get("seed_description", "")

            gen_id = self._new_id()
            gen = GeneratorAgent(
                openai_client=self.openai_client,
                model=self.model,
                scope=scope,
                seed_description=seed_description,
                seed_queue=self.seed_queue,
                workspace_dir=self.workspace_dir,
                brave_api_key=self.brave_api_key,
                sandbox=self.sandbox,
                stop_checker=self.stop_checker,
            )
            self._generators[gen_id] = gen

            # Run generator in background
            task = asyncio.create_task(gen.run())
            self._generator_tasks[gen_id] = task

            return (
                f"Generator '{gen_id}' started for scope: {scope}\n"
                f"Seeds will be yielded into the shared queue."
            ), 0.0

        registry.add(
            name="start_generator",
            description="Spawn a generator agent that iterates a scope and yields seeds.",
            parameters={
                "type": "object",
                "properties": {
                    "scope": {
                        "type": "string",
                        "description": "What to iterate through (be specific)",
                    },
                    "seed_description": {
                        "type": "string",
                        "description": "Shape of each seed (fields and their meaning)",
                    },
                },
                "required": ["scope", "seed_description"],
            },
            handler=start_generator,
        )

        # --- stop_generator ---
        async def stop_generator(args: Dict) -> tuple[str, float]:
            gen_id = args.get("generator_id", "")
            gen = self._generators.get(gen_id)
            if not gen:
                return f"Generator '{gen_id}' not found.", 0.0

            gen.stop()
            task = self._generator_tasks.get(gen_id)
            if task and not task.done():
                task.cancel()

            return f"Generator '{gen_id}' stopped. Yielded {gen.seeds_yielded} seeds.", 0.0

        registry.add(
            name="stop_generator",
            description="Stop a running generator.",
            parameters={
                "type": "object",
                "properties": {
                    "generator_id": {"type": "string"},
                },
                "required": ["generator_id"],
            },
            handler=stop_generator,
        )

        # --- check_seed_count ---
        async def check_seed_count(args: Dict) -> tuple[str, float]:
            total = sum(g.seeds_yielded for g in self._generators.values())
            queue_size = self.seed_queue.qsize()
            return (
                f"Total seeds yielded: {total}\n"
                f"Seeds in queue (pending): {queue_size}"
            ), 0.0

        registry.add(
            name="check_seed_count",
            description="Check how many seeds have been yielded by generators.",
            parameters={"type": "object", "properties": {}},
            handler=check_seed_count,
        )

        # --- check_row_count ---
        async def check_row_count(args: Dict) -> tuple[str, float]:
            # This will be populated by the job processor's integration
            return "Row count checking requires generation to be running.", 0.0

        registry.add(
            name="check_row_count",
            description="Check how many rows have been generated so far.",
            parameters={"type": "object", "properties": {}},
            handler=check_row_count,
        )

        # --- write_recipe ---
        async def write_recipe(args: Dict) -> tuple[str, float]:
            recipe = args.get("recipe", "")
            self.recipe = recipe
            return f"Recipe saved ({len(recipe)} chars).", 0.0

        registry.add(
            name="write_recipe",
            description=(
                "Write the generation recipe — detailed instructions for how to "
                "generate each row from a seed. This should be specific, reference "
                "your research findings, and describe exactly what each column should contain."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "recipe": {
                        "type": "string",
                        "description": "The full generation recipe/prompt",
                    },
                },
                "required": ["recipe"],
            },
            handler=write_recipe,
        )

        # --- begin_generation ---
        async def begin_generation(args: Dict) -> tuple[str, float]:
            if not self.recipe:
                return "Error: write_recipe() must be called first.", 0.0

            if self._generation_started:
                return "Generation already started.", 0.0

            self._generation_started = True

            # Signal to the job processor that generation should begin
            if self.on_recipe_ready:
                await self.on_recipe_ready(self.recipe, self.seed_queue)

            return (
                "Generation started. The worker pool is now consuming seeds "
                "and producing rows. Use check_row_count() to monitor progress."
            ), 0.0

        registry.add(
            name="begin_generation",
            description="Start row generation using the recipe and queued seeds.",
            parameters={"type": "object", "properties": {}},
            handler=begin_generation,
        )

        # --- done ---
        async def done(args: Dict) -> tuple[str, float]:
            reason = args.get("reason", "complete")
            self._is_done = True

            # Stop all generators
            for gen_id, gen in self._generators.items():
                gen.stop()
                task = self._generator_tasks.get(gen_id)
                if task and not task.done():
                    task.cancel()

            if self.on_done:
                await self.on_done()

            return f"Orchestrator done: {reason}", 0.0

        registry.add(
            name="done",
            description="Signal that orchestration is complete.",
            parameters={
                "type": "object",
                "properties": {
                    "reason": {"type": "string"},
                },
            },
            handler=done,
        )

    async def run(self) -> AgentResult:
        """
        Run the orchestrator. This is the main entry point.

        The orchestrator will:
        1. Read conversation context (already in system prompt)
        2. Research the domain
        3. Spawn generators for seeds
        4. Write a recipe
        5. Trigger generation
        6. Monitor and complete
        """
        result = await self._conversation.send(
            "Begin orchestrating dataset generation. "
            "Start by understanding the requirements from the conversation history, "
            "then research the domain, spawn generators for seeds, write a recipe, "
            "and begin generation.",
            exit_condition=lambda: self._is_done,
        )
        return result

    @property
    def cost_usd(self) -> float:
        """Total cost across orchestrator + all sub-agents."""
        total = self._conversation.total_cost
        for agent in self._research_agents.values():
            total += agent.cost_usd
        for gen in self._generators.values():
            total += gen.cost_usd
        return total

    async def cleanup(self) -> None:
        """Clean up all sub-agents."""
        for agent in self._research_agents.values():
            try:
                await agent.cleanup()
            except Exception as e:
                logger.warning(f"Research agent cleanup error: {e}")

        for gen in self._generators.values():
            try:
                await gen.cleanup()
            except Exception as e:
                logger.warning(f"Generator cleanup error: {e}")

        # Cancel any running generator tasks
        for task in self._generator_tasks.values():
            if not task.done():
                task.cancel()
