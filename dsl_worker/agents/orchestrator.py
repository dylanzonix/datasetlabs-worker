"""
Orchestrator agent — the brain that coordinates dataset generation.

Reads the conversation history, does research, creates a plan (pipelines +
buckets + generators), and runs generation via the blocking generate() tool.
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
from dsl_worker.agents.generator import GeneratorAgent
from dsl_worker.billing.tracked_client import TrackedOpenAIClient

logger = logging.getLogger(__name__)


ORCHESTRATOR_SYSTEM_PROMPT = """\
You are the orchestrator for a dataset generation system.

## Context

A user has described a dataset they want through a consultation chat. That
conversation — between the user and a consultation AI — is your "order".
Your job is to execute it: research the domain, plan how to source and
generate the data, then run the pipeline.

The system will automatically pause after generating a few sample rows so the
user can review quality before the full run continues. This is transparent to
you — just focus on planning and execution.

## Resources

{resources_section}
- Web search and browsing via research agents
- Code execution (Python) via research agents
- Generator agents for extracting seeds from web sources

## Column schema
{columns_description}

## Conversation history
{conversation_summary}

## Process

### 1. Understand the requirements
Read the conversation history carefully. What dataset is being requested?
What are the quality expectations? Are there specific sources mentioned?

### 2. Research
Use research() to investigate the domain. Each research agent gives a focused
answer to a specific question — you control depth by sending follow-ups via
ask_research().

Frame research tasks around understanding the subject matter — NOT around data
engineering or extraction mechanics.

Good research tasks: "What are the most popular anime characters for roleplay?",
"What fantasy book series have large fan communities?", "What makes a good X?"
Bad research tasks: "Find SPARQL queries for Wikidata", "What APIs have character
databases?", "Find open data sources with CC0 licensing"

The research agent handles its own source discovery. Your job is to ask the right
questions about the DOMAIN, not about how to technically extract data.

Strategy:
- Start with a broad question to map the landscape
- Use ask_research() to drill into specific aspects that need more depth
- Spawn separate research() agents for genuinely different topics

### 3. Plan
Call set_plan() with your strategy. The plan has three parts:

**Pipelines** — Processing instructions for turning seeds into rows. Usually
just one pipeline unless genuinely different categories need different processing.
Pipeline instructions should be specific and reference what you learned from research.

**Buckets** (optional) — Distribution controls. Use when you need to control what
proportion of the dataset comes from different sources or categories. Each bucket
has a weight and maps to a pipeline. Skip buckets entirely for simple datasets
where uniform sampling is fine.

**Generators** — Extraction agents that iterate through sources and yield seeds.
Each generator has a scope (what to extract), seed description (shape of data),
target count, and bucket assignment. Generators are rough extractors — they don't
need to be perfect, the pipeline handles cleanup.

### 4. Generate
Call generate() to run the pipeline. This blocks until one of:
- **complete** — All quotas met. Call done().
- **shortage** — Generators ran dry before quotas were met. Use add_generator()
  to add more generators to the plan, then call generate() again.
- **sampling_paused** — System paused for user review. Call done().

### 5. Handle results
- On "complete": call done("complete")
- On "shortage": analyze which buckets are short, add targeted generators, generate() again
- On "sampling_paused": call done("sampling_paused")

## Plan structure

```json
{{
  "pipelines": {{
    "main": {{
      "instructions": "Detailed instructions for turning a seed into a row..."
    }}
  }},
  "buckets": [
    {{"id": "primary", "label": "Primary sources", "weight": 0.7, "pipeline_id": "main"}},
    {{"id": "secondary", "label": "Secondary sources", "weight": 0.3, "pipeline_id": "main"}}
  ],
  "generators": [
    {{
      "id": "gen_wiki",
      "scope": "Extract all items from <specific source>",
      "seed_description": "{{field1, field2, source_url}}",
      "target_count": 200,
      "bucket_id": "primary"
    }}
  ]
}}
```

When buckets are omitted, all generators feed into a single implicit bucket and
seeds are processed uniformly.

## Quality principles

- **Prefer real data extraction over LLM generation.** If the data exists somewhere,
  extract it rather than making it up.
- **Research quality determines output quality.** Invest time in finding good sources.
- **Simplest viable strategy wins.** Don't create buckets unless you need distribution
  control. Don't create multiple pipelines unless categories need genuinely different
  processing.
- **Generators are rough extractors.** They should be fast and slightly over-produce.
  The pipeline handles cleanup and filtering via skip_seed.
- Target: {num_samples} rows.
"""


class OrchestratorAgent:
    """
    Main orchestrator. Reads conversation, researches, creates plan,
    and runs generation via the blocking generate() tool.

    Usage:
        orchestrator = OrchestratorAgent(
            chat_history=[...],
            columns=[...],
            num_samples=1000,
            openai_client=tracked_client,
            ...
            on_generate=my_generate_callback,
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
        uploaded_files: Optional[List[Dict[str, Any]]] = None,
        brave_api_key: Optional[str] = None,
        sandbox: Optional[Any] = None,
        stop_checker: Optional[Callable[[], bool]] = None,
        cost_checker: Optional[Callable[[], tuple[bool, Optional[str]]]] = None,
        on_generate: Optional[Callable[[Dict, asyncio.Queue, asyncio.Future], Awaitable]] = None,
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
        self.on_generate = on_generate

        # State
        self.plan: Optional[Dict] = None
        self.plan_version: int = 0
        self.seed_queue: asyncio.Queue = asyncio.Queue()
        self._research_agents: Dict[str, ResearchAgent] = {}
        self._generators: Dict[str, GeneratorAgent] = {}
        self._generator_tasks: Dict[str, asyncio.Task] = {}
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
            max_turns=300,
            reasoning={"effort": "medium", "summary": "detailed"},
            label="orchestrator",
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
            agent._conversation.label = f"research:{agent_id}"
            self._research_agents[agent_id] = agent

            result = await agent.ask_full(task)
            return (
                f"[Research agent {agent_id}]\n{result.text}\n\n"
                f"(cost: ${result.cost_usd:.4f}, {result.turns_taken} turns)"
            ), result.cost_usd

        registry.add(
            name="research",
            description=(
                "Spawn a research agent to investigate a specific question. "
                "The agent will research and return a focused answer. "
                "Use ask_research() for follow-up questions to dig deeper."
            ),
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
            description=(
                "Send a follow-up question to an existing research agent. "
                "Use this to drill deeper into specific aspects of their findings. "
                "The agent retains full context from previous questions."
            ),
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

        # --- set_plan ---
        async def set_plan(args: Dict) -> tuple[str, float]:
            plan = args.get("plan", args)
            if "plan" in args and isinstance(args["plan"], dict):
                plan = args["plan"]

            # Validate plan structure
            errors = self._validate_plan(plan)
            if errors:
                return f"Plan validation failed:\n" + "\n".join(f"- {e}" for e in errors), 0.0

            self.plan = plan
            self.plan_version += 1

            # Build confirmation with computed quotas
            summary_parts = [f"Plan v{self.plan_version} saved."]

            pipelines = plan.get("pipelines", {})
            summary_parts.append(f"Pipelines: {len(pipelines)} ({', '.join(pipelines.keys())})")

            buckets = plan.get("buckets", [])
            if buckets:
                total_weight = sum(b.get("weight", 1.0) for b in buckets)
                summary_parts.append(f"Buckets ({len(buckets)}):")
                for b in buckets:
                    weight = b.get("weight", 1.0)
                    quota = round(self.num_samples * weight / total_weight)
                    summary_parts.append(
                        f"  - {b.get('id')}: {b.get('label', '')} "
                        f"(weight={weight}, quota={quota} rows)"
                    )
            else:
                summary_parts.append(f"No buckets (uniform sampling, target={self.num_samples} rows)")

            generators = plan.get("generators", [])
            summary_parts.append(f"Generators ({len(generators)}):")
            for g in generators:
                summary_parts.append(
                    f"  - {g.get('id')}: scope=\"{g.get('scope', '')[:60]}\" "
                    f"target={g.get('target_count', 100)} bucket={g.get('bucket_id', 'none')}"
                )

            return "\n".join(summary_parts), 0.0

        registry.add(
            name="set_plan",
            description=(
                "Set the generation plan. Includes pipelines (processing instructions), "
                "buckets (optional distribution controls), and generators (seed extraction agents). "
                "Each call creates a new plan version."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "plan": {
                        "type": "object",
                        "description": "The plan object with pipelines, buckets (optional), and generators",
                        "properties": {
                            "pipelines": {
                                "type": "object",
                                "description": "Map of pipeline_id -> {instructions: string}",
                            },
                            "buckets": {
                                "type": "array",
                                "description": "Optional distribution controls",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "id": {"type": "string"},
                                        "label": {"type": "string"},
                                        "weight": {"type": "number"},
                                        "pipeline_id": {"type": "string"},
                                    },
                                    "required": ["id", "weight", "pipeline_id"],
                                },
                            },
                            "generators": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "id": {"type": "string"},
                                        "scope": {"type": "string"},
                                        "seed_description": {"type": "string"},
                                        "target_count": {"type": "integer"},
                                        "bucket_id": {"type": "string"},
                                    },
                                    "required": ["id", "scope", "seed_description"],
                                },
                            },
                        },
                        "required": ["pipelines", "generators"],
                    },
                },
                "required": ["plan"],
            },
            handler=set_plan,
        )

        # --- add_generator ---
        async def add_generator(args: Dict) -> tuple[str, float]:
            if not self.plan:
                return "Error: call set_plan() first.", 0.0

            generator = args.get("generator", args)
            if "generator" in args and isinstance(args["generator"], dict):
                generator = args["generator"]

            # Validate generator
            if not generator.get("id") or not generator.get("scope"):
                return "Error: generator needs at least 'id' and 'scope'.", 0.0

            # Append to current plan (no version bump)
            self.plan["generators"].append(generator)

            return (
                f"Generator '{generator.get('id')}' added to plan. "
                f"Now {len(self.plan['generators'])} generators total. "
                f"Call generate() to run."
            ), 0.0

        registry.add(
            name="add_generator",
            description=(
                "Add a generator to the current plan. Use for shortage recovery — "
                "add targeted generators for buckets that are short, then generate() again."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "generator": {
                        "type": "object",
                        "description": "Generator definition (same shape as in set_plan)",
                        "properties": {
                            "id": {"type": "string"},
                            "scope": {"type": "string"},
                            "seed_description": {"type": "string"},
                            "target_count": {"type": "integer"},
                            "bucket_id": {"type": "string"},
                        },
                        "required": ["id", "scope", "seed_description"],
                    },
                },
                "required": ["generator"],
            },
            handler=add_generator,
        )

        # --- generate ---
        async def generate(args: Dict) -> tuple[str, float]:
            if not self.plan:
                return "Error: call set_plan() first.", 0.0

            if not self.on_generate:
                return "Error: generation callback not configured.", 0.0

            # Create a fresh queue for this generation run
            self.seed_queue = asyncio.Queue()

            # Create future for the result
            result_future: asyncio.Future = asyncio.get_event_loop().create_future()

            # Signal job_processor to start the generation consumer
            await self.on_generate(self.plan, self.seed_queue, result_future)

            # Start all generator agents — they feed the seed queue
            generators_config = self.plan.get("generators", [])
            for gen_config in generators_config:
                gen_id = gen_config.get("id", self._new_id())

                # Skip generators that are already running from a previous generate() call
                if gen_id in self._generators:
                    continue

                gen = GeneratorAgent(
                    openai_client=self.openai_client,
                    model=self.model,
                    scope=gen_config.get("scope", ""),
                    seed_description=gen_config.get("seed_description", ""),
                    seed_queue=self.seed_queue,
                    workspace_dir=self.workspace_dir,
                    generator_id=gen_id,
                    target_count=gen_config.get("target_count", 100),
                    bucket_id=gen_config.get("bucket_id"),
                    brave_api_key=self.brave_api_key,
                    sandbox=self.sandbox,
                    stop_checker=self.stop_checker,
                )
                self._generators[gen_id] = gen

                task = asyncio.create_task(gen.run())
                self._generator_tasks[gen_id] = task

            logger.info(
                f"[Orchestrator] generate() started: "
                f"{len(generators_config)} generators, awaiting result..."
            )

            # Block until the consumer resolves the future
            try:
                result = await result_future
            except Exception as e:
                return f"Generation error: {e}", 0.0

            # Return result to LLM as JSON
            return json.dumps(result, indent=2), 0.0

        registry.add(
            name="generate",
            description=(
                "Run the generation pipeline. This blocks until generation completes, "
                "runs into a shortage, or pauses for user review. Returns a result "
                "object with status and details."
            ),
            parameters={"type": "object", "properties": {}},
            handler=generate,
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

            return f"Orchestrator done: {reason}", 0.0

        registry.add(
            name="done",
            description="Signal that orchestration is complete.",
            parameters={
                "type": "object",
                "properties": {
                    "reason": {
                        "type": "string",
                        "description": "Why orchestration is done (complete, sampling_paused, error)",
                    },
                },
            },
            handler=done,
        )

    def _validate_plan(self, plan: Dict) -> List[str]:
        """Validate plan structure. Returns list of error messages (empty = valid)."""
        errors = []

        if not isinstance(plan.get("pipelines"), dict):
            errors.append("'pipelines' must be an object")
        elif not plan["pipelines"]:
            errors.append("At least one pipeline is required")

        generators = plan.get("generators")
        if not isinstance(generators, list):
            errors.append("'generators' must be an array")
        elif not generators:
            errors.append("At least one generator is required")
        else:
            pipeline_ids = set(plan.get("pipelines", {}).keys())
            bucket_ids = {b.get("id") for b in plan.get("buckets", [])}

            for i, g in enumerate(generators):
                if not g.get("scope"):
                    errors.append(f"Generator {i}: 'scope' is required")

                # If buckets exist, validate bucket_id references
                if bucket_ids and g.get("bucket_id"):
                    if g["bucket_id"] not in bucket_ids:
                        errors.append(
                            f"Generator {i}: bucket_id '{g['bucket_id']}' "
                            f"not found in buckets"
                        )

            # Validate bucket pipeline_id references
            for b in plan.get("buckets", []):
                if b.get("pipeline_id") and b["pipeline_id"] not in pipeline_ids:
                    errors.append(
                        f"Bucket '{b.get('id')}': pipeline_id '{b['pipeline_id']}' "
                        f"not found in pipelines"
                    )

        return errors

    async def run(self) -> AgentResult:
        """
        Run the orchestrator. This is the main entry point.

        The orchestrator will:
        1. Read conversation context (already in system prompt)
        2. Research the domain
        3. Create a plan (set_plan)
        4. Run generation (generate — blocks until complete/shortage/paused)
        5. Handle results and complete
        """
        result = await self._conversation.send(
            "Begin orchestrating dataset generation. "
            "Read the conversation history to understand what dataset is needed, "
            "then research the domain to find sources and understand the landscape.",
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
