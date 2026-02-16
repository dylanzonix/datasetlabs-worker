"""
Orchestrator agent — the brain that coordinates dataset generation.

Reads the conversation history, does research, creates a plan (pipelines +
buckets + generators), and runs generation via set_plan().
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

## Your mission

A user has described a dataset they want through a consultation chat (below).
Your job is to figure out the **optimal strategy** for building it — what
sources to use, how to transform raw material into high-quality rows, and at
what distribution — then execute that strategy.

Strategy is about:

- **Seeds** — anchors for diversity and coverage. Each seed gives a row a
  unique starting point, preventing mode collapse. What carries the diversity
  for this dataset? The right seed form depends on where the diversity lives —
  sometimes it's extracted content because the source material IS the value,
  sometimes it's a reference or pointer because the row generator can look up
  what it needs per-row. The orchestrator should be specific about what seeds
  are and where they come from, always grounded in real sources.
- **Pipeline** — how to transform each seed into a final row. The row generator
  is a full agent with browsing, search, code execution, and randomization
  tools. Pipeline instructions can delegate significant work — looking up
  details within a seed's scope, exploring sources, applying methodology,
  synthesizing content. Don't pre-resolve work at the seed level that the row
  generator could handle per-row.

The golden case is finding source material that is **higher quality than what
you could generate yourself**. Experts, professionals, and curated real-world
content often surpass what you'd produce — authoritative references, domain
experts, proven creative work, vetted datasets. When such sources exist, use
them. When they don't, lean on synthesis but be deliberate about injecting
quality through strong pipeline instructions and research-backed domain
knowledge. Not all real data is good — seek the best, not just the real.

## Synthetic diversity risk

Not all datasets face the same risk from synthetic generation. Use this
reference when reasoning about your strategy — it tells you where to invest
in diversity engineering vs. where correctness constraints naturally suffice.

| Output type | Diversity risk | Why |
|---|---|---|
| Code, math, formal reasoning | LOW | Verifiable via execution/checks |
| Structured outputs (JSON, SQL, tool calls) | VERY LOW | Schema validation dominates |
| Information extraction & normalization | LOW | Deterministic validation possible |
| Instruction following / assistant behavior | MEDIUM | Style homogenization risk |
| Technical explanations / tutoring | MEDIUM | Tone collapses easily |
| Customer support / sales simulations | MEDIUM-HIGH | Intent accuracy easy; realism harder |
| Long-form reports & summaries | HIGH | Structure holds; discourse diversity collapses |
| Game dialogue / character writing | HIGH | Needs persona anchoring |
| Creative fiction / storytelling | VERY HIGH | Trope attractors + narrator voice collapse |
| Natural multi-turn conversation | VERY HIGH | Social nuance hard to synthesize |
| Humor / poetry / cultural voice | EXTREME | Highest human-signal dependency |

For high-risk areas, diversity must be deliberately engineered — primarily
through seed variety from real sources carrying genuine differences, and
secondarily through controlled randomization (the rng tool) for dimensions
where seed-level variety isn't sufficient or the model can genuinely go
either way.

## Resources

{resources_section}
- Web search and browsing via research agents
- Code execution (Python) via research agents
- Generator agents for extracting seeds from sources

## Column schema
{columns_description}

## Conversation history
{conversation_summary}

## How to work

1. **Strategize first.** Call strategy() with your strategic analysis before
   any other tool:
   - What does quality look like for this dataset — per-row and overall?
   - Where does the diversity live? What should seeds carry?
   - What are the possible approaches? If there are meaningfully different
     strategies (different seed designs, source mixes, pipeline splits),
     reason about the tradeoffs. If the approach is obvious, explain why.
   - What could go wrong? What are the risks with your chosen approach?
   - What do you need to confirm about sources before committing?

2. **Research to confirm.** Use research() to ground your strategy in specifics.
   Even with strong intuitions, confirm them — what specific sources exist,
   what does their content look like, are they rich enough for your target
   count? Use ask_research() to drill deeper where source structure matters
   for extraction. Resolve any uncertainty here, not through trial-and-error.

3. **Set the plan and go.** Once your strategy is confirmed, call set_plan()
   to start generation immediately:
   - Pipeline instructions telling the row generator exactly how to transform
     seeds into rows — leverage the row generator's full capabilities (browsing,
     search, code execution, rng) in these instructions
   - Generators with specific scopes for seed extraction
   - Buckets for distribution (if needed)
   The system generates sample rows first and pauses for user review.

## Tools

**strategy(analysis)** — Record your strategic analysis. Call this first, before
researching or planning. Must include: quality definition, diversity analysis,
approach reasoning (consider alternatives if non-obvious), and risk assessment.

**research(task)** — Spawn a research agent to investigate a specific question.
Give it a precise, answerable task — not a broad survey. Use ask_research(
agent_id, question) for follow-ups to dig deeper into specifics.

**set_plan(plan)** — Set and start the generation plan. Begins generation
immediately. The system generates sample rows and pauses for user review.

**add_generator(generator)** — Add a generator to the current plan. Use for
shortage recovery — add targeted generators for buckets that are short.

**done(reason)** — Signal orchestration is complete.

## Plan structure

A plan has three parts:

**Pipelines** — Instructions for turning seeds into rows. Tell the row
generator exactly how to interpret the seed and what to produce. The row
generator is a full agent with browsing, search, code execution, and rng
tools — pipeline instructions can leverage all of these. Use one pipeline
unless categories need fundamentally different processing; a single pipeline
can include conditional logic for moderate variation.

**Buckets** (optional) — Distribution controls. Each bucket has a weight and
maps to a pipeline. Skip entirely for uniform datasets.

**Generators** — Extraction agents that find sources and yield seeds. Each has
a scope (what to extract and from where), seed_description (what each seed
should look like), target_count, and optional bucket_id. Generators are rough
extractors; the pipeline handles transformation and enrichment.

```json
{{
  "pipelines": {{
    "main": {{
      "instructions": "How to transform each seed into a row..."
    }}
  }},
  "buckets": [
    {{"id": "primary", "label": "Primary sources", "weight": 0.7, "pipeline_id": "main"}}
  ],
  "generators": [
    {{
      "scope": "What to extract and from where — be specific",
      "seed_description": "What each seed should look like",
      "target_count": 200,
      "bucket_id": "primary"
    }}
  ]
}}
```

## Principles

- **Quality first.** Seek source material that exceeds what you'd synthesize.
  Not all real data is good — aim for the highest quality sources available.
- **Seeds are for diversity and coverage.** What carries the diversity for this
  dataset? Design seeds around that. Real sources over synthetic.
- **The row generator is capable.** It has full browsing, search, code execution,
  and randomization tools. If per-seed exploration or lookup is needed, the
  pipeline can handle it — don't pre-resolve work at the seed level that
  belongs in the pipeline.
- **Think, confirm, execute.** Reason about the best strategy, use research to
  confirm specifics, then commit. Don't over-research what you're confident
  about.
- **No licensing concerns.** Our legal team has confirmed all content is free
  to use in our jurisdiction regardless of license or copyright. Do not waste
  time researching licenses or avoiding copyrighted material — focus on quality.
- **Coverage and diversity matter.** A good strategy produces varied, well-
  distributed rows, not just individually good ones. Consult the synthetic
  diversity risk table to calibrate how much diversity engineering is needed.
- The simplest strategy that produces quality results wins.
- Generators yield raw text. The pipeline handles transformation and filtering.
- When set_plan() returns "shortage", add more generators and run again.
- When set_plan() returns "sampling_paused", call done("sampling_paused").
- Target: {num_samples} rows.
"""


class OrchestratorAgent:
    """
    Main orchestrator. Reads conversation, researches, creates plan,
    and runs generation via set_plan().

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
        previous_recipe: Optional[str] = None,
        blob_service_client: Optional[Any] = None,
        project_id: Optional[Any] = None,
        on_tool_call: Optional[Callable[[str, str], None]] = None,
        mcp_tools: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        self.chat_history = chat_history
        self.previous_recipe = previous_recipe
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
        self.blob_service_client = blob_service_client
        self.project_id = project_id
        self.on_tool_call = on_tool_call
        self.mcp_tools = mcp_tools or []

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

        if self.previous_recipe:
            convo_summary += (
                f"\n\n## Previous plan (user requested changes)\n"
                f"```json\n{self.previous_recipe}\n```\n"
                f"The user has provided feedback on the sample rows. Review the "
                f"conversation history for their feedback, then adjust the pipeline "
                f"instructions accordingly and call set_plan()."
            )

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
                details = f"json_schema defined"
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
        """Register orchestrator tools."""

        # --- strategy (think tool) ---
        async def strategy(args: Dict) -> tuple[str, float]:
            analysis = args.get("analysis", "")
            logger.info(f"[Orchestrator] strategy() called ({len(analysis)} chars)")
            return "Strategy recorded. Proceed with research or set_plan().", 0.0

        registry.add(
            name="strategy",
            description=(
                "Record your strategic analysis before researching or planning. "
                "Must include: quality definition, diversity analysis, approach "
                "reasoning (consider alternatives when non-obvious), and risk "
                "assessment. Call this BEFORE research() or set_plan()."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "analysis": {
                        "type": "string",
                        "description": (
                            "Your strategic reasoning. Cover: quality definition, "
                            "diversity analysis, approach reasoning (with alternatives "
                            "if non-obvious), and what could go wrong."
                        ),
                    },
                },
                "required": ["analysis"],
            },
            handler=strategy,
        )

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
                blob_service_client=self.blob_service_client,
                project_id=self.project_id,
                on_tool_call=self.on_tool_call,
                mcp_tools=self.mcp_tools,
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

            # Extract new_version flag (top-level arg, not part of plan data)
            new_version = args.get("new_version", False)

            # Validate plan structure
            errors = self._validate_plan(plan)
            if errors:
                return f"Plan validation failed:\n" + "\n".join(f"- {e}" for e in errors), 0.0

            # Store new_version flag in plan metadata for job_processor
            if new_version:
                plan["_new_version"] = True

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
                gen_id = g.get("id", "gen")
                summary_parts.append(
                    f"  - {gen_id}: scope=\"{g.get('scope', '')[:60]}\" "
                    f"target={g.get('target_count', 100)} bucket={g.get('bucket_id', 'none')}"
                )

            # === Start generation ===
            if not self.on_generate:
                summary_parts.append("\nWarning: no on_generate callback — generation skipped.")
                return "\n".join(summary_parts), 0.0

            # Create a fresh queue for this generation run
            self.seed_queue = asyncio.Queue()

            # Create future for the result
            result_future: asyncio.Future = asyncio.get_event_loop().create_future()

            # Signal job_processor to start the generation consumer
            await self.on_generate(self.plan, self.seed_queue, result_future)

            # Start all generator agents — they feed the seed queue
            for gen_config in generators:
                gen_id = gen_config.get("id", self._new_id())

                # Skip generators that are already running
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
                    blob_service_client=self.blob_service_client,
                    project_id=self.project_id,
                    on_tool_call=self.on_tool_call,
                    mcp_tools=self.mcp_tools,
                )
                self._generators[gen_id] = gen

                task = asyncio.create_task(gen.run())
                self._generator_tasks[gen_id] = task

            logger.info(
                f"[Orchestrator] set_plan: "
                f"{len(generators)} generators started, awaiting result..."
            )

            # Block until the consumer resolves the future
            try:
                result = await result_future
            except Exception as e:
                return f"Generation error: {e}", 0.0

            summary_parts.append(f"\n## Generation result\n{json.dumps(result, indent=2)}")

            # Auto-exit on sampling_paused
            if result.get("status") == "sampling_paused":
                self._is_done = True

            return "\n".join(summary_parts), 0.0

        registry.add(
            name="set_plan",
            description=(
                "Set and start the generation plan. Begins generation immediately. "
                "The system generates sample rows and pauses for user review."
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
                                    "required": ["scope", "seed_description"],
                                },
                            },
                            "new_version": {
                                "type": "boolean",
                                "description": "If true, creates a new dataset version instead of appending to the current one. Use when user feedback requires fundamentally different data (e.g. different schema, different approach). Default false.",
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

            if not generator.get("scope"):
                return "Error: generator needs 'scope'.", 0.0

            # Append to current plan (no version bump)
            self.plan["generators"].append(generator)

            gen_id = generator.get("id", "new_gen")
            return (
                f"Generator '{gen_id}' added to plan. "
                f"Now {len(self.plan['generators'])} generators total. "
                f"Call set_plan() again to restart generation with the updated plan."
            ), 0.0

        registry.add(
            name="add_generator",
            description=(
                "Add a generator to the current plan. Use for shortage recovery — "
                "add targeted generators for buckets that are short, then call set_plan() again."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "generator": {
                        "type": "object",
                        "description": "Generator definition with scope and seed_description",
                        "properties": {
                            "id": {"type": "string"},
                            "scope": {"type": "string"},
                            "seed_description": {"type": "string"},
                            "target_count": {"type": "integer"},
                            "bucket_id": {"type": "string"},
                        },
                        "required": ["scope", "seed_description"],
                    },
                },
                "required": ["generator"],
            },
            handler=add_generator,
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
            "Begin. Read the conversation history, then call strategy() with "
            "your strategic analysis.",
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
