"""
Phase: Sample Generation

Generates dataset samples using an agent-based approach with tool calling.
This phase is EPHEMERAL - it starts fresh on each resume.

Key design:
- Gets assigned seeds from SeedAssignmentPhase
- Generates samples one at a time
- Tracks generated count via Sample table (for progress display)
- Does NOT resume from where it left off - regenerates from scratch
"""

import logging
import json
import uuid
from typing import List, Dict, Optional
from datetime import datetime, timezone

from sqlalchemy import func as sql_func

from dsl_worker.phases.base import Phase, PhaseResult
from dsl_worker.phases.seed_assignment import SeedAssignmentPhase, AssignedSeed
from dsl_api.models.sample import Sample
from dsl_api.models.project_rag_chunk import ProjectRagChunk

logger = logging.getLogger(__name__)


class GenerationPhase(Phase):
    """
    Generate samples from assigned seeds.

    Uses agentic LLM with tool calling to:
    1. Optionally search for additional context (RAG, web)
    2. Generate the final sample matching the schema

    One execute_once() generates ONE sample.
    """

    def __init__(self, *args, assignment_phase: Optional[SeedAssignmentPhase] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.assignment_phase = assignment_phase
        self._current_index = 0  # Track which seed we're on

        # System prompt template
        self.system_prompt_template = """You are a dataset row generator. Your job is to generate a single row for a dataset.

You have access to tools to help you gather information. Use them if needed. When you have everything you need, call generate_row with the final content.

## Row Instructions
<row_instructions>
{row_instructions}
</row_instructions>

## Column Schema
<column_schema>
{column_schema}
</column_schema>

## Seed
This is your starting point - source material to build from:
<seed>
{seed}
</seed>

## Diversity Targets
The row should have this flavor:
<diversity_targets>
{diversity_targets}
</diversity_targets>

## Your Task
Use the tools to gather what you need, then call generate_row when ready. You may call generate_row immediately if the seed is sufficient."""

        # System prompt for no-seed generation
        self.system_prompt_no_seed_template = """You are a dataset row generator. Your job is to generate a single row for a dataset.

You have access to tools to help you gather information. Use them if needed. When you have everything you need, call generate_row with the final content.

## Row Instructions
<row_instructions>
{row_instructions}
</row_instructions>

## Column Schema
<column_schema>
{column_schema}
</column_schema>

## Diversity Targets
The row should have this flavor:
<diversity_targets>
{diversity_targets}
</diversity_targets>

## Your Task
Generate a creative and diverse row based on the instructions and schema above. Use tools to gather information if helpful, then call generate_row when ready."""

        # Tools for the agent
        self.tools = [
            {
                "type": "function",
                "function": {
                    "name": "rag_search",
                    "description": "Search source documents for relevant information",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "Search query"}
                        },
                        "required": ["query"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "generate_row",
                    "description": "Generate the final row. Call this when you have everything you need.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "row": {"type": "object", "description": "The complete row matching the column schema"}
                        },
                        "required": ["row"]
                    }
                }
            }
        ]

        # Add web search tool if enabled
        if self.state.use_internet:
            self.tools.insert(1, {
                "type": "function",
                "function": {
                    "name": "web_search",
                    "description": "Search the web for information",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "Search query"}
                        },
                        "required": ["query"]
                    }
                }
            })

    def should_run(self) -> bool:
        """Run if assignment is complete and we haven't generated all samples."""
        if not self.assignment_phase or not self.assignment_phase.is_complete():
            return False

        # Generate until we hit num_samples
        return self.state.samples_generated < self.state.num_samples

    async def execute_once(self) -> PhaseResult:
        """Generate ONE sample."""
        assigned_seeds = self.assignment_phase.get_assigned_seeds()

        samples_generated = self.state.samples_generated
        target = self.state.num_samples

        if samples_generated >= target:
            return PhaseResult.no_work()

        # Handle no-seed case: create synthetic empty seed
        if not assigned_seeds:
            assigned_seed = self._create_synthetic_seed(samples_generated)
            logger.info(f"[{self.name}] Generating sample {samples_generated + 1}/{target} (no source seeds)")
        else:
            # Cycle through seeds if we have fewer seeds than target samples
            seed_idx = samples_generated % len(assigned_seeds)
            assigned_seed = assigned_seeds[seed_idx]
            logger.info(f"[{self.name}] Generating sample {samples_generated + 1}/{target} (seed {seed_idx + 1}/{len(assigned_seeds)})")

        try:
            # Generate the sample (with cost tracking)
            sample_data, cost_usd = await self._generate_sample(assigned_seed)

            if sample_data:
                # Get next sequence number
                max_seq = (
                    self.db.query(sql_func.max(Sample.seq))
                    .filter(Sample.project_id == self.state.project_id)
                    .scalar() or 0
                )

                # Create sample record
                sample = Sample(
                    id=uuid.uuid4(),
                    project_id=self.state.project_id,
                    run_id=self.state.run_id,
                    seq=max_seq + 1,
                    row=sample_data,
                    tags=assigned_seed.diversity_assignments
                )
                self.db.add(sample)
                self.db.commit()

                logger.info(f"[{self.name}] Generated sample {max_seq + 1}")
                return PhaseResult.work_done(cost_usd=cost_usd)

            logger.warning(f"[{self.name}] Failed to generate sample from seed {assigned_seed.seed_id}")
            return PhaseResult(did_work=False, cost_usd=cost_usd)

        except Exception as e:
            logger.error(f"[{self.name}] Generation error: {e}", exc_info=True)
            return PhaseResult.no_work()

    def _create_synthetic_seed(self, sample_index: int) -> AssignedSeed:
        """Create a synthetic seed for generation when no source files exist."""
        # If there's a diversity spec, cycle through combinations
        diversity_assignments = {}
        if self.state.diversity_spec:
            diversity_assignments = self._get_diversity_assignment_for_index(sample_index)

        return AssignedSeed(
            seed_id=f"synthetic-{sample_index}",
            seed_text="",  # Empty seed - generation from scratch
            diversity_assignments=diversity_assignments,
            score=1.0
        )

    def _get_diversity_assignment_for_index(self, index: int) -> Dict[str, str]:
        """
        Get a diversity assignment for a given index.
        Cycles through all combinations based on index.
        """
        if not self.state.diversity_spec:
            return {}

        assignments = {}
        remaining_index = index

        for axis in self.state.diversity_spec:
            axis_name = axis.get("name")
            values = [v.get("value") for v in axis.get("values", [])]
            if values:
                value_idx = remaining_index % len(values)
                assignments[axis_name] = values[value_idx]
                remaining_index //= len(values)

        return assignments

    async def _generate_sample(self, assigned_seed: AssignedSeed) -> tuple[Optional[Dict], float]:
        """
        Generate a single sample using the agentic approach.

        Returns:
            Tuple of (sample_data, cost_usd)
        """
        total_cost_usd = 0.0

        # Choose prompt based on whether we have a seed
        if assigned_seed.seed_text:
            system_prompt = self.system_prompt_template.format(
                row_instructions=self.state.generation_prompt,
                column_schema=self._format_column_schema(),
                seed=assigned_seed.seed_text,
                diversity_targets=json.dumps(assigned_seed.diversity_assignments, indent=2)
            )
        else:
            system_prompt = self.system_prompt_no_seed_template.format(
                row_instructions=self.state.generation_prompt,
                column_schema=self._format_column_schema(),
                diversity_targets=json.dumps(assigned_seed.diversity_assignments, indent=2)
            )

        messages = [{"role": "system", "content": system_prompt}]
        max_iterations = 10

        for i in range(max_iterations):
            result = await self.openai_client.chat_completion(
                model="gpt-4o",
                messages=messages,
                tools=self.tools,
                tool_choice="auto"
            )

            total_cost_usd += result.cost.total_cost_usd
            message = result.response.choices[0].message
            messages.append(message.model_dump())

            if message.tool_calls:
                for tool_call in message.tool_calls:
                    name = tool_call.function.name
                    args = json.loads(tool_call.function.arguments)

                    logger.debug(f"Agent called tool: {name}")

                    # Handle generate_row (final output)
                    if name == "generate_row":
                        return args.get("row", args), total_cost_usd

                    # Handle other tools
                    tool_result, tool_cost = await self._handle_tool_call(name, args)
                    total_cost_usd += tool_cost

                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": tool_result
                    })
            else:
                # No tool call - agent finished without generating
                logger.warning("Agent finished without calling generate_row")
                break

        logger.error("Agent exceeded max iterations")
        return None, total_cost_usd

    async def _handle_tool_call(self, name: str, args: Dict) -> tuple[str, float]:
        """
        Handle tool calls from the agent.

        Returns:
            Tuple of (result_string, cost_usd)
        """
        if name == "rag_search":
            return await self._rag_search(args.get("query", ""))
        elif name == "web_search":
            return await self._web_search(args.get("query", ""))
        else:
            return f"Unknown tool: {name}", 0.0

    async def _rag_search(self, query: str) -> tuple[str, float]:
        """
        Search RAG chunks for relevant content.

        Returns:
            Tuple of (result_string, cost_usd)
        """
        try:
            # Get embedding for query
            result = await self.openai_client.create_embeddings(
                model="text-embedding-3-small",
                input=[query],
            )
            cost_usd = result.cost.total_cost_usd
            query_embedding = result.response.data[0].embedding

            # Convert embedding list to pgvector string format
            embedding_str = "[" + ",".join(str(x) for x in query_embedding) + "]"

            # Search chunks using pgvector
            # Use CAST() instead of :: to avoid SQLAlchemy parameter parsing issues
            from sqlalchemy import text

            db_result = self.db.execute(
                text("""
                    SELECT text, 1 - (embedding <=> CAST(:embedding AS vector)) as similarity
                    FROM project_rag_chunks
                    WHERE project_id = :project_id
                    ORDER BY embedding <=> CAST(:embedding AS vector)
                    LIMIT 3
                """),
                {
                    "embedding": embedding_str,
                    "project_id": str(self.state.project_id)
                }
            )

            chunks = db_result.fetchall()
            if not chunks:
                return "No relevant content found", cost_usd

            results = []
            for chunk in chunks:
                results.append(f"[Similarity: {chunk.similarity:.3f}]\n{chunk.text}")

            return "\n\n---\n\n".join(results), cost_usd

        except Exception as e:
            logger.error(f"RAG search failed: {e}")
            return f"Search failed: {e}", 0.0

    async def _web_search(self, query: str) -> tuple[str, float]:
        """
        Placeholder for web search.

        Returns:
            Tuple of (result_string, cost_usd)
        """
        # TODO: Implement actual web search
        return f"Web search not implemented yet. Query was: {query}", 0.0

    def _format_column_schema(self) -> str:
        """Format the column schema for the LLM prompt."""
        if not self.state.columns:
            return "No specific schema defined - generate appropriate fields"

        lines = []
        for col in self.state.columns:
            col_name = col.get('name', 'unknown')
            col_type = col.get('type', 'string')
            col_desc = col.get('description', '')
            lines.append(f"{col_name} ({col_type}): {col_desc}")

        return "\n".join(lines)

    def is_complete(self) -> bool:
        """Complete when we've generated the target number of samples."""
        return self.state.samples_generated >= self.state.num_samples