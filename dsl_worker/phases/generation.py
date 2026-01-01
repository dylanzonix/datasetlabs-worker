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

from dsl_worker.phases.base import Phase
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

        assigned_seeds = self.assignment_phase.get_assigned_seeds()
        if not assigned_seeds:
            return False

        # Generate until we hit num_samples (cycling through seeds if needed)
        return self.state.samples_generated < self.state.num_samples

    async def execute_once(self) -> bool:
        """Generate ONE sample."""
        assigned_seeds = self.assignment_phase.get_assigned_seeds()
        if not assigned_seeds:
            return False

        samples_generated = self.state.samples_generated
        target = self.state.num_samples

        if samples_generated >= target:
            return False

        # Cycle through seeds if we have fewer seeds than target samples
        seed_idx = samples_generated % len(assigned_seeds)
        assigned_seed = assigned_seeds[seed_idx]

        logger.info(f"[{self.name}] Generating sample {samples_generated + 1}/{target} (seed {seed_idx + 1}/{len(assigned_seeds)})")

        try:
            # Generate the sample
            sample_data = await self._generate_sample(assigned_seed)

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
                return True

            logger.warning(f"[{self.name}] Failed to generate sample from seed {assigned_seed.seed_id}")
            return False

        except Exception as e:
            logger.error(f"[{self.name}] Generation error: {e}", exc_info=True)
            return False

    async def _generate_sample(self, assigned_seed: AssignedSeed) -> Optional[Dict]:
        """Generate a single sample using the agentic approach."""
        # Build system prompt
        system_prompt = self.system_prompt_template.format(
            row_instructions=self.state.generation_prompt,
            column_schema=self._format_column_schema(),
            seed=assigned_seed.seed_text,
            diversity_targets=json.dumps(assigned_seed.diversity_assignments, indent=2)
        )

        messages = [{"role": "system", "content": system_prompt}]
        max_iterations = 10

        for i in range(max_iterations):
            response = await self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                tools=self.tools,
                tool_choice="auto"
            )

            message = response.choices[0].message
            messages.append(message.model_dump())

            if message.tool_calls:
                for tool_call in message.tool_calls:
                    name = tool_call.function.name
                    args = json.loads(tool_call.function.arguments)

                    logger.debug(f"Agent called tool: {name}")

                    # Handle generate_row (final output)
                    if name == "generate_row":
                        return args.get("row", args)

                    # Handle other tools
                    result = await self._handle_tool_call(name, args)

                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": result
                    })
            else:
                # No tool call - agent finished without generating
                logger.warning("Agent finished without calling generate_row")
                break

        logger.error("Agent exceeded max iterations")
        return None

    async def _handle_tool_call(self, name: str, args: Dict) -> str:
        """Handle tool calls from the agent."""
        if name == "rag_search":
            query = args.get("query", "")
            results = await self._rag_search(query)
            return json.dumps({"results": results})

        elif name == "web_search":
            if not self.state.use_internet:
                return json.dumps({"error": "Web search disabled for this project"})
            # TODO: Implement actual web search
            return json.dumps({"results": []})

        return json.dumps({})

    async def _rag_search(self, query: str, top_k: int = 5) -> List[Dict]:
        """Search for relevant chunks using vector similarity."""
        try:
            # Get query embedding
            response = await self.openai_client.embeddings.create(
                model="text-embedding-3-large",
                input=query
            )
            query_embedding = response.data[0].embedding

            # Vector search using pgvector cosine distance
            results = (
                self.db.query(ProjectRagChunk)
                .filter(ProjectRagChunk.project_id == self.state.project_id)
                .order_by(
                    ProjectRagChunk.embedding.cosine_distance(query_embedding)
                )
                .limit(top_k)
                .all()
            )

            return [
                {
                    "text": chunk.text[:1000],  # Truncate for context window
                    "chunk_id": str(chunk.id)
                }
                for chunk in results
            ]

        except Exception as e:
            logger.error(f"RAG search failed: {e}")
            return []

    def _format_column_schema(self) -> str:
        """Format the column schema for the LLM prompt."""
        if not self.state.columns:
            return "No specific schema defined"

        lines = []
        for col in self.state.columns:
            col_name = col.get('name', 'unknown')
            col_type = col.get('type', 'string')
            col_desc = col.get('description', '')
            lines.append(f"{col_name} ({col_type}): {col_desc}")

        return "\n".join(lines)

    def is_complete(self) -> bool:
        """Complete when we've generated target number of samples."""
        if not self.assignment_phase or not self.assignment_phase.is_complete():
            return False

        assigned_seeds = self.assignment_phase.get_assigned_seeds()
        if not assigned_seeds:
            return False

        return self.state.samples_generated >= self.state.num_samples

    def reset(self):
        """Reset generation state (for fresh start on resume)."""
        self._current_index = 0