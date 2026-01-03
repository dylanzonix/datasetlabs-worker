"""
Phase: Sample Generation

Generates dataset samples using an agent-based approach with tool calling.
This phase is EPHEMERAL - it starts fresh on each resume.

Key design:
- Gets assigned seeds from SeedAssignmentPhase
- Generates samples one at a time using incremental row building
- Tools: append, submit_row, rag_search, web_search, crawl
"""

import asyncio
import logging
import json
import uuid
import os
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone

import requests
from scrapingbee import ScrapingBeeClient
from sqlalchemy import func as sql_func, text
from openai import OpenAI

from dsl_worker.phases.base import Phase, PhaseResult
from dsl_worker.phases.seed_assignment import SeedAssignmentPhase, AssignedSeed
from dsl_api.models.sample import Sample

logger = logging.getLogger(__name__)


class GenerationPhase(Phase):
    """
    Generate samples from assigned seeds using agentic tool calling.

    Tools:
    - append: Build row incrementally (string concat, list append, or set value)
    - submit_row: Finalize and validate the row
    - rag_search: Search user's uploaded documents
    - web_search: Search web via Brave Search API
    - crawl: Fetch full page content via ScrapingBee
    """

    def __init__(
        self, *args, assignment_phase: Optional[SeedAssignmentPhase] = None, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.assignment_phase = assignment_phase

        # Row state (reset per sample)
        self._current_row: Dict[str, Any] = {}
        self._row_submitted: bool = False

        # Config
        self.max_iterations = 20
        self.brave_api_key = os.getenv("BRAVE_API_KEY")
        self.scrapingbee_api_key = os.getenv("SCRAPINGBEE_API_KEY")

    # =========================================================================
    # Phase interface
    # =========================================================================

    def should_run(self) -> bool:
        """Run if assignment is complete and we haven't generated all samples."""
        if not self.assignment_phase or not self.assignment_phase.is_complete():
            return False
        return self.state.samples_generated < self.state.num_samples

    async def execute_once(self) -> PhaseResult:
        """Generate ONE sample."""
        assigned_seeds = self.assignment_phase.get_assigned_seeds()

        samples_generated = self.state.samples_generated
        target = self.state.num_samples

        if samples_generated >= target:
            return PhaseResult.no_work()

        # Handle no-seed case
        if not assigned_seeds:
            assigned_seed = self._create_synthetic_seed(samples_generated)
            logger.info(
                f"[{self.name}] Generating sample {samples_generated + 1}/{target} (no source seeds)"
            )
        else:
            seed_idx = samples_generated % len(assigned_seeds)
            assigned_seed = assigned_seeds[seed_idx]
            logger.info(
                f"[{self.name}] Generating sample {samples_generated + 1}/{target} (seed {seed_idx + 1}/{len(assigned_seeds)})"
            )

        try:
            sample_data, cost_usd = await self._generate_sample(assigned_seed)

            if sample_data:
                max_seq = (
                        self.db.query(sql_func.max(Sample.seq))
                        .filter(Sample.project_id == self.state.project_id)
                        .scalar()
                        or 0
                )

                sample = Sample(
                    id=uuid.uuid4(),
                    project_id=self.state.project_id,
                    run_id=self.state.run_id,
                    seq=max_seq + 1,
                    row=sample_data,
                    tags=assigned_seed.diversity_assignments,
                )
                self.db.add(sample)
                self.db.commit()

                logger.info(f"[{self.name}] Generated sample {max_seq + 1} (cost: ${cost_usd:.4f})")
                return PhaseResult.work_done(cost_usd=cost_usd)

            logger.warning(
                f"[{self.name}] Failed to generate sample from seed {assigned_seed.seed_id}"
            )
            # Return cost even on failure - we still spent money on API calls
            return PhaseResult.work_done(cost_usd=cost_usd) if cost_usd > 0 else PhaseResult.no_work()

        except Exception as e:
            logger.error(f"[{self.name}] Generation error: {e}", exc_info=True)
            return PhaseResult.no_work()

    def is_complete(self) -> bool:
        """Complete when we've generated the target number of samples."""
        return self.state.samples_generated >= self.state.num_samples

    # =========================================================================
    # Sample generation
    # =========================================================================

    async def _generate_sample(self, assigned_seed: AssignedSeed) -> tuple[Optional[Dict], float]:
        """
        Generate a single sample using the agentic approach.

        Returns:
            Tuple of (sample_data, total_cost_usd)
            sample_data is None if generation failed
        """
        # Reset state for this sample
        self._current_row = {}
        self._row_submitted = False
        self._generation_cost_usd = 0.0  # Track cost for this sample

        system_prompt = self._build_system_prompt(assigned_seed)
        input_items = [{"role": "system", "content": system_prompt}]
        tools = self._build_tools()

        for iteration in range(self.max_iterations):
            logger.debug(f"Generation iteration {iteration + 1}")

            try:
                response, cost = await self.openai_client.responses_create(
                    model="gpt-4o",
                    input=input_items,
                    tools=tools,
                    max_output_tokens=100_000,
                )
                self._generation_cost_usd += cost.total_cost_usd
            except Exception as e:
                logger.error(f"OpenAI API error: {e}")
                return None, self._generation_cost_usd

            has_tool_calls = False

            for output_item in response.output:
                if output_item.type == "message":
                    input_items.append(
                        {
                            "role": "assistant",
                            "content": (
                                output_item.content[0].text
                                if output_item.content
                                else ""
                            ),
                        }
                    )

                elif output_item.type == "function_call":
                    has_tool_calls = True

                    name = output_item.name
                    args = json.loads(output_item.arguments)
                    call_id = output_item.call_id

                    logger.debug(f"Tool call: {name}({args})")

                    result = await self._handle_tool_call(name, args)

                    input_items.append(
                        {
                            "type": "function_call",
                            "call_id": call_id,
                            "name": name,
                            "arguments": output_item.arguments,
                        }
                    )
                    input_items.append(
                        {
                            "type": "function_call_output",
                            "call_id": call_id,
                            "output": result,
                        }
                    )

                    if self._row_submitted:
                        logger.info(f"Row submitted after {iteration + 1} iterations")
                        return self._current_row, self._generation_cost_usd

            if not has_tool_calls:
                logger.warning("Agent finished without submitting row")
                break

        logger.error(
            f"Agent exceeded {self.max_iterations} iterations without submitting"
        )
        return None, self._generation_cost_usd

    # =========================================================================
    # System prompt
    # =========================================================================

    def _build_system_prompt(self, assigned_seed: AssignedSeed) -> str:
        """Build the system prompt for generation."""

        column_schema = self._format_column_schema()
        diversity_targets = (
            json.dumps(assigned_seed.diversity_assignments, indent=2)
            if assigned_seed.diversity_assignments
            else "None"
        )

        # Base prompt
        prompt = f"""You are a dataset row generator. Your job is to generate a single high-quality row for a dataset.

## Tools Available
- append(column, content): Add content to a column
  - string columns: concatenates text (call once or multiple times to build incrementally)
  - list columns: adds an item to the array (call once per item)
  - int, float, bool, enum, dict columns: sets the value (call once)
- submit_row(): Finalize and submit the row when complete
- rag_search(query): Search the uploaded source documents
- web_search(query): Search the web (returns titles, snippets, URLs)
- crawl(url): Fetch full page content from a URL

## Row Instructions
<row_instructions>
{self.state.generation_prompt}
</row_instructions>

## Column Schema
<column_schema>
{column_schema}
</column_schema>

## Diversity Targets
<diversity_targets>
{diversity_targets}
</diversity_targets>
"""

        # Add seed if present
        if assigned_seed.seed_text:
            prompt += f"""
## Seed
This is your starting point - source material to build from:
<seed>
{assigned_seed.seed_text}
</seed>
"""

        prompt += """
## Your Task
1. Use tools to gather any information you need
2. Use append() to build each column
3. Call submit_row() when the row is complete

Build a high-quality, accurate row that follows the instructions and schema."""

        return prompt

    def _format_column_schema(self) -> str:
        """Format the column schema for the LLM prompt."""
        if not self.state.columns:
            return "No specific schema defined - generate appropriate fields"

        lines = []
        for col in self.state.columns:
            col_name = col.get("name", "unknown")
            col_type = col.get("type", "string")
            col_desc = col.get("description", "")

            line = f"- {col_name} ({col_type})"
            if col_desc:
                line += f": {col_desc}"

            # Add enum values if applicable
            if col_type == "enum" and col.get("enum_values"):
                line += f" [values: {', '.join(col['enum_values'])}]"

            # Add list item type if applicable
            # if col_type == "list" and col.get("item_type"):
            #     line += f" [items: {col['item_type']}]"

            lines.append(line)

        return "\n".join(lines)

    # =========================================================================
    # Tools definition
    # =========================================================================

    def _build_tools(self) -> List[Dict]:
        """Build the tools list based on project config."""
        tools = [
            {
                "type": "function",
                "name": "append",
                "description": """Add content to a column. Behavior by type:
- string: concatenates to existing text (call once with full content, or multiple times to build incrementally)
- list: adds an item to the array (call once per item)
- int, float, bool, enum, dict: sets the value (call once)""",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "column": {"type": "string", "description": "The column name"},
                        "content": {"description": "The content to add"},
                    },
                    "required": ["column", "content"],
                },
            },
            {
                "type": "function",
                "name": "submit_row",
                "description": "Finalize and submit the completed row. Call when all required columns are populated.",
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
            {
                "type": "function",
                "name": "rag_search",
                "description": "Search the uploaded source documents for relevant information.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query"}
                    },
                    "required": ["query"],
                },
            },
        ]

        # Add web tools if internet is enabled
        if self.state.use_internet:
            tools.append(
                {
                    "type": "function",
                    "name": "web_search",
                    "description": "Search the web for information. Returns titles, snippets, and URLs. Use crawl() to get full page content if needed.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "Search query"}
                        },
                        "required": ["query"],
                    },
                }
            )
            tools.append(
                {
                    "type": "function",
                    "name": "crawl",
                    "description": "Fetch the full content of a web page as markdown.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "url": {"type": "string", "description": "The URL to crawl"}
                        },
                        "required": ["url"],
                    },
                }
            )

        return tools

    # =========================================================================
    # Tool handlers
    # =========================================================================

    async def _handle_tool_call(self, name: str, args: Dict) -> str:
        """Route tool calls to handlers."""
        handlers = {
            "append": self._handle_append,
            "submit_row": self._handle_submit_row,
            "rag_search": self._handle_rag_search,
            "web_search": self._handle_web_search,
            "crawl": self._handle_crawl,
        }

        handler = handlers.get(name)
        if not handler:
            return json.dumps({"error": f"Unknown tool: {name}"})

        try:
            if asyncio.iscoroutinefunction(handler):
                return await handler(**args)
            else:
                return handler(**args)
        except Exception as e:
            logger.error(f"Tool {name} failed: {e}", exc_info=True)
            return json.dumps({"error": str(e)})

    def _handle_append(self, column: str, content: Any) -> str:
        """Append content to a column."""
        col_schema = self._get_column_schema(column)
        if not col_schema:
            return json.dumps({"error": f"Unknown column: {column}"})

        col_type = col_schema.get("type", "string")

        if col_type == "string":
            current = self._current_row.get(column, "")
            self._current_row[column] = current + str(content)
            return json.dumps(
                {
                    "success": True,
                    "column": column,
                    "length": len(self._current_row[column]),
                }
            )

        elif col_type == "list":
            if column not in self._current_row:
                self._current_row[column] = []
            self._current_row[column].append(content)
            return json.dumps(
                {
                    "success": True,
                    "column": column,
                    "items": len(self._current_row[column]),
                }
            )

        else:
            self._current_row[column] = content
            return json.dumps({"success": True, "column": column})

    def _handle_submit_row(self) -> str:
        """Validate and submit the row."""
        missing = self._get_missing_columns()
        if missing:
            return json.dumps(
                {
                    "error": "Missing required columns",
                    "missing": missing,
                    "current_columns": list(self._current_row.keys()),
                }
            )

        errors = self._validate_column_types()
        if errors:
            return json.dumps({"error": "Validation failed", "errors": errors})

        self._row_submitted = True
        return json.dumps({"success": True, "row": self._current_row})

    async def _handle_rag_search(self, query: str) -> str:
        """Search uploaded docs via pgvector."""
        try:
            # Use tracked client for embeddings
            embed_response, embed_cost = await self.openai_client.embeddings_create(
                model="text-embedding-3-large",
                input=[query],
            )
            self._generation_cost_usd += embed_cost.total_cost_usd

            query_embedding = embed_response.data[0].embedding
            embedding_str = "[" + ",".join(str(x) for x in query_embedding) + "]"

            result = self.db.execute(
                text(
                    """
                    SELECT text, 1 - (embedding <=> CAST(:embedding AS vector)) as similarity
                    FROM project_rag_chunks
                    WHERE project_id = :project_id
                    ORDER BY embedding <=> CAST(:embedding AS vector)
                    LIMIT 5
                """
                ),
                {"embedding": embedding_str, "project_id": str(self.state.project_id)},
            )

            chunks = result.fetchall()
            if not chunks:
                return json.dumps(
                    {"results": [], "message": "No relevant content found"}
                )

            results = []
            for chunk in chunks:
                results.append(
                    {
                        "text": chunk.text[:2000],
                        "similarity": round(chunk.similarity, 3),
                    }
                )

            return json.dumps({"results": results})

        except Exception as e:
            logger.error(f"RAG search failed: {e}")
            return json.dumps({"error": f"Search failed: {str(e)}"})

    async def _handle_web_search(self, query: str) -> str:
        """Search web via Brave Search API."""
        if not self.brave_api_key:
            return json.dumps({"error": "Web search not configured"})

        try:
            response = requests.get(
                "https://api.search.brave.com/res/v1/web/search",
                headers={
                    "Accept": "application/json",
                    "Accept-Encoding": "gzip",
                    "X-Subscription-Token": self.brave_api_key,
                },
                params={"q": query, "count": 5},
                timeout=10,
            )
            response.raise_for_status()
            data = response.json()

            results = []
            for item in data.get("web", {}).get("results", [])[:5]:
                results.append(
                    {
                        "title": item.get("title"),
                        "url": item.get("url"),
                        "snippet": item.get("description"),
                    }
                )

            return json.dumps({"results": results})

        except Exception as e:
            logger.error(f"Web search failed: {e}")
            return json.dumps({"error": f"Web search failed: {str(e)}"})

    async def _handle_crawl(self, url: str) -> str:
        """Crawl a web page via ScrapingBee."""
        if not self.scrapingbee_api_key:
            return json.dumps({"error": "Crawl not configured"})

        try:
            client = ScrapingBeeClient(api_key=self.scrapingbee_api_key)
            response = client.get(
                url, params={"render_js": "False", "return_page_source": "True"}
            )

            if response.status_code == 200:
                content = response.content.decode("utf-8", errors="ignore")
                if len(content) > 15000:
                    content = content[:15000] + "\n\n[Content truncated...]"
                return json.dumps({"url": url, "content": content})
            else:
                return json.dumps(
                    {"error": f"Crawl failed with status {response.status_code}"}
                )

        except Exception as e:
            logger.error(f"Crawl failed: {e}")
            return json.dumps({"error": f"Crawl failed: {str(e)}"})

    # =========================================================================
    # Helpers
    # =========================================================================

    def _get_column_schema(self, column_name: str) -> Optional[Dict]:
        """Get schema for a column by name."""
        for col in self.state.columns or []:
            if col.get("name") == column_name:
                return col
        return None

    def _get_missing_columns(self) -> List[str]:
        """Get list of required columns that are missing."""
        missing = []
        for col in self.state.columns or []:
            col_name = col.get("name")
            if col_name not in self._current_row:
                missing.append(col_name)
        return missing

    def _validate_column_types(self) -> List[str]:
        """Validate column types. Returns list of error messages."""
        errors = []

        for col in self.state.columns or []:
            col_name = col.get("name")
            col_type = col.get("type", "string")

            if col_name not in self._current_row:
                continue

            value = self._current_row[col_name]

            if col_type == "string" and not isinstance(value, str):
                errors.append(
                    f"{col_name}: expected string, got {type(value).__name__}"
                )
            elif col_type == "int" and not isinstance(value, int):
                errors.append(f"{col_name}: expected int, got {type(value).__name__}")
            elif col_type == "float" and not isinstance(value, (int, float)):
                errors.append(f"{col_name}: expected float, got {type(value).__name__}")
            elif col_type == "bool" and not isinstance(value, bool):
                errors.append(f"{col_name}: expected bool, got {type(value).__name__}")
            elif col_type == "list" and not isinstance(value, list):
                errors.append(f"{col_name}: expected list, got {type(value).__name__}")
            elif col_type == "dict" and not isinstance(value, dict):
                errors.append(f"{col_name}: expected dict, got {type(value).__name__}")
            elif col_type == "enum":
                enum_values = col.get("enum_values", [])
                if value not in enum_values:
                    errors.append(
                        f"{col_name}: '{value}' not in allowed values {enum_values}"
                    )

        return errors

    def _create_synthetic_seed(self, sample_index: int) -> AssignedSeed:
        """Create a synthetic seed when no source files exist."""
        diversity_assignments = {}
        if self.state.diversity_spec:
            diversity_assignments = self._get_diversity_assignment_for_index(
                sample_index
            )

        return AssignedSeed(
            seed_id=f"synthetic-{sample_index}",
            seed_text="",
            diversity_assignments=diversity_assignments,
            score=1.0,
        )

    def _get_diversity_assignment_for_index(self, index: int) -> Dict[str, str]:
        """Get diversity assignment for an index (cycles through combinations)."""
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
