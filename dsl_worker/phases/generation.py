"""
Phase: Sample Generation (Worker Pool)

Generates samples using an agentic tool-calling approach.
Each sample is committed to the database IMMEDIATELY when it completes,
providing real-time visibility to users.

Uses a worker pool pattern where:
- N workers run continuously
- Each worker grabs the next sample index atomically
- Samples are saved and committed one-by-one as they complete
- No batch concept - workers keep going until target is reached
- PAUSE/STOP immediately cancels all in-flight workers
"""

import asyncio
import json
import logging
import os
import time
import uuid
import re
import httpx
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple
from sqlalchemy import func as sql_func

from dsl_worker.phases.base import Phase, PhaseResult
from dsl_worker.phases.seed_assignment import SeedAssignmentPhase, AssignedSeed
from dsl_api.models.sample import Sample
from dsl_api.models.project_rag_chunk import ProjectRagChunk

import numpy as np

logger = logging.getLogger(__name__)

# Timeout for a single sample generation
GENERATION_TIMEOUT = 300.0

# How often to check for pause/stop (seconds) - keep this LOW for responsiveness
PAUSE_CHECK_INTERVAL = 0.5

GENERATION_MODEL = os.getenv("GENERATION_MODEL", "gpt-5-nano")


@dataclass
class SampleGenerationContext:
    """
    Isolated context for generating a single sample.

    Allows multiple samples to be generated concurrently without
    shared state conflicts.
    """
    sample_index: int
    assigned_seed: AssignedSeed
    current_row: Dict[str, Any] = field(default_factory=dict)
    row_submitted: bool = False
    generation_cost_usd: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    input_items: List[Dict[str, Any]] = field(default_factory=list)


class GenerationPhase(Phase):
    """
    Generate samples from assigned seeds using agentic tool calling.

    Uses a TRUE worker pool pattern:
    - Workers run continuously and pick up work as they finish
    - Each completed sample is committed to DB immediately
    - Costs are tracked and charged incrementally during generation
    - PAUSE/STOP immediately cancels all in-flight API calls

    Tools:
    - append: Build row incrementally (string concat, list append, or set value)
    - submit_row: Finalize and validate the row
    - rag_search: Search user's uploaded documents
    - web_search: Search web via Brave Search API
    - crawl: Fetch full page content via ScrapingBee
    """

    def __init__(
        self,
        *args,
        assignment_phase: Optional[SeedAssignmentPhase] = None,
        parallel_samples: int = 30,  # Number of concurrent workers
        stop_checker: Optional[Callable[[], bool]] = None,  # Callback to check if we should stop
        cost_tracker: Optional[Any] = None,  # CostTracker for incremental charging
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.assignment_phase = assignment_phase
        self.parallel_samples = parallel_samples
        self.stop_checker = stop_checker
        self.cost_tracker = cost_tracker

        # Config
        self.max_iterations = 20
        self.brave_api_key = os.getenv("BRAVE_API_KEY")
        self.scrapingbee_api_key = os.getenv("SCRAPINGBEE_API_KEY")

        # Lock for database operations (sequence number allocation + commit)
        self._db_lock = asyncio.Lock()

        # Atomic counter for sample index allocation
        self._next_sample_index = 0
        self._index_lock = asyncio.Lock()

        # Track accumulated cost
        self._total_cost = 0.0
        self._cost_lock = asyncio.Lock()

        # Track results
        self._success_count = 0
        self._fail_count = 0
        self._cancelled_count = 0
        self._results_lock = asyncio.Lock()

        # Stop event and worker tasks for IMMEDIATE cancellation
        self._stop_event = asyncio.Event()
        self._stop_requested = False
        self._worker_tasks: List[asyncio.Task] = []
        self._cancel_lock = asyncio.Lock()

    # =========================================================================
    # Phase interface
    # =========================================================================

    def should_run(self) -> bool:
        """Run if assignment is complete and we haven't generated all samples."""
        if not self.assignment_phase or not self.assignment_phase.is_complete():
            return False
        return self.state.samples_generated < self.state.num_samples

    async def execute_once(self) -> PhaseResult:
        """
        Generate samples using a continuous worker pool.

        Each worker grabs work, generates, and commits immediately.
        Costs are tracked and charged incrementally.
        PAUSE/STOP cancels all in-flight API calls IMMEDIATELY.
        """
        assigned_seeds = self.assignment_phase.get_assigned_seeds()
        samples_generated = self.state.samples_generated
        target = self.state.num_samples
        remaining = target - samples_generated

        if remaining <= 0:
            return PhaseResult.no_work()

        # Reset counters for this run
        self._next_sample_index = samples_generated
        self._total_cost = 0.0
        self._success_count = 0
        self._fail_count = 0
        self._cancelled_count = 0
        self._stop_requested = False
        self._stop_event.clear()
        self._worker_tasks = []

        logger.info(
            f"[{self.name}] Starting {self.parallel_samples} workers for {remaining} remaining samples"
        )

        # Track last pause check time
        last_pause_check = [time.time()]

        def should_stop() -> bool:
            """
            Check if we should stop (pause requested or external stop).
            """
            # Fast path: already stopped
            if self._stop_requested:
                return True

            # Check external stop callback (no DB hit)
            if self.stop_checker and self.stop_checker():
                self._stop_requested = True
                return True

            # Throttled DB check for pause
            now = time.time()
            if now - last_pause_check[0] >= PAUSE_CHECK_INTERVAL:
                last_pause_check[0] = now
                self.state.refresh()

                if self.state.paused:
                    self._trigger_immediate_stop("Pause detected from database")
                    return True

            return False

        def _trigger_immediate_stop(reason: str):
            """Trigger immediate stop and cancel ALL worker tasks."""
            if self._stop_requested:
                return  # Already stopping

            logger.info(f"[{self.name}] ⏸️  {reason} - cancelling all workers IMMEDIATELY")
            self._stop_requested = True
            self._stop_event.set()

            # Cancel ALL running worker tasks immediately
            # This will cause CancelledError to be raised at the next await point
            cancelled_count = 0
            for task in self._worker_tasks:
                if not task.done():
                    task.cancel()
                    cancelled_count += 1

            if cancelled_count > 0:
                logger.info(f"[{self.name}] Cancelled {cancelled_count} in-flight workers")

        # Make _trigger_immediate_stop available to should_stop closure
        self._trigger_immediate_stop = _trigger_immediate_stop

        async def add_cost_and_maybe_charge(
            cost_usd: float,
            input_tokens: int = 0,
            output_tokens: int = 0,
            model: str = "",
        ):
            """Add cost and charge if threshold reached."""
            if cost_usd <= 0:
                return

            async with self._cost_lock:
                self._total_cost += cost_usd

            if self.cost_tracker:
                self.cost_tracker.add_cost(
                    phase=self.name,
                    cost_usd=cost_usd,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    model=model,
                )
                # Check if we should charge
                charged = self.cost_tracker.charge_if_needed()
                if charged:
                    logger.info(f"[{self.name}] Charged {charged}¢ to balance")

                    # After charging, check if we've exceeded limits
                    if self.cost_tracker.would_exceed_spend_limit(0):
                        _trigger_immediate_stop("Spend limit exceeded after charge")
                    elif not self.cost_tracker.has_sufficient_balance():
                        _trigger_immediate_stop("Balance depleted after charge")

        async def worker(worker_id: int):
            """Worker: grab index -> generate -> save immediately -> repeat"""
            while True:
                # Check for stop BEFORE grabbing new work
                if should_stop():
                    return

                # Atomically get next sample index
                async with self._index_lock:
                    if self._next_sample_index >= target:
                        return
                    if self._stop_requested:  # Double-check under lock
                        return
                    sample_idx = self._next_sample_index
                    self._next_sample_index += 1

                # Get seed
                if not assigned_seeds:
                    seed = self._create_synthetic_seed(sample_idx)
                else:
                    seed = assigned_seeds[sample_idx % len(assigned_seeds)]

                ctx = SampleGenerationContext(
                    sample_index=sample_idx,
                    assigned_seed=seed,
                )

                try:
                    sample_data, cost, was_cancelled = await self._generate_sample_with_cancel_check(ctx)

                    # Track and charge cost IMMEDIATELY with token info
                    await add_cost_and_maybe_charge(
                        cost,
                        input_tokens=ctx.input_tokens,
                        output_tokens=ctx.output_tokens,
                        model=GENERATION_MODEL,
                    )

                    if was_cancelled:
                        # Don't count cancelled samples as failures - they'll be retried on resume
                        async with self._results_lock:
                            self._cancelled_count += 1
                        logger.info(f"[Worker {worker_id}] ⏸ Sample {sample_idx + 1} cancelled (will retry on resume)")
                    elif sample_data:
                        await self._save_sample(ctx, sample_data)
                        async with self._results_lock:
                            self._success_count += 1
                        logger.info(f"[Worker {worker_id}] ✓ Sample {sample_idx + 1}/{target} saved")
                    else:
                        async with self._results_lock:
                            self._fail_count += 1
                        logger.warning(f"[Worker {worker_id}] ✗ Sample {sample_idx + 1} failed")

                except asyncio.CancelledError:
                    # Task was cancelled (pause/stop) - track any accumulated cost
                    logger.info(f"[Worker {worker_id}] ⏸ Cancelled at sample {sample_idx + 1}")
                    if ctx.generation_cost_usd > 0:
                        await add_cost_and_maybe_charge(
                            ctx.generation_cost_usd,
                            input_tokens=ctx.input_tokens,
                            output_tokens=ctx.output_tokens,
                            model=GENERATION_MODEL,
                        )
                    async with self._results_lock:
                        self._cancelled_count += 1
                    # Don't re-raise - just exit the worker gracefully
                    return

                except Exception as e:
                    logger.error(f"[Worker {worker_id}] Sample {sample_idx + 1} exception: {e}")
                    async with self._results_lock:
                        self._fail_count += 1
                    # Still track any cost incurred before the error
                    if ctx.generation_cost_usd > 0:
                        await add_cost_and_maybe_charge(
                            ctx.generation_cost_usd,
                            input_tokens=ctx.input_tokens,
                            output_tokens=ctx.output_tokens,
                            model=GENERATION_MODEL,
                        )

        # Start workers
        self._worker_tasks = [
            asyncio.create_task(worker(i))
            for i in range(self.parallel_samples)
        ]

        # Wait for all workers, handling cancellation gracefully
        try:
            results = await asyncio.gather(*self._worker_tasks, return_exceptions=True)

            # Log any unexpected exceptions (but not CancelledError - that's expected on pause)
            for i, result in enumerate(results):
                if isinstance(result, Exception) and not isinstance(result, asyncio.CancelledError):
                    logger.error(f"Worker {i} failed with: {result}")

        except asyncio.CancelledError:
            # The entire execute_once was cancelled - cancel all workers
            logger.info(f"[{self.name}] execute_once cancelled, stopping all workers")
            for task in self._worker_tasks:
                if not task.done():
                    task.cancel()
            # Wait for workers to finish cancelling (with short timeout)
            try:
                await asyncio.wait_for(
                    asyncio.gather(*self._worker_tasks, return_exceptions=True),
                    timeout=2.0
                )
            except asyncio.TimeoutError:
                logger.warning(f"[{self.name}] Timed out waiting for workers to cancel")
            raise

        stopped_early = self._stop_requested
        logger.info(
            f"[{self.name}] {'Stopped' if stopped_early else 'Done'}: "
            f"{self._success_count} succeeded, {self._fail_count} failed, "
            f"{self._cancelled_count} cancelled, cost ${self._total_cost:.4f}"
        )

        # Return 0 cost since we already tracked it incrementally
        return PhaseResult.work_done(cost_usd=0.0) if self._success_count > 0 else PhaseResult.no_work()

    def request_stop(self):
        """
        Request workers to stop and cancel in-flight tasks IMMEDIATELY.

        This is called by the orchestrator when pause/stop is detected.
        It's safe to call multiple times.
        """
        if self._stop_requested:
            return  # Already stopping

        logger.info(f"[{self.name}] Stop requested externally, cancelling {len(self._worker_tasks)} workers")
        self._stop_requested = True
        self._stop_event.set()

        # Cancel all running worker tasks IMMEDIATELY
        cancelled_count = 0
        for task in self._worker_tasks:
            if not task.done():
                task.cancel()
                cancelled_count += 1

        if cancelled_count > 0:
            logger.info(f"[{self.name}] Cancelled {cancelled_count} in-flight workers")

    async def _generate_sample_with_cancel_check(
        self,
        ctx: SampleGenerationContext,
    ) -> Tuple[Optional[Dict], float, bool]:
        """
        Generate sample with timeout protection and proper cancellation handling.

        Returns:
            Tuple of (sample_data, cost_usd, was_cancelled)
            - sample_data: The generated row, or None if failed/cancelled
            - cost_usd: Cost incurred so far
            - was_cancelled: True if the generation was cancelled (not a failure)
        """
        try:
            # Simple timeout - no shield, no polling
            # When task.cancel() is called, CancelledError will propagate through
            sample_data, cost = await asyncio.wait_for(
                self._generate_sample(ctx, lambda: self._stop_requested),
                timeout=GENERATION_TIMEOUT
            )
            return sample_data, cost, False

        except asyncio.CancelledError:
            # Generation was cancelled (pause/stop requested)
            logger.debug(f"Sample {ctx.sample_index} generation cancelled")
            return None, ctx.generation_cost_usd, True

        except asyncio.TimeoutError:
            logger.error(f"Sample {ctx.sample_index} timed out after {GENERATION_TIMEOUT}s")
            return None, ctx.generation_cost_usd, False

    async def _save_sample(self, ctx: SampleGenerationContext, sample_data: Dict) -> None:
        """
        Save a single sample to database immediately with commit.

        Uses a lock to ensure sequence numbers are allocated atomically.
        The commit happens immediately so users see the sample right away.
        """
        async with self._db_lock:
            # Get next sequence number for this version
            max_seq = (
                    self.db.query(sql_func.max(Sample.seq))
                    .filter(Sample.version_id == self.state.version_id)
                    .scalar() or 0
            )

            sample = Sample(
                id=uuid.uuid4(),
                project_id=self.state.project_id,
                version_id=self.state.version_id,
                seq=max_seq + 1,
                row=sample_data,
                tags=ctx.assigned_seed.diversity_assignments,
            )
            self.db.add(sample)

            # Commit immediately - this makes the sample visible to users
            self.db.commit()

    def is_complete(self) -> bool:
        """Complete when we've generated the target number of samples."""
        return self.state.samples_generated >= self.state.num_samples

    def get_status(self) -> "PhaseStatus":
        """Get current progress of sample generation."""
        from dsl_worker.phases.base import PhaseStatus

        if self.is_complete():
            status = "complete"
        elif self.should_run():
            status = "active"
        else:
            status = "pending"

        return PhaseStatus(
            phase_name=self.name,
            status=status,
            progress=f"{self.state.samples_generated}/{self.state.num_samples} samples"
        )

    # =========================================================================
    # Sample generation (per-context)
    # =========================================================================

    async def _generate_sample(
            self,
            ctx: SampleGenerationContext,
            should_stop_fn: Callable[[], bool] = None,
    ) -> Tuple[Optional[Dict], float]:
        """
        Generate a single sample using the agentic approach.

        Returns:
            Tuple of (sample_data, total_cost_usd)
            sample_data is None if generation failed
        """
        system_prompt = self._build_system_prompt(ctx.assigned_seed)
        ctx.input_items = [{"role": "system", "content": system_prompt}]
        tools = self._build_tools()
        logger.debug(f"Sample {ctx.sample_index}: columns={[c.get('name') for c in self.state.columns or []]}")

        for iteration in range(self.max_iterations):
            # Check for stop at the start of each iteration
            if should_stop_fn and should_stop_fn():
                logger.debug(f"Sample {ctx.sample_index}: stopping at iteration {iteration + 1}")
                return None, ctx.generation_cost_usd

            logger.debug(f"Sample {ctx.sample_index}: iteration {iteration + 1}")

            try:
                response, cost = await self.openai_client.responses_create(
                    model=GENERATION_MODEL,
                    input=ctx.input_items,
                    tools=tools,
                    max_output_tokens=100_000,
                )
                ctx.generation_cost_usd += cost.total_cost_usd
                ctx.input_tokens += cost.input_tokens
                ctx.output_tokens += cost.output_tokens
            except asyncio.CancelledError:
                # Propagate cancellation
                raise
            except Exception as e:
                logger.error(f"OpenAI API error for sample {ctx.sample_index}: {e}")
                return None, ctx.generation_cost_usd

            has_tool_calls = False
            tool_names_this_iteration = []

            for output_item in response.output:
                # Check for stop between processing output items
                if should_stop_fn and should_stop_fn():
                    return None, ctx.generation_cost_usd

                if output_item.type == "message":
                    msg_content = output_item.content[0].text if output_item.content else ""
                    logger.debug(f"Sample {ctx.sample_index}: message: {msg_content[:200]}...")
                    ctx.input_items.append({
                        "role": "assistant",
                        "content": msg_content,
                    })

                elif output_item.type == "function_call":
                    has_tool_calls = True

                    name = output_item.name
                    args = json.loads(output_item.arguments)
                    call_id = output_item.call_id
                    tool_names_this_iteration.append(name)

                    logger.debug(f"Sample {ctx.sample_index}: tool {name}({list(args.keys())})")

                    result = await self._handle_tool_call(ctx, name, args)

                    # Log the result for debugging
                    result_preview = result[:200] if len(result) > 200 else result
                    logger.debug(f"Sample {ctx.sample_index}: tool result: {result_preview}")

                    ctx.input_items.append({
                        "type": "function_call",
                        "call_id": call_id,
                        "name": name,
                        "arguments": output_item.arguments,
                    })
                    ctx.input_items.append({
                        "type": "function_call_output",
                        "call_id": call_id,
                        "output": result,
                    })

                    if ctx.row_submitted:
                        logger.info(f"Sample {ctx.sample_index}: submitted after {iteration + 1} iterations")
                        return ctx.current_row, ctx.generation_cost_usd

            if not has_tool_calls:
                # Model stopped making tool calls - try to auto-submit
                missing = self._get_missing_columns(ctx)
                if missing:
                    logger.warning(
                        f"Sample {ctx.sample_index}: agent stopped with missing columns: {missing}"
                    )
                    return None, ctx.generation_cost_usd

                # All columns present - validate and submit
                errors = self._validate_column_types(ctx)
                if errors:
                    logger.warning(
                        f"Sample {ctx.sample_index}: validation errors on auto-submit: {errors}"
                    )
                    return None, ctx.generation_cost_usd

                # Success - auto-submit
                ctx.row_submitted = True
                logger.info(
                    f"Sample {ctx.sample_index}: auto-submitted after {iteration + 1} iterations"
                )
                return ctx.current_row, ctx.generation_cost_usd
            else:
                logger.debug(f"Sample {ctx.sample_index}: iteration {iteration + 1} tools: {tool_names_this_iteration}")

        logger.error(f"Sample {ctx.sample_index}: exceeded {self.max_iterations} iterations")
        return None, ctx.generation_cost_usd

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

        prompt = f"""You are a dataset row generator. Your job is to generate a single high-quality row for a dataset.

## Tools Available
- append(column, content): Add content to a column
  - string columns: concatenates text (call once with full content, or multiple times to build incrementally)
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

        if assigned_seed.seed_text:
            prompt += f"""
## Seed
This is your starting point - source material to build from:
<seed>
{assigned_seed.seed_text}
</seed>
"""

        prompt += """
## Instructions
1. Use rag_search to find relevant information from uploaded documents if needed
2. Use web_search and crawl for web research if needed
3. Use append() to build each column's content
4. Call submit_row() when the row is complete
5. Try to meet the diversity targets if specified

Generate a high-quality, accurate row now.
"""

        return prompt

    def _format_column_schema(self) -> str:
        """Format column schema for the prompt."""
        if not self.state.columns:
            return "No specific schema defined"

        lines = []
        for col in self.state.columns:
            col_name = col.get("name", "unknown")
            col_type = col.get("type", "string")
            col_desc = col.get("description", "")

            line = f"- {col_name} ({col_type})"
            if col_desc:
                line += f": {col_desc}"

            # Add enum values if present
            if col_type == "enum" and "enum_values" in col:
                line += f" [values: {', '.join(col['enum_values'])}]"

            lines.append(line)

        return "\n".join(lines)

    # =========================================================================
    # Tools
    # =========================================================================

    def _build_tools(self) -> List[Dict]:
        """Build the tool definitions for the agent."""
        tools = [
            {
                "type": "function",
                "name": "append",
                "description": "Add content to a column. For strings, concatenates. For lists, appends item. For other types, sets value.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "column": {
                            "type": "string",
                            "description": "The column name to append to"
                        },
                        "content": {
                            "description": "The content to append/set. Can be string, number, boolean, object, or array.",
                            "anyOf": [
                                {"type": "string"},
                                {"type": "number"},
                                {"type": "boolean"},
                                {"type": "object"},
                                {"type": "array", "items": {}}
                            ]
                        }
                    },
                    "required": ["column", "content"]
                }
            },
            {
                "type": "function",
                "name": "submit_row",
                "description": "Finalize and submit the completed row. Call this when all columns have been filled.",
                "parameters": {
                    "type": "object",
                    "properties": {}
                }
            },
            {
                "type": "function",
                "name": "rag_search",
                "description": "Search the uploaded source documents for relevant information.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The search query"
                        }
                    },
                    "required": ["query"]
                }
            },
        ]

        # Add web tools if API keys are available
        if self.brave_api_key:
            tools.append({
                "type": "function",
                "name": "web_search",
                "description": "Search the web using Brave Search. Returns titles, snippets, and URLs.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The search query"
                        }
                    },
                    "required": ["query"]
                }
            })

        if self.scrapingbee_api_key:
            tools.append({
                "type": "function",
                "name": "crawl",
                "description": "Fetch the full content of a web page.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "url": {
                            "type": "string",
                            "description": "The URL to fetch"
                        }
                    },
                    "required": ["url"]
                }
            })

        return tools

    async def _handle_tool_call(
        self, ctx: SampleGenerationContext, name: str, args: Dict
    ) -> str:
        """Handle a tool call and return the result."""
        if name == "append":
            return self._handle_append(ctx, args)
        elif name == "submit_row":
            return self._handle_submit_row(ctx)
        elif name == "rag_search":
            return await self._handle_rag_search(args)
        elif name == "web_search":
            return await self._handle_web_search(args)
        elif name == "crawl":
            return await self._handle_crawl(args)
        else:
            return f"Unknown tool: {name}"

    def _handle_append(self, ctx: SampleGenerationContext, args: Dict) -> str:
        """Handle append tool call."""
        column = args.get("column", "")
        content = args.get("content", "")

        if not column:
            return "Error: column name is required"

        # Find column type
        col_type = "string"
        for col in self.state.columns or []:
            if col.get("name") == column:
                col_type = col.get("type", "string")
                break

        # Handle based on type
        if col_type == "list":
            if column not in ctx.current_row:
                ctx.current_row[column] = []
            ctx.current_row[column].append(content)
        elif col_type == "string":
            if column not in ctx.current_row:
                ctx.current_row[column] = ""
            ctx.current_row[column] += str(content)
        else:
            # int, float, bool, enum, dict - just set the value
            ctx.current_row[column] = content

        return f"OK: {column} updated"

    def _handle_submit_row(self, ctx: SampleGenerationContext) -> str:
        """Handle submit_row tool call."""
        missing = self._get_missing_columns(ctx)
        if missing:
            return f"Error: missing required columns: {', '.join(missing)}"

        errors = self._validate_column_types(ctx)
        if errors:
            return f"Error: validation failed: {'; '.join(errors)}"

        ctx.row_submitted = True
        return "OK: row submitted successfully"

    def _get_missing_columns(self, ctx: SampleGenerationContext) -> List[str]:
        """Get list of missing required columns."""
        missing = []
        for col in self.state.columns or []:
            col_name = col.get("name")
            if col_name and col_name not in ctx.current_row:
                missing.append(col_name)
        return missing

    def _validate_column_types(self, ctx: SampleGenerationContext) -> List[str]:
        """Validate column types and return list of errors."""
        errors = []

        for col in self.state.columns or []:
            col_name = col.get("name")
            col_type = col.get("type", "string")

            if col_name not in ctx.current_row:
                continue

            value = ctx.current_row[col_name]

            if col_type == "int":
                if not isinstance(value, int):
                    try:
                        ctx.current_row[col_name] = int(value)
                    except (ValueError, TypeError):
                        errors.append(f"{col_name}: expected int, got {type(value).__name__}")
            elif col_type == "float":
                if not isinstance(value, (int, float)):
                    try:
                        ctx.current_row[col_name] = float(value)
                    except (ValueError, TypeError):
                        errors.append(f"{col_name}: expected float, got {type(value).__name__}")
            elif col_type == "bool":
                if not isinstance(value, bool):
                    if isinstance(value, str):
                        ctx.current_row[col_name] = value.lower() in ("true", "yes", "1")
                    else:
                        errors.append(f"{col_name}: expected bool, got {type(value).__name__}")
            elif col_type == "enum":
                enum_values = col.get("enum_values", [])
                # Only validate if enum_values is actually defined
                if enum_values and value not in enum_values:
                    errors.append(f"{col_name}: value '{value}' not in enum {enum_values}")
            elif col_type == "list":
                if not isinstance(value, list):
                    errors.append(f"{col_name}: expected list, got {type(value).__name__}")

        return errors

    # =========================================================================
    # RAG Search
    # =========================================================================

    async def _handle_rag_search(self, args: Dict) -> str:
        """Search uploaded documents using embeddings."""
        query = args.get("query", "")
        if not query:
            return "Error: query is required"

        try:
            # Get query embedding
            result = await self.openai_client.create_embeddings(
                model="text-embedding-3-small",
                input=[query],
            )
            query_embedding = np.array(result.response.data[0].embedding, dtype=np.float32)

            # Search chunks
            chunks = (
                self.db.query(ProjectRagChunk)
                .filter(ProjectRagChunk.project_id == self.state.project_id)
                .all()
            )

            if not chunks:
                return "No documents uploaded to search."

            # Calculate similarities
            scored = []
            for chunk in chunks:
                if chunk.embedding is not None:
                    chunk_emb = np.array(chunk.embedding, dtype=np.float32)
                    similarity = np.dot(query_embedding, chunk_emb) / (
                        np.linalg.norm(query_embedding) * np.linalg.norm(chunk_emb) + 1e-8
                    )
                    scored.append((similarity, chunk))

            # Get top results
            scored.sort(key=lambda x: x[0], reverse=True)
            top_results = scored[:5]

            if not top_results:
                return "No relevant results found."

            # Format results
            results_text = []
            for i, (score, chunk) in enumerate(top_results, 1):
                text_preview = chunk.text[:500] + "..." if len(chunk.text) > 500 else chunk.text
                results_text.append(f"[{i}] (score: {score:.3f})\n{text_preview}")

            return "\n\n".join(results_text)

        except Exception as e:
            logger.error(f"RAG search error: {e}")
            return f"Error searching documents: {e}"

    # =========================================================================
    # Web Search
    # =========================================================================

    async def _handle_web_search(self, args: Dict) -> str:
        """Search the web using Brave Search API."""
        query = args.get("query", "")
        if not query:
            return "Error: query is required"

        if not self.brave_api_key:
            return "Error: web search not configured"

        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    "https://api.search.brave.com/res/v1/web/search",
                    params={"q": query, "count": 5},
                    headers={"X-Subscription-Token": self.brave_api_key},
                    timeout=30.0,
                )
                response.raise_for_status()
                data = response.json()

            results = data.get("web", {}).get("results", [])
            if not results:
                return "No results found."

            results_text = []
            for i, result in enumerate(results[:5], 1):
                title = result.get("title", "")
                description = result.get("description", "")
                url = result.get("url", "")
                results_text.append(f"[{i}] {title}\n{description}\nURL: {url}")

            return "\n\n".join(results_text)

        except Exception as e:
            logger.error(f"Web search error: {e}")
            return f"Error searching web: {e}"

    # =========================================================================
    # Crawl
    # =========================================================================

    async def _handle_crawl(self, args: Dict) -> str:
        """Fetch full page content using ScrapingBee."""
        url = args.get("url", "")
        if not url:
            return "Error: url is required"

        if not self.scrapingbee_api_key:
            return "Error: crawling not configured"

        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    "https://app.scrapingbee.com/api/v1/",
                    params={
                        "api_key": self.scrapingbee_api_key,
                        "url": url,
                        "render_js": "false",
                        "extract_rules": json.dumps({"text": "body"}),
                    },
                    timeout=60.0,
                )
                response.raise_for_status()
                data = response.json()

            text = data.get("text", "")
            if not text:
                return "No content extracted from page."

            # Clean and truncate
            text = re.sub(r'\s+', ' ', text).strip()
            if len(text) > 10000:
                text = text[:10000] + "... [truncated]"

            return text

        except Exception as e:
            logger.error(f"Crawl error: {e}")
            return f"Error fetching page: {e}"

    # =========================================================================
    # Helpers
    # =========================================================================

    def _create_synthetic_seed(self, sample_idx: int) -> AssignedSeed:
        """Create a synthetic seed when no seeds are available."""
        return AssignedSeed(
            seed_text=f"Sample #{sample_idx + 1}",
            seed_id=None,
            diversity_assignments={},
            score=1.0,
        )