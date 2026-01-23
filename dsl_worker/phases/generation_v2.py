"""
Phase: Sample Generation (Updated for Research-based seeds)

Key changes from original:
1. Pulls seeds from ResearchPhase instead of SeedAssignmentPhase
2. Passes coverage gaps to generation agent
3. Generation agent decides which diversity slot to fill
4. Reports back slot filled, updates coverage

Seeds are minimal: {text, note, source_url}
No pre-assignment - generation has full autonomy.
"""

import asyncio
import json
import logging
import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple
from collections import defaultdict

from sqlalchemy import func as sql_func

from dsl_api.models import ProjectVersion
from dsl_worker.phases.base import Phase, PhaseResult
from dsl_worker.phases.research import ResearchPhase, Seed
from dsl_api.models.sample import Sample

logger = logging.getLogger(__name__)

GENERATION_TIMEOUT = 300.0
GENERATION_MODEL = os.getenv("GENERATION_MODEL", "gpt-4.1")


def sanitize_row(row: Dict[str, Any]) -> Dict[str, Any]:
    """Remove NULL bytes from row data before database insertion."""
    if row is None:
        return row
    json_str = json.dumps(row, ensure_ascii=False)
    clean_str = json_str.replace('\\u0000', '').replace('\x00', '')
    return json.loads(clean_str)


@dataclass
class GenerationContext:
    """Context for generating a single sample."""
    sample_index: int
    seed: Seed
    coverage_gaps: Dict[str, Dict[str, int]]  # {axis: {value: count_needed}}
    current_row: Dict[str, Any] = field(default_factory=dict)
    assigned_slot: Optional[Dict[str, str]] = None  # What slot generation filled
    row_submitted: bool = False
    generation_cost_usd: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0


class GenerationPhase(Phase):
    """
    Generate samples from seeds using agentic tool calling.
    
    Updated flow:
    1. Pull seed from research phase
    2. Show generation agent the coverage gaps
    3. Agent decides: can this seed fill a gap?
       - Yes: generate row, report slot filled
       - No: reject seed, pull next
    4. Update coverage tracking
    """
    
    def __init__(
        self,
        *args,
        research_phase: Optional[ResearchPhase] = None,
        parallel_samples: int = 30,
        stop_checker: Optional[Callable[[], bool]] = None,
        cost_tracker: Optional[Any] = None,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.research_phase = research_phase
        self.parallel_samples = parallel_samples
        self.stop_checker = stop_checker
        self.cost_tracker = cost_tracker
        
        # Coverage tracking
        self._coverage: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self._coverage_targets: Dict[str, Dict[str, int]] = {}
        self._coverage_lock = asyncio.Lock()
        
        # Seed management
        self._used_seed_ids: set = set()
        self._seed_lock = asyncio.Lock()
        
        # Counters
        self._success_count = 0
        self._fail_count = 0
        self._reject_count = 0  # Seeds rejected (can't fill gap)
        
        # Worker management
        self._db_lock = asyncio.Lock()
        self._stop_requested = False
        self._worker_tasks: List[asyncio.Task] = []
        
    def should_run(self) -> bool:
        """Run if research has seeds and we haven't generated all samples."""
        if not self.research_phase:
            return False
        if self.research_phase.get_seed_count() == 0:
            return False
        return self.state.samples_generated < self.state.num_samples
        
    async def execute_once(self) -> PhaseResult:
        """Generate samples using worker pool."""
        
        # Initialize coverage targets on first run
        if not self._coverage_targets:
            self._init_coverage_targets()
            
        samples_generated = self.state.samples_generated
        target = self.state.num_samples
        remaining = target - samples_generated
        
        if remaining <= 0:
            return PhaseResult.no_work()
            
        # Reset counters
        self._success_count = 0
        self._fail_count = 0
        self._reject_count = 0
        self._stop_requested = False
        
        logger.info(f"[{self.name}] Starting generation, {remaining} remaining")
        
        # Create worker tasks
        num_workers = min(self.parallel_samples, remaining)
        self._worker_tasks = [
            asyncio.create_task(self._worker(i))
            for i in range(num_workers)
        ]
        
        # Wait for all workers
        await asyncio.gather(*self._worker_tasks, return_exceptions=True)
        
        logger.info(
            f"[{self.name}] Done: {self._success_count} succeeded, "
            f"{self._fail_count} failed, {self._reject_count} rejected"
        )
        
        return PhaseResult.work_done() if self._success_count > 0 else PhaseResult.no_work()
        
    def _init_coverage_targets(self):
        """Initialize coverage targets from diversity spec."""
        if not self.state.diversity_spec:
            return
            
        target = self.state.num_samples
        
        for axis in self.state.diversity_spec:
            axis_name = axis.get("name")
            values = axis.get("values", [])
            
            total_weight = sum(v.get("weight", 1.0) for v in values)
            
            self._coverage_targets[axis_name] = {}
            for v in values:
                value_name = v.get("value")
                weight = v.get("weight", 1.0)
                # Calculate target count for this value
                count = int((weight / total_weight) * target)
                self._coverage_targets[axis_name][value_name] = max(1, count)
                
    def _get_coverage_gaps(self) -> Dict[str, Dict[str, int]]:
        """Get current coverage gaps (what we still need)."""
        gaps = {}
        
        for axis_name, targets in self._coverage_targets.items():
            gaps[axis_name] = {}
            for value_name, target_count in targets.items():
                current = self._coverage.get(axis_name, {}).get(value_name, 0)
                needed = max(0, target_count - current)
                if needed > 0:
                    gaps[axis_name][value_name] = needed
                    
        return gaps
        
    async def _worker(self, worker_id: int):
        """Worker: pull seed → generate → save → repeat."""
        
        while True:
            # Check stop
            if self._stop_requested:
                return
            if self.stop_checker and self.stop_checker():
                return
                
            # Check if we've hit target
            if self.state.samples_generated >= self.state.num_samples:
                return
                
            # Pull a seed
            seed = await self._pull_seed()
            if seed is None:
                logger.info(f"[Worker {worker_id}] No more seeds available")
                return
                
            # Get current gaps
            gaps = self._get_coverage_gaps()
            
            # Create context
            ctx = GenerationContext(
                sample_index=self.state.samples_generated,
                seed=seed,
                coverage_gaps=gaps,
            )
            
            try:
                # Generate
                result = await self._generate_sample(ctx)
                
                if result == "rejected":
                    # Seed couldn't fill any gap
                    self._reject_count += 1
                    logger.info(f"[Worker {worker_id}] Seed rejected (no gap fit)")
                    continue
                    
                if result == "failed":
                    self._fail_count += 1
                    logger.warning(f"[Worker {worker_id}] Generation failed")
                    continue
                    
                # Success - save and update coverage
                await self._save_sample(ctx)
                
                if ctx.assigned_slot:
                    await self._update_coverage(ctx.assigned_slot)
                    
                self._success_count += 1
                logger.info(f"[Worker {worker_id}] ✓ Sample generated")
                
            except asyncio.CancelledError:
                return
            except Exception as e:
                logger.error(f"[Worker {worker_id}] Error: {e}")
                self._fail_count += 1
                
    async def _pull_seed(self) -> Optional[Seed]:
        """Pull an unused seed from the pool."""
        async with self._seed_lock:
            for seed in self.research_phase.get_seeds():
                if seed.id not in self._used_seed_ids:
                    self._used_seed_ids.add(seed.id)
                    return seed
        return None
        
    async def _generate_sample(self, ctx: GenerationContext) -> str:
        """
        Generate a sample from a seed.
        
        Returns:
            "success" - row generated
            "rejected" - seed can't fill any gap
            "failed" - generation error
        """
        system_prompt = self._build_system_prompt(ctx)
        tools = self._build_tools()
        
        input_items = [{"role": "system", "content": system_prompt}]
        
        for iteration in range(20):  # Max iterations
            if self._stop_requested:
                return "failed"
                
            try:
                response, cost = await self.openai_client.responses_create(
                    model=GENERATION_MODEL,
                    input=input_items,
                    tools=tools,
                    max_output_tokens=100_000,
                )
                ctx.generation_cost_usd += cost.total_cost_usd
                ctx.input_tokens += cost.input_tokens
                ctx.output_tokens += cost.output_tokens
                
            except Exception as e:
                logger.error(f"API error: {e}")
                return "failed"
                
            # Process response
            for item in response.output:
                if item.type == "function_call":
                    name = item.name
                    args = json.loads(item.arguments)
                    call_id = item.call_id
                    
                    result = await self._handle_tool_call(ctx, name, args)
                    
                    input_items.append({
                        "type": "function_call",
                        "call_id": call_id,
                        "name": name,
                        "arguments": item.arguments,
                    })
                    input_items.append({
                        "type": "function_call_output",
                        "call_id": call_id,
                        "output": result,
                    })
                    
                    if name == "reject_seed":
                        return "rejected"
                        
                    if ctx.row_submitted:
                        return "success"
                        
        return "failed"  # Max iterations exceeded
        
    def _build_system_prompt(self, ctx: GenerationContext) -> str:
        """Build system prompt for generation."""
        
        # Format schema
        schema_lines = []
        for col in self.state.columns or []:
            line = f"- {col.get('name')} ({col.get('type')})"
            if col.get('description'):
                line += f": {col['description']}"
            if col.get('type') == 'enum' and col.get('enum_values'):
                line += f" [values: {', '.join(col['enum_values'])}]"
            schema_lines.append(line)
        schema = "\n".join(schema_lines)
        
        # Format coverage gaps
        gaps_lines = []
        for axis, values in ctx.coverage_gaps.items():
            if values:
                vals = ", ".join([f"{v}: need {n}" for v, n in values.items()])
                gaps_lines.append(f"- {axis}: {vals}")
        gaps = "\n".join(gaps_lines) if gaps_lines else "All quotas filled"
        
        # Format seed
        if ctx.seed.text:
            seed_content = f"Content:\n{ctx.seed.text}"
        else:
            seed_content = "(No content - this is a synthetic/assignment seed)"
            
        return f"""You are generating a dataset row from a seed.

## Dataset Instructions
{self.state.generation_prompt}

## Column Schema
{schema}

## Coverage Gaps (what we still need)
{gaps}

## Seed
{seed_content}

Note from research: {ctx.seed.note}
Source: {ctx.seed.source_url or "N/A"}

## Your Task

1. First, decide: Can this seed fill ANY of the coverage gaps?
   - Look at the seed content and the gaps we need to fill
   - If no reasonable fit exists, call reject_seed()

2. If the seed can work:
   - Decide which gap it fits best
   - Use append() to build each column
   - Call submit_row() with the slot you're filling

## Tools
- append(column, content): Add content to a column
- reject_seed(reason): This seed can't fill any gap
- submit_row(slot): Submit completed row with diversity slot filled

Be creative but grounded. Use the seed as your anchor."""

    def _build_tools(self) -> List[Dict]:
        """Build tool definitions."""
        
        # Build slot schema from diversity spec
        slot_properties = {}
        for axis in self.state.diversity_spec or []:
            axis_name = axis.get("name")
            values = [v.get("value") for v in axis.get("values", [])]
            slot_properties[axis_name] = {
                "type": "string",
                "enum": values,
                "description": f"The {axis_name} this row fills"
            }
            
        return [
            {
                "type": "function",
                "name": "append",
                "description": "Add content to a column.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "column": {"type": "string"},
                        "content": {}
                    },
                    "required": ["column", "content"]
                }
            },
            {
                "type": "function",
                "name": "reject_seed",
                "description": "Reject this seed - it cannot reasonably fill any coverage gap.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "reason": {"type": "string", "description": "Why this seed doesn't fit"}
                    },
                    "required": ["reason"]
                }
            },
            {
                "type": "function",
                "name": "submit_row",
                "description": "Submit the completed row with diversity slot assignment.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "slot": {
                            "type": "object",
                            "properties": slot_properties,
                            "description": "The diversity slot this row fills"
                        }
                    },
                    "required": ["slot"]
                }
            },
            # Include web tools for generation-time research
            {
                "type": "function",
                "name": "web_search",
                "description": "Search the web for additional information.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"}
                    },
                    "required": ["query"]
                }
            },
        ]
        
    async def _handle_tool_call(self, ctx: GenerationContext, name: str, args: Dict) -> str:
        """Handle tool calls during generation."""
        
        if name == "append":
            return self._handle_append(ctx, args)
        elif name == "reject_seed":
            reason = args.get("reason", "No reason given")
            logger.info(f"Seed rejected: {reason}")
            return f"OK: seed rejected - {reason}"
        elif name == "submit_row":
            return self._handle_submit(ctx, args)
        elif name == "web_search":
            return await self._handle_web_search(args)
        else:
            return f"Unknown tool: {name}"
            
    def _handle_append(self, ctx: GenerationContext, args: Dict) -> str:
        """Handle append tool."""
        column = args.get("column", "")
        content = args.get("content", "")
        
        if not column:
            return "Error: column name required"
            
        # Find column type
        col_type = "string"
        for col in self.state.columns or []:
            if col.get("name") == column:
                col_type = col.get("type", "string")
                break
                
        # Handle by type
        if col_type == "list":
            if column not in ctx.current_row:
                ctx.current_row[column] = []
            ctx.current_row[column].append(content)
        elif col_type == "string":
            if column not in ctx.current_row:
                ctx.current_row[column] = ""
            ctx.current_row[column] += str(content)
        else:
            ctx.current_row[column] = content
            
        return f"OK: {column} updated"
        
    def _handle_submit(self, ctx: GenerationContext, args: Dict) -> str:
        """Handle submit_row tool."""
        slot = args.get("slot", {})
        
        # Validate all columns present
        missing = []
        for col in self.state.columns or []:
            col_name = col.get("name")
            if col_name and col_name not in ctx.current_row:
                missing.append(col_name)
                
        if missing:
            return f"Error: missing columns: {', '.join(missing)}"
            
        # Validate slot
        for axis_name in self._coverage_targets.keys():
            if axis_name not in slot:
                return f"Error: slot must include {axis_name}"
            value = slot[axis_name]
            if value not in self._coverage_targets[axis_name]:
                return f"Error: invalid value '{value}' for {axis_name}"
                
        ctx.assigned_slot = slot
        ctx.row_submitted = True
        return "OK: row submitted"
        
    async def _handle_web_search(self, args: Dict) -> str:
        """Handle web search during generation."""
        # Delegate to research phase's brave search
        query = args.get("query", "")
        if not query:
            return "Error: query required"
            
        # Simplified search
        try:
            import httpx
            brave_key = os.getenv("BRAVE_API_KEY")
            if not brave_key:
                return "Error: search not configured"
                
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    "https://api.search.brave.com/res/v1/web/search",
                    params={"q": query, "count": 3},
                    headers={"X-Subscription-Token": brave_key},
                    timeout=30.0,
                )
                response.raise_for_status()
                data = response.json()
                
            results = []
            for r in data.get("web", {}).get("results", [])[:3]:
                results.append(f"- {r.get('title')}: {r.get('description')}")
                
            return "\n".join(results) if results else "No results found"
            
        except Exception as e:
            return f"Search error: {e}"
            
    async def _update_coverage(self, slot: Dict[str, str]):
        """Update coverage tracking."""
        async with self._coverage_lock:
            for axis_name, value in slot.items():
                self._coverage[axis_name][value] += 1
                
    async def _save_sample(self, ctx: GenerationContext):
        """Save generated sample to database."""
        row = sanitize_row(ctx.current_row)
        tags = ctx.assigned_slot or {}
        
        async with self._db_lock:
            try:
                self.db.query(ProjectVersion).filter(
                    ProjectVersion.id == self.state.version_id
                ).with_for_update().first()
                
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
                    row=row,
                    tags=tags,
                )
                self.db.add(sample)
                
                self.db.query(ProjectVersion).filter(
                    ProjectVersion.id == self.state.version_id
                ).update(
                    {ProjectVersion.generated_count: ProjectVersion.generated_count + 1},
                    synchronize_session=False
                )
                
                self.db.commit()
                
            except Exception as e:
                logger.error(f"Save failed: {e}")
                self.db.rollback()
                raise
                
    def is_complete(self) -> bool:
        """Complete when target reached."""
        return self.state.samples_generated >= self.state.num_samples
        
    def get_status(self) -> "PhaseStatus":
        """Get current progress."""
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