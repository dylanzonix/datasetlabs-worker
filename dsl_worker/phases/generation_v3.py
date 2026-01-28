"""
Phase: Generation (v3)

Generates rows from assigned seeds.

Seeds are pointers + notes. Generation:
1. Resolves the pointer to get actual content
2. Uses content + note to generate a row
3. Tracks diversity slot fulfillment

Starts early (1% threshold) and takes top quality seeds first.
"""

import asyncio
import json
import logging
import os
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from sqlalchemy import func as sql_func

from dsl_worker.phases.base import Phase, PhaseResult
from dsl_worker.phases.seed_assignment import SeedAssignmentPhase, AssignedSeed
from dsl_api.models.project_version import ProjectVersion
from dsl_api.models.sample import Sample

logger = logging.getLogger(__name__)

GENERATION_MODEL = os.getenv("GENERATION_MODEL", "gpt-5.2")


def sanitize_row(row: Dict[str, Any]) -> Dict[str, Any]:
    """Remove NULL bytes from row data."""
    if row is None:
        return row
    json_str = json.dumps(row, ensure_ascii=False)
    clean_str = json_str.replace('\\u0000', '').replace('\x00', '')
    return json.loads(clean_str)


@dataclass
class GenerationContext:
    """Context for generating a single row."""
    seed: AssignedSeed
    resolved_content: str  # Actual content from pointer
    current_row: Dict[str, Any] = field(default_factory=dict)
    row_submitted: bool = False
    generation_cost: float = 0.0


class GenerationPhaseV3(Phase):
    """
    Generate rows from assigned seeds.
    
    Start condition: 1% of target rows have assigned seeds
    Selection: Top quality seeds first
    """
    
    def __init__(
        self,
        *args,
        assignment_phase: Optional[SeedAssignmentPhase] = None,
        workspace_dir: Optional[str] = None,
        parallel_samples: int = 10,
        start_threshold: float = 0.01,
        quality_percentile: float = 0.1,
        stop_checker: Optional[Callable[[], bool]] = None,
        cost_tracker: Optional[Any] = None,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        
        self.assignment_phase = assignment_phase
        self.workspace_dir = Path(workspace_dir) if workspace_dir else None
        self.parallel_samples = parallel_samples
        self.start_threshold = start_threshold
        self.quality_percentile = quality_percentile
        self.stop_checker = stop_checker
        self.cost_tracker = cost_tracker
        
        # Stats
        self._success_count = 0
        self._fail_count = 0
        
        # Locks
        self._db_lock = asyncio.Lock()
        self._stop_requested = False
    
    def should_run(self) -> bool:
        """Run when we have enough seeds and haven't generated all samples."""
        if not self.assignment_phase:
            return False
        
        # Check if we've hit target
        if self.state.samples_generated >= self.state.num_samples:
            return False
        
        # Check threshold
        assigned = self.assignment_phase.get_assigned_seeds()
        threshold = max(1, int(self.state.num_samples * self.start_threshold))
        
        return len(assigned) >= threshold
    
    async def execute_once(self) -> PhaseResult:
        """Generate rows from seeds."""
        
        if not self.assignment_phase:
            return PhaseResult.no_work()
        
        # Get top quality seeds
        seeds = self.assignment_phase.get_top_seeds(fraction=self.quality_percentile)
        
        if not seeds:
            return PhaseResult.no_work()
        
        # Limit batch size
        remaining = self.state.num_samples - self.state.samples_generated
        batch_size = min(self.parallel_samples, len(seeds), remaining)
        batch = seeds[:batch_size]
        
        logger.info(f"[Generation] Generating {len(batch)} rows (remaining: {remaining})")
        
        # Generate in parallel
        tasks = [self._generate_row(seed) for seed in batch]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        total_cost = 0.0
        
        for seed, result in zip(batch, results):
            if isinstance(result, Exception):
                logger.error(f"[Generation] Failed: {result}")
                self._fail_count += 1
                continue
            
            success, cost = result
            total_cost += cost
            
            if success:
                self.assignment_phase.mark_seed_used(seed.seed_id)
                self._success_count += 1
            else:
                self._fail_count += 1
        
        if self.cost_tracker and total_cost > 0:
            self.cost_tracker.add_cost(
                phase=self.name,
                cost_usd=total_cost,
                model=GENERATION_MODEL,
            )
        
        logger.info(f"[Generation] Generated {self._success_count} rows total")
        
        return PhaseResult.work_done(cost_usd=total_cost)
    
    async def _generate_row(self, seed: AssignedSeed) -> Tuple[bool, float]:
        """Generate a single row from a seed."""
        
        if self._stop_requested or (self.stop_checker and self.stop_checker()):
            return False, 0.0
        
        # Resolve the pointer to get actual content
        resolved_content = self._resolve_seed_pointer(seed.source, seed.note)
        
        if not resolved_content:
            logger.warning(f"[Generation] Could not resolve seed: {seed.source}")
            return False, 0.0
        
        ctx = GenerationContext(
            seed=seed,
            resolved_content=resolved_content,
        )
        
        # Build prompt
        prompt = self._build_prompt(seed, resolved_content)
        tools = self._build_tools()
        
        messages = [{"role": "user", "content": prompt}]
        
        for iteration in range(20):
            if self._stop_requested:
                return False, ctx.generation_cost
            
            try:
                response, cost = await self.openai_client.responses_create(
                    model=GENERATION_MODEL,
                    input=messages,
                    tools=tools,
                    max_output_tokens=8000,
                )
                ctx.generation_cost += cost.total_cost_usd
                
            except Exception as e:
                logger.error(f"[Generation] API error: {e}")
                return False, ctx.generation_cost
            
            # Process response
            for item in response.output:
                if item.type == "function_call":
                    name = item.name
                    args = json.loads(item.arguments)
                    call_id = item.call_id
                    
                    result = self._handle_tool_call(ctx, name, args)
                    
                    messages.append({
                        "type": "function_call",
                        "call_id": call_id,
                        "name": name,
                        "arguments": item.arguments,
                    })
                    messages.append({
                        "type": "function_call_output",
                        "call_id": call_id,
                        "output": result,
                    })
                    
                    if ctx.row_submitted:
                        await self._save_row(ctx)
                        return True, ctx.generation_cost
            
            # If no tool calls, break
            if not any(item.type == "function_call" for item in response.output):
                break
        
        return False, ctx.generation_cost
    
    def _resolve_seed_pointer(self, source: str, note: str) -> Optional[str]:
        """
        Resolve a seed pointer to actual content.
        
        Source formats:
        - "path/to/file.md" - entire file
        - "path/to/file.md:lines 45-67" - specific lines
        - "path/to/file.md:Starting with 'The customer...'" - text search
        """
        if not self.workspace_dir:
            # Try to infer workspace from source path
            if "/" in source:
                pass  # Use source as-is
            else:
                return f"[Could not resolve: no workspace configured]\n\nNote: {note}"
        
        # Parse source
        file_path = source
        selector = None
        
        if ":" in source and not source.startswith("/"):
            # Could be "file:selector" or "file:lines X-Y"
            parts = source.split(":", 1)
            # Check if first part looks like a file
            if "/" in parts[0] or parts[0].endswith((".md", ".txt", ".json", ".csv")):
                file_path = parts[0]
                selector = parts[1].strip()
        
        # Resolve file path
        if not file_path.startswith("/"):
            if self.workspace_dir:
                full_path = self.workspace_dir / file_path
            else:
                full_path = Path(file_path)
        else:
            full_path = Path(file_path)
        
        if not full_path.exists():
            # Try with workspace subdirs
            if self.workspace_dir:
                for subdir in ["web", "uploads", "extracted"]:
                    alt_path = self.workspace_dir / subdir / file_path
                    if alt_path.exists():
                        full_path = alt_path
                        break
        
        if not full_path.exists():
            logger.warning(f"[Generation] File not found: {full_path}")
            return f"[File not found: {file_path}]\n\nNote: {note}"
        
        # Read content
        try:
            content = full_path.read_text(encoding='utf-8', errors='ignore')
        except Exception as e:
            return f"[Error reading file: {e}]\n\nNote: {note}"
        
        # Apply selector if present
        if selector:
            content = self._apply_selector(content, selector)
        
        # Truncate if too long (keep first 10k chars)
        if len(content) > 10000:
            content = content[:10000] + "\n\n[truncated...]"
        
        return content
    
    def _apply_selector(self, content: str, selector: str) -> str:
        """Apply a selector to content."""
        selector = selector.strip()
        
        # Lines selector: "lines 45-67" or "lines 45"
        lines_match = re.match(r"lines?\s+(\d+)(?:\s*-\s*(\d+))?", selector, re.IGNORECASE)
        if lines_match:
            start = int(lines_match.group(1))
            end = int(lines_match.group(2)) if lines_match.group(2) else start
            
            lines = content.split("\n")
            selected = lines[start-1:end]  # 1-indexed
            return "\n".join(selected)
        
        # Text search: "Starting with 'The customer...'"
        if selector.lower().startswith("starting with"):
            search_text = selector[13:].strip().strip("'\"")
            idx = content.find(search_text)
            if idx >= 0:
                # Return from that point, up to 2000 chars
                return content[idx:idx+2000]
        
        # Paragraph/item selector: "paragraph 3" or "item 5"
        item_match = re.match(r"(paragraph|item|section)\s+(\d+)", selector, re.IGNORECASE)
        if item_match:
            item_type = item_match.group(1).lower()
            item_num = int(item_match.group(2))
            
            # Split by double newlines for paragraphs
            parts = re.split(r"\n\n+", content)
            if 0 < item_num <= len(parts):
                return parts[item_num - 1]
        
        # Fallback: return full content
        return content
    
    def _build_prompt(self, seed: AssignedSeed, resolved_content: str) -> str:
        """Build generation prompt."""
        
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
        
        # Format diversity slot
        slot_str = ", ".join([f"{k}={v}" for k, v in seed.diversity_slot.items()])
        
        return f"""Generate a dataset row based on this seed content.

## Dataset Instructions
{self.state.generation_prompt}

## Column Schema
{schema}

## Diversity Slot
{slot_str}

## Seed Note
{seed.note}

## Seed Content
{resolved_content}

## Your Task
Generate a row for this dataset, grounded in the seed content.
- Use append(column, content) to build each column
- Call submit_row() when done
- Stay true to the seed content - don't hallucinate facts not in the content
- If info is missing, note that it's inferred or make reasonable assumptions"""

    def _build_tools(self) -> List[Dict]:
        """Build tool definitions."""
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
                "name": "submit_row",
                "description": "Submit the completed row.",
                "parameters": {
                    "type": "object",
                    "properties": {}
                }
            },
        ]
    
    def _handle_tool_call(self, ctx: GenerationContext, name: str, args: Dict) -> str:
        """Handle tool calls."""
        
        if name == "append":
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
        
        elif name == "submit_row":
            # Validate columns
            missing = []
            for col in self.state.columns or []:
                col_name = col.get("name")
                if col_name and col_name not in ctx.current_row:
                    missing.append(col_name)
            
            if missing:
                return f"Error: missing columns: {', '.join(missing)}"
            
            ctx.row_submitted = True
            return "OK: row submitted"
        
        return f"Unknown tool: {name}"
    
    async def _save_row(self, ctx: GenerationContext):
        """Save generated row to database."""
        
        row = sanitize_row(ctx.current_row)
        tags = ctx.seed.diversity_slot
        
        async with self._db_lock:
            try:
                # Lock version row
                self.db.query(ProjectVersion).filter(
                    ProjectVersion.id == self.state.version_id
                ).with_for_update().first()
                
                # Get next sequence number
                max_seq = (
                    self.db.query(sql_func.max(Sample.seq))
                    .filter(Sample.version_id == self.state.version_id)
                    .scalar() or 0
                )
                
                # Create sample
                sample = Sample(
                    id=uuid.uuid4(),
                    project_id=self.state.project_id,
                    version_id=self.state.version_id,
                    seq=max_seq + 1,
                    row=row,
                    tags=tags,
                )
                self.db.add(sample)
                
                # Update version count
                self.db.query(ProjectVersion).filter(
                    ProjectVersion.id == self.state.version_id
                ).update(
                    {ProjectVersion.generated_count: ProjectVersion.generated_count + 1},
                    synchronize_session=False
                )
                
                self.db.commit()
                
            except Exception as e:
                logger.error(f"[Generation] Save failed: {e}")
                self.db.rollback()
                raise
    
    def is_complete(self) -> bool:
        """Complete when target reached."""
        return self.state.samples_generated >= self.state.num_samples
    
    def get_status(self):
        """Get current status."""
        from dsl_worker.phases.base import PhaseStatus
        
        return PhaseStatus(
            phase_name=self.name,
            status="complete" if self.is_complete() else "active",
            progress=f"{self.state.samples_generated}/{self.state.num_samples} samples"
        )


# Type hint import
from typing import Tuple