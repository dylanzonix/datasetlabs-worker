"""
Phase: Seed Extraction (v2)

Processes files (from web, uploads, browser marks) into seeds.

Seeds are simple:
- source: pointer to the file/region
- note: natural language description of what's there

Multiple seeds per file is normal. The LLM identifies distinct items
that could each become a row.
"""

import asyncio
import json
import logging
import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import tiktoken

from dsl_worker.phases.base import Phase, PhaseResult
from dsl_api.models.project_seed import ProjectSeed

logger = logging.getLogger(__name__)

EXTRACTION_MODEL = os.getenv("EXTRACTION_MODEL", "gpt-5-nano")
CHUNK_THRESHOLD = 30000  # Chunk files above this token count


@dataclass
class ExtractedSeed:
    """A seed extracted from content."""
    source: str  # File path or file:lines pointer
    note: str    # Natural language description


@dataclass
class ExtractionResult:
    """Result from processing one file."""
    source_file: str
    seeds: List[ExtractedSeed]
    error: Optional[str] = None


class SeedExtractionPhase(Phase):
    """
    Extract seeds from queued files.
    
    Seeds are pointers + notes, not extracted content.
    Generation phase will use the pointer to get actual content.
    """
    
    def __init__(
        self,
        *args,
        extraction_queue: Optional[asyncio.Queue] = None,
        max_concurrent: int = 5,
        stop_checker: Optional[Callable[[], bool]] = None,
        cost_tracker: Optional[Any] = None,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        
        self.extraction_queue = extraction_queue or asyncio.Queue()
        self.max_concurrent = max_concurrent
        self.stop_checker = stop_checker
        self.cost_tracker = cost_tracker
        
        # Stats
        self._total_seeds = 0
        self._total_quality_sum = 0.0
        self._quality_counts = {"8-10": 0, "5-7": 0, "0-4": 0}
        self._processed_files: List[str] = []
        
        # Tokenizer
        try:
            self._encoding = tiktoken.get_encoding("cl100k_base")
        except:
            self._encoding = None
        
        # Semaphore
        self._semaphore = asyncio.Semaphore(max_concurrent)
    
    def _count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        if self._encoding:
            return len(self._encoding.encode(text))
        return len(text) // 4
    
    def should_run(self) -> bool:
        """Run if there are items in the queue."""
        return not self.extraction_queue.empty()
    
    async def execute_once(self) -> PhaseResult:
        """Process items from extraction queue."""
        
        if self.extraction_queue.empty():
            return PhaseResult.no_work()
        
        # Collect items
        items = []
        while not self.extraction_queue.empty() and len(items) < self.max_concurrent:
            try:
                item = self.extraction_queue.get_nowait()
                items.append(item)
            except asyncio.QueueEmpty:
                break
        
        if not items:
            return PhaseResult.no_work()
        
        logger.info(f"[Extraction] Processing {len(items)} files")
        
        # Process in parallel
        tasks = [self._process_file(item) for item in items]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        total_cost = 0.0
        total_seeds = 0
        
        for item, result in zip(items, results):
            if isinstance(result, Exception):
                logger.error(f"Extraction failed for {item.get('file_path', 'unknown')}: {result}")
                continue
            
            extraction_result, cost = result
            total_cost += cost
            total_seeds += len(extraction_result.seeds)
        
        if self.cost_tracker and total_cost > 0:
            self.cost_tracker.add_cost(
                phase=self.name,
                cost_usd=total_cost,
                model=EXTRACTION_MODEL,
            )
        
        logger.info(f"[Extraction] Extracted {total_seeds} seeds from {len(items)} files")
        
        return PhaseResult.work_done(cost_usd=total_cost)
    
    async def _process_file(self, item: Dict) -> Tuple[ExtractionResult, float]:
        """Process a single file and extract seeds."""
        
        async with self._semaphore:
            file_path = item.get("file_path", "")
            source_url = item.get("source_url", "")
            
            path = Path(file_path)
            if not path.exists():
                logger.warning(f"[Extraction] File not found: {file_path}")
                return ExtractionResult(source_file=file_path, seeds=[], error="File not found"), 0.0
            
            # Read content
            content = path.read_text(encoding='utf-8', errors='ignore')
            
            # Check if we need to chunk
            token_count = self._count_tokens(content)
            
            if token_count > CHUNK_THRESHOLD:
                # Process in chunks
                chunks = self._chunk_content(content, token_count)
                all_seeds = []
                total_cost = 0.0
                
                for i, chunk in enumerate(chunks):
                    if self.stop_checker and self.stop_checker():
                        break
                    
                    seeds, cost = await self._extract_from_content(
                        chunk, 
                        file_path,
                        chunk_info=f"chunk {i+1}/{len(chunks)}"
                    )
                    all_seeds.extend(seeds)
                    total_cost += cost
                
                await self._store_seeds(all_seeds, file_path)
                
                return ExtractionResult(
                    source_file=file_path,
                    seeds=all_seeds,
                ), total_cost
            else:
                # Process whole file
                seeds, cost = await self._extract_from_content(content, file_path)
                await self._store_seeds(seeds, file_path)
                
                return ExtractionResult(
                    source_file=file_path,
                    seeds=seeds,
                ), cost
    
    def _chunk_content(self, content: str, token_count: int) -> List[str]:
        """Chunk content for processing."""
        chunk_size = CHUNK_THRESHOLD
        overlap = 500
        
        if self._encoding:
            tokens = self._encoding.encode(content)
            chunks = []
            start = 0
            while start < len(tokens):
                end = min(start + chunk_size, len(tokens))
                chunk_tokens = tokens[start:end]
                chunks.append(self._encoding.decode(chunk_tokens))
                start += chunk_size - overlap
            return chunks
        else:
            # Fallback: character-based
            char_chunk = chunk_size * 4
            char_overlap = overlap * 4
            chunks = []
            start = 0
            while start < len(content):
                end = min(start + char_chunk, len(content))
                chunks.append(content[start:end])
                start += char_chunk - char_overlap
            return chunks
    
    async def _extract_from_content(
        self,
        content: str,
        file_path: str,
        chunk_info: str = "",
    ) -> Tuple[List[ExtractedSeed], float]:
        """Extract seeds from content using LLM."""
        
        prompt = f"""You are identifying seeds in content for dataset generation.

## What is a seed?
A seed is a distinct piece of content that could become ONE row in the dataset.
- It should be self-contained (makes sense on its own)
- It should have enough information to generate a full dataset row
- Different seeds from the same file = different rows

## Dataset Goal
{self.state.generation_prompt}

## Column Schema
{self._format_schema()}

## Source File
{file_path}{f' ({chunk_info})' if chunk_info else ''}

## Content
{content[:50000]}

## Your Task
Identify seeds in this content. For each seed, provide:
1. **source**: A pointer to locate it. Can be:
   - The file path if the whole file is one seed
   - "lines 45-67" if it's a specific section
   - "item 3" or similar identifier
   - First few words "Starting with 'The customer reported...'" 
2. **note**: Natural language description of what this seed contains and how it relates to the dataset goal

Return JSON:
{{
  "seeds": [
    {{"source": "lines 23-45", "note": "Customer complaint about delayed shipping, mentions order #12345"}},
    {{"source": "lines 67-89", "note": "Product review for headphones, 4-star rating, mentions sound quality"}}
  ]
}}

If no valid seeds exist, return {{"seeds": []}}

Be generous in identifying seeds but accurate in descriptions."""

        try:
            response, cost = await self.openai_client.responses_create(
                model=EXTRACTION_MODEL,
                input=[{"role": "user", "content": prompt}],
            )
            
            seeds = self._parse_response(response.output_text, file_path)
            
            # Update stats
            self._total_seeds += len(seeds)
            for seed in seeds:
                # Estimate quality based on note length (rough heuristic)
                quality = min(10, 5 + len(seed.note) // 20)
                self._total_quality_sum += quality
                if quality >= 8:
                    self._quality_counts["8-10"] += 1
                elif quality >= 5:
                    self._quality_counts["5-7"] += 1
                else:
                    self._quality_counts["0-4"] += 1
            
            self._processed_files.append(file_path)
            
            return seeds, cost.total_cost_usd
            
        except Exception as e:
            logger.error(f"[Extraction] LLM call failed: {e}")
            return [], 0.0
    
    def _format_schema(self) -> str:
        """Format column schema for prompt."""
        if not self.state.columns:
            return "No schema defined"
        lines = []
        for col in self.state.columns:
            line = f"- {col.get('name')} ({col.get('type')})"
            if col.get('description'):
                line += f": {col['description']}"
            lines.append(line)
        return "\n".join(lines)
    
    def _parse_response(self, response: str, file_path: str) -> List[ExtractedSeed]:
        """Parse LLM response into seeds."""
        seeds = []
        
        try:
            # Extract JSON
            response = response.strip()
            if response.startswith("```"):
                response = response.split("```")[1]
                if response.startswith("json"):
                    response = response[4:]
            
            data = json.loads(response)
            
            for s in data.get("seeds", []):
                source = s.get("source", "")
                note = s.get("note", "")
                
                if not note:
                    continue
                
                # Build full source reference
                if source and not source.startswith(file_path):
                    full_source = f"{file_path}:{source}"
                else:
                    full_source = file_path
                
                seeds.append(ExtractedSeed(
                    source=full_source,
                    note=note,
                ))
                
        except json.JSONDecodeError as e:
            logger.warning(f"[Extraction] Failed to parse response: {e}")
        
        return seeds
    
    async def _store_seeds(self, seeds: List[ExtractedSeed], source_file: str):
        """Store seeds in database."""
        
        for seed in seeds:
            db_seed = ProjectSeed(
                id=uuid.uuid4(),
                project_id=self.state.project_id,
                version_id=self.state.version_id,
                text=seed.source,  # Store pointer as text field
                extraction_metadata={
                    "note": seed.note,
                    "source_file": source_file,
                    "extracted_at": datetime.now(timezone.utc).isoformat(),
                },
            )
            self.db.add(db_seed)
        
        self.db.commit()
    
    def get_stats(self) -> Dict:
        """Get current extraction stats for feedback."""
        avg_quality = 0.0
        if self._total_seeds > 0:
            avg_quality = self._total_quality_sum / self._total_seeds
        
        return {
            "total_seeds": self._total_seeds,
            "avg_quality": avg_quality,
            "quality_distribution": dict(self._quality_counts),
            "files_processed": len(self._processed_files),
        }
    
    def is_complete(self) -> bool:
        """Extraction processes queue, never 'complete'."""
        return False
    
    def get_status(self):
        """Get current status."""
        from dsl_worker.phases.base import PhaseStatus
        
        return PhaseStatus(
            phase_name=self.name,
            status="active" if not self.extraction_queue.empty() else "idle",
            progress=f"{self._total_seeds} seeds from {len(self._processed_files)} files"
        )