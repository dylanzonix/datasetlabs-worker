"""
Phase: Research (v2)

The intelligent research agent that orchestrates source discovery and exploration.

Responsibilities:
- Brave search for source discovery (auto-fetches, summarizes, saves to files)
- Fetch specific URLs found in content
- Dispatch browser agents for complex interactive sites
- Code execution for deep inspection of files
- Maintain smart context (summaries + file paths persist)
- Feedback loop with extraction phase

Tools:
- brave_search(queries): Search + auto-fetch + summarize + save to files
- fetch_page(url): Fetch a specific URL, summarize if large, save to file
- browse_complex(url, goal): Dispatch browser agent for interactive sites
- code_exec(script): Run Python to inspect files, grep, parse, etc.
- done(reason): Mark research as complete
"""

import asyncio
import json
import logging
import os
import re
import uuid
import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import httpx
import tiktoken

from dsl_worker.phases.base import Phase, PhaseResult
from dsl_worker.phases.browser_pool import BrowserPool
from dsl_worker.phases.sandbox import SandboxExecutor

logger = logging.getLogger(__name__)

RESEARCH_MODEL = os.getenv("RESEARCH_MODEL", "gpt-5.2")
SUMMARIZE_MODEL = os.getenv("SUMMARIZE_MODEL", "gpt-5-nano")
EXTRACTION_MODEL = os.getenv("EXTRACTION_MODEL", "gpt-5-nano")

# Token thresholds
FULL_CONTENT_THRESHOLD = 4000  # Show full content if under this
SUMMARY_TARGET_TOKENS = 500    # Target summary length
MAX_TOOL_RESPONSE_TOKENS = 100_000  # Hard limit - error if exceeded
WARN_TOOL_RESPONSE_TOKENS = 30_000  # Soft warning


@dataclass
class SourceInfo:
    """Tracked information about a discovered source."""
    url: str
    file_path: str  # Path in workspace
    summary: str
    token_count: int
    fetched_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


def url_to_filename(url: str) -> str:
    """Convert URL to safe filename. Always .md for web content."""
    # Hash the URL for uniqueness
    url_hash = hashlib.md5(url.encode()).hexdigest()[:8]
    
    # Extract domain for readability
    from urllib.parse import urlparse
    parsed = urlparse(url)
    domain = parsed.netloc.replace(".", "_").replace(":", "_")
    
    # Check if it's a downloadable file type (not HTML)
    path_lower = parsed.path.lower()
    if path_lower.endswith('.pdf'):
        ext = ".pdf.txt"  # Parsed PDF text
    elif path_lower.endswith(('.docx', '.doc')):
        ext = ".docx.txt"
    elif path_lower.endswith(('.xlsx', '.xls')):
        ext = ".xlsx.txt"
    elif path_lower.endswith(('.pptx', '.ppt')):
        ext = ".pptx.txt"
    else:
        # All web pages become markdown
        ext = ".md"
    
    return f"{domain}_{url_hash}{ext}"


class ResearchPhaseV2(Phase):
    """
    Research agent that explores sources and dispatches extraction.
    
    Maintains:
    - Unified workspace for all files
    - Persistent context (summaries + file paths)
    - Feedback loop with extraction phase
    """
    
    def __init__(
        self,
        *args,
        extraction_queue: Optional[asyncio.Queue] = None,
        feedback_callback: Optional[Callable[[], Dict]] = None,
        stop_checker: Optional[Callable[[], bool]] = None,
        cost_tracker: Optional[Any] = None,
        workspace_dir: Optional[str] = None,
        browser_pool_size: int = 5,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        
        self.extraction_queue = extraction_queue or asyncio.Queue()
        self.feedback_callback = feedback_callback
        self.stop_checker = stop_checker
        self.cost_tracker = cost_tracker
        
        # Unified workspace
        self.workspace_dir = Path(workspace_dir or f"./workspace_{self.state.project_id}")
        self.workspace_dir.mkdir(parents=True, exist_ok=True)
        (self.workspace_dir / "uploads").mkdir(exist_ok=True)
        (self.workspace_dir / "web").mkdir(exist_ok=True)
        (self.workspace_dir / "extracted").mkdir(exist_ok=True)
        
        # Brave API
        self.brave_api_key = os.getenv("BRAVE_API_KEY")
        if not self.brave_api_key:
            logger.warning("[Research] BRAVE_API_KEY not set - web search will fail!")
        
        # Browser pool for concurrency
        self.browser_pool = BrowserPool(
            size=browser_pool_size,
            headless=os.getenv("BROWSER_HEADLESS", "false").lower() in ("true", "1"),
        )
        self._browser_started = False
        self._browser_start_lock = asyncio.Lock()
        
        # Sandbox for code execution
        self._sandbox: Optional[SandboxExecutor] = None
        
        # Tokenizer
        try:
            self._encoding = tiktoken.get_encoding("cl100k_base")
        except:
            self._encoding = tiktoken.get_encoding("cl100k_base")
        
        # Discovered sources (for reference)
        self._sources: Dict[str, SourceInfo] = {}  # url -> SourceInfo
        
        # Extraction stats (for relative quality feedback)
        self._extraction_stats = {
            "files_processed": 0,
            "total_seeds": 0,
        }
        
        # Conversation messages (tool calls and results)
        self._messages: List[Dict[str, str]] = []
        
        # Turn counter
        self._turn_count = 0
        self._total_cost = 0.0
        
        # Files loaded flag
        self._files_loaded = False
        
    def _count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        return len(self._encoding.encode(text))
    
    def _get_current_stats(self) -> Dict:
        """Get current seed/quota stats."""
        # Use feedback callback if available (reads from assignment phase)
        if self.feedback_callback:
            stats = self.feedback_callback()
            # Merge with internal extraction stats
            stats["total_seeds"] = max(stats.get("total_seeds", 0), self._extraction_stats["total_seeds"])
            return stats
        
        # Fallback to internal tracking
        return {
            "total_seeds": self._extraction_stats["total_seeds"],
            "avg_quality": 0.0,
            "remaining_quotas": {},
            "total_remaining": self.state.num_samples,
        }
    
    def _check_should_stop(self) -> bool:
        """Check if we should stop/pause."""
        if self.stop_checker and self.stop_checker():
            logger.info("[Research] Pause/stop requested")
            return True
        return False
    
    # =========================================================================
    # Context Building
    # =========================================================================
    
    def _build_persistent_context(self) -> str:
        """
        Build the persistent part of context (always included).
        
        Includes:
        - Project goal and schema
        - Diversity quotas (remaining)
        - Discovered sources (summaries + file paths)
        - Extraction feedback
        """
        # Schema
        schema_lines = []
        for col in self.state.columns or []:
            line = f"- {col.get('name')} ({col.get('type')})"
            if col.get('description'):
                line += f": {col['description']}"
            schema_lines.append(line)
        schema = "\n".join(schema_lines) if schema_lines else "No schema defined"
        
        # Diversity
        diversity_lines = []
        for axis in self.state.diversity_spec or []:
            values = [v.get('value') for v in axis.get('values', [])]
            diversity_lines.append(f"- {axis.get('name')}: {', '.join(values)}")
        diversity = "\n".join(diversity_lines) if diversity_lines else "No diversity requirements"
        
        # Quotas
        stats = self._get_current_stats()
        remaining = stats.get("remaining_quotas", {})
        if remaining:
            quotas_lines = [f"- {k}: need {v} more" for k, v in remaining.items() if v > 0]
            quotas = "\n".join(quotas_lines) if quotas_lines else "All quotas filled!"
        else:
            quotas = f"Need {self.state.num_samples} total samples"
        
        # Sources
        if self._sources:
            sources_lines = []
            for url, info in list(self._sources.items())[-30:]:  # Last 30
                sources_lines.append(
                    f"**{url[:60]}{'...' if len(url) > 60 else ''}**\n"
                    f"  {info.summary[:200]}{'...' if len(info.summary) > 200 else ''}\n"
                    f"  [{info.file_path}] ({info.token_count} tokens)"
                )
            sources = "\n\n".join(sources_lines)
        else:
            sources = "No sources discovered yet"
        
        # Uploaded files
        files_lines = []
        uploads_dir = self.workspace_dir / "uploads"
        if uploads_dir.exists():
            for f in uploads_dir.iterdir():
                if f.is_file():
                    files_lines.append(f"- {f.name} [workspace/uploads/{f.name}]")
        uploaded = "\n".join(files_lines) if files_lines else "No files uploaded"
        
        # Feedback
        feedback = (
            f"- Total seeds: {stats.get('total_seeds', 0)}\n"
            f"- Avg quality: {stats.get('avg_quality', 0):.1f}/10"
        )
        
        return f"""## Project Goal
{self.state.generation_prompt}

## Column Schema
{schema}

## Diversity Requirements
{diversity}

## Remaining Quotas
{quotas}

## Uploaded Files
{uploaded}

## Discovered Sources
{sources}

## Extraction Feedback
{feedback}
"""

    def _build_system_prompt(self) -> str:
        """Build system prompt for research agent."""
        # Format schema
        schema_lines = []
        for col in self.state.columns or []:
            line = f"- {col.get('name')} ({col.get('type')})"
            if col.get('description'):
                line += f": {col['description']}"
            schema_lines.append(line)
        schema_str = "\n".join(schema_lines) if schema_lines else "No schema defined"
        
        # Format quotas
        stats = self._get_current_stats()
        remaining = stats.get("remaining_quotas", {})
        total_remaining = stats.get("total_remaining", self.state.num_samples)
        
        if remaining:
            # Group by axis for cleaner display
            by_axis = {}
            for key, count in remaining.items():
                if ":" in key and count > 0:
                    axis, value = key.split(":", 1)
                    if axis not in by_axis:
                        by_axis[axis] = []
                    by_axis[axis].append(f"{value}: {count}")
            
            quota_lines = [f"**Total rows remaining: {total_remaining}**\n"]
            for axis, values in by_axis.items():
                quota_lines.append(f"{axis}: {', '.join(values)}")
            quotas_str = "\n".join(quota_lines)
        else:
            quotas_str = f"Need {self.state.num_samples} total rows"
        
        return f"""You are a research agent finding seeds for dataset generation.

## Your Goal
Find and queue content (seeds) that can be used to generate dataset rows. Seeds come in all shapes and sizes - some map directly to the target format, others provide partial info, others just inspiration or context. It's a spectrum. The closer to the target row the better, but any content that helps generate a unique row has value.

Your job is coverage and exploration, not detailed extraction. If a source looks promising, mark it. If it's obviously irrelevant, skip it. Don't get caught up verifying every detail - the extraction phase handles that and will tell you how it went.

## Tools

**brave_search(queries)**
Search the web. Results are automatically fetched and saved to files.
- Full page content returned if under ~4k tokens
- Summarized if larger (full content still saved to file for deeper inspection via code_exec)

**fetch_page(url)**
Fetch any URL and save it to a file. Works for web pages, PDFs, documents.
Same pattern: full content if small, summarized if large.

**browse_complex(url, goal)**
Dispatch a browser agent when interaction is needed - login, clicking through pages, filling search/forms, pagination, scrolling to load content.

The browser agent is an AI that navigates naturally - you give it plain language goals, not selectors or programmatic steps.

It has two special tools:
- `mark_for_extraction`: Saves current page and queues it for seed extraction
- `checkpoint`: Reports back to you for guidance

You control the session through checkpoints:
1. Give a clear initial goal and when to checkpoint (e.g., "login, then checkpoint" or "find the listings page, then checkpoint")
2. At each checkpoint, evaluate progress and give next steps or end the session
3. Continue until you've got what you need or it's a dead end

**mark_for_extraction(file_path, description)**
Queue a file for seed extraction. Runs extraction immediately and returns feedback.
You'll learn how many seeds were found and their quality relative to other sources - use this to decide whether to dig deeper or move on.

**code_exec(script)**
Execute Python to inspect files. Files at /workspace/uploads/, /workspace/web/, /workspace/extracted/.
Use for grep, parsing, viewing specific sections when summaries aren't enough.

**done(reason)**
Mark research complete when quotas are met.

Always respond with a tool call. Never respond with just text.

<dataset_instructions>
{self.state.generation_prompt}
</dataset_instructions>

<row_schema>
Exact format of each dataset row:
{schema_str}
</row_schema>

<diversity_quotas>
{quotas_str}
</diversity_quotas>"""

    # =========================================================================
    # Tool Implementations
    # =========================================================================
    
    async def _do_brave_search(self, args: Dict) -> Tuple[str, float]:
        """
        Search with Brave, auto-fetch all results, summarize, save to files.
        """
        queries = args.get("queries", [])
        
        if not self.brave_api_key:
            return "Error: BRAVE_API_KEY not configured", 0.0
        
        logger.info(f"[Research] Brave search: {queries}")
        
        all_results = []
        cost = 0.0
        
        # Search
        async with httpx.AsyncClient() as client:
            for query in queries:
                if self._check_should_stop():
                    break
                    
                try:
                    response = await client.get(
                        "https://api.search.brave.com/res/v1/web/search",
                        params={"q": query, "count": 10},
                        headers={"X-Subscription-Token": self.brave_api_key},
                        timeout=30.0,
                    )
                    response.raise_for_status()
                    data = response.json()
                    
                    for r in data.get("web", {}).get("results", []):
                        url = r.get("url", "")
                        if url and url not in self._sources:
                            all_results.append({
                                "url": url,
                                "title": r.get("title", ""),
                                "snippet": r.get("description", ""),
                            })
                            
                except Exception as e:
                    logger.error(f"[Research] Search failed for '{query}': {e}")
                    
                await asyncio.sleep(1.1)  # Rate limit
        
        if not all_results:
            return "No results found. Try different queries.", 0.0
        
        # Fetch and summarize each result
        logger.info(f"[Research] Fetching {len(all_results)} results...")
        
        fetch_tasks = []
        for r in all_results[:15]:  # Limit to 15 results
            if self._check_should_stop():
                break
            fetch_tasks.append(self._fetch_and_save(r["url"]))
        
        fetch_results = await asyncio.gather(*fetch_tasks, return_exceptions=True)
        
        # Build response
        response_parts = [f"Found {len(all_results)} results, fetched {len(fetch_tasks)}:\n"]
        
        for r, fetch_result in zip(all_results[:15], fetch_results):
            if isinstance(fetch_result, Exception):
                response_parts.append(f"\n**{r['title']}**\n{r['url']}\nFetch failed: {fetch_result}\n")
            else:
                source_info, fetch_cost = fetch_result
                cost += fetch_cost
                if source_info:
                    response_parts.append(
                        f"\n**{r['title']}**\n"
                        f"{source_info.summary}\n"
                        f"[{source_info.file_path}] ({source_info.token_count} tokens)\n"
                    )
        
        return "".join(response_parts), cost
    
    async def _do_fetch_page(self, args: Dict) -> Tuple[str, float]:
        """Fetch a specific URL, summarize if large, save to file."""
        url = args.get("url", "")
        
        if not url:
            return "Error: URL required", 0.0
        
        if url in self._sources:
            info = self._sources[url]
            return f"Already fetched: {info.summary}\n[{info.file_path}]", 0.0
        
        source_info, cost = await self._fetch_and_save(url)
        
        if source_info is None:
            return f"Failed to fetch {url}", cost
        
        # If small enough, include full content
        if source_info.token_count < FULL_CONTENT_THRESHOLD:
            content = (self.workspace_dir / source_info.file_path).read_text(errors='ignore')
            return f"Fetched {url}:\n\n{content}\n\n[Saved to {source_info.file_path}]", cost
        else:
            return (
                f"Fetched {url} ({source_info.token_count} tokens):\n\n"
                f"{source_info.summary}\n\n"
                f"[Full content at {source_info.file_path} - use code_exec to inspect]"
            ), cost
    
    async def _fetch_and_save(self, url: str) -> Tuple[Optional[SourceInfo], float]:
        """Fetch URL, save to file, summarize, return SourceInfo."""
        cost = 0.0
        
        try:
            # Use browser pool's simple_fetch (handles PDFs, DOCX, etc.)
            async with self._browser_start_lock:
                if not self._browser_started:
                    await self.browser_pool.start()
                    self._browser_started = True
            
            content = await self.browser_pool.simple_fetch(url)
            
            if not content or len(content.strip()) < 50:
                logger.warning(f"[Research] Empty or minimal content from {url}")
                return None, 0.0
            
            # Save to file
            filename = url_to_filename(url)
            file_path = f"web/{filename}"
            full_path = self.workspace_dir / file_path
            full_path.write_text(content, encoding='utf-8')
            
            # Count tokens
            token_count = self._count_tokens(content)
            
            # Summarize
            if token_count > FULL_CONTENT_THRESHOLD:
                summary, sum_cost = await self._summarize(content, url)
                cost += sum_cost
            else:
                # Short content - use first part as summary
                summary = content[:500] + ("..." if len(content) > 500 else "")
            
            source_info = SourceInfo(
                url=url,
                file_path=file_path,
                summary=summary,
                token_count=token_count,
            )
            self._sources[url] = source_info
            
            logger.info(f"[Research] Saved {url} -> {file_path} ({token_count} tokens)")
            
            return source_info, cost
            
        except Exception as e:
            logger.error(f"[Research] Fetch failed for {url}: {e}")
            return None, cost
    
    async def _summarize(self, content: str, source: str) -> Tuple[str, float]:
        """Summarize content using small model."""
        # Truncate for summarization
        max_chars = 50000
        if len(content) > max_chars:
            content = content[:max_chars] + "\n\n[truncated]"
        
        prompt = f"""Summarize this content in 2-3 sentences. Focus on what data/information is available that could be useful for dataset generation.

Source: {source}

Content:
{content}

Summary:"""

        try:
            response, cost = await self.openai_client.responses_create(
                model=SUMMARIZE_MODEL,
                input=[{"role": "user", "content": prompt}],
            )
            return response.output_text.strip(), cost.total_cost_usd
        except Exception as e:
            logger.error(f"[Research] Summarization failed: {e}")
            return content[:300] + "...", 0.0
    
    async def _do_browse_complex(self, args: Dict) -> Tuple[str, float]:
        """Dispatch browser agent for interactive sites."""
        url = args.get("url", "")
        goal = args.get("goal", "")
        
        if not url or not goal:
            return "Error: url and goal required", 0.0
        
        logger.info(f"[Research] Starting browser agent: {url}")
        logger.info(f"[Research] Goal: {goal[:100]}...")
        
        async with self._browser_start_lock:
            if not self._browser_started:
                await self.browser_pool.start()
                self._browser_started = True
        
        # Create checkpoint callback that routes back to research agent
        async def handle_checkpoint(current_url: str, status: str) -> str:
            return await self._handle_browser_checkpoint(current_url, status)
        
        # Run browser agent with checkpoint pattern
        result = await self.browser_pool.browse_with_checkpoints(
            url=url,
            goal=goal,
            llm=self._get_browser_llm(),
            checkpoint_callback=handle_checkpoint,
            extraction_dir=self.workspace_dir / "extracted",
            extraction_queue=self.extraction_queue,
            stop_checker=self.stop_checker,
        )
        
        pages_marked = result.get("pages_marked", 0)
        success = result.get("success", False)
        error = result.get("error")
        
        response = f"Browser agent finished.\n- Pages marked for extraction: {pages_marked}\n- Success: {success}"
        if error:
            response += f"\n- Error: {error}"
        
        return response, 0.0
    
    async def _handle_browser_checkpoint(self, current_url: str, status: str) -> str:
        """
        Handle checkpoint from browser agent.
        Research agent decides what browser should do next.
        """
        stats = self._get_current_stats()
        
        prompt = f"""Browser agent checkpoint.

Current URL: {current_url}

Browser agent reports:
{status}

Current stats:
- Seeds collected: {stats.get('total_seeds', 0)}
- Remaining quotas: {stats.get('remaining_quotas', {})}

What should the browser agent do next? Options:
1. Continue exploring (give specific instructions)
2. Mark current page for extraction (if it has useful content)
3. Navigate somewhere else (give URL or instructions)
4. End session (if done with this site)

Be concise and specific. This goes directly to the browser agent."""

        try:
            response, cost = await self.openai_client.responses_create(
                model=RESEARCH_MODEL,
                input=[{"role": "user", "content": prompt}],
                max_output_tokens=500,
            )
            self._total_cost += cost.total_cost_usd
            return response.output_text.strip()
        except Exception as e:
            logger.error(f"[Research] Checkpoint handling failed: {e}")
            return "Continue with your best judgment, then checkpoint again."
    
    async def _do_code_exec(self, args: Dict) -> Tuple[str, float]:
        """Execute Python code in sandbox with access to workspace."""
        script = args.get("script", "")
        
        if not script:
            return "Error: script required", 0.0
        
        logger.info(f"[Research] Executing code...")
        
        # Initialize sandbox if needed
        if self._sandbox is None:
            self._sandbox = SandboxExecutor(use_pool=True, pool_size=2)
        
        # Execute with workspace mounted
        result = self._sandbox.execute(
            script=script,
            workspace_dir=str(self.workspace_dir),
            timeout=120,
        )
        
        if result.success:
            output = result.stdout
            if result.stderr:
                output += f"\n\nStderr:\n{result.stderr}"
            
            # Check token count
            tokens = self._count_tokens(output)
            if tokens > MAX_TOOL_RESPONSE_TOKENS:
                return (
                    f"Error: Output too large ({tokens} tokens). "
                    "Refine your query to return less data, or write results to a file."
                ), 0.0
            
            if tokens > WARN_TOOL_RESPONSE_TOKENS:
                logger.warning(f"[Research] Large code output: {tokens} tokens")
            
            return output if output.strip() else "Code executed successfully (no output)", 0.0
        else:
            return f"Error: {result.error}\n\nStderr:\n{result.stderr}", 0.0
    
    async def _do_mark_for_extraction(self, args: Dict) -> Tuple[str, float]:
        """Mark a file for seed extraction - runs extraction synchronously and returns feedback."""
        file_path = args.get("file_path", "")
        description = args.get("description", "")
        
        if not file_path:
            return "Error: file_path required", 0.0
        
        # Resolve path
        if not file_path.startswith("/"):
            full_path = self.workspace_dir / file_path
        else:
            full_path = Path(file_path)
        
        # Check file exists
        if not full_path.exists():
            # Try workspace subdirs
            for subdir in ["web", "uploads", "extracted"]:
                alt_path = self.workspace_dir / subdir / file_path
                if alt_path.exists():
                    full_path = alt_path
                    break
        
        if not full_path.exists():
            return f"Error: File not found: {file_path}", 0.0
        
        logger.info(f"[Research] Extracting seeds from: {file_path}")
        
        # Run extraction synchronously
        seeds, cost = await self._extract_seeds_from_file(str(full_path), description)
        
        # Track stats
        self._extraction_stats["files_processed"] += 1
        self._extraction_stats["total_seeds"] += len(seeds)
        
        # Calculate relative quality
        quality_desc = ""
        if len(seeds) > 0:
            files_processed = self._extraction_stats["files_processed"]
            avg_per_file = self._extraction_stats["total_seeds"] / files_processed if files_processed > 0 else 0
            
            if len(seeds) > avg_per_file * 1.5:
                quality_desc = "Above average yield"
            elif len(seeds) < avg_per_file * 0.5:
                quality_desc = "Below average yield"
            else:
                quality_desc = "Average yield"
        
        # Build response
        if len(seeds) == 0:
            return f"Processed: {file_path}\nSeeds found: 0\nThis source didn't yield usable seeds.", cost
        else:
            # Summarize diversity coverage from seed notes
            response = f"Processed: {file_path}\nSeeds found: {len(seeds)}\n{quality_desc}"
            
            # Add a few example seed notes
            if seeds:
                response += "\n\nSample seeds:"
                for seed in seeds[:3]:
                    note = seed.get("note", "")[:100]
                    response += f"\n- {note}{'...' if len(seed.get('note', '')) > 100 else ''}"
                if len(seeds) > 3:
                    response += f"\n... and {len(seeds) - 3} more"
            
            return response, cost
    
    async def _extract_seeds_from_file(self, file_path: str, description: str) -> Tuple[List[Dict], float]:
        """Run seed extraction on a single file synchronously."""
        
        # Read content
        try:
            content = Path(file_path).read_text(encoding='utf-8', errors='ignore')
        except Exception as e:
            logger.error(f"[Research] Failed to read {file_path}: {e}")
            return [], 0.0
        
        # Truncate if huge
        if len(content) > 50000:
            content = content[:50000] + "\n\n[truncated]"
        
        # Format schema
        schema_lines = []
        for col in self.state.columns or []:
            line = f"- {col.get('name')} ({col.get('type')})"
            if col.get('description'):
                line += f": {col['description']}"
            schema_lines.append(line)
        schema_str = "\n".join(schema_lines) if schema_lines else "No schema defined"
        
        prompt = f"""Identify seeds in this content for dataset generation.

A seed is content that can help generate ONE dataset row. Seeds come in all shapes - direct matches, partial info, or just useful context. Identify distinct pieces that could each become a row.

<dataset_instructions>
{self.state.generation_prompt}
</dataset_instructions>

<row_schema>
{schema_str}
</row_schema>

<source_file>
{file_path}
</source_file>

<source_description>
{description}
</source_description>

<content>
{content}
</content>

For each seed found, provide:
- source: How to locate it (e.g., "lines 23-45", "section 3", "item about X")
- note: What this seed contains and how it helps generate a row

Return JSON:
{{
  "seeds": [
    {{"source": "lines 23-45", "note": "Customer complaint about shipping delays, order #12345"}},
    {{"source": "paragraph 3", "note": "Product review, 4 stars, mentions durability issues"}}
  ]
}}

If no usable seeds, return {{"seeds": []}}"""

        try:
            response, cost = await self.openai_client.responses_create(
                model=EXTRACTION_MODEL,
                input=[{"role": "user", "content": prompt}],
            )
            
            # Parse response
            response_text = response.output_text.strip()
            if response_text.startswith("```"):
                response_text = response_text.split("```")[1]
                if response_text.startswith("json"):
                    response_text = response_text[4:]
            
            data = json.loads(response_text)
            seeds = data.get("seeds", [])
            
            # Store seeds in DB
            for seed in seeds:
                from dsl_api.models.project_seed import ProjectSeed
                db_seed = ProjectSeed(
                    id=uuid.uuid4(),
                    project_id=self.state.project_id,
                    version_id=self.state.version_id,
                    text=f"{file_path}:{seed.get('source', '')}",
                    extraction_metadata={
                        "note": seed.get("note", ""),
                        "source_file": file_path,
                        "extracted_at": datetime.now(timezone.utc).isoformat(),
                    },
                )
                self.db.add(db_seed)
            
            if seeds:
                self.db.commit()
            
            return seeds, cost.total_cost_usd
            
        except json.JSONDecodeError as e:
            logger.warning(f"[Research] Failed to parse extraction response: {e}")
            return [], 0.0
        except Exception as e:
            logger.error(f"[Research] Extraction failed: {e}")
            return [], 0.0
    
    def _get_browser_llm(self):
        """Get LLM for browser agent."""
        from browser_use.llm.openai.chat import ChatOpenAI
        return ChatOpenAI(model="gpt-5.2")
    
    def _get_tools(self) -> List[Dict]:
        """Tool definitions for research agent."""
        return [
            {
                "type": "function",
                "name": "brave_search",
                "description": "Search the web. Results are automatically fetched, summarized, and saved to files. Returns summaries and file paths.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "queries": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Search queries to run"
                        }
                    },
                    "required": ["queries"]
                }
            },
            {
                "type": "function",
                "name": "fetch_page",
                "description": "Fetch any URL and save it to a file. Works for web pages, PDFs, documents. Summarizes if large.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "url": {"type": "string", "description": "URL to fetch"}
                    },
                    "required": ["url"]
                }
            },
            {
                "type": "function",
                "name": "browse_complex",
                "description": "Dispatch browser agent when interaction is needed (login, pagination, clicking, forms). Provide clear goal. Agent checkpoints back for guidance.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "url": {"type": "string", "description": "Starting URL"},
                        "goal": {"type": "string", "description": "What to accomplish - be specific and focused"}
                    },
                    "required": ["url", "goal"]
                }
            },
            {
                "type": "function",
                "name": "mark_for_extraction",
                "description": "Queue a file for seed extraction. Use when you've found content that likely contains seeds.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "file_path": {"type": "string", "description": "Path to file in workspace (e.g., 'web/example_com_abc.md')"},
                        "description": {"type": "string", "description": "What kind of seeds might be in this file"}
                    },
                    "required": ["file_path"]
                }
            },
            {
                "type": "function",
                "name": "code_exec",
                "description": "Execute Python to inspect files. Files at /workspace/uploads/, /workspace/web/, /workspace/extracted/. Libraries: pandas, BeautifulSoup, re, json.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "script": {"type": "string", "description": "Python code to execute"}
                    },
                    "required": ["script"]
                }
            },
            {
                "type": "function",
                "name": "done",
                "description": "Mark research as complete.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "reason": {"type": "string", "description": "Why research is complete"}
                    },
                    "required": ["reason"]
                }
            },
        ]
    
    def _parse_action(self, response) -> Dict:
        """Parse action from LLM response."""
        for item in response.output:
            if item.type == "function_call":
                return {
                    "type": item.name,
                    "args": json.loads(item.arguments),
                    "call_id": item.call_id,
                }
        return {"type": "message", "content": response.output_text}
    
    async def _execute_action(self, action: Dict) -> Tuple[str, float]:
        """Execute an action and return response + cost."""
        action_type = action.get("type")
        args = action.get("args", {})
        
        logger.info(f"[Research] Action: {action_type}")
        
        if action_type == "brave_search":
            return await self._do_brave_search(args)
        elif action_type == "fetch_page":
            return await self._do_fetch_page(args)
        elif action_type == "browse_complex":
            return await self._do_browse_complex(args)
        elif action_type == "mark_for_extraction":
            return await self._do_mark_for_extraction(args)
        elif action_type == "code_exec":
            return await self._do_code_exec(args)
        elif action_type == "done":
            logger.info(f"[Research] Complete: {args.get('reason')}")
            return f"Research complete: {args.get('reason')}", 0.0
        elif action_type == "message":
            return action.get("content", ""), 0.0
        else:
            return f"Unknown action: {action_type}", 0.0
    
    # =========================================================================
    # Main Loop
    # =========================================================================
    
    def should_run(self) -> bool:
        """Run if we haven't filled quotas."""
        stats = self._get_current_stats()
        remaining = stats.get("remaining_quotas", {})
        total_remaining = sum(remaining.values()) if remaining else self.state.num_samples
        return total_remaining > 0
    
    async def execute_once(self) -> PhaseResult:
        """One turn of the research loop."""
        self._turn_count += 1
        stats = self._get_current_stats()
        
        logger.info(f"")
        logger.info(f"{'='*60}")
        logger.info(f"[Research] TURN {self._turn_count}")
        logger.info(f"[Research] Seeds: {stats.get('total_seeds', 0)} | Sources: {len(self._sources)}")
        logger.info(f"{'='*60}")
        
        if self._check_should_stop():
            return PhaseResult.no_work()
        
        # Load uploaded files on first run
        if not self._files_loaded:
            await self._load_uploaded_files()
            self._files_loaded = True
        
        # Build messages for this turn
        messages = self._build_messages()
        
        # Get action from LLM
        try:
            response, cost = await self.openai_client.responses_create(
                model=RESEARCH_MODEL,
                input=messages,
                tools=self._get_tools(),
            )
            self._total_cost += cost.total_cost_usd
        except Exception as e:
            logger.error(f"[Research] LLM call failed: {e}")
            return PhaseResult.no_work()
        
        if self._check_should_stop():
            return PhaseResult.no_work()
        
        # Parse and execute action
        action = self._parse_action(response)
        action_type = action.get("type")
        
        logger.info(f"[Research] Decided: {action_type}")
        
        result_text, action_cost = await self._execute_action(action)
        self._total_cost += action_cost
        
        # Track cost
        total_turn_cost = cost.total_cost_usd + action_cost
        if self.cost_tracker and total_turn_cost > 0:
            self.cost_tracker.add_cost(
                phase=self.name,
                cost_usd=total_turn_cost,
                model=RESEARCH_MODEL,
            )
        
        # Record this turn in history (for context in next turn)
        self._record_turn(action, result_text)
        
        # Check if done
        if action_type == "done":
            return PhaseResult.work_done(cost_usd=total_turn_cost)
        
        logger.info(f"[Research] Turn {self._turn_count} complete")
        
        return PhaseResult.work_done(cost_usd=total_turn_cost)
    
    def _build_messages(self) -> List[Dict]:
        """Build messages for LLM call."""
        # System prompt (includes dataset instructions, schema, quotas)
        messages = [
            {"role": "system", "content": self._build_system_prompt()},
        ]
        
        # Add conversation history (tool calls and results)
        for turn in self._messages[-30:]:  # Last 30 turns
            messages.append(turn)
        
        # Add progress update
        stats = self._get_current_stats()
        total_remaining = stats.get("total_remaining", self.state.num_samples)
        
        # List uploaded files if any
        uploaded_info = ""
        uploads_dir = self.workspace_dir / "uploads"
        if uploads_dir.exists():
            files = list(uploads_dir.iterdir())
            if files:
                uploaded_info = f"\n\nUploaded files available:\n" + "\n".join([f"- uploads/{f.name}" for f in files if f.is_file()])
        
        progress_msg = f"Seeds extracted so far: {stats.get('total_seeds', 0)}\nRows remaining: {total_remaining}{uploaded_info}\n\nContinue."
        
        messages.append({
            "role": "user",
            "content": progress_msg
        })
        
        return messages
    
    def _record_turn(self, action: Dict, result: str):
        """Record a turn in conversation history for Responses API format."""
        action_type = action.get("type")
        args = action.get("args", {})
        
        # Don't record "message" type (that's a fallback)
        if action_type == "message":
            return
        
        call_id = action.get("call_id", f"call_{self._turn_count}")
        
        # Responses API format: function_call item
        self._messages.append({
            "type": "function_call",
            "call_id": call_id,
            "name": action_type,
            "arguments": json.dumps(args),
        })
        
        # Truncate result if too long
        result_truncated = result[:3000] if len(result) > 3000 else result
        if len(result) > 3000:
            result_truncated += f"\n\n[... truncated, {len(result)} total chars]"
        
        # Responses API format: function_call_output item
        self._messages.append({
            "type": "function_call_output",
            "call_id": call_id,
            "output": result_truncated,
        })
    
    async def _load_uploaded_files(self):
        """Download uploaded files from Azure to workspace."""
        if not hasattr(self.state, 'files_snapshot') or not self.state.files_snapshot:
            logger.info("[Research] No uploaded files")
            return
        
        logger.info(f"[Research] Loading {len(self.state.files_snapshot)} uploaded files...")
        
        uploads_dir = self.workspace_dir / "uploads"
        
        for f in self.state.files_snapshot:
            filename = f.get('filename')
            blob_path = f.get('blob_path')
            
            if not filename or not blob_path:
                continue
            
            local_path = uploads_dir / filename
            
            try:
                if self.blob_service_client:
                    container = os.getenv("AZURE_STORAGE_CONTAINER", "uploads")
                    blob_client = self.blob_service_client.get_blob_client(
                        container=container,
                        blob=blob_path
                    )
                    with open(local_path, "wb") as f:
                        f.write(blob_client.download_blob().readall())
                    logger.info(f"[Research] Downloaded: {filename}")
                else:
                    logger.warning(f"[Research] No blob client, can't download {filename}")
            except Exception as e:
                logger.error(f"[Research] Failed to download {filename}: {e}")
    
    def is_complete(self) -> bool:
        """Complete when quotas filled."""
        stats = self._get_current_stats()
        remaining = stats.get("remaining_quotas", {})
        if not remaining:
            return False
        return sum(remaining.values()) == 0
    
    def get_status(self):
        """Get current status."""
        from dsl_worker.phases.base import PhaseStatus
        stats = self._get_current_stats()
        
        return PhaseStatus(
            phase_name=self.name,
            status="complete" if self.is_complete() else "active",
            progress=f"{stats.get('total_seeds', 0)} seeds, {len(self._sources)} sources"
        )
    
    async def cleanup(self):
        """Cleanup resources."""
        if self._browser_started:
            await self.browser_pool.stop()
            self._browser_started = False
        
        if self._sandbox:
            self._sandbox.close()
            self._sandbox = None