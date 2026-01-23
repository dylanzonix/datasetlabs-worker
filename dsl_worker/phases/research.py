"""
Phase: Research (Intelligent Seed Collection)

Replaces: seed_extraction, seed_scoring, seed_assignment

A meta-agent that:
1. Explores sources via brave search (breadth)
2. Dispatches autonomous browser agents (depth)
3. Extracts seeds via code execution
4. Synthesizes assignment seeds when needed

Seeds are minimal: {text, note, source_url}
Generation phase handles diversity assignment.
"""

import asyncio
import json
import logging
import os
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple

import httpx

from dsl_worker.phases.base import Phase, PhaseResult
from dsl_worker.phases.browser_pool import BrowserPool, BrowseResult

logger = logging.getLogger(__name__)

# Models
RESEARCH_MODEL = os.getenv("RESEARCH_MODEL", "gpt-4.1")
SUMMARIZE_MODEL = os.getenv("SUMMARIZE_MODEL", "gpt-4.1-nano")


@dataclass
class Seed:
    """Minimal seed structure."""
    id: uuid.UUID
    text: Optional[str]      # None for synthetic/assignment seeds
    note: str                # Context/instructions for generation
    source_url: Optional[str] = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class SearchResult:
    """Result from brave search."""
    url: str
    title: str
    snippet: str
    summary: Optional[str] = None  # Populated after fetch + summarize


class ResearchPhase(Phase):
    """
    Intelligent research agent that explores and extracts seeds.
    
    Maintains a long-running conversation with accumulated understanding.
    Uses tools to search, fetch, browse, and extract.
    """
    
    def __init__(
        self,
        *args,
        browser_pool_size: int = None,  # Default from env or 1
        target_seed_multiplier: float = 1.2,  # Collect 20% extra
        stop_checker: Optional[Callable[[], bool]] = None,
        cost_tracker: Optional[Any] = None,
        blob_service_client: Optional[Any] = None,  # Azure blob client
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        
        # Browser pool size: from param, env, or default 1
        if browser_pool_size is None:
            browser_pool_size = int(os.getenv("BROWSER_POOL_SIZE", "3"))
        
        self.target_seed_multiplier = target_seed_multiplier
        self.stop_checker = stop_checker
        self.cost_tracker = cost_tracker
        self.blob_service_client = blob_service_client
        
        # Browser pool
        self.browser_pool = BrowserPool(
            size=browser_pool_size,
            profiles_dir="./browser_profiles",
            headless=os.getenv("BROWSER_HEADLESS", "").lower() in ("true", "1", "yes"),
        )
        self._browser_started = False
        
        # Seed pool
        self.seed_pool: List[Seed] = []
        self._seed_lock = asyncio.Lock()
        
        # Research state (accumulated understanding)
        self.research_messages: List[Dict[str, str]] = []
        self.sources_explored: List[str] = []
        self.research_notes: str = ""
        
        # API clients
        self.brave_api_key = os.getenv("BRAVE_API_KEY")
        if self.brave_api_key:
            logger.info(f"[{self.name}] BRAVE_API_KEY is set (length: {len(self.brave_api_key)})")
        else:
            logger.warning(f"[{self.name}] ⚠️ BRAVE_API_KEY is NOT set - web search will fail!")
        
        # Cost tracking
        self._total_cost = 0.0
        
        # Turn counter for debugging
        self._turn_count = 0
        
        # Local file storage for uploaded and downloaded files
        import tempfile
        self._files_dir = tempfile.mkdtemp(prefix="research_files_")
        self._uploaded_files: Dict[str, str] = {}  # filename -> local_path
        self._downloaded_files: Dict[str, str] = {}  # url -> local_path
        self._files_loaded = False
        logger.info(f"[{self.name}] Files directory: {self._files_dir}")
        
    @property
    def target_seeds(self) -> int:
        """Target number of seeds to collect."""
        return int(self.state.num_samples * self.target_seed_multiplier)
        
    def should_run(self) -> bool:
        """Run if we haven't collected enough seeds."""
        return len(self.seed_pool) < self.target_seeds
    
    def _check_should_stop(self) -> bool:
        """Check if we should stop/pause. Call this at granular points."""
        if self.stop_checker and self.stop_checker():
            logger.info(f"[{self.name}] ⏸️ Pause/stop requested")
            return True
        return False
        
    async def execute_once(self) -> PhaseResult:
        """One iteration of the research loop."""
        
        self._turn_count += 1
        logger.info(f"")
        logger.info(f"{'='*60}")
        logger.info(f"[{self.name}] 🔄 TURN {self._turn_count} | Seeds: {len(self.seed_pool)}/{self.target_seeds}")
        logger.info(f"{'='*60}")
        
        # Initialize browser pool on first run
        if not self._browser_started:
            await self.browser_pool.start()
            self._browser_started = True
            
        # Check for stop
        if self._check_should_stop():
            return PhaseResult.no_work()
            
        # Build context for meta-agent
        logger.info(f"[{self.name}] 📋 Building context for LLM...")
        context = self._build_meta_context()
        
        # Check again before LLM call
        if self._check_should_stop():
            return PhaseResult.no_work()
        
        # Get next action from meta-agent
        logger.info(f"[{self.name}] 🤖 Asking LLM: What should we do next?")
        try:
            action, cost = await self._meta_decide(context)
            self._total_cost += cost
        except Exception as e:
            logger.error(f"Meta-agent decision failed: {e}")
            return PhaseResult.no_work()
        
        # Check after LLM call
        if self._check_should_stop():
            return PhaseResult.no_work()
        
        # Log the decision
        logger.info(f"[{self.name}] 📌 LLM decided: {action.get('type')} ")
            
        # Execute action
        logger.info(f"[{self.name}] ▶️ Executing action: {action.get('type')}")
        action_cost = await self._execute_action(action)
        self._total_cost += action_cost
        
        # Track cost
        if self.cost_tracker:
            self.cost_tracker.add_cost(
                phase=self.name,
                cost_usd=cost + action_cost,
                model=RESEARCH_MODEL,
            )
            
        logger.info(f"[{self.name}] ✅ Turn {self._turn_count} complete | Seeds: {len(self.seed_pool)}/{self.target_seeds}")
        
        return PhaseResult.work_done(cost_usd=cost + action_cost)
        
    def _build_meta_context(self) -> str:
        """Build context for meta-agent decision."""
        
        # Format schema
        schema_lines = []
        for col in self.state.columns or []:
            line = f"- {col.get('name')} ({col.get('type')})"
            if col.get('description'):
                line += f": {col['description']}"
            schema_lines.append(line)
        schema = "\n".join(schema_lines) if schema_lines else "No schema defined"
        
        # Format diversity spec
        diversity_lines = []
        for axis in self.state.diversity_spec or []:
            values = [v.get('value') for v in axis.get('values', [])]
            diversity_lines.append(f"- {axis.get('name')}: {', '.join(values)}")
        diversity = "\n".join(diversity_lines) if diversity_lines else "No diversity requirements"
        
        # Format uploaded files
        files = self.state.get_uploaded_files() if hasattr(self.state, 'get_uploaded_files') else []
        files_info = "\n".join([
            f"- {f.get('filename')} ({f.get('size_bytes', 0)} bytes)"
            for f in files
        ]) if files else "No files uploaded"
        
        # Recent research summary
        recent = self.research_messages[-10:] if self.research_messages else []
        recent_summary = "\n".join([
            f"[{m['role']}]: {m['content'][:200]}..."
            for m in recent
        ]) if recent else "No research yet"
        
        return f"""## Dataset Goal
{self.state.generation_prompt}

## Schema
{schema}

## Diversity Requirements  
{diversity}

## Current Progress
- Seeds collected: {len(self.seed_pool)} / {self.target_seeds}
- Sources explored: {len(self.sources_explored)}

## Uploaded Files
{files_info}

## Research Notes
{self.research_notes or "None yet"}

## Recent Activity
{recent_summary}
"""

    async def _meta_decide(self, context: str) -> Tuple[Dict, float]:
        """Get next action from meta-agent."""
        
        system_prompt = self._build_system_prompt()
        
        messages = [
            {"role": "system", "content": system_prompt},
        ]
        
        # Add conversation history
        for msg in self.research_messages[-20:]:  # Keep last 20 turns
            messages.append(msg)
            
        # Add current context
        messages.append({
            "role": "user",
            "content": f"Current state:\n\n{context}\n\nWhat should we do next?"
        })
        
        logger.debug(f"[{self.name}] Sending {len(messages)} messages to LLM")
        logger.debug(f"[{self.name}] Context preview: {context[:300]}...")
        
        # Log message structure for debugging
        logger.info(f"[{self.name}] 📤 Message history ({len(self.research_messages)} prior messages):")
        for i, msg in enumerate(self.research_messages[-5:]):  # Last 5
            role = msg.get("role", "?")
            content = msg.get("content", "")[:100]
            logger.info(f"[{self.name}]   [{i}] {role}: {content}...")
        
        # Call LLM
        response, cost = await self.openai_client.responses_create(
            model=RESEARCH_MODEL,
            input=messages,
            tools=self._get_tools(),
        )
        
        # Log raw response for debugging
        logger.info(f"[{self.name}] 📥 Raw response output_text: {response.output_text[:300] if response.output_text else 'None'}...")
        logger.info(f"[{self.name}] 📥 Response has {len(response.output)} output items")
        for item in response.output:
            logger.info(f"[{self.name}]   - type={item.type}, {'name=' + item.name if hasattr(item, 'name') else ''}")
        
        # Parse response
        action = self._parse_action(response)
        
        # Log what we got
        if action["type"] == "message":
            logger.warning(f"[{self.name}] ⚠️ LLM did not call a tool, returned text instead")
        else:
            logger.info(f"[{self.name}] ✅ LLM called tool: {action['type']} with args: {json.dumps(action.get('args', {}))[:200]}")
        
        # Update conversation history
        self.research_messages.append({
            "role": "user",
            "content": f"State update:\n{context[:500]}..."
        })
        self.research_messages.append({
            "role": "assistant",
            "content": json.dumps(action)
        })
        
        return action, cost.total_cost_usd
        
    def _build_system_prompt(self) -> str:
        """Build system prompt for meta-agent."""
        return """You are a research agent collecting seeds for dataset generation.

## Your Job
Find content and collect it as seeds. Seeds become rows in the final dataset.

## Two Approaches for Seed Collection

### Approach 1: Simple Pages (fetch_page + add_seeds)
For static pages where content is immediately visible:
1. fetch_page(url) → get content
2. Read the content, identify valuable items
3. add_seeds([...]) → save the seeds you found

Use this for: articles, documentation, simple listings, static content

### Approach 2: Interactive Sites (start_browse)
For sites requiring clicks, scrolling, navigation, or JS interaction:
1. start_browse(url, goal) → browser agent explores AND collects seeds
2. Seeds are automatically extracted from agent's exploration

Use this for: forums with pagination, sites with popups, JS-heavy apps, multi-page content

**The browser agent IS the seed collector for interactive sites.** You don't need to call add_seeds after - the agent does it during exploration.

## What Makes a Good Seed
- COMPLETE: Enough info to generate a full dataset row
- CONTIGUOUS: From one section/post/item (not pieced together)
- STANDALONE: Makes sense without other context

## Tools

**brave_search(queries)**: Find sources. Start here.

**fetch_page(url)**: Get static page/document content. Works for HTML, PDF, DOCX, XLSX.
→ After this, YOU must call add_seeds() with extracted content.

**add_seeds(seeds)**: Save seeds from fetch_page content.
```json
{"seeds": [{"text": "Content...", "note": "Category", "source_url": "..."}]}
```

**start_browse(url, goal)**: Launch browser agent for interactive sites.
→ Browser agent collects seeds during exploration. No add_seeds needed after.
→ Goal should describe WHAT to find, agent handles the HOW.

**code_exec(script)**: Parse complex structured content with Python.

## Decision Guide
- Static page, content visible immediately? → fetch_page + add_seeds
- Need to click, scroll, navigate, handle popups? → start_browse
- Complex parsing needed? → fetch_page + code_exec + add_seeds
"""

    def _get_tools(self) -> List[Dict]:
        """Define tools for meta-agent."""
        return [
            {
                "type": "function",
                "name": "brave_search",
                "description": "Search the web. Returns summaries of top results.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "queries": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Search queries (will be run sequentially)"
                        },
                        "results_per_query": {
                            "type": "integer",
                            "default": 5,
                            "description": "Results per query"
                        }
                    },
                    "required": ["queries"]
                }
            },
            {
                "type": "function",
                "name": "fetch_page",
                "description": "Fetch content from URL. Works for HTML pages, PDFs, DOCX, XLSX. Returns text/markdown.",
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
                "name": "start_browse",
                "description": "Launch browser agent for interactive tasks (clicking, scrolling, forms, popups). Goal should be focused on extraction. Do NOT use for external search engines.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "url": {"type": "string", "description": "Starting URL"},
                        "goal": {"type": "string", "description": "Focused task: what to extract or interact with"}
                    },
                    "required": ["url", "goal"]
                }
            },
            {
                "type": "function",
                "name": "browse_action",
                "description": "Continue browser session with instruction.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "instruction": {"type": "string", "description": "What to do next"}
                    },
                    "required": ["instruction"]
                }
            },
            {
                "type": "function",
                "name": "end_browse",
                "description": "End browser session.",
                "parameters": {"type": "object", "properties": {}}
            },
            {
                "type": "function",
                "name": "code_exec",
                "description": """Execute Python code in a secure sandbox.

Available variables:
- page_markdown: Last fetched page content
- page_url: URL of last fetched page

Available functions:
- add_seeds(seeds): Add seeds to collection (REQUIRED to save work)
- read_file(filename): Read uploaded file from sandbox
- list_files(): List available uploaded files

Available libraries: pandas, BeautifulSoup, openpyxl, re, json, os

Example:
```python
# Parse fetched page
from bs4 import BeautifulSoup
soup = BeautifulSoup(page_markdown, 'html.parser')
items = soup.find_all('div', class_='item')
seeds = [{'text': item.text, 'source_url': page_url} for item in items]
add_seeds(seeds)
```""",
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
                "name": "add_seeds",
                "description": """CRITICAL: This is how you save your work. After fetching a page, extract the valuable content and call this tool.

Each seed becomes one row in the final dataset. Seeds should be:
- Complete: Contains enough info to generate a full dataset row
- Contiguous: From one section/item, not pieced together
- Standalone: Makes sense without external context

You MUST call this after finding good content. Just reading pages does nothing - you must explicitly add seeds.""",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "seeds": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "text": {"type": "string", "description": "The actual content (a paragraph, post, item, etc.)"},
                                    "note": {"type": "string", "description": "Category, context, or instructions for generation"},
                                    "source_url": {"type": "string", "description": "Where this came from"}
                                },
                                "required": ["text"]
                            },
                            "description": "List of seeds to add"
                        }
                    },
                    "required": ["seeds"]
                }
            },
            {
                "type": "function",
                "name": "update_notes",
                "description": "Update your research notes (persistent across iterations).",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "notes": {"type": "string", "description": "Updated research notes"}
                    },
                    "required": ["notes"]
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
        
        # No tool call - treat as thinking/message
        return {"type": "message", "content": response.output_text}
        
    async def _execute_action(self, action: Dict) -> float:
        """Execute an action and return cost."""
        action_type = action.get("type")
        args = action.get("args", {})
        
        logger.info(f"[{self.name}] Action: {action_type}")
        
        if action_type == "brave_search":
            return await self._do_brave_search(args)
        elif action_type == "fetch_page":
            return await self._do_fetch_page(args)
        elif action_type == "start_browse":
            return await self._do_start_browse(args)
        elif action_type == "browse_action":
            return await self._do_browse_action(args)
        elif action_type == "end_browse":
            return await self._do_end_browse()
        elif action_type == "code_exec":
            return await self._do_code_exec(args)
        elif action_type == "add_seeds":
            return await self._do_add_seeds(args)
        elif action_type == "update_notes":
            self.research_notes = args.get("notes", "")
            return 0.0
        elif action_type == "done":
            logger.info(f"[{self.name}] Research complete: {args.get('reason')}")
            return 0.0
        elif action_type == "message":
            # LLM responded with text instead of tool call - log it
            content = action.get("content", "")
            logger.warning(f"[{self.name}] LLM returned message instead of tool call:")
            logger.warning(f"[{self.name}] >>> {content[:500]}{'...' if len(content) > 500 else ''}")
            return 0.0
        else:
            logger.warning(f"Unknown action type: {action_type}")
            return 0.0

    # =========================================================================
    # Tool implementations
    # =========================================================================
    
    async def _do_brave_search(self, args: Dict) -> float:
        """Execute brave search with concurrent queries."""
        queries = args.get("queries", [])
        results_per_query = args.get("results_per_query", 5)
        
        logger.info(f"[{self.name}] 🔍 Brave search with {len(queries)} queries: {queries}")
        
        if not self.brave_api_key:
            logger.error(f"[{self.name}] ❌ BRAVE_API_KEY not set!")
            self._add_message("assistant", "Error: Brave API key not configured")
            return 0.0
        
        # Rate limit: Brave free tier is 1 req/sec, paid is higher but still limited
        # Serialize queries with delay to avoid 429s
        all_results: List[List[SearchResult]] = []
        
        async def search_one(query: str) -> List[SearchResult]:
            try:
                async with httpx.AsyncClient() as client:
                    response = await client.get(
                        "https://api.search.brave.com/res/v1/web/search",
                        params={"q": query, "count": results_per_query},
                        headers={"X-Subscription-Token": self.brave_api_key},
                        timeout=30.0,
                    )
                    response.raise_for_status()
                    data = response.json()
                    
                results = []
                for r in data.get("web", {}).get("results", []):
                    results.append(SearchResult(
                        url=r.get("url", ""),
                        title=r.get("title", ""),
                        snippet=r.get("description", ""),
                    ))
                logger.info(f"[{self.name}] Query '{query[:50]}...' returned {len(results)} results")
                return results
            except Exception as e:
                logger.error(f"[{self.name}] Search failed for '{query[:50]}...': {e}")
                return []
        
        # Execute queries sequentially with delay to avoid rate limiting
        for i, query in enumerate(queries):
            # Check for pause between queries
            if self._check_should_stop():
                logger.info(f"[{self.name}] ⏸️ Pausing during search (after {i} queries)")
                break
                
            if i > 0:
                await asyncio.sleep(1.1)  # Brave rate limit: ~1 req/sec
            results = await search_one(query)
            all_results.append(results)
        
        # Flatten and dedupe by URL
        seen_urls = set()
        unique_results = []
        for results in all_results:
            for r in results:
                if r.url not in seen_urls:
                    seen_urls.add(r.url)
                    unique_results.append(r)
        
        logger.info(f"[{self.name}] 🔍 Total unique results: {len(unique_results)}")
        for r in unique_results[:5]:  # Log first 5
            logger.info(f"[{self.name}]   - {r.title[:60]}... ({r.url[:50]}...)")
        
        if not unique_results:
            self._add_message("assistant", "Search returned no results. Try different queries.")
            return 0.0
        
        # Check before fetching
        if self._check_should_stop():
            self._add_message("assistant", f"Found {len(unique_results)} results but pausing before fetch.")
            return 0.0
                    
        # Fetch and summarize results (with concurrency limit)
        logger.info(f"[{self.name}] 📄 Fetching and summarizing {len(unique_results)} pages...")
        cost = await self._fetch_and_summarize_results(unique_results)
        
        # Format for conversation
        result_text = f"Found {len(unique_results)} unique results:\n\n"
        for r in unique_results:
            result_text += f"**{r.title}**\n{r.url}\n{r.summary or r.snippet}\n\n"
            
        self._add_message("assistant", result_text)
        self.sources_explored.extend([r.url for r in unique_results])
        
        logger.info(f"[{self.name}] ✅ Search complete, added results to context")
        
        return cost
        
    async def _fetch_and_summarize_results(self, results: List[SearchResult]) -> float:
        """Fetch pages and summarize with maximum concurrency.
        
        Fetches and summarizations run independently - we don't wait for 
        summarization before starting the next fetch.
        """
        
        fetch_semaphore = asyncio.Semaphore(5)  # Max concurrent fetches
        summarize_semaphore = asyncio.Semaphore(5)  # Max concurrent LLM calls
        
        stop_requested = False
        costs = []
        summarize_tasks = []
        
        async def summarize_one(result: SearchResult, markdown: str) -> float:
            """Summarize a single page."""
            nonlocal stop_requested
            
            if stop_requested or self._check_should_stop():
                stop_requested = True
                return 0.0
                
            async with summarize_semaphore:
                if stop_requested or self._check_should_stop():
                    stop_requested = True
                    return 0.0
                    
                try:
                    logger.info(f"[{self.name}] 🤖 Summarizing: {result.url[:50]}... ({len(markdown)} chars)")
                    summary, cost = await self._summarize_page(markdown, result.title)
                    result.summary = summary
                    logger.info(f"[{self.name}] ✅ Summary: {summary[:80]}...")
                    return cost if isinstance(cost, (int, float)) else cost.total_cost_usd
                except Exception as e:
                    logger.warning(f"[{self.name}] ❌ Summarize failed for {result.url[:50]}: {e}")
                    result.summary = result.snippet
                    return 0.0
        
        async def fetch_one(idx: int, result: SearchResult):
            """Fetch a single page and spawn summarization task."""
            nonlocal stop_requested
            
            if stop_requested or self._check_should_stop():
                stop_requested = True
                return
            
            async with fetch_semaphore:
                if stop_requested or self._check_should_stop():
                    stop_requested = True
                    return
                    
                try:
                    logger.info(f"[{self.name}] 📥 Fetching [{idx+1}/{len(results)}]: {result.url[:70]}...")
                    markdown = await self.browser_pool.simple_fetch(result.url)
                    
                    # Don't wait for summarization - fire it off and continue
                    if markdown and len(markdown) > 50:
                        task = asyncio.create_task(summarize_one(result, markdown))
                        summarize_tasks.append(task)
                    else:
                        result.summary = result.snippet
                        
                except Exception as e:
                    logger.warning(f"[{self.name}] ❌ Fetch failed for {result.url[:50]}: {e}")
                    result.summary = result.snippet
        
        # Fire off all fetches concurrently
        await asyncio.gather(*[fetch_one(i, r) for i, r in enumerate(results)])
        
        # Wait for any remaining summarizations to complete
        if summarize_tasks:
            logger.info(f"[{self.name}] ⏳ Waiting for {len(summarize_tasks)} summarizations to complete...")
            summarize_costs = await asyncio.gather(*summarize_tasks)
            costs.extend(summarize_costs)
        
        if stop_requested:
            logger.info(f"[{self.name}] ⏸️ Fetch/summarize interrupted by pause")
        
        return sum(costs)
        
    async def _summarize_page(self, markdown: str, title: str) -> Tuple[str, float]:
        """Summarize page content with small model."""
        
        # Truncate if too long
        if len(markdown) > 15000:
            markdown = markdown[:15000] + "\n\n[truncated]"
            
        prompt = f"""Summarize this webpage in 2-3 sentences. Focus on what content/data is available.

Title: {title}

Content:
{markdown}

Summary:"""
        
        response, cost = await self.openai_client.responses_create(
            model=SUMMARIZE_MODEL,
            input=[{"role": "user", "content": prompt}],
        )
        
        return response.output_text.strip(), cost.total_cost_usd
        
    async def _do_fetch_page(self, args: Dict) -> float:
        """Fetch a single page or download a file."""
        url = args.get("url")
        
        # Check for pause before fetching
        if self._check_should_stop():
            self._add_message("assistant", f"Fetch of {url} skipped - pausing.")
            return 0.0
        
        try:
            # Browser pool handles both HTML pages and file downloads
            content = await self.browser_pool.simple_fetch(url)
            
            # Check if we got meaningful content
            if not content or len(content.strip()) < 50:
                self._add_message("assistant", f"Fetched {url} but got minimal content. Page may require JS or login.")
                return 0.0
            
            # Truncate for conversation
            preview = content[:5000] + "\n\n[truncated]" if len(content) > 5000 else content
            self._add_message("assistant", f"Fetched {url}:\n\n{preview}")
            
            # Store full content for code_exec
            self._current_page_markdown = content
            self._current_page_url = url
            
        except Exception as e:
            self._add_message("assistant", f"Failed to fetch {url}: {e}")
            
        return 0.0  # No LLM cost for fetch
        
    async def _do_start_browse(self, args: Dict) -> float:
        """Start autonomous browser agent - PRIMARY seed collection method for interactive sites."""
        url = args.get("url")
        goal = args.get("goal")
        
        # Simplified goal - the browser agent has add_seeds() tool injected
        agent_goal = f"""## Your Task
{goal}

## Collecting Seeds
You have an `add_seeds` tool available. Use it to save valuable content you find.

A "seed" is content that becomes ONE row in a dataset. Good seeds are:
- COMPLETE: A full post, listing, section, or item with all its details
- CONTIGUOUS: From one place, not pieced from different parts
- STANDALONE: Makes sense on its own

## How to Use add_seeds
When you find good content, call:
```
add_seeds([
    {{"text": "The complete content...", "note": "Category or context"}},
    {{"text": "Another item...", "note": "Another note"}}
])
```

## Rules
- Call add_seeds() as you find content - don't wait until the end
- Each seed should be substantial (not just titles or fragments)
- Do NOT use external search engines (Google, DuckDuckGo, etc.)
- When done exploring, call done()
"""
        
        logger.info(f"[{self.name}] 🤖 Starting browser agent")
        logger.info(f"[{self.name}] 📋 URL: {url}")
        logger.info(f"[{self.name}] 📋 Goal: {goal[:100]}...")
        
        # Check before starting browser agent
        if self._check_should_stop():
            self._add_message("assistant", "Browser agent not started - pausing.")
            return 0.0
        
        # Use autonomous browse (add_seeds tool is injected by browser_pool)
        llm = self._get_browse_llm()
        result = await self.browser_pool.autonomous_browse(
            url=url,
            goal=agent_goal,
            llm=llm,
            max_steps=30,
            stop_checker=self.stop_checker,  # Pass stop checker for fine-grained pausing
        )
        
        # Store page state
        self._current_page_markdown = result.page_markdown
        self._current_page_url = result.final_url
        
        # extracted_data contains seeds collected via add_seeds() tool
        seeds_added = 0
        if result.extracted_data:
            # extracted_data is already a list of seed dicts from the add_seeds tool
            seeds_added = await self._add_seeds_from_list(result.extracted_data, url)
        
        # Report back
        status = "succeeded" if result.success else "failed"
        
        self._add_message(
            "assistant",
            f"Browser agent {status}.\n"
            f"Final URL: {result.final_url}\n"
            f"Seeds collected: {seeds_added}"
        )
        
        if result.error:
            self._add_message("assistant", f"Agent error: {result.error}")
            
        logger.info(f"[{self.name}] 🤖 Agent done. Status: {status}, Seeds: {seeds_added}")
            
        return 0.0
    
    async def _add_seeds_from_list(self, seed_list: List[Dict], default_url: str) -> int:
        """Convert seed dicts to Seed objects and add to pool."""
        seeds = []
        for s in seed_list:
            if isinstance(s, dict):
                text = s.get('text', '')
                if text and len(str(text).strip()) > 20:
                    seeds.append(Seed(
                        id=uuid.uuid4(),
                        text=str(text)[:5000],
                        note=s.get('note', ''),
                        source_url=s.get('source_url') or default_url,
                    ))
        
        if seeds:
            async with self._seed_lock:
                self.seed_pool.extend(seeds)
            logger.info(f"[{self.name}] 🌱 Added {len(seeds)} seeds from browser agent")
        
        return len(seeds)
    
    async def _parse_agent_output_to_seeds(self, extracted_data: Any, source_url: str) -> int:
        """Parse agent's extracted content and add as seeds."""
        import uuid
        
        seeds = []
        
        logger.info(f"[{self.name}] Parsing agent output: {type(extracted_data)}")
        logger.info(f"[{self.name}] Output preview: {str(extracted_data)[:500]}...")
        
        # Try to parse as our expected JSON format first
        parsed_seeds = self._try_parse_seed_json(extracted_data)
        if parsed_seeds:
            for s in parsed_seeds:
                text = s.get('text', '')
                if text and len(text.strip()) > 20:
                    seeds.append(Seed(
                        id=uuid.uuid4(),
                        text=text[:5000],
                        note=s.get('note', 'From browser agent'),
                        source_url=s.get('source_url', source_url),
                    ))
            logger.info(f"[{self.name}] Parsed {len(seeds)} seeds from JSON format")
        else:
            # Fallback: try to extract from generic formats
            seeds = self._parse_generic_output(extracted_data, source_url)
        
        # Add seeds to pool
        if seeds:
            async with self._seed_lock:
                self.seed_pool.extend(seeds)
            logger.info(f"[{self.name}] 🌱 Added {len(seeds)} seeds from browser agent")
        else:
            logger.warning(f"[{self.name}] ⚠️ Browser agent returned no parseable seeds")
        
        return len(seeds)
    
    def _try_parse_seed_json(self, data: Any) -> Optional[List[Dict]]:
        """Try to parse data as our expected seed JSON format."""
        import json
        
        # If it's a string, try to parse as JSON
        if isinstance(data, str):
            # Try to find JSON in the string
            try:
                # Direct parse
                parsed = json.loads(data)
                if isinstance(parsed, dict) and 'seeds' in parsed:
                    return parsed['seeds']
                if isinstance(parsed, list):
                    return parsed
            except json.JSONDecodeError:
                pass
            
            # Try to extract JSON from text (agent might have wrapped it)
            import re
            json_match = re.search(r'\{[\s\S]*"seeds"[\s\S]*\}', data)
            if json_match:
                try:
                    parsed = json.loads(json_match.group())
                    if 'seeds' in parsed:
                        return parsed['seeds']
                except json.JSONDecodeError:
                    pass
            
            # Try to find JSON array
            array_match = re.search(r'\[[\s\S]*\]', data)
            if array_match:
                try:
                    parsed = json.loads(array_match.group())
                    if isinstance(parsed, list) and len(parsed) > 0:
                        # Check if items look like seeds
                        if isinstance(parsed[0], dict) and ('text' in parsed[0] or 'content' in parsed[0]):
                            return parsed
                except json.JSONDecodeError:
                    pass
        
        # If it's already a dict
        elif isinstance(data, dict):
            if 'seeds' in data:
                return data['seeds']
            # Single seed as dict
            if 'text' in data or 'content' in data:
                return [data]
        
        # If it's a list
        elif isinstance(data, list):
            if len(data) > 0 and isinstance(data[0], dict):
                return data
        
        return None
    
    def _parse_generic_output(self, extracted_data: Any, source_url: str) -> List[Seed]:
        """Fallback parser for non-JSON agent output."""
        import uuid
        seeds = []
        
        if isinstance(extracted_data, str):
            # Single text block - add as one seed if substantial
            if len(extracted_data.strip()) > 100:
                seeds.append(Seed(
                    id=uuid.uuid4(),
                    text=extracted_data[:5000],
                    note="Extracted by browser agent (raw)",
                    source_url=source_url,
                ))
        elif isinstance(extracted_data, list):
            for item in extracted_data:
                if isinstance(item, str) and len(item.strip()) > 50:
                    seeds.append(Seed(
                        id=uuid.uuid4(),
                        text=item[:5000],
                        note="Extracted by browser agent",
                        source_url=source_url,
                    ))
                elif isinstance(item, dict):
                    text = item.get('text') or item.get('content') or str(item)
                    if len(str(text).strip()) > 50:
                        seeds.append(Seed(
                            id=uuid.uuid4(),
                            text=str(text)[:5000],
                            note=item.get('note', 'Extracted by browser agent'),
                            source_url=source_url,
                        ))
        
        return seeds
        
    async def _do_browse_action(self, args: Dict) -> float:
        """Continue browser session."""
        # For now, treat as new browse from current URL
        instruction = args.get("instruction")
        
        if hasattr(self, '_current_page_url') and self._current_page_url:
            return await self._do_start_browse({
                "url": self._current_page_url,
                "goal": instruction
            })
        else:
            self._add_message("assistant", "No active browser session. Use start_browse first.")
            return 0.0
            
    async def _do_end_browse(self) -> float:
        """End browser session."""
        self._current_page_markdown = None
        self._current_page_url = None
        self._add_message("assistant", "Browser session ended.")
        return 0.0
        
    async def _do_code_exec(self, args: Dict) -> float:
        """Execute Python code in secure sandbox."""
        script = args.get("script", "")
        
        # Check for pause before code execution
        if self._check_should_stop():
            self._add_message("assistant", "Code execution skipped - pausing.")
            return 0.0
        
        logger.info(f"[{self.name}] 🐍 Executing code in sandbox...")
        logger.info(f"[{self.name}] Script preview: {script[:200]}...")
        
        # Initialize sandbox executor if needed
        if not hasattr(self, '_sandbox') or self._sandbox is None:
            from dsl_worker.phases.sandbox import SandboxExecutor
            self._sandbox = SandboxExecutor(
                use_pool=True,
                pool_size=2,
            )
        
        # Ensure uploaded files are loaded
        if not self._files_loaded:
            self._load_uploaded_files()
            self._files_loaded = True
        
        # Execute in sandbox
        result = self._sandbox.execute(
            script=script,
            page_markdown=getattr(self, '_current_page_markdown', None),
            page_url=getattr(self, '_current_page_url', None),
            uploaded_files=self._uploaded_files,
            timeout=120,
        )
        
        # Process results
        if result.success:
            # Convert seed dicts to Seed objects and add to pool
            if result.seeds:
                seeds = []
                for s in result.seeds:
                    seeds.append(Seed(
                        id=uuid.uuid4(),
                        text=s.get("text"),
                        note=s.get("note", ""),
                        source_url=s.get("source_url") or getattr(self, '_current_page_url', None),
                    ))
                
                async with self._seed_lock:
                    self.seed_pool.extend(seeds)
                
                logger.info(f"[{self.name}] 🌱 Code added {len(seeds)} seeds")
                self._add_message(
                    "assistant", 
                    f"Code executed successfully. Added {len(seeds)} seeds.\n\nOutput:\n{result.stdout[:1500]}"
                )
            else:
                self._add_message(
                    "assistant", 
                    f"Code executed successfully.\n\nOutput:\n{result.stdout[:2000]}"
                )
        else:
            logger.error(f"[{self.name}] ❌ Code execution failed: {result.error}")
            self._add_message(
                "assistant", 
                f"Code execution failed: {result.error}\n\nStderr:\n{result.stderr[:1000]}"
            )
            
        return 0.0
        
    async def _do_add_seeds(self, args: Dict) -> float:
        """Add seeds directly."""
        seeds_data = args.get("seeds", [])
        
        seeds = []
        for s in seeds_data:
            seeds.append(Seed(
                id=uuid.uuid4(),
                text=s.get("text"),
                note=s.get("note", ""),
                source_url=s.get("source_url"),
            ))
            
        async with self._seed_lock:
            self.seed_pool.extend(seeds)
            
        self._add_message("assistant", f"Added {len(seeds)} seeds. Total: {len(self.seed_pool)}")
        return 0.0

    # =========================================================================
    # Helpers
    # =========================================================================
    
    def _add_message(self, role: str, content: str):
        """Add message to conversation history."""
        self.research_messages.append({"role": role, "content": content})
        
    def _get_uploaded_files_dict(self) -> Dict[str, str]:
        """Get dict of uploaded files (filename -> local_path)."""
        # Download files from Azure on first call
        if not self._files_loaded:
            self._load_uploaded_files()
            self._files_loaded = True
        
        return self._uploaded_files
    
    def _load_uploaded_files(self):
        """Download uploaded files from Azure blob to local temp directory."""
        if not hasattr(self.state, 'get_uploaded_files'):
            return
            
        files = self.state.get_uploaded_files() or []
        if not files:
            logger.info(f"[{self.name}] No uploaded files to load")
            return
            
        logger.info(f"[{self.name}] Loading {len(files)} uploaded files from Azure...")
        
        for f in files:
            filename = f.get('filename')
            blob_path = f.get('blob_path')
            
            if not filename or not blob_path:
                continue
                
            local_path = os.path.join(self._files_dir, filename)
            
            try:
                if self.blob_service_client:
                    # Download from Azure blob
                    container_name = os.getenv("AZURE_STORAGE_CONTAINER", "uploads")
                    blob_client = self.blob_service_client.get_blob_client(
                        container=container_name,
                        blob=blob_path
                    )
                    
                    # Download to local file
                    with open(local_path, "wb") as download_file:
                        download_file.write(blob_client.download_blob().readall())
                    
                    self._uploaded_files[filename] = local_path
                    logger.info(f"[{self.name}] ✅ Downloaded: {filename} -> {local_path}")
                else:
                    # No blob client - store blob_path for reference
                    self._uploaded_files[filename] = f"azure://{blob_path}"
                    logger.warning(f"[{self.name}] ⚠️ No blob client, storing reference: {filename}")
                    
            except Exception as e:
                logger.error(f"[{self.name}] ❌ Failed to download {filename}: {e}")
                
    def _get_browse_llm(self):
        """Get LLM instance for browser agent."""
        # This would return a browser-use compatible LLM
        # For now, return a placeholder
        from browser_use import ChatOpenAI
        return ChatOpenAI(model="gpt-4.1-mini")

    # =========================================================================
    # Phase interface
    # =========================================================================
    
    def get_seeds(self, n: Optional[int] = None) -> List[Seed]:
        """Get seeds for generation phase."""
        if n is None:
            return self.seed_pool.copy()
        return self.seed_pool[:n]
        
    def get_seed_count(self) -> int:
        """Get current seed count."""
        return len(self.seed_pool)
        
    def is_complete(self) -> bool:
        """Complete when we have enough seeds."""
        return len(self.seed_pool) >= self.target_seeds
        
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
            progress=f"{len(self.seed_pool)}/{self.target_seeds} seeds"
        )
        
    async def cleanup(self):
        """Cleanup resources."""
        if self._browser_started:
            await self.browser_pool.stop()
            self._browser_started = False
        
        # Clean up sandbox executor
        if hasattr(self, '_sandbox') and self._sandbox:
            self._sandbox.close()
            self._sandbox = None
            logger.info(f"[{self.name}] Sandbox executor closed")
        
        # Clean up temp files directory
        if hasattr(self, '_files_dir') and self._files_dir:
            import shutil
            try:
                shutil.rmtree(self._files_dir)
                logger.info(f"[{self.name}] Cleaned up temp files: {self._files_dir}")
            except Exception as e:
                logger.warning(f"[{self.name}] Failed to clean up temp files: {e}")