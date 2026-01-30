"""
Research Tools - ChatGPT-style browsing for research agent.

Tools:
- brave_search(query, response_length) → search results artifact
- open(ref_id_or_url, start_line, response_length) → page viewport
- find(ref_id, pattern, response_length) → matching lines  
- click(ref_id, link_id, response_length) → new page viewport
- note(content) → add to notes
- conclude_research(summary) → transition to decision mode
- breakdown(children) → split scope
- extract_seeds(ref_id, line_ranges) → create assignments from source
- write_seeds(seeds) → create synthetic assignments
- interact(url_or_ref_id, task) → browser agent for complex interactions
"""

import asyncio
import json
import logging
import random
import string
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import httpx
from markdownify import markdownify as md

from dsl_worker.phases.artifacts import (
    ArtifactStore,
    SearchResults,
    SearchResult,
    PageView,
    PageLink,
    RESPONSE_LENGTHS,
    extract_links_from_markdown,
    format_viewport,
    format_links_table,
    format_search_results,
    find_in_lines,
)

logger = logging.getLogger(__name__)

PAGE_LOAD_WAIT = 2.0


def short_id(length: int = 6) -> str:
    """Generate short alphanumeric ID."""
    return ''.join(random.choices(string.ascii_lowercase + string.digits, k=length))


class ResearchState(Enum):
    """State machine for research process."""
    RESEARCHING = "researching"
    CONCLUDED = "concluded"


@dataclass
class ResearchScope:
    """Current research scope context."""
    id: str
    description: str
    quota: int
    depth: int = 0
    notes: List[str] = field(default_factory=list)
    parent_notes: List[str] = field(default_factory=list)


class ResearchTools:
    """
    Tools for research agent to explore and understand a domain.
    
    State machine:
    - RESEARCHING: Can search, browse, take notes
    - CONCLUDED: Can breakdown or create seeds
    
    Must call conclude_research() before breakdown/seeding.
    """
    
    def __init__(
        self,
        workspace_dir: Path,
        schema: List[Dict],
        brave_api_key: Optional[str] = None,
        browser: Optional[Any] = None,
        openai_client: Optional[Any] = None,
        model: str = "gpt-4o",
        stop_checker: Optional[Callable[[], bool]] = None,
    ):
        self.workspace_dir = Path(workspace_dir)
        self.schema = schema
        self.brave_api_key = brave_api_key
        self.browser = browser
        self.openai_client = openai_client
        self.model = model
        self.stop_checker = stop_checker
        
        # Artifact storage
        self.artifacts = ArtifactStore()
        
        # State machine
        self.state = ResearchState.RESEARCHING
        self.research_summary: Optional[str] = None
        
        # Current scope
        self.scope: Optional[ResearchScope] = None
        
        # Track breakdown
        self.breakdown_children: Optional[List[Dict]] = None
        
        # Track created assignments
        self.assignment_dirs: List[str] = []
        
        # Ensure workspace
        (self.workspace_dir / "assignments").mkdir(parents=True, exist_ok=True)
    
    def set_scope(self, scope: ResearchScope):
        """Set current scope being researched."""
        self.scope = scope
        self.breakdown_children = None
        self.state = ResearchState.RESEARCHING
        self.research_summary = None
    
    def _should_stop(self) -> bool:
        return self.stop_checker and self.stop_checker()
    
    def _get_config(self, response_length: str) -> Dict[str, int]:
        return RESPONSE_LENGTHS.get(response_length, RESPONSE_LENGTHS["medium"])
    
    def _require_state(self, required: ResearchState, action: str) -> Optional[str]:
        """Check state requirement, return error message if not met."""
        if self.state != required:
            if required == ResearchState.CONCLUDED:
                return f"Cannot {action} yet. You must call conclude_research() first to summarize what you learned."
            else:
                return f"Cannot {action} in {self.state.value} state."
        return None
    
    # =========================================================================
    # brave_search
    # =========================================================================
    
    async def brave_search(self, query: str, response_length: str = "medium") -> Tuple[str, float]:
        """Search the web using Brave Search API."""
        if not self.brave_api_key:
            return "Error: Brave API key not configured", 0.0
        
        config = self._get_config(response_length)
        count = config["results"]
        
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(
                    "https://api.search.brave.com/res/v1/web/search",
                    params={"q": query, "count": count},
                    headers={"X-Subscription-Token": self.brave_api_key},
                    timeout=30.0,
                )
                resp.raise_for_status()
                data = resp.json()
            
            results = []
            for i, r in enumerate(data.get("web", {}).get("results", [])):
                results.append(SearchResult(
                    id=i,
                    title=r.get("title", ""),
                    url=r.get("url", ""),
                    snippet=r.get("description", ""),
                    date=r.get("age"),
                ))
            
            artifact = SearchResults(query=query, results=results)
            ref_id = self.artifacts.store_search(artifact)
            
            return format_search_results(artifact, ref_id, count), 0.0
            
        except Exception as e:
            logger.error(f"[ResearchTools] brave_search failed: {e}")
            return f"Search error: {e}", 0.0
    
    # =========================================================================
    # open
    # =========================================================================
    
    async def open(
        self,
        ref_id_or_url: str,
        start_line: int = 0,
        response_length: str = "medium",
    ) -> Tuple[str, float]:
        """Open a URL or navigate within existing page artifact."""
        config = self._get_config(response_length)
        num_lines = config["lines"]
        
        artifact = self.artifacts.get(ref_id_or_url)
        
        if artifact:
            if isinstance(artifact, PageView):
                viewport = format_viewport(
                    artifact.lines, start_line, num_lines,
                    ref_id_or_url, artifact.url
                )
                links = format_links_table(artifact.links)
                return f"{viewport}\n{links}", 0.0
            
            elif isinstance(artifact, SearchResults):
                return format_search_results(artifact, ref_id_or_url, config["results"]), 0.0
            
            else:
                return f"Unknown artifact type for: {ref_id_or_url}", 0.0
        
        url = ref_id_or_url
        if not url.startswith(("http://", "https://")):
            url = "https://" + url
        
        try:
            markdown = await self._fetch_page(url)
            
            lines = markdown.split('\n')
            links = extract_links_from_markdown(markdown, url)
            
            page_view = PageView(
                url=url,
                fetched_at=datetime.now(timezone.utc).isoformat(),
                lines=lines,
                total_lines=len(lines),
                links=links,
            )
            ref_id = self.artifacts.store_page(page_view)
            
            viewport = format_viewport(lines, start_line, num_lines, ref_id, url)
            links_table = format_links_table(links)
            
            return f"{viewport}\n{links_table}", 0.0
            
        except Exception as e:
            logger.error(f"[ResearchTools] open failed for {url}: {e}")
            return f"Failed to open {url}: {e}", 0.0
    
    async def _fetch_page(self, url: str) -> str:
        """Fetch page content using Browser."""
        if not self.browser:
            raise RuntimeError("Browser not initialized")
        
        page = await self.browser.new_page(url)
        await asyncio.sleep(PAGE_LOAD_WAIT)
        
        try:
            html = await page.evaluate('() => document.body.innerHTML')
            markdown = md(html, heading_style='ATX', strip=['script', 'style'])
            return markdown
        finally:
            await self.browser.close_page(page)
    
    # =========================================================================
    # find
    # =========================================================================
    
    async def find(
        self,
        ref_id: str,
        pattern: str,
        response_length: str = "medium",
    ) -> Tuple[str, float]:
        """Find pattern in a page artifact."""
        page = self.artifacts.get_page(ref_id)
        if not page:
            return f"Page not found: {ref_id}", 0.0
        
        config = self._get_config(response_length)
        return find_in_lines(page.lines, pattern, config["matches"]), 0.0
    
    # =========================================================================
    # click
    # =========================================================================
    
    async def click(
        self,
        ref_id: str,
        link_id: int,
        response_length: str = "medium",
    ) -> Tuple[str, float]:
        """Click a link in a page artifact."""
        page = self.artifacts.get_page(ref_id)
        if not page:
            return f"Page not found: {ref_id}", 0.0
        
        link = None
        for l in page.links:
            if l.id == link_id:
                link = l
                break
        
        if not link:
            return f"Link {link_id} not found in {ref_id}. Available: 0-{len(page.links)-1}", 0.0
        
        return await self.open(link.href, start_line=0, response_length=response_length)
    
    # =========================================================================
    # note
    # =========================================================================
    
    def note(self, content: str) -> Tuple[str, float]:
        """Record a note about this domain."""
        if not self.scope:
            return "No active scope", 0.0
        
        self.scope.notes.append(content)
        return f"Noted ({len(self.scope.notes)} total)", 0.0
    
    # =========================================================================
    # conclude_research
    # =========================================================================
    
    def conclude_research(self, summary: str) -> Tuple[str, float]:
        """
        Conclude research phase and transition to decision mode.
        
        Must be called before breakdown() or write_seeds()/extract_seeds().
        """
        if not self.scope:
            return "No active scope", 0.0
        
        if not summary or len(summary.strip()) < 10:
            return "Please provide a meaningful summary of your research findings.", 0.0
        
        self.state = ResearchState.CONCLUDED
        self.research_summary = summary
        
        return (
            f"Research concluded. Summary recorded.\n\n"
            f"You can now either:\n"
            f"- breakdown(children) to split into smaller scopes\n"
            f"- extract_seeds(ref_id, line_ranges) to extract seeds from sources\n"
            f"- write_seeds(seeds) to write synthetic seeds"
        ), 0.0
    
    # =========================================================================
    # breakdown
    # =========================================================================
    
    def breakdown(self, children: List[Dict]) -> Tuple[str, float]:
        """Break scope into smaller sub-scopes."""
        error = self._require_state(ResearchState.CONCLUDED, "breakdown")
        if error:
            return error, 0.0
        
        if not self.scope:
            return "No active scope", 0.0
        
        if not children:
            return "No children provided", 0.0
        
        self.breakdown_children = children
        
        total_weight = sum(c.get("weight", 1.0) for c in children)
        lines = [f"Breaking into {len(children)} sub-scopes:"]
        
        for c in children:
            weight = c.get("weight", 1.0) / total_weight
            approx_quota = int(self.scope.quota * weight)
            lines.append(f"  - {c.get('description', '?')} (~{approx_quota} rows)")
        
        return '\n'.join(lines), 0.0
    
    # =========================================================================
    # extract_seeds
    # =========================================================================
    
    async def extract_seeds(
        self,
        ref_id: str,
        line_ranges: List[List[int]],
    ) -> Tuple[str, float]:
        """Extract seeds from page content by line ranges."""
        error = self._require_state(ResearchState.CONCLUDED, "extract seeds")
        if error:
            return error, 0.0
        
        page = self.artifacts.get_page(ref_id)
        if not page:
            return f"Page not found: {ref_id}", 0.0
        
        if not self.scope:
            return "No active scope", 0.0
        
        if not line_ranges:
            return "No line ranges provided", 0.0
        
        extracted = []
        for start, end in line_ranges:
            start = max(0, start)
            end = min(len(page.lines), end + 1)
            content = '\n'.join(page.lines[start:end]).strip()
            if content:
                extracted.append(content)
        
        if not extracted:
            return "No content extracted from ranges", 0.0
        
        return self._write_assignments(
            seeds=extracted,
            synthetic=False,
            source_ref=ref_id,
            source_url=page.url,
        ), 0.0
    
    # =========================================================================
    # write_seeds
    # =========================================================================
    
    def write_seeds(self, seeds: List[str]) -> Tuple[str, float]:
        """Write synthetic seeds."""
        error = self._require_state(ResearchState.CONCLUDED, "write seeds")
        if error:
            return error, 0.0
        
        if not self.scope:
            return "No active scope", 0.0
        
        if not seeds:
            return "No seeds provided", 0.0
        
        return self._write_assignments(
            seeds=seeds,
            synthetic=True,
            source_ref=None,
            source_url=None,
        ), 0.0
    
    def _write_assignments(
        self,
        seeds: List[str],
        synthetic: bool,
        source_ref: Optional[str],
        source_url: Optional[str],
    ) -> str:
        """Write assignment files for seeds."""
        dir_id = short_id()
        assignment_dir = self.workspace_dir / "assignments" / dir_id
        assignment_dir.mkdir(parents=True, exist_ok=True)
        
        all_notes = self.scope.parent_notes + self.scope.notes
        
        written = 0
        for i, seed in enumerate(seeds):
            if not seed.strip():
                continue
            
            assignment = {
                "scope_id": self.scope.id,
                "scope_description": self.scope.description,
                "seed": seed,
                "notes": all_notes,
                "research_summary": self.research_summary,
                "schema": self.schema,
                "synthetic": synthetic,
                "source_ref": source_ref,
                "source_url": source_url,
            }
            
            filepath = assignment_dir / f"{i:04d}.json"
            filepath.write_text(
                json.dumps(assignment, indent=2, ensure_ascii=False),
                encoding='utf-8'
            )
            written += 1
        
        self.assignment_dirs.append(str(assignment_dir))
        
        kind = "synthetic" if synthetic else "extracted"
        return f"Created {written} {kind} assignments in {dir_id}/"
    
    # =========================================================================
    # interact (Browser Agent)
    # =========================================================================
    
    async def interact(self, url_or_ref_id: str, task: str) -> Tuple[str, float]:
        """
        Use Browser Agent for complex interactions on a page.
        
        The browser agent performs actions (clicking, typing, navigating).
        It calls checkpoint() to report status and get instructions from you.
        You stay in control - the browser agent just executes actions.
        
        Args:
            url_or_ref_id: URL to start at, or ref_id of existing page
            task: Initial task description for the browser agent
        """
        if not self.browser:
            return "Browser not initialized", 0.0
        
        if not self.openai_client:
            return "OpenAI client not initialized for interact()", 0.0
        
        # Resolve URL
        url = url_or_ref_id
        page = self.artifacts.get_page(url_or_ref_id)
        if page:
            url = page.url
        elif not url.startswith(("http://", "https://")):
            url = "https://" + url
        
        try:
            from browser_use import Agent
            from browser_use.llm.openai.chat import ChatOpenAI
        except ImportError:
            return "browser-use not installed", 0.0
        
        # Create browser-use LLM
        browser_llm = ChatOpenAI(model=self.model)
        
        # Track total cost from browser agent
        total_cost = 0.0
        
        # Checkpoint state
        checkpoint_count = 0
        should_stop_session = False
        final_page_content = None
        final_url = url
        
        # Custom checkpoint tool for browser agent
        from browser_use import Tools
        tools = Tools()
        
        @tools.action(description="""Report your status and get instructions from the research coordinator.
Call this when you:
- Complete an action (logged in, clicked something, loaded page)
- Reach a decision point
- Need guidance on what to do next
- Encounter an obstacle

Describe what you did and what you see now.""")
        async def checkpoint(status: str, browser_session) -> str:
            nonlocal checkpoint_count, should_stop_session, final_page_content, final_url, total_cost
            
            checkpoint_count += 1
            
            if self._should_stop():
                should_stop_session = True
                return "STOP - Session ending. Call done() now."
            
            try:
                # Capture current page
                current_url = await browser_session.page.evaluate('() => window.location.href')
                html = await browser_session.page.evaluate('() => document.body.innerHTML')
                markdown = md(html, heading_style='ATX', strip=['script', 'style'])
                
                final_url = current_url
                final_page_content = markdown
                
                # Create page artifact for research agent to see
                lines = markdown.split('\n')
                links = extract_links_from_markdown(markdown, current_url)
                
                page_view = PageView(
                    url=current_url,
                    fetched_at=datetime.now(timezone.utc).isoformat(),
                    lines=lines,
                    total_lines=len(lines),
                    links=links,
                )
                ref_id = self.artifacts.store_page(page_view)
                
                # Format page content for research agent
                config = self._get_config("medium")
                viewport = format_viewport(lines, 0, config["lines"], ref_id, current_url)
                links_table = format_links_table(links)
                
                # Build prompt for research agent
                checkpoint_prompt = f"""Browser agent checkpoint #{checkpoint_count}

Status from browser agent: {status}

Current page:
{viewport}
{links_table}

What should the browser agent do next? 
- Give a brief instruction for the next action
- Or say "done" to end the browser session and continue with this page"""

                # Call research agent LLM for instructions
                messages = [
                    {"role": "system", "content": "You are coordinating a browser agent. Give brief, clear instructions for the next action, or say 'done' to end the session."},
                    {"role": "user", "content": checkpoint_prompt}
                ]
                
                response, cost = await self.openai_client.responses_create(
                    model=self.model,
                    input=messages,
                    max_output_tokens=500,
                )
                total_cost += cost.total_cost_usd
                
                # Extract response text
                instructions = ""
                for item in response.output:
                    if item.type == "message":
                        for content in item.content:
                            if hasattr(content, 'text'):
                                instructions += content.text
                
                instructions = instructions.strip()
                
                logger.info(f"[interact] Checkpoint {checkpoint_count}: {status[:50]}...")
                logger.info(f"[interact] Instructions: {instructions[:100]}...")
                
                # Check if research agent wants to end
                if any(phrase in instructions.lower() for phrase in ["done", "end session", "that's enough", "stop"]):
                    should_stop_session = True
                    return f"{instructions}\n\nCall done() to end the browser session."
                
                return instructions
                
            except Exception as e:
                logger.error(f"[interact] Checkpoint failed: {e}")
                return f"Checkpoint error: {e}. Continue with your best judgment or call done()."
        
        # Run browser agent
        try:
            agent_task = f"""Navigate to {url} and: {task}

IMPORTANT: Call checkpoint() frequently to report status and get instructions.
- After completing any action
- When you see new content
- When you're unsure what to do

The research coordinator will guide you through checkpoint responses."""

            agent = Agent(
                task=agent_task,
                browser=self.browser,
                llm=browser_llm,
                tools=tools,
                calculate_cost=True,
            )
            
            history = await agent.run(max_steps=30)
            
            # Get browser agent cost
            if history.usage:
                total_cost += history.usage.total_cost
            
            # Create final artifact if we have page content
            final_ref_id = None
            if final_page_content:
                lines = final_page_content.split('\n')
                links = extract_links_from_markdown(final_page_content, final_url)
                
                page_view = PageView(
                    url=final_url,
                    fetched_at=datetime.now(timezone.utc).isoformat(),
                    lines=lines,
                    total_lines=len(lines),
                    links=links,
                )
                final_ref_id = self.artifacts.store_page(page_view)
            
            success = history.is_successful() if hasattr(history, 'is_successful') else True
            
            result = f"Browser session completed ({checkpoint_count} checkpoints)"
            if final_ref_id:
                result += f"\nFinal page stored as {final_ref_id} - use open({final_ref_id}) to explore"
            
            return result, total_cost
            
        except Exception as e:
            logger.error(f"[interact] Browser agent failed: {e}")
            return f"Browser agent error: {e}", total_cost
    
    # =========================================================================
    # Tool Definitions for LLM
    # =========================================================================
    
    def get_tool_definitions(self) -> List[Dict]:
        """Get Responses API format tool definitions."""
        return [
            {
                "type": "function",
                "name": "brave_search",
                "description": "Search the web. Returns results with URLs you can open.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query"
                        },
                        "response_length": {
                            "type": "string",
                            "enum": ["short", "medium", "long"],
                            "description": "short=5, medium=10, long=20 results"
                        },
                    },
                    "required": ["query"]
                }
            },
            {
                "type": "function",
                "name": "open",
                "description": "Open a URL or view lines from existing page. Returns line-numbered content.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "ref_id_or_url": {
                            "type": "string",
                            "description": "URL to fetch, or ref_id (p0, p1...) to view existing"
                        },
                        "start_line": {
                            "type": "integer",
                            "description": "Line to start from (default: 0)"
                        },
                        "response_length": {
                            "type": "string",
                            "enum": ["short", "medium", "long"],
                            "description": "short=60, medium=150, long=300 lines"
                        },
                    },
                    "required": ["ref_id_or_url"]
                }
            },
            {
                "type": "function",
                "name": "find",
                "description": "Find text/pattern in a page. Returns matching lines with numbers.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "ref_id": {
                            "type": "string",
                            "description": "Page ref_id (p0, p1...)"
                        },
                        "pattern": {
                            "type": "string",
                            "description": "Text or regex to find"
                        },
                        "response_length": {
                            "type": "string",
                            "enum": ["short", "medium", "long"],
                            "description": "Controls max matches returned"
                        },
                    },
                    "required": ["ref_id", "pattern"]
                }
            },
            {
                "type": "function",
                "name": "click",
                "description": "Click a link in a page. Opens the linked URL.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "ref_id": {
                            "type": "string",
                            "description": "Page ref_id"
                        },
                        "link_id": {
                            "type": "integer",
                            "description": "Link ID from links table"
                        },
                        "response_length": {
                            "type": "string",
                            "enum": ["short", "medium", "long"],
                            "description": "Viewport size for new page"
                        },
                    },
                    "required": ["ref_id", "link_id"]
                }
            },
            {
                "type": "function",
                "name": "note",
                "description": "Record a note about this domain. Use freely for facts, patterns, constraints, observations.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "content": {
                            "type": "string",
                            "description": "What you observed or learned"
                        },
                    },
                    "required": ["content"]
                }
            },
            {
                "type": "function",
                "name": "conclude_research",
                "description": "Conclude research and transition to decision mode. MUST be called before breakdown() or seeding. Briefly summarize what you learned.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "summary": {
                            "type": "string",
                            "description": "Brief summary of research findings and what you learned about the domain"
                        },
                    },
                    "required": ["summary"]
                }
            },
            {
                "type": "function",
                "name": "breakdown",
                "description": "Break scope into sub-scopes. Use when domain is too broad. Requires conclude_research() first.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "children": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "description": {"type": "string"},
                                    "weight": {"type": "number"},
                                },
                                "required": ["description"]
                            },
                            "description": "Sub-scopes with descriptions and relative weights"
                        },
                    },
                    "required": ["children"]
                }
            },
            {
                "type": "function",
                "name": "extract_seeds",
                "description": "Extract seeds from page content by line ranges. Use when source has actual row items. Requires conclude_research() first.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "ref_id": {
                            "type": "string",
                            "description": "Page ref_id containing items"
                        },
                        "line_ranges": {
                            "type": "array",
                            "items": {
                                "type": "array",
                                "items": {"type": "integer"},
                                "minItems": 2,
                                "maxItems": 2
                            },
                            "description": "[[start, end], ...] line ranges to extract"
                        },
                    },
                    "required": ["ref_id", "line_ranges"]
                }
            },
            {
                "type": "function",
                "name": "write_seeds",
                "description": "Write seeds. Use when you understand what rows should be. Requires conclude_research() first.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "seeds": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Seed descriptions - each becomes one row assignment"
                        },
                    },
                    "required": ["seeds"]
                }
            },
            {
                "type": "function",
                "name": "interact",
                "description": "Use Browser Agent for complex page interactions (login, forms, JS-heavy pages, pagination). You stay in control via checkpoints.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "url_or_ref_id": {
                            "type": "string",
                            "description": "URL or ref_id to interact with"
                        },
                        "task": {
                            "type": "string",
                            "description": "What to accomplish (e.g., 'login', 'click Load More')"
                        },
                    },
                    "required": ["url_or_ref_id", "task"]
                }
            },
        ]
    
    async def execute_tool(self, name: str, args: Dict) -> Tuple[str, float]:
        """Execute tool by name with args. Returns (result, cost)."""
        try:
            if name == "brave_search":
                return await self.brave_search(
                    query=args.get("query", ""),
                    response_length=args.get("response_length", "medium"),
                )
            
            elif name == "open":
                return await self.open(
                    ref_id_or_url=args.get("ref_id_or_url", ""),
                    start_line=args.get("start_line", 0),
                    response_length=args.get("response_length", "medium"),
                )
            
            elif name == "find":
                return await self.find(
                    ref_id=args.get("ref_id", ""),
                    pattern=args.get("pattern", ""),
                    response_length=args.get("response_length", "medium"),
                )
            
            elif name == "click":
                return await self.click(
                    ref_id=args.get("ref_id", ""),
                    link_id=args.get("link_id", 0),
                    response_length=args.get("response_length", "medium"),
                )
            
            elif name == "note":
                return self.note(content=args.get("content", ""))
            
            elif name == "conclude_research":
                return self.conclude_research(summary=args.get("summary", ""))
            
            elif name == "breakdown":
                return self.breakdown(children=args.get("children", []))
            
            elif name == "extract_seeds":
                return await self.extract_seeds(
                    ref_id=args.get("ref_id", ""),
                    line_ranges=args.get("line_ranges", []),
                )
            
            elif name == "write_seeds":
                return self.write_seeds(seeds=args.get("seeds", []))
            
            elif name == "interact":
                return await self.interact(
                    url_or_ref_id=args.get("url_or_ref_id", ""),
                    task=args.get("task", ""),
                )
            
            else:
                return f"Unknown tool: {name}", 0.0
                
        except Exception as e:
            logger.error(f"[ResearchTools] {name} failed: {e}")
            return f"Tool error: {e}", 0.0