"""
Scope Processor

Recursive processor that:
1. Researches a scope (becomes domain expert)
2. Assesses: do I have knowledge, examples, uniqueness?
3. Either breaks down (if not ready) or writes assignments (if ready)

No seeds, no slots, no quotas to fill. Just:
- Understand the domain
- Break down until scopes are small enough
- Write assignment files for generation
"""

import asyncio
import json
import logging
import os
import random
import string
import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from dsl_worker.phases.base import Phase, PhaseResult
from dsl_worker.phases.browser_pool import BrowserPool
from dsl_worker.phases.sandbox import SandboxExecutor

logger = logging.getLogger(__name__)

RESEARCH_MODEL = os.getenv("RESEARCH_MODEL", "gpt-4o")
BREAKDOWN_MODEL = os.getenv("BREAKDOWN_MODEL", "gpt-4o")

# Thresholds
MIN_QUOTA_FOR_BREAKDOWN = 10  # Don't break down scopes smaller than this
MAX_RESEARCH_TURNS = 20  # Max research iterations per scope
MAX_DEPTH = 5  # Max tree depth


def short_id(length: int = 6) -> str:
    """Generate a short alphanumeric ID."""
    return ''.join(random.choices(string.ascii_lowercase + string.digits, k=length))


@dataclass
class Source:
    """A source file in the workspace."""
    file_path: str
    url: Optional[str]
    summary: str
    example_potential: Optional[str] = None  # e.g., "~50 dentist listings"


@dataclass
class Scope:
    """A scope to process."""
    description: str
    quota: int
    knowledge: List[str] = field(default_factory=list)  # Inherited + local observations
    sources: List[Source] = field(default_factory=list)
    parent: Optional['Scope'] = None
    children: Optional[List['Scope']] = None
    depth: int = 0
    
    def inherit_knowledge(self) -> List[str]:
        """Get all knowledge from ancestors + self."""
        if self.parent:
            return self.parent.inherit_knowledge() + self.knowledge
        return self.knowledge.copy()


class ScopeProcessor:
    """
    Processes scopes recursively.
    
    Main entry: process(scope) -> writes assignment files
    """
    
    def __init__(
        self,
        workspace_dir: Path,
        schema: List[Dict],
        openai_client: Any,
        brave_api_key: Optional[str] = None,
        browser_pool: Optional[BrowserPool] = None,
        sandbox: Optional[SandboxExecutor] = None,
        stop_checker: Optional[Callable[[], bool]] = None,
        cost_tracker: Optional[Any] = None,
    ):
        self.workspace_dir = Path(workspace_dir)
        self.schema = schema
        self.openai_client = openai_client
        self.brave_api_key = brave_api_key
        self.browser_pool = browser_pool
        self.sandbox = sandbox or SandboxExecutor(use_pool=True, pool_size=2)
        self.stop_checker = stop_checker
        self.cost_tracker = cost_tracker
        
        # Ensure directories exist
        (self.workspace_dir / "uploads").mkdir(parents=True, exist_ok=True)
        (self.workspace_dir / "web").mkdir(parents=True, exist_ok=True)
        (self.workspace_dir / "assignments").mkdir(parents=True, exist_ok=True)
        
        self._total_cost = 0.0
        self._assignment_dirs: List[str] = []  # Track created assignment directories
    
    def _should_stop(self) -> bool:
        """Check if we should stop processing."""
        return self.stop_checker and self.stop_checker()
    
    async def process(self, scope: Scope) -> List[str]:
        """
        Process a scope recursively.
        
        Returns list of assignment directory paths created.
        """
        if self._should_stop():
            return []
        
        logger.info(f"[ScopeProcessor] Processing: {scope.description[:80]}... (quota={scope.quota}, depth={scope.depth})")
        
        # Research until we understand this scope
        await self._research(scope)
        
        if self._should_stop():
            return []
        
        # Assess: ready to create assignments?
        ready, reason = self._assess(scope)
        
        if ready:
            logger.info(f"[ScopeProcessor] Ready to generate: {reason}")
            assignment_dirs = await self._write_assignments(scope)
            self._assignment_dirs.extend(assignment_dirs)
            return assignment_dirs
        else:
            logger.info(f"[ScopeProcessor] Breaking down: {reason}")
            
            if scope.depth >= MAX_DEPTH:
                logger.warning(f"[ScopeProcessor] Max depth reached, forcing assignment creation")
                assignment_dirs = await self._write_assignments(scope)
                self._assignment_dirs.extend(assignment_dirs)
                return assignment_dirs
            
            # Break down into children
            children = await self._breakdown(scope)
            scope.children = children
            
            # Process each child
            all_dirs = []
            for child in children:
                if self._should_stop():
                    break
                dirs = await self.process(child)
                all_dirs.extend(dirs)
            
            return all_dirs
    
    # =========================================================================
    # Research
    # =========================================================================
    
    async def _research(self, scope: Scope):
        """
        Research loop - accumulate knowledge and find sources.
        
        Uses tools:
        - brave_search(query)
        - browse(url, task)
        - code_exec(script)
        """
        logger.info(f"[ScopeProcessor] Researching: {scope.description[:60]}...")
        
        # Build system prompt
        system_prompt = self._build_research_system_prompt(scope)
        
        # Messages for conversation
        messages = [{"role": "user", "content": f"Research this scope and become an expert on it:\n\n{scope.description}"}]
        
        for turn in range(MAX_RESEARCH_TURNS):
            if self._should_stop():
                break
            
            # Check if we have enough
            if self._has_enough_for_scope(scope):
                logger.info(f"[ScopeProcessor] Research complete after {turn} turns")
                break
            
            try:
                response, cost = await self.openai_client.responses_create(
                    model=RESEARCH_MODEL,
                    input=[{"role": "system", "content": system_prompt}] + messages,
                    tools=self._get_research_tools(),
                )
                self._total_cost += cost.total_cost_usd
                
                # Process response
                action = self._parse_action(response)
                
                if action["type"] == "done":
                    logger.info(f"[ScopeProcessor] Research agent says done: {action.get('reason', '')}")
                    break
                
                if action["type"] == "message":
                    # No tool call, just add to messages and continue
                    messages.append({"role": "assistant", "content": action.get("content", "")})
                    messages.append({"role": "user", "content": "Continue researching or call done() if finished."})
                    continue
                
                # Execute tool
                result, tool_cost = await self._execute_research_tool(scope, action)
                self._total_cost += tool_cost
                
                # Add to messages
                messages.append({
                    "type": "function_call",
                    "call_id": action.get("call_id", f"call_{turn}"),
                    "name": action["type"],
                    "arguments": json.dumps(action.get("args", {})),
                })
                messages.append({
                    "type": "function_call_output",
                    "call_id": action.get("call_id", f"call_{turn}"),
                    "output": result[:10000],  # Truncate if too long
                })
                
            except Exception as e:
                logger.error(f"[ScopeProcessor] Research error: {e}")
                break
        
        logger.info(f"[ScopeProcessor] Research done. Knowledge: {len(scope.knowledge)}, Sources: {len(scope.sources)}")
    
    def _build_research_system_prompt(self, scope: Scope) -> str:
        """Build system prompt for research agent."""
        inherited = scope.inherit_knowledge()
        knowledge_str = "\n".join(f"- {k}" for k in inherited) if inherited else "None yet"
        
        sources_str = ""
        if scope.sources:
            sources_str = "\n".join(
                f"- {s.file_path}: {s.summary}" + (f" ({s.example_potential})" if s.example_potential else "")
                for s in scope.sources
            )
        else:
            sources_str = "None yet"
        
        schema_str = "\n".join(
            f"- {col.get('name')} ({col.get('type')}): {col.get('description', '')}"
            for col in self.schema
        )
        
        return f"""You are a research agent. Your job is to become an expert on this scope and find sources for dataset generation.

## Current Scope
{scope.description}

## Dataset Schema
{schema_str}

## Quota
{scope.quota} rows needed

## Current Knowledge
{knowledge_str}

## Current Sources
{sources_str}

## Tools

**brave_search(query)** - Search the web. Returns list of results with URLs.

**browse(url, task)** - Navigate to URL with browser agent. Handles JS, pagination, Cloudflare. Saves page to workspace. Task examples: "summarize this page", "extract all listings", "find documentation about X".

**code_exec(script)** - Execute Python. Files at /workspace/uploads/, /workspace/web/. Use for inspecting files, parsing, searching.

**add_knowledge(observation)** - Record an observation/fact you learned. This becomes part of the knowledge base.

**note_source(file_path, summary, example_potential)** - Note a source file. example_potential is like "~50 dentist listings" or null if not extractable items.

**done(reason)** - Mark research complete.

## Guidelines

1. First understand what this scope needs
2. Search for relevant sources
3. Browse promising results
4. Inspect files with code_exec if needed
5. Record knowledge as you learn
6. Note sources that could provide examples
7. Call done() when you have enough understanding

Focus on finding:
- Facts/knowledge about the domain
- Sources with extractable examples (if they exist)
- Patterns and structure of the data

You don't need perfect coverage. Just enough understanding to either:
- Create assignments directly (if sources have items)
- Break down further (if scope is too broad)
"""

    def _get_research_tools(self) -> List[Dict]:
        """Tool definitions for research."""
        return [
            {
                "type": "function",
                "name": "brave_search",
                "description": "Search the web for information.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query"}
                    },
                    "required": ["query"]
                }
            },
            {
                "type": "function",
                "name": "browse",
                "description": "Browse a URL with full browser. Handles JS, pagination, Cloudflare. Saves content to workspace.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "url": {"type": "string", "description": "URL to browse"},
                        "task": {"type": "string", "description": "What to do - e.g., 'summarize', 'extract listings', 'find X'"}
                    },
                    "required": ["url", "task"]
                }
            },
            {
                "type": "function",
                "name": "code_exec",
                "description": "Execute Python code. Files at /workspace/uploads/, /workspace/web/. Has pandas, BeautifulSoup, re, json.",
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
                "name": "add_knowledge",
                "description": "Record an observation or fact you learned about this domain.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "observation": {"type": "string", "description": "What you learned"}
                    },
                    "required": ["observation"]
                }
            },
            {
                "type": "function",
                "name": "note_source",
                "description": "Note a source file that could be useful.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "file_path": {"type": "string", "description": "Path in workspace"},
                        "summary": {"type": "string", "description": "What this source contains"},
                        "example_potential": {"type": "string", "description": "e.g., '~50 listings' or null if not items"}
                    },
                    "required": ["file_path", "summary"]
                }
            },
            {
                "type": "function",
                "name": "done",
                "description": "Mark research complete.",
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
    
    async def _execute_research_tool(self, scope: Scope, action: Dict) -> Tuple[str, float]:
        """Execute a research tool and return result + cost."""
        tool_name = action["type"]
        args = action.get("args", {})
        cost = 0.0
        
        logger.info(f"[ScopeProcessor] Tool: {tool_name}")
        
        if tool_name == "brave_search":
            return await self._do_brave_search(args.get("query", ""))
        
        elif tool_name == "browse":
            return await self._do_browse(args.get("url", ""), args.get("task", ""))
        
        elif tool_name == "code_exec":
            return self._do_code_exec(args.get("script", ""))
        
        elif tool_name == "add_knowledge":
            observation = args.get("observation", "")
            if observation:
                scope.knowledge.append(observation)
            return f"Added: {observation}", 0.0
        
        elif tool_name == "note_source":
            file_path = args.get("file_path", "")
            summary = args.get("summary", "")
            example_potential = args.get("example_potential")
            
            source = Source(
                file_path=file_path,
                url=None,
                summary=summary,
                example_potential=example_potential,
            )
            scope.sources.append(source)
            return f"Noted source: {file_path}", 0.0
        
        elif tool_name == "done":
            return f"Research complete: {args.get('reason', '')}", 0.0
        
        return f"Unknown tool: {tool_name}", 0.0
    
    async def _do_brave_search(self, query: str) -> Tuple[str, float]:
        """Execute brave search."""
        if not self.brave_api_key:
            return "Error: BRAVE_API_KEY not configured", 0.0
        
        import httpx
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    "https://api.search.brave.com/res/v1/web/search",
                    params={"q": query, "count": 10},
                    headers={"X-Subscription-Token": self.brave_api_key},
                    timeout=30.0,
                )
                response.raise_for_status()
                data = response.json()
                
                results = []
                for r in data.get("web", {}).get("results", []):
                    results.append({
                        "title": r.get("title", ""),
                        "url": r.get("url", ""),
                        "snippet": r.get("description", ""),
                    })
                
                return json.dumps(results, indent=2), 0.0
                
        except Exception as e:
            return f"Search error: {e}", 0.0
    
    async def _do_browse(self, url: str, task: str) -> Tuple[str, float]:
        """Browse a URL and save to workspace."""
        if not self.browser_pool:
            return "Error: Browser not available", 0.0
        
        try:
            # Use simple_fetch for now (no complex browsing)
            content = await self.browser_pool.simple_fetch(url)
            
            if not content or len(content.strip()) < 50:
                return f"No content from {url}", 0.0
            
            # Save to workspace
            url_hash = hashlib.md5(url.encode()).hexdigest()[:8]
            filename = f"web_{url_hash}.md"
            file_path = self.workspace_dir / "web" / filename
            
            # Add metadata header
            full_content = f"""---
url: {url}
task: {task}
fetched_at: {datetime.now(timezone.utc).isoformat()}
---

{content}
"""
            file_path.write_text(full_content, encoding='utf-8')
            
            # Return summary
            summary = content[:2000]
            if len(content) > 2000:
                summary += f"\n\n[... {len(content)} total chars, saved to {filename}]"
            
            return f"Saved to web/{filename}:\n\n{summary}", 0.0
            
        except Exception as e:
            return f"Browse error: {e}", 0.0
    
    def _do_code_exec(self, script: str) -> Tuple[str, float]:
        """Execute code in sandbox."""
        result = self.sandbox.execute(
            script=script,
            workspace_dir=str(self.workspace_dir),
            timeout=120,
        )
        
        if result.success:
            output = result.stdout
            if result.stderr:
                output += f"\n\nStderr:\n{result.stderr}"
            return output if output.strip() else "Code executed (no output)", 0.0
        else:
            return f"Error: {result.error}\n\n{result.stderr}", 0.0
    
    def _has_enough_for_scope(self, scope: Scope) -> bool:
        """Check if we have enough to proceed (generate or breakdown)."""
        # At minimum, need some knowledge
        if len(scope.knowledge) < 2:
            return False
        
        # If small quota and have sources, probably good
        if scope.quota <= MIN_QUOTA_FOR_BREAKDOWN and scope.sources:
            return True
        
        # If have good sources with examples, good
        if any(s.example_potential for s in scope.sources):
            return True
        
        # If have decent knowledge, can at least break down
        if len(scope.knowledge) >= 5:
            return True
        
        return False
    
    # =========================================================================
    # Assess
    # =========================================================================
    
    def _assess(self, scope: Scope) -> Tuple[bool, str]:
        """
        Assess if scope is ready for assignment creation.
        
        Returns (ready, reason).
        """
        # Small enough quota?
        if scope.quota <= MIN_QUOTA_FOR_BREAKDOWN:
            return True, f"Small quota ({scope.quota} <= {MIN_QUOTA_FOR_BREAKDOWN})"
        
        # Have sources with extractable items?
        sources_with_items = [s for s in scope.sources if s.example_potential]
        if sources_with_items:
            # Estimate total items
            # Try to parse numbers from example_potential like "~50 listings"
            total_items = 0
            for s in sources_with_items:
                import re
                match = re.search(r'(\d+)', s.example_potential or "")
                if match:
                    total_items += int(match.group(1))
            
            if total_items >= scope.quota * 0.5:
                return True, f"Have ~{total_items} items from sources"
        
        # No good sources, need to break down
        return False, f"Need to break down (quota={scope.quota}, sources with items={len(sources_with_items)})"
    
    # =========================================================================
    # Breakdown
    # =========================================================================
    
    async def _breakdown(self, scope: Scope) -> List[Scope]:
        """Break down scope into smaller children."""
        logger.info(f"[ScopeProcessor] Breaking down: {scope.description[:60]}...")
        
        knowledge_str = "\n".join(f"- {k}" for k in scope.inherit_knowledge())
        
        prompt = f"""Break down this scope into smaller, more specific scopes.

## Current Scope
{scope.description}

## Quota
{scope.quota} rows total

## Knowledge
{knowledge_str}

## Guidelines
- Create 2-5 children that together cover the scope
- Assign quotas that sum to {scope.quota}
- Quotas should reflect natural distribution (not necessarily equal)
- Each child should be more specific than parent
- Children should be mutually exclusive (no overlap)

Return JSON:
{{
  "children": [
    {{"description": "specific scope 1", "quota": 150}},
    {{"description": "specific scope 2", "quota": 200}},
    ...
  ]
}}
"""

        try:
            response, cost = await self.openai_client.responses_create(
                model=BREAKDOWN_MODEL,
                input=[{"role": "user", "content": prompt}],
            )
            self._total_cost += cost.total_cost_usd
            
            # Parse response
            text = response.output_text.strip()
            if text.startswith("```"):
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
            
            data = json.loads(text)
            
            children = []
            total_quota = 0
            for item in data.get("children", []):
                child = Scope(
                    description=item.get("description", ""),
                    quota=item.get("quota", 0),
                    knowledge=[],  # Will inherit from parent
                    sources=[],  # Will research fresh
                    parent=scope,
                    depth=scope.depth + 1,
                )
                children.append(child)
                total_quota += child.quota
            
            # Normalize quotas if they don't sum correctly
            if total_quota != scope.quota and total_quota > 0:
                ratio = scope.quota / total_quota
                for child in children:
                    child.quota = max(1, int(child.quota * ratio))
                
                # Fix rounding
                diff = scope.quota - sum(c.quota for c in children)
                if diff != 0 and children:
                    children[0].quota += diff
            
            logger.info(f"[ScopeProcessor] Broke into {len(children)} children")
            return children
            
        except Exception as e:
            logger.error(f"[ScopeProcessor] Breakdown failed: {e}")
            # Fallback: just return scope as-is (will create assignments)
            return []
    
    # =========================================================================
    # Write Assignments
    # =========================================================================
    
    async def _write_assignments(self, scope: Scope) -> List[str]:
        """
        Write assignment files for this scope.
        
        If sources have extractable items, parse them.
        Otherwise, create synthetic assignments.
        """
        logger.info(f"[ScopeProcessor] Writing assignments for: {scope.description[:60]}...")
        
        assignment_dir_id = short_id()
        assignment_dir = self.workspace_dir / "assignments" / assignment_dir_id
        assignment_dir.mkdir(parents=True, exist_ok=True)
        
        # Gather inherited knowledge
        all_knowledge = scope.inherit_knowledge()
        
        # Check for sources with extractable items
        sources_with_items = [s for s in scope.sources if s.example_potential]
        
        if sources_with_items:
            # Parse items from sources
            assignments = await self._parse_assignments_from_sources(scope, sources_with_items, all_knowledge)
        else:
            # Create synthetic assignments
            assignments = self._create_synthetic_assignments(scope, all_knowledge)
        
        # Write assignment files
        for i, assignment in enumerate(assignments):
            filename = f"{i:04d}.json"
            filepath = assignment_dir / filename
            filepath.write_text(json.dumps(assignment, indent=2, ensure_ascii=False), encoding='utf-8')
        
        logger.info(f"[ScopeProcessor] Wrote {len(assignments)} assignments to {assignment_dir_id}/")
        
        return [str(assignment_dir)]
    
    async def _parse_assignments_from_sources(
        self,
        scope: Scope,
        sources: List[Source],
        knowledge: List[str],
    ) -> List[Dict]:
        """Parse extractable items from sources into assignments."""
        
        assignments = []
        
        for source in sources:
            if len(assignments) >= scope.quota:
                break
            
            # Use code_exec to parse items from the source
            parse_prompt = f"""Parse extractable items from this source file.

Source: {source.file_path}
Summary: {source.summary}
Expected: {source.example_potential}

Write Python code that:
1. Reads the file from /workspace/{source.file_path}
2. Parses individual items (each could become a row)
3. Prints each item as JSON on its own line

Use simple parsing - regex, string splitting, etc.
Items don't need to be perfect, just distinct.

Example output format (one JSON per line):
{{"content": "item 1 text..."}}
{{"content": "item 2 text..."}}
"""
            
            try:
                response, cost = await self.openai_client.responses_create(
                    model=RESEARCH_MODEL,
                    input=[{"role": "user", "content": parse_prompt}],
                )
                self._total_cost += cost.total_cost_usd
                
                # Extract code from response
                code = response.output_text
                if "```python" in code:
                    code = code.split("```python")[1].split("```")[0]
                elif "```" in code:
                    code = code.split("```")[1].split("```")[0]
                
                # Execute the parsing code
                result = self.sandbox.execute(
                    script=code,
                    workspace_dir=str(self.workspace_dir),
                    timeout=60,
                )
                
                if result.success and result.stdout:
                    # Parse items from stdout
                    for line in result.stdout.strip().split("\n"):
                        if len(assignments) >= scope.quota:
                            break
                        try:
                            item = json.loads(line)
                            assignment = {
                                "scope": scope.description,
                                "example": item.get("content", str(item)),
                                "knowledge": knowledge,
                                "schema": self.schema,
                                "source_file": source.file_path,
                            }
                            assignments.append(assignment)
                        except json.JSONDecodeError:
                            continue
                            
            except Exception as e:
                logger.warning(f"[ScopeProcessor] Failed to parse {source.file_path}: {e}")
        
        # If we didn't get enough from parsing, add synthetic ones
        while len(assignments) < scope.quota:
            assignments.append({
                "scope": scope.description,
                "example": None,
                "knowledge": knowledge,
                "schema": self.schema,
                "source_file": None,
            })
        
        return assignments[:scope.quota]
    
    def _create_synthetic_assignments(self, scope: Scope, knowledge: List[str]) -> List[Dict]:
        """Create synthetic assignments when no extractable sources."""
        assignments = []
        
        for _ in range(scope.quota):
            assignments.append({
                "scope": scope.description,
                "example": None,
                "knowledge": knowledge,
                "schema": self.schema,
                "source_file": None,
            })
        
        return assignments
    
    # =========================================================================
    # Public Interface
    # =========================================================================
    
    def get_assignment_dirs(self) -> List[str]:
        """Get list of assignment directories created."""
        return self._assignment_dirs.copy()
    
    def get_total_cost(self) -> float:
        """Get total cost incurred."""
        return self._total_cost
    
    async def cleanup(self):
        """Cleanup resources."""
        if self.sandbox:
            self.sandbox.close()