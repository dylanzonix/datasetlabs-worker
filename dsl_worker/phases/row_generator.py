"""
Row Generator

Agent that generates a single row from a seed.

Has full tool access:
- web_search
- browse  
- code_exec

Takes a seed (scope, notes, content, schema) and produces one row.
"""

import asyncio
import json
import logging
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from dsl_worker.config import settings
from dsl_worker.phases.research_tools import Seed

logger = logging.getLogger(__name__)

MAX_GENERATION_TURNS = 15

# Browser-use imports
try:
    from browser_use import Browser
    BROWSER_AVAILABLE = True
except ImportError:
    Browser = None
    BROWSER_AVAILABLE = False


@dataclass
class GeneratedRow:
    """Result of row generation."""
    success: bool
    row: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    cost_usd: float = 0.0


class RowGenerator:
    """
    Generates a single row from a seed.
    
    Each generator can create its own browser if needed for web tasks.
    """
    
    def __init__(
        self,
        workspace_dir: Path,
        openai_client: Any,
        brave_api_key: Optional[str] = None,
        sandbox: Optional[Any] = None,
        stop_checker: Optional[Callable[[], bool]] = None,
    ):
        self.workspace_dir = Path(workspace_dir)
        self.openai_client = openai_client
        self.brave_api_key = brave_api_key
        self.sandbox = sandbox
        self.stop_checker = stop_checker
        
        # Browser - lazy initialized if needed
        self._browser: Optional[Any] = None
    
    def _should_stop(self) -> bool:
        return self.stop_checker and self.stop_checker()
    
    async def _get_browser(self) -> Any:
        """Get or create browser for this generator."""
        if not BROWSER_AVAILABLE:
            return None
        
        if self._browser is None:
            self._browser = Browser(
                headless=False,
                downloads_path=str(self.workspace_dir / "downloads"),
                auto_download_pdfs=True,
            )
            await self._browser.start()
            logger.debug("[RowGenerator] Browser started")
        
        return self._browser
    
    async def cleanup(self):
        """Cleanup browser."""
        if self._browser:
            try:
                await self._browser.stop()
            except Exception as e:
                logger.warning(f"[RowGenerator] Browser cleanup error: {e}")
            self._browser = None
    
    async def generate(self, assignment: Dict) -> GeneratedRow:
        """Generate a single row from an assignment."""
        if self._should_stop():
            return GeneratedRow(success=False, error="Stopped")
        
        scope = assignment.get("scope_description", "")
        seed = assignment.get("seed", "")
        notes = assignment.get("notes", [])
        research_summary = assignment.get("research_summary", "")
        schema = assignment.get("schema", [])
        
        system_prompt = self._build_system_prompt(scope, notes, research_summary, schema, seed)
        
        current_row = {}
        total_cost = 0.0
        
        user_msg = f"Generate a dataset row based on this seed:\n\n{seed}"
        messages = [{"role": "user", "content": user_msg}]
        
        for turn in range(MAX_GENERATION_TURNS):
            if self._should_stop():
                return GeneratedRow(success=False, error="Stopped", cost_usd=total_cost)
            
            try:
                response, cost = await self.openai_client.responses_create(
                    model=settings.generation_model,
                    input=[{"role": "system", "content": system_prompt}] + messages,
                    tools=self._get_tools(),
                    max_output_tokens=8000,
                )
                total_cost += cost.total_cost_usd
                
                submitted = False
                
                for item in response.output:
                    if item.type == "function_call":
                        name = item.name
                        args = json.loads(item.arguments)
                        call_id = item.call_id
                        
                        result, row_update, is_submit = await self._handle_tool(
                            name, args, current_row
                        )
                        
                        if row_update:
                            current_row.update(row_update)
                        
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
                        
                        if is_submit:
                            submitted = True
                            break
                
                if submitted:
                    missing = []
                    for col in schema:
                        col_name = col.get("name")
                        if col_name and col_name not in current_row:
                            missing.append(col_name)
                    
                    if missing:
                        return GeneratedRow(
                            success=False,
                            error=f"Missing columns: {missing}",
                            row=current_row,
                            cost_usd=total_cost,
                        )
                    
                    return GeneratedRow(
                        success=True,
                        row=current_row,
                        cost_usd=total_cost,
                    )
                
                if not any(item.type == "function_call" for item in response.output):
                    messages.append({"role": "assistant", "content": response.output_text})
                    messages.append({
                        "role": "user",
                        "content": "Continue. Use set_column() to fill columns and submit_row() when done."
                    })
                
            except Exception as e:
                logger.error(f"[RowGenerator] Error: {e}")
                return GeneratedRow(success=False, error=str(e), cost_usd=total_cost)
        
        return GeneratedRow(
            success=False,
            error="Max turns reached",
            row=current_row,
            cost_usd=total_cost,
        )
    
    def _build_system_prompt(
        self,
        scope: str,
        notes: List[str],
        research_summary: str,
        schema: List[Dict],
        seed: str,
    ) -> str:
        """Build system prompt for generation."""
        
        notes_str = "\n".join(f"- {n}" for n in notes) if notes else "None"
        
        schema_str = "\n".join(
            f"- {col.get('name')} ({col.get('type', 'string')}): {col.get('description', '')}"
            for col in schema
        )
        
        summary_section = ""
        if research_summary:
            summary_section = f"""
<research_summary>
{research_summary}
</research_summary>
"""
        
        notes_section = ""
        if notes:
            notes_section = f"""
<notes>
{notes_str}
</notes>
"""
        
        return f"""You are generating a single dataset row.

<scope>
{scope}
</scope>

<seed>
{seed}
</seed>

<schema>
{schema_str}
</schema>
{notes_section}{summary_section}
## Tools

- set_column(name, value) - Set a column value
- web_search(query) - Search the web
- browse(url) - Browse a URL
- code_exec(script) - Execute Python (files at /workspace/)
- submit_row() - Submit completed row

## Guidelines

1. The seed tells you what this row is about - use it as your anchor
2. Fill each column based on the seed and your knowledge
3. Use tools if you need to look up specific information
4. Call submit_row() when all columns are filled

Be accurate. Follow the notes/constraints.
"""

    def _get_tools(self) -> List[Dict]:
        """Tool definitions."""
        return [
            {
                "type": "function",
                "name": "set_column",
                "description": "Set a column value for the row.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "description": "Column name"},
                        "value": {"description": "Column value (string, number, list, etc.)"}
                    },
                    "required": ["name", "value"]
                }
            },
            {
                "type": "function",
                "name": "web_search",
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
                "description": "Browse a URL to get information.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "url": {"type": "string", "description": "URL to browse"}
                    },
                    "required": ["url"]
                }
            },
            {
                "type": "function",
                "name": "code_exec",
                "description": "Execute Python code. Files at /workspace/.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "script": {"type": "string", "description": "Python code"}
                    },
                    "required": ["script"]
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
    
    async def _handle_tool(
        self,
        name: str,
        args: Dict,
        current_row: Dict,
    ) -> Tuple[str, Optional[Dict], bool]:
        """Handle a tool call. Returns (result_text, row_update, is_submit)."""
        if name == "set_column":
            col_name = args.get("name", "")
            value = args.get("value")
            return f"Set {col_name}", {col_name: value}, False
        
        elif name == "submit_row":
            return "Row submitted", None, True
        
        elif name == "web_search":
            result = await self._do_web_search(args.get("query", ""))
            return result, None, False
        
        elif name == "browse":
            result = await self._do_browse(args.get("url", ""))
            return result, None, False
        
        elif name == "code_exec":
            result = self._do_code_exec(args.get("script", ""))
            return result, None, False
        
        return f"Unknown tool: {name}", None, False
    
    async def _do_web_search(self, query: str) -> str:
        """Execute web search."""
        if not self.brave_api_key:
            return "Web search not available"
        
        import httpx
        
        max_retries = 3
        base_delay = 1.0
        
        for attempt in range(max_retries):
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
                    
                    results = []
                    for r in data.get("web", {}).get("results", []):
                        results.append(
                            f"- {r.get('title', '')}: {r.get('description', '')}\n"
                            f"  {r.get('url', '')}"
                        )
                    
                    return "\n".join(results) if results else "No results"
                    
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 429 or e.response.status_code >= 500:
                    if attempt < max_retries - 1:
                        delay = base_delay * (2 ** attempt)
                        await asyncio.sleep(delay)
                        continue
                return f"Search error: {e}"
                
            except Exception as e:
                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt)
                    await asyncio.sleep(delay)
                    continue
                return f"Search error: {e}"
        
        return "Search failed after retries"
    
    async def _do_browse(self, url: str) -> str:
        """Browse a URL."""
        browser = await self._get_browser()
        if not browser:
            return "Browser not available"
        
        from markdownify import markdownify as md
        
        try:
            page = await browser.new_page(url)
            await asyncio.sleep(2.0)
            
            try:
                html = await page.evaluate('() => document.body.innerHTML')
                markdown = md(html, heading_style='ATX')
                
                if len(markdown) > 3000:
                    return markdown[:3000] + f"\n\n[...truncated, {len(markdown)} total chars]"
                return markdown
            finally:
                await browser.close_page(page)
                
        except Exception as e:
            return f"Browse error: {e}"
    
    def _do_code_exec(self, script: str) -> str:
        """Execute code."""
        if not self.sandbox:
            return "Code execution not available"
        
        result = self.sandbox.execute(
            script=script,
            workspace_dir=str(self.workspace_dir),
            timeout=60,
        )
        
        if result.success:
            return result.stdout if result.stdout else "Code executed (no output)"
        return f"Error: {result.error}"


class GenerationWorkerPool:
    """Pool of workers that process seeds."""
    
    def __init__(
        self,
        workspace_dir: Path,
        openai_client: Any,
        db_session: Any,
        project_id: Any,
        version_id: Any,
        brave_api_key: Optional[str] = None,
        browser: Optional[Any] = None,  # Ignored - each generator creates own if needed
        sandbox: Optional[Any] = None,
        num_workers: int = 10,
        stop_checker: Optional[Callable[[], bool]] = None,
        cost_tracker: Optional[Any] = None,
        checkpoint_callback: Optional[Callable[[int, bool, Optional[str]], Any]] = None,
    ):
        self.workspace_dir = Path(workspace_dir)
        self.openai_client = openai_client
        self.db = db_session
        self.project_id = project_id
        self.version_id = version_id
        self.brave_api_key = brave_api_key
        self.sandbox = sandbox
        self.num_workers = num_workers
        self.stop_checker = stop_checker
        self.cost_tracker = cost_tracker
        self.checkpoint_callback = checkpoint_callback
        
        self._total_cost = 0.0
        self._rows_generated = 0
        self._errors = 0
    
    def _should_stop(self) -> bool:
        return self.stop_checker and self.stop_checker()
    
    async def process_seeds(self, seeds: List[Seed], schema: List[Dict]) -> Tuple[int, int]:
        """Process seeds directly. Returns (success_count, error_count)."""
        if not seeds:
            logger.warning("[GenerationPool] No seeds to process")
            return 0, 0
        
        logger.info(f"[GenerationPool] Processing {len(seeds)} seeds with {self.num_workers} workers")
        
        # Convert seeds to assignment dicts
        assignments = []
        for seed in seeds:
            assignment = {
                "scope_id": seed.scope_id,
                "scope_description": seed.scope_description,
                "seed": seed.content,
                "notes": seed.notes,
                "research_summary": seed.research_summary,
                "schema": schema,
                "source_ref": seed.source_ref,
                "source_url": seed.source_url,
            }
            assignments.append(assignment)
        
        # Create queue with (index, assignment) pairs
        queue = asyncio.Queue()
        for i, assignment in enumerate(assignments):
            await queue.put((i, assignment))
        
        success_count = 0
        error_count = 0
        lock = asyncio.Lock()
        
        async def worker():
            nonlocal success_count, error_count
            
            # Each worker gets its own generator (with potential for own browser)
            generator = RowGenerator(
                workspace_dir=self.workspace_dir,
                openai_client=self.openai_client,
                brave_api_key=self.brave_api_key,
                sandbox=self.sandbox,
                stop_checker=self.stop_checker,
            )
            
            try:
                while True:
                    if self._should_stop():
                        break
                    
                    try:
                        index, assignment = queue.get_nowait()
                    except asyncio.QueueEmpty:
                        break
                    
                    try:
                        result = await generator.generate(assignment)
                        
                        self._total_cost += result.cost_usd
                        
                        row_id = None
                        if result.success and result.row:
                            row_id = await self._save_row(result.row, assignment)
                            
                            async with lock:
                                success_count += 1
                                self._rows_generated += 1
                            
                            if success_count % 10 == 0:
                                logger.info(f"[GenerationPool] Generated {success_count} rows...")
                            
                            # Checkpoint callback
                            if self.checkpoint_callback:
                                await self.checkpoint_callback(index, True, row_id)
                        else:
                            async with lock:
                                error_count += 1
                                self._errors += 1
                            logger.warning(f"[GenerationPool] Failed: {result.error}")
                            
                            # Checkpoint callback for failure
                            if self.checkpoint_callback:
                                await self.checkpoint_callback(index, False, None)
                            
                    except Exception as e:
                        logger.error(f"[GenerationPool] Error processing seed: {e}")
                        async with lock:
                            error_count += 1
                            self._errors += 1
                        
                        if self.checkpoint_callback:
                            await self.checkpoint_callback(index, False, None)
            finally:
                await generator.cleanup()
        
        workers = [asyncio.create_task(worker()) for _ in range(self.num_workers)]
        await asyncio.gather(*workers)
        
        logger.info(f"[GenerationPool] Completed: {success_count} success, {error_count} errors")
        
        return success_count, error_count
    
    async def _save_row(self, row: Dict, assignment: Dict) -> Optional[str]:
        """Save generated row to database. Returns row ID."""
        from sqlalchemy import func as sql_func
        from dsl_api.models.project_version import ProjectVersion
        from dsl_api.models.sample import Sample
        
        row_json = json.dumps(row, ensure_ascii=False)
        clean_json = row_json.replace('\\u0000', '').replace('\x00', '')
        clean_row = json.loads(clean_json)
        
        row_id = str(uuid.uuid4())
        
        try:
            self.db.query(ProjectVersion).filter(
                ProjectVersion.id == self.version_id
            ).with_for_update().first()
            
            max_seq = (
                self.db.query(sql_func.max(Sample.seq))
                .filter(Sample.version_id == self.version_id)
                .scalar() or 0
            )
            
            sample = Sample(
                id=uuid.UUID(row_id),
                project_id=self.project_id,
                version_id=self.version_id,
                seq=max_seq + 1,
                row=clean_row,
                tags={"scope": assignment.get("scope_description", "")},
            )
            self.db.add(sample)
            
            self.db.query(ProjectVersion).filter(
                ProjectVersion.id == self.version_id
            ).update(
                {ProjectVersion.generated_count: ProjectVersion.generated_count + 1},
                synchronize_session=False
            )
            
            self.db.commit()
            
            return row_id
            
        except Exception as e:
            logger.error(f"[GenerationPool] Save failed: {e}")
            self.db.rollback()
            raise
    
    def get_stats(self) -> Dict:
        """Get current stats."""
        return {
            "rows_generated": self._rows_generated,
            "errors": self._errors,
            "total_cost": self._total_cost,
        }