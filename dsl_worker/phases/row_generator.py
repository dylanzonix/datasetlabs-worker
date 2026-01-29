"""
Row Generator

Agent that generates a single row from an assignment.

Has full tool access:
- web_search
- browse
- code_exec

Takes an assignment (scope, knowledge, example, schema) and produces one row.
"""

import asyncio
import json
import logging
import os
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

GENERATION_MODEL = os.getenv("GENERATION_MODEL", "gpt-4o")
MAX_GENERATION_TURNS = 15


@dataclass
class GeneratedRow:
    """Result of row generation."""
    success: bool
    row: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    cost_usd: float = 0.0


class RowGenerator:
    """
    Generates a single row from an assignment.
    
    Each assignment file contains:
    - scope: description of what this row should be
    - example: optional concrete example/seed
    - knowledge: list of facts/constraints
    - schema: column definitions
    - source_file: optional reference to source
    """
    
    def __init__(
        self,
        workspace_dir: Path,
        openai_client: Any,
        brave_api_key: Optional[str] = None,
        browser_pool: Optional[Any] = None,
        sandbox: Optional[Any] = None,
        stop_checker: Optional[Callable[[], bool]] = None,
    ):
        self.workspace_dir = Path(workspace_dir)
        self.openai_client = openai_client
        self.brave_api_key = brave_api_key
        self.browser_pool = browser_pool
        self.sandbox = sandbox
        self.stop_checker = stop_checker
    
    def _should_stop(self) -> bool:
        return self.stop_checker and self.stop_checker()
    
    async def generate(self, assignment: Dict) -> GeneratedRow:
        """
        Generate a single row from an assignment.
        
        Args:
            assignment: Dict with scope, example, knowledge, schema, source_file
            
        Returns:
            GeneratedRow with success, row data, error, cost
        """
        if self._should_stop():
            return GeneratedRow(success=False, error="Stopped")
        
        scope = assignment.get("scope", "")
        example = assignment.get("example")
        knowledge = assignment.get("knowledge", [])
        schema = assignment.get("schema", [])
        source_file = assignment.get("source_file")
        
        # Build system prompt
        system_prompt = self._build_system_prompt(scope, knowledge, schema, example, source_file)
        
        # Current row state
        current_row = {}
        total_cost = 0.0
        
        # Messages
        if example:
            user_msg = f"Generate a dataset row based on this example:\n\n{example}"
        else:
            user_msg = f"Generate a dataset row for this scope:\n\n{scope}"
        
        messages = [{"role": "user", "content": user_msg}]
        
        for turn in range(MAX_GENERATION_TURNS):
            if self._should_stop():
                return GeneratedRow(success=False, error="Stopped", cost_usd=total_cost)
            
            try:
                response, cost = await self.openai_client.responses_create(
                    model=GENERATION_MODEL,
                    input=[{"role": "system", "content": system_prompt}] + messages,
                    tools=self._get_tools(),
                    max_output_tokens=8000,
                )
                total_cost += cost.total_cost_usd
                
                # Process tool calls
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
                    # Validate row has required columns
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
                
                # If no tool calls, check if we should continue
                if not any(item.type == "function_call" for item in response.output):
                    # Add the assistant's text response and prompt to continue
                    messages.append({"role": "assistant", "content": response.output_text})
                    messages.append({"role": "user", "content": "Continue. Use set_column() to fill columns and submit_row() when done."})
                
            except Exception as e:
                logger.error(f"[RowGenerator] Error: {e}")
                return GeneratedRow(success=False, error=str(e), cost_usd=total_cost)
        
        # Max turns reached
        return GeneratedRow(
            success=False,
            error="Max turns reached",
            row=current_row,
            cost_usd=total_cost,
        )
    
    def _build_system_prompt(
        self,
        scope: str,
        knowledge: List[str],
        schema: List[Dict],
        example: Optional[str],
        source_file: Optional[str],
    ) -> str:
        """Build system prompt for generation."""
        
        knowledge_str = "\n".join(f"- {k}" for k in knowledge) if knowledge else "None"
        
        schema_str = "\n".join(
            f"- {col.get('name')} ({col.get('type')}): {col.get('description', '')}"
            for col in schema
        )
        
        example_section = ""
        if example:
            example_section = f"""
## Example/Seed
Use this as the basis for your row:
{example}
"""
        
        source_section = ""
        if source_file:
            source_section = f"""
## Source Reference
This assignment came from: {source_file}
You can read this file with code_exec if you need more context.
"""
        
        return f"""You are generating a single dataset row.

## Scope
{scope}

## Schema (columns to fill)
{schema_str}

## Knowledge (facts/constraints)
{knowledge_str}
{example_section}
{source_section}

## Tools

**set_column(name, value)** - Set a column value. Call once per column.

**web_search(query)** - Search the web for information.

**browse(url, task)** - Browse a URL to get information.

**code_exec(script)** - Execute Python. Files at /workspace/.

**submit_row()** - Submit the completed row. Call after all columns are set.

## Guidelines

1. Analyze what you need to generate
2. If you have an example, use it as the basis
3. Use tools to look up any missing information
4. Set each column using set_column()
5. Call submit_row() when all columns are filled

Be accurate. Use the knowledge constraints. Make sure the row makes sense.
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
                        "url": {"type": "string", "description": "URL to browse"},
                        "task": {"type": "string", "description": "What to extract"}
                    },
                    "required": ["url", "task"]
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
        """
        Handle a tool call.
        
        Returns (result_text, row_update, is_submit)
        """
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
            result = await self._do_browse(args.get("url", ""), args.get("task", ""))
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
                    results.append(f"- {r.get('title', '')}: {r.get('description', '')}\n  {r.get('url', '')}")
                
                return "\n".join(results) if results else "No results"
                
        except Exception as e:
            return f"Search error: {e}"
    
    async def _do_browse(self, url: str, task: str) -> str:
        """Browse a URL."""
        if not self.browser_pool:
            return "Browser not available"
        
        try:
            content = await self.browser_pool.simple_fetch(url)
            # Return truncated content
            if len(content) > 3000:
                return content[:3000] + f"\n\n[...truncated, {len(content)} total chars]"
            return content
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
    """
    Pool of workers that process assignment directories.
    """
    
    def __init__(
        self,
        workspace_dir: Path,
        openai_client: Any,
        db_session: Any,
        project_id: Any,
        version_id: Any,
        brave_api_key: Optional[str] = None,
        browser_pool: Optional[Any] = None,
        sandbox: Optional[Any] = None,
        num_workers: int = 10,
        stop_checker: Optional[Callable[[], bool]] = None,
        cost_tracker: Optional[Any] = None,
    ):
        self.workspace_dir = Path(workspace_dir)
        self.openai_client = openai_client
        self.db = db_session
        self.project_id = project_id
        self.version_id = version_id
        self.brave_api_key = brave_api_key
        self.browser_pool = browser_pool
        self.sandbox = sandbox
        self.num_workers = num_workers
        self.stop_checker = stop_checker
        self.cost_tracker = cost_tracker
        
        self._total_cost = 0.0
        self._rows_generated = 0
        self._errors = 0
    
    def _should_stop(self) -> bool:
        return self.stop_checker and self.stop_checker()
    
    async def process_directory(self, assignment_dir: str) -> Tuple[int, int]:
        """
        Process all assignments in a directory.
        
        Returns (success_count, error_count)
        """
        dir_path = Path(assignment_dir)
        if not dir_path.exists():
            logger.warning(f"[GenerationPool] Directory not found: {assignment_dir}")
            return 0, 0
        
        # Get all assignment files
        assignment_files = sorted(dir_path.glob("*.json"))
        
        if not assignment_files:
            logger.warning(f"[GenerationPool] No assignments in: {assignment_dir}")
            return 0, 0
        
        logger.info(f"[GenerationPool] Processing {len(assignment_files)} assignments from {dir_path.name}")
        
        # Create work queue
        queue = asyncio.Queue()
        for f in assignment_files:
            await queue.put(f)
        
        # Stats
        success_count = 0
        error_count = 0
        lock = asyncio.Lock()
        
        async def worker():
            nonlocal success_count, error_count
            
            generator = RowGenerator(
                workspace_dir=self.workspace_dir,
                openai_client=self.openai_client,
                brave_api_key=self.brave_api_key,
                browser_pool=self.browser_pool,
                sandbox=self.sandbox,
                stop_checker=self.stop_checker,
            )
            
            while True:
                if self._should_stop():
                    break
                
                try:
                    assignment_file = queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
                
                try:
                    # Read assignment
                    assignment = json.loads(assignment_file.read_text())
                    
                    # Generate row
                    result = await generator.generate(assignment)
                    
                    self._total_cost += result.cost_usd
                    
                    if result.success and result.row:
                        # Save to database
                        await self._save_row(result.row, assignment)
                        
                        async with lock:
                            success_count += 1
                            self._rows_generated += 1
                        
                        if success_count % 10 == 0:
                            logger.info(f"[GenerationPool] Generated {success_count} rows...")
                    else:
                        async with lock:
                            error_count += 1
                            self._errors += 1
                        logger.warning(f"[GenerationPool] Failed: {result.error}")
                        
                except Exception as e:
                    logger.error(f"[GenerationPool] Error processing {assignment_file}: {e}")
                    async with lock:
                        error_count += 1
                        self._errors += 1
        
        # Run workers
        workers = [asyncio.create_task(worker()) for _ in range(self.num_workers)]
        await asyncio.gather(*workers)
        
        logger.info(f"[GenerationPool] Completed {dir_path.name}: {success_count} success, {error_count} errors")
        
        return success_count, error_count
    
    async def _save_row(self, row: Dict, assignment: Dict):
        """Save generated row to database."""
        from sqlalchemy import func as sql_func
        from dsl_api.models.project_version import ProjectVersion
        from dsl_api.models.sample import Sample
        
        # Sanitize row (remove NULL bytes)
        row_json = json.dumps(row, ensure_ascii=False)
        clean_json = row_json.replace('\\u0000', '').replace('\x00', '')
        clean_row = json.loads(clean_json)
        
        try:
            # Lock version row
            self.db.query(ProjectVersion).filter(
                ProjectVersion.id == self.version_id
            ).with_for_update().first()
            
            # Get next sequence number
            max_seq = (
                self.db.query(sql_func.max(Sample.seq))
                .filter(Sample.version_id == self.version_id)
                .scalar() or 0
            )
            
            # Create sample
            sample = Sample(
                id=uuid.uuid4(),
                project_id=self.project_id,
                version_id=self.version_id,
                seq=max_seq + 1,
                row=clean_row,
                tags={"scope": assignment.get("scope", "")},
            )
            self.db.add(sample)
            
            # Update version count
            self.db.query(ProjectVersion).filter(
                ProjectVersion.id == self.version_id
            ).update(
                {ProjectVersion.generated_count: ProjectVersion.generated_count + 1},
                synchronize_session=False
            )
            
            self.db.commit()
            
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