"""
Scope Processor

Recursive processor that:
1. Researches a scope (becomes domain expert)
2. Calls conclude_research() to transition to decision mode
3. Decides: breakdown into smaller scopes OR create seeds
"""

import asyncio
import json
import logging
import random
import string
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from dsl_worker.phases.research_tools import ResearchTools, ResearchScope
from dsl_worker.config import settings

logger = logging.getLogger(__name__)

MAX_RESEARCH_TURNS = 30
MAX_DEPTH = 5


def short_id(length: int = 6) -> str:
    return ''.join(random.choices(string.ascii_lowercase + string.digits, k=length))


@dataclass
class Scope:
    """A scope to process."""
    id: str = ""
    description: str = ""
    quota: int = 0
    notes: List[str] = field(default_factory=list)
    parent: Optional['Scope'] = None
    children: Optional[List['Scope']] = None
    depth: int = 0
    
    def __post_init__(self):
        if not self.id:
            self.id = short_id()
    
    def get_lineage(self) -> List[str]:
        """Get lineage from root to this scope."""
        if self.parent:
            return self.parent.get_lineage() + [self.description]
        return [self.description]
    
    def get_all_parent_notes(self) -> List[str]:
        """Get flattened notes from all ancestors."""
        if self.parent:
            return self.parent.get_all_parent_notes() + self.parent.notes
        return []


class ScopeProcessor:
    """Processes scopes recursively."""
    
    def __init__(
        self,
        workspace_dir: Path,
        schema: List[Dict],
        project_instructions: str,
        openai_client: Any,
        brave_api_key: Optional[str] = None,
        browser: Optional[Any] = None,
        stop_checker: Optional[Callable[[], bool]] = None,
    ):
        self.workspace_dir = Path(workspace_dir)
        self.schema = schema
        self.project_instructions = project_instructions
        self.openai_client = openai_client
        self.brave_api_key = brave_api_key
        self.browser = browser
        self.stop_checker = stop_checker
        
        (self.workspace_dir / "assignments").mkdir(parents=True, exist_ok=True)
        
        self._total_cost = 0.0
        self._all_assignment_dirs: List[str] = []
    
    def _should_stop(self) -> bool:
        return self.stop_checker and self.stop_checker()
    
    async def process(self, scope: Scope) -> List[str]:
        """Process a scope recursively. Returns assignment directory paths."""
        if self._should_stop():
            return []
        
        logger.info(f"[ScopeProcessor] Processing: {scope.description[:80]}... (quota={scope.quota}, depth={scope.depth})")
        
        # Create tools for this scope
        tools = ResearchTools(
            workspace_dir=self.workspace_dir,
            schema=self.schema,
            brave_api_key=self.brave_api_key,
            browser=self.browser,
            openai_client=self.openai_client,
            model=settings.research_model,
            stop_checker=self.stop_checker,
        )
        
        # Set scope with inherited notes
        parent_notes = scope.get_all_parent_notes()
        research_scope = ResearchScope(
            id=scope.id,
            description=scope.description,
            quota=scope.quota,
            depth=scope.depth,
            notes=[],
            parent_notes=parent_notes,
        )
        tools.set_scope(research_scope)
        
        # Run research loop
        await self._research_loop(scope, tools)
        
        if self._should_stop():
            return tools.assignment_dirs
        
        # Check outcome
        if tools.breakdown_children:
            logger.info(f"[ScopeProcessor] Breaking down into {len(tools.breakdown_children)} children")
            
            if scope.depth >= MAX_DEPTH:
                logger.warning(f"[ScopeProcessor] Max depth reached, forcing seed generation")
                self._all_assignment_dirs.extend(tools.assignment_dirs)
                return tools.assignment_dirs
            
            total_weight = sum(c.get("weight", 1.0) for c in tools.breakdown_children)
            
            all_dirs = []
            for child_spec in tools.breakdown_children:
                weight = child_spec.get("weight", 1.0) / total_weight
                child_quota = max(1, int(scope.quota * weight))
                
                child = Scope(
                    description=child_spec["description"],
                    quota=child_quota,
                    notes=research_scope.notes.copy(),
                    parent=scope,
                    depth=scope.depth + 1,
                )
                
                if self._should_stop():
                    break
                
                dirs = await self.process(child)
                all_dirs.extend(dirs)
            
            return all_dirs
        
        if tools.assignment_dirs:
            logger.info(f"[ScopeProcessor] Created {len(tools.assignment_dirs)} assignment directories")
            self._all_assignment_dirs.extend(tools.assignment_dirs)
            return tools.assignment_dirs
        
        logger.warning(f"[ScopeProcessor] No breakdown or assignments for: {scope.description[:60]}")
        return []
    
    async def _research_loop(self, scope: Scope, tools: ResearchTools):
        """Run research loop until agent decides to breakdown or generate."""
        
        system_prompt = self._build_system_prompt(scope, tools)
        
        messages = [{
            "role": "user",
            "content": "Research this scope and decide how to proceed."
        }]
        
        for turn in range(MAX_RESEARCH_TURNS):
            if self._should_stop():
                break
            
            if tools.breakdown_children or tools.assignment_dirs:
                break
            
            try:
                full_messages = [{"role": "system", "content": system_prompt}] + messages
                
                response, cost = await self.openai_client.responses_create(
                    model=settings.research_model,
                    input=full_messages,
                    tools=tools.get_tool_definitions(),
                )
                
                self._total_cost += cost.total_cost_usd
                
                output_text = ""
                tool_calls = []
                
                for item in response.output:
                    if item.type == "message":
                        for content in item.content:
                            if hasattr(content, 'text'):
                                output_text += content.text
                    elif item.type == "function_call":
                        tool_calls.append(item)
                
                if tool_calls:
                    messages.append({
                        "role": "assistant",
                        "content": output_text or None,
                        "tool_calls": [
                            {"id": tc.id, "type": "function", "function": {"name": tc.name, "arguments": tc.arguments}}
                            for tc in tool_calls
                        ]
                    })
                    
                    for tc in tool_calls:
                        name = tc.name
                        args = json.loads(tc.arguments)
                        
                        logger.info(f"[ScopeProcessor] Tool: {name}")
                        result, tool_cost = await tools.execute_tool(name, args)
                        self._total_cost += tool_cost
                        
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": result[:15000],
                        })
                        
                        if tools.breakdown_children or tools.assignment_dirs:
                            break
                else:
                    messages.append({"role": "assistant", "content": output_text or ""})
                    messages.append({
                        "role": "user",
                        "content": "Continue researching. When ready, call conclude_research() to summarize findings, then breakdown() or write_seeds()/extract_seeds()."
                    })
                
            except Exception as e:
                logger.error(f"[ScopeProcessor] Research error: {e}")
                messages.append({"role": "user", "content": f"Error: {e}. Continue or try different approach."})
        
        notes_count = len(tools.scope.notes) if tools.scope else 0
        logger.info(f"[ScopeProcessor] Research done. Turns: {turn + 1}, Notes: {notes_count}, Assignments: {len(tools.assignment_dirs)}")
    
    def _build_system_prompt(self, scope: Scope, tools: ResearchTools) -> str:
        """Build system prompt for research agent."""
        
        schema_lines = []
        for col in self.schema:
            schema_lines.append(f"- {col.get('name')} ({col.get('type', 'string')}): {col.get('description', '')}")
        schema_str = "\n".join(schema_lines)
        
        # Lineage only shown if we have parent scopes
        lineage = scope.get_lineage()
        lineage_section = ""
        if len(lineage) > 1:
            lineage_str = " → ".join(lineage)
            lineage_section = f"""
<lineage>
{lineage_str}
</lineage>
"""
        
        all_notes = []
        if tools.scope:
            all_notes = tools.scope.parent_notes + tools.scope.notes
        
        notes_section = ""
        if all_notes:
            notes_str = "\n".join(f"- {n}" for n in all_notes)
            notes_section = f"""
<notes>
{notes_str}
</notes>
"""
        
        return f'''You are a research agent preparing to generate dataset rows.

<project_instructions>
{self.project_instructions}
</project_instructions>

<schema>
{schema_str}
</schema>

<scope>
{scope.description}
</scope>

<quota>{scope.quota}</quota>
{lineage_section}{notes_section}
---

Your goal: Master this domain so you can define what each row should be.

Key question: "Can I write {scope.quota} distinct seeds that will produce accurate, diverse, realistic rows?"

## Process

1. **Research** - Always do some research, even for simple domains. At minimum, search for inspiration and examples.

2. **Take notes** - Record facts, patterns, constraints via note().

3. **Conclude research** - Call conclude_research(summary) with a brief summary. REQUIRED before proceeding.

4. **Decide** - After concluding:
   - Space too broad? → breakdown(children)
   - Ready to define rows? → write_seeds() or extract_seeds()

## Seeds

Seeds anchor individual rows. Make them as complete as possible without overstepping.

Good seeds:
- Contain key facts/constraints for that specific row
- Are distinct from other seeds (no overlap)
- Don't predict things that need tool calls to determine

When sources have actual items → use extract_seeds()
When no direct examples exist → use write_seeds() with specific descriptions

## Tools

Research: brave_search(query), open(url_or_ref_id), find(ref_id, pattern), click(ref_id, link_id), interact(url, task)
Notes: note(content)
Conclude: conclude_research(summary) - REQUIRED before breakdown/seeding
Decide: breakdown(children), extract_seeds(ref_id, line_ranges), write_seeds(seeds)

Artifacts: Pages as p0, p1... Searches as s0, s1... Content is line-numbered (L0, L1...).

## Important

- Always research first, at least lightly
- Must call conclude_research() before breakdown or seeding
- Seeds should be complete enough for unique rows
- Breaking down = going deeper, only if needed for scope size
'''
    
    def get_total_cost(self) -> float:
        return self._total_cost
    
    def get_all_assignment_dirs(self) -> List[str]:
        return self._all_assignment_dirs.copy()