"""
Scope Processor

Recursive processor that:
1. Researches a scope (becomes domain expert)
2. Calls conclude_research() to transition to decision mode
3. Decides: breakdown into smaller scopes OR create seeds

Children scopes run IN PARALLEL - each gets its own browser.
"""

import asyncio
import json
import logging
import random
import string
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from dsl_worker.phases.research_tools import ResearchTools, ResearchScope, ResearchState, Seed
from dsl_worker.config import settings

logger = logging.getLogger(__name__)

MAX_RESEARCH_TURNS = 500
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
    """
    Processes scopes recursively.
    
    Each scope gets its own ResearchTools instance with its own browser.
    Children scopes run in parallel.
    """
    
    def __init__(
        self,
        workspace_dir: Path,
        schema: List[Dict],
        project_instructions: str,
        openai_client: Any,
        brave_api_key: Optional[str] = None,
        sandbox: Optional[Any] = None,
        files_metadata: Optional[List[str]] = None,
        stop_checker: Optional[Callable[[], bool]] = None,
    ):
        self.workspace_dir = Path(workspace_dir)
        self.schema = schema
        self.project_instructions = project_instructions
        self.openai_client = openai_client
        self.brave_api_key = brave_api_key
        self.sandbox = sandbox
        self.files_metadata = files_metadata or []
        self.stop_checker = stop_checker
        
        self._total_cost = 0.0
        self._all_seeds: List[Seed] = []
    
    def _should_stop(self) -> bool:
        return self.stop_checker and self.stop_checker()
    
    async def process(self, scope: Scope) -> List[Seed]:
        """
        Process a scope recursively. Returns seeds for row generation.
        
        Each scope gets its own browser (via ResearchTools).
        Children scopes run in parallel.
        """
        if self._should_stop():
            return []
        
        logger.info(
            f"[ScopeProcessor] Processing: {scope.description[:80]}... "
            f"(quota={scope.quota}, depth={scope.depth})"
        )
        
        # Create tools for this scope - it will create its own browser
        tools = ResearchTools(
            workspace_dir=self.workspace_dir,
            schema=self.schema,
            brave_api_key=self.brave_api_key,
            openai_client=self.openai_client,
            model=settings.research_model,
            sandbox=self.sandbox,
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
        
        try:
            # Run research loop
            await self._research_loop(scope, tools)
            
            if self._should_stop():
                return tools.seeds
            
            # Check outcome
            if tools.breakdown_children:
                logger.info(
                    f"[ScopeProcessor] Breaking down into {len(tools.breakdown_children)} "
                    f"children (parallel)"
                )
                
                if scope.depth >= MAX_DEPTH:
                    logger.warning(
                        f"[ScopeProcessor] Max depth reached, forcing seed generation"
                    )
                    self._all_seeds.extend(tools.seeds)
                    return tools.seeds
                
                # Build child scopes
                total_weight = sum(c.get("weight", 1.0) for c in tools.breakdown_children)
                children = []
                
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
                    children.append(child)
                
                if self._should_stop():
                    return tools.seeds
                
                # Process all children IN PARALLEL
                # Each child will create its own ResearchTools with its own browser
                child_tasks = [self.process(child) for child in children]
                results = await asyncio.gather(*child_tasks, return_exceptions=True)
                
                # Collect seeds, handle any errors
                all_seeds = []
                for i, result in enumerate(results):
                    if isinstance(result, Exception):
                        logger.error(
                            f"[ScopeProcessor] Child scope failed: "
                            f"{children[i].description[:50]}... - {result}"
                        )
                    else:
                        all_seeds.extend(result)
                
                return all_seeds
            
            if tools.seeds:
                logger.info(f"[ScopeProcessor] Created {len(tools.seeds)} seeds")
                self._all_seeds.extend(tools.seeds)
                return tools.seeds
            
            logger.warning(
                f"[ScopeProcessor] No breakdown or seeds for: {scope.description[:60]}"
            )
            return []
            
        finally:
            # Always cleanup browser
            await tools.cleanup()
    
    async def _research_loop(self, scope: Scope, tools: ResearchTools):
        """Run research loop until agent decides to breakdown or generate."""
        
        system_prompt = self._build_system_prompt(scope, tools)
        
        messages = [{
            "role": "user",
            "content": "Research this scope. Your only tools right now are for research - searching, browsing, reading, and note-taking. Build understanding of this domain, then call conclude_research() when ready to move to the decision phase."
        }]
        
        phase_transitioned = False
        
        for turn in range(MAX_RESEARCH_TURNS):
            if self._should_stop():
                break
            
            # Exit if we've made a decision (breakdown or finished seeding)
            if tools.breakdown_children or tools.quota_filled or tools.is_done:
                break
            
            try:
                full_input = [{"role": "system", "content": system_prompt}] + messages
                
                response, cost = await self.openai_client.responses_create(
                    model=settings.research_model,
                    input=full_input,
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
                    # Add assistant message with text (if any)
                    if output_text:
                        messages.append({"role": "assistant", "content": output_text})
                    
                    # Process tool calls
                    for tc in tool_calls:
                        messages.append({
                            "type": "function_call",
                            "call_id": tc.call_id,
                            "name": tc.name,
                            "arguments": tc.arguments,
                        })
                        
                        name = tc.name
                        args = json.loads(tc.arguments)
                        
                        logger.info(f"[ScopeProcessor] Tool: {name}")
                        result, tool_cost = await tools.execute_tool(name, args)
                        self._total_cost += tool_cost
                        
                        messages.append({
                            "type": "function_call_output",
                            "call_id": tc.call_id,
                            "output": result[:15000],
                        })
                        
                        if tools.breakdown_children or tools.quota_filled or tools.is_done:
                            break
                        
                        # Detect phase transition after conclude_research
                        if name == "conclude_research" and tools.state == ResearchState.CONCLUDED and not phase_transitioned:
                            phase_transitioned = True
                            messages.append({
                                "role": "user",
                                "content": (
                                    f"Research complete. Your tools have changed - you now have: breakdown(), submit_seed(), and done().\n\n"
                                    f"Based on your research, decide:\n"
                                    f"- breakdown() if sub-areas need their own dedicated research agents\n"
                                    f"- submit_seed() to produce seeds from your understanding (quota: {tools.remaining_quota})\n"
                                    f"- done() if seeds are exhausted before quota"
                                )
                            })
                    
                    # Nudge to continue if still seeding
                    if (tools.seeds_submitted > 0 and 
                        not tools.quota_filled and 
                        not tools.is_done and 
                        not tools.breakdown_children):
                        messages.append({
                            "role": "user",
                            "content": (
                                f"Continue submitting seeds. You've submitted "
                                f"{tools.seeds_submitted}, quota remaining: "
                                f"{tools.remaining_quota}."
                            )
                        })
                else:
                    messages.append({"role": "assistant", "content": output_text or ""})
                    if not phase_transitioned:
                        messages.append({
                            "role": "user",
                            "content": (
                                "Continue researching. When you've built sufficient understanding, "
                                "call conclude_research() to move to the decision phase."
                            )
                        })
                    else:
                        messages.append({
                            "role": "user",
                            "content": (
                                "Use your tools to proceed: breakdown(), submit_seed(), or done()."
                            )
                        })
                pass
                
            except Exception as e:
                logger.error(f"[ScopeProcessor] Research error: {e}")
                messages.append({
                    "role": "user",
                    "content": f"Error: {e}. Continue or try different approach."
                })
        
        notes_count = len(tools.scope.notes) if tools.scope else 0
        logger.info(
            f"[ScopeProcessor] Research done. Turns: {turn + 1}, "
            f"Notes: {notes_count}, Seeds: {len(tools.seeds)}"
        )
    
    def _build_system_prompt(self, scope: Scope, tools: ResearchTools) -> str:
        """Build system prompt for research agent."""
        
        schema_lines = []
        for col in self.schema:
            schema_lines.append(
                f"- {col.get('name')} ({col.get('type', 'string')}): "
                f"{col.get('description', '')}"
            )
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
        
        # Files section
        files_section = ""
        if self.files_metadata:
            files_lines = [f"- {f}" for f in self.files_metadata]
            files_str = "\n".join(files_lines)
            files_section = f"""
<files>
{files_str}
</files>
"""
        
        # Scope description
        if scope.depth == 0:
            scope_description = "Root scope: entire dataset as defined in project instructions"
        else:
            scope_description = scope.description
        
        return f'''You are a research agent in a dataset generation pipeline.

## Why You Exist

You exist because your scope was identified as needing dedicated, thorough research to understand properly. That's what makes you a research agent - not that you search for things, but that you deeply understand a domain through research. Seeds are the output of that understanding.

## Key Concepts

**Scope** - The domain you're responsible for understanding. May be the entire dataset (root) or a narrower category (after breakdown).

**Seeds** - What you produce after understanding your scope. A seed is a specification that tells the generation agent what row to produce. Never write final row content yourself - that's the generation agent's job. A seed can be an extracted real-world example, a scenario description, a set of constraints, or a reference to source material. What matters is it gives the generation agent enough direction to produce one unique, high-quality row.

**Generation agent** - A separate agent that takes your seed and produces the actual row. It has its own tools (search, browsing, code execution). Your job is to set it up for success with a good seed, not to do its work.

**Notes** - Your thinking out loud. Use notes to reason progressively: what questions you're trying to answer, what you've learned, what new questions came up, how your understanding is evolving, what gaps remain. Notes are inherited by child scopes and generation agents.

**Quota** - Rows needed from your scope. Larger quotas mean more diversity is needed, which generally means more thorough research to understand the full breadth. If seeds are finite (real things that exist), you may exhaust them before quota - call done(). Don't invent seeds for things that don't exist.

## Two Phases

Your work happens in two distinct phases with different tools.

**Phase 1: Research** (current phase)
Your tools are for research only: searching, browsing, reading, and note-taking. No seeding or breakdown tools are available yet. Your job is to build genuine understanding of your scope's domain.

**Phase 2: Decision & Seeding** (after conclude_research)
Your tools change to: breakdown(), submit_seed(), done(). You use your research understanding to either break down into sub-scopes or produce seeds.

## Phase 1: How to Research

### Start broad

You need the full picture before going specific. What exists in this space? What are the major players, works, conventions? What does quality look like? What do real people in this domain actually care about?

Before any tool calls, think about what you need to understand and note your initial questions. Then research broadly to answer them. As you learn, answers will lead to new questions - that's expected and good.

Your first searches should establish fundamentals, not hunt for specific items. "What are the best cozy fantasy books" before "cozy fantasy bakery witch prompts." "What do real EV owners in India complain about" before "EV charging safety earthing RCCB."

### Go deep where it matters

Once you have the broad picture, go deeper into areas that are most important or that you understand least. Your research should be thorough enough that your seeds reflect real understanding, not surface-level pattern matching.

### Always research

Research is always valuable - it's why you exist. For factual domains, it prevents inaccuracy. For creative domains, it prevents mediocrity. Your implicit knowledge is generic compared to what real sources reveal about how people actually talk, what they actually care about, what authentic examples look like.

Never conclude "nothing to look up." There is always something to learn from sources about what good output looks like in your scope. Your implicit knowledge alone is not sufficient for any scope, no matter how familiar it seems.

### Look for high-signal sources

Prioritize sources that are authoritative, specific, and information-dense. Skip SEO fluff and shallow overviews. The best sources depend on the domain - use judgment.

### Think out loud

Use notes throughout research. Good notes evolve:
- Early: questions to answer, initial observations, the broad landscape
- Middle: patterns emerging, understanding deepening, new questions arising
- Late: coverage assessment, gaps identified, ready to transition

### When to conclude

Call conclude_research() when you have built sufficient understanding of your scope to make informed decisions about seeds. This means you should be able to explain what you learned from sources, not just restate what you already knew.

## Phase 2: The Decision

After conclude_research(), your tools change. You either break down or produce seeds.

**Break down when** you've discovered through research that a sub-area within your scope is substantial enough to need its own dedicated research agent. This means you understand it well enough to know it's important, how it fits within your scope, and roughly how much of the quota it deserves based on its relevance and prevalence. Don't break down things you haven't researched - you need to understand a topic's importance before delegating it.

Do NOT break down when:
- You already understand the scope well enough to produce good seeds. Breakdown is for when deeper research is needed, not for organizing output.
- You're just splitting quota across labels. That's busywork, not research.
- Quota is tiny (single digits). At that point just produce seeds directly.

**Produce seeds when** you understand the space well enough to specify what each row should be. Seeds should reflect real understanding from your research - grounded in examples, patterns, and domain knowledge you've built up.

## Research Tools (Phase 1)

brave_search(query), open(url_or_ref_id), find(ref_id, pattern), click(ref_id, link_id), interact(url, task), note(content), list_files(), code_exec(script), conclude_research(summary)

## Decision Tools (Phase 2, after conclude_research)

breakdown(children), submit_seed(ref_id, lines, content), done(reason), note(content), brave_search(query), open(url_or_ref_id)

Artifacts: Pages stored as p0, p1... Searches as s0, s1... Content is line-numbered (L0, L1...) for precise extraction.

<project_instructions>
{self.project_instructions}
</project_instructions>

<schema>
{schema_str}
</schema>

<scope>
{scope_description}
</scope>

<quota>
{scope.quota}
</quota>
{files_section}{lineage_section}{notes_section}'''
    
    def get_total_cost(self) -> float:
        return self._total_cost
    
    def get_all_seeds(self) -> List[Seed]:
        return self._all_seeds.copy()