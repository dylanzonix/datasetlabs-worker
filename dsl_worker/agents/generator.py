"""
Generator agent — iterates through a scope and yields seeds into a shared queue.

The orchestrator spawns one or more generators, each with a specific scope
to iterate through. The generator researches its scope and yields structured
seeds as it discovers them.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from dsl_worker.agents.base import AgentConversation, AgentResult
from dsl_worker.agents.tools import ToolRegistry
from dsl_worker.billing.tracked_client import TrackedOpenAIClient
from dsl_worker.phases.research_tools import ResearchTools, ResearchScope

logger = logging.getLogger(__name__)


GENERATOR_SYSTEM_PROMPT = """\
You are a seed extraction agent. Your job is to iterate through a scope and yield
raw text chunks as fast as possible. You are a rough extractor — the downstream
pipeline will clean up and transform your output.

## Your assignment

Scope: {scope}

Seed description: {seed_description}

Target: ~{target_count} seeds

## What seeds look like

Seeds are raw text — not structured JSON. Grab content directly from sources:
- A paragraph or section describing an item
- A table row or block of tabular data
- A formatted text block (lists, specs, profiles, descriptions)
- A line range from a page containing the relevant content

Each seed should contain enough context that a downstream agent can work with it.
Include source attribution when practical (URL, page title, section heading) as
part of the text.

## Strategy: speed over depth

You are NOT a research agent. Do not write summaries, analyze findings, or deliberate.
Your loop is: search → open source → identify line ranges → yield seeds → next source.

### How to yield seeds efficiently

**Always prefer line references over copying text.** When you open a page and see
relevant content, yield it by ref_id + line range instead of copying the text:

```
yield_seed(ref_id="p0", lines=[45, 120])  // Good — zero output tokens for content
yield_seed(text="<500 words copied>")      // Bad — expensive output tokens
```

The system resolves the text from the cached page automatically. This is ~10x cheaper.

### Workflow

1. **Open a source** — note the ref_id (e.g. "p0") and scan for relevant sections
2. **Identify line ranges** — each relevant item/section is a seed
3. **yield_seed(ref_id, lines)** for each item — one call per seed
4. **Use open(ref_id, start_line)** to scroll through long pages
5. **Move to next source** when current one is exhausted

### When to use text instead of line references

- Content from code_exec output (not in a page)
- Content you need to combine from multiple places
- Very short items where a line ref would be overhead

### Other techniques

- **Programmatic extraction**: When a source has >20 items in structured format (tables,
  lists, CSV, JSON), use code_exec to extract them all at once and yield each via text.
- **Multiple sources**: If one source doesn't cover your scope, search for more. Cast a
  wide net — directories, databases, lists, rankings, registries.

Seeds don't need to be perfect — rough and complete beats polished and sparse.

## Tools

- brave_search(query): Search the web
- open(ref_id_or_url, start_line): View a page (use start_line to jump to sections)
- find(ref_id, pattern): Search within a loaded page
- click(ref_id, link_id): Follow a link
- list_files(directory): List workspace files
- code_exec(script, description): Execute Python for bulk extraction
- interact(url_or_ref_id, task): Browser agent for JS-heavy pages
- yield_seed(ref_id, lines): Yield lines from a loaded page (preferred — cheap)
- yield_seed(text): Yield raw text (fallback — expensive, avoid when possible)

## When to stop

- You've reached or exceeded your target count (~{target_count} seeds)
- You've exhausted your scope (no more sources to extract from)
- Prefer to over-produce slightly rather than under-produce
"""


class GeneratorAgent:
    """
    Iterates through a scope and yields seeds into a shared async queue.

    Usage:
        queue = asyncio.Queue()
        gen = GeneratorAgent(
            openai_client=tracked_client,
            model="gpt-5.2",
            scope="Top 100 US universities by enrollment",
            seed_description="name, location, enrollment_count, type (public/private)",
            seed_queue=queue,
            workspace_dir=Path("/workspace"),
            brave_api_key="...",
        )

        # Run in background
        task = asyncio.create_task(gen.run())

        # Consume seeds as they arrive
        while True:
            seed = await queue.get()
            if seed is None:  # Poison pill = done
                break
            process(seed)
    """

    def __init__(
        self,
        openai_client: TrackedOpenAIClient,
        model: str,
        scope: str,
        seed_description: str,
        seed_queue: asyncio.Queue,
        workspace_dir: Path,
        generator_id: str = "generator",
        target_count: int = 100,
        bucket_id: Optional[str] = None,
        brave_api_key: Optional[str] = None,
        sandbox: Optional[Any] = None,
        stop_checker: Optional[Callable[[], bool]] = None,
        max_turns: int = 200,
        blob_service_client: Optional[Any] = None,
        project_id: Optional[Any] = None,
        on_tool_call: Optional[Callable[[str, str], None]] = None,
        uploaded_file_urls: Optional[Dict[str, str]] = None,
    ) -> None:
        self.scope = scope
        self.seed_description = seed_description
        self.seed_queue = seed_queue
        self.generator_id = generator_id
        self.target_count = target_count
        self.bucket_id = bucket_id
        self.seeds_yielded: int = 0
        self._stopped = False
        self.blob_service_client = blob_service_client
        self.project_id = project_id

        # Combine _stopped flag with external stop checker so gen.stop() works
        def combined_stop_checker():
            if self._stopped:
                return True
            return stop_checker() if stop_checker else False

        # Create ResearchTools directly (no need for a full ResearchAgent)
        self._impl = ResearchTools(
            workspace_dir=workspace_dir,
            schema=[],
            brave_api_key=brave_api_key,
            openai_client=openai_client,
            model=model,
            sandbox=sandbox,
            stop_checker=stop_checker,
            blob_service_client=blob_service_client,
            project_id=project_id,
            uploaded_file_urls=uploaded_file_urls,
        )
        self._impl.set_scope(ResearchScope(id="generator", description="", quota=0))

        # Build tools — browsing tools + yield_seed
        registry = ToolRegistry()
        self._impl.register_on(registry)
        self._register_yield_seed(registry)

        prompt = GENERATOR_SYSTEM_PROMPT.format(
            scope=scope,
            seed_description=seed_description,
            target_count=target_count,
        )

        self._conversation = AgentConversation(
            openai_client=openai_client,
            model=model,
            system_prompt=prompt,
            tools=registry,
            stop_checker=combined_stop_checker,
            max_turns=max_turns,
            reasoning={"effort": "medium", "summary": "detailed"},
            label=f"generator:{generator_id}",
            on_tool_call=on_tool_call,
        )

    def _register_yield_seed(self, registry: ToolRegistry) -> None:
        """Register the yield_seed tool (generator-specific)."""
        impl = self._impl

        async def yield_seed(args: Dict) -> tuple[str, float]:
            ref_id = args.get("ref_id")
            lines = args.get("lines")
            text = args.get("text", "")

            # Resolve text from page cache if ref_id + lines provided
            if ref_id and lines and len(lines) == 2:
                page = impl.artifacts.get_page(ref_id)
                if not page:
                    return f"Error: page '{ref_id}' not found in cache.", 0.0
                start, end = lines
                start = max(0, start)
                end = min(len(page.lines), end + 1)
                if start >= end:
                    return f"Error: invalid line range [{start}, {end}].", 0.0
                text = "\n".join(page.lines[start:end]).strip()
                if page.url:
                    text = f"Source: {page.url} (lines {start}-{end-1})\n\n{text}"

            if not text:
                return "Error: provide either 'text' or 'ref_id' + 'lines'.", 0.0

            # Wrap in envelope with bucket and generator metadata
            envelope = {
                "data": text,
                "bucket_id": self.bucket_id,
                "generator_id": self.generator_id,
            }

            await self.seed_queue.put(envelope)
            self.seeds_yielded += 1

            remaining = self.target_count - self.seeds_yielded
            if remaining <= 0:
                return (
                    f"Seed #{self.seeds_yielded} yielded. "
                    f"Target reached ({self.target_count}). You can stop now.",
                    0.0,
                )
            return f"Seed #{self.seeds_yielded} yielded. {remaining} remaining to target.", 0.0

        registry.add(
            name="yield_seed",
            description=(
                "Yield a seed. PREFERRED: use ref_id + lines to reference content "
                "from a loaded page (avoids copying text). Fallback: use text for "
                "content not from a page."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "ref_id": {
                        "type": "string",
                        "description": "Page ref_id (e.g. 'p0') — from open() or click()",
                    },
                    "lines": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": "[start_line, end_line] — inclusive range from the page",
                    },
                    "text": {
                        "type": "string",
                        "description": "Raw text content (only if not referencing a page)",
                    },
                },
            },
            handler=yield_seed,
        )

    async def run(self) -> AgentResult:
        """
        Run the generator until scope is exhausted or stopped.
        Sends a poison pill (None) to the queue when done.
        """
        try:
            result = await self._conversation.send(
                f"Begin extracting seeds from your scope. "
                f"Target: ~{self.target_count} seeds. Scope: {self.scope}"
            )
            return result
        finally:
            # Signal completion with poison pill
            await self.seed_queue.put(None)

    def stop(self) -> None:
        """Signal the generator to stop."""
        self._stopped = True

    @property
    def cost_usd(self) -> float:
        return self._conversation.total_cost

    async def cleanup(self) -> None:
        """Clean up resources."""
        try:
            await self._impl.cleanup()
        except Exception as e:
            logger.warning(f"Generator cleanup error: {e}")
