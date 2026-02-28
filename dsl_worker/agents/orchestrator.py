"""
Orchestrator agent — delegation layer that plans and delegates topic work.

V4: The orchestrator understands the request, does light research to figure
out how to slice the work, plans, and delegates:
1. Reads conversation history and uploaded files
2. Light research directly (search, browse, code) — just enough to plan
3. Plans: dataset brief, topics, targets
4. Delegates all topics in one delegate_topics() call
5. Done
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional

from dsl_worker.agents.base import AgentConversation, AgentResult
from dsl_worker.agents.tools import ToolRegistry
from dsl_worker.billing.tracked_client import TrackedOpenAIClient

logger = logging.getLogger(__name__)

READ_FILE_LIMIT = 30_000


ORCHESTRATOR_SYSTEM_PROMPT = """\
You are the orchestrator for a dataset generation system.

## Your Mission

A user described a dataset through a consultation chat (below). Your job is to understand
the request, plan how to slice it into topics, and delegate everything.

## How to Work

1. **Plan FIRST.** Call plan() as your very first action. You already have the conversation
   history, uploaded files, and schema — that's enough to plan from. Think about:
   - **Dataset type**: Is this extractive (rows are real entities to find), synthetic
     (rows are designed content), judgment (rows evaluate existing data), or hybrid?
   - **Where knowledge lives**: What sources are most authoritative for this domain?
     Uploaded files, specific websites, code repos, APIs, MCP servers, or the model's
     own knowledge? Don't go explore these yourself — reason about them and tell
     downstream agents where to look.
   - **Research approach**: What should topic agents focus on? What should row generators
     verify vs. trust? Should they go to specific platforms, read actual code, or is
     web search sufficient?
   - **Topics, targets, and briefings**: How to slice the work, how many rows each,
     and what each topic agent needs to know.

2. **Optional: quick lookup.** If your plan revealed something you're genuinely unsure
   about (e.g., you don't know what sub-areas exist for a niche domain), do ONE targeted
   search to fill the gap. But most of the time you can go straight to delegating. Do NOT
   browse extensively — topic agents and row generators do the real research.

3. **Delegate.** Call delegate_topics() with a dataset brief, topics, and targets.
   The system handles everything from here.

4. **Done.** Call done() immediately after delegating.

## Dataset Brief

The brief is written for row generators — it describes what kind of row to produce.
Unlike a template with {{variables}}, the brief describes the row holistically. Topic agents
will write specific row assignments that build on this brief.

A good brief has two parts:
1. **Row description** — what each row looks like, format, quality expectations
2. **Research approach** — how row generators should find and verify information

Example brief (extractive):
```
Generate a row profiling a real open-source Python library. Each row should include the
library name, description, primary use case, and a realistic code example.

Research approach: Look up the actual library on PyPI and its GitHub repo. Read the real
README and docs — don't guess at APIs. Verify the latest version number. Code examples
should be tested against the real library interface, not invented. Prefer official docs
over blog posts.
```

Example brief (synthetic):
```
Generate a single-turn expert Q&A about DayZ. A player asks a question and an expert
answers with specific, accurate game knowledge. The expert should sound like someone who
actually plays, not a wiki article.

Research approach: Ground your answers in real game mechanics. Look up actual stats,
crafting recipes, and game mechanics on the DayZ wiki or community resources like WOBO's
data sheets. Don't rely on what you know from training — game balance changes with patches.
Cross-check any specific numbers (damage values, spawn rates) against recent sources.
```

Example brief (judgment):
```
Score each user message for explicitness on a 0-1 scale with reasoning.

Research approach: No external research needed. Apply your judgment to the provided
message text. The value is in your reasoning, not in looking things up.
```

Rules:
- Written as a direct task for the row generator
- Describes the row format, quality expectations, and approach
- MUST include a "Research approach" section — this tells row generators where to look,
  what to verify, and when research is unnecessary
- Does NOT describe the schema — it's shown separately
- Does NOT include meta-instructions about the system

## Recognizing Dataset Types

Think about what kind of dataset this is. Different types need very different approaches.

**Extractive datasets** — rows describe real things that exist in the world.
- Examples: company profiles, library documentation, product listings, event histories
- Topic agents discover what entities exist and name specific ones in assignments
- Row generators must research each entity and report real facts — no fabrication
- Brief should emphasize: verify at primary sources, check freshness
- Topic briefings should say where to find entities (directories, registries, indexes)

**Synthetic datasets** — rows are invented/designed content.
- Examples: training conversations, Q&A pairs, creative writing, hypothetical scenarios
- Topic agents map the variation space and design diverse assignments
- Row generators create content — they may research to ground details in reality, but
  the content itself is original
- Brief should emphasize: diversity of angle/difficulty/style, realistic feel
- Topic briefings should describe what sub-areas to cover and what variety to aim for

**Judgment datasets** — rows evaluate, score, or classify existing data.
- Examples: content ratings, code quality assessments, annotation tasks
- Topic agents batch the data to evaluate (from uploaded files, MCP, or generated input)
- Row generators analyze what's presented — they rarely need external research
- Brief should emphasize: consistent criteria, rubric clarity
- Topic briefings should describe what data to pull from and evaluation criteria

**Hybrid datasets** — combine types. A codebase expert dataset is extractive (discover
what's in the code) + synthetic (design conversations about it). Note which parts
are which in the brief so row generators know when to research vs. create.

Tailor your brief, topics, and topic briefings based on the type.

## Topic Briefings

Each topic briefing guides a topic agent. Good briefings include:
- What this topic covers (scope boundaries)
- Where to find information or entities (specific sites, file paths, registries, MCP tools)
- What kind of diversity to aim for within this topic
- Whether this topic is extractive, synthetic, or judgment work

Example (extractive): "Cover Python web frameworks. Find real frameworks by checking
PyPI, GitHub trending, and awesome-python lists. Include popular and lesser-known ones.
Each assignment should name a specific real framework."

Example (synthetic): "Cover beginner-level Python questions. Design questions about
variables, loops, conditionals, functions. Mix styles: how-do-I, what's-wrong-with-this,
explain-the-difference. Aim for questions a first-week student would actually ask."

## Topics and Scale

Topics exist to divide large datasets into manageable chunks. Each topic agent holds its
entire area in context, ensuring diversity and avoiding duplicates within that chunk.

- For {num_samples} rows, create enough topics so each has ~10-30 rows
- Small datasets (< 20 rows): 1-2 topics is fine
- Don't fragment unnecessarily — a topic agent can handle a broad area
- **Topic names must be short, natural language labels** (e.g., "Getting Started",
  "Troubleshooting", "Advanced Workflows"). No numbering prefixes, no underscores,
  no code-style names.

## Resources

<resources>
{resources_section}
</resources>

## Column Schema

<schema>
{columns_description}
</schema>

## Conversation History

<conversation>
{conversation_summary}
</conversation>

## Tools

- brave_search(query): Search the web.
- open(ref_id_or_url): View a page or search result.
- find(ref_id, pattern): Search within a loaded page.
- click(ref_id, link_id): Follow a link.
- code_exec(script, description): Execute Python for data exploration.
- read_file(path): Read a workspace file.
- plan(strategy): Articulate your plan before delegating. Describe the brief, topics, reasoning.
- delegate_topics(dataset_brief, topics): Delegate all topics at once.
- done(reason): Signal orchestration is complete.

## What Happens After You Delegate

After you call delegate_topics(), the system runs each topic agent to produce sample rows,
then the user reviews them. If the user has feedback, the topic agents adjust. Then full
generation proceeds. You don't participate after delegation.

## Principles

- **You are a delegation layer.** Figure out the blueprint, don't do the work. Topic
  agents handle the details, row generators do the heavy lifting.
- **Research to delegate properly.** You need to know enough about the domain to create
  good topic areas. But always light — a quick search, reading an uploaded file.
- **Plan proportionally.** Simple, clear requests need minimal planning. Ambiguous or
  complex requests deserve more thought.
- **Think about research strategy.** Your most important contribution isn't just splitting
  rows into topics — it's telling downstream agents where to look and what to verify.
  A dataset about real Python libraries needs completely different research guidance than
  a dataset of synthetic customer support conversations.
- Target: {num_samples} rows total.

## Feedback Iterations

Sometimes you'll receive feedback about previously generated samples. Your initial message
will contain the previous plan and user feedback. Re-plan based on the feedback — adjust
the brief, topics, and targets as needed. The same tools and principles apply.
"""


class OrchestratorAgent:
    """
    V4 Orchestrator. Delegation layer — plans and delegates topic work.

    Usage:
        orchestrator = OrchestratorAgent(
            chat_history=[...],
            columns=[...],
            num_samples=100,
            on_delegate_topics=my_callback,
            openai_client=tracked_client,
            ...
        )
        await orchestrator.run()
    """

    def __init__(
        self,
        chat_history: List[Dict[str, str]],
        columns: List[Dict[str, Any]],
        num_samples: int,
        openai_client: TrackedOpenAIClient,
        model: str,
        workspace_dir: Path,
        on_delegate_topics: Callable[[Dict], Awaitable[Dict]],
        uploaded_files: Optional[List[Dict[str, Any]]] = None,
        brave_api_key: Optional[str] = None,
        sandbox: Optional[Any] = None,
        stop_checker: Optional[Callable[[], bool]] = None,
        cost_checker: Optional[Callable[[], tuple[bool, Optional[str]]]] = None,
        blob_service_client: Optional[Any] = None,
        project_id: Optional[Any] = None,
        on_tool_call: Optional[Callable[[str, str], None]] = None,
        on_cost: Optional[Callable] = None,
        uploaded_file_urls: Optional[Dict[str, str]] = None,
        mcp_tools: Optional[List[Dict[str, Any]]] = None,
        feedback_context: Optional[Dict[str, Any]] = None,
        langfuse_parent: Optional[Any] = None,
    ) -> None:
        self.feedback_context = feedback_context
        self.chat_history = chat_history
        self.columns = columns
        self.num_samples = num_samples
        self.workspace_dir = Path(workspace_dir)
        self.openai_client = openai_client
        self.model = model
        self.on_delegate_topics = on_delegate_topics
        self.brave_api_key = brave_api_key
        self.sandbox = sandbox
        self.stop_checker = stop_checker
        self.cost_checker = cost_checker
        self.blob_service_client = blob_service_client
        self.project_id = project_id
        self.on_tool_call = on_tool_call
        self.on_cost = on_cost
        self.uploaded_file_urls = uploaded_file_urls
        self.mcp_tools = mcp_tools or []

        # State
        self._is_done = False

        # Build tools
        registry = ToolRegistry()
        self._register_tools(registry)

        # Build system prompt
        columns_desc = self._format_columns()
        convo_summary = self._format_conversation()
        resources_section = self._format_resources(uploaded_files)

        system_prompt = ORCHESTRATOR_SYSTEM_PROMPT.format(
            num_samples=num_samples,
            columns_description=columns_desc,
            conversation_summary=convo_summary,
            resources_section=resources_section,
        )

        self._conversation = AgentConversation(
            openai_client=openai_client,
            model=model,
            system_prompt=system_prompt,
            tools=registry,
            stop_checker=stop_checker,
            max_turns=30,
            reasoning={"effort": "high", "summary": "detailed"},
            label="orchestrator",
            continue_on_text=True,
            on_tool_call=on_tool_call,
            on_cost=on_cost,
            extra_tools=self.mcp_tools,
            langfuse_parent=langfuse_parent,
        )

    def _format_columns(self) -> str:
        if not self.columns:
            return "(no columns defined)"
        return json.dumps(self.columns, indent=2)

    def _format_conversation(self) -> str:
        if not self.chat_history:
            return "(no conversation history)"

        parts = []
        for msg in self.chat_history:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            parts.append(f"**{role}**: {content}")
        return "\n\n".join(parts)

    def _format_resources(self, uploaded_files: Optional[List[Dict[str, Any]]]) -> str:
        lines = []
        if uploaded_files:
            lines.append("Uploaded files:")
            for f in uploaded_files:
                name = f.get("filename", "unknown")
                size = f.get("size_bytes", 0)
                ctype = f.get("content_type", "")
                if size > 1_000_000:
                    size_str = f"{size / 1_000_000:.1f} MB"
                elif size > 1_000:
                    size_str = f"{size / 1_000:.0f} KB"
                else:
                    size_str = f"{size} bytes"
                lines.append(f"  - {name} ({ctype}, {size_str})")
        else:
            lines.append("No uploaded files.")
        return "\n".join(lines)

    def _register_tools(self, registry: ToolRegistry) -> None:
        """Register orchestrator tools — research + delegation."""

        # --- Research tools (brave_search, open, find, click, code_exec, etc.) ---
        from dsl_worker.infra.research_tools import ResearchTools, ResearchScope

        self._impl = ResearchTools(
            workspace_dir=self.workspace_dir,
            schema=[],
            brave_api_key=self.brave_api_key,
            openai_client=self.openai_client,
            model=self.model,
            sandbox=self.sandbox,
            stop_checker=self.stop_checker,
            blob_service_client=self.blob_service_client,
            project_id=self.project_id,
            uploaded_file_urls=self.uploaded_file_urls,
        )
        self._impl.set_scope(ResearchScope(
            id="orchestrator",
            description="",
            quota=0,
        ))
        self._impl.register_on(registry)

        # --- read_file ---
        async def read_file(args: Dict) -> tuple[str, float]:
            path_str = args.get("path", "")
            try:
                path = Path(path_str)
                if not path.is_absolute():
                    candidate = self.workspace_dir / path
                    if not candidate.exists():
                        candidate = self.workspace_dir / "sources" / path
                    path = candidate

                if not path.exists():
                    return f"File not found: {path_str}", 0.0

                content = path.read_text(encoding="utf-8")
                if len(content) > READ_FILE_LIMIT:
                    content = content[:READ_FILE_LIMIT] + f"\n\n[Truncated at {READ_FILE_LIMIT} chars]"
                return content, 0.0
            except Exception as e:
                return f"Error reading file: {e}", 0.0

        registry.add(
            name="read_file",
            description="Read a file from the workspace (uploads, downloads, etc.).",
            parameters={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "File path (relative to workspace)",
                    },
                },
                "required": ["path"],
            },
            handler=read_file,
        )

        # --- plan ---
        async def plan(args: Dict) -> tuple[str, float]:
            strategy = args.get("strategy", "")
            return (
                "Plan recorded. Before you delegate, make sure your dataset brief "
                "includes a 'Research approach' section telling row generators where "
                "to look and what to verify. And include source guidance in each "
                "topic's briefing."
            ), 0.0

        registry.add(
            name="plan",
            description=(
                "Articulate your strategy before delegating. Describe the dataset "
                "brief, topics, and reasoning. This is your thinking step."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "strategy": {
                        "type": "string",
                        "description": (
                            "Your plan: what the dataset brief should say, what topics "
                            "to create, how many rows each, and your reasoning."
                        ),
                    },
                },
                "required": ["strategy"],
            },
            handler=plan,
        )

        # --- delegate_topics ---
        async def delegate_topics(args: Dict) -> tuple[str, float]:
            dataset_brief = args.get("dataset_brief", "")
            topics = args.get("topics", [])

            if not dataset_brief:
                return "Error: dataset_brief is required", 0.0
            if not topics:
                return "Error: at least one topic is required", 0.0

            # Validate topics
            errors = []
            total_target = 0
            for i, topic in enumerate(topics):
                if not isinstance(topic, dict):
                    errors.append(f"Topic {i}: must be an object")
                    continue
                if not topic.get("name"):
                    errors.append(f"Topic {i}: 'name' is required")
                if not topic.get("briefing"):
                    errors.append(f"Topic {i}: 'briefing' is required")
                target = topic.get("target", 10)
                total_target += target
            if errors:
                return "Validation errors:\n" + "\n".join(f"- {e}" for e in errors), 0.0

            # Normalize targets to match num_samples
            if total_target != self.num_samples and total_target > 0:
                ratio = self.num_samples / total_target
                for topic in topics:
                    topic["target"] = max(1, round(topic.get("target", 10) * ratio))
                # Adjust the last topic to hit exact target
                adjusted_total = sum(t.get("target", 10) for t in topics)
                if adjusted_total != self.num_samples:
                    topics[-1]["target"] += self.num_samples - adjusted_total

            # Dispatch via callback — the job processor handles spawning topic agents
            config = {
                "dataset_brief": dataset_brief,
                "topics": topics,
            }

            await self.on_delegate_topics(config)
            topic_count = len(topics)
            total = sum(t.get("target", 10) for t in topics)

            return (
                f"Delegated {topic_count} topics ({total} total target rows). "
                f"Topic agents will research, produce assignments, and dispatch row generators. "
                f"Your job is done — call done() now."
            ), 0.0

        registry.add(
            name="delegate_topics",
            description=(
                "Delegate all topics to topic agents in one call. "
                "Specify the dataset brief and a list of topics with names, "
                "targets, and briefings. Topic agents run in parallel."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "dataset_brief": {
                        "type": "string",
                        "description": (
                            "Natural language brief for row generators. Describes "
                            "what kind of row to produce — format, quality, approach."
                        ),
                    },
                    "topics": {
                        "type": "array",
                        "description": "Topics to delegate",
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": {
                                    "type": "string",
                                    "description": "Topic name",
                                },
                                "target": {
                                    "type": "integer",
                                    "description": "Target number of rows for this topic",
                                },
                                "briefing": {
                                    "type": "string",
                                    "description": (
                                        "What this topic covers. Guide the topic agent — "
                                        "what to research, what subtopics to cover."
                                    ),
                                },
                            },
                            "required": ["name", "target", "briefing"],
                        },
                    },
                },
                "required": ["dataset_brief", "topics"],
            },
            handler=delegate_topics,
        )

        # --- done ---
        async def done(args: Dict) -> tuple[str, float]:
            reason = args.get("reason", "complete")
            self._is_done = True
            return f"Orchestrator done: {reason}", 0.0

        registry.add(
            name="done",
            description="Signal that orchestration is complete. Call after delegate_topics.",
            parameters={
                "type": "object",
                "properties": {
                    "reason": {
                        "type": "string",
                        "description": "Why orchestration is done",
                    },
                },
            },
            handler=done,
        )

    async def run(self) -> AgentResult:
        """Run the orchestrator."""
        if self.feedback_context:
            prev = self.feedback_context["previous_config"]
            feedback = self.feedback_context["user_feedback"]

            topics_desc = "\n".join(
                f"- {t['name']} ({t.get('target', '?')} rows): {t['briefing']}"
                for t in prev.get("topics", [])
            )
            message = (
                f"FEEDBACK ITERATION: The user reviewed sample rows and gave feedback.\n\n"
                f"Previous dataset brief:\n{prev.get('dataset_brief', '')}\n\n"
                f"Previous topics:\n{topics_desc}\n\n"
                f"User feedback: \"{feedback}\"\n\n"
                f"The previous samples were discarded. Re-plan based on this feedback. "
                f"You can adjust the dataset brief, add/remove/modify topics, or change "
                f"targets. Then call delegate_topics() and done()."
            )
        else:
            message = (
                "Begin. Read the conversation history and uploaded files, then plan, "
                "delegate topics, and call done."
            )

        result = await self._conversation.send(
            message,
            exit_condition=lambda: self._is_done,
        )
        return result

    @property
    def cost_usd(self) -> float:
        """Total cost accumulated by the orchestrator."""
        return self._conversation.total_cost

    async def cleanup(self) -> None:
        """Clean up browser, sandbox, and other resources."""
        try:
            await self._impl.cleanup()
        except Exception as e:
            logger.warning(f"Orchestrator cleanup error: {e}")
