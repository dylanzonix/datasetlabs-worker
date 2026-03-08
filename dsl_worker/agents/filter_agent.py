"""
Filter agent — lightweight validation/classification of seeds.

V5: Filters run between seed yielding and row generation. They validate
seeds against criteria defined in the pipeline config.

Two modes:
- Simple: Single-turn cheap LLM call for classification (no tools)
- Judgment: Short-lived agent with research tools for validation
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from dsl_worker.agents.base import AgentConversation
from dsl_worker.agents.tools import ToolRegistry
from dsl_worker.billing.tracked_client import TrackedOpenAIClient
from dsl_worker.infra.pipeline import FilterConfig, Seed

logger = logging.getLogger(__name__)


class FilterAgent:
    """Lightweight agent that validates/classifies a seed."""

    @staticmethod
    async def run_simple(
        seed: Seed,
        filter_config: FilterConfig,
        openai_client: TrackedOpenAIClient,
        model: str,
    ) -> Tuple[bool, Dict, float]:
        """
        Single-turn filter. No tools, just classification.

        Returns (passed, findings, cost_usd).
        """
        prompt = (
            f"Evaluate this seed against the filter criteria.\n\n"
            f"Seed values: {json.dumps(seed.values, ensure_ascii=False)}\n"
            f"Seed metadata: {json.dumps(seed.metadata, ensure_ascii=False)}\n\n"
            f"Filter: {filter_config.name} — {filter_config.description}\n\n"
            f"Respond with ONLY a JSON object (no markdown, no explanation):\n"
            f'{{"pass": true/false, "reason": "...", "findings": {{...}}}}'
        )

        try:
            response, cost = await openai_client.responses_create(
                model=model,
                input=[{"role": "user", "content": prompt}],
                max_output_tokens=1000,
            )

            # Extract text from response
            text = ""
            for item in response.output:
                if item.type == "message":
                    for block in item.content:
                        if hasattr(block, "text"):
                            text += block.text

            # Parse JSON response
            text = text.strip()
            # Handle potential markdown code block wrapping
            if text.startswith("```"):
                text = text.split("\n", 1)[1] if "\n" in text else text[3:]
                if text.endswith("```"):
                    text = text[:-3]
                text = text.strip()

            result = json.loads(text)
            passed = bool(result.get("pass", True))
            findings = {
                "reason": result.get("reason", ""),
                **result.get("findings", {}),
            }

            return passed, findings, cost.total_cost_usd

        except json.JSONDecodeError:
            logger.warning(
                f"[FilterAgent] Failed to parse response for filter "
                f"'{filter_config.name}': {text[:200]}"
            )
            # On parse failure, pass the seed through
            return True, {"reason": "filter response unparseable, passing"}, 0.0
        except Exception as e:
            logger.error(f"[FilterAgent] Error in simple filter: {e}")
            # On error, pass the seed through
            return True, {"reason": f"filter error: {e}"}, 0.0

    @staticmethod
    async def run_judgment(
        seed: Seed,
        filter_config: FilterConfig,
        openai_client: TrackedOpenAIClient,
        model: str,
        workspace_dir: Path,
        brave_api_key: Optional[str] = None,
        sandbox: Optional[Any] = None,
        stop_checker: Optional[Callable[[], bool]] = None,
        blob_service_client: Optional[Any] = None,
        project_id: Optional[Any] = None,
        on_browser_started: Optional[Callable] = None,
    ) -> Tuple[bool, Dict, float]:
        """
        Multi-turn filter with research tools. For filters that need
        to look things up to validate a seed.

        Returns (passed, findings, cost_usd).
        """
        from dsl_worker.infra.research_tools import ResearchTools, ResearchScope

        impl = ResearchTools(
            workspace_dir=workspace_dir,
            schema=[],
            brave_api_key=brave_api_key,
            openai_client=openai_client,
            model=model,
            sandbox=sandbox,
            stop_checker=stop_checker,
            blob_service_client=blob_service_client,
            project_id=project_id,
            on_browser_started=on_browser_started,
        )
        impl.set_scope(ResearchScope(
            id=f"filter:{filter_config.name}",
            description="",
            quota=0,
        ))

        registry = ToolRegistry()
        impl.register_on(registry)

        # respond() tool for the filter to submit its verdict
        responded = False
        response_data: Dict = {}

        async def respond(args: Dict) -> tuple[str, float]:
            nonlocal responded, response_data
            responded = True
            response_data = {
                "pass": args.get("pass", True),
                "reason": args.get("reason", ""),
                "findings": args.get("findings", {}),
            }
            return "Verdict recorded.", 0.0

        registry.add(
            name="respond",
            description="Submit your filter verdict.",
            parameters={
                "type": "object",
                "properties": {
                    "pass": {
                        "type": "boolean",
                        "description": "Whether the seed passes the filter",
                    },
                    "reason": {
                        "type": "string",
                        "description": "Why it passed or failed",
                    },
                    "findings": {
                        "type": "object",
                        "description": "Any useful findings to pass to the row generator",
                    },
                },
                "required": ["pass", "reason"],
            },
            handler=respond,
        )

        system_prompt = (
            f"You are a filter agent validating a seed for dataset generation.\n\n"
            f"## Seed\n"
            f"Values: {json.dumps(seed.values, ensure_ascii=False)}\n"
            f"Metadata: {json.dumps(seed.metadata, ensure_ascii=False)}\n\n"
            f"## Filter Criteria\n"
            f"**{filter_config.name}**: {filter_config.description}\n\n"
            f"## Instructions\n"
            f"Research as needed to validate this seed against the criteria. "
            f"Then call respond() with your verdict.\n\n"
            f"Keep it brief — 1-2 searches max. The goal is quick validation, "
            f"not exhaustive research."
        )

        conversation = AgentConversation(
            openai_client=openai_client,
            model=model,
            system_prompt=system_prompt,
            tools=registry,
            stop_checker=stop_checker,
            max_turns=8,
            reasoning={"effort": "low", "summary": "auto"},
            label=f"filter:{filter_config.name}",
        )

        try:
            await conversation.send(
                "Validate this seed.",
                exit_condition=lambda: responded,
            )
        finally:
            await impl.cleanup()

        cost = conversation.total_cost

        if responded:
            passed = bool(response_data.get("pass", True))
            findings = {
                "reason": response_data.get("reason", ""),
                **response_data.get("findings", {}),
            }
            return passed, findings, cost

        # If filter didn't respond, pass the seed through
        logger.warning(
            f"[FilterAgent] Judgment filter '{filter_config.name}' "
            f"did not respond, passing seed through"
        )
        return True, {"reason": "filter did not respond"}, cost
