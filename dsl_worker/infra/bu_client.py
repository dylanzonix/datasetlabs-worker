"""
Browser Use V3 SDK client — single interface for all web interactions.

Uses browser_use_sdk.v3.AsyncBrowserUse which handles everything server-side:
browser sessions, CAPTCHA solving, proxies, model selection.

Two modes:
- extract(task) → (structured items, cost_usd) for harvesters
- research(task) → (plain text, cost_usd) for row generators, web_search_agent

Cost tracking: BU reports llm_cost_usd, proxy_cost_usd, browser_cost_usd per
session. We extract total_cost_usd and return it alongside results so it flows
through the existing on_cost callback chain.

Pause support: If stop_event is provided, BU calls are raced against it —
a pause request cancels the in-flight HTTP call immediately.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class ExtractedItem(BaseModel):
    """Generic extracted item — BU fills in whatever it finds."""
    data: Dict[str, Any] = Field(
        default_factory=dict,
        description="All extracted fields for this item (title, url, description, price, date, etc.)",
    )


class ExtractionResult(BaseModel):
    """Structured extraction output from BU."""
    items: List[ExtractedItem] = Field(
        default_factory=list,
        description="All items extracted from the page(s)",
    )


class BUClient:
    """
    Shared Browser Use V3 SDK client for all web interactions.

    Usage:
        client = BUClient(api_key="bu__xxx", model="bu-mini")

        # Extraction (harvesters)
        items, cost = await client.extract("Navigate to ... and extract all listings")

        # Research (row generators, orchestrator)
        text, cost = await client.research("Find the CEO of Acme Corp")
    """

    def __init__(
        self,
        api_key: str,
        model: str = "bu-mini",
        proxy_country: Optional[str] = "us",
        stop_event: Optional[asyncio.Event] = None,
    ) -> None:
        from browser_use_sdk.v3 import AsyncBrowserUse

        self._client = AsyncBrowserUse(api_key=api_key)
        self._model = model
        self._default_proxy = proxy_country
        self._stop_event = stop_event

    async def _run_cancellable(self, coro):
        """Run a coroutine, cancelling it if stop_event fires."""
        if not self._stop_event:
            return await coro

        task = asyncio.create_task(coro)
        stop_fut = asyncio.ensure_future(self._stop_event.wait())

        done, pending = await asyncio.wait(
            {task, stop_fut}, return_when=asyncio.FIRST_COMPLETED,
        )

        for p in pending:
            p.cancel()

        if task in done:
            return task.result()

        # Stop was requested — cancel the BU call
        raise asyncio.CancelledError("Stop requested during BU call")

    @staticmethod
    def _parse_cost(result) -> float:
        """Extract total cost USD from a BU SessionResult."""
        try:
            # SessionResult delegates to SessionResponse via __getattr__
            cost_str = getattr(result, "total_cost_usd", None)
            if cost_str is None and hasattr(result, "session"):
                cost_str = getattr(result.session, "total_cost_usd", None)
            return float(cost_str) if cost_str else 0.0
        except (ValueError, TypeError):
            return 0.0

    @staticmethod
    def _log_cost_breakdown(result, label: str) -> None:
        """Log detailed cost breakdown for debugging."""
        try:
            session = getattr(result, "session", result)
            llm = float(getattr(session, "llm_cost_usd", 0) or 0)
            proxy = float(getattr(session, "proxy_cost_usd", 0) or 0)
            browser = float(getattr(session, "browser_cost_usd", 0) or 0)
            total = float(getattr(session, "total_cost_usd", 0) or 0)
            steps = getattr(session, "step_count", 0) or 0
            logger.info(
                f"[BUClient] {label} cost: ${total:.4f} "
                f"(llm=${llm:.4f}, proxy=${proxy:.4f}, browser=${browser:.4f}, "
                f"{steps} steps)"
            )
        except Exception:
            pass

    async def extract(
        self,
        task: str,
        proxy_country: Optional[str] = None,
        profile_id: Optional[str] = None,
    ) -> Tuple[List[Dict[str, Any]], float]:
        """
        Extract structured items from a page. For harvesters.

        Returns (items, cost_usd) — each item is a dict of extracted fields.
        """
        try:
            result = await self._run_cancellable(
                self._client.run(
                    task,
                    model=self._model,
                    output_schema=ExtractionResult,
                    proxy_country_code=proxy_country or self._default_proxy,
                    profile_id=profile_id,
                )
            )
            cost = self._parse_cost(result)
            self._log_cost_breakdown(result, "extract")
            if result.output and hasattr(result.output, "items"):
                items = [item.data for item in result.output.items]
                return items, cost
            return [], cost
        except asyncio.CancelledError:
            logger.info("[BUClient] extract cancelled (stop requested)")
            return [], 0.0
        except Exception as e:
            logger.error(f"[BUClient] extract error: {e}")
            return [], 0.0

    async def research(
        self,
        task: str,
        proxy_country: Optional[str] = None,
        profile_id: Optional[str] = None,
    ) -> Tuple[str, float]:
        """
        Research a question on the web. For row generators and web_search_agent.

        Returns (text, cost_usd).
        """
        try:
            result = await self._run_cancellable(
                self._client.run(
                    task,
                    model=self._model,
                    proxy_country_code=proxy_country or self._default_proxy,
                    profile_id=profile_id,
                )
            )
            cost = self._parse_cost(result)
            self._log_cost_breakdown(result, "research")
            text = result.output or ""
            if not isinstance(text, str):
                text = str(text)
            return text, cost
        except asyncio.CancelledError:
            logger.info("[BUClient] research cancelled (stop requested)")
            return "", 0.0
        except Exception as e:
            logger.error(f"[BUClient] research error: {e}")
            return f"Research error: {e}", 0.0

    async def close(self) -> None:
        """Clean up the client."""
        try:
            await self._client.close()
        except Exception:
            pass
