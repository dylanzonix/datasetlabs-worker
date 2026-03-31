"""
Browser Use V3 SDK client — single interface for all web interactions.

Uses browser_use_sdk.v3.AsyncBrowserUse which handles everything server-side:
browser sessions, CAPTCHA solving, proxies, model selection.

Three modes:
- extract(task) → (items, cost, session_id) for harvesters
- research(task) → (text, cost, session_id) for row generators, web_search_agent
- Session management: keep_alive sessions for multi-batch harvesting

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
        client = BUClient(api_key="bu__xxx", model="bu-max")

        # Extraction (harvesters)
        items, cost, sid, summary = await client.extract("Navigate to ... and extract all listings")

        # Research (row generators, orchestrator)
        text, cost = await client.research("Find the CEO of Acme Corp")
    """

    def __init__(
        self,
        api_key: str,
        model: str = "bu-max",
        proxy_country: Optional[str] = "us",
        stop_event: Optional[asyncio.Event] = None,
    ) -> None:
        from browser_use_sdk.v3 import AsyncBrowserUse

        self._client = AsyncBrowserUse(api_key=api_key)
        self._model = model
        self._default_proxy = proxy_country
        self._stop_event = stop_event

    async def _run_cancellable(self, session_run):
        """Run an AsyncSessionRun, cancelling it if stop_event fires.

        If stop fires, also stops the BU cloud session so it doesn't
        keep running (and costing money) after we've cancelled locally.
        """
        if not self._stop_event:
            return await session_run

        # Check before starting — skip the call entirely if already stopped
        if self._stop_event.is_set():
            raise asyncio.CancelledError("Stop already requested")

        # Wrap in a coroutine so create_task works with any awaitable
        async def _wrapper():
            return await session_run

        task = asyncio.create_task(_wrapper())
        stop_fut = asyncio.ensure_future(self._stop_event.wait())

        done, pending = await asyncio.wait(
            {task, stop_fut}, return_when=asyncio.FIRST_COMPLETED,
        )

        for p in pending:
            p.cancel()

        # Stop takes priority — if both finished, prefer stopping
        if stop_fut in done:
            task.cancel()
            # Wait briefly for the task to die (avoids orphaned coroutines)
            try:
                await asyncio.wait({task}, timeout=3.0)
            except Exception:
                pass
            # Stop the BU cloud session so it doesn't keep running
            sid = getattr(session_run, 'session_id', None)
            if not sid:
                sid = getattr(session_run, '_session_id', None)
            if sid:
                try:
                    await self._client.sessions.stop(sid)
                    logger.info(f"[BUClient] stopped cloud session {sid[:12]}... on pause")
                except Exception:
                    pass
            raise asyncio.CancelledError("Stop requested during BU call")

        return task.result()

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

    @staticmethod
    def _get_session_id(result) -> Optional[str]:
        """Extract session ID from a BU result for session reuse."""
        try:
            sid = getattr(result, "id", None)
            if sid:
                return str(sid)
            session = getattr(result, "session", None)
            if session:
                return str(getattr(session, "id", ""))
            return None
        except Exception:
            return None

    async def extract(
        self,
        task: str,
        proxy_country: Optional[str] = None,
        profile_id: Optional[str] = None,
        session_id: Optional[str] = None,
        keep_alive: bool = False,
        timeout: float = 900,
    ) -> Tuple[List[Dict[str, Any]], float, Optional[str], str]:
        """
        Extract structured items from a page. For harvesters.

        Returns (items, cost_usd, session_id, summary).
        summary contains BU's last step summary — useful for pagination/nav reports.
        Pass session_id from a previous call to reuse the same browser session.
        Set keep_alive=True to keep the session open for subsequent calls.
        """
        try:
            session_run = self._client.run(
                task,
                model=self._model,
                output_schema=ExtractionResult,
                proxy_country_code=proxy_country or self._default_proxy,
                profile_id=profile_id,
                session_id=session_id,
                keep_alive=keep_alive,
            )
            # SDK defaults to 14400s (4h) — override with our safety-net
            # timeout. SDK doesn't expose timeout via run(), so we set the
            # private attr directly.
            session_run._timeout = timeout
            result = await self._run_cancellable(session_run)
            cost = self._parse_cost(result)
            self._log_cost_breakdown(result, "extract")
            sid = self._get_session_id(result)
            # Extract the last step summary for pagination/nav reporting
            summary = ""
            try:
                session = getattr(result, "session", result)
                summary = getattr(session, "last_step_summary", "") or ""
                if not summary:
                    summary = getattr(session, "lastStepSummary", "") or ""
            except Exception:
                pass
            if result.output and hasattr(result.output, "items"):
                items = [item.data for item in result.output.items]
                return items, cost, sid, summary
            return [], cost, sid, summary
        except asyncio.CancelledError:
            logger.info("[BUClient] extract cancelled (stop requested)")
            return [], 0.0, session_id, ""
        except Exception as e:
            logger.error(f"[BUClient] extract error: {e}")
            return [], 0.0, session_id, ""

    async def research(
        self,
        task: str,
        proxy_country: Optional[str] = None,
        profile_id: Optional[str] = None,
        session_id: Optional[str] = None,
        keep_alive: bool = False,
        timeout: float = 900,
    ) -> Tuple[str, float, Optional[str]]:
        """
        Research a question on the web. For row generators and web_search_agent.

        Returns (text, cost_usd, session_id).
        """
        try:
            session_run = self._client.run(
                task,
                model=self._model,
                proxy_country_code=proxy_country or self._default_proxy,
                profile_id=profile_id,
                session_id=session_id,
                keep_alive=keep_alive,
            )
            # SDK defaults to 14400s (4h) — override with our safety-net
            session_run._timeout = timeout
            result = await self._run_cancellable(session_run)
            cost = self._parse_cost(result)
            self._log_cost_breakdown(result, "research")
            sid = self._get_session_id(result)
            text = result.output or ""
            if not isinstance(text, str):
                text = str(text)
            return text, cost, sid
        except asyncio.CancelledError:
            logger.info("[BUClient] research cancelled (stop requested)")
            return "", 0.0, session_id
        except Exception as e:
            logger.error(f"[BUClient] research error: {e}")
            return f"Research error: {e}", 0.0, session_id

    async def stop_session(self, session_id: str) -> None:
        """Stop a keep_alive session to free browser resources."""
        try:
            await self._client.sessions.stop(session_id)
            logger.info(f"[BUClient] stopped session {session_id[:12]}...")
        except Exception as e:
            logger.warning(f"[BUClient] failed to stop session {session_id[:12]}...: {e}")

    async def close(self) -> None:
        """Clean up the client."""
        try:
            await self._client.close()
        except Exception:
            pass
