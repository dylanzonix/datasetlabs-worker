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


# Hard efficiency guard prepended to every BU task. BU's internal agent
# is prone to looping on the same page, exhaustively reading every
# element it sees, and bloating output with commentary — each behaviour
# costs LLM tokens and clock time. The wording is deliberate: short,
# imperative, and uses concrete numeric ceilings so the model can
# self-check against them. Tune the numbers here (not at the caller),
# because every caller — chat_v2 browser_use source, web_search_agent,
# the bulk_browser path — benefits equally from a tighter floor.
BU_EFFICIENCY_GUARD = (
    "EFFICIENCY RULES (read first, follow strictly):\n"
    "1. Hard budget: at most ~30 navigation actions and ~3 minutes per session. "
    "If the answer isn't in reach within that, STOP and return what you have.\n"
    "2. No looping. If the same action (scroll, click, search) didn't change the "
    "page in any useful way once, do NOT repeat it — try a different approach or stop.\n"
    "3. No exhaustive reading. Extract only the fields the task asks for. Don't "
    "open detail pages, don't read sidebars, don't summarize the page, don't "
    "narrate what you see. Skip cookie banners and chat widgets.\n"
    "4. Output is rows, not prose. Return ONLY the structured output. No analysis, "
    "no recap, no 'here is what I found'. Empty list is a valid answer.\n"
    "5. If the page blocks you (login wall, CAPTCHA you can't solve, hard 404, "
    "JS that won't render), stop immediately and return [] — do NOT try alternate "
    "URLs, do NOT search the web, do NOT improvise.\n"
    "6. One scope per session. The task below targets one page or one search. "
    "Don't expand scope, don't open external links, don't 'also check' anywhere "
    "else even if it seems related.\n\n"
    "--- TASK ---\n"
)


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

        # If the BU task completed, return its result even if stop also fired.
        # Don't throw away finished work.
        if task in done and not task.cancelled() and task.exception() is None:
            return task.result()

        # Stop fired and task isn't done — cancel it
        if stop_fut in done:
            task.cancel()
            try:
                await asyncio.wait({task}, timeout=3.0)
            except Exception:
                pass
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
        max_cost_usd: Optional[float] = None,
    ) -> Tuple[List[Dict[str, Any]], float, Optional[str], str]:
        """
        Extract structured items from a page. For harvesters.

        Returns (items, cost_usd, session_id, summary).
        summary contains BU's last step summary — useful for pagination/nav reports.
        Pass session_id from a previous call to reuse the same browser session.
        Set keep_alive=True to keep the session open for subsequent calls.
        max_cost_usd (if set) is passed to BU's SDK so it self-limits inside
        the session — used by the cell agent to enforce per-row credit caps.
        """
        try:
            run_kwargs = dict(
                model=self._model,
                output_schema=ExtractionResult,
                proxy_country_code=proxy_country or self._default_proxy,
                profile_id=profile_id,
                session_id=session_id,
                keep_alive=keep_alive,
            )
            if max_cost_usd is not None and max_cost_usd > 0:
                run_kwargs["max_cost_usd"] = max_cost_usd
            session_run = self._client.run(
                BU_EFFICIENCY_GUARD + task,
                **run_kwargs,
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
                if items:
                    return items, cost, sid, summary

            # SDK returned 0 items. For keep_alive sessions, BU may still be
            # updating the output after going idle. Re-poll the session to check.
            if sid and keep_alive:
                for retry in range(3):
                    await asyncio.sleep(2.0)
                    status_data = await self.get_session_status(sid)
                    if not status_data:
                        break
                    # Check raw output via API
                    import httpx
                    async with httpx.AsyncClient(timeout=5.0) as http:
                        resp = await http.get(
                            f"https://api.browser-use.com/api/v3/sessions/{sid}",
                            headers={"X-Browser-Use-API-Key": self._get_api_key()},
                        )
                        if resp.status_code == 200:
                            d = resp.json()
                            raw_output = d.get("output")
                            if isinstance(raw_output, dict):
                                raw_items = raw_output.get("items", [])
                                if raw_items:
                                    logger.info(
                                        f"[BUClient] Retry poll #{retry+1} recovered "
                                        f"{len(raw_items)} items from session {sid[:12]}"
                                    )
                                    try:
                                        parsed = ExtractionResult.model_validate(raw_output)
                                        items = [item.data for item in parsed.items]
                                        cost = float(d.get("totalCostUsd", 0) or 0)
                                        return items, cost, sid, summary
                                    except Exception:
                                        pass

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
                BU_EFFICIENCY_GUARD + task,
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

    def _get_api_key(self) -> str:
        """Extract the API key from the SDK's inner HTTP client."""
        return self._client._http._client.headers["x-browser-use-api-key"]

    async def get_session_status(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Poll a BU session for live status (step count, cost, last step).

        Returns dict with keys: status, step_count, total_cost_usd, llm_cost_usd,
        browser_cost_usd, last_step, title. Returns None on error.
        """
        try:
            import httpx
            async with httpx.AsyncClient(timeout=5.0) as http:
                resp = await http.get(
                    f"https://api.browser-use.com/api/v3/sessions/{session_id}",
                    headers={"X-Browser-Use-API-Key": self._get_api_key()},
                )
                if resp.status_code == 200:
                    d = resp.json()
                    return {
                        "status": d.get("status", "unknown"),
                        "step_count": d.get("stepCount", 0),
                        "total_cost_usd": float(d.get("totalCostUsd", 0) or 0),
                        "llm_cost_usd": float(d.get("llmCostUsd", 0) or 0),
                        "browser_cost_usd": float(d.get("browserCostUsd", 0) or 0),
                        "last_step": d.get("lastStepSummary") or "",
                        "title": d.get("title") or "",
                    }
                return None
        except Exception as e:
            logger.debug(f"[BUClient] session status poll failed: {e}")
            return None

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


# ---------------------------------------------------------------------------
# Module-level wrapper for chat_v2 source adapter
# ---------------------------------------------------------------------------


async def bu_extract_rows(
    url: str,
    task: str,
    candidate_description: str = "",
    *,
    max_cost_usd: Optional[float] = None,
) -> Tuple[List[Dict[str, Any]], float]:
    """Convenience wrapper used by `sources_v2/browser_use.py`.

    BU's native `extract` takes one composite task; this folds the
    starting URL and a candidate-description into the task so the
    caller only thinks about (url, task, candidate_description).
    Returns (rows, cost_usd).

    `max_cost_usd` (if set) is passed straight through to BU's SDK so
    BU self-limits inside its session — no further spend after the cap
    is hit. The cell agent uses this to enforce its per-row credit cap.
    """
    parts = [f"Navigate to {url}.", task.strip()]
    if candidate_description:
        parts.append(f"Each item should look like: {candidate_description}")
    composed_task = " ".join(parts)
    import os
    api_key = os.getenv("BROWSER_USE_API_KEY")
    if not api_key:
        logger.warning("BROWSER_USE_API_KEY not set — bu_extract_rows returning empty")
        return [], 0.0
    client = BUClient(
        api_key=api_key,
        proxy_country=os.getenv("BROWSER_USE_PROXY_COUNTRY", "us"),
    )
    try:
        items, cost, _sid, _summary = await client.extract(composed_task, max_cost_usd=max_cost_usd)
        return list(items or []), float(cost or 0.0)
    finally:
        await client.close()
