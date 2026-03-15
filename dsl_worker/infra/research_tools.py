"""
Research Tools - ChatGPT-style browsing for research agent.

Each ResearchTools instance has its OWN cloud browser (one per scope).
Browser runs on Browser Use Cloud, controlled via Playwright CDP.
Lazy-initialized on first use, cleaned up when scope ends.

Tools:
- brave_search(query, response_length) → search results artifact
- open(ref_id_or_url, start_line, response_length) → page viewport
- find(ref_id, pattern, response_length) → matching lines  
- click(ref_id, link_id, response_length) → new page viewport
- note(content) → add to notes
- list_files(directory) → show available files with metadata
- code_exec(script) → execute Python with submit_seed()
- conclude_research(summary) → transition to decision mode
- breakdown(children) → split scope
- submit_seed(ref_id, lines, content) → create seed from source
- done(reason) → finish when seeds exhausted
- interact(url_or_ref_id, task) → browser agent for complex interactions
"""

import asyncio
import csv
import json
import logging
import random
import string
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import httpx
from markdownify import markdownify as md

# OTel tracing is optional
try:
    from opentelemetry import trace as _otel_trace
    from openinference.semconv.trace import SpanAttributes as _SpanAttributes

    def _get_tracer():
        return _otel_trace.get_tracer(__name__)
except ImportError:
    def _get_tracer():
        return None

from dsl_worker.infra.artifacts import (
    ArtifactStore,
    SearchResults,
    SearchResult,
    PageView,
    PageLink,
    RESPONSE_LENGTHS,
    extract_links_from_markdown,
    format_viewport,
    format_links_table,
    format_search_results,
    find_in_lines,
)

logger = logging.getLogger(__name__)

# Browser Use Cloud SDK
try:
    from browser_use_sdk import AsyncBrowserUse
    from playwright.async_api import async_playwright
    BROWSER_AVAILABLE = True
except ImportError:
    AsyncBrowserUse = None
    async_playwright = None
    BROWSER_AVAILABLE = False


def short_id(length: int = 6) -> str:
    """Generate short alphanumeric ID."""
    return ''.join(random.choices(string.ascii_lowercase + string.digits, k=length))


class ResearchState(Enum):
    """State machine for research process."""
    RESEARCHING = "researching"
    CONCLUDED = "concluded"


@dataclass
class Seed:
    """A seed for row generation."""
    content: str
    scope_id: str
    scope_description: str
    notes: List[str]
    research_summary: Optional[str] = None
    source_ref: Optional[str] = None
    source_url: Optional[str] = None
    bucket_id: Optional[str] = None


@dataclass
class ResearchScope:
    """Current research scope context."""
    id: str
    description: str
    quota: int
    depth: int = 0
    notes: List[str] = field(default_factory=list)
    parent_notes: List[str] = field(default_factory=list)


# =============================================================================
# dsl_tools library injected into sandbox
# =============================================================================

DSL_TOOLS_LIBRARY = '''
"""
DSL Tools - Available in code execution sandbox.

Usage:
    from dsl_tools import submit_seed, list_files, file_info
"""

import json
import os
from pathlib import Path

_WORKSPACE = "/workspace"
_SEEDS_FILE = os.path.join(_WORKSPACE, ".dsl_seeds.jsonl")


def submit_seed(content: str, source: str = None) -> None:
    """
    Submit a seed for row generation.
    
    Args:
        content: The seed content - what this row should be about
        source: Optional source reference (e.g., "data.pdf page 3")
    
    Example:
        submit_seed("Tesla Model 3", source="ev_list.csv row 15")
    """
    seed = {"content": content, "source": source}
    
    with open(_SEEDS_FILE, "a") as f:
        f.write(json.dumps(seed) + "\\n")
    
    print(f"[seed] {content[:60]}...")


def list_files(directory: str = "all") -> list:
    """List available files. Returns list of paths."""
    files = []
    
    if directory in ("uploads", "all"):
        uploads = Path(_WORKSPACE) / "uploads"
        if uploads.exists():
            files.extend([str(f) for f in uploads.iterdir() if f.is_file()])
    
    if directory in ("downloads", "all"):
        downloads = Path(_WORKSPACE) / "downloads"
        if downloads.exists():
            files.extend([str(f) for f in downloads.iterdir() if f.is_file()])
    
    return files


def file_info(path: str) -> dict:
    """Get file info: name, size_bytes, extension."""
    p = Path(path)
    if not p.exists():
        return {"exists": False}
    return {
        "exists": True,
        "name": p.name,
        "size_bytes": p.stat().st_size,
        "extension": p.suffix.lower(),
    }
'''


class ResearchTools:
    """
    Tools for research agent to explore and understand a domain.
    
    Each instance has its OWN browser (one per scope).
    
    State machine:
    - RESEARCHING: Can search, browse, take notes
    - CONCLUDED: Can breakdown or create seeds
    
    Must call conclude_research() before breakdown/seeding.
    """
    
    def __init__(
        self,
        workspace_dir: Path,
        schema: List[Dict],
        brave_api_key: Optional[str] = None,
        openai_client: Optional[Any] = None,
        model: str = "gpt-4o",
        sandbox: Optional[Any] = None,
        stop_checker: Optional[Callable[[], bool]] = None,
        blob_service_client: Optional[Any] = None,
        project_id: Optional[str] = None,
        uploaded_file_urls: Optional[Dict[str, str]] = None,
        on_browser_started: Optional[Callable] = None,
        on_browser_stopped: Optional[Callable] = None,
    ):
        self.workspace_dir = Path(workspace_dir)
        self.schema = schema
        self.brave_api_key = brave_api_key
        self.openai_client = openai_client
        self.model = model
        self.sandbox = sandbox
        self.stop_checker = stop_checker
        self.blob_service_client = blob_service_client
        self.project_id = str(project_id) if project_id else None
        self.uploaded_file_urls = uploaded_file_urls
        self._on_browser_started = on_browser_started
        self._on_browser_stopped = on_browser_stopped

        # Ensure downloads dir exists (uploads go direct to sandbox via SAS URLs)
        (self.workspace_dir / "downloads").mkdir(parents=True, exist_ok=True)

        # Browser Use Cloud - lazy initialized, owned by this instance
        self._browser_context: Optional[Any] = None  # Playwright BrowserContext
        self._browser_init_lock = asyncio.Lock()
        self._cloud_client: Optional[Any] = None      # AsyncBrowserUse SDK client
        self._cloud_session_id: Optional[str] = None   # Cloud browser session ID
        self._cdp_url: Optional[str] = None             # CDP URL for shared session
        self._live_url: Optional[str] = None            # Live debugging URL
        self._playwright: Optional[Any] = None          # Playwright instance
        self._cloud_files_seen: set = set()             # Track known remote files
        self._bu_agent: Optional[Any] = None             # Persistent Browser Use agent

        # Sandbox session - lazy initialized, owned by this instance
        self._sandbox_session: Optional[Any] = None
        self._sandbox_session_lock = asyncio.Lock()
        
        # Artifact storage
        self.artifacts = ArtifactStore()
        
        # State machine
        self.state = ResearchState.RESEARCHING
        self.research_summary: Optional[str] = None
        self.research_actions: int = 0  # Track searches/opens/reads
        
        # Current scope
        self.scope: Optional[ResearchScope] = None
        
        # Track breakdown
        self.breakdown_children: Optional[List[Dict]] = None
        
        # Track seeds (in-memory)
        self.seeds: List[Seed] = []
        self.seeds_submitted: int = 0
        self.is_done: bool = False
    
    def set_scope(self, scope: ResearchScope):
        """Set current scope being researched."""
        self.scope = scope
        self.breakdown_children = None
        self.state = ResearchState.RESEARCHING
        self.research_summary = None
        self.research_actions = 0
        self.seeds = []
        self.seeds_submitted = 0
        self.is_done = False
    
    def _should_stop(self) -> bool:
        return self.stop_checker and self.stop_checker()
    
    def _get_config(self, response_length: str) -> Dict[str, int]:
        return RESPONSE_LENGTHS.get(response_length, RESPONSE_LENGTHS["medium"])
    
    def _require_state(self, required: ResearchState, action: str) -> Optional[str]:
        """Check state requirement, return error message if not met."""
        if self.state != required:
            if required == ResearchState.CONCLUDED:
                return f"Cannot {action} yet. You must call conclude_research() first to summarize what you learned."
            else:
                return f"Cannot {action} in {self.state.value} state."
        return None
    
    # =========================================================================
    # Browser Management
    # =========================================================================
    
    async def _get_browser(self) -> Any:
        """Get or create a cloud browser session via Browser Use Cloud.

        Creates a cloud browser, connects Playwright via CDP, and injects
        cookies. Returns a Playwright BrowserContext for tab management.

        Uses a lock so parallel open() calls don't race through init.
        """
        if not BROWSER_AVAILABLE:
            raise RuntimeError("browser-use-sdk or playwright not installed")

        async with self._browser_init_lock:
            if self._browser_context is None:
                from dsl_worker.config import settings

                # Create cloud browser session via Browser Use SDK
                self._cloud_client = AsyncBrowserUse(
                    api_key=settings.browser_use_api_key,
                )
                cloud_browser = await self._cloud_client.browsers.create_browser_session(
                    proxy_country_code=settings.browser_use_proxy_country,
                )
                self._cloud_session_id = cloud_browser.id
                self._cdp_url = cloud_browser.cdp_url
                self._live_url = cloud_browser.live_url

                scope_id = self.scope.id if self.scope else "unknown"
                logger.info(
                    f"[ResearchTools] Cloud browser created for scope {scope_id}. "
                    f"Live URL: {self._live_url}"
                )

                # Notify caller about live URL (for progress_detail surfacing)
                if self._on_browser_started:
                    try:
                        result = self._on_browser_started(self._live_url, self._cloud_session_id)
                        if asyncio.iscoroutine(result):
                            await result
                    except Exception as e:
                        logger.warning(f"[ResearchTools] on_browser_started callback failed: {e}")

                # Connect Playwright to cloud browser via CDP
                self._playwright = await async_playwright().start()
                browser = await self._playwright.chromium.connect_over_cdp(
                    cloud_browser.cdp_url,
                )
                self._browser_context = browser.contexts[0]

                # Inject cookies from Azure Blob (global + per-project)
                if self.blob_service_client and self.project_id:
                    from dsl_worker.infra.cookie_manager import load_cookies
                    storage_state = load_cookies(
                        self.blob_service_client,
                        settings.azure_storage_container_name,
                        self.project_id,
                        settings.browser_global_cookies_blob_path,
                    )
                    if storage_state and storage_state.get("cookies"):
                        await self._browser_context.add_cookies(storage_state["cookies"])

        return self._browser_context
    
    # =========================================================================
    # Sandbox Session Management
    # =========================================================================

    async def _get_sandbox_session(self):
        """Get or create a persistent sandbox session for this instance.

        Lazy-creates a SandboxSession on first call, uploads workspace files
        once, and returns the same session for subsequent calls.
        """
        from dsl_worker.infra.sandbox import SandboxSession
        from sandbox_service.models import SessionConfig

        async with self._sandbox_session_lock:
            if self._sandbox_session is None:
                config = SessionConfig(network_enabled=True, memory_limit="4g")
                session_client = await self.sandbox.create_session(config)
                self._sandbox_session = SandboxSession(session_client, self.sandbox)
                await self._sandbox_session.upload_workspace(
                    self.workspace_dir,
                    file_urls=self.uploaded_file_urls,
                )
                scope_id = self.scope.id if self.scope else "unknown"
                logger.info(f"[ResearchTools] Sandbox session created for scope {scope_id}")

        return self._sandbox_session

    async def _disconnect_playwright(self):
        """Disconnect Playwright from the cloud browser.

        Used before BU Agent runs to avoid dual CDP connections.
        The cloud browser session stays alive — only the Playwright
        CDP connection is dropped.
        """
        if self._playwright:
            try:
                await self._playwright.stop()
            except Exception as e:
                logger.debug(f"[ResearchTools] Error disconnecting Playwright: {e}")
            self._playwright = None
            self._browser_context = None

    async def _reconnect_playwright(self):
        """Reconnect Playwright to the cloud browser after BU Agent finishes.

        Assumes the cloud browser session is still alive. If connection
        fails (e.g., browser died), nullifies state so _get_browser()
        will create a fresh session on next call.
        """
        if not self._cdp_url:
            return
        try:
            self._playwright = await async_playwright().start()
            browser = await self._playwright.chromium.connect_over_cdp(
                self._cdp_url,
            )
            self._browser_context = browser.contexts[0]
        except Exception as e:
            logger.warning(f"[ResearchTools] Failed to reconnect Playwright: {e}")
            # Browser session likely died — clear state so _get_browser()
            # creates a fresh session on next call
            self._playwright = None
            self._browser_context = None
            self._cdp_url = None
            self._cloud_session_id = None

    async def _stop_bu_cdp_client(self, agent: Any) -> None:
        """Force-stop the BU Agent's internal CDPClient.

        With keep_alive=True, agent.close() does NOT stop the CDPClient.
        We must reach in and stop it explicitly to avoid a lingering CDP
        WebSocket connection that conflicts with Playwright's.

        CRITICAL: Must set _intentional_stop=True BEFORE stopping the
        CDPClient. Otherwise BU's auto-reconnect callback fires when
        the WebSocket drops and reconnects the CDPClient within seconds.
        """
        try:
            session = getattr(agent, 'browser_session', None)
            if session is None:
                return
            # Prevent auto-reconnect (3 retries) when WebSocket drops
            session._intentional_stop = True
            cdp_client = getattr(session, '_cdp_client_root', None)
            if cdp_client is not None:
                await cdp_client.stop()
                logger.debug("[ResearchTools] BU CDPClient stopped")
        except Exception as e:
            logger.debug(f"[ResearchTools] Error stopping BU CDPClient: {e}")

    async def cleanup(self):
        """Cleanup browser, BU agent, and sandbox sessions."""
        # Cleanup BU agent — stop its CDPClient first, then close
        if self._bu_agent:
            try:
                await self._stop_bu_cdp_client(self._bu_agent)
                await self._bu_agent.close()
                logger.info("[ResearchTools] BU agent closed")
            except Exception as e:
                logger.warning(f"[ResearchTools] Error closing BU agent: {e}")
            self._bu_agent = None

        # Cleanup sandbox session
        if self._sandbox_session:
            try:
                await self._sandbox_session.close()
                scope_id = self.scope.id if self.scope else "unknown"
                logger.info(f"[ResearchTools] Sandbox session closed for scope {scope_id}")
            except Exception as e:
                logger.warning(f"[ResearchTools] Error closing sandbox session: {e}")
            self._sandbox_session = None

        # Cleanup cloud browser session
        if self._browser_context or self._cloud_session_id:
            # Save project cookies before stopping
            if self._browser_context:
                try:
                    if self.blob_service_client and self.project_id:
                        from dsl_worker.config import settings
                        from dsl_worker.infra.cookie_manager import save_project_cookies_from_context
                        try:
                            await save_project_cookies_from_context(
                                self._browser_context,
                                self.blob_service_client,
                                settings.azure_storage_container_name,
                                self.project_id,
                            )
                        except Exception as e:
                            logger.warning(f"[ResearchTools] Failed to save cookies: {e}")
                except Exception as e:
                    logger.warning(f"[ResearchTools] Error during cookie save: {e}")

            # Stop cloud browser session FIRST (before disconnecting Playwright)
            # so the API call goes through while we still have connectivity
            if self._cloud_client and self._cloud_session_id:
                stopped_session_id = self._cloud_session_id
                try:
                    await self._cloud_client.browsers.update_browser_session(
                        self._cloud_session_id, action="stop",
                    )
                    scope_id = self.scope.id if self.scope else "unknown"
                    logger.info(f"[ResearchTools] Cloud browser session stopped for scope {scope_id}")
                except Exception as e:
                    logger.warning(f"[ResearchTools] Error stopping cloud session: {e}")
                self._cloud_session_id = None
                if self._on_browser_stopped and stopped_session_id:
                    try:
                        result = self._on_browser_stopped(stopped_session_id)
                        if asyncio.iscoroutine(result):
                            await result
                    except Exception as e:
                        logger.warning(f"[ResearchTools] on_browser_stopped callback failed: {e}")

            # Close Playwright connection (cloud session already stopped above)
            try:
                if self._playwright:
                    await self._playwright.stop()
            except Exception as e:
                logger.warning(f"[ResearchTools] Error stopping Playwright: {e}")
            self._playwright = None
            self._browser_context = None
    
    def _is_browser_dead(self, error: Exception) -> bool:
        """Check if an exception indicates the browser session has died."""
        err_str = str(error).lower()
        dead_signals = [
            "target closed",
            "target detached",
            "connection closed",
            "browser has been closed",
            "cdp session closed",
            "websocket",
            "not found",
            "session closed",
        ]
        return any(signal in err_str for signal in dead_signals)

    async def _reset_browser_state(self):
        """Reset all browser state so _get_browser() creates a fresh session."""
        stopped_session_id = self._cloud_session_id
        try:
            if self._playwright:
                await self._playwright.stop()
        except Exception:
            pass
        self._playwright = None
        self._browser_context = None
        self._cdp_url = None
        self._cloud_session_id = None
        self._bu_agent = None
        if self._on_browser_stopped and stopped_session_id:
            try:
                result = self._on_browser_stopped(stopped_session_id)
                if asyncio.iscoroutine(result):
                    await result
            except Exception as e:
                logger.warning(f"[ResearchTools] on_browser_stopped callback failed: {e}")
        logger.info("[ResearchTools] Browser state reset — next operation will create fresh session")

    # =========================================================================
    # brave_search
    # =========================================================================
    
    async def brave_search(self, query: str, response_length: str = "medium") -> Tuple[str, float]:
        """Search the web using Brave Search API with retry."""
        if not self.brave_api_key:
            return "Error: Brave API key not configured", 0.0
        
        config = self._get_config(response_length)
        count = config["results"]
        
        max_retries = 3
        base_delay = 1.0
        
        for attempt in range(max_retries):
            try:
                async with httpx.AsyncClient() as client:
                    resp = await client.get(
                        "https://api.search.brave.com/res/v1/web/search",
                        params={"q": query, "count": count},
                        headers={"X-Subscription-Token": self.brave_api_key},
                        timeout=30.0,
                    )
                    resp.raise_for_status()
                    data = resp.json()
                
                results = []
                for i, r in enumerate(data.get("web", {}).get("results", [])):
                    results.append(SearchResult(
                        id=i,
                        title=r.get("title", ""),
                        url=r.get("url", ""),
                        snippet=r.get("description", ""),
                        date=r.get("age"),
                    ))
                
                artifact = SearchResults(query=query, results=results)
                ref_id = self.artifacts.store_search(artifact)
                
                return format_search_results(artifact, ref_id, count), 0.0
                
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 429 or e.response.status_code >= 500:
                    if attempt < max_retries - 1:
                        delay = base_delay * (2 ** attempt)
                        logger.warning(
                            f"[ResearchTools] Brave search failed ({e.response.status_code}), "
                            f"retry {attempt + 1}/{max_retries} in {delay}s"
                        )
                        await asyncio.sleep(delay)
                        continue
                logger.error(f"[ResearchTools] Brave search HTTP error: {e}")
                return f"Search error: {e}", 0.0
                
            except Exception as e:
                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt)
                    logger.warning(
                        f"[ResearchTools] Brave search failed ({e}), "
                        f"retry {attempt + 1}/{max_retries} in {delay}s"
                    )
                    await asyncio.sleep(delay)
                    continue
                logger.error(f"[ResearchTools] brave_search failed: {e}")
                return f"Search error: {e}", 0.0
        
        return "Search failed after retries", 0.0
    
    # =========================================================================
    # open
    # =========================================================================
    
    # Max time for a single open() call (navigation + content extraction)
    OPEN_TIMEOUT = 60

    async def open(
        self,
        ref_id_or_url: str,
        start_line: int = 0,
        response_length: str = "medium",
    ) -> Tuple[str, float]:
        """Open a URL or navigate within existing page artifact.

        Each call opens a NEW browser tab so multiple open() calls can run
        in parallel without fighting over the same page. Tabs are closed
        when done. All tabs share the same cloud browser session (shared cookies).
        """
        config = self._get_config(response_length)
        num_lines = config["lines"]

        artifact = self.artifacts.get(ref_id_or_url)

        if artifact:
            if isinstance(artifact, PageView):
                viewport = format_viewport(
                    artifact.lines, start_line, num_lines,
                    ref_id_or_url, artifact.url
                )
                links = format_links_table(artifact.links)
                return f"{viewport}\n{links}", 0.0

            elif isinstance(artifact, SearchResults):
                return format_search_results(artifact, ref_id_or_url, config["results"]), 0.0

            else:
                return f"Unknown artifact type for: {ref_id_or_url}", 0.0

        url = ref_id_or_url
        if not url.startswith(("http://", "https://")):
            url = "https://" + url

        for attempt in range(2):
            try:
                return await asyncio.wait_for(
                    self._open_in_new_tab(url, start_line, num_lines),
                    timeout=self.OPEN_TIMEOUT,
                )
            except asyncio.TimeoutError:
                logger.warning(f"[ResearchTools] open() timed out after {self.OPEN_TIMEOUT}s for {url}")
                return f"Timed out loading {url} after {self.OPEN_TIMEOUT}s", 0.0
            except Exception as e:
                if attempt == 0 and self._is_browser_dead(e):
                    logger.warning(
                        f"[ResearchTools] Browser session appears dead ({e}), "
                        "creating fresh session and retrying"
                    )
                    await self._reset_browser_state()
                    continue
                logger.error(f"[ResearchTools] open failed for {url}: {e}")
                return f"Failed to open {url}: {e}", 0.0
        return f"Failed to open {url} after session recovery", 0.0

    async def _get_cloud_files(self) -> set:
        """Get the set of file names currently on the cloud browser session."""
        if not self._cloud_client or not self._cloud_session_id:
            return set()
        try:
            files = await self._cloud_client.sessions.files(
                self._cloud_session_id, include_urls=True,
            )
            return {(f.name, getattr(f, 'url', None)) for f in files} if files else set()
        except Exception:
            return set()

    async def _download_remote_file(self, name: str, url: str) -> Path:
        """Download a file from cloud browser session to local workspace."""
        local_path = self.workspace_dir / "downloads" / name
        async with httpx.AsyncClient() as client:
            resp = await client.get(url, timeout=60.0)
            resp.raise_for_status()
            local_path.write_bytes(resp.content)
        logger.info(f"[ResearchTools] Downloaded remote file: {name} ({len(resp.content)} bytes)")
        return local_path

    async def _open_in_new_tab(
        self, url: str, start_line: int, num_lines: int,
    ) -> Tuple[str, float]:
        """Open a URL in a new browser tab, extract content, close tab.

        The browser runs on Browser Use Cloud. Playwright controls it via CDP.
        """
        context = await self._get_browser()
        page = None

        try:
            # Snapshot cloud files before navigation (for download detection)
            files_before = await self._get_cloud_files()

            # Create a new tab and navigate
            page = await context.new_page()
            try:
                await page.goto(url, wait_until="domcontentloaded", timeout=30000)
            except Exception as nav_error:
                # Timeout is expected for PDF downloads — continue and check
                logger.debug(f"[ResearchTools] Navigation exception (may be expected): {nav_error}")

            # Check for new downloads on cloud session
            await asyncio.sleep(1.0)
            files_after = await self._get_cloud_files()
            new_files = files_after - files_before

            if new_files:
                name, file_url = list(new_files)[0]
                if file_url:
                    local_path = await self._download_remote_file(name, file_url)
                    self._upload_download_to_blob(local_path)
                    return await self._handle_download(
                        {str(local_path)}, url, start_line, num_lines,
                    )

            # Try to extract HTML content from this tab
            markdown = ""
            max_attempts = 4

            for attempt in range(max_attempts):
                try:
                    html = await page.evaluate('() => document.body.innerHTML')
                    if html and len(html.strip()) > 0:
                        markdown = md(html, heading_style='ATX')
                        break
                except Exception as e:
                    logger.debug(f"[ResearchTools] HTML extraction attempt {attempt + 1} failed: {e}")
                await asyncio.sleep(1.5)

            if not markdown:
                # One more check for late downloads (e.g. PDF redirect)
                await asyncio.sleep(2.0)
                files_after = await self._get_cloud_files()
                new_files = files_after - files_before

                if new_files:
                    name, file_url = list(new_files)[0]
                    if file_url:
                        local_path = await self._download_remote_file(name, file_url)
                        self._upload_download_to_blob(local_path)
                        return await self._handle_download(
                            {str(local_path)}, url, start_line, num_lines,
                        )

                markdown = "Page loaded but no content extracted"

            lines = markdown.split('\n')
            links = extract_links_from_markdown(markdown, url)

            page_view = PageView(
                url=url,
                fetched_at=datetime.now(timezone.utc).isoformat(),
                lines=lines,
                total_lines=len(lines),
                links=links,
            )
            ref_id = self.artifacts.store_page(page_view)

            viewport = format_viewport(lines, start_line, num_lines, ref_id, url)
            links_table = format_links_table(links)

            return f"{viewport}\n{links_table}", 0.0

        finally:
            # Always close the tab to prevent leaks
            if page is not None:
                try:
                    await page.close()
                except Exception:
                    pass

    def _upload_download_to_blob(self, local_path: Path) -> Optional[str]:
        """Upload a downloaded file to Azure Blob for persistence. Returns blob path or None."""
        if not self.blob_service_client or not self.project_id:
            return None

        from dsl_worker.config import settings

        blob_path = f"projects/{self.project_id}/downloads/{local_path.name}"
        try:
            blob_client = self.blob_service_client.get_blob_client(
                container=settings.azure_storage_container_name,
                blob=blob_path,
            )
            with open(local_path, "rb") as f:
                blob_client.upload_blob(f, overwrite=True)
            logger.info(f"[ResearchTools] Uploaded download to blob: {blob_path}")
            return blob_path
        except Exception as e:
            logger.warning(f"[ResearchTools] Failed to upload download to blob: {e}")
            return None

    async def _handle_download(
        self,
        new_downloads: set,
        url: str,
        start_line: int,
        num_lines: int,
    ) -> Tuple[str, float]:
        """Process a downloaded file detected during open()."""
        downloaded_path = Path(list(new_downloads)[0])
        logger.info(f"[ResearchTools] File downloaded: {downloaded_path.name}")

        # Wait a bit for large files to finish writing
        await asyncio.sleep(1.0)

        # Upload to Azure Blob for durability (file also stays local for immediate use)
        self._upload_download_to_blob(downloaded_path)

        content, file_info = await self._extract_file_content(downloaded_path)

        if content:
            lines = content.split('\n')

            page_view = PageView(
                url=url,
                fetched_at=datetime.now(timezone.utc).isoformat(),
                lines=lines,
                total_lines=len(lines),
                links=[],
            )
            ref_id = self.artifacts.store_page(page_view)

            header = f"Downloaded: {downloaded_path.name} ({file_info})\n"
            header += f"Source: {url}\n"
            header += "=" * 50 + "\n"

            viewport = format_viewport(lines, start_line, num_lines, ref_id, url)
            return f"{header}{viewport}", 0.0
        else:
            size_kb = downloaded_path.stat().st_size / 1024
            return (
                f"File downloaded: {downloaded_path.name} ({size_kb:.1f}KB)\n"
                f"Location: /workspace/downloads/{downloaded_path.name}\n"
                f"Source: {url}\n\n"
                f"Could not extract text. Use code_exec() with appropriate library to process."
            ), 0.0
    
    async def _extract_file_content(self, file_path: Path) -> Tuple[str, str]:
        """
        Extract text content from a downloaded file.
        
        Returns (content, file_info) tuple.
        content is the extracted text, or empty string if extraction failed.
        file_info is a brief description like "PDF, 15 pages" or "245KB".
        """
        ext = file_path.suffix.lower()
        size_kb = file_path.stat().st_size / 1024
        
        try:
            if ext == '.pdf':
                try:
                    import pdfplumber
                    text_lines = []
                    page_count = 0
                    
                    with pdfplumber.open(file_path) as pdf:
                        page_count = len(pdf.pages)
                        for i, page in enumerate(pdf.pages):
                            text = page.extract_text() or ""
                            if text.strip():
                                text_lines.append(f"--- Page {i + 1} ---")
                                text_lines.extend(text.split('\n'))
                    
                    if text_lines:
                        return '\n'.join(text_lines), f"PDF, {page_count} pages"
                    else:
                        return "", f"PDF, {page_count} pages (no extractable text)"
                        
                except ImportError:
                    logger.warning("[ResearchTools] pdfplumber not installed")
                    return "", f"PDF, {size_kb:.1f}KB (pdfplumber not installed)"
            
            elif ext in ('.txt', '.md', '.csv', '.json', '.xml', '.html', '.htm'):
                content = file_path.read_text(errors='ignore')
                line_count = len(content.split('\n'))
                return content, f"{ext[1:].upper()}, {line_count} lines"
            
            elif ext in ('.xlsx', '.xls'):
                try:
                    import pandas as pd
                    
                    # Read all sheets
                    xlsx = pd.ExcelFile(file_path)
                    all_text = []
                    
                    for sheet_name in xlsx.sheet_names:
                        df = pd.read_excel(xlsx, sheet_name=sheet_name)
                        all_text.append(f"--- Sheet: {sheet_name} ---")
                        all_text.append(df.to_string())
                    
                    return '\n'.join(all_text), f"Excel, {len(xlsx.sheet_names)} sheets"
                    
                except ImportError:
                    return "", f"Excel, {size_kb:.1f}KB (pandas not installed)"
            
            elif ext == '.docx':
                try:
                    from docx import Document
                    doc = Document(file_path)
                    paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
                    return '\n\n'.join(paragraphs), f"Word doc, {len(paragraphs)} paragraphs"
                except ImportError:
                    return "", f"Word doc, {size_kb:.1f}KB (python-docx not installed)"
            
            else:
                # Unknown type - can't extract
                return "", f"{size_kb:.1f}KB"
                
        except Exception as e:
            logger.error(f"[ResearchTools] File extraction failed for {file_path}: {e}")
            return "", f"{size_kb:.1f}KB (extraction error: {e})"
    
    # =========================================================================
    # find
    # =========================================================================
    
    async def find(
        self,
        ref_id: str,
        pattern: str,
        response_length: str = "medium",
    ) -> Tuple[str, float]:
        """Find pattern in a page artifact."""
        page = self.artifacts.get_page(ref_id)
        if not page:
            return f"Page not found: {ref_id}", 0.0
        
        config = self._get_config(response_length)
        return find_in_lines(page.lines, pattern, config["matches"]), 0.0
    
    # =========================================================================
    # click
    # =========================================================================
    
    async def click(
        self,
        ref_id: str,
        link_id: int,
        response_length: str = "medium",
    ) -> Tuple[str, float]:
        """Click a link in a page artifact."""
        page = self.artifacts.get_page(ref_id)
        if not page:
            return f"Page not found: {ref_id}", 0.0
        
        link = None
        for l in page.links:
            if l.id == link_id:
                link = l
                break
        
        if not link:
            return f"Link {link_id} not found in {ref_id}. Available: 0-{len(page.links)-1}", 0.0
        
        return await self.open(link.href, start_line=0, response_length=response_length)
    
    # =========================================================================
    # note
    # =========================================================================
    
    def note(self, content: str) -> Tuple[str, float]:
        """Record a note about this domain."""
        if not self.scope:
            return "No active scope", 0.0
        
        self.scope.notes.append(content)
        return f"Noted ({len(self.scope.notes)} total)", 0.0
    
    # =========================================================================
    # list_files
    # =========================================================================
    
    async def list_files(self, directory: str = "all") -> Tuple[str, float]:
        """List available files with metadata preview."""
        output = []

        # Uploads: show from uploaded_file_urls keys (files are in sandbox, not on disk)
        if directory in ("uploads", "all") and self.uploaded_file_urls:
            output.append("📁 Uploads:")
            for filename in sorted(self.uploaded_file_urls.keys()):
                output.append(f"  {filename}")

        # Downloads: show from local disk (browser downloads)
        if directory in ("downloads", "all"):
            dir_path = self.workspace_dir / "downloads"
            if dir_path.exists():
                files = sorted([f for f in dir_path.iterdir() if f.is_file()])
                if files:
                    output.append("📥 Downloads:")
                    for f in files:
                        size_kb = f.stat().st_size / 1024
                        meta = self._get_file_metadata(f)
                        output.append(f"  {f.name} ({size_kb:.1f}KB)")
                        if meta:
                            output.append(f"    └─ {meta}")

        if not output:
            return "No files found. Upload files or browse web to download.", 0.0

        output.append("\nUse code_exec() to parse files and submit_seed() for items.")
        return '\n'.join(output), 0.0
    
    def _get_file_metadata(self, path: Path) -> str:
        """Quick metadata preview for common file types."""
        ext = path.suffix.lower()
        
        try:
            if ext == '.csv':
                with open(path, 'r', errors='ignore') as f:
                    reader = csv.reader(f)
                    header = next(reader, None)
                    row_count = sum(1 for _ in f) + 1
                if header:
                    cols = header[:5]
                    more = f" +{len(header)-5} more" if len(header) > 5 else ""
                    return f"Columns: {cols}{more} | ~{row_count} rows"
                return f"~{row_count} rows"
            
            elif ext == '.json':
                text = path.read_text(errors='ignore')[:10000]
                data = json.loads(text)
                if isinstance(data, list):
                    return f"Array with {len(data)} items"
                elif isinstance(data, dict):
                    keys = list(data.keys())[:5]
                    more = f" +{len(data)-5}" if len(data) > 5 else ""
                    return f"Object keys: {keys}{more}"
            
            elif ext in ('.xlsx', '.xls'):
                return "Excel spreadsheet"
            
            elif ext == '.pdf':
                return "PDF document"
            
            elif ext in ('.db', '.sqlite', '.sqlite3'):
                return "SQLite database"
            
            elif ext in ('.txt', '.md'):
                lines = len(path.read_text(errors='ignore').split('\n'))
                return f"~{lines} lines"
                
        except Exception:
            pass
        
        return ""
    
    # =========================================================================
    # code_exec
    # =========================================================================
    
    async def code_exec(self, script: str, description: str = "") -> Tuple[str, float]:
        """
        Execute Python code with access to files and submit_seed().

        Available in script:
        - submit_seed(content, source=None) - Submit a seed for row generation
        - list_files() - List available files
        - file_info(path) - Get file metadata

        Files are at:
        - /workspace/uploads/ - User uploaded files
        - /workspace/downloads/ - Browser downloaded files
        """
        if not self.sandbox:
            return "Code execution not available", 0.0

        tracer = _get_tracer()
        if tracer:
            with tracer.start_as_current_span(
                "tool:code_exec",
                attributes={_SpanAttributes.OPENINFERENCE_SPAN_KIND: "TOOL"},
            ) as span:
                span.set_attribute("input.value", str({"script": script[:500], "description": description})[:500])
                result_text, cost = await self._do_code_exec(script)
                span.set_attribute("output.value", result_text[:300])
                return result_text, cost
        return await self._do_code_exec(script)

    async def _do_code_exec(self, script: str) -> Tuple[str, float]:
        """Execute Python code (implementation)."""
        # Build code with dsl_tools injected
        full_script = f"""
{DSL_TOOLS_LIBRARY}

# Common libraries
try:
    import pandas as pd
except ImportError:
    pd = None

try:
    import pdfplumber
except ImportError:
    pdfplumber = None

try:
    from openpyxl import load_workbook
except ImportError:
    load_workbook = None

# Clear previous seeds file
import os
_seeds_path = os.path.join("/workspace", ".dsl_seeds.jsonl")
if os.path.exists(_seeds_path):
    os.remove(_seeds_path)

# User script
{script}
"""

        session = await self._get_sandbox_session()
        result = await session.execute(
            script=full_script,
            timeout=120,
        )

        # Read any seeds that were submitted (from sandbox filesystem)
        new_seeds = []
        try:
            seeds_content = await session.read_file(".dsl_seeds.jsonl")
            for line in seeds_content.strip().split('\n'):
                if line:
                    try:
                        seed_data = json.loads(line)
                        new_seeds.append(seed_data)
                    except json.JSONDecodeError:
                        pass
        except Exception:
            pass  # No seeds file = no seeds submitted

        # Add seeds to our collection
        for seed_data in new_seeds:
            self._add_seed_from_code(seed_data)

        # Build response
        output_parts = []

        if result.stdout:
            output_parts.append(result.stdout)

        if not result.success and result.stderr:
            output_parts.append(f"Error: {result.stderr}")

        if new_seeds:
            output_parts.append(f"\n✓ {len(new_seeds)} seeds submitted")
            remaining = self.scope.quota - self.seeds_submitted if self.scope else 0
            output_parts.append(f"  Remaining quota: {remaining}")

        return '\n'.join(output_parts) if output_parts else "Code executed (no output)", 0.0
    
    # =========================================================================
    # shell_exec
    # =========================================================================

    async def shell_exec(self, command: str, description: str = "") -> Tuple[str, float]:
        """Execute a shell command in the sandbox."""
        if not self.sandbox:
            return "Shell execution not available", 0.0

        tracer = _get_tracer()
        if tracer:
            with tracer.start_as_current_span(
                "tool:shell_exec",
                attributes={_SpanAttributes.OPENINFERENCE_SPAN_KIND: "TOOL"},
            ) as span:
                span.set_attribute("input.value", str({"command": command[:500], "description": description})[:500])
                result_text, cost = await self._do_shell_exec(command)
                span.set_attribute("output.value", result_text[:300])
                return result_text, cost
        return await self._do_shell_exec(command)

    async def _do_shell_exec(self, command: str) -> Tuple[str, float]:
        """Execute a shell command (implementation)."""
        session = await self._get_sandbox_session()
        result = await session.exec_shell(command=command, timeout=60)

        output_parts = []
        if result.stdout:
            output_parts.append(result.stdout)
        if not result.success and result.stderr:
            output_parts.append(f"Error: {result.stderr}")

        return '\n'.join(output_parts) if output_parts else "Command executed (no output)", 0.0

    def _add_seed_from_code(self, seed_data: Dict):
        """Add a seed that was submitted via code execution."""
        if not self.scope:
            return
        
        seed = Seed(
            content=seed_data.get("content", ""),
            scope_id=self.scope.id,
            scope_description=self.scope.description,
            notes=self.scope.parent_notes + self.scope.notes,
            research_summary=self.research_summary,
            source_ref=seed_data.get("source"),
            source_url=None,
        )
        self.seeds.append(seed)
        self.seeds_submitted += 1
    
    # =========================================================================
    # conclude_research
    # =========================================================================
    
    def conclude_research(self, summary: str) -> Tuple[str, float]:
        """
        Conclude research phase and transition to decision mode.
        
        Must be called before breakdown() or submit_seed().
        Requires actual research to have been performed.
        """
        if not self.scope:
            return "No active scope", 0.0
        
        if not summary or len(summary.strip()) < 10:
            return "Please provide a meaningful summary of your research findings.", 0.0
        
        if self.research_actions == 0:
            return (
                "Cannot conclude research without having done any. "
                "You need to use your research tools (brave_search, open, etc.) "
                "to build understanding before concluding. Start by searching broadly "
                "to understand your scope's domain."
            ), 0.0
        
        self.state = ResearchState.CONCLUDED
        self.research_summary = summary
        
        return (
            f"Research concluded. Summary recorded.\n\n"
            f"You can now either:\n"
            f"- breakdown(children) to split into smaller scopes\n"
            f"- submit_seed() to submit seeds for row generation\n"
            f"- done(reason) if seeds are exhausted before quota"
        ), 0.0
    
    # =========================================================================
    # breakdown
    # =========================================================================
    
    def breakdown(self, children: List[Dict]) -> Tuple[str, float]:
        """Break scope into smaller sub-scopes."""
        error = self._require_state(ResearchState.CONCLUDED, "breakdown")
        if error:
            return error, 0.0
        
        if not self.scope:
            return "No active scope", 0.0
        
        if not children:
            return "No children provided", 0.0
        
        self.breakdown_children = children
        
        total_weight = sum(c.get("weight", 1.0) for c in children)
        lines = [f"Breaking into {len(children)} sub-scopes:"]
        
        for c in children:
            weight = c.get("weight", 1.0) / total_weight
            approx_quota = int(self.scope.quota * weight)
            lines.append(f"  - {c.get('description', '?')} (~{approx_quota} rows)")
        
        return '\n'.join(lines), 0.0
    
    # =========================================================================
    # submit_seed
    # =========================================================================
    
    def submit_seed(
        self,
        ref_id: Optional[str] = None,
        lines: Optional[List[int]] = None,
        content: Optional[str] = None,
    ) -> Tuple[str, float]:
        """Submit a single seed for row generation."""
        error = self._require_state(ResearchState.CONCLUDED, "submit seed")
        if error:
            return error, 0.0
        
        if not self.scope:
            return "No active scope", 0.0
        
        # Build seed content
        seed_content = ""
        source_ref = None
        source_url = None
        
        # Extract from source if ref_id + lines provided
        if ref_id and lines and len(lines) == 2:
            page = self.artifacts.get_page(ref_id)
            if not page:
                return f"Page not found: {ref_id}", 0.0
            
            start, end = lines
            start = max(0, start)
            end = min(len(page.lines), end + 1)
            extracted = '\n'.join(page.lines[start:end]).strip()
            
            if extracted:
                seed_content = extracted
                source_ref = ref_id
                source_url = page.url
        
        # Add written content
        if content:
            if seed_content:
                seed_content = f"{seed_content}\n\n{content}"
            else:
                seed_content = content
        
        if not seed_content.strip():
            return "Empty seed - provide ref_id+lines and/or content", 0.0
        
        # Create seed
        all_notes = self.scope.parent_notes + self.scope.notes
        seed = Seed(
            content=seed_content,
            scope_id=self.scope.id,
            scope_description=self.scope.description,
            notes=all_notes,
            research_summary=self.research_summary,
            source_ref=source_ref,
            source_url=source_url,
        )
        self.seeds.append(seed)
        self.seeds_submitted += 1
        
        remaining = self.scope.quota - self.seeds_submitted
        
        if remaining > 0:
            return f"Seed {self.seeds_submitted} submitted | Remaining quota: {remaining}", 0.0
        else:
            return f"Seed {self.seeds_submitted} submitted | Quota filled!", 0.0
    
    # =========================================================================
    # done
    # =========================================================================
    
    def done(self, reason: str) -> Tuple[str, float]:
        """Finish when seeds are exhausted before reaching quota."""
        error = self._require_state(ResearchState.CONCLUDED, "done")
        if error:
            return error, 0.0
        
        if not self.scope:
            return "No active scope", 0.0
        
        if not reason or len(reason.strip()) < 5:
            return "Please provide a reason why seeds are exhausted", 0.0
        
        self.is_done = True
        remaining = self.scope.quota - self.seeds_submitted
        
        return f"Done. Submitted {self.seeds_submitted} seeds. Remaining {remaining} could not be filled. Reason: {reason}", 0.0
    
    @property
    def remaining_quota(self) -> int:
        """Get remaining quota for this scope."""
        if not self.scope:
            return 0
        return max(0, self.scope.quota - self.seeds_submitted)
    
    @property
    def quota_filled(self) -> bool:
        """Check if quota is filled."""
        return self.remaining_quota == 0
    
    # =========================================================================
    # interact (Browser Agent)
    # =========================================================================
    
    # BU system prompt extension — set once when the agent is created.
    # BU keeps its own navigation/DOM prompt; this just clarifies its role.
    _BU_SYSTEM_PROMPT = (
        "\n\n## Your Role\n\n"
        "You are a navigation assistant inside a larger system. A senior AI agent "
        "is controlling you — it can see the page directly and makes all decisions.\n\n"
        "Your job is simple: execute the specific navigation task you're given, then "
        "call done(). That's it.\n\n"
        "Rules:\n"
        "- Do exactly what's asked. No more, no less.\n"
        "- Do NOT summarize, describe, or extract page content — the controller "
        "sees the page and doesn't need your interpretation.\n"
        "- Do NOT research, explore links, or investigate beyond the task.\n"
        "- Do NOT make decisions about what to do next — you'll be given the next task.\n"
        "- Keep done() messages brief: \"Done — clicked Next\", \"Done — page loaded\", "
        "\"Failed — button not found\".\n"
    )

    async def _get_or_create_bu_agent(self, task: str) -> Any:
        """Create a fresh BU agent for an interact() call.

        A new agent is created each time — interact() cleans up after each
        call to avoid lingering CDPClient connections.
        """
        from browser_use import Agent
        from browser_use.browser.profile import BrowserProfile
        from browser_use.llm.browser_use import ChatBrowserUse

        if self._bu_agent is not None:
            # Reuse existing agent with new task
            self._bu_agent.add_new_task(task)
            return self._bu_agent

        # First call — create the agent
        from dsl_worker.config import settings
        import os
        os.environ.setdefault("BROWSER_USE_API_KEY", settings.browser_use_api_key)

        # Connect to our existing cloud browser. keep_alive=True so
        # agent.close() doesn't kill the browser — we manage that in cleanup().
        profile = BrowserProfile(
            cdp_url=self._cdp_url,
            keep_alive=True,
        )

        # Disable search action — BU should never navigate to Google/DuckDuckGo.
        # Keep extract enabled — BU may need it to see the page for navigation
        # decisions, but our agent always gets raw markdown regardless.
        try:
            from browser_use.tools.service import Tools as BUTools
            bu_tools = BUTools(exclude_actions=['search'])
        except ImportError:
            bu_tools = None

        async def should_stop():
            return bool(self.stop_checker and self.stop_checker())

        llm = ChatBrowserUse(model='bu-2-0')

        agent_kwargs = dict(
            task=task,
            llm=llm,
            browser_profile=profile,
            extend_system_message=self._BU_SYSTEM_PROMPT,
            register_should_stop_callback=should_stop,
            max_actions_per_step=3,
            use_judge=False,           # We decide success, not BU's judge
            directly_open_url=True,    # BU navigates to URL, triggering cloud captcha solver
        )
        if bu_tools is not None:
            agent_kwargs["tools"] = bu_tools

        self._bu_agent = Agent(**agent_kwargs)
        return self._bu_agent

    def _make_bu_supervisor(
        self, checkpoint_interval: int = 10, hard_ceiling: int = 50,
    ) -> Callable:
        """Create an on_step_end callback for BU supervision.

        Logs every step for visibility, plus safety checks.
        Collects step data in supervisor.steps for tracing.
        """
        _recent_urls: list = []
        scope_id = self.scope.id if self.scope else "unknown"

        async def on_step_end(agent) -> None:
            step = agent.state.n_steps
            step_time = time.time()

            # --- Extract step info for logging ---
            current_url = None
            last_action = None
            if agent.history and agent.history.history:
                last = agent.history.history[-1]
                if hasattr(last, 'state') and last.state:
                    current_url = last.state.url
                if hasattr(last, 'model_output') and last.model_output:
                    mo = last.model_output
                    if hasattr(mo, 'action') and mo.action:
                        actions = mo.action if isinstance(mo.action, list) else [mo.action]
                        action_names = []
                        for a in actions:
                            if hasattr(a, 'model_dump'):
                                d = a.model_dump(exclude_unset=True)
                                action_names.extend(d.keys())
                            elif isinstance(a, dict):
                                action_names.extend(a.keys())
                        last_action = ", ".join(action_names) if action_names else None

            short_url = current_url[:80] if current_url else "?"
            action_str = last_action or "?"
            logger.info(
                f"[BU {scope_id}] step {step}: {action_str} | {short_url}"
            )

            # Collect for tracing span
            on_step_end.steps.append({
                "step": step,
                "action": action_str,
                "url": short_url,
                "t": round(step_time - on_step_end.t0, 1),
            })

            _recent_urls.append(current_url)

            # --- Per-step checks ---

            if self.stop_checker and self.stop_checker():
                agent.state.stopped = True
                on_step_end.stop_reason = "external stop"
                logger.info(f"[BU {scope_id}] Stopped at step {step}: external stop")
                return

            if agent.state.consecutive_failures >= 3:
                agent.state.stopped = True
                on_step_end.stop_reason = f"{agent.state.consecutive_failures} consecutive failures"
                logger.info(
                    f"[BU {scope_id}] Stopped at step {step}: "
                    f"{agent.state.consecutive_failures} consecutive failures"
                )
                return

            # --- Hard ceiling ---
            if step >= hard_ceiling:
                agent.state.stopped = True
                on_step_end.stop_reason = f"hard ceiling ({hard_ceiling})"
                logger.info(f"[BU {scope_id}] Stopped at step {step}: hard ceiling ({hard_ceiling})")
                return

            # --- Checkpoint evaluation (every N steps) ---
            if step > 0 and step % checkpoint_interval == 0:
                window = _recent_urls[-checkpoint_interval:]
                unique_in_window = set(u for u in window if u)

                if len(unique_in_window) <= 1:
                    agent.state.stopped = True
                    on_step_end.stop_reason = f"same URL for {checkpoint_interval} steps"
                    logger.info(
                        f"[BU {scope_id}] Stopped at step {step}: "
                        f"same URL for {checkpoint_interval} steps"
                    )
                    return

                if len(unique_in_window) >= checkpoint_interval // 2:
                    agent.state.stopped = True
                    on_step_end.stop_reason = (
                        f"spiraling ({len(unique_in_window)} unique URLs "
                        f"in {checkpoint_interval} steps)"
                    )
                    logger.info(
                        f"[BU {scope_id}] Stopped at step {step}: "
                        f"spiraling ({len(unique_in_window)} unique URLs "
                        f"in {checkpoint_interval} steps)"
                    )
                    return

        # Attach data collectors to the callback
        on_step_end.steps = []
        on_step_end.t0 = time.time()
        on_step_end.stop_reason = None

        return on_step_end

    async def _extract_page_content(self, target_url: str) -> Tuple[Optional[str], Optional[str]]:
        """Extract page content from our Playwright connection after BU acts.

        Returns (formatted_content, ref_id) or (None, None) if extraction fails.
        """
        context = self._browser_context
        if not context or not context.pages:
            return None, None

        # Find the best page: prefer one matching target URL,
        # otherwise use the last page (most recently active/opened)
        page = context.pages[-1]
        for p in context.pages:
            try:
                p_url = p.url
                if target_url in p_url:
                    page = p
                    break
            except Exception:
                continue

        current_url = page.url

        html = ""
        for attempt in range(3):
            if self.stop_checker and self.stop_checker():
                break
            try:
                html = await page.evaluate('() => document.body.innerHTML')
                if html and len(html.strip()) > 100:
                    break
            except Exception:
                pass
            await asyncio.sleep(1.0)

        if not html or len(html.strip()) == 0:
            return None, None

        markdown = md(html, heading_style='ATX')
        lines = markdown.split('\n')
        links = extract_links_from_markdown(markdown, current_url)

        page_view = PageView(
            url=current_url,
            fetched_at=datetime.now(timezone.utc).isoformat(),
            lines=lines,
            total_lines=len(lines),
            links=links,
        )
        ref_id = self.artifacts.store_page(page_view)

        viewport = format_viewport(lines, 0, 150, ref_id, current_url)
        links_table = format_links_table(links)

        return f"{viewport}\n{links_table}", ref_id

    async def interact(self, url_or_ref_id: str, task: str) -> Tuple[str, float]:
        """
        Use Browser Use agent to perform an action on the shared cloud browser.

        interact() shares the same cloud browser session as open()/find()/click().
        A fresh BU agent is created per call to avoid lingering CDP connections.

        CRITICAL: Playwright and BU's CDPClient MUST NOT be connected to the
        same browser simultaneously. Both use Target.setAutoAttach which causes
        CDP target management conflicts, leading to "Target.detachedFromTarget"
        crashes. We disconnect Playwright before BU runs, then reconnect after.

        After BU finishes, we extract raw page content via our Playwright
        connection and return line-numbered markdown (same format as open()).
        """
        tracer = _get_tracer()
        if tracer:
            with tracer.start_as_current_span(
                "tool:interact",
                attributes={_SpanAttributes.OPENINFERENCE_SPAN_KIND: "TOOL"},
            ) as span:
                span.set_attribute("input.value", str({"task": task[:200], "url_or_ref_id": url_or_ref_id})[:500])
                result_text, cost = await self._do_interact(url_or_ref_id, task, span)
                return result_text, cost
        return await self._do_interact(url_or_ref_id, task, None)

    async def _do_interact(
        self, url_or_ref_id: str, task: str, span: Any,
    ) -> Tuple[str, float]:
        """interact() implementation — optionally traced via OTel span."""
        try:
            from browser_use import Agent
        except ImportError:
            return "browser-use not installed", 0.0

        # Resolve URL
        url = url_or_ref_id
        page_artifact = self.artifacts.get_page(url_or_ref_id)
        if page_artifact:
            url = page_artifact.url
        elif not url.startswith(("http://", "https://")):
            url = "https://" + url

        scope_id = self.scope.id if self.scope else "unknown"

        try:
            # Ensure shared cloud browser exists (reuses existing session)
            await self._get_browser()

            if not self._cdp_url:
                return "No cloud browser session available", 0.0

            # CRITICAL: Disconnect Playwright before BU runs.
            await self._disconnect_playwright()

            # Truncate task to prevent agents from embedding research
            # goals into BU's task.
            clean_task = task[:120].split("\n")[0]

            bu_task = (
                f"{clean_task}\n\n"
                f"Navigate to: {url}"
            )

            logger.info(
                f"[interact {scope_id}] START task={clean_task!r} url={url[:80]}"
            )
            t0 = time.time()

            bu_summary = "action failed"
            bu_steps = 0
            supervisor = None
            bu_history_steps = []
            try:
                agent = await self._get_or_create_bu_agent(bu_task)
                t_agent = time.time()

                supervisor = self._make_bu_supervisor()
                try:
                    result = await agent.run(max_steps=100, on_step_end=supervisor)
                    bu_summary = result.final_result() if result.final_result() else "action completed"
                    bu_steps = agent.state.n_steps
                except Exception as run_err:
                    logger.warning(f"[interact {scope_id}] BU run error (will still extract): {run_err}")
                    bu_summary = "action completed (with BU internal error)"
                    bu_steps = getattr(agent.state, 'n_steps', 0)

                t_run = time.time()

                # Extract BU's own per-step timing from agent.history
                try:
                    for h in agent.history.history:
                        step_info = {}
                        if h.metadata:
                            step_info["step"] = h.metadata.step_number
                            step_info["duration_s"] = round(h.metadata.duration_seconds, 1)
                        if h.state:
                            step_info["url"] = (h.state.url or "")[:100]
                        if h.model_output:
                            if h.model_output.action:
                                actions = h.model_output.action if isinstance(h.model_output.action, list) else [h.model_output.action]
                                names = []
                                for a in actions:
                                    if hasattr(a, 'model_dump'):
                                        names.extend(a.model_dump(exclude_unset=True).keys())
                                step_info["actions"] = names
                            if h.model_output.thinking:
                                step_info["thinking"] = h.model_output.thinking[:200]
                        if h.result:
                            for r in h.result:
                                if r.error:
                                    step_info["error"] = str(r.error)[:200]
                                if r.is_done:
                                    step_info["is_done"] = True
                        bu_history_steps.append(step_info)
                except Exception as hist_err:
                    logger.warning(f"[interact {scope_id}] History extraction error: {hist_err}")

                logger.info(
                    f"[interact {scope_id}] BU done: {bu_steps} steps, "
                    f"agent_create={t_agent - t0:.1f}s, "
                    f"run={t_run - t_agent:.1f}s, "
                    f"summary={bu_summary!r}"
                )
                for hs in bu_history_steps:
                    logger.info(f"[interact {scope_id}]   step {hs.get('step','?')}: "
                                f"{hs.get('duration_s','?')}s | {hs.get('actions',[])} | "
                                f"{hs.get('thinking','')[:80]}")
            finally:
                if self._bu_agent:
                    await self._stop_bu_cdp_client(self._bu_agent)
                    try:
                        await self._bu_agent.close()
                    except Exception:
                        pass
                    self._bu_agent = None

                await self._reconnect_playwright()

            # Extract raw page content via Playwright
            try:
                content, ref_id = await self._extract_page_content(url)
                t_extract = time.time()

                logger.info(
                    f"[interact {scope_id}] DONE total={t_extract - t0:.1f}s "
                    f"(agent={t_agent - t0:.1f}s + run={t_run - t_agent:.1f}s + "
                    f"reconnect+extract={t_extract - t_run:.1f}s) "
                    f"steps={bu_steps} content={'yes' if content else 'no'}"
                )

                if span:
                    span.set_attribute("output.value", str({
                        "summary": bu_summary[:200],
                        "step_count": bu_steps,
                        "total_seconds": round(t_extract - t0, 1),
                        "stop_reason": supervisor.stop_reason if supervisor else None,
                        "content_extracted": bool(content),
                    })[:1000])

                if content:
                    return (
                        f"Browser action: {bu_summary}\n\n"
                        f"{content}"
                    ), 0.0

                return f"Browser action: {bu_summary} (no page content extracted)", 0.0

            except Exception as extract_err:
                logger.warning(f"[interact {scope_id}] Extraction failed: {extract_err}")
                if span:
                    span.set_attribute("output.value", str({
                        "summary": bu_summary[:200],
                        "step_count": bu_steps,
                        "error": f"extraction failed: {extract_err}",
                    })[:1000])
                return f"Browser action: {bu_summary} (extraction failed: {extract_err})", 0.0

        except Exception as e:
            logger.error(f"[interact {scope_id}] Failed: {e}", exc_info=True)
            self._bu_agent = None
            if span:
                span.set_attribute("output.value", str({"error": str(e)})[:500])
            return f"Browser agent error: {e}", 0.0
    
    # =========================================================================
    # Tool Registration for Agent Framework
    # =========================================================================

    def register_on(self, registry: 'ToolRegistry', exclude: Optional[List[str]] = None) -> None:
        """
        Register browsing/research tools onto an agent ToolRegistry.

        Registers: brave_search, open, find, click, list_files, code_exec,
        interact. Skips scope-dependent tools (note, conclude_research,
        submit_seed, breakdown, done).

        Args:
            registry: The tool registry to add tools to.
            exclude: Optional list of tool names to skip (e.g. ["open", "find", "click"]).

        This is the single place adapter functions live — agents call this
        instead of duplicating the boilerplate.
        """
        exclude = set(exclude or [])
        # Adapter closures that translate args dict -> method calls
        handlers = {
            "brave_search": lambda args: self.brave_search(
                query=args.get("query", ""),
                response_length=args.get("response_length", "medium"),
            ),
            "open": lambda args: self.open(
                ref_id_or_url=args.get("ref_id_or_url", ""),
                start_line=args.get("start_line", 0),
                response_length=args.get("response_length", "medium"),
            ),
            "find": lambda args: self.find(
                ref_id=args.get("ref_id", ""),
                pattern=args.get("pattern", ""),
                response_length=args.get("response_length", "medium"),
            ),
            "click": lambda args: self.click(
                ref_id=args.get("ref_id", ""),
                link_id=args.get("link_id", 0),
                response_length=args.get("response_length", "medium"),
            ),
            "list_files": lambda args: self.list_files(
                directory=args.get("directory", "all"),
            ),
            "code_exec": lambda args: self.code_exec(
                script=args.get("script", ""),
                description=args.get("description", ""),
            ),
            "shell_exec": lambda args: self.shell_exec(
                command=args.get("command", ""),
                description=args.get("description", ""),
            ),
            "interact": lambda args: self.interact(
                url_or_ref_id=args.get("url_or_ref_id", ""),
                task=args.get("task", ""),
            ),
        }

        for defn in self._research_tools():
            name = defn.get("name")
            if name in handlers and name not in exclude:
                registry.add(
                    name=name,
                    description=defn.get("description", ""),
                    parameters=defn.get("parameters", {}),
                    handler=handlers[name],
                )

        # Add OpenAI native web search tool — the model can use this to read
        # web pages directly without going through Browser Use Cloud.
        registry.add_builtin({
            "type": "web_search_preview",
            "search_context_size": "medium",
        })

    # =========================================================================
    # Tool Definitions for LLM
    # =========================================================================

    def get_tool_definitions(self, phase: str = None) -> List[Dict]:
        """Get tool definitions for current or specified phase."""
        if phase is None:
            phase = "research" if self.state == ResearchState.RESEARCHING else "decision"
        
        if phase == "research":
            return self._research_tools()
        else:
            return self._decision_tools()
    
    def _research_tools(self) -> List[Dict]:
        """Tools available during research phase."""
        return [
            {
                "type": "function",
                "name": "brave_search",
                "description": "Search the web. Returns results with URLs you can open.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query"
                        },
                        "response_length": {
                            "type": "string",
                            "enum": ["short", "medium", "long"],
                            "description": "short=5, medium=10, long=20 results"
                        },
                    },
                    "required": ["query"]
                }
            },
            {
                "type": "function",
                "name": "open",
                "description": "Open a URL or view lines from existing page. Returns line-numbered content.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "ref_id_or_url": {
                            "type": "string",
                            "description": "URL to fetch, or ref_id (p0, p1...) to view existing"
                        },
                        "start_line": {
                            "type": "integer",
                            "description": "Line to start from (default: 0)"
                        },
                        "response_length": {
                            "type": "string",
                            "enum": ["short", "medium", "long"],
                            "description": "short=60, medium=150, long=300 lines"
                        },
                    },
                    "required": ["ref_id_or_url"]
                }
            },
            {
                "type": "function",
                "name": "find",
                "description": "Find text/pattern in a page. Returns matching lines with numbers.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "ref_id": {
                            "type": "string",
                            "description": "Page ref_id (p0, p1...)"
                        },
                        "pattern": {
                            "type": "string",
                            "description": "Text or regex to find"
                        },
                        "response_length": {
                            "type": "string",
                            "enum": ["short", "medium", "long"],
                            "description": "Controls max matches returned"
                        },
                    },
                    "required": ["ref_id", "pattern"]
                }
            },
            {
                "type": "function",
                "name": "click",
                "description": "Click a link in a page. Opens the linked URL.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "ref_id": {
                            "type": "string",
                            "description": "Page ref_id"
                        },
                        "link_id": {
                            "type": "integer",
                            "description": "Link ID from links table"
                        },
                        "response_length": {
                            "type": "string",
                            "enum": ["short", "medium", "long"],
                            "description": "Viewport size for new page"
                        },
                    },
                    "required": ["ref_id", "link_id"]
                }
            },
            {
                "type": "function",
                "name": "note",
                "description": "Record a note about this domain. Use freely for facts, patterns, constraints, observations.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "content": {
                            "type": "string",
                            "description": "What you observed or learned"
                        },
                    },
                    "required": ["content"]
                }
            },
            {
                "type": "function",
                "name": "list_files",
                "description": "List available files (uploads and downloads) with metadata preview.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "directory": {
                            "type": "string",
                            "enum": ["uploads", "downloads", "all"],
                            "description": "Which directory to list (default: all)"
                        }
                    }
                }
            },
            {
                "type": "function",
                "name": "code_exec",
                "description": """Execute Python code with file access.

Available:
- list_files(), file_info(path) - File utilities
- pandas, pdfplumber, openpyxl (if installed)

Files at /workspace/uploads/ and /workspace/downloads/""",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "script": {
                            "type": "string",
                            "description": "Python code to execute"
                        },
                        "description": {
                            "type": "string",
                            "description": "What this code does"
                        }
                    },
                    "required": ["script"]
                }
            },
            {
                "type": "function",
                "name": "shell_exec",
                "description": """Execute a shell command in the sandbox.

Use for system-level operations: installing packages, running CLI tools,
file manipulation, piping commands, etc.

Working directory is /workspace. Files at /workspace/uploads/ and /workspace/downloads/""",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {
                            "type": "string",
                            "description": "Shell command to execute"
                        },
                        "description": {
                            "type": "string",
                            "description": "What this command does"
                        }
                    },
                    "required": ["command"]
                }
            },
            {
                "type": "function",
                "name": "conclude_research",
                "description": "Conclude research phase and transition to decision phase. Summarize what you learned about the domain. After this, your tools will change to breakdown/seeding tools.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "summary": {
                            "type": "string",
                            "description": "Summary of research findings and domain understanding"
                        },
                    },
                    "required": ["summary"]
                }
            },
            {
                "type": "function",
                "name": "interact",
                "description": "Send a SHORT, ATOMIC action to a browser navigation agent. The agent is dumb — it only clicks, scrolls, and bypasses challenges. It shouldn't research, extract, or strategize. You see the page content yourself after it acts. ONLY use when you need to do an action on the website to navigate to content.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "url_or_ref_id": {
                            "type": "string",
                            "description": "URL or ref_id of the page"
                        },
                        "task": {
                            "type": "string",
                            "description": "One short action. Examples: 'bypass Cloudflare', 'click Next Page', 'scroll down', 'click Accept Cookies'. Do NOT include your research goal — the agent shouldn't help with that.",
                            "maxLength": 100
                        },
                    },
                    "required": ["url_or_ref_id", "task"]
                }
            },
        ]
    
    def _decision_tools(self) -> List[Dict]:
        """Tools available during decision/seeding phase."""
        return [
            {
                "type": "function",
                "name": "breakdown",
                "description": "Break scope into sub-scopes. Each child becomes its own research agent.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "children": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "description": {"type": "string"},
                                    "weight": {"type": "number"},
                                },
                                "required": ["description"]
                            },
                            "description": "Sub-scopes with descriptions and relative weights"
                        },
                    },
                    "required": ["children"]
                }
            },
            {
                "type": "function",
                "name": "submit_seed",
                "description": "Submit a seed for row generation. Provide source extraction (ref_id + lines), written content, or both.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "ref_id": {
                            "type": "string",
                            "description": "Page ref_id to extract from (optional)"
                        },
                        "lines": {
                            "type": "array",
                            "items": {"type": "integer"},
                            "minItems": 2,
                            "maxItems": 2,
                            "description": "[start, end] line range to extract (requires ref_id)"
                        },
                        "content": {
                            "type": "string",
                            "description": "Written seed content or additional context (optional)"
                        },
                    }
                }
            },
            {
                "type": "function",
                "name": "done",
                "description": "Finish when seeds are exhausted before reaching quota. Use when seeds are finite (real items that exist or don't) and you've found all that exist. Don't use to quit early.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "reason": {
                            "type": "string",
                            "description": "Why seeds are exhausted"
                        },
                    },
                    "required": ["reason"]
                }
            },
            {
                "type": "function",
                "name": "note",
                "description": "Record a note. Use to track coverage while seeding.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "content": {
                            "type": "string",
                            "description": "What you observed or decided"
                        },
                    },
                    "required": ["content"]
                }
            },
            {
                "type": "function",
                "name": "brave_search",
                "description": "Search the web. Still available if you need to check something while seeding.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query"
                        },
                        "response_length": {
                            "type": "string",
                            "enum": ["short", "medium", "long"],
                            "description": "short=5, medium=10, long=20 results"
                        },
                    },
                    "required": ["query"]
                }
            },
            {
                "type": "function",
                "name": "open",
                "description": "Open a URL or view lines from existing page. Still available for reference.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "ref_id_or_url": {
                            "type": "string",
                            "description": "URL to fetch, or ref_id (p0, p1...) to view existing"
                        },
                        "start_line": {
                            "type": "integer",
                            "description": "Line to start from (default: 0)"
                        },
                        "response_length": {
                            "type": "string",
                            "enum": ["short", "medium", "long"],
                            "description": "short=60, medium=150, long=300 lines"
                        },
                    },
                    "required": ["ref_id_or_url"]
                }
            },
        ]
    
    async def execute_tool(self, name: str, args: Dict) -> Tuple[str, float]:
        """Execute tool by name with args. Returns (result, cost)."""
        # Track research activity
        if name in ("brave_search", "open", "click", "find", "interact"):
            self.research_actions += 1
        
        try:
            if name == "brave_search":
                return await self.brave_search(
                    query=args.get("query", ""),
                    response_length=args.get("response_length", "medium"),
                )
            
            elif name == "open":
                return await self.open(
                    ref_id_or_url=args.get("ref_id_or_url", ""),
                    start_line=args.get("start_line", 0),
                    response_length=args.get("response_length", "medium"),
                )
            
            elif name == "find":
                return await self.find(
                    ref_id=args.get("ref_id", ""),
                    pattern=args.get("pattern", ""),
                    response_length=args.get("response_length", "medium"),
                )
            
            elif name == "click":
                return await self.click(
                    ref_id=args.get("ref_id", ""),
                    link_id=args.get("link_id", 0),
                    response_length=args.get("response_length", "medium"),
                )
            
            elif name == "note":
                return self.note(content=args.get("content", ""))
            
            elif name == "list_files":
                return await self.list_files(directory=args.get("directory", "all"))
            
            elif name == "code_exec":
                return await self.code_exec(
                    script=args.get("script", ""),
                    description=args.get("description", ""),
                )

            elif name == "shell_exec":
                return await self.shell_exec(
                    command=args.get("command", ""),
                    description=args.get("description", ""),
                )

            elif name == "conclude_research":
                return self.conclude_research(summary=args.get("summary", ""))
            
            elif name == "breakdown":
                return self.breakdown(children=args.get("children", []))
            
            elif name == "submit_seed":
                return self.submit_seed(
                    ref_id=args.get("ref_id"),
                    lines=args.get("lines"),
                    content=args.get("content"),
                )
            
            elif name == "done":
                return self.done(reason=args.get("reason", ""))
            
            elif name == "interact":
                return await self.interact(
                    url_or_ref_id=args.get("url_or_ref_id", ""),
                    task=args.get("task", ""),
                )
            
            else:
                return f"Unknown tool: {name}", 0.0
                
        except Exception as e:
            logger.error(f"[ResearchTools] {name} failed: {e}")
            return f"Tool error: {e}", 0.0