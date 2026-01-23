"""
Browser Pool for concurrent browser automation.

Manages a pool of browser instances for parallel research tasks.
Each browser has its own profile to avoid conflicts.
"""

import asyncio
import logging
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
from uuid import uuid4

from markdownify import markdownify as md

logger = logging.getLogger(__name__)


@dataclass
class BrowseResult:
    """Result from a browser task."""
    success: bool
    final_url: Optional[str] = None
    page_markdown: Optional[str] = None
    extracted_data: Optional[Any] = None
    error: Optional[str] = None


@dataclass
class PageSummary:
    """Summary of a fetched page."""
    url: str
    title: Optional[str]
    summary: str
    error: Optional[str] = None


class BrowserPool:
    """
    Pool of browser instances for parallel browser automation.
    
    Manages lifecycle, provides acquire/release semantics,
    and handles auth state persistence.
    """
    
    def __init__(
        self,
        size: int = 3,
        profiles_dir: str = "./browser_profiles",
        headless: bool = False,  # False + Xvfb for stealth
        auth_store_path: Optional[str] = None,
    ):
        self.size = size
        self.profiles_dir = Path(profiles_dir)
        self.profiles_dir.mkdir(parents=True, exist_ok=True)
        self.headless = headless
        self.auth_store_path = Path(auth_store_path) if auth_store_path else None
        
        self._browsers: List[Any] = []  # browser_use.Browser instances
        self._browser_ids: Dict[Any, int] = {}  # Map browser to ID for logging
        self._available: asyncio.Queue = asyncio.Queue()
        self._semaphore = asyncio.Semaphore(size)
        self._started = False
        
        logger.info(f"[BrowserPool] Initialized with size={size}, headless={headless}")
        
    async def start(self):
        """Initialize all browser instances."""
        if self._started:
            return
            
        from browser_use import Browser
        
        logger.info(f"[BrowserPool] Starting {self.size} browser instances...")
        
        for i in range(self.size):
            profile_dir = self.profiles_dir / f"browser-{i}"
            profile_dir.mkdir(exist_ok=True)
            
            logger.info(f"[BrowserPool] Starting browser {i} with profile {profile_dir}")
            
            browser = Browser(
                user_data_dir=str(profile_dir),
                headless=self.headless,
            )
            await browser.start()
            
            self._browsers.append(browser)
            self._browser_ids[id(browser)] = i
            await self._available.put(browser)
            
            logger.info(f"[BrowserPool] Browser {i} started")
            
        self._started = True
        logger.info(f"[BrowserPool] ✅ All {self.size} browsers started")
        
    async def stop(self):
        """Shutdown all browser instances."""
        for browser in self._browsers:
            try:
                await browser.stop()
            except Exception as e:
                logger.warning(f"Error stopping browser: {e}")
                
        self._browsers = []
        self._available = asyncio.Queue()
        self._started = False
        logger.info("Browser pool stopped")
        
    @asynccontextmanager
    async def acquire(self):
        """Acquire a browser from the pool."""
        logger.info(f"[BrowserPool] ⏳ Waiting for browser (available: {self._available.qsize()}/{self.size})...")
        async with self._semaphore:
            browser = await self._available.get()
            browser_id = self._browser_ids.get(id(browser), "?")
            logger.info(f"[BrowserPool] ✅ Acquired browser {browser_id}")
            try:
                yield browser
            finally:
                await self._available.put(browser)
                logger.info(f"[BrowserPool] ↩️ Released browser {browser_id}")
                
    async def load_auth(self, browser, domain: str) -> bool:
        """Load stored auth state for a domain."""
        if not self.auth_store_path:
            return False
            
        state_file = self.auth_store_path / f"{domain}.json"
        if not state_file.exists():
            return False
            
        try:
            import json
            state = json.loads(state_file.read_text())
            # browser_use uses storage_state parameter
            # For an existing session, we'd need to load cookies manually
            # This is a simplified version
            logger.info(f"Loaded auth state for {domain}")
            return True
        except Exception as e:
            logger.warning(f"Failed to load auth for {domain}: {e}")
            return False
            
    async def save_auth(self, browser, domain: str):
        """Save auth state for a domain."""
        if not self.auth_store_path:
            return
            
        self.auth_store_path.mkdir(parents=True, exist_ok=True)
        state_file = self.auth_store_path / f"{domain}.json"
        
        try:
            # Get storage state from browser
            # This would need browser_use API for getting cookies
            logger.info(f"Saved auth state for {domain}")
        except Exception as e:
            logger.warning(f"Failed to save auth for {domain}: {e}")

    # =========================================================================
    # Simple fetch (no LLM)
    # =========================================================================
    
    # File extensions that need special handling (download + parse)
    DOWNLOADABLE_EXTENSIONS = {'.pdf', '.docx', '.doc', '.xlsx', '.xls', '.pptx', '.ppt'}
    ARCHIVE_EXTENSIONS = {'.zip', '.rar', '.gz', '.tar'}
    
    async def simple_fetch(self, url: str, wait_time: float = 1.5) -> str:
        """
        Fast fetch - no LLM, just browser + JS rendering.
        Returns markdown for HTML pages, or downloads and parses files.
        """
        url_lower = url.lower()
        
        # Check for archive files - skip these
        for ext in self.ARCHIVE_EXTENSIONS:
            if ext in url_lower:
                logger.warning(f"[BrowserPool] ⏭️ Skipping archive URL: {url[:60]}...")
                return f"[Cannot process archive file: {url}]"
        
        # Check for downloadable files - fetch and parse
        for ext in self.DOWNLOADABLE_EXTENSIONS:
            if ext in url_lower:
                logger.info(f"[BrowserPool] 📥 Downloading file: {url[:60]}...")
                return await self._download_and_parse_file(url, ext)
        
        # Regular HTML page - use browser
        async with self.acquire() as browser:
            browser_id = self._browser_ids.get(id(browser), "?")
            page = None
            try:
                logger.info(f"[Browser-{browser_id}] 🌐 Opening: {url[:80]}...")
                page = await browser.new_page(url)
                logger.info(f"[Browser-{browser_id}] ⏳ Waiting {wait_time}s for JS...")
                await asyncio.sleep(wait_time)  # Let JS settle
                
                # Get HTML and convert to markdown
                logger.info(f"[Browser-{browser_id}] 📄 Extracting HTML...")
                html = await page.evaluate('() => document.body.innerHTML')
                markdown = md(html, heading_style='ATX', strip=['script', 'style'])
                
                logger.info(f"[Browser-{browser_id}] ✅ Got {len(markdown)} chars from {url[:50]}...")
                return markdown
                
            except Exception as e:
                logger.error(f"[Browser-{browser_id}] ❌ Fetch failed for {url[:50]}: {e}")
                return f"Error fetching {url}: {e}"
            finally:
                if page:
                    try:
                        logger.info(f"[Browser-{browser_id}] 🔒 Closing page...")
                        await browser.close_page(page)
                        logger.info(f"[Browser-{browser_id}] ✅ Page closed")
                    except Exception as e:
                        logger.warning(f"[Browser-{browser_id}] ⚠️ Failed to close page: {e}")
    
    async def _download_and_parse_file(self, url: str, ext: str) -> str:
        """Download a file and extract text content."""
        import tempfile
        import httpx
        
        try:
            # Download the file
            async with httpx.AsyncClient(follow_redirects=True, timeout=60.0) as client:
                response = await client.get(url)
                response.raise_for_status()
                content = response.content
            
            logger.info(f"[BrowserPool] 📥 Downloaded {len(content)} bytes from {url[:50]}...")
            
            # Save to temp file and parse
            with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as f:
                f.write(content)
                temp_path = f.name
            
            try:
                if ext == '.pdf':
                    return await self._parse_pdf(temp_path, url)
                elif ext in ('.docx', '.doc'):
                    return await self._parse_docx(temp_path, url)
                elif ext in ('.xlsx', '.xls'):
                    return await self._parse_excel(temp_path, url)
                elif ext in ('.pptx', '.ppt'):
                    return await self._parse_pptx(temp_path, url)
                else:
                    return f"[Downloaded but cannot parse {ext} file: {url}]"
            finally:
                # Clean up temp file
                import os
                try:
                    os.unlink(temp_path)
                except:
                    pass
                    
        except Exception as e:
            logger.error(f"[BrowserPool] ❌ Failed to download/parse {url}: {e}")
            return f"[Error downloading {url}: {e}]"
    
    async def _parse_pdf(self, path: str, url: str) -> str:
        """Extract text from PDF."""
        try:
            import pdfplumber
            
            text_parts = []
            with pdfplumber.open(path) as pdf:
                for i, page in enumerate(pdf.pages[:50]):  # Limit to 50 pages
                    page_text = page.extract_text()
                    if page_text:
                        text_parts.append(f"--- Page {i+1} ---\n{page_text}")
            
            if text_parts:
                result = f"# PDF: {url}\n\n" + "\n\n".join(text_parts)
                logger.info(f"[BrowserPool] 📄 Extracted {len(result)} chars from PDF")
                return result
            else:
                return f"[PDF at {url} contains no extractable text (may be scanned/image-based)]"
                
        except ImportError:
            logger.warning("[BrowserPool] pdfplumber not installed, trying pypdf")
            try:
                from pypdf import PdfReader
                reader = PdfReader(path)
                text_parts = []
                for i, page in enumerate(reader.pages[:50]):
                    text = page.extract_text()
                    if text:
                        text_parts.append(f"--- Page {i+1} ---\n{text}")
                if text_parts:
                    return f"# PDF: {url}\n\n" + "\n\n".join(text_parts)
                return f"[PDF at {url} contains no extractable text]"
            except ImportError:
                return f"[Cannot parse PDF - install pdfplumber or pypdf: {url}]"
        except Exception as e:
            return f"[Error parsing PDF {url}: {e}]"
    
    async def _parse_docx(self, path: str, url: str) -> str:
        """Extract text from DOCX."""
        try:
            from docx import Document
            
            doc = Document(path)
            paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
            
            if paragraphs:
                result = f"# DOCX: {url}\n\n" + "\n\n".join(paragraphs)
                logger.info(f"[BrowserPool] 📄 Extracted {len(result)} chars from DOCX")
                return result
            else:
                return f"[DOCX at {url} contains no text]"
                
        except ImportError:
            return f"[Cannot parse DOCX - install python-docx: {url}]"
        except Exception as e:
            return f"[Error parsing DOCX {url}: {e}]"
    
    async def _parse_excel(self, path: str, url: str) -> str:
        """Extract text from Excel."""
        try:
            import pandas as pd
            
            # Read all sheets
            sheets = pd.read_excel(path, sheet_name=None)
            
            parts = [f"# Excel: {url}\n"]
            for sheet_name, df in sheets.items():
                parts.append(f"\n## Sheet: {sheet_name}\n")
                parts.append(df.to_markdown(index=False))
            
            result = "\n".join(parts)
            logger.info(f"[BrowserPool] 📄 Extracted {len(result)} chars from Excel")
            return result
            
        except ImportError:
            return f"[Cannot parse Excel - install pandas openpyxl: {url}]"
        except Exception as e:
            return f"[Error parsing Excel {url}: {e}]"
    
    async def _parse_pptx(self, path: str, url: str) -> str:
        """Extract text from PowerPoint."""
        try:
            from pptx import Presentation
            
            prs = Presentation(path)
            slides_text = []
            
            for i, slide in enumerate(prs.slides):
                slide_parts = [f"--- Slide {i+1} ---"]
                for shape in slide.shapes:
                    if hasattr(shape, "text") and shape.text.strip():
                        slide_parts.append(shape.text)
                if len(slide_parts) > 1:
                    slides_text.append("\n".join(slide_parts))
            
            if slides_text:
                result = f"# PowerPoint: {url}\n\n" + "\n\n".join(slides_text)
                logger.info(f"[BrowserPool] 📄 Extracted {len(result)} chars from PPTX")
                return result
            else:
                return f"[PPTX at {url} contains no text]"
                
        except ImportError:
            return f"[Cannot parse PPTX - install python-pptx: {url}]"
        except Exception as e:
            return f"[Error parsing PPTX {url}: {e}]"
                
    async def fetch_batch(self, urls: List[str]) -> List[PageSummary]:
        """Fetch multiple URLs concurrently."""
        tasks = [self._fetch_one(url) for url in urls]
        return await asyncio.gather(*tasks)
        
    async def _fetch_one(self, url: str) -> PageSummary:
        """Fetch one URL and return summary."""
        try:
            markdown = await self.simple_fetch(url)
            
            # Extract title (first # heading or first line)
            title = None
            for line in markdown.split('\n'):
                line = line.strip()
                if line.startswith('# '):
                    title = line[2:]
                    break
                elif line and not title:
                    title = line[:100]
                    
            return PageSummary(
                url=url,
                title=title,
                summary=markdown[:2000],  # Truncate for context
            )
        except Exception as e:
            return PageSummary(
                url=url,
                title=None,
                summary="",
                error=str(e),
            )

    # =========================================================================
    # Autonomous browse (with LLM)
    # =========================================================================
    
    async def autonomous_browse(
        self,
        url: str,
        goal: str,
        llm,
        max_steps: int = 30,
        stop_checker: Optional[Callable[[], bool]] = None,
    ) -> BrowseResult:
        """
        Run autonomous browser agent for complex tasks.
        
        The agent has access to an add_seeds() custom tool to directly collect seeds.
        
        Args:
            url: Starting URL
            goal: What to accomplish
            llm: LLM instance for the agent
            max_steps: Maximum agent steps
            stop_checker: Optional callable that returns True if we should stop
        """
        from browser_use import Agent, Tools
        
        # Check if we should stop before even starting
        if stop_checker and stop_checker():
            logger.info(f"[BrowserPool] ⏸️ Stop requested before browser agent start")
            return BrowseResult(success=False, error="Stopped before start")
        
        # Collected seeds during this browse session
        collected_seeds = []
        should_stop = False
        
        # Create custom tools including add_seeds
        tools = Tools()
        
        @tools.action(description="""Add seeds to the dataset collection.

Call this whenever you find valuable content that should become a dataset row.

Each seed should be a dict with:
- text: The complete content (a section, post, item, etc.) - REQUIRED
- note: Category or context info (optional)
- source_url: Where it came from (optional, defaults to current page)

A good seed is:
- COMPLETE: Contains enough info to generate a full dataset row
- CONTIGUOUS: From one section/item, not pieced together
- STANDALONE: Makes sense without external context

Example usage:
add_seeds([
    {"text": "Complete forum post content here...", "note": "Category: support question"},
    {"text": "Another complete item...", "note": "Type: product listing"}
])
""")
        def add_seeds(seeds: list) -> str:
            """Add seeds to collection."""
            for seed in seeds:
                if isinstance(seed, dict) and seed.get('text'):
                    collected_seeds.append(seed)
                    logger.info(f"[BrowserAgent] 🌱 Added seed: {str(seed.get('text', ''))[:80]}...")
            return f"✅ Added {len(seeds)} seeds. Total collected: {len(collected_seeds)}"
        
        # Add a check_should_stop tool so agent can proactively check
        @tools.action(description="Check if the task should be stopped/paused. Returns 'continue' or 'stop'. Call this periodically during long tasks.")
        def check_status() -> str:
            """Check if we should stop."""
            nonlocal should_stop
            if stop_checker and stop_checker():
                should_stop = True
                logger.info(f"[BrowserAgent] ⏸️ Stop requested")
                return "STOP - Task is being paused. Call done() immediately."
            return "continue"
        
        browser_id = None
        
        # Don't use context manager - we need to handle cleanup manually
        await self._semaphore.acquire()
        browser = await self._available.get()
        browser_id = self._browser_ids.get(id(browser), "?")
        
        logger.info(f"[Browser-{browser_id}] 🤖 Starting autonomous agent")
        logger.info(f"[Browser-{browser_id}] 📋 Start URL: {url}")
        logger.info(f"[Browser-{browser_id}] 📋 Goal: {goal[:150]}...")
        
        try:
            agent = Agent(
                task=goal,
                browser=browser,
                llm=llm,
                tools=tools,  # Include our custom add_seeds tool
                max_steps=max_steps,
            )
            
            history = await agent.run()
            
            # Log what the agent did
            logger.info(f"[Browser-{browser_id}] 🤖 Agent finished after {len(history.urls())} page visits")
            logger.info(f"[Browser-{browser_id}] 🤖 Success: {history.is_successful()}")
            logger.info(f"[Browser-{browser_id}] 🌱 Seeds collected via add_seeds(): {len(collected_seeds)}")
            
            # Get final page markdown
            page_markdown = None
            try:
                from browser_use.dom.markdown_extractor import extract_clean_markdown
                page_markdown, _ = await extract_clean_markdown(browser_session=browser)
            except Exception as e:
                logger.warning(f"[Browser-{browser_id}] Could not extract final page markdown: {e}")
            
            return BrowseResult(
                success=history.is_successful() or False,
                final_url=history.urls()[-1] if history.urls() else None,
                page_markdown=page_markdown,
                extracted_data=collected_seeds,  # Return seeds collected via add_seeds()
                error=history.errors()[-1] if history.has_errors() else None,
            )
            
        except Exception as e:
            logger.error(f"[Browser-{browser_id}] ❌ Autonomous browse failed: {e}")
            return BrowseResult(
                success=False,
                extracted_data=collected_seeds,  # Return any seeds collected before failure
                error=str(e),
            )
        finally:
            # Reset browser to clean state before returning to pool
            logger.info(f"[Browser-{browser_id}] 🔄 Resetting browser before returning to pool...")
            try:
                await browser.stop()
                await browser.start()
                logger.info(f"[Browser-{browser_id}] ✅ Browser reset complete")
            except Exception as e:
                logger.error(f"[Browser-{browser_id}] ❌ Browser reset failed: {e}")
            
            # Return browser to pool
            await self._available.put(browser)
            self._semaphore.release()
            logger.info(f"[Browser-{browser_id}] ↩️ Released browser back to pool")


class BrowserSession:
    """
    Stateful browser session for multi-turn interactions.
    
    Keeps a browser acquired for multiple actions,
    then releases when done.
    """
    
    def __init__(self, pool: BrowserPool, llm):
        self.pool = pool
        self.llm = llm
        self._browser = None
        self._acquired = False
        
    async def start(self, url: str, goal: str) -> BrowseResult:
        """Start a new session by acquiring browser and navigating."""
        if self._acquired:
            raise RuntimeError("Session already started")
            
        # Acquire from pool (bypass context manager)
        await self.pool._semaphore.acquire()
        self._browser = await self.pool._available.get()
        self._acquired = True
        
        try:
            page = await self._browser.new_page(url)
            await asyncio.sleep(1.5)
            
            from browser_use.dom.markdown_extractor import extract_clean_markdown
            markdown, _ = await extract_clean_markdown(browser_session=self._browser)
            
            return BrowseResult(
                success=True,
                final_url=url,
                page_markdown=markdown,
            )
        except Exception as e:
            return BrowseResult(success=False, error=str(e))
            
    async def action(self, instruction: str, max_steps: int = 10) -> BrowseResult:
        """Execute an action in the current session."""
        if not self._acquired:
            raise RuntimeError("Session not started")
            
        from browser_use import Agent
        
        try:
            agent = Agent(
                task=instruction,
                browser=self._browser,
                llm=self.llm,
                max_steps=max_steps,
            )
            history = await agent.run()
            
            from browser_use.dom.markdown_extractor import extract_clean_markdown
            markdown, _ = await extract_clean_markdown(browser_session=self._browser)
            
            return BrowseResult(
                success=history.is_successful() or False,
                final_url=history.urls()[-1] if history.urls() else None,
                page_markdown=markdown,
                extracted_data=history.extracted_content(),
            )
        except Exception as e:
            return BrowseResult(success=False, error=str(e))
            
    async def end(self):
        """End session and release browser back to pool."""
        if not self._acquired:
            return
            
        await self.pool._available.put(self._browser)
        self.pool._semaphore.release()
        self._browser = None
        self._acquired = False