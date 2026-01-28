"""
Browser agent tools for communication with research agent.

Two main tools:
- mark_for_extraction: Save current page content immediately (crash resilient)
- checkpoint: Report status and get instructions from research agent
"""

import asyncio
import logging
import os
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Awaitable
from urllib.parse import urlparse

from markdownify import markdownify as md

logger = logging.getLogger(__name__)


@dataclass
class ExtractionItem:
    """Item queued for seed extraction."""
    file_path: str
    source_url: str
    description: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass 
class BrowserToolContext:
    """
    Context shared between browser agent and research agent.
    
    Provides:
    - Extraction queue for mark_for_extraction
    - Callback to research agent for checkpoint
    - Workspace directory for saving files
    - Stats tracking
    """
    workspace_dir: Path
    extraction_queue: asyncio.Queue
    checkpoint_callback: Callable[[str, str], Awaitable[str]]
    
    # Stats
    pages_extracted: int = 0
    current_url: str = ""
    
    # Seed stats (updated by extraction workers)
    total_seeds: int = 0
    avg_quality: float = 0.0
    remaining_quotas: Dict[str, int] = field(default_factory=dict)


def url_to_filename(url: str) -> str:
    """Convert URL to safe filename."""
    parsed = urlparse(url)
    domain = parsed.netloc.replace(".", "_").replace(":", "_")
    path = parsed.path.replace("/", "_").replace(".", "_")
    
    # Truncate if too long
    name = f"web_{domain}{path}"
    if len(name) > 100:
        name = name[:100]
    
    # Add unique suffix
    suffix = uuid.uuid4().hex[:8]
    return f"{name}_{suffix}.md"


def create_browser_tools(context: BrowserToolContext):
    """
    Create browser-use Tools instance with mark_for_extraction and checkpoint.
    
    Args:
        context: Shared context for communication
        
    Returns:
        Tools instance to pass to browser-use Agent
    """
    from browser_use import Tools, ActionResult, BrowserSession
    
    tools = Tools()
    
    @tools.action('Save current page content for seed extraction. Call this whenever you see content worth extracting - saves immediately so progress is preserved even if you crash later.')
    async def mark_for_extraction(description: str, browser_session: BrowserSession) -> ActionResult:
        """
        Dump current page to markdown file and queue for extraction.
        
        Args:
            description: Brief description of what content this page contains
        """
        try:
            # Get current URL
            current_url = await browser_session.page.evaluate('() => window.location.href')
            context.current_url = current_url
            
            # Get HTML and convert to markdown
            html = await browser_session.page.evaluate('() => document.body.innerHTML')
            markdown = md(html, heading_style='ATX', strip=['script', 'style'])
            
            # Add metadata header
            content = f"""---
source_url: {current_url}
extracted_at: {datetime.now(timezone.utc).isoformat()}
description: {description}
---

{markdown}
"""
            
            # Save to workspace
            filename = url_to_filename(current_url)
            filepath = context.workspace_dir / filename
            filepath.write_text(content, encoding='utf-8')
            
            # Queue for extraction
            item = ExtractionItem(
                file_path=str(filepath),
                source_url=current_url,
                description=description,
            )
            await context.extraction_queue.put(item)
            
            context.pages_extracted += 1
            
            logger.info(f"[BrowserTools] Saved page {context.pages_extracted}: {filename}")
            
            return ActionResult(
                extracted_content=f"Saved to {filename}. Total pages saved: {context.pages_extracted}",
                long_term_memory=f"Saved page: {description} ({filename})"
            )
            
        except Exception as e:
            logger.error(f"[BrowserTools] mark_for_extraction failed: {e}")
            return ActionResult(
                error=f"Failed to save page: {e}"
            )
    
    @tools.action('Checkpoint - report your current status and get instructions on what to do next. Call this when you reach a decision point, complete a section, or need guidance.')
    async def checkpoint(status: str, browser_session: BrowserSession) -> ActionResult:
        """
        Report status to research agent and get next instructions.
        
        Args:
            status: What you've found/done and any questions
        """
        try:
            # Get current URL
            current_url = await browser_session.page.evaluate('() => window.location.href')
            context.current_url = current_url
            
            # Call research agent for guidance
            instructions = await context.checkpoint_callback(current_url, status)
            
            logger.info(f"[BrowserTools] Checkpoint at {current_url[:50]}...")
            logger.info(f"[BrowserTools] Status: {status[:100]}...")
            logger.info(f"[BrowserTools] Instructions: {instructions[:100]}...")
            
            return ActionResult(
                extracted_content=instructions,
                long_term_memory=f"Checkpoint: {status[:50]}... → Instructions received"
            )
            
        except Exception as e:
            logger.error(f"[BrowserTools] checkpoint failed: {e}")
            return ActionResult(
                error=f"Checkpoint failed: {e}. Continue with your best judgment."
            )
    
    return tools


class BrowserAgentDispatcher:
    """
    Dispatches browser agent tasks and handles communication.
    
    Manages:
    - Browser pool lifecycle
    - Tool context setup
    - Agent execution
    - Extraction queue processing
    """
    
    def __init__(
        self,
        workspace_dir: Path,
        extraction_queue: asyncio.Queue,
        checkpoint_callback: Callable[[str, str], Awaitable[str]],
        browser_pool_size: int = 3,
        headless: bool = False,
    ):
        self.workspace_dir = Path(workspace_dir)
        self.workspace_dir.mkdir(parents=True, exist_ok=True)
        
        self.extraction_queue = extraction_queue
        self.checkpoint_callback = checkpoint_callback
        self.browser_pool_size = browser_pool_size
        self.headless = headless
        
        self._browser = None
        self._started = False
        
    async def start(self):
        """Initialize browser."""
        if self._started:
            return
            
        from browser_use import Browser
        
        self._browser = Browser(headless=self.headless)
        await self._browser.start()
        self._started = True
        
        logger.info(f"[BrowserDispatcher] Browser started (headless={self.headless})")
        
    async def stop(self):
        """Shutdown browser."""
        if self._browser:
            try:
                await self._browser.stop()
            except Exception as e:
                logger.warning(f"[BrowserDispatcher] Error stopping browser: {e}")
        self._browser = None
        self._started = False
        
    async def dispatch(
        self,
        url: str,
        goal: str,
        model: str = "gpt-5.2",
        max_steps: int = 50,
    ) -> Dict[str, Any]:
        """
        Dispatch browser agent to navigate and extract.
        
        Args:
            url: Starting URL
            goal: What to accomplish (navigate, extract, etc.)
            model: LLM model for agent
            max_steps: Maximum agent steps
            
        Returns:
            Dict with results:
            - success: bool
            - pages_extracted: int
            - error: Optional[str]
        """
        if not self._started:
            await self.start()
            
        from browser_use import Agent, ChatOpenAI
        
        # Create tool context
        context = BrowserToolContext(
            workspace_dir=self.workspace_dir,
            extraction_queue=self.extraction_queue,
            checkpoint_callback=self.checkpoint_callback,
        )
        
        # Create tools
        tools = create_browser_tools(context)
        
        # Create LLM
        llm = ChatOpenAI(model=model)
        
        # Build task with guidance
        task = f"""Navigate to {url} and accomplish this goal:

{goal}

IMPORTANT:
- When you find pages with content worth extracting, call mark_for_extraction() to save them
- Call checkpoint() when you reach a decision point or need guidance
- Don't try to extract data yourself - just save pages and let the extraction system handle it
- If you encounter pagination or infinite scroll, navigate through all pages and save each one
"""
        
        logger.info(f"[BrowserDispatcher] Starting agent: {url}")
        logger.info(f"[BrowserDispatcher] Goal: {goal[:100]}...")
        
        try:
            agent = Agent(
                task=task,
                browser=self._browser,
                llm=llm,
                tools=tools,
                max_steps=max_steps,
            )
            
            history = await agent.run()
            
            success = history.is_successful() if hasattr(history, 'is_successful') else True
            
            logger.info(f"[BrowserDispatcher] Agent finished. Pages extracted: {context.pages_extracted}")
            
            return {
                "success": success,
                "pages_extracted": context.pages_extracted,
                "error": history.errors()[-1] if hasattr(history, 'errors') and history.errors() else None,
            }
            
        except Exception as e:
            logger.error(f"[BrowserDispatcher] Agent failed: {e}")
            return {
                "success": False,
                "pages_extracted": context.pages_extracted,
                "error": str(e),
            }