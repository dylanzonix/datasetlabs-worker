"""
Artifacts for research session.

Stores search results and page views with ref_ids for browsing.
"""

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from urllib.parse import urljoin, urlparse


@dataclass
class SearchResult:
    """Single search result."""
    id: int
    title: str
    url: str
    snippet: str
    date: Optional[str] = None


@dataclass
class SearchResults:
    """Search results artifact."""
    type: str = "search_results"
    query: str = ""
    results: List[SearchResult] = field(default_factory=list)


@dataclass
class PageLink:
    """Link found on a page."""
    id: int
    text: str
    href: str


@dataclass
class PageView:
    """Page view artifact - line-numbered markdown content."""
    type: str = "page_view"
    url: str = ""
    fetched_at: str = ""
    lines: List[str] = field(default_factory=list)
    total_lines: int = 0
    links: List[PageLink] = field(default_factory=list)


RESPONSE_LENGTHS = {
    "short": {"lines": 60, "results": 5, "matches": 5},
    "medium": {"lines": 150, "results": 10, "matches": 10},
    "long": {"lines": 300, "results": 20, "matches": 20},
}


class ArtifactStore:
    """
    Session-scoped artifact storage.
    
    ref_ids: s0, s1, ... for searches; p0, p1, ... for pages
    """
    
    def __init__(self):
        self._artifacts: Dict[str, Any] = {}
        self._search_counter = 0
        self._page_counter = 0
    
    def store_search(self, results: SearchResults) -> str:
        """Store search results, return ref_id."""
        ref_id = f"s{self._search_counter}"
        self._search_counter += 1
        self._artifacts[ref_id] = results
        return ref_id
    
    def store_page(self, page_view: PageView) -> str:
        """Store page view, return ref_id."""
        ref_id = f"p{self._page_counter}"
        self._page_counter += 1
        self._artifacts[ref_id] = page_view
        return ref_id
    
    def get(self, ref_id: str) -> Optional[Any]:
        """Get artifact by ref_id."""
        return self._artifacts.get(ref_id)
    
    def get_search(self, ref_id: str) -> Optional[SearchResults]:
        """Get search results by ref_id."""
        artifact = self._artifacts.get(ref_id)
        if artifact and isinstance(artifact, SearchResults):
            return artifact
        return None
    
    def get_page(self, ref_id: str) -> Optional[PageView]:
        """Get page view by ref_id."""
        artifact = self._artifacts.get(ref_id)
        if artifact and isinstance(artifact, PageView):
            return artifact
        return None
    
    def clear(self):
        """Clear all artifacts."""
        self._artifacts.clear()
        self._search_counter = 0
        self._page_counter = 0


def extract_links_from_markdown(markdown: str, base_url: str) -> List[PageLink]:
    """Extract links from markdown content."""
    links = []
    seen_urls = set()
    pattern = r'\[([^\]]+)\]\(([^)]+)\)'
    
    link_id = 0
    for match in re.finditer(pattern, markdown):
        text = match.group(1).strip()
        href = match.group(2).strip()
        
        if href and not href.startswith(('http://', 'https://', 'mailto:', 'tel:', '#')):
            href = urljoin(base_url, href)
        
        if href and not href.startswith('#') and href not in seen_urls:
            links.append(PageLink(id=link_id, text=text[:100], href=href))
            seen_urls.add(href)
            link_id += 1
    
    return links


def format_viewport(lines: List[str], start: int, count: int, ref_id: str, url: str) -> str:
    """Format lines as numbered viewport with header."""
    total = len(lines)
    end = min(start + count, total)
    
    header = f"[{ref_id}] {url}\n"
    header += f"Showing lines {start}-{end-1} of {total}\n"
    header += "-" * 40
    
    output_lines = [header]
    for i in range(start, end):
        output_lines.append(f"L{i}: {lines[i]}")
    
    return '\n'.join(output_lines)


def format_links_table(links: List[PageLink], limit: int = 20) -> str:
    """Format links as table for LLM."""
    if not links:
        return "No links found."
    
    output = ["\nLinks (use click to follow):"]
    for link in links[:limit]:
        text_preview = link.text[:50] + "..." if len(link.text) > 50 else link.text
        output.append(f"  [{link.id}] {text_preview}")
    
    if len(links) > limit:
        output.append(f"  ... and {len(links) - limit} more links")
    
    return '\n'.join(output)


def format_search_results(results: SearchResults, ref_id: str, limit: int) -> str:
    """Format search results for LLM."""
    if not results.results:
        return f"[{ref_id}] No results found for: {results.query}"
    
    output = [f"[{ref_id}] Search results for: {results.query}\n"]
    for r in results.results[:limit]:
        # Include date if available
        date_str = f" ({r.date})" if r.date else ""
        output.append(f"[{r.id}] {r.title}{date_str}")
        output.append(f"    {r.snippet[:200]}")
        output.append(f"    URL: {r.url}")
        output.append("")
    
    return '\n'.join(output)


def find_in_lines(lines: List[str], pattern: str, max_matches: int, context: int = 2) -> str:
    """Find pattern in lines, return formatted matches with context."""
    try:
        regex = re.compile(pattern, re.IGNORECASE)
        use_regex = True
    except re.error:
        use_regex = False
    
    matches = []
    for i, line in enumerate(lines):
        if use_regex:
            if regex.search(line):
                matches.append(i)
        else:
            if pattern.lower() in line.lower():
                matches.append(i)
    
    if not matches:
        return f"No matches found for: {pattern}"
    
    output = [f"Found {len(matches)} matches for '{pattern}':\n"]
    
    shown = 0
    for match_line in matches:
        if shown >= max_matches:
            remaining = len(matches) - shown
            output.append(f"\n... and {remaining} more matches")
            break
        
        start = max(0, match_line - context)
        end = min(len(lines), match_line + context + 1)
        
        output.append("")
        for i in range(start, end):
            prefix = "→ " if i == match_line else "  "
            output.append(f"{prefix}L{i}: {lines[i]}")
        
        shown += 1
    
    return '\n'.join(output)