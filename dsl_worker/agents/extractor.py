"""
Candidate extractor — batch extraction from dumped pages using cheap models.

V8: Non-agentic extraction. Takes pages dumped by crawlers, chunks them,
sends each chunk to a mini model for candidate extraction.
No agent loop, no tools — just LLM calls with JSON output.

Candidates are returned as strings or dicts — whatever is natural for the page.
No variable schema enforcement.

Massively parallel: asyncio.gather across all chunks from all pages.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple

from dsl_worker.billing.tracked_client import TrackedOpenAIClient
from dsl_worker.infra.pipeline import Seed

try:
    from langfuse import get_client as _get_langfuse_client

    def _get_langfuse():
        try:
            return _get_langfuse_client()
        except Exception:
            return None
except ImportError:
    def _get_langfuse():
        return None

logger = logging.getLogger(__name__)

DEFAULT_CHUNK_SIZE = 8000
DEFAULT_CHUNK_OVERLAP = 500
MAX_CONCURRENT_EXTRACTIONS = 20


class CandidateExtractor:
    """
    Extracts candidate items from dumped pages using cheap batch LLM calls.

    No agent loop — single structured LLM call per page chunk.
    Each extracted candidate becomes a Seed submitted to SeedProcessor.

    Usage:
        extractor = CandidateExtractor(
            openai_client=tracked_client,
            model="gpt-5-mini",
            candidate_description="churches with name and address",
            on_submit=seed_processor.submit_seed,
        )
        count, cost = await extractor.extract_page(page, semaphore)
    """

    def __init__(
        self,
        openai_client: TrackedOpenAIClient,
        model: str,
        candidate_description: str,
        on_submit: Callable[[Seed, str], Awaitable[Dict[str, Any]]],
        on_cost: Optional[Callable] = None,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    ) -> None:
        self.openai_client = openai_client
        self.model = model
        self.candidate_description = candidate_description
        self.on_submit = on_submit
        self.on_cost = on_cost
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.total_cost: float = 0.0

    def _chunk_page(self, content: str) -> List[str]:
        """Split page content into overlapping chunks."""
        if len(content) <= self.chunk_size:
            return [content]

        chunks = []
        start = 0
        while start < len(content):
            end = start + self.chunk_size
            chunks.append(content[start:end])
            start = end - self.chunk_overlap

        return chunks

    async def _extract_from_chunk(
        self,
        chunk: str,
        page_url: str,
    ) -> Tuple[List[Any], float]:
        """Extract candidates from a single page chunk.

        Returns (candidates, cost_usd). Candidates are strings or dicts.
        """
        prompt = (
            f"Extract all {self.candidate_description} from this page content.\n\n"
            f"For each candidate found, output it as an element in a JSON array. "
            f"Each element can be a string (e.g. a URL or name) or an object with "
            f"relevant fields — use whatever captures the most useful information.\n\n"
            f"Page URL: {page_url}\n\n"
            f"Page content:\n{chunk}\n\n"
            f"Output a JSON object with a single key \"candidates\" containing the array. "
            f"Only include items that are clearly {self.candidate_description}. "
            f"If no candidates are found, return {{\"candidates\": []}}."
        )

        try:
            input_items = [{"role": "user", "content": prompt}]

            # Trace the LLM call via Langfuse if available
            lf = _get_langfuse()
            obs_ctx = None
            if lf:
                try:
                    obs_ctx = lf.start_as_current_observation(
                        as_type="generation",
                        name="extractor:llm",
                        model=self.model,
                        input=input_items,
                    )
                except Exception:
                    obs_ctx = None

            response, cost = await self.openai_client.responses_create(
                model=self.model,
                input=input_items,
                text={"format": {"type": "json_object"}},
            )

            if obs_ctx is not None:
                try:
                    usage_details = None
                    if hasattr(response, "usage") and response.usage:
                        usage_details = {
                            "input": getattr(response.usage, "input_tokens", 0),
                            "output": getattr(response.usage, "output_tokens", 0),
                        }
                    obs_ctx.__enter__().update(
                        output=self._response_text(response),
                        usage_details=usage_details,
                        metadata={"cost_usd": round(cost.total_cost_usd, 6), "page_url": page_url},
                    )
                    obs_ctx.__exit__(None, None, None)
                except Exception:
                    pass

            if self.on_cost and cost.total_cost_usd > 0:
                await self.on_cost(cost.total_cost_usd, "extractor")

            candidates = self._parse_candidates(response, page_url)
            return candidates, cost.total_cost_usd

        except Exception as e:
            logger.warning(f"[Extractor] Chunk extraction failed for {page_url}: {e}")
            return [], 0.0

    @staticmethod
    def _response_text(response: Any) -> str:
        """Extract text content from an OpenAI Responses API response."""
        parts = []
        for item in response.output:
            if item.type == "message":
                for block in item.content:
                    if hasattr(block, "text"):
                        parts.append(block.text)
        return "".join(parts)

    def _parse_candidates(self, response: Any, page_url: str) -> List[Any]:
        """Parse candidate list from LLM response."""
        text = ""
        for item in response.output:
            if item.type == "message":
                for content_block in item.content:
                    if hasattr(content_block, "text"):
                        text += content_block.text

        if not text.strip():
            return []

        try:
            parsed = json.loads(text)

            if isinstance(parsed, dict):
                candidates = parsed.get("candidates", [])
                if isinstance(candidates, list):
                    return candidates
                return []

            if isinstance(parsed, list):
                return parsed

            return []

        except json.JSONDecodeError as e:
            logger.warning(
                f"[Extractor] JSON parse error for {page_url}: {e} "
                f"(response: {text[:200]})"
            )
            return []

    async def extract_page(
        self,
        page: Dict[str, Any],
        semaphore: asyncio.Semaphore,
    ) -> Tuple[int, float]:
        """Extract candidates from a single page (called as pages arrive).

        Args:
            page: Dict with keys: url, content, description, crawler_id
            semaphore: Shared semaphore for concurrency limiting

        Returns:
            (candidates_accepted, cost_usd)
        """
        chunks = self._chunk_page(page["content"])
        page_url = page.get("url", "unknown")

        async def extract_one(chunk: str) -> Tuple[List[Any], float]:
            async with semaphore:
                return await self._extract_from_chunk(chunk, page_url)

        results = await asyncio.gather(
            *[extract_one(c) for c in chunks],
            return_exceptions=True,
        )

        page_candidates = 0
        page_cost = 0.0

        for result in results:
            if isinstance(result, BaseException):
                logger.warning(f"[Extractor] Chunk failed for {page_url}: {result}")
                continue

            candidates, cost = result
            page_cost += cost

            for candidate in candidates:
                if not candidate:
                    continue
                seed = Seed(
                    values=candidate,
                    metadata={"source_url": page_url},
                )
                try:
                    status = await self.on_submit(seed, "extractor")
                    if status.get("accepted"):
                        page_candidates += 1
                except Exception as e:
                    logger.warning(f"[Extractor] Seed submission error: {e}")

        self.total_cost += page_cost
        logger.info(
            f"[Extractor] Page {page_url[:60]}: "
            f"{page_candidates} candidates from {len(chunks)} chunks, "
            f"cost=${page_cost:.4f}"
        )

        return page_candidates, page_cost

    async def run(self, pages: List[Dict[str, Any]]) -> Tuple[int, float]:
        """Extract candidates from all dumped pages (batch mode)."""
        if not pages:
            return 0, 0.0

        logger.info(f"[Extractor] Extracting from {len(pages)} pages using {self.model}")

        semaphore = asyncio.Semaphore(MAX_CONCURRENT_EXTRACTIONS)
        results = await asyncio.gather(
            *[self.extract_page(p, semaphore) for p in pages],
            return_exceptions=True,
        )

        total_candidates = 0
        total_cost = 0.0
        for result in results:
            if isinstance(result, BaseException):
                continue
            count, cost = result
            total_candidates += count
            total_cost += cost

        logger.info(
            f"[Extractor] Done: {total_candidates} candidates accepted "
            f"from {len(pages)} pages, cost=${total_cost:.4f}"
        )

        return total_candidates, total_cost
