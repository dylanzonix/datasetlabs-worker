"""
Source Manager — persistent source storage with manifest indexing.

Sources are files saved to /workspace/sources/ during research.
The manifest (manifest.json) indexes all sources with metadata:
path, url, description, tags, authority score, source type.

Agents use save_source() to persist research material.
Row generators read the manifest to discover relevant sources.
"""

import asyncio
import json
import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Max file size for read_source (10MB)
MAX_SOURCE_READ_BYTES = 10 * 1024 * 1024


@dataclass
class SourceEntry:
    """A single source in the manifest."""
    path: str
    description: str
    tags: List[str]
    authority: float
    source_type: str
    url: Optional[str] = None


class SourceManager:
    """
    Manages source files and the manifest index.

    Thread-safe for concurrent writes from multiple subagents via asyncio.Lock.

    Usage:
        manager = SourceManager(workspace_dir)
        await manager.initialize()

        await manager.save_source(
            content="# DayZ Wiki: Combat Mechanics\n...",
            path="combat/wiki_lean_peek.md",
            description="Official wiki page on lean/peek mechanics",
            tags=["combat", "peeking", "mechanics"],
            authority=0.9,
            source_type="wiki",
            url="https://dayz.wiki/combat/lean",
        )

        summary = manager.get_manifest_summary()
        content = manager.read_source("combat/wiki_lean_peek.md")
    """

    def __init__(self, workspace_dir: Path) -> None:
        self.workspace_dir = Path(workspace_dir)
        self.sources_dir = self.workspace_dir / "sources"
        self.manifest_path = self.sources_dir / "manifest.json"
        self._manifest: List[Dict[str, Any]] = []
        self._lock = asyncio.Lock()

    async def initialize(self) -> None:
        """Create sources directory and load existing manifest."""
        self.sources_dir.mkdir(parents=True, exist_ok=True)

        if self.manifest_path.exists():
            try:
                data = json.loads(self.manifest_path.read_text(encoding="utf-8"))
                if isinstance(data, list):
                    self._manifest = data
                    logger.info(
                        f"[SourceManager] Loaded manifest with "
                        f"{len(self._manifest)} entries"
                    )
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f"[SourceManager] Could not load manifest: {e}")
                self._manifest = []
        else:
            self._manifest = []
            logger.info("[SourceManager] Initialized empty manifest")

    async def save_source(
        self,
        content: str,
        path: str,
        description: str,
        tags: List[str],
        authority: float,
        source_type: str,
        url: Optional[str] = None,
    ) -> str:
        """
        Save a source file and update the manifest.

        Args:
            content: File content to save.
            path: Relative path within /sources/ (e.g., "combat/wiki_peek.md").
            description: What this source contains (for manifest).
            tags: Freeform tags for filtering.
            authority: 0-1 score (wiki=0.9, expert=0.7, forum=0.5, random=0.3).
            source_type: Category (wiki, forum, article, code, data, upload, etc.).
            url: Original URL if this is a web source.

        Returns:
            The full path of the saved file.
        """
        async with self._lock:
            # Ensure path doesn't escape sources dir
            clean_path = Path(path)
            if clean_path.is_absolute() or ".." in clean_path.parts:
                raise ValueError(f"Invalid source path: {path}")

            full_path = self.sources_dir / clean_path
            full_path.parent.mkdir(parents=True, exist_ok=True)
            full_path.write_text(content, encoding="utf-8")

            # Clamp authority
            authority = max(0.0, min(1.0, authority))

            entry = SourceEntry(
                path=str(clean_path),
                description=description,
                tags=tags,
                authority=authority,
                source_type=source_type,
                url=url,
            )

            # Deduplicate by path — update existing or append new
            self._manifest = [
                e for e in self._manifest if e.get("path") != str(clean_path)
            ]
            self._manifest.append(asdict(entry))

            # Write manifest
            self.manifest_path.write_text(
                json.dumps(self._manifest, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )

            logger.info(
                f"[SourceManager] Saved source: {clean_path} "
                f"({len(content)} chars, authority={authority}, "
                f"tags={tags})"
            )

            return str(full_path)

    def get_manifest(self) -> List[Dict[str, Any]]:
        """Get raw manifest data."""
        return list(self._manifest)

    def get_manifest_summary(self) -> str:
        """
        Format manifest as a readable summary for LLM context.

        Returns a table-like format that's easy for models to scan.
        """
        if not self._manifest:
            return "No sources saved yet."

        lines = [
            f"Source manifest ({len(self._manifest)} files):\n",
            "| Path | Description | Tags | Authority | Type |",
            "|------|-------------|------|-----------|------|",
        ]

        for entry in self._manifest:
            path = entry.get("path", "?")
            desc = entry.get("description", "")
            # Truncate long descriptions
            if len(desc) > 80:
                desc = desc[:77] + "..."
            tags = ", ".join(entry.get("tags", []))
            auth = entry.get("authority", 0)
            stype = entry.get("source_type", "?")
            lines.append(f"| {path} | {desc} | {tags} | {auth} | {stype} |")

        return "\n".join(lines)

    def read_source(self, path: str) -> str:
        """
        Read a source file by its manifest path.

        Args:
            path: Relative path within /sources/ (as listed in manifest).

        Returns:
            File content as string.

        Raises:
            FileNotFoundError: If the source file doesn't exist.
            ValueError: If path tries to escape sources directory.
        """
        clean_path = Path(path)
        if clean_path.is_absolute() or ".." in clean_path.parts:
            raise ValueError(f"Invalid source path: {path}")

        full_path = self.sources_dir / clean_path
        if not full_path.exists():
            raise FileNotFoundError(f"Source not found: {path}")

        size = full_path.stat().st_size
        if size > MAX_SOURCE_READ_BYTES:
            return (
                f"[Source too large: {size / 1024 / 1024:.1f}MB. "
                f"Max: {MAX_SOURCE_READ_BYTES / 1024 / 1024:.0f}MB]"
            )

        return full_path.read_text(encoding="utf-8")

    @property
    def source_count(self) -> int:
        """Number of sources in the manifest."""
        return len(self._manifest)
