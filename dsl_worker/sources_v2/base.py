"""Source adapter base class — one adapter per source enum value.

Each adapter takes `query_params` (source-specific shape, validated locally),
runs the source, and returns a `FetchResult` with rows + cost + a continuation
signal. The orchestrator never thinks about per-source pagination internals;
it just calls table_extend with new query_params and the adapter handles them.

Predictable sources (`predictable=True`) have a fixed `default_columns` map —
field names + types — so `table_create` can immediately commit rows without
a column_map_set round-trip.

Unpredictable sources (`predictable=False`) return whatever fields the source
emits. `table_create` returns first ~10 rows as preview; agent calls
`column_map_set` to commit mapping; server then completes the fetch in the
background.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class FetchResult:
    """What a source adapter returns from a fetch."""

    # Rows in the order the source returned them. Each row is a dict of
    # source-native field names → values. The orchestrator's column_map
    # decides which fields to keep and what to name/type them.
    rows: List[Dict[str, Any]]

    # All field names present in the returned rows. For predictable sources,
    # this matches default_columns. For unpredictable, this is whatever the
    # source emitted on this fetch.
    schema: List[str]

    # Credit cost of this fetch. Empirical; what the user actually pays.
    cost_credits: float

    # True if the source can't yield more rows for THIS query. Doesn't
    # block future table_extend calls — agent may still extend with a
    # different query angle.
    exhausted: bool = False

    # Optional continuation hint stored back on the table's source_cursor
    # for the agent to inspect via project state. Per-source shape — e.g.
    # {"oldest_seen_date": "2026-05-09"} or {"next_page_token": "..."}.
    cursor: Optional[Dict[str, Any]] = None

    # Optional dedup hint — the adapter's recommended dedup_key_column if
    # the rows have a clear unique field. Only used on first fetch.
    dedup_key_column_hint: Optional[str] = None


@dataclass
class ColumnDef:
    """A column definition — name + soft type."""
    name: str
    type: str  # one of: text, number, url, email, date, bool, enum


class SourceAdapter(abc.ABC):
    """Base class. Each concrete source registers exactly one adapter."""

    # Source enum value — apollo_companies, fullenrich_people, etc.
    # For apify_actor:<id>, name is "apify_actor" and the actor_id is
    # carried in query_params.
    name: str = ""

    # Predictable sources have fixed default columns. Unpredictable
    # sources require column_map_set after the agent inspects a preview.
    predictable: bool = True

    # Default column mapping for predictable sources:
    # [{source_field, column_name, type}, ...]
    default_columns: List[Dict[str, str]] = []

    # Recommended dedup_key_column for this source (used on first create).
    default_dedup_key_column: Optional[str] = None

    @abc.abstractmethod
    async def fetch(
        self,
        query_params: Dict[str, Any],
        n: int,
        prior_cursor: Optional[Dict[str, Any]] = None,
    ) -> FetchResult:
        """Run the source and return up to `n` rows.

        Args:
            query_params: source-specific shape, validated by validate_query_params.
            n: max rows to return on this call.
            prior_cursor: continuation state from the previous fetch on this
                table (None on first fetch).

        Returns:
            FetchResult with rows, schema, cost, exhausted, cursor.
        """
        raise NotImplementedError

    def validate_query_params(self, query_params: Dict[str, Any]) -> Optional[str]:
        """Return None if valid, or a human-readable error string with hint.

        Default: accept anything. Concrete adapters override to enforce
        per-source schema.
        """
        return None

    @classmethod
    def query_params_schema(cls) -> Dict[str, Any]:
        """JSON Schema for this adapter's query_params.

        Used to emit per-source `table_create_<source>` tool variants
        with a strict params schema so the LLM gets schema-level
        rejection of invalid keys instead of a runtime error.

        Default: accept any object. Concrete adapters override to
        enumerate exact params + types.
        """
        return {"type": "object", "additionalProperties": True}

    @classmethod
    def tool_description(cls) -> str:
        """One-line description for the per-source table_create tool. Override
        to give source-specific guidance to the agent."""
        return f"Create a table from {cls.name}."


# Adapter registry — populated by each adapter module on import.
_REGISTRY: Dict[str, SourceAdapter] = {}


def register(adapter: SourceAdapter) -> SourceAdapter:
    """Register a concrete adapter. Called once at module import time."""
    if not adapter.name:
        raise ValueError(f"Adapter {adapter.__class__.__name__} has no name")
    _REGISTRY[adapter.name] = adapter
    return adapter


def get_adapter(source: str) -> SourceAdapter:
    """Resolve a `source` enum value to an adapter instance.

    For apify_actor:<id>, this strips the ":<id>" and resolves to the
    "apify_actor" adapter; the actor_id is read from query_params.
    """
    base = source.split(":", 1)[0]
    if base not in _REGISTRY:
        raise KeyError(f"No adapter registered for source={source!r}; available: {list(_REGISTRY)}")
    return _REGISTRY[base]


def list_sources() -> List[str]:
    """All registered source names. Used to validate the source enum."""
    return sorted(_REGISTRY)
