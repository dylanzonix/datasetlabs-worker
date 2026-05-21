"""v-next source adapter registry.

Importing this module registers every adapter into the global registry
(see base.py). Resolve a source string to an adapter with:

    from dsl_worker.sources import get_adapter
    adapter = get_adapter("apollo_companies")
    # or "apify_actor:clearpath/reddit-search-scraper"
"""

from dsl_worker.sources.base import (
    ColumnDef,
    FetchResult,
    SourceAdapter,
    SourceDescription,
    describe_source,
    get_adapter,
    list_sources,
)

# Side-effect imports register adapters into the registry.
from dsl_worker.sources import apollo_companies  # noqa: F401
from dsl_worker.sources import fullenrich_people  # noqa: F401
from dsl_worker.sources import google_maps  # noqa: F401
from dsl_worker.sources import apify_actor  # noqa: F401
from dsl_worker.sources import web_harvest  # noqa: F401
from dsl_worker.sources import browser_use  # noqa: F401
from dsl_worker.sources import file as file_source  # noqa: F401
from dsl_worker.sources import llm  # noqa: F401


__all__ = [
    "ColumnDef",
    "FetchResult",
    "SourceAdapter",
    "SourceDescription",
    "describe_source",
    "get_adapter",
    "list_sources",
]
