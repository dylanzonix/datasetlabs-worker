"""v-next source adapter registry.

Importing this module registers every adapter into the global registry
(see base.py). Resolve a source string to an adapter with:

    from dsl_worker.sources_v2 import get_adapter
    adapter = get_adapter("apollo_companies")
    # or "apify_actor:clearpath/reddit-search-scraper"
"""

from dsl_worker.sources_v2.base import (
    ColumnDef,
    FetchResult,
    SourceAdapter,
    SourceDescription,
    describe_source,
    get_adapter,
    list_sources,
)

# Side-effect imports register adapters into the registry.
from dsl_worker.sources_v2 import apollo_companies  # noqa: F401
from dsl_worker.sources_v2 import fullenrich_people  # noqa: F401
from dsl_worker.sources_v2 import google_maps  # noqa: F401
from dsl_worker.sources_v2 import apify_actor  # noqa: F401
from dsl_worker.sources_v2 import web_harvest  # noqa: F401
from dsl_worker.sources_v2 import browser_use  # noqa: F401
from dsl_worker.sources_v2 import file as file_source  # noqa: F401
from dsl_worker.sources_v2 import llm  # noqa: F401


__all__ = [
    "ColumnDef",
    "FetchResult",
    "SourceAdapter",
    "SourceDescription",
    "describe_source",
    "get_adapter",
    "list_sources",
]
