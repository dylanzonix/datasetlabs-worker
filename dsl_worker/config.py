"""Worker configuration."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class WorkerSettings(BaseSettings):
    """Worker configuration from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",  # Allow extra env vars (e.g., from shared .env with API)
    )

    # Database
    database_url: str

    # Azure Service Bus
    azure_service_bus_connection_string: str
    azure_service_bus_queue_name: str = "jobs"

    # Azure Blob Storage (match API's approach)
    azure_storage_account_name: str
    azure_storage_account_key: str
    azure_storage_container_name: str = "datasetlabs"

    # Azure OpenAI
    azure_openai_api_key: str
    azure_openai_endpoint: str  # e.g. "https://found-mlr2zdw9-eastus2.cognitiveservices.azure.com"
    azure_openai_api_version: str = "2025-04-01-preview"

    # Worker settings
    max_concurrent_jobs: int = 1
    heartbeat_interval_seconds: int = 30
    checkpoint_interval_seconds: int = 60

    # Generation settings
    generation_parallel_samples: int = 10  # Number of samples to generate in parallel

    # Billing settings
    compute_cost_per_credit: float = 0.10  # How much raw OpenAI spend 1 credit covers
    billing_charge_threshold_cents: int = 100  # $1 - charge when accumulated
    billing_charge_interval_seconds: int = 60  # Charge at least every 60 seconds

    brave_api_key: str
    brave_search_rps: float = 0.5  # requests per second

    research_model: str = "gpt-5.4"
    generation_model: str = "gpt-5.4"

    # V10 pipeline settings
    research_subagent_model: str = "gpt-5.4"           # model for research subagents
    seed_yielder_model: str = "gpt-5-mini"              # model for harvesters
    max_research_subagents: int = 10             # cap on parallel research
    max_seed_yielders: int = 10                  # cap on parallel seed yielders
    orchestrator_max_turns: int = 40             # hard cap on orchestrator turns
    orchestrator_soft_limit: int = 25            # soft nudge to wrap up

    # Sandbox service
    sandbox_service_url: str = "http://localhost:8010"

    # Browser Use Cloud (https://cloud.browser-use.com)
    browser_use_api_key: str = ""                  # API key from cloud.browser-use.com/settings
    browser_use_proxy_country: str = "us"          # Residential proxy country code (us, gb, de, etc.)

    # Credential pool service URL (serves authenticated cookies per session)
    credential_pool_url: str = ""

    # Apollo.io (optional — leave blank to disable)
    apollo_api_key: str = ""
    apollo_cost_per_credit: float = 0.0238  # $60 / 2520 credits

    # Langfuse observability (optional — leave blank to disable)
    langfuse_secret_key: str = ""
    langfuse_public_key: str = ""
    langfuse_base_url: str = "https://cloud.langfuse.com"


settings = WorkerSettings()