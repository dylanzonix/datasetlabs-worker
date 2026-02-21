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
    generation_parallel_samples: int = 30  # Number of samples to generate in parallel

    # Billing settings
    billing_margin_multiplier: float = 4.0  # 2x = 100% margin (charge $2 for $1 cost)
    billing_charge_threshold_cents: int = 100  # $1 - charge when accumulated
    billing_charge_interval_seconds: int = 60  # Charge at least every 60 seconds

    brave_api_key: str
    brave_search_rps: float = 0.5  # requests per second

    research_model: str = "gpt-5.2"
    generation_model: str = "gpt-5.2"
    summarize_model: str = "gpt-5-nano"

    # Browser proxy (Bright Data residential, optional — leave blank to disable)
    browser_proxy_server: str = ""       # e.g. "http://brd.superproxy.io:22225"
    browser_proxy_username: str = ""     # includes geo-pin: brd-customer-XXX-zone-residential-country-us-state-newyork
    browser_proxy_password: str = ""

    # Cookie persistence blob path (global pre-auth cookies shared across all projects)
    browser_global_cookies_blob_path: str = "browser/global_cookies.json"

    # Langfuse observability (optional — leave blank to disable)
    langfuse_secret_key: str = ""
    langfuse_public_key: str = ""
    langfuse_base_url: str = "https://cloud.langfuse.com"


settings = WorkerSettings()