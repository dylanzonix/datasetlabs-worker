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

    # OpenAI
    openai_api_key: str

    # Worker settings
    max_concurrent_jobs: int = 1
    heartbeat_interval_seconds: int = 30
    checkpoint_interval_seconds: int = 60


settings = WorkerSettings()