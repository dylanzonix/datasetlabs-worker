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

    # Direct OpenAI (used when use_direct_openai=true to bypass Azure content filters)
    openai_api_key: str = ""
    use_direct_openai: bool = False

    # LLM provider: "openai" (default) or "anthropic". When "anthropic",
    # all agent calls are routed through TrackedAnthropicClient using
    # anthropic_model. Pricing and cost tracking flow through unchanged.
    llm_provider: str = "openai"
    anthropic_api_key: str = ""
    anthropic_model: str = "claude-opus-4-7"

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
    seed_yielder_model: str = "gpt-5.4"               # model for harvesters
    max_research_subagents: int = 10             # cap on parallel research
    max_seed_yielders: int = 10                  # cap on parallel seed yielders
    orchestrator_max_turns: int = 200            # hard cap on orchestrator turns
    orchestrator_soft_limit: int = 150           # soft nudge to wrap up

    # Sandbox service
    sandbox_service_url: str = "http://localhost:8010"

    # Browser Use Cloud (https://cloud.browser-use.com)
    browser_use_api_key: str = ""                  # API key from cloud.browser-use.com/settings
    browser_use_proxy_country: str = "us"          # Residential proxy country code (us, gb, de, etc.)

    # Apollo.io (optional — leave blank to disable)
    apollo_api_key: str = ""
    apollo_cost_per_credit: float = 0.0238  # $60 / 2520 credits

    # Google APIs (Maps Places + YouTube Data v3, optional)
    google_api_key: str = ""

    # Apify (optional — leave blank to disable)
    apify_api_key: str = ""

    # FullEnrich (optional — leave blank to disable)
    fullenrich_api_key: str = ""
    fullenrich_cost_per_credit: float = 0.055  # ~$55/1000 credits (Pro plan)

    pipeline_version: str = "v13"

    # Langfuse observability (optional — leave blank to disable)
    langfuse_secret_key: str = ""
    langfuse_public_key: str = ""
    langfuse_base_url: str = "https://cloud.langfuse.com"

    def get_model(self, role: str) -> str:
        """Resolve the model for a role, respecting llm_provider.

        When llm_provider == "anthropic", every role maps to anthropic_model.
        Otherwise returns the role-specific OpenAI model.
        """
        if self.llm_provider == "anthropic":
            return self.anthropic_model
        role_map = {
            "generation": self.generation_model,
            "research": self.research_model,
            "seed_yielder": self.seed_yielder_model,
            "subagent": self.research_subagent_model,
        }
        return role_map.get(role, self.generation_model)


settings = WorkerSettings()