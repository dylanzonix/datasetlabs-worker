from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",       # only needed in dev
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ---- App basics ----
    app_env: str = Field("development", alias="APP_ENV")
    app_name: str = Field("dataset-engine", alias="APP_NAME")
    log_level: str = Field("INFO", alias="LOG_LEVEL")

    # ---- LLM / AI ----
    openai_api_key: str | None = Field(default=None, alias="OPENAI_API_KEY")


# singleton-style helper
settings = Settings()
