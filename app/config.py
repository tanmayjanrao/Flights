"""
Central app configuration.

Uses pydantic-settings so every value is typed, validated once at startup,
and read from `.env` automatically. Nothing else in the codebase should call
os.getenv() directly - import `settings` from here instead.
"""
from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Aviationstack
    aviationstack_api_key: str = ""
    aviationstack_base_url: str = "https://api.aviationstack.com/v1"

    # AirLabs
    airlabs_api_key: str = ""
    airlabs_base_url: str = "https://airlabs.co/api/v9"

    # Provider orchestration
    primary_provider: Literal["airlabs", "aviationstack"] = "airlabs"
    http_timeout: int = 10

    # CORS
    allowed_origins: str = "*"

    @property
    def cors_origins(self) -> list[str]:
        if self.allowed_origins.strip() == "*":
            return ["*"]
        return [o.strip() for o in self.allowed_origins.split(",") if o.strip()]

    def provider_configured(self, name: str) -> bool:
        if name == "airlabs":
            return bool(self.airlabs_api_key)
        if name == "aviationstack":
            return bool(self.aviationstack_api_key)
        return False


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()

if not settings.airlabs_api_key and not settings.aviationstack_api_key:
    print("Warning: no flight API keys configured - set AIRLABS_API_KEY and/or "
          "AVIATIONSTACK_API_KEY in your .env file.")
