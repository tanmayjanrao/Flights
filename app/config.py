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

    # QA tool - local Ollama (CPU-only, no GPU) running qwen3:4b
    ollama_base_url: str = "http://localhost:11434"
    qa_model: str = "qwen3:4b"
    # CPU-bound generation: keep this generous - retrying a slow call just doubles the wait.
    qa_timeout_seconds: int = 180
    # Structured JSON output only, no chain-of-thought in the answer -> doesn't need many tokens.
    qa_num_predict: int = 700
    # Few-shot prompt + one transcript comfortably fits well under 4k tokens.
    qa_num_ctx: int = 4096
    qa_temperature: float = 0.2
    # qwen3 has a "thinking" mode that burns a lot of CPU time on reasoning tokens before
    # it ever gets to the answer. Not needed for a rubric-scoring task, so off by default.
    qa_disable_thinking: bool = True

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
