"""Application settings, read once from environment variables (and `.env` if present).

Every setting has a matching upper-case environment variable, e.g. `llm_model` is read from
`LLM_MODEL`. Call `get_settings()` instead of building `Settings` directly, so the whole
process shares one validated instance.
"""

from functools import lru_cache
from typing import Literal

from pydantic import PostgresDsn, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

LLMProvider = Literal["claude"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore", frozen=True)

    # Required: there is no sensible default for where the data lives.
    database_url: PostgresDsn

    # Only Claude is implemented so far; OpenAI and OpenAI-compatible endpoints come later.
    llm_provider: LLMProvider = "claude"
    llm_model: str = "claude-opus-5-5"
    # Optional here because the Anthropic SDK also reads ANTHROPIC_API_KEY on its own.
    anthropic_api_key: SecretStr | None = None

    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    # mypy can't see that pydantic-settings fills the required fields from the environment.
    return Settings()  # type: ignore[call-arg]
