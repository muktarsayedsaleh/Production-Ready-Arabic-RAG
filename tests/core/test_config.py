import pytest
from pydantic import ValidationError

from core.config import Settings, get_settings

DB_URL = "postgresql://rag:rag@localhost:5432/rag"


@pytest.fixture(autouse=True)
def clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("DATABASE_URL", "LLM_PROVIDER", "LLM_MODEL", "ANTHROPIC_API_KEY", "LOG_LEVEL"):
        monkeypatch.delenv(name, raising=False)
    get_settings.cache_clear()


def load() -> Settings:
    # _env_file=None keeps a developer's local .env out of the tests.
    return Settings(_env_file=None)  # type: ignore[call-arg]


def test_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATABASE_URL", DB_URL)

    settings = load()

    assert str(settings.database_url) == DB_URL
    assert settings.llm_provider == "claude"
    assert settings.llm_model == "claude-opus-5-5"
    assert settings.anthropic_api_key is None
    assert settings.log_level == "INFO"


def test_reads_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATABASE_URL", DB_URL)
    monkeypatch.setenv("LLM_MODEL", "claude-sonnet-5-5")
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")

    settings = load()

    assert settings.llm_model == "claude-sonnet-5-5"
    assert settings.log_level == "DEBUG"


def test_database_url_is_required() -> None:
    with pytest.raises(ValidationError, match="database_url"):
        load()


def test_rejects_non_postgres_database_url(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATABASE_URL", "mysql://user:pass@localhost/db")

    with pytest.raises(ValidationError, match="database_url"):
        load()


def test_rejects_unknown_llm_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATABASE_URL", DB_URL)
    monkeypatch.setenv("LLM_PROVIDER", "gemini")

    with pytest.raises(ValidationError, match="llm_provider"):
        load()


def test_api_key_is_not_leaked_in_repr(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATABASE_URL", DB_URL)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-secret")

    settings = load()

    assert "sk-ant-secret" not in repr(settings)
    assert settings.anthropic_api_key is not None
    assert settings.anthropic_api_key.get_secret_value() == "sk-ant-secret"


def test_get_settings_returns_one_shared_instance(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATABASE_URL", DB_URL)

    assert get_settings() is get_settings()
