"""Integration tests for the Alembic migrations against a real Postgres.

Set TEST_DATABASE_URL to a server where the user may create databases (the docker compose
`rag` user can). Each test runs in a fresh, temporary database that is dropped afterwards.
"""

import os
import uuid
from collections.abc import Iterator
from pathlib import Path

import psycopg
import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy.engine import make_url

TEST_DATABASE_URL = os.environ.get("TEST_DATABASE_URL")
ALEMBIC_INI = Path(__file__).resolve().parents[2] / "alembic.ini"

pytestmark = pytest.mark.skipif(not TEST_DATABASE_URL, reason="TEST_DATABASE_URL is not set")


@pytest.fixture
def alembic_config() -> Config:
    return Config(str(ALEMBIC_INI))


@pytest.fixture
def database_url(monkeypatch: pytest.MonkeyPatch) -> Iterator[str]:
    assert TEST_DATABASE_URL
    name = f"test_{uuid.uuid4().hex[:12]}"
    url = make_url(TEST_DATABASE_URL).set(database=name).render_as_string(hide_password=False)

    with psycopg.connect(TEST_DATABASE_URL, autocommit=True) as admin:
        admin.execute(f'CREATE DATABASE "{name}"')
    monkeypatch.setenv("DATABASE_URL", url)
    try:
        yield url
    finally:
        with psycopg.connect(TEST_DATABASE_URL, autocommit=True) as admin:
            admin.execute(f'DROP DATABASE "{name}" WITH (FORCE)')


def installed_extensions(url: str) -> set[str]:
    with psycopg.connect(url) as conn:
        rows = conn.execute("SELECT extname FROM pg_extension").fetchall()
    return {row[0] for row in rows}


def pg_search_available(url: str) -> bool:
    with psycopg.connect(url) as conn:
        row = conn.execute(
            "SELECT 1 FROM pg_available_extensions WHERE name = 'pg_search'"
        ).fetchone()
    return row is not None


def test_upgrade_enables_vector(database_url: str, alembic_config: Config) -> None:
    command.upgrade(alembic_config, "head")

    assert "vector" in installed_extensions(database_url)


def test_upgrade_enables_pg_search_when_the_server_has_it(
    database_url: str, alembic_config: Config
) -> None:
    if not pg_search_available(database_url):
        pytest.skip("this Postgres server does not ship pg_search")

    command.upgrade(alembic_config, "head")

    assert "pg_search" in installed_extensions(database_url)


def test_upgrade_is_repeatable(database_url: str, alembic_config: Config) -> None:
    command.upgrade(alembic_config, "head")
    command.upgrade(alembic_config, "head")

    assert "vector" in installed_extensions(database_url)


def test_downgrade_removes_extensions(database_url: str, alembic_config: Config) -> None:
    command.upgrade(alembic_config, "head")
    command.downgrade(alembic_config, "base")

    assert installed_extensions(database_url).isdisjoint({"vector", "pg_search"})


def test_env_refuses_to_run_without_database_url(
    monkeypatch: pytest.MonkeyPatch, alembic_config: Config
) -> None:
    monkeypatch.delenv("DATABASE_URL", raising=False)

    with pytest.raises(RuntimeError, match="DATABASE_URL is not set"):
        command.upgrade(alembic_config, "head")
