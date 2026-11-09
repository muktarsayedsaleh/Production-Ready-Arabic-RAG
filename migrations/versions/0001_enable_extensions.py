"""Enable the vector and (when available) pg_search extensions

pgvector is required: dense and sparse embeddings are stored with it.

pg_search is optional. It ships with the ParadeDB image used in docker compose, but managed
Postgres such as RDS and Aurora doesn't allow it. There the migration skips it, and the plain
full-text and sparse-vector keyword backends still work.

Revision ID: 0001
Revises:
Create Date: 2026-11-09
"""

import logging
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

logger = logging.getLogger("alembic.runtime.migration")


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    available = op.get_bind().scalar(
        sa.text("SELECT 1 FROM pg_available_extensions WHERE name = 'pg_search'")
    )
    if available:
        op.execute("CREATE EXTENSION IF NOT EXISTS pg_search")
    else:
        logger.warning(
            "pg_search is not available on this server; skipping it. "
            "The 'paradedb' keyword backend will not work here."
        )


def downgrade() -> None:
    op.execute("DROP EXTENSION IF EXISTS pg_search")
    op.execute("DROP EXTENSION IF EXISTS vector")
