"""create predictions table

Revision ID: 1f008e33fe8d
Revises:
Create Date: 2026-02-01 00:27:31.436488

"""

from collections.abc import Sequence
from typing import Union

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "1f008e33fe8d"
down_revision: str | Sequence[str] | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        "predictions",
        sa.Column("filename", sa.String(length=255), nullable=False),
        sa.Column("prediction", sa.String(length=50), nullable=False),
        sa.Column("confidence", sa.Float(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("id", sa.Integer(), nullable=False),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_predictions")),
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_table("predictions")
