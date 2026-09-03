"""Add workflow_instances.autonomy_mode for LangGraph orchestration.

Revision ID: r5g4h3i2j1k0
Revises: q4f3g2h1i0j9
Create Date: 2026-09-03
"""

from alembic import op
import sqlalchemy as sa

revision = 'r5g4h3i2j1k0'
down_revision = 'q4f3g2h1i0j9'
branch_labels = None
depends_on = None


def upgrade():
    bind = op.get_bind()
    insp = sa.inspect(bind)

    cols = {c["name"] for c in insp.get_columns("workflow_instances")}
    if "autonomy_mode" not in cols:
        op.add_column(
            "workflow_instances",
            sa.Column("autonomy_mode", sa.String(20), nullable=True, server_default="co-pilot"),
        )


def downgrade():
    op.drop_column("workflow_instances", "autonomy_mode")
