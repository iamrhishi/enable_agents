"""Add project budget fields and cm_projects.platform_project_id link.

Revision ID: q4f3g2h1i0j9
Revises: p3e2f1a0b9c8
Create Date: 2026-07-26
"""

from alembic import op
import sqlalchemy as sa

revision = 'q4f3g2h1i0j9'
down_revision = 'p3e2f1a0b9c8'
branch_labels = None
depends_on = None


def upgrade():
    bind = op.get_bind()
    insp = sa.inspect(bind)

    projects_cols = {c["name"] for c in insp.get_columns("projects")}
    if "monthly_budget_usd" not in projects_cols:
        op.add_column("projects", sa.Column("monthly_budget_usd", sa.Float(), nullable=True))
    if "budget_alert_month" not in projects_cols:
        op.add_column("projects", sa.Column("budget_alert_month", sa.String(7), nullable=True))

    if "cm_projects" in insp.get_table_names():
        cm_cols = {c["name"] for c in insp.get_columns("cm_projects")}
        if "platform_project_id" not in cm_cols:
            op.add_column("cm_projects", sa.Column("platform_project_id", sa.String(36), nullable=True))
            op.create_index("ix_cm_projects_platform_project_id", "cm_projects", ["platform_project_id"], unique=False)


def downgrade():
    op.drop_column("projects", "monthly_budget_usd")
    op.drop_column("projects", "budget_alert_month")
    op.drop_index("ix_cm_projects_platform_project_id", table_name="cm_projects")
    op.drop_column("cm_projects", "platform_project_id")
