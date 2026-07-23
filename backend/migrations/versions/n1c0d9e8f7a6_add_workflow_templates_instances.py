"""Add workflow_templates and workflow_instances tables.

These tables backed the Workflows feature from the start but were only
ever created ad-hoc (via a manual db.create_all() during local dev),
never through a real migration - which is dead code in this app since
the container command is `flask run`, not `python app.py`. That left
production without these tables entirely.

Revision ID: n1c0d9e8f7a6
Revises: m0b9c8d7e6f5
Create Date: 2026-07-23
"""

from alembic import op
import sqlalchemy as sa

revision = 'n1c0d9e8f7a6'
down_revision = 'm0b9c8d7e6f5'
branch_labels = None
depends_on = None


def upgrade():
    bind = op.get_bind()
    insp = sa.inspect(bind)
    existing = set(insp.get_table_names())

    if 'workflow_templates' not in existing:
        op.create_table(
            'workflow_templates',
            sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
            sa.Column('template_id', sa.String(100), nullable=False),
            sa.Column('name', sa.String(255), nullable=False),
            sa.Column('description', sa.Text(), nullable=True),
            sa.Column('category', sa.String(100), nullable=True),
            sa.Column('icon', sa.String(50), nullable=True, server_default='workflow'),
            sa.Column('is_system', sa.Boolean(), nullable=True, server_default='false'),
            sa.Column('is_active', sa.Boolean(), nullable=True, server_default='true'),
            sa.Column('stages', sa.Text(), nullable=False, server_default='[]'),
            sa.Column('created_at', sa.DateTime(), nullable=True),
            sa.Column('updated_at', sa.DateTime(), nullable=True),
            sa.PrimaryKeyConstraint('id'),
            sa.UniqueConstraint('template_id'),
        )
        op.create_index('ix_workflow_templates_template_id', 'workflow_templates', ['template_id'], unique=True)
        op.create_index('ix_workflow_templates_category', 'workflow_templates', ['category'], unique=False)
        op.create_index('ix_workflow_templates_is_active', 'workflow_templates', ['is_active'], unique=False)

    if 'workflow_instances' not in existing:
        op.create_table(
            'workflow_instances',
            sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
            sa.Column('instance_id', sa.String(36), nullable=False),
            sa.Column('template_id', sa.String(100), sa.ForeignKey('workflow_templates.template_id', ondelete='CASCADE'), nullable=False),
            sa.Column('user_id', sa.String(255), nullable=False),
            sa.Column('project_id', sa.String(36), nullable=True),
            sa.Column('name', sa.String(255), nullable=False),
            sa.Column('status', sa.String(50), nullable=True, server_default='pending'),
            sa.Column('current_stage_index', sa.Integer(), nullable=True, server_default='0'),
            sa.Column('stage_states', sa.Text(), nullable=True, server_default='{}'),
            sa.Column('context', sa.Text(), nullable=True, server_default='{}'),
            sa.Column('started_at', sa.DateTime(), nullable=True),
            sa.Column('completed_at', sa.DateTime(), nullable=True),
            sa.Column('created_at', sa.DateTime(), nullable=True),
            sa.Column('updated_at', sa.DateTime(), nullable=True),
            sa.PrimaryKeyConstraint('id'),
            sa.UniqueConstraint('instance_id'),
        )
        op.create_index('ix_workflow_instances_instance_id', 'workflow_instances', ['instance_id'], unique=True)
        op.create_index('ix_workflow_instances_template_id', 'workflow_instances', ['template_id'], unique=False)
        op.create_index('ix_workflow_instances_user_id', 'workflow_instances', ['user_id'], unique=False)
        op.create_index('ix_workflow_instances_project_id', 'workflow_instances', ['project_id'], unique=False)
        op.create_index('ix_workflow_instances_status', 'workflow_instances', ['status'], unique=False)


def downgrade():
    op.drop_table('workflow_instances')
    op.drop_table('workflow_templates')
