"""Add project_settings and ai_usage_log tables.

Revision ID: p3e2f1a0b9c8
Revises: o2d1e0f9a8b7
Create Date: 2026-07-25
"""

from alembic import op
import sqlalchemy as sa

revision = 'p3e2f1a0b9c8'
down_revision = 'o2d1e0f9a8b7'
branch_labels = None
depends_on = None


def upgrade():
    bind = op.get_bind()
    insp = sa.inspect(bind)
    existing = set(insp.get_table_names())

    if 'project_settings' not in existing:
        op.create_table(
            'project_settings',
            sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
            sa.Column('project_id', sa.String(36), sa.ForeignKey('projects.project_id', ondelete='CASCADE'), nullable=False),
            sa.Column('category', sa.String(50), nullable=False),
            sa.Column('key', sa.String(100), nullable=False),
            sa.Column('value_encrypted', sa.Text(), nullable=False),
            sa.Column('created_at', sa.DateTime(), nullable=True),
            sa.Column('updated_at', sa.DateTime(), nullable=True),
            sa.PrimaryKeyConstraint('id'),
            sa.UniqueConstraint('project_id', 'category', 'key', name='uq_project_setting'),
        )
        op.create_index('ix_project_settings_project_id', 'project_settings', ['project_id'], unique=False)
        op.create_index('ix_project_settings_project_category', 'project_settings', ['project_id', 'category'], unique=False)

    if 'ai_usage_log' not in existing:
        op.create_table(
            'ai_usage_log',
            sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
            sa.Column('user_id', sa.String(255), nullable=False),
            sa.Column('project_id', sa.String(36), nullable=True),
            sa.Column('team_id', sa.String(36), nullable=True),
            sa.Column('agent', sa.String(100), nullable=False),
            sa.Column('provider', sa.String(20), nullable=False),
            sa.Column('model', sa.String(100), nullable=False),
            sa.Column('key_source', sa.String(20), nullable=False),
            sa.Column('prompt_tokens', sa.Integer(), nullable=False, server_default='0'),
            sa.Column('completion_tokens', sa.Integer(), nullable=False, server_default='0'),
            sa.Column('total_tokens', sa.Integer(), nullable=False, server_default='0'),
            sa.Column('estimated_cost_usd', sa.Float(), nullable=False, server_default='0'),
            sa.Column('created_at', sa.DateTime(), nullable=True),
            sa.PrimaryKeyConstraint('id'),
        )
        op.create_index('ix_ai_usage_log_user_id', 'ai_usage_log', ['user_id'], unique=False)
        op.create_index('ix_ai_usage_log_project_id', 'ai_usage_log', ['project_id'], unique=False)
        op.create_index('ix_ai_usage_log_team_id', 'ai_usage_log', ['team_id'], unique=False)
        op.create_index('ix_ai_usage_log_agent', 'ai_usage_log', ['agent'], unique=False)
        op.create_index('ix_ai_usage_log_created_at', 'ai_usage_log', ['created_at'], unique=False)


def downgrade():
    op.drop_table('ai_usage_log')
    op.drop_table('project_settings')
