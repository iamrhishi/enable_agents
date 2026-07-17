"""add agent_context table

Revision ID: c1a4d2e9b7f0
Revises: 7c2fba01add3
Create Date: 2026-05-15
"""
from alembic import op
import sqlalchemy as sa


revision = 'c1a4d2e9b7f0'
down_revision = '7c2fba01add3'
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        'agent_context',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('user_id', sa.String(length=255), nullable=False),
        sa.Column('session_id', sa.String(length=36), nullable=True),
        sa.Column('agent_id', sa.String(length=100), nullable=False),
        sa.Column('key', sa.String(length=255), nullable=False),
        sa.Column('value', sa.Text(), nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('user_id', 'agent_id', 'key', name='uq_agent_context'),
    )
    op.create_index('ix_agent_context_user_id', 'agent_context', ['user_id'], unique=False)
    op.create_index('ix_agent_context_session_id', 'agent_context', ['session_id'], unique=False)
    op.create_index('ix_agent_context_agent_id', 'agent_context', ['agent_id'], unique=False)
    op.create_index('ix_agent_context_key', 'agent_context', ['key'], unique=False)
    op.create_index('ix_agent_context_user_key', 'agent_context', ['user_id', 'key'], unique=False)


def downgrade():
    op.drop_index('ix_agent_context_user_key', table_name='agent_context')
    op.drop_index('ix_agent_context_key', table_name='agent_context')
    op.drop_index('ix_agent_context_agent_id', table_name='agent_context')
    op.drop_index('ix_agent_context_session_id', table_name='agent_context')
    op.drop_index('ix_agent_context_user_id', table_name='agent_context')
    op.drop_table('agent_context')
