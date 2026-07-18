"""Add teams and projects tables for platform-wide persistence.

Revision ID: i6d5e4f3a2b1
Revises: h5c4d3e2f1a0
Create Date: 2026-07-18
"""

from alembic import op
import sqlalchemy as sa

revision = 'i6d5e4f3a2b1'
down_revision = 'h5c4d3e2f1a0'
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        'teams',
        sa.Column('id', sa.Integer, primary_key=True, autoincrement=True),
        sa.Column('team_id', sa.String(36), nullable=False, unique=True, index=True),
        sa.Column('owner_id', sa.String(255), nullable=False, index=True),
        sa.Column('name', sa.String(255), nullable=True),
        sa.Column('created_at', sa.DateTime, server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime, server_default=sa.func.now(), onupdate=sa.func.now()),
    )

    op.create_table(
        'team_members',
        sa.Column('id', sa.Integer, primary_key=True, autoincrement=True),
        sa.Column('member_id', sa.String(36), nullable=False, unique=True, index=True),
        sa.Column('team_id', sa.String(36), sa.ForeignKey('teams.team_id', ondelete='CASCADE'), nullable=False, index=True),
        sa.Column('user_id', sa.String(255), nullable=False, index=True),
        sa.Column('name', sa.String(255), nullable=True),
        sa.Column('role', sa.String(50), nullable=False, server_default='member'),
        sa.Column('joined_at', sa.DateTime, server_default=sa.func.now()),
        sa.UniqueConstraint('team_id', 'user_id', name='uq_team_member'),
    )

    op.create_table(
        'projects',
        sa.Column('id', sa.Integer, primary_key=True, autoincrement=True),
        sa.Column('project_id', sa.String(36), nullable=False, unique=True, index=True),
        sa.Column('team_id', sa.String(36), sa.ForeignKey('teams.team_id', ondelete='CASCADE'), nullable=False, index=True),
        sa.Column('owner_id', sa.String(255), nullable=False, index=True),
        sa.Column('name', sa.String(255), nullable=False),
        sa.Column('description', sa.Text, nullable=True),
        sa.Column('status', sa.String(50), nullable=False, server_default='active', index=True),
        sa.Column('agents', sa.Text, nullable=False, server_default='[]'),
        sa.Column('data', sa.Text, nullable=False, server_default='{}'),
        sa.Column('created_at', sa.DateTime, server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime, server_default=sa.func.now(), onupdate=sa.func.now()),
    )

    op.create_table(
        'pending_invites',
        sa.Column('id', sa.Integer, primary_key=True, autoincrement=True),
        sa.Column('invite_id', sa.String(36), nullable=False, unique=True, index=True),
        sa.Column('team_id', sa.String(36), sa.ForeignKey('teams.team_id', ondelete='CASCADE'), nullable=False, index=True),
        sa.Column('email', sa.String(255), nullable=False, index=True),
        sa.Column('role', sa.String(50), nullable=False, server_default='member'),
        sa.Column('invited_by', sa.String(255), nullable=False),
        sa.Column('created_at', sa.DateTime, server_default=sa.func.now()),
        sa.UniqueConstraint('team_id', 'email', name='uq_pending_invite'),
    )

    # Add FK from processed_documents.project_id to projects.project_id
    # (deferred from h5c4d3e2f1a0 since projects table didn't exist)
    op.create_foreign_key(
        'fk_processed_documents_project',
        'processed_documents',
        'projects',
        ['project_id'],
        ['project_id'],
        ondelete='SET NULL'
    )


def downgrade():
    # Drop FK before dropping projects table
    op.drop_constraint('fk_processed_documents_project', 'processed_documents', type_='foreignkey')
    op.drop_table('pending_invites')
    op.drop_table('projects')
    op.drop_table('team_members')
    op.drop_table('teams')
