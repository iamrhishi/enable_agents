"""Add event networking tables.

Revision ID: l9a8b7c6d5e4
Revises: k8f7a6b5c4d3
Create Date: 2026-07-22
"""

from alembic import op
import sqlalchemy as sa

revision = 'l9a8b7c6d5e4'
down_revision = 'k8f7a6b5c4d3'
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        'en_events',
        sa.Column('event_id', sa.String(36), primary_key=True),
        sa.Column('user_id', sa.String(255), nullable=False, index=True),
        sa.Column('name', sa.String(255), nullable=False),
        sa.Column('description', sa.Text, nullable=True),
        sa.Column('date', sa.String(20), nullable=True),
        sa.Column('location', sa.String(255), nullable=True),
        sa.Column('created_at', sa.DateTime, server_default=sa.func.now()),
    )

    op.create_table(
        'en_attendees',
        sa.Column('attendee_id', sa.String(36), primary_key=True),
        sa.Column('event_id', sa.String(36), sa.ForeignKey('en_events.event_id', ondelete='CASCADE'), nullable=False, index=True),
        sa.Column('name', sa.String(255), nullable=False),
        sa.Column('email', sa.String(255), nullable=True),
        sa.Column('company', sa.String(255), nullable=True),
        sa.Column('role', sa.String(255), nullable=True),
        sa.Column('linkedin', sa.String(500), nullable=True),
        sa.Column('interests', sa.Text, nullable=True, server_default='[]'),
        sa.Column('notes', sa.Text, nullable=True),
        sa.Column('priority', sa.String(20), nullable=True, server_default='medium'),
        sa.Column('last_contact', sa.String(20), nullable=True),
        sa.Column('follow_up_date', sa.String(20), nullable=True),
        sa.Column('created_at', sa.DateTime, server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime, server_default=sa.func.now(), onupdate=sa.func.now()),
    )


def downgrade():
    op.drop_table('en_attendees')
    op.drop_table('en_events')
