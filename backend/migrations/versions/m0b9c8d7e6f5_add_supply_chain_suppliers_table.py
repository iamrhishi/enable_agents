"""Add supply chain suppliers table.

Revision ID: m0b9c8d7e6f5
Revises: l9a8b7c6d5e4
Create Date: 2026-07-22
"""

from alembic import op
import sqlalchemy as sa

revision = 'm0b9c8d7e6f5'
down_revision = 'l9a8b7c6d5e4'
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        'sc_suppliers',
        sa.Column('supplier_id', sa.String(36), primary_key=True),
        sa.Column('project_id', sa.String(36), nullable=False, index=True),
        sa.Column('user_id', sa.String(255), nullable=False, index=True),
        sa.Column('name', sa.String(255), nullable=False),
        sa.Column('location', sa.String(255), nullable=True),
        sa.Column('capacity', sa.String(255), nullable=True),
        sa.Column('certifications', sa.Text, nullable=True, server_default='[]'),
        sa.Column('capabilities', sa.Text, nullable=True, server_default='[]'),
        sa.Column('audit_status', sa.String(20), nullable=True, server_default='pending'),
        sa.Column('score', sa.Integer, nullable=True),
        sa.Column('audit_date', sa.String(20), nullable=True),
        sa.Column('created_at', sa.DateTime, server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime, server_default=sa.func.now(), onupdate=sa.func.now()),
    )


def downgrade():
    op.drop_table('sc_suppliers')
