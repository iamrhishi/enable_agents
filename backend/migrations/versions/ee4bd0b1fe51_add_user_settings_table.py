"""add user_settings table

Revision ID: ee4bd0b1fe51
Revises: d2e5f3a1b8c9
Create Date: 2026-05-28 04:04:18.475730

"""
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = 'ee4bd0b1fe51'
down_revision = 'd2e5f3a1b8c9'
branch_labels = None
depends_on = None


def upgrade():
    """Create user_settings table for encrypted credential storage."""
    op.create_table(
        'user_settings',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('user_id', sa.String(length=255), nullable=False),
        sa.Column('category', sa.String(length=50), nullable=False),
        sa.Column('key', sa.String(length=100), nullable=False),
        sa.Column('value_encrypted', sa.Text(), nullable=False),
        sa.Column('created_at', sa.DateTime(), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(), server_default=sa.text('now()'), nullable=False),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('user_id', 'category', 'key', name='uq_user_setting')
    )
    op.create_index('ix_user_settings_user_id', 'user_settings', ['user_id'])
    op.create_index('ix_user_settings_category', 'user_settings', ['category'])


def downgrade():
    """Drop user_settings table."""
    op.drop_index('ix_user_settings_category', table_name='user_settings')
    op.drop_index('ix_user_settings_user_id', table_name='user_settings')
    op.drop_table('user_settings')
