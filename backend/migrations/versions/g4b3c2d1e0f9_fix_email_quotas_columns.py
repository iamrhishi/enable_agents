"""Fix email_extraction_quotas column names

Revision ID: g4b3c2d1e0f9
Revises: f3a2b1c4d5e6
Create Date: 2026-07-12

Rename columns to match model:
- total_allowed -> monthly_limit
- used_count -> emails_used_this_month
Add reset_date column
"""
from alembic import op
import sqlalchemy as sa


revision = 'g4b3c2d1e0f9'
down_revision = 'f3a2b1c4d5e6'
branch_labels = None
depends_on = None


def upgrade():
    with op.batch_alter_table('email_extraction_quotas', schema=None) as batch_op:
        # Rename columns to match model
        batch_op.alter_column('total_allowed', new_column_name='monthly_limit')
        batch_op.alter_column('used_count', new_column_name='emails_used_this_month')
        # Add reset_date column
        batch_op.add_column(sa.Column('reset_date', sa.DateTime(), nullable=True))


def downgrade():
    with op.batch_alter_table('email_extraction_quotas', schema=None) as batch_op:
        batch_op.drop_column('reset_date')
        batch_op.alter_column('emails_used_this_month', new_column_name='used_count')
        batch_op.alter_column('monthly_limit', new_column_name='total_allowed')
