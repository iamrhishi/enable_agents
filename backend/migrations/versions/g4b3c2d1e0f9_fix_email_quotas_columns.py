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
    # Guarded with existence checks: some environments already had this
    # migration's changes applied out-of-band, which made the plain
    # version fail (rename source column missing / column already exists).
    bind = op.get_bind()
    insp = sa.inspect(bind)
    cols = {c['name'] for c in insp.get_columns('email_extraction_quotas')}

    with op.batch_alter_table('email_extraction_quotas', schema=None) as batch_op:
        if 'total_allowed' in cols and 'monthly_limit' not in cols:
            batch_op.alter_column('total_allowed', new_column_name='monthly_limit')
        if 'used_count' in cols and 'emails_used_this_month' not in cols:
            batch_op.alter_column('used_count', new_column_name='emails_used_this_month')
        if 'reset_date' not in cols:
            batch_op.add_column(sa.Column('reset_date', sa.DateTime(), nullable=True))


def downgrade():
    with op.batch_alter_table('email_extraction_quotas', schema=None) as batch_op:
        batch_op.drop_column('reset_date')
        batch_op.alter_column('emails_used_this_month', new_column_name='used_count')
        batch_op.alter_column('monthly_limit', new_column_name='total_allowed')
