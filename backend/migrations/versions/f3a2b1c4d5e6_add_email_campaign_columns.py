"""Add missing email campaign columns

Revision ID: f3a2b1c4d5e6
Revises: ee4bd0b1fe51
Create Date: 2026-06-29

Add body_template, sent_at, status to email_campaigns.
Add company, personalised_body, reply_subject, reply_snippet, reply_body, error_message to email_campaign_recipients.
"""
from alembic import op
import sqlalchemy as sa


revision = 'f3a2b1c4d5e6'
down_revision = 'ee4bd0b1fe51'
branch_labels = None
depends_on = None


def upgrade():
    # Add missing columns to email_campaigns
    with op.batch_alter_table('email_campaigns', schema=None) as batch_op:
        batch_op.add_column(sa.Column('body_template', sa.Text(), nullable=True))
        batch_op.add_column(sa.Column('sent_at', sa.DateTime(), nullable=True))
        batch_op.add_column(sa.Column('status', sa.String(length=50), nullable=True, server_default='draft'))

    # Add missing columns to email_campaign_recipients
    with op.batch_alter_table('email_campaign_recipients', schema=None) as batch_op:
        batch_op.add_column(sa.Column('company', sa.String(length=255), nullable=True))
        batch_op.add_column(sa.Column('personalised_body', sa.Text(), nullable=True))
        batch_op.add_column(sa.Column('reply_subject', sa.String(length=512), nullable=True))
        batch_op.add_column(sa.Column('reply_snippet', sa.Text(), nullable=True))
        batch_op.add_column(sa.Column('reply_body', sa.Text(), nullable=True))
        batch_op.add_column(sa.Column('error_message', sa.Text(), nullable=True))


def downgrade():
    with op.batch_alter_table('email_campaign_recipients', schema=None) as batch_op:
        batch_op.drop_column('error_message')
        batch_op.drop_column('reply_body')
        batch_op.drop_column('reply_snippet')
        batch_op.drop_column('reply_subject')
        batch_op.drop_column('personalised_body')
        batch_op.drop_column('company')

    with op.batch_alter_table('email_campaigns', schema=None) as batch_op:
        batch_op.drop_column('status')
        batch_op.drop_column('sent_at')
        batch_op.drop_column('body_template')
