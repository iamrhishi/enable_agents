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
    # Uses IF NOT EXISTS because some environments already had a subset of
    # these columns applied out-of-band before this migration existed,
    # which made the plain add_column version fail with DuplicateColumn.
    op.execute("ALTER TABLE email_campaigns ADD COLUMN IF NOT EXISTS body_template TEXT")
    op.execute("ALTER TABLE email_campaigns ADD COLUMN IF NOT EXISTS sent_at TIMESTAMP")
    op.execute("ALTER TABLE email_campaigns ADD COLUMN IF NOT EXISTS status VARCHAR(50) DEFAULT 'draft'")

    op.execute("ALTER TABLE email_campaign_recipients ADD COLUMN IF NOT EXISTS company VARCHAR(255)")
    op.execute("ALTER TABLE email_campaign_recipients ADD COLUMN IF NOT EXISTS personalised_body TEXT")
    op.execute("ALTER TABLE email_campaign_recipients ADD COLUMN IF NOT EXISTS reply_subject VARCHAR(512)")
    op.execute("ALTER TABLE email_campaign_recipients ADD COLUMN IF NOT EXISTS reply_snippet TEXT")
    op.execute("ALTER TABLE email_campaign_recipients ADD COLUMN IF NOT EXISTS reply_body TEXT")
    op.execute("ALTER TABLE email_campaign_recipients ADD COLUMN IF NOT EXISTS error_message TEXT")


def downgrade():
    op.execute("ALTER TABLE email_campaign_recipients DROP COLUMN IF EXISTS error_message")
    op.execute("ALTER TABLE email_campaign_recipients DROP COLUMN IF EXISTS reply_body")
    op.execute("ALTER TABLE email_campaign_recipients DROP COLUMN IF EXISTS reply_snippet")
    op.execute("ALTER TABLE email_campaign_recipients DROP COLUMN IF EXISTS reply_subject")
    op.execute("ALTER TABLE email_campaign_recipients DROP COLUMN IF EXISTS personalised_body")
    op.execute("ALTER TABLE email_campaign_recipients DROP COLUMN IF EXISTS company")

    op.execute("ALTER TABLE email_campaigns DROP COLUMN IF EXISTS status")
    op.execute("ALTER TABLE email_campaigns DROP COLUMN IF EXISTS sent_at")
    op.execute("ALTER TABLE email_campaigns DROP COLUMN IF EXISTS body_template")
