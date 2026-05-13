"""core app tables (users, oauth, quotas, campaigns, saved projects)

Revision ID: 7c2fba01add3
Revises: 62214ff70af8
Create Date: 2026-05-13

The prior "initial" revision only created email_extraction_usage_logs; app models
require these tables for login and related features.
"""
from alembic import op
import sqlalchemy as sa


revision = '7c2fba01add3'
down_revision = '62214ff70af8'
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        'users',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('username', sa.String(length=80), nullable=False),
        sa.Column('password', sa.String(length=512), nullable=False),
        sa.Column('first_name', sa.String(length=80), nullable=True),
        sa.Column('last_name', sa.String(length=80), nullable=True),
        sa.Column('email', sa.String(length=120), nullable=True),
        sa.Column('company', sa.String(length=120), nullable=True),
        sa.Column('linkedin', sa.String(length=256), nullable=True),
        sa.Column('short_intro', sa.String(length=256), nullable=True),
        sa.Column('company_intro', sa.String(length=256), nullable=True),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('username'),
    )

    op.create_table(
        'google_oauth_tokens',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('username', sa.String(length=80), nullable=False),
        sa.Column('token', sa.Text(), nullable=False),
        sa.Column('refresh_token', sa.Text(), nullable=True),
        sa.Column('token_uri', sa.String(length=512), nullable=True),
        sa.Column('client_id', sa.String(length=512), nullable=True),
        sa.Column('client_secret', sa.String(length=512), nullable=True),
        sa.Column('scopes', sa.Text(), nullable=True),
        sa.ForeignKeyConstraint(['username'], ['users.username'], ),
        sa.PrimaryKeyConstraint('id'),
    )

    op.create_table(
        'email_extraction_quotas',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('username', sa.String(length=120), nullable=False),
        sa.Column('total_allowed', sa.Integer(), nullable=False),
        sa.Column('used_count', sa.Integer(), nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint('id'),
    )
    with op.batch_alter_table('email_extraction_quotas', schema=None) as batch_op:
        batch_op.create_index(batch_op.f('ix_email_extraction_quotas_username'), ['username'], unique=True)

    op.create_table(
        'email_campaigns',
        sa.Column('id', sa.String(length=36), nullable=False),
        sa.Column('name', sa.String(length=255), nullable=False),
        sa.Column('subject', sa.String(length=255), nullable=False),
        sa.Column('username', sa.String(length=120), nullable=True),
        sa.Column('sender_email', sa.String(length=255), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint('id'),
    )
    with op.batch_alter_table('email_campaigns', schema=None) as batch_op:
        batch_op.create_index(batch_op.f('ix_email_campaigns_username'), ['username'], unique=False)
        batch_op.create_index(batch_op.f('ix_email_campaigns_sender_email'), ['sender_email'], unique=False)

    op.create_table(
        'email_campaign_recipients',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('campaign_id', sa.String(length=36), nullable=False),
        sa.Column('receiver_email', sa.String(length=255), nullable=False),
        sa.Column('receiver_name', sa.String(length=255), nullable=True),
        sa.Column('status', sa.String(length=50), nullable=True),
        sa.Column('reply_status', sa.String(length=50), nullable=True),
        sa.Column('message_id', sa.String(length=255), nullable=True),
        sa.Column('thread_id', sa.String(length=255), nullable=True),
        sa.Column('sent_at', sa.DateTime(), nullable=False),
        sa.Column('replied_at', sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(['campaign_id'], ['email_campaigns.id'], ),
        sa.PrimaryKeyConstraint('id'),
    )
    with op.batch_alter_table('email_campaign_recipients', schema=None) as batch_op:
        batch_op.create_index(batch_op.f('ix_email_campaign_recipients_receiver_email'), ['receiver_email'], unique=False)

    op.create_table(
        'saved_projects',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('username', sa.String(length=120), nullable=False),
        sa.Column('name', sa.String(length=255), nullable=False),
        sa.Column('query_used', sa.String(length=512), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint('id'),
    )
    with op.batch_alter_table('saved_projects', schema=None) as batch_op:
        batch_op.create_index(batch_op.f('ix_saved_projects_username'), ['username'], unique=False)

    op.create_table(
        'saved_leads',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('project_id', sa.Integer(), nullable=False),
        sa.Column('name', sa.String(length=255), nullable=True),
        sa.Column('website', sa.String(length=512), nullable=True),
        sa.Column('phone', sa.String(length=100), nullable=True),
        sa.Column('address', sa.String(length=512), nullable=True),
        sa.Column('emails', sa.Text(), nullable=True),
        sa.Column('linkedin_links', sa.Text(), nullable=True),
        sa.Column('social_links', sa.Text(), nullable=True),
        sa.Column('has_extracted', sa.Boolean(), nullable=True),
        sa.Column('raw_data', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(['project_id'], ['saved_projects.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id'),
    )


def downgrade():
    op.drop_table('saved_leads')
    with op.batch_alter_table('saved_projects', schema=None) as batch_op:
        batch_op.drop_index(batch_op.f('ix_saved_projects_username'))

    op.drop_table('saved_projects')
    with op.batch_alter_table('email_campaign_recipients', schema=None) as batch_op:
        batch_op.drop_index(batch_op.f('ix_email_campaign_recipients_receiver_email'))

    op.drop_table('email_campaign_recipients')
    with op.batch_alter_table('email_campaigns', schema=None) as batch_op:
        batch_op.drop_index(batch_op.f('ix_email_campaigns_sender_email'))
        batch_op.drop_index(batch_op.f('ix_email_campaigns_username'))

    op.drop_table('email_campaigns')
    with op.batch_alter_table('email_extraction_quotas', schema=None) as batch_op:
        batch_op.drop_index(batch_op.f('ix_email_extraction_quotas_username'))

    op.drop_table('email_extraction_quotas')
    op.drop_table('google_oauth_tokens')
    op.drop_table('users')
