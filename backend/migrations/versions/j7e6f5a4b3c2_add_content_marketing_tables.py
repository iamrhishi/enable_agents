"""Add content marketing tables to PostgreSQL.

Migrates from SQLite to PostgreSQL for single-database architecture.

Revision ID: j7e6f5a4b3c2
Revises: i6d5e4f3a2b1
Create Date: 2026-07-18
"""

from alembic import op
import sqlalchemy as sa

revision = 'j7e6f5a4b3c2'
down_revision = 'i6d5e4f3a2b1'
branch_labels = None
depends_on = None


def upgrade():
    # Guarded with existence checks: cm_projects/cm_documents/cm_knowledge_graphs
    # were already created out-of-band (ad-hoc db.create_all()) in some
    # environments before this migration was written.
    bind = op.get_bind()
    insp = sa.inspect(bind)
    existing = set(insp.get_table_names())

    if 'cm_projects' not in existing:
        op.create_table(
            'cm_projects',
            sa.Column('project_id', sa.String(36), primary_key=True),
            sa.Column('user_id', sa.String(255), nullable=False, index=True),
            sa.Column('project_name', sa.String(255), nullable=False),
            sa.Column('description', sa.Text, nullable=True),
            sa.Column('industry', sa.String(255), nullable=True),
            sa.Column('sector', sa.String(255), nullable=True),
            sa.Column('function', sa.String(255), nullable=True),
            sa.Column('role', sa.String(255), nullable=True),
            sa.Column('metadata', sa.Text, nullable=True, server_default='{}'),
            sa.Column('created_at', sa.DateTime, server_default=sa.func.now()),
            sa.Column('updated_at', sa.DateTime, server_default=sa.func.now(), onupdate=sa.func.now()),
        )

    if 'cm_documents' not in existing:
        op.create_table(
            'cm_documents',
            sa.Column('doc_id', sa.String(36), primary_key=True),
            sa.Column('project_id', sa.String(36), sa.ForeignKey('cm_projects.project_id', ondelete='CASCADE'), nullable=False, index=True),
            sa.Column('file_name', sa.String(255), nullable=False),
            sa.Column('file_type', sa.String(50), nullable=True),
            sa.Column('file_path', sa.Text, nullable=True),
            sa.Column('file_size', sa.Integer, nullable=True),
            sa.Column('upload_date', sa.DateTime, server_default=sa.func.now()),
            sa.Column('document_type', sa.String(100), nullable=True),
            sa.Column('extracted_content', sa.Text, nullable=True),
        )

    if 'cm_knowledge_graphs' not in existing:
        op.create_table(
            'cm_knowledge_graphs',
            sa.Column('kg_id', sa.String(36), primary_key=True),
            sa.Column('project_id', sa.String(36), sa.ForeignKey('cm_projects.project_id', ondelete='CASCADE'), nullable=False, index=True),
            sa.Column('kg_data', sa.Text, nullable=True, server_default='{}'),
            sa.Column('entities', sa.Integer, nullable=True, server_default='0'),
            sa.Column('relationships', sa.Integer, nullable=True, server_default='0'),
            sa.Column('created_at', sa.DateTime, server_default=sa.func.now()),
        )

    if 'cm_generated_content' not in existing:
        op.create_table(
            'cm_generated_content',
            sa.Column('content_id', sa.String(36), primary_key=True),
            sa.Column('project_id', sa.String(36), sa.ForeignKey('cm_projects.project_id', ondelete='CASCADE'), nullable=False, index=True),
            sa.Column('channel', sa.String(50), nullable=True),
            sa.Column('content_type', sa.String(50), nullable=True),
            sa.Column('content', sa.Text, nullable=True),
            sa.Column('source_docs', sa.Text, nullable=True, server_default='[]'),
            sa.Column('domain_context', sa.Text, nullable=True, server_default='{}'),
            sa.Column('created_at', sa.DateTime, server_default=sa.func.now()),
            sa.Column('modified_at', sa.DateTime, server_default=sa.func.now(), onupdate=sa.func.now()),
        )

    if 'cm_conversations' not in existing:
        op.create_table(
            'cm_conversations',
            sa.Column('msg_id', sa.String(36), primary_key=True),
            sa.Column('project_id', sa.String(36), sa.ForeignKey('cm_projects.project_id', ondelete='CASCADE'), nullable=False, index=True),
            sa.Column('user_message', sa.Text, nullable=True),
            sa.Column('agent_response', sa.Text, nullable=True),
            sa.Column('context', sa.Text, nullable=True, server_default='{}'),
            sa.Column('timestamp', sa.DateTime, server_default=sa.func.now()),
        )


def downgrade():
    op.drop_table('cm_conversations')
    op.drop_table('cm_generated_content')
    op.drop_table('cm_knowledge_graphs')
    op.drop_table('cm_documents')
    op.drop_table('cm_projects')
