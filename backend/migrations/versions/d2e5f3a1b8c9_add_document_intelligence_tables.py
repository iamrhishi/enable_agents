"""add document intelligence tables with pgvector

Revision ID: d2e5f3a1b8c9
Revises: c1a4d2e9b7f0
Create Date: 2026-05-27
"""
from alembic import op
import sqlalchemy as sa
from pgvector.sqlalchemy import Vector


revision = 'd2e5f3a1b8c9'
down_revision = 'c1a4d2e9b7f0'
branch_labels = None
depends_on = None


def upgrade():
    # Enable pgvector extension
    op.execute('CREATE EXTENSION IF NOT EXISTS vector')

    # Create processed_documents table
    op.create_table(
        'processed_documents',
        sa.Column('document_id', sa.String(36), primary_key=True),
        sa.Column('user_id', sa.String(255), nullable=False),
        sa.Column('file_name', sa.String(255), nullable=False),
        sa.Column('file_type', sa.String(50), nullable=True),
        sa.Column('file_size', sa.Integer(), nullable=True),
        sa.Column('file_path', sa.Text(), nullable=True),
        sa.Column('status', sa.String(50), nullable=True, default='pending'),
        sa.Column('processing_stage', sa.String(100), nullable=True),
        sa.Column('processing_progress', sa.Float(), nullable=True, default=0.0),
        sa.Column('error_message', sa.Text(), nullable=True),
        sa.Column('extracted_text', sa.Text(), nullable=True),
        sa.Column('page_count', sa.Integer(), nullable=True),
        sa.Column('word_count', sa.Integer(), nullable=True),
        sa.Column('chunk_count', sa.Integer(), nullable=True, default=0),
        sa.Column('entity_count', sa.Integer(), nullable=True, default=0),
        sa.Column('document_type', sa.String(100), nullable=True),
        sa.Column('metadata', sa.Text(), nullable=True, default='{}'),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.Column('updated_at', sa.DateTime(), nullable=True),
        sa.Column('processed_at', sa.DateTime(), nullable=True),
    )
    op.create_index('ix_processed_documents_user_id', 'processed_documents', ['user_id'])
    op.create_index('ix_processed_documents_status', 'processed_documents', ['status'])

    # Create document_chunks table with pgvector
    op.create_table(
        'document_chunks',
        sa.Column('chunk_id', sa.String(36), primary_key=True),
        sa.Column('document_id', sa.String(36), sa.ForeignKey('processed_documents.document_id', ondelete='CASCADE'), nullable=False),
        sa.Column('chunk_index', sa.Integer(), nullable=False),
        sa.Column('content', sa.Text(), nullable=False),
        sa.Column('embedding', Vector(1536), nullable=True),
        sa.Column('metadata', sa.Text(), nullable=True, default='{}'),
        sa.Column('created_at', sa.DateTime(), nullable=True),
    )
    op.create_index('ix_document_chunks_document_id', 'document_chunks', ['document_id'])

    # Create ivfflat index for vector similarity search
    # Note: This requires at least 100 rows to work efficiently
    # For small datasets, you can use HNSW instead or skip the index
    op.execute('''
        CREATE INDEX IF NOT EXISTS ix_document_chunks_embedding
        ON document_chunks
        USING ivfflat (embedding vector_cosine_ops)
        WITH (lists = 100)
    ''')

    # Create document_entities table
    op.create_table(
        'document_entities',
        sa.Column('entity_id', sa.String(36), primary_key=True),
        sa.Column('document_id', sa.String(36), sa.ForeignKey('processed_documents.document_id', ondelete='CASCADE'), nullable=False),
        sa.Column('entity_type', sa.String(100), nullable=False),
        sa.Column('entity_name', sa.String(500), nullable=False),
        sa.Column('normalized_name', sa.String(500), nullable=True),
        sa.Column('properties', sa.Text(), nullable=True, default='{}'),
        sa.Column('chunk_ids', sa.Text(), nullable=True, default='[]'),
        sa.Column('confidence', sa.Float(), nullable=True, default=1.0),
        sa.Column('created_at', sa.DateTime(), nullable=True),
    )
    op.create_index('ix_document_entities_document_id', 'document_entities', ['document_id'])
    op.create_index('ix_document_entities_entity_type', 'document_entities', ['entity_type'])
    op.create_index('ix_document_entities_normalized_name', 'document_entities', ['normalized_name'])


def downgrade():
    op.drop_index('ix_document_entities_normalized_name', table_name='document_entities')
    op.drop_index('ix_document_entities_entity_type', table_name='document_entities')
    op.drop_index('ix_document_entities_document_id', table_name='document_entities')
    op.drop_table('document_entities')

    op.execute('DROP INDEX IF EXISTS ix_document_chunks_embedding')
    op.drop_index('ix_document_chunks_document_id', table_name='document_chunks')
    op.drop_table('document_chunks')

    op.drop_index('ix_processed_documents_status', table_name='processed_documents')
    op.drop_index('ix_processed_documents_user_id', table_name='processed_documents')
    op.drop_table('processed_documents')

    # Note: We don't drop the vector extension as other tables might use it
