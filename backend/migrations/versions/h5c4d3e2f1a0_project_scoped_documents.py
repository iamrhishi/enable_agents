"""Add project_id to documents for project-scoped document library.

Documents now belong to a project instead of a user.
- Add project_id column
- Rename user_id to uploaded_by (track who uploaded, not owner)
- Add index on project_id for faster queries

Revision ID: h5c4d3e2f1a0
Revises: g4b3c2d1e0f9
Create Date: 2026-07-16
"""

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'h5c4d3e2f1a0'
down_revision = 'g4b3c2d1e0f9'
branch_labels = None
depends_on = None


def upgrade():
    bind = op.get_bind()
    insp = sa.inspect(bind)
    cols = {c['name'] for c in insp.get_columns('processed_documents')}
    indexes = {ix['name'] for ix in insp.get_indexes('processed_documents')}

    # Add project_id column (nullable initially for existing data)
    if 'project_id' not in cols:
        op.add_column(
            'processed_documents',
            sa.Column('project_id', sa.String(36), nullable=True)
        )

    # Add index on project_id
    if 'ix_processed_documents_project_id' not in indexes:
        op.create_index(
            'ix_processed_documents_project_id',
            'processed_documents',
            ['project_id']
        )

    # Rename user_id to uploaded_by
    if 'user_id' in cols and 'uploaded_by' not in cols:
        op.alter_column(
            'processed_documents',
            'user_id',
            new_column_name='uploaded_by'
        )

    # Note: FK to projects table will be added by a later migration
    # (i6d5e4f3a2b1) after the projects table is created.
    # Don't create FK here as projects table doesn't exist yet.


def downgrade():
    # Remove foreign key if exists
    try:
        op.drop_constraint(
            'fk_processed_documents_project',
            'processed_documents',
            type_='foreignkey'
        )
    except Exception:
        pass

    # Rename uploaded_by back to user_id
    op.alter_column(
        'processed_documents',
        'uploaded_by',
        new_column_name='user_id'
    )

    # Drop index
    op.drop_index(
        'ix_processed_documents_project_id',
        'processed_documents'
    )

    # Drop project_id column
    op.drop_column('processed_documents', 'project_id')
