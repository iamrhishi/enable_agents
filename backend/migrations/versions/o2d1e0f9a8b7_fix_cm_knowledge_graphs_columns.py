"""Fix cm_knowledge_graphs columns to match the current model.

Some environments had this table created ad-hoc from an older version of
the CMKnowledgeGraph model (graph_data/updated_at, no entities/
relationships) before the real migration for this table existed. That
migration only creates the table if missing, so it never fixed already-
drifted columns - this backfills them.

Revision ID: o2d1e0f9a8b7
Revises: n1c0d9e8f7a6
Create Date: 2026-07-24
"""

from alembic import op
import sqlalchemy as sa

revision = 'o2d1e0f9a8b7'
down_revision = 'n1c0d9e8f7a6'
branch_labels = None
depends_on = None


def upgrade():
    bind = op.get_bind()
    insp = sa.inspect(bind)
    cols = {c['name'] for c in insp.get_columns('cm_knowledge_graphs')}

    if 'kg_data' not in cols:
        op.execute("ALTER TABLE cm_knowledge_graphs ADD COLUMN kg_data TEXT DEFAULT '{}'")
        if 'graph_data' in cols:
            op.execute("UPDATE cm_knowledge_graphs SET kg_data = graph_data WHERE graph_data IS NOT NULL")
    if 'entities' not in cols:
        op.execute("ALTER TABLE cm_knowledge_graphs ADD COLUMN entities INTEGER DEFAULT 0")
    if 'relationships' not in cols:
        op.execute("ALTER TABLE cm_knowledge_graphs ADD COLUMN relationships INTEGER DEFAULT 0")


def downgrade():
    op.execute("ALTER TABLE cm_knowledge_graphs DROP COLUMN IF EXISTS relationships")
    op.execute("ALTER TABLE cm_knowledge_graphs DROP COLUMN IF EXISTS entities")
    op.execute("ALTER TABLE cm_knowledge_graphs DROP COLUMN IF EXISTS kg_data")
