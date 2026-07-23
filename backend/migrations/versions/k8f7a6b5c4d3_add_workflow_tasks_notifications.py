"""Add workflow_tasks and notifications tables

Revision ID: k8f7a6b5c4d3
Revises: j7e6f5a4b3c2
Create Date: 2026-07-21 05:10:00.000000

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'k8f7a6b5c4d3'
down_revision = 'j7e6f5a4b3c2'
branch_labels = None
depends_on = None


def upgrade():
    # Create workflow_tasks table
    op.create_table('workflow_tasks',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('task_id', sa.String(36), nullable=False),
        sa.Column('instance_id', sa.String(36), nullable=False),
        sa.Column('stage_id', sa.String(100), nullable=False),
        sa.Column('title', sa.String(255), nullable=False),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('assigned_to', sa.String(255), nullable=True),
        sa.Column('status', sa.String(20), nullable=False, server_default='pending'),
        sa.Column('is_required', sa.Boolean(), nullable=False, server_default='true'),
        sa.Column('source', sa.String(20), nullable=False, server_default='manual'),
        sa.Column('created_by', sa.String(255), nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.Column('completed_at', sa.DateTime(), nullable=True),
        sa.Column('completed_by', sa.String(255), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_workflow_tasks_task_id', 'workflow_tasks', ['task_id'], unique=True)
    op.create_index('ix_workflow_tasks_instance_id', 'workflow_tasks', ['instance_id'], unique=False)
    op.create_index('ix_workflow_tasks_stage_id', 'workflow_tasks', ['stage_id'], unique=False)
    op.create_index('ix_workflow_tasks_assigned_to', 'workflow_tasks', ['assigned_to'], unique=False)
    op.create_index('ix_workflow_tasks_instance_stage', 'workflow_tasks', ['instance_id', 'stage_id'], unique=False)

    # Create notifications table
    op.create_table('notifications',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('notification_id', sa.String(36), nullable=False),
        sa.Column('user_id', sa.String(255), nullable=False),
        sa.Column('type', sa.String(50), nullable=False),
        sa.Column('title', sa.String(255), nullable=False),
        sa.Column('message', sa.Text(), nullable=True),
        sa.Column('link', sa.String(500), nullable=True),
        sa.Column('is_read', sa.Boolean(), nullable=False, server_default='false'),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.Column('workflow_id', sa.String(36), nullable=True),
        sa.Column('task_id', sa.String(36), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_notifications_notification_id', 'notifications', ['notification_id'], unique=True)
    op.create_index('ix_notifications_user_id', 'notifications', ['user_id'], unique=False)
    op.create_index('ix_notifications_user_unread', 'notifications', ['user_id', 'is_read'], unique=False)


def downgrade():
    op.drop_index('ix_notifications_user_unread', table_name='notifications')
    op.drop_index('ix_notifications_user_id', table_name='notifications')
    op.drop_index('ix_notifications_notification_id', table_name='notifications')
    op.drop_table('notifications')

    op.drop_index('ix_workflow_tasks_instance_stage', table_name='workflow_tasks')
    op.drop_index('ix_workflow_tasks_assigned_to', table_name='workflow_tasks')
    op.drop_index('ix_workflow_tasks_stage_id', table_name='workflow_tasks')
    op.drop_index('ix_workflow_tasks_instance_id', table_name='workflow_tasks')
    op.drop_index('ix_workflow_tasks_task_id', table_name='workflow_tasks')
    op.drop_table('workflow_tasks')
