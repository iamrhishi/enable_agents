"""Executive Assistant Agent — SQLAlchemy models."""
from datetime import datetime
import json
from typing import Dict, Any, List
from core.database import db


class ExecTask(db.Model):
    """Task managed by Executive Assistant."""
    __tablename__ = "exec_tasks"

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    task_id = db.Column(db.String(36), unique=True, nullable=False, index=True)
    user_id = db.Column(db.String(255), nullable=False, index=True)
    project_id = db.Column(db.String(36), nullable=True, index=True)  # Links to platform Project
    title = db.Column(db.String(500), nullable=False)
    description = db.Column(db.Text, nullable=True)
    assigned_to = db.Column(db.String(36), nullable=True, index=True)  # Stakeholder ID
    due_date = db.Column(db.DateTime, nullable=True)
    priority = db.Column(db.String(20), default="Medium")  # Low | Medium | High | Urgent
    status = db.Column(db.String(50), default="Pending", index=True)  # Pending | In Progress | Completed | Cancelled
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    reminders = db.relationship("ExecReminder", back_populates="task", cascade="all, delete-orphan")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.task_id,
            "projectId": self.project_id,
            "title": self.title,
            "description": self.description,
            "assignedTo": self.assigned_to,
            "dueDate": self.due_date.strftime("%Y-%m-%d") if self.due_date else None,
            "priority": self.priority,
            "status": self.status,
            "createdAt": self.created_at.isoformat() if self.created_at else None,
            "updatedAt": self.updated_at.isoformat() if self.updated_at else None,
        }


class ExecReminder(db.Model):
    """Reminder for a task."""
    __tablename__ = "exec_reminders"

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    reminder_id = db.Column(db.String(36), unique=True, nullable=False, index=True)
    task_id = db.Column(db.String(36), db.ForeignKey("exec_tasks.task_id", ondelete="CASCADE"), nullable=False, index=True)
    user_id = db.Column(db.String(255), nullable=False, index=True)
    remind_at = db.Column(db.DateTime, nullable=False)
    channel = db.Column(db.String(50), default="email")  # email | whatsapp | sms | push
    recipient = db.Column(db.String(255), nullable=True)  # Email or phone
    message = db.Column(db.Text, nullable=True)
    sent = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    task = db.relationship("ExecTask", back_populates="reminders")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.reminder_id,
            "taskId": self.task_id,
            "remindAt": self.remind_at.isoformat() if self.remind_at else None,
            "channel": self.channel,
            "recipient": self.recipient,
            "message": self.message,
            "sent": self.sent,
            "createdAt": self.created_at.isoformat() if self.created_at else None,
        }


class ExecStakeholder(db.Model):
    """Stakeholder / team member for task assignment."""
    __tablename__ = "exec_stakeholders"

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    stakeholder_id = db.Column(db.String(36), unique=True, nullable=False, index=True)
    user_id = db.Column(db.String(255), nullable=False, index=True)  # Owner of this contact
    name = db.Column(db.String(255), nullable=False)
    email = db.Column(db.String(255), nullable=True)
    phone = db.Column(db.String(50), nullable=True)
    role = db.Column(db.String(100), nullable=True)
    _project_ids = db.Column("project_ids", db.Text, default="[]")  # JSON array
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    @property
    def project_ids(self) -> List[str]:
        return json.loads(self._project_ids or "[]")

    @project_ids.setter
    def project_ids(self, value: List[str]):
        self._project_ids = json.dumps(value)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.stakeholder_id,
            "name": self.name,
            "email": self.email,
            "phone": self.phone,
            "role": self.role,
            "projects": self.project_ids,
            "createdAt": self.created_at.isoformat() if self.created_at else None,
        }
