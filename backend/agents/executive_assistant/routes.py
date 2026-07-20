"""Executive Assistant Agent — Flask Blueprint."""
from flask import Blueprint
from . import service

executive_assistant_bp = Blueprint("executive_assistant", __name__, url_prefix="/api/executive-assistant")

# =============================================================================
# Tasks
# =============================================================================

@executive_assistant_bp.get("/tasks")
def list_tasks():
    return service.list_tasks()


@executive_assistant_bp.post("/tasks")
def create_task():
    return service.create_task()


@executive_assistant_bp.get("/tasks/<task_id>")
def get_task(task_id: str):
    return service.get_task(task_id)


@executive_assistant_bp.put("/tasks/<task_id>")
def update_task(task_id: str):
    return service.update_task(task_id)


@executive_assistant_bp.delete("/tasks/<task_id>")
def delete_task(task_id: str):
    return service.delete_task(task_id)


# =============================================================================
# Reminders
# =============================================================================

@executive_assistant_bp.get("/reminders")
def list_reminders():
    return service.list_reminders()


@executive_assistant_bp.post("/reminders")
def create_reminder():
    return service.create_reminder()


@executive_assistant_bp.delete("/reminders/<reminder_id>")
def delete_reminder(reminder_id: str):
    return service.delete_reminder(reminder_id)


# =============================================================================
# Stakeholders
# =============================================================================

@executive_assistant_bp.get("/stakeholders")
def list_stakeholders():
    return service.list_stakeholders()


@executive_assistant_bp.post("/stakeholders")
def create_stakeholder():
    return service.create_stakeholder()


@executive_assistant_bp.get("/stakeholders/<stakeholder_id>")
def get_stakeholder(stakeholder_id: str):
    return service.get_stakeholder(stakeholder_id)


@executive_assistant_bp.put("/stakeholders/<stakeholder_id>")
def update_stakeholder(stakeholder_id: str):
    return service.update_stakeholder(stakeholder_id)


@executive_assistant_bp.delete("/stakeholders/<stakeholder_id>")
def delete_stakeholder(stakeholder_id: str):
    return service.delete_stakeholder(stakeholder_id)
