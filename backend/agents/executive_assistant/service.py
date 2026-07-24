"""Executive Assistant Agent — Service layer."""
from flask import request, jsonify, g
from datetime import datetime
import uuid

from core.database import db
from .models import ExecTask, ExecReminder, ExecStakeholder


def get_user_id() -> str:
    """Authenticated user ID, set by the @require_auth decorator from a
    verified session token - never trust a client-supplied X-User-Id header,
    it's spoofable by any caller."""
    return getattr(g, "user_id", "") or ""


# =============================================================================
# Tasks
# =============================================================================

def list_tasks():
    """List all tasks for the current user, optionally filtered by project."""
    user_id = get_user_id()
    if not user_id:
        return jsonify({"error": "Not authenticated"}), 401

    project_id = request.args.get("project_id")
    status = request.args.get("status")

    query = ExecTask.query.filter_by(user_id=user_id)
    if project_id:
        query = query.filter_by(project_id=project_id)
    if status:
        query = query.filter_by(status=status)

    tasks = query.order_by(ExecTask.due_date.asc().nullslast(), ExecTask.priority.desc()).all()
    return jsonify({"success": True, "tasks": [t.to_dict() for t in tasks]})


def create_task():
    """Create a new task."""
    user_id = get_user_id()
    if not user_id:
        return jsonify({"error": "Not authenticated"}), 401

    data = request.get_json()
    title = data.get("title", "").strip()
    if not title:
        return jsonify({"error": "Title is required"}), 400

    due_date = None
    if data.get("dueDate"):
        try:
            due_date = datetime.strptime(data["dueDate"], "%Y-%m-%d")
        except ValueError:
            pass

    task = ExecTask(
        task_id=str(uuid.uuid4()),
        user_id=user_id,
        project_id=data.get("projectId"),
        title=title,
        description=data.get("description", ""),
        assigned_to=data.get("assignedTo"),
        due_date=due_date,
        priority=data.get("priority", "Medium"),
        status=data.get("status", "Pending"),
    )
    db.session.add(task)
    db.session.commit()
    return jsonify({"success": True, "task": task.to_dict()})


def get_task(task_id: str):
    """Get a single task by ID."""
    user_id = get_user_id()
    if not user_id:
        return jsonify({"error": "Not authenticated"}), 401

    task = ExecTask.query.filter_by(task_id=task_id, user_id=user_id).first()
    if not task:
        return jsonify({"error": "Task not found"}), 404
    return jsonify({"success": True, "task": task.to_dict()})


def update_task(task_id: str):
    """Update a task."""
    user_id = get_user_id()
    if not user_id:
        return jsonify({"error": "Not authenticated"}), 401

    task = ExecTask.query.filter_by(task_id=task_id, user_id=user_id).first()
    if not task:
        return jsonify({"error": "Task not found"}), 404

    data = request.get_json()
    if "title" in data:
        task.title = data["title"].strip()
    if "description" in data:
        task.description = data["description"]
    if "projectId" in data:
        task.project_id = data["projectId"]
    if "assignedTo" in data:
        task.assigned_to = data["assignedTo"]
    if "priority" in data and data["priority"] in ["Low", "Medium", "High", "Urgent"]:
        task.priority = data["priority"]
    if "status" in data and data["status"] in ["Pending", "In Progress", "Completed", "Cancelled"]:
        task.status = data["status"]
    if "dueDate" in data:
        if data["dueDate"]:
            try:
                task.due_date = datetime.strptime(data["dueDate"], "%Y-%m-%d")
            except ValueError:
                pass
        else:
            task.due_date = None

    db.session.commit()
    return jsonify({"success": True, "task": task.to_dict()})


def delete_task(task_id: str):
    """Delete a task."""
    user_id = get_user_id()
    if not user_id:
        return jsonify({"error": "Not authenticated"}), 401

    task = ExecTask.query.filter_by(task_id=task_id, user_id=user_id).first()
    if not task:
        return jsonify({"error": "Task not found"}), 404

    db.session.delete(task)
    db.session.commit()
    return jsonify({"success": True, "message": "Task deleted"})


# =============================================================================
# Reminders
# =============================================================================

def list_reminders():
    """List all reminders for the current user."""
    user_id = get_user_id()
    if not user_id:
        return jsonify({"error": "Not authenticated"}), 401

    task_id = request.args.get("task_id")
    pending_only = request.args.get("pending") == "true"

    query = ExecReminder.query.filter_by(user_id=user_id)
    if task_id:
        query = query.filter_by(task_id=task_id)
    if pending_only:
        query = query.filter_by(sent=False).filter(ExecReminder.remind_at >= datetime.utcnow())

    reminders = query.order_by(ExecReminder.remind_at.asc()).all()
    return jsonify({"success": True, "reminders": [r.to_dict() for r in reminders]})


def create_reminder():
    """Create a reminder for a task."""
    user_id = get_user_id()
    if not user_id:
        return jsonify({"error": "Not authenticated"}), 401

    data = request.get_json()
    task_id = data.get("taskId")

    # Verify task exists and belongs to user
    task = ExecTask.query.filter_by(task_id=task_id, user_id=user_id).first()
    if not task:
        return jsonify({"error": "Task not found"}), 404

    try:
        remind_at = datetime.fromisoformat(data["remindAt"].replace("Z", "+00:00"))
    except (KeyError, ValueError):
        return jsonify({"error": "Valid remindAt datetime required"}), 400

    reminder = ExecReminder(
        reminder_id=str(uuid.uuid4()),
        task_id=task_id,
        user_id=user_id,
        remind_at=remind_at,
        channel=data.get("channel", "email"),
        recipient=data.get("recipient"),
        message=data.get("message"),
    )
    db.session.add(reminder)
    db.session.commit()
    return jsonify({"success": True, "reminder": reminder.to_dict()})


def delete_reminder(reminder_id: str):
    """Delete a reminder."""
    user_id = get_user_id()
    if not user_id:
        return jsonify({"error": "Not authenticated"}), 401

    reminder = ExecReminder.query.filter_by(reminder_id=reminder_id, user_id=user_id).first()
    if not reminder:
        return jsonify({"error": "Reminder not found"}), 404

    db.session.delete(reminder)
    db.session.commit()
    return jsonify({"success": True, "message": "Reminder deleted"})


# =============================================================================
# Stakeholders
# =============================================================================

def list_stakeholders():
    """List all stakeholders/contacts for the current user."""
    user_id = get_user_id()
    if not user_id:
        return jsonify({"error": "Not authenticated"}), 401

    stakeholders = ExecStakeholder.query.filter_by(user_id=user_id).order_by(ExecStakeholder.name.asc()).all()
    return jsonify({"success": True, "stakeholders": [s.to_dict() for s in stakeholders]})


def create_stakeholder():
    """Create a new stakeholder."""
    user_id = get_user_id()
    if not user_id:
        return jsonify({"error": "Not authenticated"}), 401

    data = request.get_json()
    name = data.get("name", "").strip()
    if not name:
        return jsonify({"error": "Name is required"}), 400

    stakeholder = ExecStakeholder(
        stakeholder_id=str(uuid.uuid4()),
        user_id=user_id,
        name=name,
        email=data.get("email", "").strip() or None,
        phone=data.get("phone", "").strip() or None,
        role=data.get("role", "").strip() or None,
    )
    if data.get("projects"):
        stakeholder.project_ids = data["projects"]

    db.session.add(stakeholder)
    db.session.commit()
    return jsonify({"success": True, "stakeholder": stakeholder.to_dict()})


def get_stakeholder(stakeholder_id: str):
    """Get a single stakeholder."""
    user_id = get_user_id()
    if not user_id:
        return jsonify({"error": "Not authenticated"}), 401

    stakeholder = ExecStakeholder.query.filter_by(stakeholder_id=stakeholder_id, user_id=user_id).first()
    if not stakeholder:
        return jsonify({"error": "Stakeholder not found"}), 404
    return jsonify({"success": True, "stakeholder": stakeholder.to_dict()})


def update_stakeholder(stakeholder_id: str):
    """Update a stakeholder."""
    user_id = get_user_id()
    if not user_id:
        return jsonify({"error": "Not authenticated"}), 401

    stakeholder = ExecStakeholder.query.filter_by(stakeholder_id=stakeholder_id, user_id=user_id).first()
    if not stakeholder:
        return jsonify({"error": "Stakeholder not found"}), 404

    data = request.get_json()
    if "name" in data:
        stakeholder.name = data["name"].strip()
    if "email" in data:
        stakeholder.email = data["email"].strip() or None
    if "phone" in data:
        stakeholder.phone = data["phone"].strip() or None
    if "role" in data:
        stakeholder.role = data["role"].strip() or None
    if "projects" in data:
        stakeholder.project_ids = data["projects"]

    db.session.commit()
    return jsonify({"success": True, "stakeholder": stakeholder.to_dict()})


def delete_stakeholder(stakeholder_id: str):
    """Delete a stakeholder."""
    user_id = get_user_id()
    if not user_id:
        return jsonify({"error": "Not authenticated"}), 401

    stakeholder = ExecStakeholder.query.filter_by(stakeholder_id=stakeholder_id, user_id=user_id).first()
    if not stakeholder:
        return jsonify({"error": "Stakeholder not found"}), 404

    db.session.delete(stakeholder)
    db.session.commit()
    return jsonify({"success": True, "message": "Stakeholder deleted"})
