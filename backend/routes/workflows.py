"""
Workflow Templates API Routes.

Handles workflow template CRUD, instance management, and stage transitions.
"""
from flask import Blueprint, request, jsonify
from datetime import datetime
import uuid
import json
from pathlib import Path

from core.database import db
from models.workflow import WorkflowTemplate, WorkflowInstance

workflows_bp = Blueprint('workflows', __name__)

TEMPLATES_DIR = Path(__file__).parent.parent / "config" / "workflow-templates"


def get_user_id() -> str:
    return request.headers.get("X-User-Id", "")


def load_system_templates():
    """Load system templates from JSON files on startup."""
    if not TEMPLATES_DIR.exists():
        return

    for json_file in TEMPLATES_DIR.glob("*.json"):
        try:
            data = json.loads(json_file.read_text())
            template_id = data.get("id")
            if not template_id:
                continue

            existing = WorkflowTemplate.query.filter_by(template_id=template_id).first()
            if existing:
                # Update if system template
                if existing.is_system:
                    existing.name = data.get("name", existing.name)
                    existing.description = data.get("description", existing.description)
                    existing.category = data.get("category", existing.category)
                    existing.icon = data.get("icon", existing.icon)
                    existing.stages = data.get("stages", [])
                continue

            template = WorkflowTemplate(
                template_id=template_id,
                name=data.get("name", template_id),
                description=data.get("description", ""),
                category=data.get("category", "general"),
                icon=data.get("icon", "workflow"),
                is_system=True,
                is_active=True,
            )
            template.stages = data.get("stages", [])
            db.session.add(template)
        except Exception as e:
            print(f"[workflows] Error loading template {json_file}: {e}")

    db.session.commit()


# =============================================================================
# Templates
# =============================================================================

@workflows_bp.route('/api/workflows/templates', methods=['GET'])
def list_templates():
    """List all active workflow templates."""
    category = request.args.get("category")
    query = WorkflowTemplate.query.filter_by(is_active=True)
    if category:
        query = query.filter_by(category=category)

    templates = query.order_by(WorkflowTemplate.name.asc()).all()
    return jsonify({"success": True, "templates": [t.to_dict() for t in templates]})


@workflows_bp.route('/api/workflows/templates/<template_id>', methods=['GET'])
def get_template(template_id: str):
    """Get a single template with full stage details."""
    template = WorkflowTemplate.query.filter_by(template_id=template_id).first()
    if not template:
        return jsonify({"error": "Template not found"}), 404
    return jsonify({"success": True, "template": template.to_dict()})


@workflows_bp.route('/api/workflows/templates', methods=['POST'])
def create_template():
    """Create a custom workflow template."""
    user_id = get_user_id()
    if not user_id:
        return jsonify({"error": "Not authenticated"}), 401

    data = request.get_json()
    name = data.get("name", "").strip()
    if not name:
        return jsonify({"error": "Name is required"}), 400

    template_id = data.get("id") or f"custom-{uuid.uuid4().hex[:8]}"

    template = WorkflowTemplate(
        template_id=template_id,
        name=name,
        description=data.get("description", ""),
        category=data.get("category", "custom"),
        icon=data.get("icon", "workflow"),
        is_system=False,
        is_active=True,
    )
    template.stages = data.get("stages", [])

    db.session.add(template)
    db.session.commit()
    return jsonify({"success": True, "template": template.to_dict()})


@workflows_bp.route('/api/workflows/templates/<template_id>', methods=['DELETE'])
def delete_template(template_id: str):
    """Delete a custom template (not system templates)."""
    user_id = get_user_id()
    if not user_id:
        return jsonify({"error": "Not authenticated"}), 401

    template = WorkflowTemplate.query.filter_by(template_id=template_id).first()
    if not template:
        return jsonify({"error": "Template not found"}), 404
    if template.is_system:
        return jsonify({"error": "Cannot delete system templates"}), 403

    db.session.delete(template)
    db.session.commit()
    return jsonify({"success": True, "message": "Template deleted"})


# =============================================================================
# Instances
# =============================================================================

@workflows_bp.route('/api/workflows/instances', methods=['GET'])
def list_instances():
    """List workflow instances for the current user."""
    user_id = get_user_id()
    if not user_id:
        return jsonify({"error": "Not authenticated"}), 401

    status = request.args.get("status")
    project_id = request.args.get("project_id")

    query = WorkflowInstance.query.filter_by(user_id=user_id)
    if status:
        query = query.filter_by(status=status)
    if project_id:
        query = query.filter_by(project_id=project_id)

    instances = query.order_by(WorkflowInstance.updated_at.desc()).all()
    return jsonify({"success": True, "instances": [i.to_dict() for i in instances]})


@workflows_bp.route('/api/workflows/instances', methods=['POST'])
def create_instance():
    """Start a new workflow instance from a template."""
    user_id = get_user_id()
    if not user_id:
        return jsonify({"error": "Not authenticated"}), 401

    data = request.get_json()
    template_id = data.get("templateId")
    if not template_id:
        return jsonify({"error": "templateId is required"}), 400

    template = WorkflowTemplate.query.filter_by(template_id=template_id).first()
    if not template or not template.is_active:
        return jsonify({"error": "Template not found or inactive"}), 404

    instance = WorkflowInstance(
        instance_id=str(uuid.uuid4()),
        template_id=template_id,
        user_id=user_id,
        project_id=data.get("projectId"),
        name=data.get("name", f"{template.name} - {datetime.utcnow().strftime('%Y-%m-%d')}"),
        status="pending",
        current_stage_index=0,
    )

    # Initialize context with any provided inputs
    if data.get("inputs"):
        instance.context = data["inputs"]

    db.session.add(instance)
    db.session.commit()
    return jsonify({"success": True, "instance": instance.to_dict()})


@workflows_bp.route('/api/workflows/instances/<instance_id>', methods=['GET'])
def get_instance(instance_id: str):
    """Get a workflow instance with full state."""
    user_id = get_user_id()
    if not user_id:
        return jsonify({"error": "Not authenticated"}), 401

    instance = WorkflowInstance.query.filter_by(instance_id=instance_id, user_id=user_id).first()
    if not instance:
        return jsonify({"error": "Instance not found"}), 404

    return jsonify({"success": True, "instance": instance.to_dict()})


@workflows_bp.route('/api/workflows/instances/<instance_id>/start', methods=['POST'])
def start_instance(instance_id: str):
    """Start a pending workflow instance."""
    user_id = get_user_id()
    if not user_id:
        return jsonify({"error": "Not authenticated"}), 401

    instance = WorkflowInstance.query.filter_by(instance_id=instance_id, user_id=user_id).first()
    if not instance:
        return jsonify({"error": "Instance not found"}), 404

    if instance.status not in ["pending", "paused"]:
        return jsonify({"error": f"Cannot start instance with status '{instance.status}'"}), 400

    instance.status = "running"
    if not instance.started_at:
        instance.started_at = datetime.utcnow()

    db.session.commit()
    return jsonify({"success": True, "instance": instance.to_dict()})


@workflows_bp.route('/api/workflows/instances/<instance_id>/complete-stage', methods=['POST'])
def complete_stage(instance_id: str):
    """Complete the current stage and advance to next."""
    user_id = get_user_id()
    if not user_id:
        return jsonify({"error": "Not authenticated"}), 401

    instance = WorkflowInstance.query.filter_by(instance_id=instance_id, user_id=user_id).first()
    if not instance:
        return jsonify({"error": "Instance not found"}), 404

    if instance.status != "running":
        return jsonify({"error": "Instance is not running"}), 400

    data = request.get_json() or {}
    stage_data = data.get("data", {})

    instance.advance_stage(stage_data)
    db.session.commit()

    return jsonify({"success": True, "instance": instance.to_dict()})


@workflows_bp.route('/api/workflows/instances/<instance_id>/pause', methods=['POST'])
def pause_instance(instance_id: str):
    """Pause a running workflow."""
    user_id = get_user_id()
    if not user_id:
        return jsonify({"error": "Not authenticated"}), 401

    instance = WorkflowInstance.query.filter_by(instance_id=instance_id, user_id=user_id).first()
    if not instance:
        return jsonify({"error": "Instance not found"}), 404

    if instance.status != "running":
        return jsonify({"error": "Can only pause running instances"}), 400

    instance.status = "paused"
    db.session.commit()
    return jsonify({"success": True, "instance": instance.to_dict()})


@workflows_bp.route('/api/workflows/instances/<instance_id>', methods=['DELETE'])
def delete_instance(instance_id: str):
    """Delete a workflow instance."""
    user_id = get_user_id()
    if not user_id:
        return jsonify({"error": "Not authenticated"}), 401

    instance = WorkflowInstance.query.filter_by(instance_id=instance_id, user_id=user_id).first()
    if not instance:
        return jsonify({"error": "Instance not found"}), 404

    db.session.delete(instance)
    db.session.commit()
    return jsonify({"success": True, "message": "Instance deleted"})
