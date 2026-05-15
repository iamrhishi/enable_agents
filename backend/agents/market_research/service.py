"""Market Research Agent — service layer."""
import json
from datetime import datetime
from uuid import uuid4

from flask import jsonify, request

from core.database import db
from .models import ResearchProject, ResearchResult


def create_project():
    data = request.get_json(silent=True) or {}
    required = ["user_id", "company_name", "industry"]
    missing = [k for k in required if not data.get(k)]
    if missing:
        return jsonify({"error": f"Missing fields: {missing}"}), 400

    project = ResearchProject(
        project_id=str(uuid4()),
        user_id=data["user_id"],
        company_name=data["company_name"],
        industry=data["industry"],
        research_goals=json.dumps(data.get("research_goals", [])),
    )
    db.session.add(project)
    db.session.commit()
    try:
        from core.context import ContextStore

        ContextStore().set(
            data["user_id"],
            "market_research",
            f"project:{project.project_id}",
            {
                "project_id": project.project_id,
                "company_name": project.company_name,
                "industry": project.industry,
                "status": project.status,
                "updated_at": datetime.utcnow().isoformat(),
            },
        )
    except Exception:
        pass
    return jsonify({"success": True, "project_id": project.project_id}), 201


def get_project(project_id: str):
    project = ResearchProject.query.filter_by(project_id=project_id).first()
    if not project:
        return jsonify({"error": "Project not found"}), 404
    return jsonify({
        "success": True,
        "project": {
            "project_id": project.project_id,
            "company_name": project.company_name,
            "industry": project.industry,
            "status": project.status,
            "created_at": project.created_at.isoformat(),
        },
    })


def run_research(project_id: str):
    from agents.market_research.tasks import run_research_async
    result = run_research_async.delay(project_id)
    return jsonify({"success": True, "task_id": result.id})


def get_results(project_id: str):
    results = ResearchResult.query.filter_by(project_id=project_id).all()
    return jsonify({"success": True, "results": [
        {"result_type": r.result_type, "content": r.content, "source_url": r.source_url}
        for r in results
    ]})


def generate_requirements():
    try:
        from app import generate_requirements as legacy  # noqa
        return legacy()
    except (ImportError, AttributeError):
        return jsonify({"error": "Not yet migrated"}), 501


def simple_search():
    try:
        from app import simple_search_endpoint as legacy  # noqa
        return legacy()
    except (ImportError, AttributeError):
        return jsonify({"error": "Not yet migrated"}), 501


def recommend_agents():
    try:
        from app import recommend_agents as legacy  # noqa
        return legacy()
    except (ImportError, AttributeError):
        return jsonify({"error": "Not yet migrated"}), 501
