"""
Content Marketing Agent — service layer.

Uses SQLAlchemy models (CMProject, CMDocument) for all DB operations.
Legacy handlers that haven't been migrated yet return 501 until extracted.
"""
from datetime import datetime
from uuid import uuid4

from flask import jsonify, request

from core.database import db
from .models import CMProject, CMDocument


def create_project():
    data = request.get_json(silent=True) or {}
    required = ["user_id", "project_name"]
    missing = [k for k in required if not data.get(k)]
    if missing:
        return jsonify({"error": f"Missing fields: {missing}"}), 400

    project = CMProject(
        project_id=str(uuid4()),
        user_id=data["user_id"],
        project_name=data["project_name"],
        description=data.get("description"),
        industry=data.get("industry"),
        sector=data.get("sector"),
        function=data.get("function"),
        role=data.get("role"),
    )
    db.session.add(project)
    db.session.commit()
    return jsonify({"success": True, "project_id": project.project_id}), 201


def get_project(project_id: str):
    project = CMProject.query.filter_by(project_id=project_id).first()
    if not project:
        return jsonify({"error": "Project not found"}), 404
    return jsonify({
        "success": True,
        "project": {
            "project_id": project.project_id,
            "user_id": project.user_id,
            "project_name": project.project_name,
            "industry": project.industry,
            "created_at": project.created_at.isoformat() if project.created_at else None,
        },
    })


def upload_documents():
    return jsonify({"error": "Not yet migrated from legacy handler"}), 501


def list_documents(project_id: str):
    docs = CMDocument.query.filter_by(project_id=project_id).all()
    return jsonify({"success": True, "documents": [
        {"doc_id": d.doc_id, "file_name": d.file_name, "file_type": d.file_type}
        for d in docs
    ]})


def generate_content():
    return jsonify({"error": "Not yet migrated from legacy handler"}), 501


def chat():
    return jsonify({"error": "Not yet migrated from legacy handler"}), 501


def get_knowledge_graph(project_id: str):
    from .models import CMKnowledgeGraph
    kg = CMKnowledgeGraph.query.filter_by(project_id=project_id).first()
    if not kg:
        return jsonify({"error": "Knowledge graph not found"}), 404
    return jsonify({"success": True, "graph_data": kg.graph_data})
