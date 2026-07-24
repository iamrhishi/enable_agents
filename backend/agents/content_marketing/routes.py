"""
Content Marketing Agent — Flask Blueprint.

Routes are registered under /api/content-marketing (defined in manifest.json).
Business logic lives in service.py; this file is intentionally thin.
"""
from flask import Blueprint

from core.auth import require_auth

from . import service

content_marketing_bp = Blueprint(
    "content_marketing",
    __name__,
    url_prefix="/api/content-marketing",
)


@content_marketing_bp.post("/projects")
@require_auth
def create_project():
    return service.create_project()


@content_marketing_bp.get("/projects/<project_id>")
@require_auth
def get_project(project_id: str):
    return service.get_project(project_id)


@content_marketing_bp.post("/documents/upload")
@require_auth
def upload_documents():
    return service.upload_documents()


@content_marketing_bp.get("/documents/<project_id>")
@require_auth
def list_documents(project_id: str):
    return service.list_documents(project_id)


@content_marketing_bp.post("/generate-content")
@require_auth
def generate_content():
    return service.generate_content()


@content_marketing_bp.post("/chat")
@require_auth
def chat():
    return service.chat()


@content_marketing_bp.get("/knowledge-graph/<project_id>")
@require_auth
def get_knowledge_graph(project_id: str):
    return service.get_knowledge_graph(project_id)
