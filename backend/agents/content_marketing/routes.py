"""
Content Marketing Agent — Flask Blueprint.

Routes are registered under /api/content-marketing (defined in manifest.json).
Business logic lives in service.py; this file is intentionally thin.
"""
from flask import Blueprint
from . import service

content_marketing_bp = Blueprint(
    "content_marketing",
    __name__,
    url_prefix="/api/content-marketing",
)


@content_marketing_bp.post("/projects")
def create_project():
    return service.create_project()


@content_marketing_bp.get("/projects/<project_id>")
def get_project(project_id: str):
    return service.get_project(project_id)


@content_marketing_bp.post("/documents/upload")
def upload_documents():
    return service.upload_documents()


@content_marketing_bp.get("/documents/<project_id>")
def list_documents(project_id: str):
    return service.list_documents(project_id)


@content_marketing_bp.post("/generate-content")
def generate_content():
    return service.generate_content()


@content_marketing_bp.post("/chat")
def chat():
    return service.chat()


@content_marketing_bp.get("/knowledge-graph/<project_id>")
def get_knowledge_graph(project_id: str):
    return service.get_knowledge_graph(project_id)
