"""
Document Intelligence Agent — Flask routes.

Documents are PROJECT-SCOPED: they belong to a project and can be
accessed by any team member with project access.

Endpoints:
- POST /upload — upload a document to a project
- GET /status/<document_id> — get processing status
- GET /documents — list project's documents
- DELETE /documents/<document_id> — delete a document
- POST /process/<document_id> — trigger processing (or auto on upload)
- POST /chat — query documents
"""

import logging
from flask import Blueprint, request, jsonify, g

from core.auth import require_auth, user_can_access_project
from agents.document_intelligence.service import DocumentService

logger = logging.getLogger(__name__)

bp = Blueprint("document_intelligence", __name__, url_prefix="/api/document-intelligence")
service = DocumentService()


def get_user_id() -> str:
    """Authenticated user ID, set by the @require_auth decorator from a
    verified session token."""
    return g.user_id


def get_project_id() -> str:
    """Get project ID from request (query param, form data, or JSON body)."""
    project_id = request.args.get("project_id")
    if project_id:
        return project_id

    project_id = request.form.get("project_id")
    if project_id:
        return project_id

    data = request.get_json(silent=True) or {}
    return data.get("project_id")


def _require_project_access(project_id):
    """Returns an error response if project_id is missing or the caller
    doesn't have access to it, else None. Documents are shared within a
    project, so ownership of the project - not who uploaded the document -
    is the real access-control boundary here."""
    if not project_id:
        return jsonify({"error": "project_id is required"}), 400
    if not user_can_access_project(g.user_id, project_id):
        return jsonify({"error": "Project not found"}), 404
    return None


@bp.route("/upload", methods=["POST"])
@require_auth
def upload_document():
    """
    Upload a document to a project for processing.

    Request: multipart/form-data with 'file', 'project_id', and optional 'document_type'
    Response: { document_id, file_name, project_id, status }

    Documents are project-scoped and shared among all project members.
    """
    try:
        user_id = get_user_id()
        project_id = get_project_id()

        access_error = _require_project_access(project_id)
        if access_error:
            return access_error

        if "file" not in request.files:
            return jsonify({"error": "No file provided"}), 400

        file = request.files["file"]
        document_type = request.form.get("document_type")

        result = service.upload_document(
            file=file,
            user_id=user_id,
            project_id=project_id,
            document_type=document_type,
        )

        # Trigger async processing
        from agents.document_intelligence.tasks import process_document_task

        process_document_task.delay(result["document_id"])

        return jsonify(result), 201

    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        logger.exception("Upload failed")
        return jsonify({"error": "Upload failed", "detail": str(e)}), 500


@bp.route("/status/<document_id>", methods=["GET"])
@require_auth
def get_status(document_id: str):
    """
    Get document processing status.

    Query params:
    - project_id (required): verify document belongs to a project the caller can access

    Response: { document_id, status, processing_stage, processing_progress, ... }
    """
    try:
        project_id = get_project_id()
        access_error = _require_project_access(project_id)
        if access_error:
            return access_error

        result = service.get_document_status(document_id, project_id=project_id)
        return jsonify(result), 200

    except ValueError as e:
        return jsonify({"error": str(e)}), 404
    except Exception as e:
        logger.exception("Status check failed")
        return jsonify({"error": "Status check failed"}), 500


@bp.route("/documents", methods=["GET"])
@require_auth
def list_documents():
    """
    List project's documents.

    Query params:
    - project_id (required): filter by project
    - limit (default 50): max results

    Response: { documents: [...] }
    """
    try:
        project_id = get_project_id()
        access_error = _require_project_access(project_id)
        if access_error:
            return access_error

        limit = request.args.get("limit", 50, type=int)

        documents = service.list_documents(project_id=project_id, limit=limit)
        return jsonify({"documents": documents}), 200

    except Exception as e:
        logger.exception("List documents failed")
        return jsonify({"error": "List documents failed"}), 500


@bp.route("/documents/<document_id>", methods=["DELETE"])
@require_auth
def delete_document(document_id: str):
    """
    Delete a document and all associated data.

    Query params:
    - project_id (required): verify document belongs to a project the caller can access

    Response: { success: true }
    """
    try:
        project_id = get_project_id()
        access_error = _require_project_access(project_id)
        if access_error:
            return access_error

        service.delete_document(document_id, project_id=project_id)
        return jsonify({"success": True}), 200

    except ValueError as e:
        return jsonify({"error": str(e)}), 404
    except Exception as e:
        logger.exception("Delete document failed")
        return jsonify({"error": "Delete document failed"}), 500


@bp.route("/process/<document_id>", methods=["POST"])
@require_auth
def process_document(document_id: str):
    """
    Trigger document processing (if not already processing).

    Response: { document_id, status }
    """
    try:
        project_id = get_project_id()
        access_error = _require_project_access(project_id)
        if access_error:
            return access_error

        # Verify ownership (by project, matching get_document_status's contract)
        status = service.get_document_status(document_id, project_id)

        if status["status"] == "processing":
            return jsonify({"message": "Already processing", **status}), 200

        # Trigger async processing
        from agents.document_intelligence.tasks import process_document_task

        process_document_task.delay(document_id)

        return jsonify({"message": "Processing started", "document_id": document_id}), 202

    except ValueError as e:
        return jsonify({"error": str(e)}), 404
    except Exception as e:
        logger.exception("Process document failed")
        return jsonify({"error": "Process document failed"}), 500


@bp.route("/documents/<document_id>/insight", methods=["GET"])
@require_auth
def get_insight(document_id: str):
    """
    Get (or generate) structured analysis for a document: summary, key
    facts, recommendations, and source citations - derived from the
    document's real extracted text and vector search, not demo data.

    Query params:
    - project_id (required): verify document belongs to a project the caller can access

    Response: { status, summary, keyFacts, recommendations, sources }
    """
    try:
        user_id = get_user_id()
        project_id = get_project_id()
        access_error = _require_project_access(project_id)
        if access_error:
            return access_error

        result = service.get_document_insight(document_id, project_id=project_id, user_id=user_id)
        return jsonify(result), 200

    except ValueError as e:
        return jsonify({"error": str(e)}), 404
    except Exception as e:
        logger.exception("Insight generation failed")
        return jsonify({"error": "Insight generation failed", "detail": str(e)}), 500


@bp.route("/chat", methods=["POST"])
@require_auth
def chat():
    """
    Chat with documents using RAG.

    Request body:
    {
        "query": "What is the pricing strategy?",
        "project_id": "...",  // required - scopes which documents can be searched
        "document_ids": ["uuid1", "uuid2"],  // optional, defaults to all in project
        "use_entity_boost": false  // optional
    }

    Response: { answer, sources, chunk_count }
    """
    try:
        user_id = get_user_id()
        data = request.get_json() or {}

        query = data.get("query", "").strip()
        if not query:
            return jsonify({"error": "Query is required"}), 400

        project_id = data.get("project_id") or get_project_id()
        access_error = _require_project_access(project_id)
        if access_error:
            return access_error

        document_ids = data.get("document_ids")
        use_entity_boost = data.get("use_entity_boost", False)

        result = service.chat(
            query=query,
            user_id=user_id,
            document_ids=document_ids,
            use_entity_boost=use_entity_boost,
            project_id=project_id,
        )

        return jsonify(result), 200

    except Exception as e:
        logger.exception("Chat failed")
        return jsonify({"error": "Chat failed", "detail": str(e)}), 500


@bp.route("/search", methods=["POST"])
@require_auth
def search():
    """
    Search documents (raw vector search), scoped to the authenticated user.

    Request body:
    {
        "query": "search terms",
        "document_ids": ["uuid1"],  // optional
        "top_k": 5  // optional
    }

    Response: { results: [...] }
    """
    try:
        user_id = get_user_id()
        data = request.get_json() or {}

        query = data.get("query", "").strip()
        if not query:
            return jsonify({"error": "Query is required"}), 400

        document_ids = data.get("document_ids")
        top_k = data.get("top_k", 5)

        from agents.document_intelligence.retrieval import DocumentRetriever

        retriever = DocumentRetriever()
        results = retriever.search(
            query=query,
            user_id=user_id,
            document_ids=document_ids,
            top_k=top_k,
        )

        return jsonify({"results": results}), 200

    except Exception as e:
        logger.exception("Search failed")
        return jsonify({"error": "Search failed"}), 500
