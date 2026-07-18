"""
Content Marketing Agent — service layer.

Uses SQLAlchemy models for all DB operations.
All SQLite code has been migrated to PostgreSQL.
"""
from datetime import datetime
from uuid import uuid4
import json
import os

from flask import jsonify, request, current_app
from werkzeug.utils import secure_filename

from core.database import db
from .models import CMProject, CMDocument, CMKnowledgeGraph, CMGeneratedContent, CMConversation


# =============================================================================
# Project Operations
# =============================================================================

def create_project():
    """Create a new content marketing project."""
    data = request.get_json(silent=True) or {}

    user_id = data.get('user_id', 'default_user')
    project_name = data.get('project_name', 'Untitled Project')

    project = CMProject(
        project_id=f"project_{uuid4().hex[:12]}",
        user_id=user_id,
        project_name=project_name,
        description=data.get("description"),
        industry=data.get("industry"),
        sector=data.get("sector"),
        function=data.get("function"),
        role=data.get("role"),
    )
    db.session.add(project)
    db.session.commit()

    # Store in context lake
    try:
        from core.context import ContextStore
        ContextStore().set(
            user_id,
            "content_marketing",
            f"project:{project.project_id}",
            {
                "project_id": project.project_id,
                "project_name": project.project_name,
                "industry": project.industry,
                "updated_at": datetime.utcnow().isoformat(),
            },
        )
    except Exception:
        pass

    return jsonify({
        "success": True,
        "project_id": project.project_id,
        "message": f'Project "{project_name}" created successfully'
    }), 201


def get_project(project_id: str):
    """Get project details with statistics."""
    project = CMProject.query.filter_by(project_id=project_id).first()
    if not project:
        return jsonify({"success": False, "error": "Project not found"}), 404

    doc_count = CMDocument.query.filter_by(project_id=project_id).count()
    kg = CMKnowledgeGraph.query.filter_by(project_id=project_id).first()

    return jsonify({
        "success": True,
        "project": project.to_dict(),
        "statistics": {
            "documents": doc_count,
            "has_knowledge_graph": kg is not None
        }
    })


# =============================================================================
# Document Operations
# =============================================================================

def upload_documents(analyzer=None, upload_folder=None):
    """
    Upload documents to project.
    Extracts text and creates initial knowledge graph.

    Args:
        analyzer: DomainSpecializationAnalyzer instance (passed from app.py)
        upload_folder: Path to upload folder (passed from app.py)
    """
    project_id = request.form.get('project_id')
    if not project_id:
        return jsonify({'success': False, 'error': 'project_id required'}), 400

    uploaded_files = request.files.getlist('files')
    if not uploaded_files:
        return jsonify({'success': False, 'error': 'No files provided'}), 400

    # Verify project exists
    project = CMProject.query.filter_by(project_id=project_id).first()
    if not project:
        return jsonify({'success': False, 'error': 'Project not found'}), 404

    extracted_documents = []
    doc_ids = []

    for file in uploaded_files:
        if not file.filename:
            continue

        filename = secure_filename(file.filename)
        file_type = filename.split('.')[-1].lower()

        # Save file
        file_path = os.path.join(upload_folder, project_id, filename)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        file.save(file_path)

        # Extract content
        extracted_content = ""
        if analyzer:
            try:
                extracted_content = analyzer.extract_text(file_path)
            except Exception:
                pass

        # Create document record
        doc_id = f"doc_{uuid4().hex[:12]}"
        doc = CMDocument(
            doc_id=doc_id,
            project_id=project_id,
            file_name=filename,
            file_type=file_type,
            file_path=file_path,
            file_size=os.path.getsize(file_path),
            document_type=file_type,
            extracted_content=extracted_content
        )
        db.session.add(doc)
        doc_ids.append(doc_id)
        extracted_documents.append({
            'doc_id': doc_id,
            'file_name': filename,
            'content_preview': extracted_content[:500] if extracted_content else ''
        })

    db.session.commit()

    # Build knowledge graph if we have documents
    if extracted_documents and analyzer:
        try:
            all_content = "\n\n".join([d.get('content_preview', '') for d in extracted_documents])
            kg_data = analyzer.build_knowledge_graph(all_content) if hasattr(analyzer, 'build_knowledge_graph') else {}

            kg = CMKnowledgeGraph(
                kg_id=f"kg_{uuid4().hex[:12]}",
                project_id=project_id,
            )
            kg.kg_data = kg_data
            kg.entities = len(kg_data.get('entities', []))
            kg.relationships = len(kg_data.get('relationships', []))
            db.session.add(kg)
            db.session.commit()
        except Exception:
            pass

    return jsonify({
        'success': True,
        'uploaded_count': len(doc_ids),
        'documents': extracted_documents
    })


def list_documents(project_id: str):
    """List all documents in a project."""
    docs = CMDocument.query.filter_by(project_id=project_id).all()
    return jsonify({
        "success": True,
        "documents": [d.to_dict() for d in docs]
    })


# =============================================================================
# Content Generation
# =============================================================================

def generate_content(content_generator=None):
    """
    Generate content using RAG and knowledge graph.

    Args:
        content_generator: RAGContentGenerator instance (passed from app.py)
    """
    data = request.get_json(silent=True) or {}
    project_id = data.get('project_id')
    channel = data.get('channel', 'linkedin')
    content_type = data.get('content_type', 'post')
    prompt = data.get('prompt', '')

    if not project_id:
        return jsonify({'success': False, 'error': 'project_id required'}), 400

    # Get project and documents
    project = CMProject.query.filter_by(project_id=project_id).first()
    if not project:
        return jsonify({'success': False, 'error': 'Project not found'}), 404

    docs = CMDocument.query.filter_by(project_id=project_id).all()

    # Generate content
    generated_text = ""
    if content_generator:
        try:
            context = "\n\n".join([d.extracted_content or '' for d in docs])
            generated_text = content_generator.generate(
                prompt=prompt,
                context=context,
                channel=channel,
                content_type=content_type
            ) if hasattr(content_generator, 'generate') else f"Generated {content_type} for {channel}"
        except Exception as e:
            generated_text = f"Error generating content: {str(e)}"
    else:
        generated_text = f"Content generation placeholder for {channel} {content_type}"

    # Store generated content
    content = CMGeneratedContent(
        content_id=f"content_{uuid4().hex[:12]}",
        project_id=project_id,
        channel=channel,
        content_type=content_type,
        content=generated_text,
    )
    content.source_docs = [d.doc_id for d in docs]
    content.domain_context = {
        "industry": project.industry,
        "prompt": prompt
    }
    db.session.add(content)
    db.session.commit()

    return jsonify({
        'success': True,
        'content_id': content.content_id,
        'channel': channel,
        'content_type': content_type,
        'generated_content': generated_text
    })


# =============================================================================
# Chat / Conversation
# =============================================================================

def chat(content_generator=None):
    """
    Chat with the content marketing agent.

    Args:
        content_generator: RAGContentGenerator instance (passed from app.py)
    """
    data = request.get_json(silent=True) or {}
    project_id = data.get('project_id')
    user_message = data.get('message', '')

    if not project_id:
        return jsonify({'success': False, 'error': 'project_id required'}), 400

    # Get project context
    project = CMProject.query.filter_by(project_id=project_id).first()
    if not project:
        return jsonify({'success': False, 'error': 'Project not found'}), 404

    docs = CMDocument.query.filter_by(project_id=project_id).all()

    # Generate response
    agent_response = ""
    if content_generator:
        try:
            context = "\n\n".join([d.extracted_content or '' for d in docs])
            agent_response = content_generator.chat(
                message=user_message,
                context=context
            ) if hasattr(content_generator, 'chat') else f"Response to: {user_message}"
        except Exception as e:
            agent_response = f"Error: {str(e)}"
    else:
        agent_response = f"Chat response placeholder for: {user_message}"

    # Store conversation
    conv = CMConversation(
        msg_id=f"msg_{uuid4().hex[:12]}",
        project_id=project_id,
        user_message=user_message,
        agent_response=agent_response,
    )
    conv.context = {
        "project_name": project.project_name,
        "doc_count": len(docs)
    }
    db.session.add(conv)
    db.session.commit()

    return jsonify({
        'success': True,
        'response': agent_response,
        'msg_id': conv.msg_id
    })


# =============================================================================
# Knowledge Graph
# =============================================================================

def get_knowledge_graph(project_id: str):
    """Retrieve knowledge graph for visualization."""
    kg = CMKnowledgeGraph.query.filter_by(project_id=project_id).first()
    if not kg:
        return jsonify({"success": False, "error": "Knowledge graph not found"}), 404

    return jsonify({
        "success": True,
        "kg_id": kg.kg_id,
        "kg_data": kg.kg_data,
        "entities": kg.entities,
        "relationships": kg.relationships,
        "created_at": kg.created_at.isoformat() if kg.created_at else None
    })
