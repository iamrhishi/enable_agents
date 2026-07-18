"""
Content Marketing Agent — SQLAlchemy models.

These replace the raw sqlite3 CREATE TABLE statements in app.py's
init_content_marketing_db(). All tables prefixed with 'cm_' to avoid
conflicts with platform tables.
"""
import json
from datetime import datetime
from typing import Dict, Any, List

from core.database import db


class CMProject(db.Model):
    """Content Marketing project."""
    __tablename__ = "cm_projects"

    project_id = db.Column(db.String(36), primary_key=True)
    user_id = db.Column(db.String(255), nullable=False, index=True)
    project_name = db.Column(db.String(255), nullable=False)
    description = db.Column(db.Text)
    industry = db.Column(db.String(255))
    sector = db.Column(db.String(255))
    function = db.Column(db.String(255))
    role = db.Column(db.String(255))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    _metadata = db.Column("metadata", db.Text, default="{}")

    documents = db.relationship("CMDocument", back_populates="project", cascade="all, delete-orphan")
    knowledge_graphs = db.relationship("CMKnowledgeGraph", back_populates="project", cascade="all, delete-orphan")
    generated_content = db.relationship("CMGeneratedContent", back_populates="project", cascade="all, delete-orphan")
    conversations = db.relationship("CMConversation", back_populates="project", cascade="all, delete-orphan")

    @property
    def metadata_dict(self) -> Dict[str, Any]:
        return json.loads(self._metadata or "{}")

    @metadata_dict.setter
    def metadata_dict(self, value: Dict[str, Any]):
        self._metadata = json.dumps(value, default=str)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "project_id": self.project_id,
            "user_id": self.user_id,
            "project_name": self.project_name,
            "description": self.description,
            "industry": self.industry,
            "sector": self.sector,
            "function": self.function,
            "role": self.role,
            "metadata": self.metadata_dict,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


class CMDocument(db.Model):
    """Content Marketing document."""
    __tablename__ = "cm_documents"

    doc_id = db.Column(db.String(36), primary_key=True)
    project_id = db.Column(db.String(36), db.ForeignKey("cm_projects.project_id", ondelete="CASCADE"), nullable=False, index=True)
    file_name = db.Column(db.String(255), nullable=False)
    file_type = db.Column(db.String(50))
    file_path = db.Column(db.Text)
    file_size = db.Column(db.Integer)
    upload_date = db.Column(db.DateTime, default=datetime.utcnow)
    document_type = db.Column(db.String(100))
    extracted_content = db.Column(db.Text)

    project = db.relationship("CMProject", back_populates="documents")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "project_id": self.project_id,
            "file_name": self.file_name,
            "file_type": self.file_type,
            "file_path": self.file_path,
            "file_size": self.file_size,
            "document_type": self.document_type,
            "extracted_content": self.extracted_content,
            "upload_date": self.upload_date.isoformat() if self.upload_date else None,
        }


class CMKnowledgeGraph(db.Model):
    """Content Marketing knowledge graph."""
    __tablename__ = "cm_knowledge_graphs"

    kg_id = db.Column(db.String(36), primary_key=True)
    project_id = db.Column(db.String(36), db.ForeignKey("cm_projects.project_id", ondelete="CASCADE"), nullable=False, index=True)
    _kg_data = db.Column("kg_data", db.Text, default="{}")
    entities = db.Column(db.Integer, default=0)
    relationships = db.Column(db.Integer, default=0)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    project = db.relationship("CMProject", back_populates="knowledge_graphs")

    @property
    def kg_data(self) -> Dict[str, Any]:
        return json.loads(self._kg_data or "{}")

    @kg_data.setter
    def kg_data(self, value: Dict[str, Any]):
        self._kg_data = json.dumps(value, default=str)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "kg_id": self.kg_id,
            "project_id": self.project_id,
            "kg_data": self.kg_data,
            "entities": self.entities,
            "relationships": self.relationships,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class CMGeneratedContent(db.Model):
    """Content Marketing generated content."""
    __tablename__ = "cm_generated_content"

    content_id = db.Column(db.String(36), primary_key=True)
    project_id = db.Column(db.String(36), db.ForeignKey("cm_projects.project_id", ondelete="CASCADE"), nullable=False, index=True)
    channel = db.Column(db.String(50))
    content_type = db.Column(db.String(50))
    content = db.Column(db.Text)
    _source_docs = db.Column("source_docs", db.Text, default="[]")
    _domain_context = db.Column("domain_context", db.Text, default="{}")
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    modified_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    project = db.relationship("CMProject", back_populates="generated_content")

    @property
    def source_docs(self) -> List[str]:
        return json.loads(self._source_docs or "[]")

    @source_docs.setter
    def source_docs(self, value: List[str]):
        self._source_docs = json.dumps(value)

    @property
    def domain_context(self) -> Dict[str, Any]:
        return json.loads(self._domain_context or "{}")

    @domain_context.setter
    def domain_context(self, value: Dict[str, Any]):
        self._domain_context = json.dumps(value, default=str)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "content_id": self.content_id,
            "project_id": self.project_id,
            "channel": self.channel,
            "content_type": self.content_type,
            "content": self.content,
            "source_docs": self.source_docs,
            "domain_context": self.domain_context,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "modified_at": self.modified_at.isoformat() if self.modified_at else None,
        }


class CMConversation(db.Model):
    """Content Marketing conversation history."""
    __tablename__ = "cm_conversations"

    msg_id = db.Column(db.String(36), primary_key=True)
    project_id = db.Column(db.String(36), db.ForeignKey("cm_projects.project_id", ondelete="CASCADE"), nullable=False, index=True)
    user_message = db.Column(db.Text)
    agent_response = db.Column(db.Text)
    _context = db.Column("context", db.Text, default="{}")
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

    project = db.relationship("CMProject", back_populates="conversations")

    @property
    def context(self) -> Dict[str, Any]:
        return json.loads(self._context or "{}")

    @context.setter
    def context(self, value: Dict[str, Any]):
        self._context = json.dumps(value, default=str)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "msg_id": self.msg_id,
            "project_id": self.project_id,
            "user_message": self.user_message,
            "agent_response": self.agent_response,
            "context": self.context,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
        }
