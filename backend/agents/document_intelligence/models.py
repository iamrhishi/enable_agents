"""
Document Intelligence Agent — SQLAlchemy models.

Tables:
- ProcessedDocument: uploaded document metadata and status
- DocumentChunk: text chunks with pgvector embeddings
- DocumentEntity: extracted entities (stored as JSON, linked to chunks)
"""

import json
from datetime import datetime
from typing import Any, Dict, List, Optional

from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, UUID

from core.database import db


class ProcessedDocument(db.Model):
    """Document metadata and processing status.

    Documents are project-scoped: they belong to a project and can be
    accessed by any team member with project access. The uploaded_by
    field tracks who originally uploaded the document.
    """

    __tablename__ = "processed_documents"

    document_id = db.Column(db.String(36), primary_key=True)

    # Project-scoped ownership (nullable for backwards compatibility)
    project_id = db.Column(db.String(36), nullable=True, index=True)

    # Track who uploaded (renamed from user_id)
    uploaded_by = db.Column("uploaded_by", db.String(255), nullable=False, index=True)

    file_name = db.Column(db.String(255), nullable=False)
    file_type = db.Column(db.String(50))
    file_size = db.Column(db.Integer)
    file_path = db.Column(db.Text)

    # Processing status: pending, processing, completed, failed
    status = db.Column(db.String(50), default="pending", index=True)
    processing_stage = db.Column(db.String(100))
    processing_progress = db.Column(db.Float, default=0.0)
    error_message = db.Column(db.Text)

    # Extracted content
    extracted_text = db.Column(db.Text)
    page_count = db.Column(db.Integer)
    word_count = db.Column(db.Integer)
    chunk_count = db.Column(db.Integer, default=0)
    entity_count = db.Column(db.Integer, default=0)

    # Metadata
    document_type = db.Column(db.String(100))
    _metadata = db.Column("metadata", db.Text, default="{}")

    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    processed_at = db.Column(db.DateTime)

    # Relationships
    chunks = db.relationship(
        "DocumentChunk", back_populates="document", cascade="all, delete-orphan"
    )
    entities = db.relationship(
        "DocumentEntity", back_populates="document", cascade="all, delete-orphan"
    )

    @property
    def metadata_dict(self) -> Dict[str, Any]:
        return json.loads(self._metadata or "{}")

    @metadata_dict.setter
    def metadata_dict(self, value: Dict[str, Any]):
        self._metadata = json.dumps(value, default=str)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "document_id": self.document_id,
            "project_id": self.project_id,
            "uploaded_by": self.uploaded_by,
            "file_name": self.file_name,
            "file_type": self.file_type,
            "file_size": self.file_size,
            "status": self.status,
            "processing_stage": self.processing_stage,
            "processing_progress": self.processing_progress,
            "error_message": self.error_message,
            "page_count": self.page_count,
            "word_count": self.word_count,
            "chunk_count": self.chunk_count,
            "entity_count": self.entity_count,
            "document_type": self.document_type,
            "metadata": self.metadata_dict,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "processed_at": self.processed_at.isoformat() if self.processed_at else None,
        }


class DocumentChunk(db.Model):
    """Document text chunk with pgvector embedding."""

    __tablename__ = "document_chunks"

    chunk_id = db.Column(db.String(36), primary_key=True)
    document_id = db.Column(
        db.String(36),
        db.ForeignKey("processed_documents.document_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    chunk_index = db.Column(db.Integer, nullable=False)
    content = db.Column(db.Text, nullable=False)

    # pgvector embedding (1536 dimensions for OpenAI ada-002)
    embedding = db.Column(Vector(1536))

    # Metadata (page number, section, etc.)
    _metadata = db.Column("metadata", db.Text, default="{}")

    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    # Relationships
    document = db.relationship("ProcessedDocument", back_populates="chunks")

    @property
    def metadata_dict(self) -> Dict[str, Any]:
        return json.loads(self._metadata or "{}")

    @metadata_dict.setter
    def metadata_dict(self, value: Dict[str, Any]):
        self._metadata = json.dumps(value, default=str)

    def to_dict(self, include_embedding: bool = False) -> Dict[str, Any]:
        result = {
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "chunk_index": self.chunk_index,
            "content": self.content,
            "metadata": self.metadata_dict,
        }
        if include_embedding and self.embedding is not None:
            result["embedding"] = list(self.embedding)
        return result


class DocumentEntity(db.Model):
    """Entity extracted from document (stored as JSON, linked to chunks)."""

    __tablename__ = "document_entities"

    entity_id = db.Column(db.String(36), primary_key=True)
    document_id = db.Column(
        db.String(36),
        db.ForeignKey("processed_documents.document_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    entity_type = db.Column(db.String(100), nullable=False, index=True)
    entity_name = db.Column(db.String(500), nullable=False)
    normalized_name = db.Column(db.String(500), index=True)

    # Properties extracted for this entity
    _properties = db.Column("properties", db.Text, default="{}")

    # Chunk IDs where this entity appears
    _chunk_ids = db.Column("chunk_ids", db.Text, default="[]")

    # Confidence score (0-1)
    confidence = db.Column(db.Float, default=1.0)

    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    # Relationships
    document = db.relationship("ProcessedDocument", back_populates="entities")

    @property
    def properties(self) -> Dict[str, Any]:
        return json.loads(self._properties or "{}")

    @properties.setter
    def properties(self, value: Dict[str, Any]):
        self._properties = json.dumps(value, default=str)

    @property
    def chunk_ids(self) -> List[str]:
        return json.loads(self._chunk_ids or "[]")

    @chunk_ids.setter
    def chunk_ids(self, value: List[str]):
        self._chunk_ids = json.dumps(value)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "entity_id": self.entity_id,
            "document_id": self.document_id,
            "entity_type": self.entity_type,
            "entity_name": self.entity_name,
            "normalized_name": self.normalized_name,
            "properties": self.properties,
            "chunk_ids": self.chunk_ids,
            "confidence": self.confidence,
        }


# Index for vector similarity search (created via migration)
# CREATE INDEX ON document_chunks USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
