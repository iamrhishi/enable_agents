"""
Content Marketing Agent — SQLAlchemy models.

These replace the raw sqlite3 CREATE TABLE statements in app.py's
init_content_marketing_db(). Flask-Migrate will generate the migration
automatically once these models are imported into the app context.
"""
import json
from datetime import datetime

from core.database import db


class CMProject(db.Model):
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

    @property
    def metadata_dict(self):
        return json.loads(self._metadata or "{}")


class CMDocument(db.Model):
    __tablename__ = "cm_documents"

    doc_id = db.Column(db.String(36), primary_key=True)
    project_id = db.Column(db.String(36), db.ForeignKey("cm_projects.project_id"), nullable=False, index=True)
    file_name = db.Column(db.String(255), nullable=False)
    file_type = db.Column(db.String(50))
    file_path = db.Column(db.Text)
    file_size = db.Column(db.Integer)
    upload_date = db.Column(db.DateTime, default=datetime.utcnow)
    document_type = db.Column(db.String(100))
    extracted_content = db.Column(db.Text)

    project = db.relationship("CMProject", back_populates="documents")


class CMKnowledgeGraph(db.Model):
    __tablename__ = "cm_knowledge_graphs"

    kg_id = db.Column(db.String(36), primary_key=True)
    project_id = db.Column(db.String(36), db.ForeignKey("cm_projects.project_id"), nullable=False, index=True)
    graph_data = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    project = db.relationship("CMProject", back_populates="knowledge_graphs")
