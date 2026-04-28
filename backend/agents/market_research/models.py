"""Market Research Agent — SQLAlchemy models."""
from datetime import datetime
from core.database import db


class ResearchProject(db.Model):
    __tablename__ = "mr_projects"

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    project_id = db.Column(db.String(36), unique=True, nullable=False, index=True)
    user_id = db.Column(db.String(255), nullable=False, index=True)
    company_name = db.Column(db.String(255))
    industry = db.Column(db.String(255))
    research_goals = db.Column(db.Text)  # JSON-serialised list
    status = db.Column(db.String(50), default="pending")  # pending | running | done | failed
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    results = db.relationship("ResearchResult", back_populates="project", cascade="all, delete-orphan")


class ResearchResult(db.Model):
    __tablename__ = "mr_results"

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    project_id = db.Column(db.String(36), db.ForeignKey("mr_projects.project_id"), nullable=False, index=True)
    result_type = db.Column(db.String(100))  # competitors | market_size | requirements | scraped_data
    content = db.Column(db.Text)
    source_url = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    project = db.relationship("ResearchProject", back_populates="results")
