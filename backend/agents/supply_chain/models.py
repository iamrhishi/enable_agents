"""Supply Chain Audit Agent — SQLAlchemy models."""
import json
from datetime import datetime
from typing import Any, Dict, List

from core.database import db


class SCSupplier(db.Model):
    """A supplier tracked for qualification audits, scoped to a project."""

    __tablename__ = "sc_suppliers"

    supplier_id = db.Column(db.String(36), primary_key=True)
    project_id = db.Column(db.String(36), nullable=False, index=True)
    user_id = db.Column(db.String(255), nullable=False, index=True)
    name = db.Column(db.String(255), nullable=False)
    location = db.Column(db.String(255))
    capacity = db.Column(db.String(255))
    _certifications = db.Column("certifications", db.Text, default="[]")
    _capabilities = db.Column("capabilities", db.Text, default="[]")
    audit_status = db.Column(db.String(20), default="pending")  # pending|scheduled|passed|failed
    score = db.Column(db.Integer)
    audit_date = db.Column(db.String(20))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    @property
    def certifications(self) -> List[str]:
        try:
            return json.loads(self._certifications or "[]")
        except Exception:
            return []

    @certifications.setter
    def certifications(self, value: List[str]):
        self._certifications = json.dumps(value or [])

    @property
    def capabilities(self) -> List[str]:
        try:
            return json.loads(self._capabilities or "[]")
        except Exception:
            return []

    @capabilities.setter
    def capabilities(self, value: List[str]):
        self._capabilities = json.dumps(value or [])

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.supplier_id,
            "name": self.name,
            "location": self.location,
            "capacity": self.capacity,
            "certifications": self.certifications,
            "capabilities": self.capabilities,
            "auditStatus": self.audit_status,
            "score": self.score,
            "auditDate": self.audit_date,
        }
