"""Supply Chain Audit Agent — service layer."""
from datetime import datetime
from uuid import uuid4

from flask import jsonify, request

from core.database import db
from .models import SCSupplier


def list_suppliers():
    project_id = request.args.get("project_id")
    if not project_id:
        return jsonify({"success": False, "error": "project_id is required"}), 400

    suppliers = SCSupplier.query.filter_by(project_id=project_id).order_by(SCSupplier.created_at.desc()).all()
    return jsonify({"success": True, "suppliers": [s.to_dict() for s in suppliers]}), 200


def create_supplier():
    data = request.get_json(silent=True) or {}
    name = (data.get("name") or "").strip()
    project_id = data.get("project_id")

    if not name:
        return jsonify({"success": False, "error": "Supplier name is required"}), 400
    if not project_id:
        return jsonify({"success": False, "error": "project_id is required"}), 400

    supplier = SCSupplier(
        supplier_id=str(uuid4()),
        project_id=project_id,
        user_id=data.get("user_id", "anonymous"),
        name=name,
        location=data.get("location"),
        capacity=data.get("capacity"),
        audit_status="pending",
    )
    supplier.certifications = data.get("certifications") or []
    supplier.capabilities = data.get("capabilities") or []
    db.session.add(supplier)
    db.session.commit()
    return jsonify({"success": True, "supplier": supplier.to_dict()}), 201


def submit_audit(supplier_id):
    supplier = SCSupplier.query.get(supplier_id)
    if not supplier:
        return jsonify({"success": False, "error": "Supplier not found"}), 404

    data = request.get_json(silent=True) or {}
    score = data.get("score")
    if score is None:
        return jsonify({"success": False, "error": "score is required"}), 400

    supplier.score = score
    supplier.audit_status = "passed" if score >= 70 else "failed"
    supplier.audit_date = datetime.utcnow().strftime("%Y-%m-%d")
    db.session.commit()
    return jsonify({"success": True, "supplier": supplier.to_dict()}), 200
