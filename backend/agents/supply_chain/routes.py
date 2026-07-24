"""Supply Chain Audit Agent — Flask Blueprint."""
from flask import Blueprint

from core.auth import require_auth

from . import service

supply_chain_bp = Blueprint(
    "supply_chain",
    __name__,
    url_prefix="/api/supply-chain",
)


@supply_chain_bp.get("/suppliers")
@require_auth
def list_suppliers():
    return service.list_suppliers()


@supply_chain_bp.post("/suppliers")
@require_auth
def create_supplier():
    return service.create_supplier()


@supply_chain_bp.put("/suppliers/<supplier_id>/audit")
@require_auth
def submit_audit(supplier_id):
    return service.submit_audit(supplier_id)
