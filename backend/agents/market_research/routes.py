"""Market Research Agent — Flask Blueprint."""
from flask import Blueprint

from core.auth import require_auth

from . import service

market_research_bp = Blueprint("market_research", __name__, url_prefix="/api/market-research")


@market_research_bp.post("/projects")
@require_auth
def create_project():
    return service.create_project()


@market_research_bp.get("/projects/<project_id>")
@require_auth
def get_project(project_id: str):
    return service.get_project(project_id)


@market_research_bp.post("/projects/<project_id>/run")
@require_auth
def run_research(project_id: str):
    return service.run_research(project_id)


@market_research_bp.get("/projects/<project_id>/results")
@require_auth
def get_results(project_id: str):
    return service.get_results(project_id)


@market_research_bp.post("/requirements")
@require_auth
def generate_requirements():
    return service.generate_requirements()


@market_research_bp.post("/search")
@require_auth
def simple_search():
    return service.simple_search()


@market_research_bp.post("/recommend-agents")
@require_auth
def recommend_agents():
    return service.recommend_agents()
