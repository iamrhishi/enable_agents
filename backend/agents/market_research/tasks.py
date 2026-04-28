"""Market Research Agent — Celery async tasks."""
from core.celery_app import celery


@celery.task(bind=True, name="market_research.run_research_async")
def run_research_async(self, project_id: str) -> dict:
    """Run market research pipeline asynchronously."""
    try:
        self.update_state(state="PROGRESS", meta={"project_id": project_id, "step": "scraping"})
        from agents.market_research.models import ResearchProject, ResearchResult
        from core.database import db

        project = ResearchProject.query.filter_by(project_id=project_id).first()
        if not project:
            return {"status": "error", "error": "Project not found"}

        project.status = "running"
        db.session.commit()

        # TODO: plug in web scraping, OpenAI analysis, and competitor extraction here

        project.status = "done"
        db.session.commit()
        return {"status": "done", "project_id": project_id}
    except Exception as exc:
        self.update_state(state="FAILURE", meta={"error": str(exc)})
        raise
