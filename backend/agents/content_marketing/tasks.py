"""
Content Marketing Agent — Celery async tasks.

Usage:
    from agents.content_marketing.tasks import generate_content_async
    result = generate_content_async.delay(project_id, payload)
"""
from core.celery_app import celery


@celery.task(bind=True, name="content_marketing.generate_content_async")
def generate_content_async(self, project_id: str, payload: dict) -> dict:
    """Run content generation in a background worker."""
    try:
        self.update_state(state="PROGRESS", meta={"step": "generating"})
        # Import service lazily to avoid circular imports at module load time
        from agents.content_marketing import service  # noqa: F401
        # TODO: call the actual generation logic here
        return {"status": "done", "project_id": project_id}
    except Exception as exc:
        self.update_state(state="FAILURE", meta={"error": str(exc)})
        raise
