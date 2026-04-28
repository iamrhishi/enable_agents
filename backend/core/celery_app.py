"""
Celery factory.

Usage (in app factory):
    from core.celery_app import make_celery
    celery = make_celery(app)

Usage (in worker command):
    celery -A core.celery_app worker
"""
import os
from celery import Celery

celery: Celery | None = None


def make_celery(app) -> Celery:
    global celery
    broker = os.environ.get("CELERY_BROKER_URL", "redis://localhost:6379/0")
    backend = os.environ.get("CELERY_RESULT_BACKEND", "redis://localhost:6379/0")

    celery = Celery(
        app.import_name,
        broker=broker,
        backend=backend,
    )
    celery.conf.update(
        task_serializer="json",
        result_serializer="json",
        accept_content=["json"],
        timezone="UTC",
        enable_utc=True,
        task_track_started=True,
    )

    class ContextTask(celery.Task):
        def __call__(self, *args, **kwargs):
            with app.app_context():
                return self.run(*args, **kwargs)

    celery.Task = ContextTask
    return celery
