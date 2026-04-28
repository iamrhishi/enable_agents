"""
Structured JSON logging configuration.

Produces single-line JSON log records for easy ingestion by log aggregators.
"""
import logging
import json
import os
from datetime import datetime, timezone


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
            "module": record.module,
            "func": record.funcName,
            "line": record.lineno,
        }
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload)


def configure_logging(app) -> None:
    """Attach structured JSON logging and request/response middleware to the Flask app."""
    log_level = os.environ.get("LOG_LEVEL", "INFO").upper()
    handler = logging.StreamHandler()
    if os.environ.get("ENVIRONMENT", "development") == "production":
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s — %(message)s"
        ))
    handler.setLevel(log_level)

    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(log_level)

    app.logger.setLevel(log_level)
    app.logger.info("Logging configured (level=%s)", log_level)

    _attach_request_logging(app)


def _attach_request_logging(app) -> None:
    """Log each request and response (method, path, status, duration)."""
    import time
    from flask import g, request

    @app.before_request
    def _before():
        g._request_start = time.monotonic()

    @app.after_request
    def _after(response):
        duration_ms = round((time.monotonic() - getattr(g, "_request_start", time.monotonic())) * 1000, 1)
        # Skip noisy health-check logging in production
        skip = (
            request.path in ("/health",)
            and os.environ.get("ENVIRONMENT") == "production"
        )
        if not skip:
            app.logger.info(
                "%s %s → %s  (%.1fms)",
                request.method,
                request.path,
                response.status_code,
                duration_ms,
            )
        return response
