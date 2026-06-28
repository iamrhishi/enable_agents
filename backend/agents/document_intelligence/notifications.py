"""
Document Intelligence Agent — SSE notification helpers.

Sends real-time notifications for document processing status via Redis pub/sub.
"""

import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


def get_redis_client():
    """Get Redis client for pub/sub."""
    try:
        import redis

        url = os.environ.get("CELERY_BROKER_URL") or os.environ.get(
            "REDIS_URL", "redis://localhost:6379/0"
        )
        client = redis.from_url(url, decode_responses=True)
        client.ping()
        return client
    except Exception as e:
        logger.warning(f"Redis unavailable for notifications: {e}")
        return None


def send_document_notification(
    user_id: str,
    document_id: str,
    stage: str,
    progress: float,
    message: str,
    data: Optional[Dict[str, Any]] = None,
) -> bool:
    """
    Send document processing notification via Redis pub/sub.

    Args:
        user_id: User ID to notify
        document_id: Document being processed
        stage: Processing stage (started, extracting, chunking, embedding, entities, completed, failed)
        progress: Progress percentage (0.0 - 1.0)
        message: Human-readable status message
        data: Optional additional data

    Returns:
        True if notification was sent
    """
    redis_client = get_redis_client()
    if not redis_client:
        return False

    notification = {
        "type": "document_processing",
        "user_id": user_id,
        "document_id": document_id,
        "stage": stage,
        "progress": progress,
        "message": message,
        "data": data or {},
        "timestamp": datetime.utcnow().isoformat(),
    }

    try:
        # Publish to user-specific channel
        channel = f"notifications:{user_id}"
        redis_client.publish(channel, json.dumps(notification))
        logger.debug(f"Sent notification to {channel}: {stage}")
        return True
    except Exception as e:
        logger.error(f"Failed to send notification: {e}")
        return False


def send_document_ready_notification(
    user_id: str,
    document_id: str,
    document_name: str,
    chunk_count: int,
    entity_count: int,
) -> bool:
    """Send notification that document is ready for queries."""
    return send_document_notification(
        user_id=user_id,
        document_id=document_id,
        stage="ready",
        progress=1.0,
        message=f"Document '{document_name}' is ready",
        data={
            "document_name": document_name,
            "chunk_count": chunk_count,
            "entity_count": entity_count,
        },
    )


def send_document_error_notification(
    user_id: str,
    document_id: str,
    error_message: str,
) -> bool:
    """Send notification that document processing failed."""
    return send_document_notification(
        user_id=user_id,
        document_id=document_id,
        stage="error",
        progress=0.0,
        message=f"Processing failed: {error_message}",
        data={"error": error_message},
    )
