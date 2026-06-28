"""
SSE Notifications — Server-Sent Events for real-time updates.

Provides:
- SSE endpoint for streaming notifications
- Redis pub/sub for cross-process notification delivery
- Connection management with heartbeat

Usage:
    # In app.py, register the routes:
    from core.notifications import register_sse_routes
    register_sse_routes(app)

    # To send a notification from anywhere:
    from core.notifications import send_notification
    send_notification(user_id, "document_ready", {"document_id": "..."})
"""

import json
import logging
import os
import time
from datetime import datetime
from typing import Any, Dict, Generator, Optional

from flask import Response, request, g

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
        logger.warning(f"Redis unavailable for SSE: {e}")
        return None


def send_notification(
    user_id: str,
    notification_type: str,
    data: Optional[Dict[str, Any]] = None,
    message: Optional[str] = None,
) -> bool:
    """
    Send notification to a user via Redis pub/sub.

    Args:
        user_id: Target user ID
        notification_type: Type of notification (e.g., "document_ready", "error")
        data: Optional payload data
        message: Optional human-readable message

    Returns:
        True if notification was published
    """
    redis_client = get_redis_client()
    if not redis_client:
        logger.warning("Cannot send notification: Redis unavailable")
        return False

    notification = {
        "type": notification_type,
        "data": data or {},
        "message": message,
        "timestamp": datetime.utcnow().isoformat(),
    }

    try:
        channel = f"notifications:{user_id}"
        redis_client.publish(channel, json.dumps(notification))
        logger.debug(f"Published notification to {channel}: {notification_type}")
        return True
    except Exception as e:
        logger.error(f"Failed to publish notification: {e}")
        return False


def _event_stream(user_id: str) -> Generator[str, None, None]:
    """
    Generate SSE events for a user.

    Subscribes to Redis pub/sub channel and yields events.
    Sends heartbeat every 30 seconds to keep connection alive.
    """
    redis_client = get_redis_client()
    if not redis_client:
        yield _format_sse({"type": "error", "message": "Notification service unavailable"})
        return

    pubsub = redis_client.pubsub()
    channel = f"notifications:{user_id}"

    try:
        pubsub.subscribe(channel)
        logger.info(f"SSE: User {user_id} connected to {channel}")

        # Send connection established message
        yield _format_sse({
            "type": "connected",
            "message": "Connected to notification stream",
            "timestamp": datetime.utcnow().isoformat(),
        })

        last_heartbeat = time.time()
        heartbeat_interval = 30  # seconds

        while True:
            # Check for messages (non-blocking with timeout)
            message = pubsub.get_message(timeout=1.0)

            if message and message["type"] == "message":
                try:
                    data = json.loads(message["data"])
                    yield _format_sse(data)
                except json.JSONDecodeError:
                    logger.warning(f"Invalid notification JSON: {message['data']}")

            # Send heartbeat periodically
            if time.time() - last_heartbeat > heartbeat_interval:
                yield _format_sse({
                    "type": "heartbeat",
                    "timestamp": datetime.utcnow().isoformat(),
                })
                last_heartbeat = time.time()

    except GeneratorExit:
        logger.info(f"SSE: User {user_id} disconnected")
    except Exception as e:
        logger.error(f"SSE error for user {user_id}: {e}")
        yield _format_sse({"type": "error", "message": str(e)})
    finally:
        try:
            pubsub.unsubscribe(channel)
            pubsub.close()
        except Exception:
            pass


def _format_sse(data: Dict[str, Any]) -> str:
    """Format data as SSE event."""
    return f"data: {json.dumps(data)}\n\n"


def _get_user_id_from_request() -> Optional[str]:
    """Extract user ID from request (token or header)."""
    # Check g.user (set by auth middleware)
    if hasattr(g, "user") and g.user:
        return g.user.get("user_id") or g.user.get("id") or g.user.get("email")

    # Check Authorization header for Bearer token
    auth_header = request.headers.get("Authorization", "")
    if auth_header.startswith("Bearer "):
        token = auth_header[7:]
        # For SSE, we accept the token as user ID in development
        # In production, this should validate the JWT
        if token:
            return token

    # Check query param (for EventSource which can't set headers)
    token = request.args.get("token")
    if token:
        return token

    # Fallback header
    return request.headers.get("X-User-Id")


def register_sse_routes(app):
    """
    Register SSE notification routes with Flask app.

    Adds:
    - GET /api/notifications/stream — SSE endpoint
    - POST /api/notifications/test — Test endpoint (dev only)
    """

    @app.route("/api/notifications/stream")
    def notification_stream():
        """
        SSE endpoint for real-time notifications.

        Query params:
        - token: User auth token (required for EventSource clients)

        Headers:
        - Authorization: Bearer <token>

        Returns:
            SSE stream of notifications
        """
        user_id = _get_user_id_from_request()

        if not user_id:
            return {"error": "Authentication required"}, 401

        return Response(
            _event_stream(user_id),
            mimetype="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",  # Disable nginx buffering
                "Access-Control-Allow-Origin": "*",
            },
        )

    @app.route("/api/notifications/test", methods=["POST"])
    def test_notification():
        """
        Send a test notification (development only).

        Request body:
        {
            "user_id": "...",
            "type": "test",
            "message": "Test notification"
        }
        """
        if os.environ.get("ENVIRONMENT") == "production":
            return {"error": "Not available in production"}, 403

        data = request.get_json() or {}
        user_id = data.get("user_id") or _get_user_id_from_request()

        if not user_id:
            return {"error": "user_id required"}, 400

        success = send_notification(
            user_id=user_id,
            notification_type=data.get("type", "test"),
            message=data.get("message", "Test notification"),
            data=data.get("data"),
        )

        return {"success": success}, 200 if success else 500

    logger.info("SSE notification routes registered")
