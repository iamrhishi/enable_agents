"""
Connector Registry — discover and instantiate connectors.

Usage:
    from core.connectors import get_connector, list_connectors

    # Get all available connectors
    available = list_connectors()
    # Returns: [{"id": "google_business", "name": "Google Business", "type": "oauth", ...}, ...]

    # Get a connector instance
    connector = get_connector("web_scraper", user_id="user123")
    if connector.connect():
        data = connector.fetch("page", params={"url": "https://example.com"})
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Type

from core.connectors.base import BaseConnector

logger = logging.getLogger(__name__)

# Registry of all available connectors
# Format: { connector_id: ConnectorClass }
CONNECTOR_REGISTRY: Dict[str, Type[BaseConnector]] = {}


def register_connector(connector_class: Type[BaseConnector]) -> Type[BaseConnector]:
    """
    Decorator to register a connector class.

    Usage:
        @register_connector
        class MyConnector(BaseConnector):
            connector_id = "my_connector"
            ...
    """
    connector_id = getattr(connector_class, "connector_id", None)
    if not connector_id:
        raise ValueError(f"Connector class {connector_class.__name__} must define connector_id")

    CONNECTOR_REGISTRY[connector_id] = connector_class
    logger.debug(f"Registered connector: {connector_id}")
    return connector_class


def get_connector(
    connector_id: str,
    user_id: str,
    config: Optional[Dict[str, Any]] = None,
) -> BaseConnector:
    """
    Get a connector instance by ID.

    Args:
        connector_id: Connector identifier
        user_id: User ID for context storage
        config: Optional configuration overrides

    Returns:
        Connector instance

    Raises:
        ValueError: If connector not found
    """
    # Lazy load connectors to avoid circular imports
    _ensure_connectors_loaded()

    if connector_id not in CONNECTOR_REGISTRY:
        available = list(CONNECTOR_REGISTRY.keys())
        raise ValueError(
            f"Connector '{connector_id}' not found. Available: {available}"
        )

    connector_class = CONNECTOR_REGISTRY[connector_id]
    return connector_class(user_id=user_id, config=config)


def list_connectors() -> List[Dict[str, Any]]:
    """
    List all registered connectors with their metadata.

    Returns:
        List of connector info dicts
    """
    _ensure_connectors_loaded()

    result = []
    for connector_id, connector_class in CONNECTOR_REGISTRY.items():
        result.append({
            "id": connector_id,
            "name": getattr(connector_class, "connector_name", connector_id),
            "type": getattr(connector_class, "connector_type", "unknown"),
            "resources": getattr(connector_class, "supported_resources", []),
        })
    return result


def _ensure_connectors_loaded() -> None:
    """Lazy-load all connector modules."""
    if CONNECTOR_REGISTRY:
        return  # Already loaded

    # Import connector modules to trigger registration
    try:
        from core.connectors import web_scraper
    except ImportError as e:
        logger.debug(f"Could not load web_scraper connector: {e}")

    try:
        from core.connectors import google_business
    except ImportError as e:
        logger.debug(f"Could not load google_business connector: {e}")

    try:
        from core.connectors import web_search
    except ImportError as e:
        logger.debug(f"Could not load web_search connector: {e}")

    try:
        from core.connectors import linkedin
    except ImportError as e:
        logger.debug(f"Could not load linkedin connector: {e}")
