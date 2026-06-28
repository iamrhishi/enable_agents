"""
Connectors — standardized external data source integrations.

Architecture
────────────
All external data sources (APIs, web scraping, search engines) are implemented
as Connectors with a consistent interface. This allows:

- Reusable auth patterns (OAuth, API keys, session-based)
- Automatic context storage (all fetched data goes to ContextStore)
- Unified error handling and retry logic
- Easy addition of new data sources

Usage
─────
    from core.connectors import get_connector

    # Get a connector instance
    connector = get_connector("google_business", user_id=user_id)

    # Check if connected/authenticated
    if connector.is_connected():
        # Fetch data (automatically stored to context)
        data = connector.fetch(resource="profile")

    # Or use the registry to list available connectors
    from core.connectors import list_connectors
    available = list_connectors()  # ["google_business", "linkedin", "web_scraper", ...]

Connector Types
───────────────
- OAuth Connectors: Google Business, LinkedIn, Salesforce
- API Key Connectors: HubSpot, SendGrid, Search APIs
- Session Connectors: Web scraping (maintains session/cookies)
- Stateless Connectors: Simple HTTP fetchers

Adding a New Connector
──────────────────────
1. Create file in core/connectors/ (e.g., hubspot.py)
2. Inherit from appropriate base (OAuthConnector, APIKeyConnector, etc.)
3. Implement required methods: connect(), fetch(), _transform()
4. Register in CONNECTOR_REGISTRY
"""

from core.connectors.base import (
    BaseConnector,
    OAuthConnector,
    APIKeyConnector,
    SessionConnector,
    ConnectorError,
    AuthenticationError,
    RateLimitError,
)
from core.connectors.registry import (
    get_connector,
    list_connectors,
    register_connector,
    CONNECTOR_REGISTRY,
)

__all__ = [
    "BaseConnector",
    "OAuthConnector",
    "APIKeyConnector",
    "SessionConnector",
    "ConnectorError",
    "AuthenticationError",
    "RateLimitError",
    "get_connector",
    "list_connectors",
    "register_connector",
    "CONNECTOR_REGISTRY",
]
