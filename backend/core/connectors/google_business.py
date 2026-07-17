"""
Google Business Connector — Google Business Profile API integration.

Usage:
    from core.connectors import get_connector

    connector = get_connector("google_business", user_id="user123")

    # Check if user has connected their Google account
    if not connector.is_connected():
        # Get OAuth URL for user to authorize
        auth_url = connector.get_authorization_url(
            redirect_uri="https://app.example.com/oauth/callback",
            scopes=["https://www.googleapis.com/auth/business.manage"]
        )
        # Redirect user to auth_url...

    # After authorization, fetch data
    if connector.connect():
        profile = connector.fetch("profile")
        reviews = connector.fetch("reviews")
        insights = connector.fetch("insights", params={"days": 30})
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from core.connectors.base import OAuthConnector, ConnectorError, AuthenticationError
from core.connectors.registry import register_connector

logger = logging.getLogger(__name__)


@register_connector
class GoogleBusinessConnector(OAuthConnector):
    """
    Google Business Profile API connector.

    Requires:
    - GOOGLE_CLIENT_ID environment variable
    - GOOGLE_CLIENT_SECRET environment variable
    - User OAuth authorization
    """

    connector_id = "google_business"
    connector_name = "Google Business Profile"
    supported_resources = ["profile", "reviews", "insights", "locations", "posts"]

    OAUTH_AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
    OAUTH_TOKEN_URL = "https://oauth2.googleapis.com/token"
    API_BASE_URL = "https://mybusiness.googleapis.com/v4"
    API_V1_BASE_URL = "https://mybusinessbusinessinformation.googleapis.com/v1"

    def __init__(self, user_id: str, config: Optional[Dict[str, Any]] = None):
        super().__init__(user_id, config)
        self.client_id = config.get("client_id") or os.environ.get("GOOGLE_CLIENT_ID")
        self.client_secret = config.get("client_secret") or os.environ.get("GOOGLE_CLIENT_SECRET")
        self.account_id: Optional[str] = None
        self.location_id: Optional[str] = None

    def get_authorization_url(
        self,
        redirect_uri: str,
        scopes: Optional[List[str]] = None,
    ) -> str:
        """Generate OAuth authorization URL."""
        import urllib.parse

        if not self.client_id:
            raise ConnectorError("GOOGLE_CLIENT_ID not configured")

        scopes = scopes or [
            "https://www.googleapis.com/auth/business.manage",
        ]

        params = {
            "client_id": self.client_id,
            "redirect_uri": redirect_uri,
            "response_type": "code",
            "scope": " ".join(scopes),
            "access_type": "offline",
            "prompt": "consent",
            "state": self.user_id,  # For CSRF protection
        }

        return f"{self.OAUTH_AUTH_URL}?{urllib.parse.urlencode(params)}"

    def exchange_code(self, code: str, redirect_uri: str) -> Dict[str, Any]:
        """Exchange authorization code for tokens."""
        import requests

        if not self.client_id or not self.client_secret:
            raise ConnectorError("Google OAuth credentials not configured")

        response = requests.post(
            self.OAUTH_TOKEN_URL,
            data={
                "client_id": self.client_id,
                "client_secret": self.client_secret,
                "code": code,
                "redirect_uri": redirect_uri,
                "grant_type": "authorization_code",
            },
        )

        if response.status_code != 200:
            raise AuthenticationError(f"Token exchange failed: {response.text}")

        token_data = response.json()

        self.access_token = token_data.get("access_token")
        self.refresh_token = token_data.get("refresh_token")

        expires_in = token_data.get("expires_in", 3600)
        self.token_expires_at = datetime.utcnow() + timedelta(seconds=expires_in)

        self._save_tokens()
        self._connected = True

        return token_data

    def refresh_access_token(self) -> bool:
        """Refresh the access token."""
        import requests

        if not self.refresh_token:
            return False

        response = requests.post(
            self.OAUTH_TOKEN_URL,
            data={
                "client_id": self.client_id,
                "client_secret": self.client_secret,
                "refresh_token": self.refresh_token,
                "grant_type": "refresh_token",
            },
        )

        if response.status_code != 200:
            logger.error(f"Token refresh failed: {response.text}")
            return False

        token_data = response.json()
        self.access_token = token_data.get("access_token")

        expires_in = token_data.get("expires_in", 3600)
        self.token_expires_at = datetime.utcnow() + timedelta(seconds=expires_in)

        self._save_tokens()
        self._connected = True

        return True

    def fetch(
        self,
        resource: str,
        params: Optional[Dict[str, Any]] = None,
        store: bool = True,
    ) -> Dict[str, Any]:
        """
        Fetch data from Google Business API.

        Args:
            resource: Resource type (profile, reviews, insights, locations, posts)
            params: Resource-specific parameters
            store: Whether to store to context
        """
        if not self.is_connected():
            if not self.connect():
                raise AuthenticationError("Not connected to Google Business")

        params = params or {}

        try:
            if resource == "locations":
                raw_data = self._fetch_locations()
            elif resource == "profile":
                raw_data = self._fetch_profile(params.get("location_id"))
            elif resource == "reviews":
                raw_data = self._fetch_reviews(params.get("location_id"))
            elif resource == "insights":
                raw_data = self._fetch_insights(
                    params.get("location_id"),
                    params.get("days", 30),
                )
            elif resource == "posts":
                raw_data = self._fetch_posts(params.get("location_id"))
            else:
                raise ConnectorError(f"Unknown resource: {resource}")

            data = self._transform(raw_data, resource)

            if store:
                self.store_to_context(data, resource)

            return data

        except Exception as e:
            logger.error(f"Google Business fetch failed: {e}")
            raise ConnectorError(f"Fetch failed: {e}")

    def _api_request(self, endpoint: str, method: str = "GET", **kwargs) -> Dict[str, Any]:
        """Make authenticated API request."""
        import requests

        headers = {
            "Authorization": f"Bearer {self.access_token}",
            "Content-Type": "application/json",
        }

        url = f"{self.API_BASE_URL}/{endpoint}"

        response = requests.request(method, url, headers=headers, **kwargs)

        if response.status_code == 401:
            # Try refresh
            if self.refresh_access_token():
                headers["Authorization"] = f"Bearer {self.access_token}"
                response = requests.request(method, url, headers=headers, **kwargs)
            else:
                raise AuthenticationError("Token expired and refresh failed")

        if response.status_code != 200:
            raise ConnectorError(f"API error: {response.status_code} - {response.text}")

        return response.json()

    def _fetch_locations(self) -> Dict[str, Any]:
        """Fetch all business locations."""
        # First get accounts
        accounts = self._api_request("accounts")
        account_list = accounts.get("accounts", [])

        if not account_list:
            return {"locations": []}

        # Use first account
        account_name = account_list[0].get("name")
        self.account_id = account_name.split("/")[-1]

        # Get locations for account
        locations = self._api_request(f"{account_name}/locations")

        return {
            "account_id": self.account_id,
            "locations": locations.get("locations", []),
        }

    def _fetch_profile(self, location_id: Optional[str] = None) -> Dict[str, Any]:
        """Fetch business profile."""
        if not location_id and not self.location_id:
            # Get first location
            locations = self._fetch_locations()
            if locations.get("locations"):
                self.location_id = locations["locations"][0].get("name", "").split("/")[-1]

        location_id = location_id or self.location_id
        if not location_id:
            return {"error": "No location found"}

        location_name = f"accounts/{self.account_id}/locations/{location_id}"
        return self._api_request(location_name)

    def _fetch_reviews(self, location_id: Optional[str] = None) -> Dict[str, Any]:
        """Fetch reviews for a location."""
        location_id = location_id or self.location_id
        if not location_id:
            return {"reviews": []}

        location_name = f"accounts/{self.account_id}/locations/{location_id}"
        return self._api_request(f"{location_name}/reviews")

    def _fetch_insights(
        self,
        location_id: Optional[str] = None,
        days: int = 30,
    ) -> Dict[str, Any]:
        """Fetch location insights/analytics."""
        location_id = location_id or self.location_id
        if not location_id:
            return {"insights": {}}

        # Note: Insights API has specific requirements
        # This is a simplified implementation
        location_name = f"accounts/{self.account_id}/locations/{location_id}"

        return self._api_request(
            f"{location_name}/insights",
            params={"basicRequest.metricRequests.metric": "ALL"},
        )

    def _fetch_posts(self, location_id: Optional[str] = None) -> Dict[str, Any]:
        """Fetch local posts."""
        location_id = location_id or self.location_id
        if not location_id:
            return {"posts": []}

        location_name = f"accounts/{self.account_id}/locations/{location_id}"
        return self._api_request(f"{location_name}/localPosts")

    def _transform(self, raw_data: Any, resource: str) -> Dict[str, Any]:
        """Transform API response to standard format."""
        return {
            "resource_type": resource,
            "fetched_at": datetime.utcnow().isoformat(),
            "connector": self.connector_id,
            **raw_data,
        }
