"""
LinkedIn Connector — LinkedIn API integration via OAuth 2.0.

Usage:
    from core.connectors import get_connector

    connector = get_connector("linkedin", user_id="user123")

    # Check if user has connected their LinkedIn account
    if not connector.is_connected():
        auth_url = connector.get_authorization_url(
            redirect_uri="https://app.example.com/oauth/callback",
            scopes=["openid", "profile", "email"]
        )
        # Redirect user to auth_url...

    # After authorization, fetch data
    if connector.connect():
        profile = connector.fetch("profile")
        connections = connector.fetch("connections")
        companies = connector.fetch("companies")
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
class LinkedInConnector(OAuthConnector):
    """
    LinkedIn API connector using OAuth 2.0.

    Requires:
    - LINKEDIN_CLIENT_ID environment variable (or in UserSettings)
    - LINKEDIN_CLIENT_SECRET environment variable (or in UserSettings)
    - User OAuth authorization

    LinkedIn API v2 endpoints used:
    - /v2/userinfo - OpenID Connect profile
    - /v2/me - Basic profile
    - /v2/connections - Connections (requires special access)
    - /v2/organizationalEntityAcls - Managed companies
    """

    connector_id = "linkedin"
    connector_name = "LinkedIn"
    supported_resources = ["profile", "connections", "companies", "posts"]

    OAUTH_AUTH_URL = "https://www.linkedin.com/oauth/v2/authorization"
    OAUTH_TOKEN_URL = "https://www.linkedin.com/oauth/v2/accessToken"
    API_BASE_URL = "https://api.linkedin.com/v2"
    OPENID_USERINFO_URL = "https://api.linkedin.com/v2/userinfo"

    DEFAULT_SCOPES = [
        "openid",
        "profile",
        "email",
    ]

    def __init__(self, user_id: str, config: Optional[Dict[str, Any]] = None):
        super().__init__(user_id, config)
        self.client_id = None
        self.client_secret = None

    def _load_credentials(self) -> None:
        """Load OAuth credentials from config, UserSettings, or environment."""
        self.client_id = (
            self._config_overrides.get("client_id")
            or os.environ.get("LINKEDIN_CLIENT_ID")
        )
        self.client_secret = (
            self._config_overrides.get("client_secret")
            or os.environ.get("LINKEDIN_CLIENT_SECRET")
        )

    def get_authorization_url(
        self,
        redirect_uri: str,
        scopes: Optional[List[str]] = None,
    ) -> str:
        """Generate OAuth authorization URL."""
        import urllib.parse

        self._load_credentials()

        if not self.client_id:
            raise ConnectorError("LINKEDIN_CLIENT_ID not configured")

        scopes = scopes or self.DEFAULT_SCOPES

        params = {
            "response_type": "code",
            "client_id": self.client_id,
            "redirect_uri": redirect_uri,
            "scope": " ".join(scopes),
            "state": self.user_id,  # For CSRF protection
        }

        return f"{self.OAUTH_AUTH_URL}?{urllib.parse.urlencode(params)}"

    def exchange_code(self, code: str, redirect_uri: str) -> Dict[str, Any]:
        """Exchange authorization code for tokens."""
        import requests

        self._load_credentials()

        if not self.client_id or not self.client_secret:
            raise ConnectorError("LinkedIn OAuth credentials not configured")

        response = requests.post(
            self.OAUTH_TOKEN_URL,
            data={
                "grant_type": "authorization_code",
                "code": code,
                "redirect_uri": redirect_uri,
                "client_id": self.client_id,
                "client_secret": self.client_secret,
            },
            headers={
                "Content-Type": "application/x-www-form-urlencoded",
            },
        )

        if response.status_code != 200:
            raise AuthenticationError(f"Token exchange failed: {response.text}")

        token_data = response.json()

        self.access_token = token_data.get("access_token")
        self.refresh_token = token_data.get("refresh_token")

        # LinkedIn tokens typically expire in 60 days (5184000 seconds)
        expires_in = token_data.get("expires_in", 5184000)
        self.token_expires_at = datetime.utcnow() + timedelta(seconds=expires_in)

        self._save_tokens()
        self._connected = True

        return token_data

    def refresh_access_token(self) -> bool:
        """
        Refresh the access token.

        Note: LinkedIn refresh tokens require special approval.
        Most apps will need to re-authorize after token expiry.
        """
        import requests

        if not self.refresh_token:
            logger.warning("No refresh token available for LinkedIn")
            return False

        self._load_credentials()

        response = requests.post(
            self.OAUTH_TOKEN_URL,
            data={
                "grant_type": "refresh_token",
                "refresh_token": self.refresh_token,
                "client_id": self.client_id,
                "client_secret": self.client_secret,
            },
            headers={
                "Content-Type": "application/x-www-form-urlencoded",
            },
        )

        if response.status_code != 200:
            logger.error(f"LinkedIn token refresh failed: {response.text}")
            return False

        token_data = response.json()
        self.access_token = token_data.get("access_token")

        expires_in = token_data.get("expires_in", 5184000)
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
        Fetch data from LinkedIn API.

        Args:
            resource: Resource type (profile, connections, companies, posts)
            params: Resource-specific parameters
            store: Whether to store to context
        """
        if not self.is_connected():
            if not self.connect():
                raise AuthenticationError("Not connected to LinkedIn")

        params = params or {}

        try:
            if resource == "profile":
                raw_data = self._fetch_profile()
            elif resource == "connections":
                raw_data = self._fetch_connections(params.get("start", 0), params.get("count", 50))
            elif resource == "companies":
                raw_data = self._fetch_companies()
            elif resource == "posts":
                raw_data = self._fetch_posts(params.get("count", 20))
            else:
                raise ConnectorError(f"Unknown resource: {resource}")

            data = self._transform(raw_data, resource)

            if store:
                self.store_to_context(data, resource)

            return data

        except Exception as e:
            logger.error(f"LinkedIn fetch failed: {e}")
            raise ConnectorError(f"Fetch failed: {e}")

    def _api_request(
        self,
        endpoint: str,
        method: str = "GET",
        use_openid: bool = False,
        **kwargs,
    ) -> Dict[str, Any]:
        """Make authenticated API request."""
        import requests

        headers = {
            "Authorization": f"Bearer {self.access_token}",
            "Content-Type": "application/json",
            "X-Restli-Protocol-Version": "2.0.0",
        }

        if use_openid:
            url = self.OPENID_USERINFO_URL
        else:
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

    def _fetch_profile(self) -> Dict[str, Any]:
        """Fetch user profile using OpenID Connect."""
        # Use OpenID userinfo endpoint (simpler, standardized)
        profile = self._api_request("", use_openid=True)

        return {
            "id": profile.get("sub"),
            "name": profile.get("name"),
            "given_name": profile.get("given_name"),
            "family_name": profile.get("family_name"),
            "email": profile.get("email"),
            "email_verified": profile.get("email_verified"),
            "picture": profile.get("picture"),
            "locale": profile.get("locale"),
        }

    def _fetch_connections(self, start: int = 0, count: int = 50) -> Dict[str, Any]:
        """
        Fetch user's connections.

        Note: This requires the 'r_1st_connections_size' scope which is restricted.
        Most apps won't have access to this.
        """
        try:
            response = self._api_request(
                f"connections?start={start}&count={count}&projection=(elements*(to~))"
            )
            return {
                "connections": response.get("elements", []),
                "total": response.get("paging", {}).get("total", 0),
            }
        except ConnectorError:
            # Connections API requires special access
            return {
                "connections": [],
                "total": 0,
                "note": "Connections API requires special LinkedIn partnership access",
            }

    def _fetch_companies(self) -> Dict[str, Any]:
        """Fetch companies the user has admin access to."""
        try:
            response = self._api_request("organizationalEntityAcls?q=roleAssignee")
            elements = response.get("elements", [])

            companies = []
            for elem in elements:
                org_urn = elem.get("organizationalTarget")
                if org_urn:
                    # org_urn format: urn:li:organization:123456
                    org_id = org_urn.split(":")[-1]
                    companies.append({
                        "id": org_id,
                        "urn": org_urn,
                        "role": elem.get("role"),
                    })

            return {"companies": companies}

        except ConnectorError:
            return {
                "companies": [],
                "note": "Organization API requires Marketing Developer Platform access",
            }

    def _fetch_posts(self, count: int = 20) -> Dict[str, Any]:
        """
        Fetch user's posts/shares.

        Note: This requires w_member_social scope.
        """
        try:
            # Get user URN first
            profile = self._fetch_profile()
            user_urn = f"urn:li:person:{profile.get('id')}"

            response = self._api_request(
                f"shares?q=owners&owners={user_urn}&count={count}"
            )

            return {
                "posts": response.get("elements", []),
                "total": response.get("paging", {}).get("total", 0),
            }

        except ConnectorError:
            return {
                "posts": [],
                "note": "Posts API requires member_social scope",
            }

    def _transform(self, raw_data: Any, resource: str) -> Dict[str, Any]:
        """Transform API response to standard format."""
        return {
            "resource_type": resource,
            "fetched_at": datetime.utcnow().isoformat(),
            "connector": self.connector_id,
            **raw_data,
        }
