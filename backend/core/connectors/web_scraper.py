"""
Web Scraper Connector — unified web scraping with context storage.

Replaces the scattered scraping functions in app.py with a proper connector.

Usage:
    from core.connectors import get_connector

    scraper = get_connector("web_scraper", user_id="user123")
    scraper.connect()

    # Scrape a page
    data = scraper.fetch("page", params={"url": "https://example.com"})

    # Scrape with specific extraction type
    data = scraper.fetch("text", params={"url": "https://example.com"})
    data = scraper.fetch("tables", params={"url": "https://example.com"})
    data = scraper.fetch("product", params={"url": "https://example.com"})
    data = scraper.fetch("json_ld", params={"url": "https://example.com"})
"""

from __future__ import annotations

import logging
import re
from datetime import datetime
from typing import Any, Dict, List, Optional
from urllib.parse import urljoin, urlparse

from core.connectors.base import SessionConnector, ConnectorError
from core.connectors.registry import register_connector

logger = logging.getLogger(__name__)


@register_connector
class WebScraperConnector(SessionConnector):
    """
    Web scraping connector with multiple extraction modes.

    Supported resources:
    - page: Full HTML content
    - text: Text-only extraction
    - tables: Extract HTML tables as structured data
    - product: E-commerce product info
    - json_ld: Schema.org JSON-LD data
    - links: Extract all links
    - custom: Custom CSS selector extraction
    """

    connector_id = "web_scraper"
    connector_name = "Web Scraper"
    supported_resources = ["page", "text", "tables", "product", "json_ld", "links", "custom"]

    def fetch(
        self,
        resource: str,
        params: Optional[Dict[str, Any]] = None,
        store: bool = True,
    ) -> Dict[str, Any]:
        """
        Fetch and extract data from a URL.

        Args:
            resource: Extraction type (page, text, tables, product, json_ld, links, custom)
            params: Must include 'url', may include 'selectors' for custom
            store: Whether to store to context

        Returns:
            Extracted data
        """
        params = params or {}
        url = params.get("url")

        if not url:
            raise ConnectorError("URL is required")

        # Validate URL
        parsed = urlparse(url)
        if not parsed.scheme or not parsed.netloc:
            raise ConnectorError(f"Invalid URL: {url}")

        try:
            # Fetch page
            response = self.session.get(url, timeout=30)
            response.raise_for_status()

            # Parse with BeautifulSoup
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(response.text, "html.parser")

            # Extract based on resource type
            if resource == "page":
                raw_data = self._extract_page(soup, url, response)
            elif resource == "text":
                raw_data = self._extract_text(soup, url)
            elif resource == "tables":
                raw_data = self._extract_tables(soup, url)
            elif resource == "product":
                raw_data = self._extract_product(soup, url)
            elif resource == "json_ld":
                raw_data = self._extract_json_ld(soup, url)
            elif resource == "links":
                raw_data = self._extract_links(soup, url)
            elif resource == "custom":
                selectors = params.get("selectors", {})
                raw_data = self._extract_custom(soup, url, selectors)
            else:
                raise ConnectorError(f"Unknown resource type: {resource}")

            # Transform to standard format
            data = self._transform(raw_data, resource)

            # Store to context
            if store:
                self.store_to_context(data, resource, metadata={"url": url})

            return data

        except Exception as e:
            logger.error(f"Scrape failed for {url}: {e}")
            raise ConnectorError(f"Scrape failed: {e}")

    def _transform(self, raw_data: Any, resource: str) -> Dict[str, Any]:
        """Transform raw data to standard format."""
        return {
            "resource_type": resource,
            "scraped_at": datetime.utcnow().isoformat(),
            **raw_data,
        }

    def _extract_page(self, soup, url: str, response) -> Dict[str, Any]:
        """Extract full page content."""
        return {
            "url": url,
            "title": soup.title.string if soup.title else None,
            "html": str(soup),
            "status_code": response.status_code,
            "content_type": response.headers.get("Content-Type"),
        }

    def _extract_text(self, soup, url: str) -> Dict[str, Any]:
        """Extract text content only."""
        # Remove script and style elements
        for element in soup(["script", "style", "nav", "footer", "header"]):
            element.decompose()

        # Get text
        text = soup.get_text(separator="\n", strip=True)

        # Clean up whitespace
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        clean_text = "\n".join(lines)

        return {
            "url": url,
            "title": soup.title.string if soup.title else None,
            "text": clean_text,
            "word_count": len(clean_text.split()),
        }

    def _extract_tables(self, soup, url: str) -> Dict[str, Any]:
        """Extract HTML tables as structured data."""
        tables = []

        for i, table in enumerate(soup.find_all("table")):
            table_data = {
                "index": i,
                "headers": [],
                "rows": [],
            }

            # Extract headers
            header_row = table.find("thead")
            if header_row:
                headers = header_row.find_all(["th", "td"])
                table_data["headers"] = [h.get_text(strip=True) for h in headers]
            else:
                # Try first row as header
                first_row = table.find("tr")
                if first_row:
                    ths = first_row.find_all("th")
                    if ths:
                        table_data["headers"] = [h.get_text(strip=True) for h in ths]

            # Extract rows
            for row in table.find_all("tr"):
                cells = row.find_all(["td", "th"])
                if cells:
                    row_data = [cell.get_text(strip=True) for cell in cells]
                    # Skip if it's the header row
                    if row_data != table_data["headers"]:
                        table_data["rows"].append(row_data)

            if table_data["rows"]:
                tables.append(table_data)

        return {
            "url": url,
            "table_count": len(tables),
            "tables": tables,
        }

    def _extract_product(self, soup, url: str) -> Dict[str, Any]:
        """Extract e-commerce product information."""
        product = {
            "url": url,
            "name": None,
            "price": None,
            "currency": None,
            "description": None,
            "images": [],
            "availability": None,
            "brand": None,
            "sku": None,
        }

        # Try JSON-LD first
        json_ld = self._extract_json_ld(soup, url)
        for item in json_ld.get("items", []):
            if item.get("@type") == "Product":
                product["name"] = item.get("name")
                product["description"] = item.get("description")
                product["brand"] = item.get("brand", {}).get("name")
                product["sku"] = item.get("sku")

                offers = item.get("offers", {})
                if isinstance(offers, list):
                    offers = offers[0] if offers else {}
                product["price"] = offers.get("price")
                product["currency"] = offers.get("priceCurrency")
                product["availability"] = offers.get("availability")

                images = item.get("image", [])
                if isinstance(images, str):
                    images = [images]
                product["images"] = images

        # Fallback to common selectors if JSON-LD didn't work
        if not product["name"]:
            # Common product name selectors
            for selector in ["h1", ".product-title", ".product-name", "[itemprop='name']"]:
                elem = soup.select_one(selector)
                if elem:
                    product["name"] = elem.get_text(strip=True)
                    break

        if not product["price"]:
            # Common price selectors
            for selector in [".price", ".product-price", "[itemprop='price']", ".amount"]:
                elem = soup.select_one(selector)
                if elem:
                    price_text = elem.get_text(strip=True)
                    # Extract numeric price
                    match = re.search(r"[\d,.]+", price_text)
                    if match:
                        product["price"] = match.group()
                    break

        return product

    def _extract_json_ld(self, soup, url: str) -> Dict[str, Any]:
        """Extract Schema.org JSON-LD data."""
        import json

        items = []

        for script in soup.find_all("script", type="application/ld+json"):
            try:
                data = json.loads(script.string)
                if isinstance(data, list):
                    items.extend(data)
                else:
                    items.append(data)
            except (json.JSONDecodeError, TypeError):
                continue

        return {
            "url": url,
            "item_count": len(items),
            "items": items,
        }

    def _extract_links(self, soup, url: str) -> Dict[str, Any]:
        """Extract all links from page."""
        links = []
        seen = set()

        for a in soup.find_all("a", href=True):
            href = a["href"]

            # Make absolute URL
            absolute_url = urljoin(url, href)

            # Skip duplicates, anchors, javascript
            if absolute_url in seen:
                continue
            if href.startswith("#") or href.startswith("javascript:"):
                continue

            seen.add(absolute_url)
            links.append({
                "url": absolute_url,
                "text": a.get_text(strip=True)[:100],  # Limit text length
                "is_external": urlparse(absolute_url).netloc != urlparse(url).netloc,
            })

        return {
            "url": url,
            "link_count": len(links),
            "links": links,
        }

    def _extract_custom(
        self,
        soup,
        url: str,
        selectors: Dict[str, str],
    ) -> Dict[str, Any]:
        """
        Extract using custom CSS selectors.

        Args:
            selectors: Dict of {field_name: css_selector}
        """
        results = {"url": url, "fields": {}}

        for field, selector in selectors.items():
            elements = soup.select(selector)
            if len(elements) == 1:
                results["fields"][field] = elements[0].get_text(strip=True)
            elif len(elements) > 1:
                results["fields"][field] = [e.get_text(strip=True) for e in elements]
            else:
                results["fields"][field] = None

        return results
