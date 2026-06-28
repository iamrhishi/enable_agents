"""
Cross-Agent Data Sharing Tests

Verifies that agents can share data via ContextStore.
Example: Document Intelligence uploads docs → Content Marketing reads them.

Run with:
    cd backend
    python -m pytest tests/test_cross_agent_sharing.py -v
"""

import pytest
from datetime import datetime
from unittest.mock import patch, MagicMock


class TestContextStoreSharing:
    """Test ContextStore enables cross-agent data sharing."""

    def test_document_intelligence_stores_data(self):
        """Document Intelligence agent stores document metadata to context."""
        from core.context import ContextStore

        ctx = ContextStore()
        user_id = "test_user_sharing"
        agent_id = "document_intelligence"

        # Simulate what document_intelligence does after processing
        doc_metadata = {
            "document_id": "doc123",
            "file_name": "product_specs.pdf",
            "document_type": "product",
            "word_count": 1500,
            "chunk_count": 5,
            "status": "completed",
            "preview": "This document describes our product features...",
        }

        ctx.set(user_id, agent_id, "document:doc123", doc_metadata)

        # Update index
        ctx.set(user_id, agent_id, "documents:index", [
            {"document_id": "doc123", "file_name": "product_specs.pdf"}
        ])

        # Verify data was stored
        stored = ctx.get(user_id, agent_id, "document:doc123")
        assert stored is not None
        assert stored["document_id"] == "doc123"
        assert stored["file_name"] == "product_specs.pdf"

    def test_content_marketing_reads_document_data(self):
        """Content Marketing agent reads document data from Document Intelligence."""
        from core.context import ContextStore

        ctx = ContextStore()
        user_id = "test_user_sharing"

        # First, simulate document_intelligence storing data
        doc_agent = "document_intelligence"
        ctx.set(user_id, doc_agent, "documents:index", [
            {"document_id": "doc456", "file_name": "brand_guidelines.pdf"},
            {"document_id": "doc789", "file_name": "competitor_analysis.pdf"},
        ])
        ctx.set(user_id, doc_agent, "document:doc456", {
            "document_id": "doc456",
            "file_name": "brand_guidelines.pdf",
            "preview": "Our brand values include...",
        })

        # Now simulate content_marketing reading it
        # The ask() interface provides a clean way to do this
        docs = ctx.ask(user_id, "documents")
        assert len(docs) >= 2

        # Content Marketing can also get specific document
        doc_context = ctx.get(user_id, doc_agent, "document:doc456")
        assert doc_context is not None
        assert "brand" in doc_context["file_name"].lower()

    def test_ask_interface_abstracts_agent_details(self):
        """The ask() interface hides which agent provides the data."""
        from core.context import ContextStore

        ctx = ContextStore()
        user_id = "test_user_ask"

        # Store some test data
        ctx.set(user_id, "document_intelligence", "documents:index", [
            {"document_id": "test1", "file_name": "test.pdf"}
        ])
        ctx.set(user_id, "external", "web_search:index", [
            {"query": "competitors", "results": 5}
        ])

        # Agents can ask for data without knowing where it comes from
        docs = ctx.ask(user_id, "documents")
        assert isinstance(docs, list)

        external = ctx.ask(user_id, "external")
        assert isinstance(external, dict)

        all_data = ctx.ask(user_id, "all")
        assert "documents" in all_data

    def test_connector_data_accessible_via_context(self):
        """Data fetched by connectors is accessible via ContextStore."""
        from core.context import ContextStore

        ctx = ContextStore()
        user_id = "test_user_connector"

        # Simulate connector storing data (what store_from_source does)
        web_search_result = {
            "query": "market research competitors",
            "results": [
                {"title": "Competitor A Analysis", "url": "https://example.com/a"},
                {"title": "Competitor B Review", "url": "https://example.com/b"},
            ],
        }

        # Connectors store via store_from_source
        key = ctx.store_from_source(
            user_id=user_id,
            source="web_search",
            data=web_search_result,
            metadata={"query": "market research competitors"}
        )

        assert key is not None

        # Other agents can access via ask("external")
        external = ctx.ask(user_id, "external")
        assert external is not None

    def test_market_research_shares_with_content_marketing(self):
        """Market Research shares company data with Content Marketing."""
        from core.context import ContextStore

        ctx = ContextStore()
        user_id = "test_user_mr_cm"

        # Market Research stores competitor analysis
        mr_data = {
            "company": "ACME Corp",
            "competitors": ["CompX", "CompY", "CompZ"],
            "market_position": "Leader in widget manufacturing",
            "swot": {
                "strengths": ["Brand recognition", "Quality"],
                "weaknesses": ["High prices"],
            },
        }
        ctx.set(user_id, "market_research", "company:acme", mr_data)

        # Content Marketing reads it to create content
        company_data = ctx.get(user_id, "market_research", "company:acme")
        assert company_data is not None
        assert company_data["company"] == "ACME Corp"
        assert "strengths" in company_data["swot"]

        # Content Marketing can use this to generate targeted content
        assert "Brand recognition" in company_data["swot"]["strengths"]


class TestCrossAgentHelpers:
    """Test cross-agent helper functions from document_intelligence."""

    def test_get_user_documents_helper(self):
        """Test get_user_documents() helper from document_intelligence."""
        from core.context import ContextStore

        ctx = ContextStore()
        user_id = "test_helper_docs"

        # Setup: document_intelligence has stored docs
        ctx.set(user_id, "document_intelligence", "documents:index", [
            {"document_id": "h1", "file_name": "helper_test.pdf"}
        ])

        # Import and use the helper
        from agents.document_intelligence.service import get_user_documents
        docs = get_user_documents(user_id)

        assert isinstance(docs, list)
        assert len(docs) >= 1
        assert docs[0]["document_id"] == "h1"

    def test_get_document_context_helper(self):
        """Test get_document_context() helper from document_intelligence."""
        from core.context import ContextStore

        ctx = ContextStore()
        user_id = "test_helper_ctx"
        doc_id = "ctx_test_doc"

        # Setup
        ctx.set(user_id, "document_intelligence", f"document:{doc_id}", {
            "document_id": doc_id,
            "file_name": "context_test.pdf",
            "preview": "Test content preview",
        })

        # Use helper
        from agents.document_intelligence.service import get_document_context
        doc_ctx = get_document_context(user_id, doc_id)

        assert doc_ctx is not None
        assert doc_ctx["document_id"] == doc_id
        assert "preview" in doc_ctx


# Run directly for quick test
if __name__ == "__main__":
    import sys

    # Quick manual test
    print("Running cross-agent sharing tests...")

    from core.context import ContextStore
    ctx = ContextStore()

    user = "manual_test_user"

    # Test 1: Store from document_intelligence
    ctx.set(user, "document_intelligence", "documents:index", [
        {"document_id": "manual1", "file_name": "test.pdf"}
    ])
    print("✓ document_intelligence stored data")

    # Test 2: Read from another "agent"
    docs = ctx.ask(user, "documents")
    assert len(docs) >= 1, "Failed to read documents"
    print(f"✓ Read {len(docs)} documents via ask()")

    # Test 3: Cross-agent access
    doc_data = ctx.get(user, "document_intelligence", "documents:index")
    assert doc_data is not None
    print("✓ Cross-agent data access works")

    print("\n✅ All manual tests passed!")
