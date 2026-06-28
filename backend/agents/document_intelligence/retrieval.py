"""
Document Intelligence Agent — Retrieval utilities.

Provides:
- pgvector similarity search
- Optional NetworkX graph entity boost
- Context assembly for LLM
"""

import logging
from typing import Any, Dict, List, Optional

from core.vector_store import VectorStore

logger = logging.getLogger(__name__)


class DocumentRetriever:
    """Retrieval service combining vector search with optional entity boost."""

    def __init__(self):
        self.vector_store = VectorStore()

    def search(
        self,
        query: str,
        user_id: str,
        document_ids: Optional[List[str]] = None,
        top_k: int = 5,
        min_score: float = 0.0,
    ) -> List[Dict[str, Any]]:
        """
        Search for relevant document chunks.

        Args:
            query: Search query text
            user_id: User ID for access control
            document_ids: Optional filter to specific documents
            top_k: Number of results to return
            min_score: Minimum similarity score (0-1)

        Returns:
            List of relevant chunks with scores
        """
        # Perform vector search
        results = self.vector_store.search(
            query=query,
            top_k=top_k,
            document_ids=document_ids,
            user_id=user_id,
            threshold=min_score,
        )

        return results

    def get_context_for_query(
        self,
        query: str,
        user_id: str,
        document_ids: Optional[List[str]] = None,
        max_chunks: int = 5,
        max_tokens: int = 3000,
    ) -> Dict[str, Any]:
        """
        Get assembled context for LLM query.

        Args:
            query: User query
            user_id: User ID
            document_ids: Optional document filter
            max_chunks: Maximum number of chunks to include
            max_tokens: Approximate token limit (~4 chars/token)

        Returns:
            Dict with 'context', 'sources', 'chunk_count'
        """
        # Search for relevant chunks
        # Use lower threshold (0.2) to capture more potentially relevant content
        results = self.search(
            query=query,
            user_id=user_id,
            document_ids=document_ids,
            top_k=max_chunks * 2,  # Get extra for filtering
            min_score=0.2,
        )

        if not results:
            return {
                "context": "",
                "sources": [],
                "chunk_count": 0,
            }

        # Assemble context within token limit
        context_parts = []
        sources = []
        total_chars = 0
        max_chars = max_tokens * 4

        for chunk in results[:max_chunks]:
            content = chunk.get("content", "")

            # Check token limit
            if total_chars + len(content) > max_chars:
                # Truncate this chunk
                remaining = max_chars - total_chars
                if remaining > 200:
                    content = content[:remaining] + "..."
                else:
                    break

            context_parts.append(content)
            total_chars += len(content)

            sources.append({
                "chunk_id": chunk.get("chunk_id"),
                "document_id": chunk.get("document_id"),
                "chunk_index": chunk.get("chunk_index"),
                "score": chunk.get("score"),
            })

        return {
            "context": "\n\n---\n\n".join(context_parts),
            "sources": sources,
            "chunk_count": len(sources),
        }

    def get_entity_enhanced_context(
        self,
        query: str,
        user_id: str,
        document_ids: Optional[List[str]] = None,
        max_chunks: int = 5,
    ) -> Dict[str, Any]:
        """
        Get context with entity-enhanced retrieval.

        First extracts key entities from query, then boosts chunks
        containing those entities.

        Args:
            query: User query
            user_id: User ID
            document_ids: Optional document filter
            max_chunks: Maximum chunks to return

        Returns:
            Dict with context and sources
        """
        from agents.document_intelligence.models import DocumentEntity

        # Get base vector search results
        base_results = self.search(
            query=query,
            user_id=user_id,
            document_ids=document_ids,
            top_k=max_chunks * 3,
            min_score=0.2,
        )

        if not base_results:
            return self.get_context_for_query(
                query=query,
                user_id=user_id,
                document_ids=document_ids,
                max_chunks=max_chunks,
            )

        # Extract query terms for entity matching
        query_terms = set(query.lower().split())

        # Find matching entities
        chunk_entity_counts: Dict[str, int] = {}

        for result in base_results:
            chunk_id = result.get("chunk_id")
            doc_id = result.get("document_id")

            # Find entities in this document that match query terms
            entities = DocumentEntity.query.filter_by(document_id=doc_id).all()

            for entity in entities:
                entity_terms = set(entity.entity_name.lower().split())
                if query_terms & entity_terms:
                    # Entity matches query, boost chunks containing it
                    for cid in entity.chunk_ids:
                        chunk_entity_counts[cid] = chunk_entity_counts.get(cid, 0) + 1

        # Re-score results with entity boost
        for result in base_results:
            chunk_id = result.get("chunk_id")
            entity_boost = chunk_entity_counts.get(chunk_id, 0) * 0.1
            result["score"] = result.get("score", 0) + entity_boost

        # Sort by boosted score
        base_results.sort(key=lambda x: x.get("score", 0), reverse=True)

        # Assemble context
        context_parts = []
        sources = []

        for chunk in base_results[:max_chunks]:
            context_parts.append(chunk.get("content", ""))
            sources.append({
                "chunk_id": chunk.get("chunk_id"),
                "document_id": chunk.get("document_id"),
                "chunk_index": chunk.get("chunk_index"),
                "score": chunk.get("score"),
            })

        return {
            "context": "\n\n---\n\n".join(context_parts),
            "sources": sources,
            "chunk_count": len(sources),
        }
