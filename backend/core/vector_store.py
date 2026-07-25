"""
VectorStore — pgvector-backed embedding storage and similarity search.

Provides:
- OpenAI ada-002 embedding generation (1536 dimensions)
- Storage in PostgreSQL via pgvector
- Cosine similarity search
- Batch operations for efficiency

Usage:
    from core.vector_store import VectorStore

    store = VectorStore()

    # Store embeddings
    store.store_chunks(document_id, chunks)

    # Search similar
    results = store.search(query, top_k=5)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

logger = logging.getLogger(__name__)

EMBEDDING_MODEL = "text-embedding-ada-002"
EMBEDDING_DIMENSION = 1536


class VectorStore:
    """
    pgvector-backed vector store for document embeddings.

    Thread-safe: all state lives in PostgreSQL, not in the object.
    """

    def __init__(self):
        pass

    def embed(self, texts: List[str], user_id: Optional[str] = None, project_id: Optional[str] = None) -> List[List[float]]:
        """
        Generate embeddings for a list of texts using OpenAI ada-002, via
        core.ai_client so this resolves the caller's project/personal key
        (falling back to the platform default) and logs usage/cost - same
        as every other LLM call in the app. user_id falls back to the
        current request's g.user_id if not given explicitly.

        Args:
            texts: List of text strings to embed
            user_id: Whose key to bill this to (optional, see above)
            project_id: Which project's key takes priority (optional)

        Returns:
            List of embedding vectors (1536 dimensions each)
        """
        if not texts:
            return []

        from core.ai_client import ai_embeddings

        # OpenAI batch limit is 2048 texts
        batch_size = 100
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            try:
                response = ai_embeddings(
                    user_id=user_id, project_id=project_id, agent="document_intelligence.vector_store",
                    model=EMBEDDING_MODEL, input=batch,
                )
                batch_embeddings = [item.embedding for item in response.data]
                all_embeddings.extend(batch_embeddings)
            except Exception as e:
                logger.error("Embedding generation failed: %s", e)
                raise

        return all_embeddings

    def embed_single(self, text: str, user_id: Optional[str] = None, project_id: Optional[str] = None) -> List[float]:
        """Embed a single text string."""
        embeddings = self.embed([text], user_id=user_id, project_id=project_id)
        return embeddings[0] if embeddings else []

    def store_chunks(
        self,
        document_id: str,
        chunks: List[Dict[str, Any]],
        user_id: Optional[str] = None,
        project_id: Optional[str] = None,
        billed_user_id: Optional[str] = None,
    ) -> List[str]:
        """
        Store document chunks with embeddings in pgvector.

        Args:
            document_id: UUID of parent document
            chunks: List of dicts with 'content' and optional 'metadata'
            user_id: Optional access-control/filtering key stored on each
                chunk's metadata (existing callers pass a project id here,
                not necessarily a real platform user - kept as-is; NOT used
                for AI key billing, see project_id/billed_user_id below).
            project_id: The platform project this document belongs to, used
                to resolve that project's AI key for the embedding calls and
                to tag their usage/cost.
            billed_user_id: Who to attribute this embedding usage to. This
                often runs from a Celery task with no request context (so
                the usual g.user_id fallback doesn't apply) - pass the
                document's uploader explicitly when known.

        Returns:
            List of chunk IDs
        """
        from core.database import db
        from agents.document_intelligence.models import DocumentChunk

        if not chunks:
            return []

        # Extract text content
        texts = [c.get("content", "") for c in chunks]

        embeddings = self.embed(texts, user_id=billed_user_id, project_id=project_id)

        chunk_ids = []
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            chunk_id = str(uuid4())
            metadata = chunk.get("metadata", {})
            if user_id:
                metadata["user_id"] = user_id

            db_chunk = DocumentChunk(
                chunk_id=chunk_id,
                document_id=document_id,
                chunk_index=i,
                content=chunk.get("content", ""),
                embedding=embedding,
                metadata=metadata,
            )
            db.session.add(db_chunk)
            chunk_ids.append(chunk_id)

        try:
            db.session.commit()
            logger.info(
                "Stored %d chunks for document %s", len(chunk_ids), document_id
            )
        except Exception as e:
            db.session.rollback()
            logger.error("Failed to store chunks: %s", e)
            raise

        return chunk_ids

    def search(
        self,
        query: str,
        top_k: int = 5,
        document_ids: Optional[List[str]] = None,
        user_id: Optional[str] = None,
        threshold: float = 0.0,
    ) -> List[Dict[str, Any]]:
        """
        Search for similar chunks using cosine similarity.

        Args:
            query: Search query text
            top_k: Number of results to return
            document_ids: Optional filter by document IDs
            user_id: Optional filter by user ID (in metadata)
            threshold: Minimum similarity score (0-1)

        Returns:
            List of dicts with 'chunk_id', 'content', 'score', 'metadata'
        """
        from sqlalchemy import text
        from core.database import db

        # Generate query embedding
        query_embedding = self.embed_single(query, user_id=user_id)
        if not query_embedding:
            return []

        # Build query with pgvector cosine distance
        # Note: pgvector uses <=> for cosine distance (1 - similarity)
        # We convert to similarity score
        # Format embedding as literal in SQL to avoid binding issues
        embedding_str = "[" + ",".join(str(x) for x in query_embedding) + "]"

        # Build base SQL with embedding as literal (safer for pgvector compatibility)
        # Join with processed_documents to filter by user_id for security
        sql = f"""
            SELECT
                dc.chunk_id,
                dc.document_id,
                dc.chunk_index,
                dc.content,
                dc.metadata,
                1 - (dc.embedding <=> '{embedding_str}'::vector) as similarity
            FROM document_chunks dc
            JOIN processed_documents pd ON dc.document_id = pd.document_id
            WHERE 1=1
        """
        params: Dict[str, Any] = {}

        # Security: always filter by uploaded_by (column is named uploaded_by,
        # not user_id, on processed_documents)
        if user_id:
            sql += " AND pd.uploaded_by = :user_id"
            params["user_id"] = user_id

        if document_ids:
            sql += " AND dc.document_id = ANY(:doc_ids)"
            params["doc_ids"] = document_ids

        sql += f"""
            ORDER BY dc.embedding <=> '{embedding_str}'::vector
            LIMIT :limit
        """
        params["limit"] = top_k

        try:
            result = db.session.execute(text(sql), params)
            rows = result.fetchall()

            results = []
            for row in rows:
                similarity = float(row.similarity)
                if similarity >= threshold:
                    results.append({
                        "chunk_id": row.chunk_id,
                        "document_id": row.document_id,
                        "chunk_index": row.chunk_index,
                        "content": row.content,
                        "metadata": row.metadata or {},
                        "score": similarity,
                    })

            return results

        except Exception as e:
            logger.error("Vector search failed: %s", e)
            return []

    def delete_document_chunks(self, document_id: str) -> int:
        """
        Delete all chunks for a document.

        Returns:
            Number of chunks deleted
        """
        from core.database import db
        from agents.document_intelligence.models import DocumentChunk

        try:
            count = DocumentChunk.query.filter_by(document_id=document_id).delete()
            db.session.commit()
            logger.info("Deleted %d chunks for document %s", count, document_id)
            return count
        except Exception as e:
            db.session.rollback()
            logger.error("Failed to delete chunks: %s", e)
            raise

    def get_document_chunks(
        self, document_id: str, include_embeddings: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Get all chunks for a document.

        Args:
            document_id: Document UUID
            include_embeddings: Whether to include embedding vectors

        Returns:
            List of chunk dicts
        """
        from agents.document_intelligence.models import DocumentChunk

        chunks = (
            DocumentChunk.query.filter_by(document_id=document_id)
            .order_by(DocumentChunk.chunk_index)
            .all()
        )

        results = []
        for chunk in chunks:
            data = {
                "chunk_id": chunk.chunk_id,
                "document_id": chunk.document_id,
                "chunk_index": chunk.chunk_index,
                "content": chunk.content,
                "metadata": chunk.metadata_dict,
            }
            if include_embeddings:
                data["embedding"] = chunk.embedding
            results.append(data)

        return results
