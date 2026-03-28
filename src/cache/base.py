"""
Abstract base class for cache backends.

Both SQLite and Redis backends implement this interface,
so the rest of the application doesn't care which one is active.
Swap via config.yaml (cache.backend: "sqlite" or "redis").
"""

from abc import ABC, abstractmethod
from typing import Optional



class CacheBackend(ABC):
    """Interface that all cache backends must implement."""

    # ── Tier 1: Exact Query Cache ──────────────────────────────
    # Hash of normalized query → full answer + sources
    # O(1) lookup, catches identical questions

    @abstractmethod
    def get_exact(self, query_hash: str) -> Optional[dict]:
        """Lookup by query hash. Returns {question, answer, sources} or None."""
        ...

    # set_exact.
    @abstractmethod
    def set_exact(
        self,
        query_hash: str,
        question: str,
        answer: str,
        sources_json: str,
        doc_version: int,
        ttl_seconds: int,
    ) -> None:
        """Store an exact cache entry."""
        ...

    # ── Tier 2: Semantic Cache ─────────────────────────────────
    # Embedding similarity → full answer + sources
    # O(N) cosine scan, catches rephrasings (cosine ≥ threshold)

    @abstractmethod
    def get_semantic(
        self, embedding: list[float], threshold: float, source_filter: str = ""
    ) -> Optional[dict]:
        """Find semantically similar cached question for the same source filter.
        Returns {question, answer, sources, similarity} or None."""
        ...

    # set_semantic.
    @abstractmethod
    def set_semantic(
        self,
        question: str,
        embedding: list[float],
        answer: str,
        sources_json: str,
        doc_version: int,
        ttl_seconds: int,
        source_filter: str = "",
    ) -> None:
        """Store a semantic cache entry with its embedding."""
        ...

    # ── Tier 3: Retrieval Cache ────────────────────────────────
    # Embedding similarity → cached retrieval chunks
    # Skips Pinecone call but still runs LLM generation

    @abstractmethod
    def get_retrieval(
        self, embedding: list[float], threshold: float, source_filter: str = ""
    ) -> Optional[dict]:
        """Find cached retrieval chunks for a similar query with same source filter.
        Returns {question, chunks} or None."""
        ...

    # set_retrieval.
    @abstractmethod
    def set_retrieval(
        self,
        question: str,
        embedding: list[float],
        chunks_json: str,
        doc_version: int,
        ttl_seconds: int,
        source_filter: str = "",
    ) -> None:
        """Store retrieval results (chunks) for a query."""
        ...

    # ── Management ─────────────────────────────────────────────

    @abstractmethod
    def get_doc_version(self) -> int:
        """Get the current document version counter."""
        ...

    # bump_doc_version.
    @abstractmethod
    def bump_doc_version(self) -> int:
        """Increment and return the new document version."""
        ...

    # clear_all.
    @abstractmethod
    def clear_all(self) -> dict:
        """Clear all 3 cache tiers. Returns {"exact": N, "semantic": N, "retrieval": N}."""
        ...

    # get_stats.
    @abstractmethod
    def get_stats(self) -> dict:
        """Return per-tier entry counts, hit counts, and backend info."""
        ...

    # cleanup_expired.
    @abstractmethod
    def cleanup_expired(self) -> int:
        """Remove expired entries (past TTL). Returns total removed count."""
        ...

    # ── Document Hash Deduplication ──────────────────────────────

    @abstractmethod
    def get_document_hash(self, file_hash: str) -> Optional[dict]:
        """Check if a file content hash was already uploaded.
        Returns {file_name, file_size, chunk_count, created_at} or None."""
        ...

    # set_document_hash.
    @abstractmethod
    def set_document_hash(self, file_hash: str, metadata: dict) -> None:
        """Store file content hash after successful upload."""
        ...

    # remove_document_hash_by_name.
    @abstractmethod
    def remove_document_hash_by_name(self, file_name: str) -> bool:
        """Remove a document hash entry by file name. Returns True if found and removed."""
        ...

    # clear_document_hashes.
    @abstractmethod
    def clear_document_hashes(self) -> int:
        """Clear all document hashes (used on vector reset). Returns count removed."""
        ...
