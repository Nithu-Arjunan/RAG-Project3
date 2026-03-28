"""
SQLite cache backend.

Stores all 3 cache tiers in a local SQLite database.
Zero infrastructure — just a file on disk.
Good for local development and teaching.

Key design decisions:
- WAL mode for concurrent read access
- Embeddings stored as BLOB (numpy float32 bytes)
- Cosine similarity computed in Python via numpy
- All embeddings loaded into memory for fast comparison
"""

import os
import sqlite3
import time
from typing import Optional

from pathlib import Path

from config_loader import get_cache_config
from .base import CacheBackend
from .embeddding_cache import (
    cosine_similarity,
    embedding_to_bytes,
    bytes_to_embedding,
)


class SQLiteCacheBackend(CacheBackend):
    """SQLite-based implementation of the 3-tier cache."""

    # __init__.
    def __init__(self, db_path: str | None = None):
        cache_cfg = get_cache_config()
        self._db_path = db_path or cache_cfg.get("database_path")
        Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_tables()

    # _get_conn.
    def _get_conn(self) -> sqlite3.Connection:
        """Create a new connection (one per operation for thread safety)."""
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    # _init_tables.
    def _init_tables(self):
        """Create all cache tables if they don't exist."""
        conn = self._get_conn()
        conn.executescript("""
            -- Global metadata (doc_version counter, etc.)
            CREATE TABLE IF NOT EXISTS cache_metadata (
                key   TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );

            -- Initialize doc_version if not exists
            INSERT OR IGNORE INTO cache_metadata (key, value) VALUES ('doc_version', '0');

            -- Tier 1: Exact query cache
            CREATE TABLE IF NOT EXISTS exact_cache (
                query_hash      TEXT PRIMARY KEY,
                question_text   TEXT NOT NULL,
                answer_text     TEXT NOT NULL,
                sources_json    TEXT NOT NULL,
                doc_version     INTEGER NOT NULL,
                created_at      REAL NOT NULL,
                ttl_seconds     INTEGER NOT NULL,
                hit_count       INTEGER DEFAULT 0,
                last_hit_at     REAL
            );

            -- Tier 2: Semantic cache
            CREATE TABLE IF NOT EXISTS semantic_cache (
                id                  INTEGER PRIMARY KEY AUTOINCREMENT,
                question_text       TEXT NOT NULL,
                question_embedding  BLOB NOT NULL,
                answer_text         TEXT NOT NULL,
                sources_json        TEXT NOT NULL,
                source_filter       TEXT NOT NULL DEFAULT '',
                doc_version         INTEGER NOT NULL,
                created_at          REAL NOT NULL,
                ttl_seconds         INTEGER NOT NULL,
                hit_count           INTEGER DEFAULT 0,
                last_hit_at         REAL
            );

            -- Document hash deduplication
            CREATE TABLE IF NOT EXISTS document_hashes (
                file_hash   TEXT PRIMARY KEY,
                file_name   TEXT NOT NULL,
                file_size   INTEGER NOT NULL,
                chunk_count INTEGER NOT NULL,
                created_at  REAL NOT NULL
            );

            -- Tier 3: Retrieval cache
            CREATE TABLE IF NOT EXISTS retrieval_cache (
                id                  INTEGER PRIMARY KEY AUTOINCREMENT,
                question_text       TEXT NOT NULL,
                question_embedding  BLOB NOT NULL,
                chunks_json         TEXT NOT NULL,
                source_filter       TEXT NOT NULL DEFAULT '',
                doc_version         INTEGER NOT NULL,
                created_at          REAL NOT NULL,
                ttl_seconds         INTEGER NOT NULL,
                hit_count           INTEGER DEFAULT 0,
                last_hit_at         REAL
            );
        """)
        conn.commit()
        # Migrate: add source_filter column if missing (existing DBs)
        for table in ("semantic_cache", "retrieval_cache"):
            cols = [r[1] for r in conn.execute(f"PRAGMA table_info({table})").fetchall()]
            if "source_filter" not in cols:
                conn.execute(f"ALTER TABLE {table} ADD COLUMN source_filter TEXT NOT NULL DEFAULT ''")
                conn.commit()
        conn.close()

    # ── Tier 1: Exact Cache ────────────────────────────────────

    def get_exact(self, query_hash: str) -> Optional[dict]:
        conn = self._get_conn()
        now = time.time()
        doc_version = self._get_doc_version_raw(conn)

        row = conn.execute(
            """SELECT question_text, answer_text, sources_json, doc_version,
                      created_at, ttl_seconds
               FROM exact_cache WHERE query_hash = ?""",
            (query_hash,),
        ).fetchone()

        if row is None:
            conn.close()
            return None

        # Check TTL expiry
        if now - row["created_at"] > row["ttl_seconds"]:
            conn.execute("DELETE FROM exact_cache WHERE query_hash = ?", (query_hash,))
            conn.commit()
            conn.close()
            return None

        # Check doc version (stale if entry is from older version)
        if row["doc_version"] < doc_version:
            conn.execute("DELETE FROM exact_cache WHERE query_hash = ?", (query_hash,))
            conn.commit()
            conn.close()
            return None

        # Cache hit — update stats
        conn.execute(
            "UPDATE exact_cache SET hit_count = hit_count + 1, last_hit_at = ? WHERE query_hash = ?",
            (now, query_hash),
        )
        conn.commit()
        conn.close()

        return {
            "question": row["question_text"],
            "answer": row["answer_text"],
            "sources_json": row["sources_json"],
        }

    # set_exact.
    def set_exact(
        self,
        query_hash: str,
        question: str,
        answer: str,
        sources_json: str,
        doc_version: int,
        ttl_seconds: int,
    ) -> None:
        conn = self._get_conn()
        conn.execute(
            """INSERT OR REPLACE INTO exact_cache
               (query_hash, question_text, answer_text, sources_json,
                doc_version, created_at, ttl_seconds, hit_count, last_hit_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, 0, NULL)""",
            (query_hash, question, answer, sources_json, doc_version, time.time(), ttl_seconds),
        )
        conn.commit()
        conn.close()

    # ── Tier 2: Semantic Cache ─────────────────────────────────

    def get_semantic(
        self, embedding: list[float], threshold: float, source_filter: str = ""
    ) -> Optional[dict]:
        conn = self._get_conn()
        now = time.time()
        doc_version = self._get_doc_version_raw(conn)

        rows = conn.execute(
            "SELECT id, question_text, question_embedding, answer_text, sources_json, "
            "source_filter, doc_version, created_at, ttl_seconds FROM semantic_cache "
            "WHERE source_filter = ?",
            (source_filter,),
        ).fetchall()
        conn.close()

        best_match = None
        best_similarity = 0.0

        for row in rows:
            # Skip expired
            if now - row["created_at"] > row["ttl_seconds"]:
                continue
            # Skip stale doc version
            if row["doc_version"] < doc_version:
                continue

            cached_embedding = bytes_to_embedding(row["question_embedding"])
            sim = cosine_similarity(embedding, cached_embedding)

            if sim >= threshold and sim > best_similarity:
                best_similarity = sim
                best_match = row

        if best_match is None:
            return None

        # Update hit stats
        conn = self._get_conn()
        conn.execute(
            "UPDATE semantic_cache SET hit_count = hit_count + 1, last_hit_at = ? WHERE id = ?",
            (now, best_match["id"]),
        )
        conn.commit()
        conn.close()

        return {
            "question": best_match["question_text"],
            "answer": best_match["answer_text"],
            "sources_json": best_match["sources_json"],
            "similarity": best_similarity,
        }

    # set_semantic.
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
        conn = self._get_conn()
        conn.execute(
            """INSERT INTO semantic_cache
               (question_text, question_embedding, answer_text, sources_json,
                source_filter, doc_version, created_at, ttl_seconds, hit_count, last_hit_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0, NULL)""",
            (question, embedding_to_bytes(embedding), answer, sources_json,
             source_filter, doc_version, time.time(), ttl_seconds),
        )
        conn.commit()
        conn.close()

    # ── Tier 3: Retrieval Cache ────────────────────────────────

    def get_retrieval(
        self, embedding: list[float], threshold: float, source_filter: str = ""
    ) -> Optional[dict]:
        conn = self._get_conn()
        now = time.time()
        doc_version = self._get_doc_version_raw(conn)

        rows = conn.execute(
            "SELECT id, question_text, question_embedding, chunks_json, "
            "source_filter, doc_version, created_at, ttl_seconds FROM retrieval_cache "
            "WHERE source_filter = ?",
            (source_filter,),
        ).fetchall()
        conn.close()

        best_match = None
        best_similarity = 0.0

        for row in rows:
            if now - row["created_at"] > row["ttl_seconds"]:
                continue
            if row["doc_version"] < doc_version:
                continue

            cached_embedding = bytes_to_embedding(row["question_embedding"])
            sim = cosine_similarity(embedding, cached_embedding)

            if sim >= threshold and sim > best_similarity:
                best_similarity = sim
                best_match = row

        if best_match is None:
            return None

        # Update hit stats
        conn = self._get_conn()
        conn.execute(
            "UPDATE retrieval_cache SET hit_count = hit_count + 1, last_hit_at = ? WHERE id = ?",
            (now, best_match["id"]),
        )
        conn.commit()
        conn.close()

        return {
            "question": best_match["question_text"],
            "chunks_json": best_match["chunks_json"],
            "similarity": best_similarity,
        }

    # set_retrieval.
    def set_retrieval(
        self,
        question: str,
        embedding: list[float],
        chunks_json: str,
        doc_version: int,
        ttl_seconds: int,
        source_filter: str = "",
    ) -> None:
        conn = self._get_conn()
        conn.execute(
            """INSERT INTO retrieval_cache
               (question_text, question_embedding, chunks_json,
                source_filter, doc_version, created_at, ttl_seconds, hit_count, last_hit_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, 0, NULL)""",
            (question, embedding_to_bytes(embedding), chunks_json,
             source_filter, doc_version, time.time(), ttl_seconds),
        )
        conn.commit()
        conn.close()

    # ── Management ─────────────────────────────────────────────

    def _get_doc_version_raw(self, conn: sqlite3.Connection) -> int:
        """Internal: get doc_version using an existing connection."""
        row = conn.execute(
            "SELECT value FROM cache_metadata WHERE key = 'doc_version'"
        ).fetchone()
        return int(row["value"]) if row else 0

    # get_doc_version.
    def get_doc_version(self) -> int:
        conn = self._get_conn()
        version = self._get_doc_version_raw(conn)
        conn.close()
        return version

    # bump_doc_version.
    def bump_doc_version(self) -> int:
        conn = self._get_conn()
        current = self._get_doc_version_raw(conn)
        new_version = current + 1
        conn.execute(
            "UPDATE cache_metadata SET value = ? WHERE key = 'doc_version'",
            (str(new_version),),
        )
        conn.commit()
        conn.close()
        return new_version

    # clear_all.
    def clear_all(self) -> dict:
        conn = self._get_conn()
        exact_count = conn.execute("SELECT COUNT(*) FROM exact_cache").fetchone()[0]
        semantic_count = conn.execute("SELECT COUNT(*) FROM semantic_cache").fetchone()[0]
        retrieval_count = conn.execute("SELECT COUNT(*) FROM retrieval_cache").fetchone()[0]

        conn.execute("DELETE FROM exact_cache")
        conn.execute("DELETE FROM semantic_cache")
        conn.execute("DELETE FROM retrieval_cache")
        conn.commit()
        conn.close()

        return {
            "exact": exact_count,
            "semantic": semantic_count,
            "retrieval": retrieval_count,
        }

    # get_stats.
    def get_stats(self) -> dict:
        conn = self._get_conn()

        # _tier_stats.
        def _tier_stats(table: str) -> dict:
            row = conn.execute(
                f"SELECT COUNT(*) as cnt, COALESCE(SUM(hit_count), 0) as hits FROM {table}"
            ).fetchone()
            return {"entries": row["cnt"], "total_hits": row["hits"]}

        stats = {
            "backend": "sqlite",
            "db_path": self._db_path,
            "doc_version": self._get_doc_version_raw(conn),
            "exact": _tier_stats("exact_cache"),
            "semantic": _tier_stats("semantic_cache"),
            "retrieval": _tier_stats("retrieval_cache"),
        }
        conn.close()
        return stats

    # cleanup_expired.
    def cleanup_expired(self) -> int:
        conn = self._get_conn()
        now = time.time()
        total = 0

        for table in ("exact_cache", "semantic_cache", "retrieval_cache"):
            cursor = conn.execute(
                f"DELETE FROM {table} WHERE (? - created_at) > ttl_seconds",
                (now,),
            )
            total += cursor.rowcount

        conn.commit()
        conn.close()
        return total

    # ── Document Hash Deduplication ──────────────────────────────

    def get_document_hash(self, file_hash: str) -> Optional[dict]:
        conn = self._get_conn()
        row = conn.execute(
            "SELECT file_name, file_size, chunk_count, created_at "
            "FROM document_hashes WHERE file_hash = ?",
            (file_hash,),
        ).fetchone()
        conn.close()
        if row is None:
            return None
        return {
            "file_name": row["file_name"],
            "file_size": row["file_size"],
            "chunk_count": row["chunk_count"],
            "created_at": row["created_at"],
        }

    # set_document_hash.
    def set_document_hash(self, file_hash: str, metadata: dict) -> None:
        conn = self._get_conn()
        conn.execute(
            """INSERT OR REPLACE INTO document_hashes
               (file_hash, file_name, file_size, chunk_count, created_at)
               VALUES (?, ?, ?, ?, ?)""",
            (
                file_hash,
                metadata["file_name"],
                metadata["file_size"],
                metadata["chunk_count"],
                time.time(),
            ),
        )
        conn.commit()
        conn.close()

    # remove_document_hash_by_name.
    def remove_document_hash_by_name(self, file_name: str) -> bool:
        conn = self._get_conn()
        cursor = conn.execute(
            "DELETE FROM document_hashes WHERE file_name = ?", (file_name,)
        )
        conn.commit()
        removed = cursor.rowcount > 0
        conn.close()
        return removed

    # clear_document_hashes.
    def clear_document_hashes(self) -> int:
        conn = self._get_conn()
        count = conn.execute("SELECT COUNT(*) FROM document_hashes").fetchone()[0]
        conn.execute("DELETE FROM document_hashes")
        conn.commit()
        conn.close()
        return count