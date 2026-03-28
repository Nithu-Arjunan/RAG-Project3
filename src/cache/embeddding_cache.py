"""
Embedding utilities for the cache system.

- embed_query()        - generate embedding vector via HuggingFaceEmbeddings
- normalize_query()    - clean/normalize query text for exact matching
- hash_query()         - SHA256 hash of normalized text
- cosine_similarity()  - compare two embedding vectors
"""

import hashlib
import re

import numpy as np
from langchain_huggingface import HuggingFaceEmbeddings

from config_loader import get_embedding_config

_embeddings_client = None


# _get_embeddings_client.
def _get_embeddings_client():
    """Get or create the embeddings client (singleton)."""
    global _embeddings_client
    if _embeddings_client is None:
        embed_cfg = get_embedding_config()
        _embeddings_client = HuggingFaceEmbeddings(
            model_name=embed_cfg.get("model_name", "sentence-transformers/all-MiniLM-L6-v2")
        )
    return _embeddings_client


# embed_query.
def embed_query(text: str) -> list[float]:
    """Generate an embedding vector for a query string using HuggingFace embeddings."""
    embeddings = _get_embeddings_client()
    return embeddings.embed_query(text)


# normalize_query.
def normalize_query(text: str) -> str:
    """Normalize a query for exact-match caching.

    Steps:
    1. Lowercase
    2. Strip leading/trailing whitespace
    3. Collapse multiple spaces into one
    4. Remove trailing punctuation (?, !, .)

    Example: "  What was  Apple's Revenue?? " -> "what was apple's revenue"
    """
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[?.!]+$", "", text).strip()
    return text


# hash_query.
def hash_query(normalized_text: str) -> str:
    """SHA256 hash of a normalized query string.

    Used as the key for Tier 1 (exact cache) lookups.
    Deterministic: same input always produces same hash.
    """
    return hashlib.sha256(normalized_text.encode("utf-8")).hexdigest()


# cosine_similarity.
def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """Compute cosine similarity between two vectors.

    Returns a float in [-1, 1] where:
    - 1.0  = identical direction (same meaning)
    - 0.0  = orthogonal (unrelated)
    - -1.0 = opposite direction

    Uses numpy for fast computation.
    """
    a = np.array(vec_a, dtype=np.float32)
    b = np.array(vec_b, dtype=np.float32)

    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return float(dot_product / (norm_a * norm_b))


# embedding_to_bytes.
def embedding_to_bytes(embedding: list[float]) -> bytes:
    """Serialize embedding list to bytes for storage (SQLite BLOB / Redis)."""
    return np.array(embedding, dtype=np.float32).tobytes()


# bytes_to_embedding.
def bytes_to_embedding(data: bytes) -> list[float]:
    """Deserialize bytes back to embedding list."""
    return np.frombuffer(data, dtype=np.float32).tolist()
