"""
Cache package — 3-tier caching for the RAG pipeline.

Usage:
    from apps.cache import get_cache_backend
    cache = get_cache_backend()     # returns SQLite or Redis based on config
    cache.get_exact(query_hash)     # Tier 1
    cache.get_semantic(embedding)   # Tier 2
    cache.get_retrieval(embedding)  # Tier 3

"""


from config_loader import get_cache_config
from .base import CacheBackend

# Singleton instance — created on first call, reused after
_backend_instance: CacheBackend | None = None


# get_cache_backend.
def get_cache_backend() -> CacheBackend:
    """Get the configured cache backend (singleton).

    Returns SQLiteCacheBackend or RedisCacheBackend based on
    the cache.backend setting in config.yaml.
    """
    global _backend_instance

    if _backend_instance is not None:
        return _backend_instance

    cache_cfg = get_cache_config()
    if cache_cfg.get("backend") == "redis":
        from .redis_backend import RedisCacheBackend
        _backend_instance = RedisCacheBackend()
    else:
        from .sqlite_backend import SQLiteCacheBackend
        _backend_instance = SQLiteCacheBackend()

    return _backend_instance


# reset_cache_backend.
def reset_cache_backend() -> None:
    """Reset the singleton (useful for testing or backend switch)."""
    global _backend_instance
    _backend_instance = None
