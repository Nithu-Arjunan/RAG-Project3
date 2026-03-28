from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = Path(__file__).resolve().parent / "config.yaml"


# _load_config.
def _load_config() -> dict:
    if not CONFIG_PATH.exists():
        return {}
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


# _resolve_path.
def _resolve_path(value: str | None, default: str) -> str:
    if not value:
        return default
    path = Path(value)
    if path.is_absolute():
        return str(path)
    return str(ROOT / path)


# get_cache_config.
def get_cache_config() -> dict:
    cfg = _load_config()
    cache_cfg = cfg.get("cache", {}) if isinstance(cfg, dict) else {}
    backend = cache_cfg.get("backend", "sqlite")
    db_default = str(ROOT / "data" / "cache.sqlite")
    db_path = _resolve_path(cache_cfg.get("database_path"), db_default)
    return {
        "backend": backend,
        "database_path": db_path,
        "exact_ttl_seconds": int(cache_cfg.get("exact_ttl_seconds", 24 * 60 * 60)),
        "semantic_ttl_seconds": int(cache_cfg.get("semantic_ttl_seconds", 24 * 60 * 60)),
        "retrieval_ttl_seconds": int(cache_cfg.get("retrieval_ttl_seconds", 24 * 60 * 60)),
        "semantic_threshold": float(cache_cfg.get("semantic_threshold", 0.9)),
        "retrieval_threshold": float(cache_cfg.get("retrieval_threshold", 0.85)),
    }


# get_rag_config.
def get_rag_config() -> dict:
    cfg = _load_config()
    rag_cfg = cfg.get("rag", {}) if isinstance(cfg, dict) else {}
    return {
        "chunk_size": int(rag_cfg.get("chunk_size", 300)),
        "overlap": int(rag_cfg.get("overlap", 50)),
        "max_words": int(rag_cfg.get("max_words", 300)),
        "k_dense": int(rag_cfg.get("k_dense", 10)),
        "k_sparse": int(rag_cfg.get("k_sparse", 10)),
        "top_n": int(rag_cfg.get("top_n", 10)),
        "max_attempts": int(rag_cfg.get("max_attempts", 6)),
    }


# get_embedding_config.
def get_embedding_config() -> dict:
    cfg = _load_config()
    embed_cfg = cfg.get("embedding", {}) if isinstance(cfg, dict) else {}
    return {
        "model_name": embed_cfg.get(
            "model_name", "sentence-transformers/all-MiniLM-L6-v2"
        )
    }


# get_reranker_config.
def get_reranker_config() -> dict:
    cfg = _load_config()
    reranker_cfg = cfg.get("reranker", {}) if isinstance(cfg, dict) else {}
    return {
        "model_name": reranker_cfg.get(
            "model_name", "cross-encoder/ms-marco-MiniLM-L-6-v2"
        ),
    }
