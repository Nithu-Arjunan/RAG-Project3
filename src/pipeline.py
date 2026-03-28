import logging
from typing import Any, Dict, List

from chunking import DoclingParser, create_parent_child_chunks
from embedding import build_vectorstore, build_bm25_index, get_reranker, hybrid_retrieve, rerank
from config_loader import get_rag_config

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO)


# infer_doc_name.
def infer_doc_name(doc_json: Dict[str, Any], fallback_name: str = "document") -> str:
    """Infer a stable document name from Docling metadata or fallback."""
    return (
        doc_json.get("origin", {}).get("filename")
        or doc_json.get("name")
        or fallback_name
    )


# build_chunks.
def build_chunks(doc_json: Dict[str, Any], doc_name: str) -> List[Dict[str, Any]]:
    """Parse Docling JSON and generate parent-child chunks for embedding."""
    parser = DoclingParser(doc_json)
    sections = parser.build_sections()
    return create_parent_child_chunks(sections, doc_name=doc_name)


# run_hybrid_rerank_from_json.
def run_hybrid_rerank_from_json(
    doc_json: Dict[str, Any],
    doc_name: str | None,
    query: str,
    k_dense: int | None = None,
    k_sparse: int | None = None,
    top_n: int | None = None,
    reranker=None,
) -> List[Dict[str, Any]]:
    """
    Chunk -> embed -> hybrid retrieve -> (optional) rerank and return structured results.

    Returns a list of dicts:
      {"text": ..., "metadata": ..., "hybrid_score": ...}
    """
    if not doc_name:
        doc_name = infer_doc_name(doc_json)

    logger.info("Stage: Chunking started.")
    chunks = build_chunks(doc_json, doc_name)
    logger.info("Stage: Chunking completed. Chunks created: %d", len(chunks))

    logger.info("Stage: Embedding started.")
    vectorstore = build_vectorstore(chunks)
    logger.info("Stage: Embedding completed.")

    logger.info("Stage: Hybrid retrieval started.")
    from embedding import _normalize_chunks
    bm25_index = build_bm25_index(_normalize_chunks(chunks))
    rag_cfg = get_rag_config()
    if k_dense is None:
        k_dense = rag_cfg.get("k_dense", 10)
    if k_sparse is None:
        k_sparse = rag_cfg.get("k_sparse", 10)
    if top_n is None:
        top_n = rag_cfg.get("top_n", 10)

    hybrid_results = hybrid_retrieve(
        query=query,
        vectorstore=vectorstore,
        bm25_index=bm25_index,
        k_dense=k_dense,
        k_sparse=k_sparse,
        top_n=top_n,
        metadata_filter=None,
    )
    logger.info("Stage: Hybrid retrieval completed. Results: %d", len(hybrid_results))

    if reranker is None:
        reranker = get_reranker()

    docs_only = [doc for doc, _ in hybrid_results]
    if reranker is not None and docs_only:
        docs_only = rerank(query, docs_only, reranker)

    score_map = {id(doc): score for doc, score in hybrid_results}
    structured: List[Dict[str, Any]] = []
    for doc in docs_only:
        structured.append(
            {
                "text": doc.page_content,
                "metadata": doc.metadata,
                "hybrid_score": score_map.get(id(doc)),
            }
        )

    return structured


if __name__ == "__main__":
    import os
    import json
    import sys

    if len(sys.argv) > 1:
        json_path = os.path.abspath(sys.argv[1])
    else:
        json_path = os.path.abspath("doc_output.json")

    if not os.path.exists(json_path):
        logger.error("JSON file not found: %s", json_path)
        raise SystemExit(1)

    with open(json_path, "r", encoding="utf-8") as f:
        doc_json = json.load(f)

    query = sys.argv[2] if len(sys.argv) > 2 else "What is Self-RAG?"

    results = run_hybrid_rerank_from_json(
        doc_json=doc_json,
        doc_name=None,
        query=query,
        k_dense=10,
        k_sparse=10,
        top_n=10,
        reranker=None,
    )

    logger.info("Demo completed. Results: %d", len(results))
    for i, item in enumerate(results, start=1):
        logger.info("Result %d | Hybrid Score: %s", i, item.get("hybrid_score"))
        logger.info("Content: %s", item.get("text", "")[:400])
        logger.info("Metadata: %s", item.get("metadata"))
