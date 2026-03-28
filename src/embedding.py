from __future__ import annotations

import os
import json
import logging
import math
import re
from pathlib import Path
from functools import lru_cache
from typing import Any, Dict, Iterable, List, Optional, Tuple

from langchain_core.documents import Document
#from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Qdrant
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, Filter, FieldCondition, MatchValue, VectorParams
from sentence_transformers import CrossEncoder
from chunking import DoclingParser, create_parent_child_chunks
from config_loader import get_embedding_config, get_rag_config, get_reranker_config

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO)

# get_reranker.
@lru_cache(maxsize=1)
def get_reranker() -> Optional[CrossEncoder]:
    cfg = get_reranker_config()
    model_name = cfg.get("model_name", "cross-encoder/ms-marco-MiniLM-L-6-v2")
    return CrossEncoder(model_name)



### Normalize the document so that it is converted to a list of LangChain Documents with consistent metadata structure.
def _normalize_chunks(chunks: Iterable[Any]) -> List[Document]:
    """
    Accepts chunks from chunking.py and returns LangChain Documents.
    Supported inputs:
    - List[Dict] with keys: "text" and optional "metadata"
    - List[str]
    """
    docs: List[Document] = []

    for chunk in chunks:
        if isinstance(chunk, str):
            docs.append(Document(page_content=chunk, metadata={}))
            continue

        if isinstance(chunk, dict):
            text = chunk.get("text", "")
            metadata = chunk.get("metadata", {})
            docs.append(Document(page_content=text, metadata=metadata))
            continue

        raise TypeError(
            f"Unsupported chunk type: {type(chunk)}. Expected str or dict with 'text'."
        )

    return docs


# _tokenize.
def _tokenize(text: str) -> List[str]:
    """Lowercase alphanumeric tokenizer for BM25 (simple and deterministic)."""
    return re.findall(r"[a-z0-9]+", text.lower())


# _doc_key.
def _doc_key(doc: Document) -> Tuple[str, Tuple[Tuple[str, Any], ...]]:
    """Stable key for document identity in fusion (content + sorted metadata)."""
    meta = doc.metadata or {}
    return (doc.page_content, tuple(sorted(meta.items())))


class BM25Index:
    """Minimal BM25 index for sparse scoring over in-memory Documents."""
    # __init__.
    def __init__(self, docs: List[Document], k1: float = 1.5, b: float = 0.75):
        self.docs = docs
        self.k1 = k1
        self.b = b
        self.doc_tokens = [_tokenize(d.page_content) for d in docs]
        self.doc_len = [len(toks) for toks in self.doc_tokens]
        self.avgdl = (sum(self.doc_len) / len(self.doc_len)) if self.doc_len else 0.0
        self.df: Dict[str, int] = {}
        for toks in self.doc_tokens:
            seen = set(toks)
            for term in seen:
                self.df[term] = self.df.get(term, 0) + 1

    # _idf.
    def _idf(self, term: str) -> float:
        """Inverse document frequency with BM25 smoothing."""
        n = len(self.docs)
        df = self.df.get(term, 0)
        return math.log(1 + (n - df + 0.5) / (df + 0.5)) if n else 0.0

    # scores.
    def scores(self, query: str) -> List[float]:
        """Return BM25 scores for each document in the index."""
        q_terms = _tokenize(query)
        scores: List[float] = []
        for toks, dl in zip(self.doc_tokens, self.doc_len):
            if dl == 0:
                scores.append(0.0)
                continue
            freq: Dict[str, int] = {}
            for t in toks:
                freq[t] = freq.get(t, 0) + 1
            score = 0.0
            for term in q_terms:
                if term not in freq:
                    continue
                idf = self._idf(term)
                tf = freq[term]
                denom = tf + self.k1 * (1 - self.b + self.b * (dl / self.avgdl))
                score += idf * ((tf * (self.k1 + 1)) / denom)
            scores.append(score)
        return scores

# Helper to build a Qdrant filter from metadata dict (for filtered retrieval)
def _build_qdrant_filter(metadata: Optional[Dict[str, Any]] = None) -> Optional[Filter]:
    """Convert a metadata dict into a Qdrant filter (exact match per field)."""
    if not metadata:
        return None

    conditions = [
        FieldCondition(key=key, match=MatchValue(value=value))
        for key, value in metadata.items()
    ]
    return Filter(must=conditions)

# Main embedding and retrieval functions
def build_vectorstore(
    chunks: Iterable[Any],
    collection_name: str = "rag_demo",
    qdrant_path: str = "",
    embedding_model: Optional[HuggingFaceEmbeddings] = None,
    client: Optional[QdrantClient] = None,
) -> Qdrant:
    """
    Build a Qdrant vectorstore from chunks produced by chunking.py.
    """
    docs = _normalize_chunks(chunks)

    if embedding_model is None:
        embed_cfg = get_embedding_config()
        embedding_model = HuggingFaceEmbeddings(
            model_name=embed_cfg.get("model_name", "sentence-transformers/all-MiniLM-L6-v2")
        )

    if client is None:
        if not qdrant_path:
            qdrant_path = str(Path(__file__).resolve().parents[1] / "qdrant_db")
        client = QdrantClient(path=qdrant_path)

    # Ensure collection exists
    try:
        client.get_collection(collection_name)
    except Exception:
        sample_vector = embedding_model.embed_query("collection_init")
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=len(sample_vector),
                distance=Distance.COSINE,
            ),
        )

    # Build vectorstore and insert documents
    vectorstore = Qdrant(
        client=client,
        collection_name=collection_name,
        embeddings=embedding_model,
    )
    if not docs:
        logger.warning("No documents provided. Vectorstore created without inserts.")
        return vectorstore

    vectorstore.add_documents(docs)
    logger.info("Inserted %d documents into collection '%s'.", len(docs), collection_name)
    return vectorstore


# rerank.
def rerank(query: str, docs: List[Document], reranker: CrossEncoder) -> List[Document]:
    """Rerank documents with a cross-encoder and return docs in new order."""
    pairs = [(query, doc.page_content) for doc in docs]
    scores = reranker.predict(pairs)

    ranked_docs = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
    return [doc for doc, _ in ranked_docs]


# retrieve.
def retrieve(
    query: str,
    vectorstore: Qdrant,
    reranker: Optional[CrossEncoder] = None,
    k: int | None = None,
    top_n: int | None = None,
    metadata_filter: Optional[Dict[str, Any]] = None,
) -> List[Document]:
    """
    Dense retrieval from Qdrant with optional reranking.
    Returns top_n documents (no scores).
    """
    qdrant_filter = _build_qdrant_filter(metadata_filter)

    rag_cfg = get_rag_config()
    if k is None:
        k = rag_cfg.get("k_dense", 10)
    if top_n is None:
        top_n = rag_cfg.get("top_n", 10)

    logger.info(
        "Retrieving with k=%d, top_n=%d, metadata_filter=%s",
        k,
        top_n,
        metadata_filter,
    )
    scored_docs = _query_points_search_with_scores(
        vectorstore=vectorstore,
        query=query,
        qdrant_filter=qdrant_filter,
        k=k,
    )
    docs = [doc for doc, _ in scored_docs]
    if not docs:
        logger.info("No relevant documents found.")
        return []
    if reranker is not None and docs:
        logger.info("Reranking %d documents.", len(docs))
        docs = rerank(query, docs, reranker)
    elif reranker is None:
        logger.info("Reranker not provided. Returning base retrieval results.")
    else:
        logger.info("No documents retrieved to rerank.")

    return docs[:top_n]


# retrieve_with_scores.
def retrieve_with_scores(
    query: str,
    vectorstore: Qdrant,
    k: int | None = None,
    metadata_filter: Optional[Dict[str, Any]] = None,
) -> List[tuple[Document, float]]:
    """
    Retrieve documents with scores using Qdrant query_points.
    """
    qdrant_filter = _build_qdrant_filter(metadata_filter)
    if k is None:
        k = get_rag_config().get("k_dense", 10)
    return _query_points_search_with_scores(
        vectorstore=vectorstore,
        query=query,
        qdrant_filter=qdrant_filter,
        k=k,
    )

# build_bm25_index.
def build_bm25_index(docs: List[Document]) -> BM25Index:
    """Build an in-memory BM25 index for sparse retrieval."""
    return BM25Index(docs)



# _query_points_search_with_scores.
def _query_points_search_with_scores(
    vectorstore: Qdrant,
    query: str,
    qdrant_filter: Optional[Filter],
    k: int,
) -> List[tuple[Document, float]]:
    """
    Stable search path that relies on QdrantClient.query_points, which is
    available even when QdrantClient.search() is not.
    """
    client = getattr(vectorstore, "client", None)
    if client is None:
        raise RuntimeError("Vectorstore has no Qdrant client attached.")

    embeddings = getattr(vectorstore, "embeddings", None) or getattr(
        vectorstore, "embedding", None
    )
    if embeddings is None:
        raise RuntimeError("Vectorstore does not expose embeddings for query.")

    query_vector = embeddings.embed_query(query)
    response = client.query_points(
        collection_name=vectorstore.collection_name,
        query=query_vector,
        limit=k,
        query_filter=qdrant_filter,
        with_payload=True,
        with_vectors=False,
    )

    content_key = getattr(vectorstore, "content_payload_key", "page_content")
    metadata_key = getattr(vectorstore, "metadata_payload_key", "metadata")
    docs: List[tuple[Document, float]] = []
    for point in response.points:
        payload = point.payload or {}
        content = payload.get(content_key, "")
        metadata = payload.get(metadata_key, {})
        if not isinstance(metadata, dict):
            metadata = {}
        if not metadata:
            metadata = {k: v for k, v in payload.items() if k != content_key}
        doc = Document(page_content=content, metadata=metadata)
        docs.append((doc, float(point.score)))

    return docs

# Hybrid late fusion function

def hybrid_retrieve(
    query: str,
    vectorstore: Qdrant,
    bm25_index: BM25Index,
    k_dense: int | None = None,
    k_sparse: int | None = None,
    top_n: int | None = None,
    metadata_filter: Optional[Dict[str, Any]] = None,
) -> List[Tuple[Document, float]]:
    """
    True hybrid retrieval: dense (Qdrant) + sparse (BM25) with RRF fusion.
    Returns (Document, hybrid_score).
    """
    rag_cfg = get_rag_config()
    if k_dense is None:
        k_dense = rag_cfg.get("k_dense", 10)
    if k_sparse is None:
        k_sparse = rag_cfg.get("k_sparse", 10)
    if top_n is None:
        top_n = rag_cfg.get("top_n", 10)

    dense_scored = retrieve_with_scores(
        query=query,
        vectorstore=vectorstore,
        k=k_dense,
        metadata_filter=metadata_filter,
    )

    sparse_scores = bm25_index.scores(query)
    sparse_ranked = sorted(
        zip(bm25_index.docs, sparse_scores),
        key=lambda x: x[1],
        reverse=True,
    )[:k_sparse]

    rrf_k = 60.0
    fused: Dict[Tuple[str, Tuple[Tuple[str, Any], ...]], float] = {}

    for rank, (doc, _) in enumerate(dense_scored, start=1):
        fused[_doc_key(doc)] = fused.get(_doc_key(doc), 0.0) + 1.0 / (rrf_k + rank)

    for rank, (doc, _) in enumerate(sparse_ranked, start=1):
        fused[_doc_key(doc)] = fused.get(_doc_key(doc), 0.0) + 1.0 / (rrf_k + rank)

    # Materialize back to docs (prefer dense docs for metadata fidelity)
    doc_map: Dict[Tuple[str, Tuple[Tuple[str, Any], ...]], Document] = {}
    for doc, _ in dense_scored:
        doc_map[_doc_key(doc)] = doc
    for doc, _ in sparse_ranked:
        doc_map.setdefault(_doc_key(doc), doc)

    fused_ranked = sorted(
        ((doc_map[k], score) for k, score in fused.items()),
        key=lambda x: x[1],
        reverse=True,
    )

    return fused_ranked[:top_n]


########## Demo ################

# def _load_doc_json(path: str) -> Dict[str, Any]:
#     with open(path, "r", encoding="utf-8") as f:
#         return json.load(f)


# def _infer_doc_name(doc_json: Dict[str, Any], fallback_path: str) -> str:
#     return (
#         doc_json.get("origin", {}).get("filename")
#         or doc_json.get("name")
#         or os.path.basename(fallback_path)
#     )


# def _build_chunks_from_doc_json(path: str) -> List[Dict[str, Any]]:
#     if DoclingParser is None or create_parent_child_chunks is None:
#         raise RuntimeError("chunking.py is not available for demo pipeline.")

#     doc_json = _load_doc_json(path)
#     parser = DoclingParser(doc_json)
#     sections = parser.build_sections()
#     doc_name = _infer_doc_name(doc_json, path)
#     return create_parent_child_chunks(sections, doc_name=doc_name)


if __name__ == "__main__":
    # --- Demo data from your sample chunks ---
    sample_chunks = [
        {
            "text": "differs from conventional RAG (Figure 1 left), which",
            "metadata": {
                "parent_title": "1 INTRODUCTION",
                "parent_text": (
                    "State-of-the-art LLMs continue to struggle with factual errors "
                    "(Mallen et al., 2023; Min et al., 2023) despite their increased "
                    "model and data scale (Ouyang et al., 2022). Retrieval-Augmented Generatio"
                ),
                "chunk_id": "child_chunk_3",
            },
        },
        {
            "text": (
                "1 Our code and trained models are available at https://selfrag.github.io/ . "
                "consistently retrieves a fixed number of documents for generation regardless "
                "of the retrieval necessity (e.g., the bottom fi"
            ),
            "metadata": {
                "parent_title": "1 INTRODUCTION",
                "parent_text": (
                    "State-of-the-art LLMs continue to struggle with factual errors "
                    "(Mallen et al., 2023; Min et al., 2023) despite their increased "
                    "model and data scale (Ouyang et al., 2022). Retrieval-Augmented Generatio"
                ),
                "chunk_id": "child_chunk_4",
            },
        },
    ]
 
    

    # --- Build vectorstore ---
    vectorstore = build_vectorstore(
        chunks=sample_chunks,
        collection_name="rag_demo",
    )
    logging.info("Vectorstore built with sample chunks. Ready for retrieval demo.")
    # --- Optional reranker ---
    reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    # --- Hybrid retrieval test (dense + BM25) ---
    query = "What are the struggles of LLMs?"
    bm25_index = build_bm25_index(_normalize_chunks(sample_chunks))
    hybrid_results = hybrid_retrieve(
        query,
        vectorstore=vectorstore,
        bm25_index=bm25_index,
        k_dense=10,
        k_sparse=10,
        top_n=5,
        metadata_filter=None,
    )

    logger.info("Query: %s", query)
    logger.info("Top results: %d", len(hybrid_results))
    for i, (doc, score) in enumerate(hybrid_results, start=1):
        logger.info("Result %d", i)
        logger.info("Hybrid Score: %s", score)
        logger.info("Content: %s", doc.page_content[:300])
        logger.info("Metadata: %s", doc.metadata)

    # --- Optional reranked results ---
    reranked = retrieve(
        query,
        vectorstore=vectorstore,
        reranker=reranker,
        k=10,
        top_n=5,
        metadata_filter=None,
    )
    logger.info("Reranked results:")
    for i, doc in enumerate(reranked, start=1):
        logger.info("Reranked %d", i)
        logger.info("Content: %s", doc.page_content[:300])
        logger.info("Metadata: %s", doc.metadata)
