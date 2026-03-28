import hashlib
import json
import os
import pickle
import logging
import uuid
from pathlib import Path
from typing import Any, Dict

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import time
from langchain_community.vectorstores import Qdrant
from langchain_huggingface import HuggingFaceEmbeddings
from qdrant_client import QdrantClient

from chunking import DoclingParser, create_parent_child_chunks
from config_loader import get_cache_config, get_embedding_config
from embedding import build_bm25_index, build_vectorstore, get_reranker, _normalize_chunks
from graph import app as rag_graph
from node import generate_from_docs, self_rag_check
from cache import get_cache_backend
from cache.embeddding_cache import embed_query, hash_query, normalize_query
from logging_config import configure_logging

configure_logging()
logger = logging.getLogger(__name__)

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS", "1")
from docling.document_converter import DocumentConverter

_cache_cfg = get_cache_config()
EXACT_TTL_SECONDS = _cache_cfg["exact_ttl_seconds"]
SEMANTIC_TTL_SECONDS = _cache_cfg["semantic_ttl_seconds"]
RETRIEVAL_TTL_SECONDS = _cache_cfg["retrieval_ttl_seconds"]
SEMANTIC_THRESHOLD = _cache_cfg["semantic_threshold"]
RETRIEVAL_THRESHOLD = _cache_cfg["retrieval_threshold"]


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
REGISTRY_PATH = DATA_DIR / "ingestion_registry.json"
BM25_DIR = DATA_DIR / "bm25"
QDRANT_PATH = ROOT / "qdrant_db"
UPLOAD_DIR = DATA_DIR / "uploads"
UI_DIST_DIR = ROOT / "ui" / "dist"

DATA_DIR.mkdir(parents=True, exist_ok=True)
BM25_DIR.mkdir(parents=True, exist_ok=True)
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

INDEX_CACHE: Dict[str, Dict[str, Any]] = {}


# _load_registry.
def _load_registry() -> Dict[str, Dict[str, Any]]:
    # Load ingestion registry from disk.
    if not REGISTRY_PATH.exists():
        return {}
    with open(REGISTRY_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


# _save_registry.
def _save_registry(registry: Dict[str, Dict[str, Any]]) -> None:
    # Persist ingestion registry to disk.
    with open(REGISTRY_PATH, "w", encoding="utf-8") as f:
        json.dump(registry, f, indent=2)


# _hash_bytes.
def _hash_bytes(data: bytes) -> str:
    # Compute a stable content hash for file de-duplication.
    return hashlib.sha256(data).hexdigest()


# _load_vectorstore.
def _load_vectorstore(collection_name: str) -> Qdrant:
    # Load Qdrant vectorstore for a collection using configured embeddings.
    embed_cfg = get_embedding_config()
    embeddings = HuggingFaceEmbeddings(
        model_name=embed_cfg.get("model_name", "sentence-transformers/all-MiniLM-L6-v2")
    )
    client = QdrantClient(path=str(QDRANT_PATH))
    return Qdrant(client=client, collection_name=collection_name, embeddings=embeddings)


# _convert_file_to_docling_json.
def _convert_file_to_docling_json(file_path: Path) -> Dict[str, Any]:
    # Convert a file to Docling JSON format.
    converter = DocumentConverter()
    result = converter.convert(str(file_path))
    doc = result.document
    return doc.export_to_dict()


# _get_indices.
def _get_indices(file_id: str) -> Dict[str, Any]:
    # Load BM25 and vectorstore indices for a file_id (cached in memory).
    if file_id in INDEX_CACHE:
        return INDEX_CACHE[file_id]

    registry = _load_registry()
    meta = registry.get(file_id)
    if not meta:
        raise HTTPException(status_code=404, detail="File not found. Please upload first.")

    bm25_path = Path(meta["bm25_path"])
    if not bm25_path.exists():
        raise HTTPException(status_code=500, detail="BM25 index missing on disk.")

    with open(bm25_path, "rb") as f:
        bm25_index = pickle.load(f)

    vectorstore = _load_vectorstore(meta["collection_name"])

    INDEX_CACHE[file_id] = {
        "vectorstore": vectorstore,
        "bm25_index": bm25_index,
        "doc_name": meta.get("doc_name"),
    }
    return INDEX_CACHE[file_id]


class QueryRequest(BaseModel):
    query: str
    file_id: str


class CacheCheckRequest(BaseModel):
    query: str
    file_id: str


app = FastAPI(title="Agentic AI API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ingest.
@app.post("/ingest")
async def ingest(file: UploadFile = File(...)):
    # Ingest an uploaded file, build indices, and register metadata.
    logger.info("ingest:start filename=%s content_type=%s", file.filename, file.content_type)
    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing filename.")

    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Empty file.")

    file_id = _hash_bytes(raw)
    registry = _load_registry()

    if file_id in registry:
        logger.info("ingest:duplicate file_id=%s filename=%s", file_id, file.filename)
        return {
            "file_id": file_id,
            "duplicate": True,
            "doc_name": registry[file_id].get("doc_name"),
        }

    file_ext = Path(file.filename).suffix.lower()
    if file_ext == ".json":
        try:
            doc_json = json.loads(raw.decode("utf-8"))
        except Exception as exc:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid JSON file: {exc}",
            ) from exc
    else:
        safe_name = Path(file.filename).name
        upload_path = UPLOAD_DIR / f"{file_id}_{safe_name}"
        try:
            with open(upload_path, "wb") as f:
                f.write(raw)
        except Exception as exc:
            raise HTTPException(
                status_code=500, detail=f"Failed to save upload: {exc}"
            ) from exc

        try:
            doc_json = _convert_file_to_docling_json(upload_path)
        except Exception as exc:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported or unreadable document: {exc}",
            ) from exc

    doc_name = (
        doc_json.get("origin", {}).get("filename")
        or doc_json.get("name")
        or file.filename
    )

    parser = DoclingParser(doc_json)
    sections = parser.build_sections()
    chunks = create_parent_child_chunks(sections, doc_name=doc_name)

    collection_name = f"rag_{file_id[:12]}"
    vectorstore = build_vectorstore(
        chunks=chunks,
        collection_name=collection_name,
        qdrant_path=str(QDRANT_PATH),
    )
    bm25_index = build_bm25_index(_normalize_chunks(chunks))

    bm25_path = BM25_DIR / f"{file_id}.pkl"
    with open(bm25_path, "wb") as f:
        pickle.dump(bm25_index, f)

    registry[file_id] = {
        "filename": file.filename,
        "doc_name": doc_name,
        "collection_name": collection_name,
        "bm25_path": str(bm25_path),
    }
    _save_registry(registry)
    try:
        cache = get_cache_backend()
        cache.bump_doc_version()
    except Exception:
        # Cache should not block ingestion
        pass

    INDEX_CACHE[file_id] = {
        "vectorstore": vectorstore,
        "bm25_index": bm25_index,
        "doc_name": doc_name,
    }

    logger.info("ingest:complete file_id=%s doc_name=%s", file_id, doc_name)
    return {"file_id": file_id, "duplicate": False, "doc_name": doc_name}


# query.
@app.post("/query")
def query(req: QueryRequest):
    # Run retrieval + generation flow for a query against an ingested file.
    trace_id = uuid.uuid4().hex[:8]
    logger.info("query:start trace_id=%s file_id=%s query=%s", trace_id, req.file_id, req.query)
    cache = get_cache_backend()
    source_filter = req.file_id
    normalized = normalize_query(req.query)
    query_hash = hash_query(normalized)
    start_ts = time.time()

    exact_hit = cache.get_exact(query_hash)
    if exact_hit:
        logger.info("query:cache_hit_exact trace_id=%s", trace_id)
        total_ms = int((time.time() - start_ts) * 1000)
        return {
            "answer": exact_hit.get("answer"),
            "decision": "cache_hit",
            "complexity": "A",
            "trace": [{"step": "cache_exact", "data": {"hit": True}}],
            "sources": json.loads(exact_hit.get("sources_json", "[]")),
            "time_ms": total_ms,
            "cache_status": "hit_exact",
            "token_usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "by_call": [],
            },
        }

    embedding = embed_query(normalized)
    semantic_hit = cache.get_semantic(
        embedding, threshold=SEMANTIC_THRESHOLD, source_filter=source_filter
    )
    if semantic_hit:
        logger.info("query:cache_hit_semantic trace_id=%s similarity=%s", trace_id, semantic_hit.get("similarity"))
        total_ms = int((time.time() - start_ts) * 1000)
        return {
            "answer": semantic_hit.get("answer"),
            "decision": "cache_hit",
            "complexity": "B",
            "trace": [
                {
                    "step": "cache_semantic",
                    "data": {"hit": True, "similarity": semantic_hit.get("similarity")},
                }
            ],
            "sources": json.loads(semantic_hit.get("sources_json", "[]")),
            "time_ms": total_ms,
            "cache_status": "hit_semantic",
            "token_usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "by_call": [],
            },
        }

    retrieval_hit = cache.get_retrieval(
        embedding, threshold=RETRIEVAL_THRESHOLD, source_filter=source_filter
    )
    if retrieval_hit:
        logger.info("query:cache_hit_retrieval trace_id=%s similarity=%s", trace_id, retrieval_hit.get("similarity"))
        cached_docs = json.loads(retrieval_hit.get("chunks_json", "[]"))
        state = {"query": req.query, "retrieved_docs": cached_docs}
        state.update(generate_from_docs(state))
        state.update(self_rag_check(state))
        total_ms = int((time.time() - start_ts) * 1000)
        return {
            "answer": state.get("answer"),
            "decision": state.get("decision"),
            "complexity": "B",
            "trace": state.get("trace", []) + [
                {"step": "cache_retrieval", "data": {"hit": True, "similarity": retrieval_hit.get("similarity")}}
            ],
            "sources": cached_docs,
            "time_ms": total_ms,
            "cache_status": "hit_retrieval",
            "token_usage": state.get("token_usage"),
        }

    indices = _get_indices(req.file_id)
    vectorstore = indices["vectorstore"]
    bm25_index = indices["bm25_index"]
    doc_name = indices.get("doc_name")
    reranker = get_reranker()

    initial_state = {
        "query": req.query,
        "vectorstore": vectorstore,
        "bm25_index": bm25_index,
        "doc_name": doc_name,
        "reranker": reranker,
        "trace_id": trace_id,
    }

    result = rag_graph.invoke(initial_state)
    answer = result.get("answer") or result.get("final_answer")
    retrieved_docs = result.get("retrieved_docs") or []
    sources = []
    for doc in retrieved_docs:
        metadata = doc.get("metadata", {}) if isinstance(doc, dict) else {}
        sources.append(
            {
                "text": doc.get("text") if isinstance(doc, dict) else str(doc),
                "page_number": metadata.get("page_number"),
                "document": metadata.get("document") or doc_name,
                "score": doc.get("hybrid_score") if isinstance(doc, dict) else None,
                "metadata": metadata,
            }
        )
    try:
        doc_version = cache.get_doc_version()
        sources_json = json.dumps(sources)
        cache.set_exact(
            query_hash=query_hash,
            question=req.query,
            answer=answer or "",
            sources_json=sources_json,
            doc_version=doc_version,
            ttl_seconds=EXACT_TTL_SECONDS,
        )
        cache.set_semantic(
            question=req.query,
            embedding=embedding,
            answer=answer or "",
            sources_json=sources_json,
            doc_version=doc_version,
            ttl_seconds=SEMANTIC_TTL_SECONDS,
            source_filter=source_filter,
        )
        cache.set_retrieval(
            question=req.query,
            embedding=embedding,
            chunks_json=json.dumps(retrieved_docs),
            doc_version=doc_version,
            ttl_seconds=RETRIEVAL_TTL_SECONDS,
            source_filter=source_filter,
        )
    except Exception:
        pass
    total_ms = int((time.time() - start_ts) * 1000)
    logger.info(
        "query:complete trace_id=%s decision=%s complexity=%s cache=miss time_ms=%d",
        trace_id,
        result.get("decision"),
        result.get("complexity"),
        total_ms,
    )
    return {
        "answer": answer,
        "decision": result.get("decision"),
        "complexity": result.get("complexity"),
        "trace": result.get("trace"),
        "sources": sources,
        "time_ms": total_ms,
        "cache_status": "miss",
        "token_usage": result.get("token_usage"),
    }


# cache_check.
@app.post("/cache/check")
def cache_check(req: CacheCheckRequest):
    # Check cache status for a query without running full retrieval.
    cache = get_cache_backend()
    source_filter = req.file_id
    normalized = normalize_query(req.query)
    query_hash = hash_query(normalized)
    start_ts = time.time()

    if cache.get_exact(query_hash):
        return {
            "status": "hit_exact",
            "time_ms": int((time.time() - start_ts) * 1000),
        }

    embedding = embed_query(normalized)

    semantic_hit = cache.get_semantic(
        embedding, threshold=SEMANTIC_THRESHOLD, source_filter=source_filter
    )
    if semantic_hit:
        return {
            "status": "hit_semantic",
            "similarity": semantic_hit.get("similarity"),
            "time_ms": int((time.time() - start_ts) * 1000),
        }

    retrieval_hit = cache.get_retrieval(
        embedding, threshold=RETRIEVAL_THRESHOLD, source_filter=source_filter
    )
    if retrieval_hit:
        return {
            "status": "hit_retrieval",
            "similarity": retrieval_hit.get("similarity"),
            "time_ms": int((time.time() - start_ts) * 1000),
        }

    return {
        "status": "miss",
        "time_ms": int((time.time() - start_ts) * 1000),
    }


# health.
@app.get("/health")
def health():
    # Health check endpoint.
    return {"status": "ok"}


# list_files.
@app.get("/files")
def list_files():
    # List ingested files and their metadata.
    registry = _load_registry()
    files = []
    for file_id, meta in registry.items():
        files.append(
            {
                "file_id": file_id,
                "filename": meta.get("filename"),
                "doc_name": meta.get("doc_name"),
            }
        )
    return {"count": len(files), "files": files}


# clear_cache.
@app.post("/cache/clear")
def clear_cache():
    # Clear all cache entries.
    cache = get_cache_backend()
    cleared = cache.clear_all()
    return {"cleared": cleared}


if UI_DIST_DIR.exists():
    app.mount("/", StaticFiles(directory=str(UI_DIST_DIR), html=True), name="ui")
    logger.info("ui:mounted path=/ dir=%s", UI_DIST_DIR)
else:
    logger.warning("ui:dist_not_found dir=%s", UI_DIST_DIR)
