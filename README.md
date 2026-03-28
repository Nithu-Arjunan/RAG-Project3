## RAG OPTMIZATION - ADAPTIVE RAG , CRAG , SELF RAG

An end-to-end, local-first Agentic RAG app with a FastAPI backend and a React (Vite) frontend. Upload a document once, then ask questions with traceable retrieval and caching.

**Highlights**
- Upload and ingest documents with Docling.
- Hybrid retrieval with Qdrant + BM25 and reranking (always on).
- Multi-tier caching (exact, semantic, retrieval) for faster responses.
- Simple React UI for upload, query, and diagnostics.

## Architecture (query flow)

`POST /query` checks a multi-tier cache first. On a full miss, the LangGraph in `src/graph.py` runs **Adaptive RAG** (`route_query`) → **CRAG** (`grade_documents`) → **SELF-RAG** (`self_rag_check`). A longer reference with the same diagram lives in [`diagram.md`](diagram.md).

```mermaid
flowchart TB
    subgraph ingress["1. Request"]
        U["User query\nPOST /query"]
    end

    subgraph cache["2. Multi-tier cache"]
        U --> N["Normalize + hash / embed"]
        N --> T1{"Tier 1: exact\nquery hash?"}
        T1 -->|hit| H1["Return answer + sources\n(cache_status: hit_exact)"]
        T1 -->|miss| T2{"Tier 2: semantic\nsimilarity ≥ threshold?"}
        T2 -->|hit| H2["Return answer + sources\n(cache_status: hit_semantic)"]
        T2 -->|miss| T3{"Tier 3: retrieval\ncached chunks?"}
        T3 -->|hit| GH["generate_from_docs\n(from cached chunks)"]
        GH --> SR0["SELF-RAG\n(self_rag_check)"]
        SR0 --> HR["Response\n(cache_status: hit_retrieval)"]
    end

    T3 -->|miss| M["Cache miss\n→ rag_graph.invoke"]

    subgraph adaptive["3. Adaptive RAG (route_query)"]
        M --> RQ["Classify complexity A / B / C"]
        RQ -->|A| SG["simple_generate"]
        RQ -->|B| RET["retrieve\n(hybrid retrieval)"]
        RQ -->|C| Cprep["generate_subquery OR\ntransform_query"]
        Cprep --> RET
    end

    subgraph crag["4. CRAG (grade_documents)"]
        RET --> GD["Grade each doc\n(binary relevance)"]
        GD --> OKD{"Any relevant\ndocs?"}
        OKD -->|yes\nrun_web_search: No| GEN["generate_from_docs"]
        OKD -->|no| RW["transform_query /\ngenerate_subquery → retrieve"]
        RW --> GD
    end

    subgraph selfrag["5. SELF-RAG (self_rag_check)"]
        GEN --> SR["Hallucination grader +\nanswer relevance grader"]
        SR --> USE{"decision\n== useful?"}
        USE -->|yes| ENDN["END"]
        USE -->|no| RWA["rewrite / retry\n→ retrieve"]
        RWA --> GD
    end

    SG --> ENDA["END"]
    ENDN --> OUT["Answer + populate cache\n(exact, semantic, retrieval)"]
    ENDA --> OUT
    H1 --> STOP1(["Done"])
    H2 --> STOP2(["Done"])
    HR --> STOP3(["Done"])

    classDef entry fill:#E3F2FD,stroke:#1565C0,stroke-width:2px,color:#0D47A1
    classDef cache fill:#E0F7FA,stroke:#00838F,stroke-width:2px,color:#004D40
    classDef miss fill:#FFFDE7,stroke:#F9A825,stroke-width:2px,color:#E65100
    classDef adaptive fill:#FFF3E0,stroke:#EF6C00,stroke-width:2px,color:#BF360C
    classDef crag fill:#F3E5F5,stroke:#7B1FA2,stroke-width:2px,color:#4A148C
    classDef generate fill:#FCE4EC,stroke:#C2185B,stroke-width:2px,color:#880E4F
    classDef selfrag fill:#E8F5E9,stroke:#2E7D32,stroke-width:2px,color:#1B5E20
    classDef terminal fill:#ECEFF1,stroke:#546E7A,stroke-width:2px,color:#37474F

    class U entry
    class N,T1,T2,T3,H1,H2 cache
    class M miss
    class RQ,SG,RET,Cprep adaptive
    class GD,OKD,RW crag
    class GH,GEN generate
    class SR0,SR,USE,RWA selfrag
    class ENDN,ENDA,OUT,HR,STOP1,STOP2,STOP3 terminal
```

| Color | Stage |
| --- | --- |
| Blue | Entry — user request |
| Teal | Cache — tiers and direct cache responses |
| Yellow | Cache miss — full pipeline |
| Orange | Adaptive RAG — A / B / C routing |
| Purple | CRAG — document grading and rewrite loops |
| Pink | Generation — `generate_from_docs` |
| Green | SELF-RAG — graders and retry |
| Gray | Terminal — end state and cache write |

**Notes:** Retrieval cache hits skip the full graph but still run generation + SELF-RAG. Path **A** (simple queries) skips retrieval, CRAG, and SELF-RAG in the graph.

## Quick Start

### Prerequisites
- Python 3.13+
- Node 18+ (for the UI)

### 1. Backend setup
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. Start the API
```powershell
cd src
uvicorn src.main:app --reload --host 0.0.0.0 --port 8000
```

Health check:
```text
http://127.0.0.1:8000/health
```

### 3. Start the UI
```powershell
cd ui
npm install
npm run dev
```

Open:
```text
http://localhost:5173
```

Note: the UI currently calls the backend at `http://127.0.0.1:8000`.

## Configuration
Main config lives in `src/config.yaml`:

- `rag`: chunking and retrieval sizes.
- `embedding`: embedding model name.
- `reranker`: cross-encoder model name.
- `cache`: caching backend and thresholds.

Default embedding and reranker models:
- `sentence-transformers/all-MiniLM-L6-v2`
- `cross-encoder/ms-marco-MiniLM-L-6-v2`

The first run may download model weights from Hugging Face.

### Sample `.env`
Create a `.env` file in the project root if you use Vertex AI or other env-based config.

```env
# Vertex AI (used by litellm in src/config.yaml)
VERTEX_PROJECT=your-gcp-project-id
VERTEX_LOCATION=us-central1

# Optional: logging level
LOG_LEVEL=INFO
```

## API

Base URL: `http://127.0.0.1:8000`

### `POST /ingest`
Upload a file and build its index.

Request:
```bash
curl -X POST -F "file=@./path/to/file.pdf" http://127.0.0.1:8000/ingest
```

Response:
```json
{"file_id":"...","duplicate":false,"doc_name":"..."}
```

### `POST /query`
Ask a question about a previously ingested file.

Request:
```json
{"query":"What is this document about?","file_id":"<file_id>"}
```

Response:
```json
{
  "answer": "...",
  "decision": "B",
  "complexity": "B",
  "trace": [],
  "sources": [],
  "time_ms": 1234,
  "cache_status": "miss",
  "token_usage": {}
}
```

### `POST /cache/check`
Checks whether the query would hit cache.

### `POST /cache/clear`
Clears all caches.

### `GET /files`
Lists ingested files.

### `GET /health`
Simple health check.

## Project Structure

```
.
├─ src
│  ├─ main.py               # FastAPI app and endpoints
│  ├─ embedding.py          # Qdrant + BM25 + reranking
│  ├─ chunking.py           # Docling parsing + chunking
│  ├─ graph.py              # LangGraph orchestration
│  ├─ node.py               # Agentic steps and grading
│  └─ cache/                # Cache backends and utilities
├─ ui                        # React frontend
└─ data                      # Local runtime data (generated)
```

Generated data directories:
- `data/uploads` for uploaded files
- `data/bm25` for BM25 index pickles
- `data/ingestion_registry.json` for file metadata (kept in Git for reference)
- `qdrant_db` for Qdrant local store

Sample Docling output:
- `doc_output.json` is a sample Docling JSON artifact created by the CLI/demo scripts (for example, `python src/ingestion.py ...`). It is **not** created by the `/ingest` API path.

Registry files:
- `data/ingestion_registry.json` is intentionally **not** ignored so it can be inspected.
- `src/file_hash_registry.json` is a local CLI registry and **is** ignored in `.gitignore`.


## Troubleshooting

**UI says “Failed to fetch”**
- Ensure the backend is running on `http://127.0.0.1:8000`.
- Restart Vite after changes to `ui/src/App.tsx`.

**`ui:dist_not_found` warning**
- This is normal during local dev. The UI is served by Vite, not by `ui/dist`.

**Upload fails with a Docling or Hugging Face error**
- The first upload may download model weights.
- Check your internet connection and retry.
