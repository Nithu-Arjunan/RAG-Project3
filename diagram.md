# Agentic RAG — project flow

Flow aligned with `/query` in `src/main.py` and the LangGraph in `src/graph.py`: multi-tier cache first; on full miss, **Adaptive RAG** → **CRAG** (`grade_documents`) → **SELF-RAG** (`self_rag_check`).

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

## Color legend

| Color | Meaning |
| --- | --- |
| Blue | **Entry** — user request |
| Teal | **Cache** — normalize, tiers, direct cache responses |
| Yellow | **Cache miss** — full pipeline |
| Orange | **Adaptive RAG** — routing A / B / C |
| Purple | **CRAG** — grade documents, rewrite / retry loops |
| Pink | **Generation** — `generate_from_docs` |
| Green | **SELF-RAG** — critique and retry decisions |
| Gray | **Terminal** — end states, final response, cache write |

## Notes

- **Cache** tiers: exact → semantic → retrieval. Retrieval hits still run `generate_from_docs` + **SELF-RAG** (no full graph).
- **Adaptive RAG** is `route_query` and A/B/C routing in `graph.py`.
- **CRAG** is `grade_documents`; irrelevant docs trigger rewrite/subquery loops.
- **SELF-RAG** is `self_rag_check` after generation; path **A** skips retrieval, grading, and self-check in the graph.
