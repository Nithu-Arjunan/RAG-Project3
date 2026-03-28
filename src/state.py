from typing import Any, Dict, List, Optional, TypedDict

class RAGState(TypedDict, total=False):
    #  Input
    query: str
    chat_history: Optional[List[str]]
    doc_json: Optional[Dict[str, Any]]
    doc_name: Optional[str]

    #  Classification
    complexity: Optional[str]  # "A", "B", "C"
    original_query: Optional[str]

    #  Retrieval
    retrieved_docs: Optional[List[Dict[str, Any]]]
    retrieved_scores: Optional[List[float]]
    retrieved_metadata: Optional[List[Dict[str, Any]]]
    reranked_docs: Optional[List[Dict[str, Any]]]
    reranker: Optional[Any]
    run_web_search: Optional[str]
    web_results: Optional[List[Dict[str, Any]]]
    vectorstore: Optional[Any]
    bm25_index: Optional[Any]

    # metadata for versioning and source tracking
    sources: Optional[List[str]]
    doc_version: Optional[int] 
    page_no: Optional[int]
    citations: Optional[List[str]]

    #  Generation
    answer: Optional[str]
    decision: Optional[str]

    #  Multi-step reasoning
    intermediate_steps: Optional[List[str]]
    iteration: Optional[int]
    attempts: Optional[int]
    sub_query: Optional[str]
    rewritten_query: Optional[str]

    #  Control / evaluation
    confidence_score: Optional[float]
    

    #  Debugging (optional but useful)
    logs: Optional[List[str]]
    trace: Optional[List[Dict[str, Any]]]

    # Token usage tracking
    token_usage: Optional[Dict[str, Any]]
    last_usage: Optional[Dict[str, int]]

    
