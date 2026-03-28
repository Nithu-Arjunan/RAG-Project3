import logging
from langgraph.graph import StateGraph, START, END
from state import RAGState
from config_loader import get_rag_config
from node import (
    route_query,
    simple_generate,
    retrieve,
    grade_documents,
    transform_query,
    generate_subquery,
    generate_from_docs,
    final_fail,
    self_rag_check,
)
from node import hallucination_grader, answer_grader
from logging_config import configure_logging
configure_logging()
logger = logging.getLogger(__name__)
workflow = StateGraph(RAGState)
MAX_ATTEMPTS = get_rag_config().get("max_attempts", 6)


workflow.add_node("route_query", route_query)
workflow.add_node("simple_generate", simple_generate)
workflow.add_node("retrieve", retrieve)
workflow.add_node("grade_documents", grade_documents)
workflow.add_node("generate_subquery", generate_subquery)
workflow.add_node("transform_query", transform_query)
workflow.add_node("generate_docs", generate_from_docs)
workflow.add_node("final_fail", final_fail)
workflow.add_node("self_rag", self_rag_check)

workflow.set_entry_point("route_query")

# Adaptive Routing
def _route_entry(state: RAGState) -> str:
    complexity = state.get("complexity")
    if complexity == "A":
        logger.info("graph:route_entry trace_id=%s route=simple_generate", state.get("trace_id", "-"))
        return "simple_generate"
    if complexity == "B":
        logger.info("graph:route_entry trace_id=%s route=retrieve", state.get("trace_id", "-"))
        return "retrieve"
    # complexity C
    if state.get("attempts", 0) >= MAX_ATTEMPTS:
        logger.info("graph:route_entry trace_id=%s route=final_fail reason=max_attempts", state.get("trace_id", "-"))
        return "final_fail"
    route = "transform_query" if state.get("iteration", 0) >= 3 else "generate_subquery"
    logger.info("graph:route_entry trace_id=%s route=%s", state.get("trace_id", "-"), route)
    return route

workflow.add_conditional_edges(
    "route_query",
    _route_entry,
    {
        "simple_generate": "simple_generate",
        "retrieve": "retrieve",
        "generate_subquery": "generate_subquery",
        "transform_query": "transform_query",
        "final_fail": "final_fail",
    },
)

workflow.add_edge("simple_generate", END)

# RAG Flow
workflow.add_edge("retrieve", "grade_documents")

# _next_on_bad_docs.
def _next_on_bad_docs(state: RAGState) -> str:
    if state.get("attempts", 0) >= MAX_ATTEMPTS:
        return "final_fail"
    if state.get("complexity") == "C" and state.get("iteration", 0) < 3:
        return "retry_c"
    return "rewrite"

workflow.add_conditional_edges(
    "grade_documents",
    lambda x: "good" if x["run_web_search"] == "No" else _next_on_bad_docs(x),
    {
        "good": "generate_docs",
        "retry_c": "generate_subquery",
        "rewrite": "transform_query",
        "final_fail": "final_fail",
    },
)

workflow.add_edge("generate_subquery", "retrieve")
workflow.add_edge("transform_query", "retrieve")

# Self RAG
workflow.add_edge("generate_docs", "self_rag")

# _next_on_bad_answer.
def _next_on_bad_answer(state: RAGState) -> str:
    if state.get("decision") == "useful":
        return "done"
    if state.get("attempts", 0) >= MAX_ATTEMPTS:
        return "final_fail"
    if state.get("complexity") == "C" and state.get("iteration", 0) < 3:
        return "retry_c"
    return "rewrite"

workflow.add_conditional_edges(
    "self_rag",
    _next_on_bad_answer,
    {
        "done": END,
        "retry_c": "generate_subquery",
        "rewrite": "transform_query",
        "final_fail": "final_fail",
    },
)

workflow.add_edge("final_fail", END)


app = workflow.compile()

