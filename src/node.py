import json
import logging
import re
import time
from typing import Dict

from pydantic import BaseModel, Field

from litellm import completion
from embedding import hybrid_retrieve, rerank as rerank_docs
from generation import generate_answer, _load_generation_config
from logging_config import configure_logging
from state import RAGState

configure_logging()
logger = logging.getLogger(__name__)

# Define the Rouers output schema
# class RouterOutput(BaseModel):
#     action: str = Field(..., description="The action to take, e.g., 'retrieve', 'generate', 'ask_for_clarification'")
#     reasoning: str = Field(..., description="The reasoning behind the chosen action")
#     complexity: str = Field(..., description="The complexity level of the query, e.g., 'A', 'B', 'C'")

def _make_fallback_response(content: str, error: Exception | None = None):
    return {
        "choices": [{"message": {"content": content}}],
        "_fallback": True,
        "_fallback_error": str(error) if error else None,
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


# _safe_completion.
def _safe_completion(*, fallback_content: str, max_retries: int = 3, backoff_s: float = 1.0, **kwargs):
    last_err = None
    for attempt in range(max_retries + 1):
        try:
            return completion(**kwargs)
        except Exception as exc:
            last_err = exc
            if attempt == max_retries:
                break
            time.sleep(backoff_s * (2 ** attempt))
    logging.warning("LLM call failed after retries: %s", last_err)
    return _make_fallback_response(fallback_content, last_err)


# _trace_id.
def _trace_id(state: RAGState | None) -> str:
    if not state:
        return "-"
    return str(state.get("trace_id") or "-")


# _clip.
def _clip(text: object, limit: int = 160) -> str:
    if text is None:
        return ""
    s = str(text)
    return s if len(s) <= limit else f"{s[:limit]}..."


# _normalize_usage.
def _normalize_usage(usage: object | None) -> dict:
    if usage is None:
        return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    if not isinstance(usage, dict):
        usage = {
            "prompt_tokens": getattr(usage, "prompt_tokens", 0),
            "completion_tokens": getattr(usage, "completion_tokens", 0),
            "total_tokens": getattr(usage, "total_tokens", 0),
            "input_tokens": getattr(usage, "input_tokens", 0),
            "output_tokens": getattr(usage, "output_tokens", 0),
        }

    prompt_tokens = usage.get("prompt_tokens") or usage.get("input_tokens") or 0
    completion_tokens = usage.get("completion_tokens") or usage.get("output_tokens") or 0
    total_tokens = usage.get("total_tokens") or (prompt_tokens + completion_tokens)

    return {
        "prompt_tokens": int(prompt_tokens or 0),
        "completion_tokens": int(completion_tokens or 0),
        "total_tokens": int(total_tokens or 0),
    }


# _extract_usage.
def _extract_usage(response: object) -> dict:
    usage = None
    if isinstance(response, dict):
        usage = response.get("usage")
    else:
        usage = getattr(response, "usage", None)
    return _normalize_usage(usage)


# _merge_usage.
def _merge_usage(state: RAGState, usage: dict, label: str | None = None) -> dict:
    normalized = _normalize_usage(usage)
    current = state.get("token_usage") or {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "by_call": [],
    }
    updated = {
        "prompt_tokens": current.get("prompt_tokens", 0) + normalized["prompt_tokens"],
        "completion_tokens": current.get("completion_tokens", 0) + normalized["completion_tokens"],
        "total_tokens": current.get("total_tokens", 0) + normalized["total_tokens"],
        "by_call": list(current.get("by_call", [])),
    }
    if label:
        updated["by_call"].append({"label": label, **normalized})
    return {"token_usage": updated, "last_usage": normalized}


# _extract_json_object.
def _extract_json_object(raw: str) -> dict:
    """Best-effort extraction of a JSON object from model output."""
    if not raw:
        raise ValueError("Empty model output")
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except Exception:
        pass
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError("No JSON object found in model output")
    return json.loads(match.group(0))

# _call_llm_direct_with_meta.
def _call_llm_direct_with_meta(query: str, model_name: str = "generator-model"):
    cfg = _load_generation_config(model_name=model_name)
    model = cfg.get("model")
    project_id = cfg.get("vertex_project")
    location = cfg.get("vertex_location")
    temperature = cfg.get("temperature", 0.2)
    max_tokens = cfg.get("max_tokens", 1024)

    if not model or not project_id or not location:
        logger.warning(
            "llm_call:missing_config model_set=%s project_set=%s location_set=%s",
            bool(model),
            bool(project_id),
            bool(location),
        )
        return "Model configuration missing in config.yaml.", True, {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    response = _safe_completion(
        model=model,
        messages=[{"role": "user", "content": query}],
        temperature=temperature,
        max_tokens=max_tokens,
        vertex_project=project_id,
        vertex_location=location,
        fallback_content="Temporary connection error. Please retry.",
    )
    answer = response["choices"][0]["message"]["content"]
    usage = _extract_usage(response)
    return answer, bool(response.get("_fallback")), usage


# _call_llm_direct.
def _call_llm_direct(query: str, model_name: str = "generator-model") -> str:
    answer, _, _ = _call_llm_direct_with_meta(query, model_name=model_name)
    return answer


# simple_generate.
def simple_generate(state: RAGState) -> Dict[str, str]:
    """Direct LLM response for simple queries (route A)."""
    query = state["query"]
    logger.info("simple_generate:start trace_id=%s query=%s", _trace_id(state), _clip(query))
    answer, used_fallback, usage = _call_llm_direct_with_meta(query)
    return {
        "answer": answer,
        **add_trace(state, "simple_generate", {"answer": answer, "fallback": used_fallback, "usage": usage}),
        **_merge_usage(state, usage, "simple_generate"),
    }




# def call_llm_and_update_state(state: RAGState) -> Dict[str, str]:
#     """Invoke LLM directly and return a state patch with the answer."""
#     query = state["query"]
#     answer = _call_llm_direct(query)
#     return {"answer": answer}

# # Function to handle case A simple generation
# def simple_generate(state: RAGState):
#     logging.info("---PATH A: SIMPLE GENERATION---")
#     return call_llm_and_update_state(state)

# # Function to handle case B retrieval + generation
# def retrieve_and_generate(state: RAGState):
#     logging.info("---PATH B: STANDARD RAG---")
#     query = state["query"]
#     doc_json = state.get("doc_json")
#     doc_name = state.get("doc_name")
#     reranker = state.get("reranker")
#     if doc_json is None:
#         raise ValueError("doc_json is required in state for retrieval.")

#     retrieved = run_hybrid_rerank_from_json(
#         doc_json=doc_json,
#         doc_name=doc_name,
#         query=query,
#         k_dense=10,
#         k_sparse=10,
#         top_n=10,
#         reranker=reranker,
#     )

#     return {
#         "retrieved_docs": retrieved,
#         "retrieved_scores": [r.get("hybrid_score") for r in retrieved],
#         "retrieved_metadata": [r.get("metadata") for r in retrieved],
#         "reranked_docs": retrieved if reranker is not None else None,
#     }

# # Schema for structured output
# class GradeDocuments(BaseModel):
#     binary_score: str = Field(
#         description="Documents are relevant to the question, 'yes' or 'no'"
#     )

# def grade_documents(state: RAGState) -> Dict:
#     logging.info("---CRAG: CHECKING DOCUMENT RELEVANCE---")
    
#     # Get the generator config
#     cfg = _load_generation_config(model_name="generator-model")
#     model = cfg.get("model")
#     project_id = cfg.get("vertex_project")
#     location = cfg.get("vertex_location")
#     temperature = cfg.get("temperature", 0.2)
#     max_tokens = cfg.get("max_tokens", 256)
#     if not model or not project_id or not location:
#         return {"retrieved_docs": [], "run_web_search": "Yes"}

#     retrieved_docs = state.get("retrieved_docs", [])
#     query = state["query"]
    
#     relevant_docs = []
#     search_needed = "No"

#     for doc in retrieved_docs:
#         # Simple binary grading prompt
#         doc_text = doc.get("text", "") if isinstance(doc, dict) else str(doc)
#         prompt = (
#             "Return ONLY JSON: {\"binary_score\": \"yes\"|\"no\"}. "
#             f"Document: {doc_text}\nQuery: {query}"
#         )
#         response = completion(
#             model=model,
#             messages=[{"role": "user", "content": prompt}],
#             temperature=temperature,
#             max_tokens=max_tokens,
#             vertex_project=project_id,
#             vertex_location=location,
#         )
#         raw = response["choices"][0]["message"]["content"] or ""
#         try:
#             res = GradeDocuments(**json.loads(raw))
#         except Exception:
#             res = GradeDocuments(binary_score="no")

#         if res.binary_score.lower() == "yes":
#             relevant_docs.append(doc)
#         else:
#             continue

#     # If all docs are irrelevant, we trigger the Web Search fallback
#     if not relevant_docs:
#         search_needed = "Yes"
        
#     return {"retrieved_docs": relevant_docs, "run_web_search": search_needed}

# def transform_query(state: RAGState):
#     logging.info("---CRAG: REWRITING QUERY FOR WEB---")
#     query = state["query"]
    
#     prompt = f"The initial retrieval failed for: {query}. Rewrite this to be a better search engine query."
#     better_query = _call_llm_direct(prompt) # Use your existing direct call function
    
#     return {"query": better_query}


# def web_search(state: RAGState) -> Dict:
#     """
#     Placeholder web search node.
#     Triggered when run_web_search == "Yes".
#     Replace this stub with a real search integration if/when available.
#     """
#     logging.info("---WEB SEARCH---")
#     if state.get("run_web_search") != "Yes":
#         return {}

#     query = state["query"]  # should be the rewritten query from transform_query

#     cfg = _load_generation_config(model_name="generator-model")
#     model = cfg.get("model")
#     project_id = cfg.get("vertex_project")
#     location = cfg.get("vertex_location")
#     temperature = cfg.get("temperature", 0.2)
#     max_tokens = cfg.get("max_tokens", 512)

#     if not model or not project_id or not location:
#         return {"web_results": [], "logs": ["Web search skipped (missing model config)."]}

#     # Tool definition for web search
#     tools = [
#         {
#             "google_search": {},
#         }
#     ]

#     response = completion(
#         model=model,
#         messages=[{"role": "user", "content": query}],
#         temperature=temperature,
#         max_tokens=max_tokens,
#         tools=tools,
#         vertex_project=project_id,
#         vertex_location=location,
#     )

#     content = response["choices"][0]["message"]["content"]
#     return {"web_results": [{"content": content}], "logs": [f"Web search executed for: {query}"]}

# # def generate_with_docs(state: RAGState):
# #     logging.info("---PATH B: GENERATION WITH DOCS---")
# #     query = state["query"]
# #     docs = state.get("retrieved_docs", [])
# #     answer = generate_answer(query, docs, model_name="generator-model")
# #     return {"answer": answer}
    

# # Multi step reasoning path for complex queries
# def multi_step_reasoning(state: RAGState):
#     logging.info("---PATH C: MULTI-STEP AGENT---")
#     query = state["query"]
#     context = state.get("retrieved_docs", [])
#     #iteration = state.get("iteration", 0)
#     max_steps = 3
    
#     for i in range(max_steps):
#         # 1. Generate a reasoning step and/or a sub-query
#         # Build a cleaner context string for the LLM
#         context_text = "\n".join(
#             f"- {d.get('text', '')}" for d in context if isinstance(d, dict)
#         )
#         prompt = f"""
#         Original Question: {query}
#         Current Context:\n{context_text}
        
#         Based on the current context, do you have enough information to answer the question?
#         If yes, provide the final answer starting with 'FINAL:'.
#         If no, write 'SEARCH:' followed by a specific search query to find the missing information.
#         """
        
#         # Use Gemini (via config.yaml) to decide next step
#         response = _call_llm_direct(prompt)
        
#         upper = response.upper()
#         if "FINAL:" in upper:
#             return {"final_answer": response.split("FINAL:", 1)[1].strip()}
        
#         elif "SEARCH:" in upper:
#             new_sub_query = response.split("SEARCH:", 1)[1]
#             # 2. Perform retrieval for the SUB-QUERY (not the original query)
#             retrieval_state = retrieve_and_generate(
#                 {
#                     **state,
#                     "query": new_sub_query.strip(),
#                 }
#             )
#             new_docs = retrieval_state.get("retrieved_docs", [])
#             context = context + new_docs
            
#     return {"final_answer": "I could not find enough information after multiple steps."}


# --- SELF-RAG: Hallucination Grader ---
class GradeHallucination(BaseModel):
    binary_score: str  # 'yes' or 'no'


# hallucination_grader.
def hallucination_grader(documents, generation):
    prompt = (
        "Return ONLY JSON: {\"binary_score\": \"yes\"|\"no\"}. "
        f"DOCUMENTS: {documents}\nGENERATION: {generation}"
    )

    cfg = _load_generation_config(model_name="generator-model")
    model = cfg.get("model")
    project_id = cfg.get("vertex_project")
    location = cfg.get("vertex_location")
    temperature = cfg.get("temperature", 0.2)
    max_tokens = cfg.get("max_tokens", 256)
    if not model or not project_id or not location:
        return "no", True, "", {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    response = _safe_completion(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens,
        vertex_project=project_id,
        vertex_location=location,
        fallback_content='{"binary_score":"no"}',
    )
    raw = response["choices"][0]["message"]["content"] or ""
    used_fallback = bool(response.get("_fallback"))
    usage = _extract_usage(response)
    try:
        res = GradeHallucination(**_extract_json_object(raw))
        return res.binary_score.lower(), used_fallback, raw, usage
    except Exception:
        return "no", used_fallback, raw, usage


# --- SELF-RAG: Answer Relevance Grader ---
class GradeAnswer(BaseModel):
    binary_score: str  # 'yes' or 'no'


# answer_grader.
def answer_grader(query, generation):
    prompt = (
        "Return ONLY JSON: {\"binary_score\": \"yes\"|\"no\"}. "
        f"QUERY: {query}\nGENERATION: {generation}"
    )

    cfg = _load_generation_config(model_name="generator-model")
    model = cfg.get("model")
    project_id = cfg.get("vertex_project")
    location = cfg.get("vertex_location")
    temperature = cfg.get("temperature", 0.2)
    max_tokens = cfg.get("max_tokens", 256)
    if not model or not project_id or not location:
        return "no", True, "", {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    response = _safe_completion(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens,
        vertex_project=project_id,
        vertex_location=location,
        fallback_content='{"binary_score":"no"}',
    )
    raw = response["choices"][0]["message"]["content"] or ""
    used_fallback = bool(response.get("_fallback"))
    usage = _extract_usage(response)
    try:
        res = GradeAnswer(**_extract_json_object(raw))
        return res.binary_score.lower(), used_fallback, raw, usage
    except Exception:
        return "no", used_fallback, raw, usage



# if __name__ == "__main__":
#     import json
#     import sys

#     logging.basicConfig(level=logging.INFO)

#     if len(sys.argv) > 1:
#         json_path = sys.argv[1]
#     else:
#         json_path = "doc_output.json"

#     with open(json_path, "r", encoding="utf-8") as f:
#         doc_json = json.load(f)

#     query = sys.argv[2] if len(sys.argv) > 2 else "What is Self-RAG?"

#     state: RAGState = {
#         "query": query,
#         "doc_json": doc_json,
#         "doc_name": None,
#     }

#     logging.info("Routing...")
#     route = route_query(state)
#     logging.info("Route result: %s", route)

#     logging.info("Retrieving...")
#     retrieval = retrieve_and_generate(state)
#     logging.info("Retrieved docs: %d", len(retrieval.get("retrieved_docs", [])))

#     state.update(retrieval)
#     logging.info("Generating answer...")
#     answer = call_llm_and_update_state(state)
#     logging.info("Answer: %s", answer.get("answer"))

#     logging.info("Multi-step reasoning...")
#     ms = multi_step_reasoning(state)
#     logging.info("Multi-step result: %s", ms)


    # If you want, I can add a retry wrapper around all LiteLLM calls in node.py 
    # and generation.py so it automatically backs off and retries on 429.



##################### New code ################################

# Function to trace the flow 
def add_trace(state, step, data):
    trace = state.get("trace", [])
    trace.append({"step": step, "data": data})
    return {"trace": trace}

# Adaptive RAG - Query router

def route_query(state: RAGState):
    logger.info("route_query:start trace_id=%s query=%s", _trace_id(state), _clip(state.get("query")))
    query = state["query"]
    q_norm = query.strip().lower()
    tokens = [t for t in q_norm.replace("?", "").split() if t]
    has_doc_context = bool(
        state.get("doc_name") or state.get("vectorstore") or state.get("bm25_index")
    )
    greetings = {
        "hi",
        "hello",
        "hey",
        "how are you",
        "how are you?",
        "good morning",
        "good afternoon",
        "good evening",
        "what's up",
        "whats up",
        "sup",
    }
    retrieval_cues = (
        "document",
        "doc",
        "pdf",
        "file",
        "dataset",
        "policy",
        "report",
        "section",
        "table",
        "figure",
        "page",
        "according to",
        "in this",
        "in the",
    )
    general_wh_starts = (
        "what is",
        "what's",
        "who is",
        "who's",
        "where is",
        "when is",
        "why is",
        "how is",
        "define ",
        "explain ",
    )
    fact_patterns = (
        q_norm.startswith("what is the capital of ")
        or q_norm.startswith("what's the capital of ")
        or q_norm.startswith("who is the president of ")
        or q_norm.startswith("who is president of ")
    )
    is_short = len(tokens) <= 12
    has_retrieval_cues = any(cue in q_norm for cue in retrieval_cues)
    general_knowledge = (
        any(q_norm.startswith(p) for p in general_wh_starts)
        and is_short
        and not has_retrieval_cues
        and not has_doc_context
    )

    if q_norm in greetings or fact_patterns or general_knowledge:
        logger.info(
            "route_query:decision trace_id=%s route=A reason=rule_based_simple_fact",
            _trace_id(state),
        )
        return {
            "complexity": "A",
            "original_query": query,
            **add_trace(state, "route_query", {"route": "A", "fallback": "rule_based_simple_fact"}),
        }
    if has_doc_context and any(q_norm.startswith(p) for p in general_wh_starts):
        logger.info(
            "route_query:decision trace_id=%s route=B reason=doc_context_prefers_retrieval",
            _trace_id(state),
        )
        return {
            "complexity": "B",
            "original_query": query,
            **add_trace(state, "route_query", {"route": "B", "fallback": "doc_context_prefers_retrieval"}),
        }
    cfg = _load_generation_config(model_name="generator-model")
    model = cfg.get("model")
    project_id = cfg.get("vertex_project")
    location = cfg.get("vertex_location")
    temperature = cfg.get("temperature", 0.0)
    max_tokens = cfg.get("max_tokens", 128)
    if not model or not project_id or not location:
        logger.info(
            "route_query:decision trace_id=%s route=B reason=missing_model_config",
            _trace_id(state),
        )
        return {
            "complexity": "B",
            "iteration": 0,
            "logs": ["Routing defaulted to B (missing model config)."],
            **add_trace(state, "route_query", {"route": "B", "fallback": "missing_config"}),
        }

    response = _safe_completion(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "### ROLE\n"
                    "You are an expert Query Classifier for an advanced Retrieval-Augmented "
                    "Generation (RAG) system. Your task is to analyze an incoming user query "
                    "and categorize it into one of three complexity levels: A, B, or C.\n\n"
                    "### CLASSIFICATION CRITERIA\n\n"
                    "- **Level A (Simple/Direct):**\n"
                    "    - Nature: Common knowledge, greetings, or basic factual questions "
                    "that do not require external documentation.\n"
                    "    - Indicators: The information is likely part of the LLM's core training "
                    "data (e.g., \"Who is the President of the US?\", \"What is the capital of "
                    "France?\", or \"Hello\").\n"
                    "    - Action: Route to direct generation.\n\n"
                    "- **Level B (Moderate/Single-Step):**\n"
                    "    - Nature: Requires specific, up-to-date, or private information found "
                    "in a single set of documents.\n"
                    "    - Indicators: Questions about specific company policies, recent news "
                    "events, or data points that require a single search query (e.g., "
                    "\"What is our company's Q3 remote work policy?\", "
                    "\"How many units were sold in the 2025 report?\").\n"
                    "    - Action: Route to the hybrid retrieval pipeline.\n\n"
                    "- **Level C (Complex/Multi-hop):**\n"
                    "    - Nature: Requires synthesizing information from multiple sources, "
                    "comparing different data points, or breaking the question into sub-steps.\n"
                    "    - Indicators: Multi-step reasoning, comparisons, or queries that "
                    "cannot be answered with a single search (e.g., "
                    "\"Compare the revenue growth of Product X in 2023 vs Product Y in 2024,\" "
                    "or \"Who is the CEO of the company that acquired Brand Z?\").\n"
                    "    - Action: Route to the multi-hop/agentic reasoning pipeline.\n\n"
                    "Return JSON {\"route\": \"A\"|\"B\"|\"C\"}."
                ),
            },
            {"role": "user", "content": query},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
        vertex_project=project_id,
        vertex_location=location,
        fallback_content='{"route":"B"}',
    )

    content = response["choices"][0]["message"]["content"] or ""
    used_fallback = bool(response.get("_fallback"))
    usage = _extract_usage(response)
    try:
        route = _extract_json_object(content).get("route", "B")
    except Exception:
        logger.warning(
            "route_query:parse_failed trace_id=%s default=B raw=%s",
            _trace_id(state),
            _clip(content),
        )
        route = "B"

    logger.info("route_query:decision trace_id=%s route=%s", _trace_id(state), route)
    return {
        "complexity": route,
        "original_query": query,
        **add_trace(state, "route_query", {"route": route, "fallback": used_fallback, "usage": usage}),
        **_merge_usage(state, usage, "route_query"),
    }

# Hybrid Retrieval

def retrieve(state: RAGState):
    query = state["query"]
    logger.info("retrieve:start trace_id=%s query=%s", _trace_id(state), _clip(query))
    vectorstore = state.get("vectorstore")
    bm25_index = state.get("bm25_index")
    reranker = state.get("reranker")
    if vectorstore is None or bm25_index is None:
        raise ValueError("vectorstore and bm25_index are required in state for retrieval.")

    hybrid_results = hybrid_retrieve(
        query=query,
        vectorstore=vectorstore,
        bm25_index=bm25_index,
        k_dense=None,
        k_sparse=None,
        top_n=None,
        metadata_filter=None,
    )
    docs_only = [doc for doc, _ in hybrid_results]
    if reranker is not None and docs_only:
        docs_only = rerank_docs(query, docs_only, reranker)
    score_map = {id(doc): score for doc, score in hybrid_results}
    retrieved = [
        {
            "text": doc.page_content,
            "metadata": doc.metadata,
            "hybrid_score": score_map.get(id(doc)),
        }
        for doc in docs_only
    ]

    return {
        "retrieved_docs": retrieved,
        **add_trace(state, "retrieve", {"query": query, "docs_found": len(retrieved)})
    }

# CRAG - Grade documents

def grade_documents(state: RAGState):
    query = state["query"]
    docs = state.get("retrieved_docs", [])
    logger.info(
        "grade_documents:start trace_id=%s docs_in=%d",
        _trace_id(state),
        len(docs),
    )

    relevant_docs = []
    used_fallback_any = False
    cfg = _load_generation_config(model_name="generator-model")
    model = cfg.get("model")
    project_id = cfg.get("vertex_project")
    location = cfg.get("vertex_location")
    temperature = cfg.get("temperature", 0.2)
    max_tokens = cfg.get("max_tokens", 256)
    if not model or not project_id or not location:
        logger.warning("grade_documents:missing_config trace_id=%s", _trace_id(state))
        return {
            "retrieved_docs": [],
            "run_web_search": "Yes",
            **add_trace(state, "grade_documents", {
                "relevant_docs": 0,
                "run_web": "Yes",
                "fallback": "missing_config",
            }),
        }

    usage_total = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    for doc in docs:
        prompt = f"""
        Query: {query}
        Document: {doc.get("text","")}

        Return JSON: {{"binary_score": "yes" or "no"}}
        """

        response = _safe_completion(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
            vertex_project=project_id,
            vertex_location=location,
            fallback_content='{"binary_score":"no"}',
        )
        raw = response["choices"][0]["message"]["content"] or ""
        usage = _extract_usage(response)
        usage_total = {
            "prompt_tokens": usage_total["prompt_tokens"] + usage["prompt_tokens"],
            "completion_tokens": usage_total["completion_tokens"] + usage["completion_tokens"],
            "total_tokens": usage_total["total_tokens"] + usage["total_tokens"],
        }
        used_fallback = bool(response.get("_fallback"))
        if used_fallback:
            used_fallback_any = True
        try:
            result = _extract_json_object(raw)
        except Exception:
            result = {"binary_score": "no"}

        if result["binary_score"] == "yes":
            relevant_docs.append(doc)
        if used_fallback:
            logging.warning("grade_documents used fallback response for doc scoring.")

    run_web = "Yes" if not relevant_docs else "No"
    logger.info(
        "grade_documents:done trace_id=%s relevant_docs=%d run_web=%s fallback=%s",
        _trace_id(state),
        len(relevant_docs),
        run_web,
        used_fallback_any,
    )

    return {
        "retrieved_docs": relevant_docs,
        "run_web_search": run_web,
        **add_trace(state, "grade_documents", {
            "relevant_docs": len(relevant_docs),
            "run_web": run_web,
            "fallback": used_fallback_any,
            "usage": usage_total,
        })
        ,
        **_merge_usage(state, usage_total, "grade_documents"),
    }

# Query rewrite

def transform_query(state: RAGState):
    query = state["query"]
    logger.info("transform_query:start trace_id=%s query=%s", _trace_id(state), _clip(query))

    prompt = (
        "Rewrite the user query to improve retrieval. "
        "Return ONLY the rewritten query as a single line of text. "
        "Do not add explanations, lists, or formatting.\n\n"
        f"User query: {query}"
    )

    new_query, used_fallback, usage = _call_llm_direct_with_meta(prompt)
    logger.info(
        "transform_query:done trace_id=%s new_query=%s fallback=%s",
        _trace_id(state),
        _clip(new_query),
        used_fallback,
    )
    if state.get("complexity") == "C":
        next_iteration = 0
    else:
        next_iteration = state.get("iteration", 0) + 1

    return {
        "query": new_query,
        "rewritten_query": new_query,
        "iteration": next_iteration,
        "attempts": state.get("attempts", 0) + 1,
        **add_trace(state, "transform_query", {
            "old_query": query,
            "new_query": new_query,
            "fallback": used_fallback,
            "usage": usage,
        })
        ,
        **_merge_usage(state, usage, "transform_query"),
    }


# generate_subquery.
def generate_subquery(state: RAGState):
    """Generate a follow-up sub-query for multi-hop retrieval (route C)."""
    original_query = state.get("original_query") or state["query"]
    docs = state.get("retrieved_docs", [])
    context = "\n".join([d.get("text", "") for d in docs])
    if len(context) > 2000:
        context = context[:2000]

    prompt = (
        "You are performing multi-hop retrieval. "
        "Given the original question and current context, "
        "produce ONE focused follow-up search query to retrieve missing info. "
        "Return only the query text.\n\n"
        f"Original Question: {original_query}\n"
        f"Current Context:\n{context}"
    )

    sub_query, used_fallback, usage = _call_llm_direct_with_meta(prompt)
    logger.info(
        "generate_subquery:done trace_id=%s sub_query=%s fallback=%s",
        _trace_id(state),
        _clip(sub_query),
        used_fallback,
    )

    return {
        "query": sub_query,
        "sub_query": sub_query,
        "iteration": state.get("iteration", 0) + 1,
        "attempts": state.get("attempts", 0) + 1,
        **add_trace(state, "generate_subquery", {
            "original_query": original_query,
            "sub_query": sub_query,
            "fallback": used_fallback,
            "usage": usage,
        }),
        **_merge_usage(state, usage, "generate_subquery"),
    }


# final_fail.
def final_fail(state: RAGState):
    last_answer = state.get("answer") or state.get("final_answer")
    logger.warning("final_fail trace_id=%s reason=max_attempts_exceeded", _trace_id(state))
    if last_answer:
        return {
            "answer": last_answer,
            "unverified": True,
            **add_trace(state, "final_fail", {
                "reason": "max_attempts_exceeded",
                "returned_last_answer": True,
            }),
        }
    message = "I could not find enough grounded information after multiple attempts."
    return {
        "answer": message,
        "unverified": True,
        **add_trace(state, "final_fail", {"reason": "max_attempts_exceeded"}),
    }

# Web search

def web_search(state: RAGState):
    query = state["query"]
    if state.get("run_web_search") != "Yes":
        return {}

    cfg = _load_generation_config(model_name="generator-model")
    model = cfg.get("model")
    project_id = cfg.get("vertex_project")
    location = cfg.get("vertex_location")
    temperature = cfg.get("temperature", 0.2)
    max_tokens = cfg.get("max_tokens", 512)
    if not model or not project_id or not location:
        return {
            "web_results": [],
            "logs": ["Web search skipped (missing model config)."],
            **add_trace(state, "web_search", {"query": query, "fallback": "missing_config"}),
        }

    response = _safe_completion(
        model=model,
        messages=[{"role": "user", "content": query}],
        tools=[{"google_search": {}}],
        temperature=temperature,
        max_tokens=max_tokens,
        vertex_project=project_id,
        vertex_location=location,
        fallback_content="",
    )

    content = response["choices"][0]["message"]["content"]
    used_fallback = bool(response.get("_fallback"))
    usage = _extract_usage(response)

    return {
        "web_results": [{"content": content}],
        **add_trace(state, "web_search", {"query": query, "fallback": used_fallback, "usage": usage}),
        **_merge_usage(state, usage, "web_search"),
    }

# Generate from retrieved documents

def generate_from_docs(state: RAGState):
    docs = state.get("retrieved_docs", [])
    query = state["query"]
    logger.info(
        "generate_docs:start trace_id=%s docs=%d query=%s",
        _trace_id(state),
        len(docs),
        _clip(query),
    )

    context = "\n".join([d.get("text", "") for d in docs])

    prompt = (
        "You are answering a user question using the provided context.\n"
        "Give a detailed, multi-paragraph answer that explains the concept clearly.\n"
        "Include key points and any relevant nuance from the context.\n"
        "Do not discuss query rewriting, retrieval strategies, or the prompt itself.\n"
        "If the context does not contain the answer, say you don't know.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {query}"
    )

    answer, used_fallback, usage = _call_llm_direct_with_meta(prompt)
    logger.info(
        "generate_docs:done trace_id=%s fallback=%s",
        _trace_id(state),
        used_fallback,
    )

    return {
        "answer": answer,
        **add_trace(state, "generate_docs", {"answer": answer, "fallback": used_fallback, "usage": usage}),
        **_merge_usage(state, usage, "generate_docs"),
    }

# Generate from web

def generate_from_web(state: RAGState):
    web = state.get("web_results", [])
    query = state["query"]

    context = "\n".join([w.get("content", "") for w in web])

    prompt = (
        "You are answering a user question using the provided web context.\n"
        "Give a detailed, multi-paragraph answer that explains the concept clearly.\n"
        "Include key points and any relevant nuance from the context.\n"
        "Do not discuss query rewriting, retrieval strategies, or the prompt itself.\n"
        "If the context does not contain the answer, say you don't know.\n\n"
        f"Web Context:\n{context}\n\n"
        f"Question: {query}"
    )

    answer, used_fallback, usage = _call_llm_direct_with_meta(prompt)

    return {
        "answer": answer,
        **add_trace(state, "generate_web", {"answer": answer, "fallback": used_fallback, "usage": usage}),
        **_merge_usage(state, usage, "generate_web"),
    }

# SELF RAG check

def self_rag_check(state: RAGState):
    query = state["query"]
    answer = state["answer"]
    docs = state.get("retrieved_docs", [])
    logger.info(
        "self_rag:start trace_id=%s docs=%d",
        _trace_id(state),
        len(docs),
    )

    hallucination, hallucination_fallback, hallucination_raw, hallucination_usage = hallucination_grader(docs, answer)
    relevance, relevance_fallback, relevance_raw, relevance_usage = answer_grader(query, answer)

    decision = "useful"

    if hallucination == "no":
        decision = "not supported"
    elif relevance == "no":
        decision = "not useful"

    logger.info(
        "self_rag:decision trace_id=%s decision=%s hallucination=%s relevance=%s",
        _trace_id(state),
        decision,
        hallucination,
        relevance,
    )
    return {
        "hallucination_score": hallucination,
        "answer_relevance_score": relevance,
        **add_trace(state, "self_rag", {
            "hallucination": hallucination,
            "relevance": relevance,
            "decision": decision,
            "hallucination_raw": hallucination_raw,
            "relevance_raw": relevance_raw,
            "fallback": hallucination_fallback or relevance_fallback,
            "usage": {
                "prompt_tokens": hallucination_usage["prompt_tokens"] + relevance_usage["prompt_tokens"],
                "completion_tokens": hallucination_usage["completion_tokens"] + relevance_usage["completion_tokens"],
                "total_tokens": hallucination_usage["total_tokens"] + relevance_usage["total_tokens"],
            },
        }),
        "decision": decision,
        **_merge_usage(
            state,
            {
                "prompt_tokens": hallucination_usage["prompt_tokens"] + relevance_usage["prompt_tokens"],
                "completion_tokens": hallucination_usage["completion_tokens"] + relevance_usage["completion_tokens"],
                "total_tokens": hallucination_usage["total_tokens"] + relevance_usage["total_tokens"],
            },
            "self_rag",
        ),
    }
