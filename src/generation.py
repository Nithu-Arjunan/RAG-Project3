from typing import List, Dict
from pathlib import Path
import os
import logging

import yaml
from litellm import completion
from logging_config import configure_logging

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover - optional dependency
    load_dotenv = None

ROOT = Path(__file__).resolve().parents[1]
configure_logging()
logger = logging.getLogger(__name__)


# _load_env_file.
def _load_env_file(path: Path) -> None:
    if not path.exists():
        return
    try:
        raw = path.read_text(encoding="utf-8")
    except Exception:
        return
    for line in raw.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        key = key.strip().lstrip("\ufeff")
        value = value.strip().strip("\"").strip("'")
        if key:
            os.environ.setdefault(key, value)

# from .reranking import rerank
# from .base import Chunk

system_prompt = """
You are a helpful assistant that provides answers based only on the provided context.

STRICT RULES:
1. Use ONLY the information present in the provided CONTEXT.
2. Do NOT invent or assume missing values.
3. If any part of the answer is incomplete in the context, state:
   "These are the information present in the document."
4. When answering:
   - Preserve full tables exactly as written.
   - Do not truncate rows.
   - Present structured data clearly in table format.
5. If answer is not found, respond exactly:
   "Not found in document."
6. Do NOT summarize unless explicitly asked.
7. Ensure the answer is complete and fully readable before finishing.
"""


# _load_generation_config.
def _load_generation_config(model_name: str = "generator-model") -> Dict:
    env_path = ROOT / ".env"
    if load_dotenv is not None:
        load_dotenv(dotenv_path=env_path, override=False)
    _load_env_file(env_path)
    config_path = Path(__file__).resolve().parent / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    model_list = data.get("model_list", [])
    selected = None
    for m in model_list:
        if m.get("model_name") == model_name:
            selected = m.get("litellm_params", {})
            break

    if selected is None:
        return {}

    # Resolve os.getenv(...) strings if present
    def _resolve(value):
        if isinstance(value, str) and value.startswith("os.getenv("):
            env_key = value.split("\"")[1]
            return os.getenv(env_key)
        return value

    resolved = {k: _resolve(v) for k, v in selected.items()}
    logger.info(
        "generation_config:loaded model_name=%s model_set=%s project_set=%s location_set=%s",
        model_name,
        bool(resolved.get("model")),
        bool(resolved.get("vertex_project")),
        bool(resolved.get("vertex_location")),
    )
    return resolved


# _format_chunks.
def _format_chunks(chunks: List[Dict]) -> str:
    parts: List[str] = []
    for i, c in enumerate(chunks, start=1):
        metadata = c.get("metadata", {}) if isinstance(c, dict) else {}
        text = c.get("chunk_text", c.get("text", ""))

        parent_id = metadata.get("parent_id", "")
        parent_title = metadata.get("parent_title", "")
        chunk_index = metadata.get("chunk_index", "")
        document = metadata.get("document", metadata.get("source", ""))
        page_number = metadata.get("page_number", metadata.get("page", ""))

        parts.append(
            f"[{i}] document={document} page={page_number} "
            f"parent_title={parent_title} parent_id={parent_id} "
            f"chunk_index={chunk_index}\n{text}"
        )
    return "\n\n".join(parts)


# generate_answer.
def generate_answer(query: str, chunks: List[Dict], model_name: str = "generator-model") -> str:
    query = query.strip()
    if not query:
        return "Empty query provided."
    if not chunks:
        return "No relevant information found in the documents."

    cfg = _load_generation_config(model_name=model_name)
    model = cfg.get("model")
    project_id = cfg.get("vertex_project")
    location = cfg.get("vertex_location")
    temperature = cfg.get("temperature", 0.2)
    max_tokens = cfg.get("max_tokens", 1024)

    if not model or not project_id or not location:
        return "Model configuration missing in config.yaml."

    prompt = f"""{system_prompt}

CONTEXT:
{_format_chunks(chunks)}

USER QUERY:
{query}
"""

    response = completion(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
        vertex_project=project_id,
        vertex_location=location,
    )

    return response.choices[0].message["content"]


if __name__ == "__main__":
    import json
    import sys
    from pathlib import Path
    from pipeline import run_hybrid_rerank_from_json

    if len(sys.argv) > 1:
        json_path = Path(sys.argv[1]).resolve()
    else:
        json_path = Path("doc_output.json").resolve()

    if not json_path.exists():
        logger.error("JSON file not found: %s", json_path)
        raise SystemExit(1)

    query = "How is SELF RAG training?"
    with open(json_path, "r", encoding="utf-8") as f:
        doc_json = json.load(f)

    retrieved = run_hybrid_rerank_from_json(
        doc_json=doc_json,
        doc_name=None,
        query=query,
        k_dense=10,
        k_sparse=10,
        top_n=5,
        reranker=None,
    )

    formatted_context = _format_chunks(retrieved)
    formatted_chunks = [{"text": formatted_context}]
    answer = generate_answer(query, formatted_chunks, model_name="generator-model")
    logger.info("Answer: %s", answer)
