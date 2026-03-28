import uuid
import logging
from typing import List, Dict, Any

from config_loader import get_rag_config

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO)

class DoclingParser:
    # __init__.
    def __init__(self, doc_json: Dict[str, Any]):
        self.doc = doc_json
        self.texts = doc_json.get("texts", [])
        self.tables = doc_json.get("tables", [])
        self.groups = doc_json.get("groups", [])
        self.body = doc_json.get("body", {})

    # -----------------------------
    # Resolve $ref
    # -----------------------------
    def resolve_ref(self, ref: str) -> Dict:
        """
        Example ref: '#/texts/12'
        """
        parts = ref.strip("#/").split("/")
        obj = self.doc

        for part in parts:
            if part.isdigit():
                obj = obj[int(part)]
            else:
                obj = obj.get(part, {})

        return obj

    # -----------------------------
    # Extract ordered nodes
    # -----------------------------
    def get_ordered_nodes(self) -> List[Dict]:
        nodes = []
        for child in self.body.get("children", []):
            if "$ref" in child:
                resolved = self.resolve_ref(child["$ref"])
                nodes.append(resolved)
        return nodes

    # -----------------------------
    # Build sections (parents)
    # -----------------------------
    def build_sections(self) -> List[Dict]:
        nodes = self.get_ordered_nodes()

        sections = []
        current_section = None

        for node in nodes:
            label = node.get("label")
            text = node.get("text", "").strip()

            # New Parent
            if label == "section_header":
                if current_section:
                    sections.append(current_section)

                current_section = {
                    "id": str(uuid.uuid4()),
                    "title": text,
                    "content": []
                }

            # Child Content
            elif label in ["text", "code", "list_item", "caption", "formula", "footnote"]:
                if current_section and text:
                    prov = node.get("prov") or []
                    page_numbers = []
                    for p in prov:
                        page_no = p.get("page_no")
                        if isinstance(page_no, int):
                            page_numbers.append(page_no)
                    current_section["content"].append(
                        {
                            "text": text.strip(),
                            "page_number": min(page_numbers) if page_numbers else None,
                        }
                    )

        # section add
        if current_section:
            sections.append(current_section)

        return sections


# --------------------------------------
#  Child Chunking
# --------------------------------------
def chunk_text(text: str, chunk_size: int | None = None, overlap: int | None = None) -> List[str]:
    cfg = get_rag_config()
    if chunk_size is None:
        chunk_size = cfg.get("chunk_size", 300)
    if overlap is None:
        overlap = cfg.get("overlap", 50)
    words = text.split()
    chunks = []

    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = words[start:end]
        chunks.append(" ".join(chunk))

        start += chunk_size - overlap

    return chunks


# --------------------------------------
#  Create parent-child chunks
# --------------------------------------

def create_parent_child_chunks(
    sections: List[Dict],
    doc_name: str,
    max_words: int | None = None
) -> List[Dict]:
    """
    Creates child chunks within each section (parent) using paragraph-aware chunking.
    """

    if max_words is None:
        max_words = get_rag_config().get("max_words", 300)

    all_chunks = []

    for section in sections:
        parent_id = section["id"]
        parent_title = section["title"]
        paragraphs = section["content"]  # list of paragraphs or dicts with page info

        current_chunk = []
        current_length = 0
        chunk_index = 0
        current_pages: List[int] = []

        for para in paragraphs:
            if isinstance(para, dict):
                para_text = para.get("text", "").strip()
                page_number = para.get("page_number")
                para_pages = [page_number] if isinstance(page_number, int) else []
            else:
                para_text = str(para).strip()
                para_pages = []

            if not para_text:
                continue

            para_len = len(para_text.split())

            # If paragraph itself is too large → split it
            if para_len > max_words:
                if current_chunk:
                    all_chunks.append({
                        "id": str(uuid.uuid4()),
                        "text": " ".join(current_chunk),
                        "metadata": {
                            "parent_id": parent_id,
                            "parent_title": parent_title,
                            "chunk_index": chunk_index,
                            "document": doc_name,
                            "page_number": min(current_pages) if current_pages else None,
                        }
                    })
                    chunk_index += 1
                    current_chunk = []
                    current_length = 0
                    current_pages = []

                words = para_text.split()

                for i in range(0, len(words), max_words):
                    sub_chunk = " ".join(words[i:i + max_words])

                    all_chunks.append({
                        "id": str(uuid.uuid4()),
                        "text": sub_chunk,
                        "metadata": {
                            "parent_id": parent_id,
                            "parent_title": parent_title,
                            "chunk_index": chunk_index,
                            "document": doc_name,
                            "page_number": para_pages[0] if para_pages else None,
                        }
                    })

                    chunk_index += 1

                continue

            # If adding paragraph exceeds limit → finalize current chunk
            if current_length + para_len > max_words:
                all_chunks.append({
                    "id": str(uuid.uuid4()),
                    "text": " ".join(current_chunk),
                    "metadata": {
                        "parent_id": parent_id,
                        "parent_title": parent_title,
                        "chunk_index": chunk_index,
                        "document": doc_name,
                        "page_number": min(current_pages) if current_pages else None,
                    }
                })

                chunk_index += 1
                current_chunk = []
                current_length = 0
                current_pages = []

            # Add paragraph to chunk
            current_chunk.append(para_text)
            current_length += para_len
            if para_pages:
                current_pages.extend(para_pages)

        # Add remaining chunk
        if current_chunk:
            all_chunks.append({
                "id": str(uuid.uuid4()),
                "text": " ".join(current_chunk),
                "metadata": {
                    "parent_id": parent_id,
                    "parent_title": parent_title,
                    "chunk_index": chunk_index,
                    "document": doc_name,
                    "page_number": min(current_pages) if current_pages else None,
                }
            })

    return all_chunks

# --------------------------------------
# Demo
# --------------------------------------
if __name__ == "__main__":
    import json
    import os
    import sys

    if len(sys.argv) > 1:
        json_path = sys.argv[1]
    else:
        default_in_src = os.path.join("src", "doc_output.json")
        json_path = default_in_src if os.path.exists(default_in_src) else "doc_output.json"

    with open(json_path, "r", encoding="utf-8") as f:
        doc_json = json.load(f)

    parser = DoclingParser(doc_json)

    # Step 1: Build sections (parents)
    sections = parser.build_sections()

    logger.info("Total sections: %d", len(sections))

    # Optional: Table info (testing)
    tables = doc_json.get("tables", [])
    logger.info("Total tables: %d", len(tables))
    if tables:
        logger.info("Sample table metadata: %s", {
            "self_ref": tables[0].get("self_ref"),
            "label": tables[0].get("label"),
            "content_layer": tables[0].get("content_layer"),
        })

    # Step 2: Create parent-child chunks
    doc_name = (
        doc_json.get("origin", {}).get("filename")
        or doc_json.get("name")
        or os.path.basename(json_path)
    )
    chunks = create_parent_child_chunks(sections, doc_name=doc_name)

    logger.info("Total chunks: %d", len(chunks))

    # Sample output
    if chunks:
        logger.info("Last chunk: %s", chunks[-1])
    else:
        logger.warning("No chunks created. Check section extraction for this JSON.")

    # Step 3: Retrieve parent section for each child chunk (demo)
    parent_store = {
        section["id"]: {
            "title": section["title"],
            "text": " ".join(
                c["text"] if isinstance(c, dict) else str(c)
                for c in section["content"]
            ),
        }
        for section in sections
    }

    for idx, chunk in enumerate(chunks[:5]):
        metadata = chunk.get("metadata", {})
        parent_id = metadata.get("parent_id", "")
        parent = parent_store.get(parent_id)

        logger.info("Child chunk %d", idx)
        logger.info("Parent title: %s", parent.get("title") if parent else "NOT FOUND")
        logger.info("Child text: %s", chunk.get("text", "")[:200])
        logger.info("Parent text: %s", (parent.get("text") if parent else "")[:200])
        logger.info("Page number: %s", metadata.get("page_number"))
