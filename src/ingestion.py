import os
import json, logging
from pathlib import Path
from datetime import datetime
import hashlib
from docling.document_converter import DocumentConverter

# Avoid Windows symlink permission errors when downloading HF models
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS", "1")


logging.basicConfig(level=logging.INFO)

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
REGISTRY_FILE = "file_hash_registry.json"


# Standardize file paths to prevent duplicates due to different path formats
def _normalize_path(path: str) -> str:
    return os.path.normcase(os.path.abspath(path))



# Load existing hashes from a file or database to check for duplicates
def load_registry():
    if not os.path.exists(REGISTRY_FILE):
        return {}

    with open(REGISTRY_FILE, "r") as f:
        try:
            data = json.load(f)
            return data if isinstance(data, dict) else {}
        except json.JSONDecodeError:
            return {}
    
# Save registry to a file or database after processing a new file
def save_registry(registry):
    with open(REGISTRY_FILE, "w") as f:
        json.dump(registry, f, indent=2)

# Duplicate check using file hash
def is_duplicate(file_hash, registry):
    return file_hash in registry


# generate_file_hash.
def generate_file_hash(file_path):
    hasher = hashlib.sha256()

    with open(file_path, "rb") as f:
        while chunk := f.read(8192):
            hasher.update(chunk)

    return hasher.hexdigest()

# Add new file hash to registry after successful processing with timestamp
def add_file_hash(file_hash, file_name, registry):
    registry[file_hash] = {
        "file_name": file_name,
        "processed_at": datetime.utcnow().isoformat()
    }

# document_exists.
def document_exists(file_path: str) -> bool:
    """Check if a document with the same file path already exists using hashing."""
    normalized_path = _normalize_path(file_path)
    

    file_hash = generate_file_hash(file_path)
    registry = load_registry()

    if is_duplicate(file_hash, registry):
        logging.info(f"Duplicate document detected: {file_path} (hash: {file_hash})")
        return True 
    else:
        return False




# Document ingestion function that checks for duplicates and processes the document if it's new
def document_ingestion(file_path: str) -> list:
    """Ingest a document using docling and convert the document into a JSON format so that it can be chunked and saved into the vector database."""
    if document_exists(file_path):
        logging.info(f"Document already exists: {file_path}")
        return []

    # Initialize converter
    converter = DocumentConverter()

    # Convert document
    result = converter.convert(file_path)

    # Extract structured document
    doc = result.document

    # Export JSON
    doc_json = doc.export_to_dict()

    # Save JSON file
    with open("doc_output.json", "w", encoding="utf-8") as f:
        json.dump(doc_json, f, indent=2)

    logging.info("Document converted successfully.")

    registry = load_registry()
    file_hash = generate_file_hash(file_path)
    add_file_hash(file_hash, os.path.basename(file_path), registry)
    save_registry(registry)

    return doc_json



#####################   Demo / Testing Code   #####################

if __name__ == "__main__":
    import sys

    default_path = DATA_DIR / "SELF_RAG.pdf"
    file_arg = sys.argv[1] if len(sys.argv) > 1 else str(default_path)
    if not os.path.exists(file_arg):
        logging.error("File not found: %s", file_arg)
        logging.info("Usage: python src/ingestion.py <file_path>")
        raise SystemExit(1)

    output = document_ingestion(file_arg)
    if output:
        logging.info("JSON output ready and saved to doc_output.json")
