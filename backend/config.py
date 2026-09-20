import os
from pathlib import Path

def csv_env(name: str, default: str) -> list[str]:
    """Read a comma-separated environment variable into a clean list."""
    return [item.strip().rstrip("/") for item in os.getenv(name, default).split(",") if item.strip()]

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
STORAGE_DIR = PROJECT_ROOT / "storage"
DATABASE_DIR = PROJECT_ROOT / "db"

HANDBOOK_PDF = DATA_DIR / "handbook.pdf"
INDEX_NAME = "handbook-v1"
INDEX_DIR = STORAGE_DIR / INDEX_NAME

EVALUATION_LOG = PROJECT_ROOT / "logs" / "evaluations.jsonl"

UPLOAD_DIR = PROJECT_ROOT / "uploads"
MAX_UPLOAD_BYTES = 10 * 1024 * 1024

# Browsers may only call the API from these origins (set CORS_ORIGINS in production).
CORS_ORIGINS = csv_env("CORS_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000")

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"
DEVICE = "cpu"

CHAT_MODEL = "gpt-4.1-mini"
CHAT_TIMEOUT_SECONDS = 30
EVALUATION_TIMEOUT_SECONDS = 60
MAX_RETRIES = 1

CHUNK_SIZE = 200
CHUNK_OVERLAP = 40

CANDIDATE_TOP_K = 20
FINAL_TOP_K = 3
