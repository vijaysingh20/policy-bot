from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
STORAGE_DIR = PROJECT_ROOT / "storage"
DATABASE_DIR = PROJECT_ROOT / "db"

HANDBOOK_PDF = DATA_DIR / "handbook.pdf"
INDEX_NAME = "handbook-v1"
INDEX_DIR = STORAGE_DIR / INDEX_NAME

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
