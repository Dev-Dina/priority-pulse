"""
Central Configuration Management
"""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# Project root: backend/app/config.py → go up 2 levels → backend/
PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """Application settings loaded from .env"""

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )

    # ── App ───────────────────────────────────────────────────────
    APP_NAME: str = "Priority Pulse"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False
    API_V1_PREFIX: str = "/api/v1"
    BACKEND_PORT: int = 8000

    # ── Qdrant (vector store) ─────────────────────────────────────
    # Inside Docker Compose the host name matches the service name.
    # Locally it's localhost.
    QDRANT_HOST: str = "localhost"
    QDRANT_PORT: int = 6333
    QDRANT_COLLECTION: str = "tickets"

    # ── LLM ───────────────────────────────────────────────────────
    GEMINI_API_KEY: str | None = None
    LLM_MODEL: str = "gemini-2.0-flash-lite"
    LLM_MAX_TOKENS: int = 1024
    LLM_TEMPERATURE: float = 0.7

    # ── LLM fallback (Groq — OpenAI-compatible API) ───────────────
    # Groq serves only open-weight models (e.g. Llama, gpt-oss), so the
    # fallback's behaviour and answer quality differ from Gemini-proper.
    GROQ_API_KEY: str | None = None
    GROQ_BASE_URL: str = "https://api.groq.com/openai/v1"
    GROQ_MODEL: str = "llama-3.3-70b-versatile"

    # ── Embeddings ────────────────────────────────────────────────
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    EMBEDDING_DIM: int = 384

    # ── RAG ───────────────────────────────────────────────────────
    RAG_TOP_K: int = 5
    RAG_SIMILARITY_THRESHOLD: float = 0.3

    # ── Models ────────────────────────────────────────────────────
    MODEL_DIR: Path = PROJECT_ROOT / "models"

    # ── Data paths ────────────────────────────────────────────────
    DATA_RAW_DIR: Path = PROJECT_ROOT / "data" / "raw"
    DATA_PROCESSED_DIR: Path = PROJECT_ROOT / "data" / "processed"
    DATA_EMBEDDINGS_DIR: Path = PROJECT_ROOT / "data" / "embeddings"

    RAW_DATASET_PATH: Path = DATA_RAW_DIR / "twcs.csv"
    CLEANED_DATASET_PATH: Path = DATA_PROCESSED_DIR / "cleaned_tickets.csv"
    LABELED_DATASET_PATH: Path = DATA_PROCESSED_DIR / "labeled_tickets.csv"
    ML_FEATURES_PATH: Path = DATA_PROCESSED_DIR / "ml_features.csv"

    # ── Logging ───────────────────────────────────────────────────
    LOGS_DIR: Path = PROJECT_ROOT / "logs"
    APP_LOG_PATH: Path = LOGS_DIR / "app.log"
    DATA_LOG_PATH: Path = LOGS_DIR / "data_processing.log"
    ML_LOG_PATH: Path = LOGS_DIR / "ml_training.log"
    QUERY_LOG_PATH: Path = LOGS_DIR / "queries.jsonl"

    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "text"      # "text" or "json"
    LOG_TO_FILE: bool = True
    LOG_TO_DB: bool = False       # DB logging removed; JSONL is the audit trail


settings = Settings()

# Ensure runtime directories exist
settings.LOGS_DIR.mkdir(parents=True, exist_ok=True)
settings.DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
settings.DATA_EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
settings.MODEL_DIR.mkdir(parents=True, exist_ok=True)
