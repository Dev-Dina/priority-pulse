"""
Structured Logging System

All query audit logs go to a JSONL file (one JSON object per line).
Set LOG_TO_FILE=true in .env — that is the only persistence layer needed.
"""

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from loguru import logger

from app.config import settings


class StructuredLogger:
    """Multi-output logger: console (coloured) + rotating file."""

    def __init__(self, module_name: str, log_path: Optional[Path] = None):
        self.module_name = module_name
        self.log_path = log_path or settings.APP_LOG_PATH
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._configure_logger()

    def _configure_logger(self):
        logger.remove()

        # Console — pretty for development
        logger.add(
            sys.stdout,
            format=(
                "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
                "<level>{level: <8}</level> | "
                "<cyan>{extra[module]}</cyan> | "
                "<level>{message}</level>"
            ),
            level=settings.LOG_LEVEL,
            colorize=True,
        )

        # File — rotating, compressed
        if settings.LOG_TO_FILE:
            logger.add(
                self.log_path,
                format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {extra[module]} | {message}",
                level=settings.LOG_LEVEL,
                rotation="100 MB",
                retention="30 days",
                compression="zip",
                serialize=(settings.LOG_FORMAT == "json"),
            )

        logger.configure(extra={"module": self.module_name})

    def info(self, message: str, **kwargs):
        logger.bind(module=self.module_name).info(message, **kwargs)

    def error(self, message: str, **kwargs):
        logger.bind(module=self.module_name).error(message, **kwargs)

    def warning(self, message: str, **kwargs):
        logger.bind(module=self.module_name).warning(message, **kwargs)

    def debug(self, message: str, **kwargs):
        logger.bind(module=self.module_name).debug(message, **kwargs)

    def log_query(self, query_data: Dict[str, Any]) -> None:
        """
        Append one query audit record to queries.jsonl.

        Each line is a self-contained JSON object with timestamp + all four
        outputs (RAG, non-RAG, ML, LLM zero-shot) so you can replay or
        analyse the log without touching the live system.
        """
        if not settings.LOG_TO_FILE:
            return

        record = {"timestamp": datetime.utcnow().isoformat(), **query_data}
        with open(settings.QUERY_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


# Module-specific logger instances
app_logger = StructuredLogger("app", settings.APP_LOG_PATH)
data_logger = StructuredLogger("data_processing", settings.DATA_LOG_PATH)
ml_logger = StructuredLogger("ml_training", settings.ML_LOG_PATH)
