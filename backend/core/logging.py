"""
Structured JSON logging for the Christianity-Focused AI Assistant.
Provides a consistent logging interface with contextual metadata.
"""

import json
import logging
import sys
import traceback
from datetime import datetime, timezone
from typing import Any


class JSONFormatter(logging.Formatter):
    """
    Formats log records as structured JSON for log aggregation tools
    (e.g., Datadog, CloudWatch, Loki).
    """

    def format(self, record: logging.LogRecord) -> str:
        log_data: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Attach extra context fields
        if hasattr(record, "extra"):
            log_data.update(record.extra)  # type: ignore[arg-type]

        # Attach exception info if present
        if record.exc_info:
            log_data["exception"] = {
                "type": record.exc_info[0].__name__ if record.exc_info[0] else None,
                "message": str(record.exc_info[1]),
                "traceback": traceback.format_exception(*record.exc_info),
            }

        return json.dumps(log_data, default=str)


def setup_logging(level: str = "INFO") -> logging.Logger:
    """
    Configure and return the root application logger.

    Args:
        level: Log level string (DEBUG, INFO, WARNING, ERROR).

    Returns:
        Configured root logger.
    """
    root_logger = logging.getLogger("christian_ai")
    root_logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Avoid duplicate handlers on hot-reload
    if root_logger.handlers:
        return root_logger

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JSONFormatter())
    root_logger.addHandler(handler)

    # Suppress noisy third-party loggers
    for noisy in ("httpx", "httpcore", "anthropic", "openai", "pinecone"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    return root_logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a named child logger. All children inherit the root configuration.

    Usage:
        logger = get_logger(__name__)
        logger.info("Processing query", extra={"extra": {"session_id": "abc"}})
    """
    return logging.getLogger(f"christian_ai.{name}")
