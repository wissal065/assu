"""
Structured logging configuration for the AutoAssur API.

Produces JSON-formatted logs, compatible with monitoring tools
such as ELK Stack, Datadog, CloudWatch, etc.

Example log output:
{
  "timestamp": "2024-01-15T10:23:45.123Z",
  "level": "INFO",
  "logger": "auto_insurance.api.endpoints.predict",
  "message": "Prediction successful",
  "endpoint": "/predict/premium",
  "method": "POST",
  "status_code": 200,
  "latency_ms": 42.7,
  "model_version": "v1.0"
}
"""

import json
import logging
import sys
from datetime import datetime, timezone


class JSONFormatter(logging.Formatter):
    """
    Custom log formatter that serializes each log record as a single JSON line.
    Compatible with log aggregation systems (ELK, Datadog, etc.).
    """

    def format(self, record: logging.LogRecord) -> str:
        log_entry: dict = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Append business-level fields if injected via logger.info(..., extra={...})
        for field in (
            "endpoint",
            "method",
            "status_code",
            "latency_ms",
            "model_version",
            "vehicle_brand",
            "error_type",
            "error_detail",
            "request_id",
        ):
            if hasattr(record, field):
                log_entry[field] = getattr(record, field)

        # Append exception traceback if present
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_entry, ensure_ascii=False)


def setup_logging(level: str = "INFO") -> None:
    """
    Configure application-wide logging.

    Must be called once at startup, in main.py, before any module
    creates a logger. The log level can be overridden via the
    LOG_LEVEL environment variable.

    Args:
        level: Log level string ("DEBUG", "INFO", "WARNING", "ERROR").
    """
    # Stream handler writing to stdout (captured by Docker, Render, etc.)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JSONFormatter())

    # Root logger — all application loggers inherit from it
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    root_logger.handlers.clear()
    root_logger.addHandler(handler)

    # Reduce noise from third-party libraries
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.error").setLevel(logging.WARNING)
    logging.getLogger("shap").setLevel(logging.WARNING)
