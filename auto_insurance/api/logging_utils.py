"""Utilities for application logging."""

from __future__ import annotations

import logging
import os


def setup_logging() -> None:
    """Configure a simple application-wide logging format."""
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        force=True,
    )
