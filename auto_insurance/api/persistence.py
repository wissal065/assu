"""Optional SQLite persistence for prediction audit logs."""

from __future__ import annotations

import json
import logging
import os
import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class PredictionAuditRepository:
    """Stores successful predictions in a local SQLite database."""

    def __init__(self, enabled: bool, db_path: str) -> None:
        self.enabled = enabled
        self.db_path = Path(db_path)
        if self.enabled:
            self._init_db()

    def _connect(self) -> sqlite3.Connection:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        return connection

    def _init_db(self) -> None:
        with closing(self._connect()) as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS prediction_audit (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT NOT NULL,
                    endpoint TEXT NOT NULL,
                    request_payload TEXT NOT NULL,
                    response_payload TEXT NOT NULL,
                    niveau_risque TEXT
                )
                """
            )
            connection.commit()
        logger.info("Prediction audit database ready at %s", self.db_path)

    def save_prediction(
        self,
        endpoint: str,
        request_payload: dict[str, Any],
        response_payload: dict[str, Any],
        niveau_risque: str | None = None,
    ) -> None:
        """Persist a successful prediction when audit storage is enabled."""
        if not self.enabled:
            return

        with closing(self._connect()) as connection:
            connection.execute(
                """
                INSERT INTO prediction_audit (
                    created_at,
                    endpoint,
                    request_payload,
                    response_payload,
                    niveau_risque
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (
                    datetime.now(timezone.utc).isoformat(),
                    endpoint,
                    json.dumps(request_payload, ensure_ascii=False),
                    json.dumps(response_payload, ensure_ascii=False),
                    niveau_risque,
                ),
            )
            connection.commit()

    def get_status(self) -> dict[str, Any]:
        """Return audit storage status for health checks."""
        status: dict[str, Any] = {
            "enabled": self.enabled,
            "database_path": str(self.db_path),
        }
        if not self.enabled:
            status["records"] = 0
            return status

        with closing(self._connect()) as connection:
            record_count = connection.execute(
                "SELECT COUNT(*) FROM prediction_audit"
            ).fetchone()[0]
        status["records"] = int(record_count)
        return status


def build_audit_repository() -> PredictionAuditRepository:
    """Create an audit repository from environment variables."""
    enabled = os.getenv("PREDICTION_AUDIT_ENABLED", "false").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    db_path = os.getenv("PREDICTION_AUDIT_DB_PATH", "data/prediction_audit.db")
    return PredictionAuditRepository(enabled=enabled, db_path=db_path)
