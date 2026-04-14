"""FastAPI dependency providers for the application."""

from auto_insurance.api.persistence import (
    PredictionAuditRepository,
    build_audit_repository,
)
from auto_insurance.src.pipeline import PredictionPipeline

_pipeline: PredictionPipeline | None = None
_audit_repository: PredictionAuditRepository | None = None


def get_pipeline() -> PredictionPipeline:
    """Return the prediction pipeline singleton."""
    global _pipeline
    if _pipeline is None:
        _pipeline = PredictionPipeline()
    return _pipeline


def get_audit_repository() -> PredictionAuditRepository:
    """Return the audit repository singleton."""
    global _audit_repository
    if _audit_repository is None:
        _audit_repository = build_audit_repository()
    return _audit_repository
