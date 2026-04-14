"""Prediction endpoints for the auto insurance API."""

import logging
import time

import shap
from fastapi import APIRouter, Depends, HTTPException

from auto_insurance.api.dependencies import get_audit_repository, get_pipeline
from auto_insurance.api.persistence import PredictionAuditRepository
from auto_insurance.api.schemas.insurance import (
    ExplainResponse,
    FrequenceResponse,
    GraviteResponse,
    InsuranceInput,
    PrimeResponse,
)
from auto_insurance.src.pipeline import PredictionPipeline

router = APIRouter(prefix="/predict", tags=["Predictions"])
logger = logging.getLogger(__name__)

# Centralised model version — update here only when deploying a new model.
MODEL_VERSION = "v1.0"


# ── Helpers ───────────────────────────────────────────────────────────────────

def _get_risk_level(frequence: float) -> str:
    """
    Return a risk level label based on predicted claim frequency.

    Thresholds calibrated on the training data distribution:
    - low      : < 5%  annual claim probability
    - moderate : 5% to 10%
    - high     : 10% to 20%
    - very high: > 20%
    """
    if frequence < 0.05:
        return "faible"
    if frequence < 0.10:
        return "modere"
    if frequence < 0.20:
        return "eleve"
    return "tres eleve"


def _get_risk_factors(
    data: InsuranceInput,
    pipeline: PredictionPipeline,
) -> list[str]:
    """
    Compute the top 3 risk factors using SHAP TreeExplainer.

    SHAP (SHapley Additive exPlanations) assigns each feature its marginal
    contribution to the prediction. A positive SHAP value means the feature
    increases predicted risk; a negative value means it decreases it.

    The frequency model is used for explanation because frequency is the
    primary driver of the risk classification.

    Args:
        data: Validated input data from the Pydantic schema.
        pipeline: Loaded prediction pipeline singleton.

    Returns:
        List of 3 strings describing the most influential features.

    Raises:
        RuntimeError: If the SHAP computation fails.
    """
    df = pipeline._build_features(data.model_dump())

    # SHAP TreeExplainer does not support pandas 'category' dtype.
    # Convert categorical columns to integer codes before computing SHAP values.
    df_shap = df.copy()
    for col in df_shap.select_dtypes(include="category").columns:
        df_shap[col] = df_shap[col].cat.codes

    explainer = shap.TreeExplainer(pipeline.model.model_frequence)
    shap_values = explainer.shap_values(df_shap)

    feature_names = df_shap.columns.tolist()
    shap_importance = dict(zip(feature_names, shap_values[0]))

    top_features = sorted(
        shap_importance.items(),
        key=lambda x: abs(x[1]),
        reverse=True,
    )[:3]

    return [
        f"{feature} {'augmente' if value > 0 else 'diminue'} le risque"
        for feature, value in top_features
    ]


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post(
    "/frequency",
    response_model=FrequenceResponse,
    summary="Claim probability",
    description=(
        "Predicts the probability that a claim will occur during the contract period. "
        "Output is clamped between 0 and 1. "
        "Powered by an XGBoost frequency model trained on historical claims data."
    ),
)
def predict_frequency(
    data: InsuranceInput,
    pipeline: PredictionPipeline = Depends(get_pipeline),
    audit_repository: PredictionAuditRepository = Depends(get_audit_repository),
) -> FrequenceResponse:
    """Predict claim frequency (probability) for a given insurance contract."""
    start = time.perf_counter()
    try:
        frequence = pipeline.predict_frequence(data.model_dump())
        response = FrequenceResponse(frequence_predite=frequence)
        audit_repository.save_prediction(
            endpoint="/predict/frequency",
            request_payload=data.model_dump(),
            response_payload=response.model_dump(),
        )
        logger.info(
            "Frequency prediction successful | postal_code=%s | latency_ms=%.2f",
            data.code_postal,
            (time.perf_counter() - start) * 1000,
        )
        return response

    except ValueError as exc:
        # Invalid business data after Pydantic validation
        # e.g. a computed feature falls outside the range expected by the model
        logger.warning("Invalid data for frequency prediction | error=%s", exc)
        raise HTTPException(
            status_code=422,
            detail=f"Invalid input data: {exc}",
        ) from exc

    except KeyError as exc:
        # A feature expected by the model is missing from the pipeline DataFrame
        logger.error("Missing feature in frequency pipeline | feature=%s", exc)
        raise HTTPException(
            status_code=400,
            detail=f"Missing feature in pipeline: {exc}. "
                   "Ensure all required input fields are provided.",
        ) from exc

    except RuntimeError as exc:
        # Internal model error (model not loaded, XGBoost incompatibility)
        logger.error("Frequency model internal error | error=%s", exc)
        raise HTTPException(
            status_code=503,
            detail="The frequency model is temporarily unavailable. "
                   "Please verify that model files are correctly loaded.",
        ) from exc

    except Exception as exc:
        # Unexpected system error — log full traceback for debugging
        logger.exception("Unexpected error on /predict/frequency | error_type=%s", type(exc).__name__)
        raise HTTPException(
            status_code=500,
            detail="Internal server error. Check server logs for details.",
        ) from exc


@router.post(
    "/severity",
    response_model=GraviteResponse,
    summary="Average claim cost",
    description=(
        "Predicts the average cost of a claim if one occurs. "
        "Output is expressed in euros (always positive). "
        "Powered by an XGBoost severity model trained on historical claims data."
    ),
)
def predict_severity(
    data: InsuranceInput,
    pipeline: PredictionPipeline = Depends(get_pipeline),
    audit_repository: PredictionAuditRepository = Depends(get_audit_repository),
) -> GraviteResponse:
    """Predict claim severity (average cost) for a given insurance contract."""
    start = time.perf_counter()
    try:
        gravite = pipeline.predict_gravite(data.model_dump())
        response = GraviteResponse(cout_moyen_predit=gravite)
        audit_repository.save_prediction(
            endpoint="/predict/severity",
            request_payload=data.model_dump(),
            response_payload=response.model_dump(),
        )
        logger.info(
            "Severity prediction successful | vehicle=%s | latency_ms=%.2f",
            data.modele_vehicule,
            (time.perf_counter() - start) * 1000,
        )
        return response

    except ValueError as exc:
        logger.warning("Invalid data for severity prediction | error=%s", exc)
        raise HTTPException(
            status_code=422,
            detail=f"Invalid input data: {exc}",
        ) from exc

    except KeyError as exc:
        logger.error("Missing feature in severity pipeline | feature=%s", exc)
        raise HTTPException(
            status_code=400,
            detail=f"Missing feature in pipeline: {exc}.",
        ) from exc

    except RuntimeError as exc:
        logger.error("Severity model internal error | error=%s", exc)
        raise HTTPException(
            status_code=503,
            detail="The severity model is temporarily unavailable.",
        ) from exc

    except Exception as exc:
        logger.exception("Unexpected error on /predict/severity | error_type=%s", type(exc).__name__)
        raise HTTPException(
            status_code=500,
            detail="Internal server error. Check server logs for details.",
        ) from exc


@router.post(
    "/premium",
    response_model=PrimeResponse,
    summary="Full pure premium",
    description=(
        "Computes the pure premium using the standard actuarial formula:\n\n"
        "`Pure Premium = Frequency x Severity`\n\n"
        "Also returns a risk level (faible / modere / eleve / tres eleve) "
        "derived from the predicted claim frequency."
    ),
)
def predict_premium(
    data: InsuranceInput,
    pipeline: PredictionPipeline = Depends(get_pipeline),
    audit_repository: PredictionAuditRepository = Depends(get_audit_repository),
) -> PrimeResponse:
    """Compute the pure premium = frequency x severity."""
    start = time.perf_counter()
    try:
        result = pipeline.predict_prime(data.model_dump())
        response = PrimeResponse(
            frequence_predite=result["frequence_predite"],
            cout_moyen_predit=result["cout_moyen_predit"],
            prime_pure=result["prime_pure"],
            niveau_risque=_get_risk_level(result["frequence_predite"]),
            model_version=MODEL_VERSION,
        )
        audit_repository.save_prediction(
            endpoint="/predict/premium",
            request_payload=data.model_dump(),
            response_payload=response.model_dump(),
            niveau_risque=response.niveau_risque,
        )
        logger.info(
            "Premium prediction successful | vehicle=%s %s | risk=%s | latency_ms=%.2f",
            data.marque_vehicule,
            data.modele_vehicule,
            response.niveau_risque,
            (time.perf_counter() - start) * 1000,
        )
        return response

    except ValueError as exc:
        logger.warning("Invalid data for premium computation | error=%s", exc)
        raise HTTPException(
            status_code=422,
            detail=f"Invalid input data: {exc}",
        ) from exc

    except KeyError as exc:
        logger.error("Missing feature in premium pipeline | feature=%s", exc)
        raise HTTPException(
            status_code=400,
            detail=f"Missing feature in pipeline: {exc}.",
        ) from exc

    except RuntimeError as exc:
        logger.error("Model internal error during premium computation | error=%s", exc)
        raise HTTPException(
            status_code=503,
            detail="Models are temporarily unavailable.",
        ) from exc

    except Exception as exc:
        logger.exception("Unexpected error on /predict/premium | error_type=%s", type(exc).__name__)
        raise HTTPException(
            status_code=500,
            detail="Internal server error. Check server logs for details.",
        ) from exc


@router.post(
    "/explain",
    response_model=ExplainResponse,
    summary="Pure premium + risk factors (SHAP)",
    description=(
        "Computes the pure premium and returns the top 3 risk factors "
        "driving the prediction, computed via SHAP TreeExplainer.\n\n"
        "SHAP (SHapley Additive exPlanations) assigns each feature its marginal "
        "contribution to the prediction, providing local explainability. "
        "This endpoint addresses the GDPR Article 22 requirement for "
        "explainability in automated decision-making."
    ),
)
def predict_explain(
    data: InsuranceInput,
    pipeline: PredictionPipeline = Depends(get_pipeline),
    audit_repository: PredictionAuditRepository = Depends(get_audit_repository),
) -> ExplainResponse:
    """Compute the pure premium and explain risk factors via SHAP."""
    start = time.perf_counter()
    try:
        result = pipeline.predict_prime(data.model_dump())
        risk_factors = _get_risk_factors(data, pipeline)
        risk_level = _get_risk_level(result["frequence_predite"])

        response = ExplainResponse(
            frequence_predite=result["frequence_predite"],
            cout_moyen_predit=result["cout_moyen_predit"],
            prime_pure=result["prime_pure"],
            niveau_risque=risk_level,
            facteurs_de_risque=risk_factors,
            model_version=MODEL_VERSION,
        )
        audit_repository.save_prediction(
            endpoint="/predict/explain",
            request_payload=data.model_dump(),
            response_payload=response.model_dump(),
            niveau_risque=response.niveau_risque,
        )
        logger.info(
            "Explain prediction successful | contract=%s | risk=%s | latency_ms=%.2f",
            data.type_contrat,
            risk_level,
            (time.perf_counter() - start) * 1000,
        )
        return response

    except ValueError as exc:
        logger.warning("Invalid data for SHAP explanation | error=%s", exc)
        raise HTTPException(
            status_code=422,
            detail=f"Invalid input data: {exc}",
        ) from exc

    except KeyError as exc:
        logger.error("Missing feature in explain pipeline | feature=%s", exc)
        raise HTTPException(
            status_code=400,
            detail=f"Missing feature in pipeline: {exc}.",
        ) from exc

    except RuntimeError as exc:
        logger.error("SHAP or model error on /predict/explain | error=%s", exc)
        raise HTTPException(
            status_code=503,
            detail="SHAP computation is temporarily unavailable.",
        ) from exc

    except Exception as exc:
        logger.exception("Unexpected error on /predict/explain | error_type=%s", type(exc).__name__)
        raise HTTPException(
            status_code=500,
            detail="Internal server error. Check server logs for details.",
        ) from exc
