"""
Pydantic schemas for the AutoAssur motor insurance pricing API.

Defines input (InsuranceInput) and output schemas
(FrequenceResponse, GraviteResponse, PrimeResponse, ExplainResponse).

The model version is stored in the MODEL_VERSION constant defined here
and imported by all endpoints. This avoids having "v1.0" hardcoded
in multiple places — update it here only when the model changes.
"""

from pydantic import BaseModel, Field, field_validator, model_validator

# ── Centralised model version ────────────────────────────────────────────────
# Update this constant only when deploying a new model version.
# Imported by predict.py to populate the model_version field in all responses.
MODEL_VERSION = "v1.0"


# ── Input schema ─────────────────────────────────────────────────────────────

class InsuranceInput(BaseModel):
    """
    Raw input data for a motor insurance contract.

    Matches exactly the features expected by the XGBoost models,
    before passing through the preprocessing and feature engineering pipeline.

    Business validations:
    - age_conducteur1: between 18 and 100 (Pydantic Field constraint)
    - anciennete_permis1: coherence with age checked by model_validator
    """

    # ── Contract ──────────────────────────────────────────────────────────────
    type_contrat: str = Field(
        ...,
        description="Contract type (A = comprehensive, B = extended third-party, C = third-party)",
        examples=["A"],
    )
    duree_contrat: float = Field(
        ..., ge=0, description="Contract duration in months"
    )
    anciennete_info: float = Field(
        ..., ge=0, description="Customer seniority in the database (years)"
    )
    freq_paiement: str = Field(
        ..., description="Payment frequency (mensuel, trimestriel, annuel)"
    )
    utilisation: str = Field(
        ..., description="Vehicle usage (prive, pro, mixte)"
    )
    code_postal: str = Field(
        ..., description="Postal code of the main driver"
    )

    # ── Main driver ───────────────────────────────────────────────────────────
    age_conducteur1: float = Field(
        ..., ge=18, le=100, description="Age of the main driver (years)"
    )
    sex_conducteur1: str = Field(
        ..., description="Gender of the main driver (M or F)"
    )
    anciennete_permis1: float = Field(
        ..., ge=0, description="Driving licence seniority (years)"
    )

    # ── Vehicle ───────────────────────────────────────────────────────────────
    anciennete_vehicule: float = Field(
        ..., ge=0, description="Vehicle age (years)"
    )
    cylindre_vehicule: float = Field(
        ..., ge=0, description="Engine displacement (cm³)"
    )
    din_vehicule: float = Field(
        ..., ge=0, description="Engine power (DIN horsepower)"
    )
    essence_vehicule: str = Field(
        ..., description="Fuel type (essence, diesel, electrique, hybride)"
    )
    marque_vehicule: str = Field(
        ..., description="Vehicle brand"
    )
    modele_vehicule: str = Field(
        ..., description="Vehicle model"
    )
    fin_vente_vehicule: float = Field(
        ..., description="Year the model was discontinued"
    )
    debut_vente_vehicule: float | None = Field(
        default=None,
        description=(
            "Year the model was first sold (optional). "
            "If absent, the pipeline defaults to fin_vente_vehicule - 5."
        ),
    )
    vitesse_vehicule: float = Field(
        ..., ge=0, description="Top speed of the vehicle (km/h)"
    )
    type_vehicule: str = Field(
        ..., description="Vehicle category (berline, suv, citadine, break, coupe)"
    )
    prix_vehicule: float = Field(
        ..., ge=0, description="Vehicle value when new (€)"
    )
    poids_vehicule: float = Field(
        ..., ge=0, description="Vehicle weight (kg)"
    )

    # ── Secondary driver (optional) ───────────────────────────────────────────
    conducteur2: str | None = Field(
        default=None,
        description="Secondary driver present (Yes / No)",
    )

    # ── Business validation ───────────────────────────────────────────────────
    @field_validator(
        "duree_contrat",
        "anciennete_info",
        "age_conducteur1",
        "anciennete_permis1",
        "anciennete_vehicule",
        "cylindre_vehicule",
        "din_vehicule",
        "fin_vente_vehicule",
        "debut_vente_vehicule",
        "vitesse_vehicule",
        "prix_vehicule",
        "poids_vehicule",
        mode="before",
    )
    @classmethod
    def parse_single_numeric_value(cls, value: object) -> object:
        """Normalize one-item numeric wrappers like ``[5.834E-2]``."""
        if value is None or isinstance(value, (int, float)):
            return value

        if isinstance(value, list):
            if len(value) != 1:
                raise ValueError("Expected a single numeric value.")
            value = value[0]

        if isinstance(value, str):
            cleaned = value.strip()
            if cleaned.startswith("[") and cleaned.endswith("]"):
                cleaned = cleaned[1:-1].strip()
            if not cleaned:
                raise ValueError("Expected a numeric value.")
            return cleaned

        return value

    @model_validator(mode="after")
    def check_licence_age_coherence(self) -> "InsuranceInput":
        """
        Verify that the computed licence-obtaining age is realistic.

        The licence age is: age_conducteur1 - anciennete_permis1.
        In France, the minimum age for a driving licence is 17 (accompanied
        driving) or 18 (standard). We accept 16 as the lower bound to cover
        licences obtained abroad or via accompanied driving programmes.

        Raises:
            ValueError: If the computed licence age is below 16 years.
        """
        licence_age = self.age_conducteur1 - self.anciennete_permis1
        if licence_age < 16:
            raise ValueError(
                f"Inconsistency detected: driver would have obtained their licence at "
                f"{licence_age:.0f} years old "
                f"(age={self.age_conducteur1:.0f}, "
                f"licence_seniority={self.anciennete_permis1:.0f}). "
                "Minimum legal age is 16."
            )
        return self

    model_config = {
        "json_schema_extra": {
            "example": {
                "type_contrat": "A",
                "duree_contrat": 12.0,
                "anciennete_info": 5.0,
                "freq_paiement": "mensuel",
                "utilisation": "prive",
                "code_postal": "75001",
                "age_conducteur1": 35.0,
                "sex_conducteur1": "M",
                "anciennete_permis1": 12.0,
                "anciennete_vehicule": 3.0,
                "cylindre_vehicule": 1600.0,
                "din_vehicule": 90.0,
                "essence_vehicule": "essence",
                "marque_vehicule": "Peugeot",
                "modele_vehicule": "308",
                "fin_vente_vehicule": 2022.0,
                "vitesse_vehicule": 180.0,
                "type_vehicule": "berline",
                "prix_vehicule": 18000.0,
                "poids_vehicule": 1200.0,
            }
        }
    }


# ── Output schemas ────────────────────────────────────────────────────────────

class FrequenceResponse(BaseModel):
    """Response schema for the frequency model."""
    frequence_predite: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Predicted claim probability, clamped between 0 and 1",
    )


class GraviteResponse(BaseModel):
    """Response schema for the severity model."""
    cout_moyen_predit: float = Field(
        ...,
        ge=0.0,
        description="Predicted average claim cost (€)",
    )


class PrimeResponse(BaseModel):
    """
    Full response: pure premium = frequency x severity.

    The model_version field is populated from the MODEL_VERSION constant.
    It must never be hardcoded directly in endpoint handlers.
    """
    frequence_predite: float = Field(
        ..., description="Predicted claim probability"
    )
    cout_moyen_predit: float = Field(
        ..., description="Predicted average claim cost (€)"
    )
    prime_pure: float = Field(
        ..., description="Pure premium = frequency x severity (€)"
    )
    niveau_risque: str = Field(
        ...,
        description="Risk level: faible | modéré | élevé | très élevé",
    )
    model_version: str = Field(
        default=MODEL_VERSION,
        description="Version of the XGBoost models used",
    )


class ExplainResponse(BaseModel):
    """
    Enriched response: pure premium + top risk factors (SHAP).

    Addresses the GDPR Article 22 requirement for explainability
    in automated decision-making systems.
    """
    frequence_predite: float = Field(
        ..., description="Predicted claim probability"
    )
    cout_moyen_predit: float = Field(
        ..., description="Predicted average claim cost (€)"
    )
    prime_pure: float = Field(
        ..., description="Pure premium = frequency x severity (€)"
    )
    niveau_risque: str = Field(
        ..., description="Risk level: faible | modéré | élevé | très élevé"
    )
    facteurs_de_risque: list[str] = Field(
        ...,
        description=(
            "Top 3 most influential risk factors, "
            "computed via SHAP TreeExplainer on the frequency model"
        ),
    )
    model_version: str = Field(
        default=MODEL_VERSION,
        description="Version of the XGBoost models used",
    )


class HealthResponse(BaseModel):
    """Response schema for the health check endpoint."""
    status: str = Field(..., description="API status (ok | degraded | down)")
    message: str = Field(..., description="Descriptive status message")
