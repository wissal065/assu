"""
Pipeline d'inférence pour l'assurance auto.
Orchestre le feature engineering et la prédiction.
"""

import logging
from pathlib import Path

import pandas as pd

from auto_insurance.src.features import FeatureEngineer
from auto_insurance.src.model import InsuranceModel
from auto_insurance.src.preprocessing import DataPreprocessor

logger = logging.getLogger(__name__)

MODELS_DIR = Path(__file__).parent.parent / "models"
ENCODER_PATH = MODELS_DIR / "encoder.pkl"

EXPECTED_COLS = [
    "type_contrat", "duree_contrat", "anciennete_info", "freq_paiement",
    "utilisation", "code_postal", "age_conducteur1", "sex_conducteur1",
    "anciennete_permis1", "anciennete_vehicule", "cylindre_vehicule",
    "din_vehicule", "essence_vehicule", "marque_vehicule", "modele_vehicule",
    "fin_vente_vehicule", "vitesse_vehicule", "type_vehicule", "prix_vehicule",
    "poids_vehicule", "ratio_poids_puissance", "age_obtention_permis",
    "duree_vie_modele", "log_prix_vehicule"
]


class PredictionPipeline:
    """
    Pipeline complet d'inférence : input brut → prime pure.

    Attributes:
        model: Modèles XGBoost fréquence et gravité.
    """

    def __init__(self) -> None:
        self.model = InsuranceModel()
        self.preprocessor = DataPreprocessor()
        self.engineer = FeatureEngineer()
        self._load()

    def _load(self) -> None:
        """Charge les modèles et l'encodeur au démarrage."""
        self.model.load_models(
            str(MODELS_DIR / "model_frequence.json"),
            str(MODELS_DIR / "model_gravite.json"),
        )
        self.preprocessor.load_encoder(str(ENCODER_PATH))
        logger.info("Pipeline chargé avec succès.")

    def _build_features(self, input_data: dict) -> pd.DataFrame:
        """
        Construit le DataFrame avec les 24 features exactes attendues par XGBoost.

        Args:
            input_data: Dictionnaire des données brutes.

        Returns:
            DataFrame aligné et prêt pour la prédiction.
        """
        data = dict(input_data)
        if data.get("debut_vente_vehicule") is None:
            data["debut_vente_vehicule"] = data["fin_vente_vehicule"] - 5

        df = self.preprocessor.transform(data)
        df = self.engineer.transform(df)

        # Alignement strict des colonnes attendues
        for col in EXPECTED_COLS:
            if col not in df.columns:
                df[col] = 0
        df = df[EXPECTED_COLS]

        # Conversion pour XGBoost
        cat_cols = df.select_dtypes(include=["object", "string"]).columns
        df[cat_cols] = df[cat_cols].astype("category")

        return df

    def predict_frequence(self, input_data: dict) -> float:
        """Prédit la fréquence de sinistres — résultat clampé entre 0 et 1."""
        df = self._build_features(input_data)
        result = float(max(0.0, min(1.0, self.model.predict_frequence(df))))
        logger.info("Fréquence prédite : %.4f", result)
        return result

    def predict_gravite(self, input_data: dict) -> float:
        """Prédit la gravité (coût moyen) — résultat toujours positif."""
        df = self._build_features(input_data)
        result = float(max(0.0, self.model.predict_gravite(df)))
        logger.info("Gravité prédite : %.2f", result)
        return result

    def predict_prime(self, input_data: dict) -> dict:
        """
        Calcule la prime pure complète.

        Args:
            input_data: Dictionnaire des données brutes.

        Returns:
            Dictionnaire avec fréquence, gravité et prime pure.
        """
        logger.info("Nouvelle prédiction pour : %s", input_data.get("marque_vehicule", "?"))
        df = self._build_features(input_data)

        frequence = float(max(0.0, min(1.0, self.model.predict_frequence(df))))
        gravite = float(max(0.0, self.model.predict_gravite(df)))
        prime = frequence * gravite

        result = {
            "frequence_predite": round(frequence, 4),
            "cout_moyen_predit": round(gravite, 2),
            "prime_pure": round(prime, 2),
        }
        logger.info("Résultat : fréquence=%.4f, gravité=%.2f, prime=%.2f",
                    frequence, gravite, prime)
        return result
