"""
Module de chargement et prediction des modeles d'assurance auto.
Gere les modeles de frequence, gravite et le calcul de la prime pure.
"""

import pandas as pd
from xgboost import XGBRegressor


class InsuranceModel:
    """
    Chargement et prediction des modeles XGBoost d'assurance auto.

    Attributes:
        model_frequence: Modele XGBoost de frequence des sinistres.
        model_gravite: Modele XGBoost de gravite (cout moyen).
    """

    def __init__(self) -> None:
        self.model_frequence: XGBRegressor = XGBRegressor()
        self.model_gravite: XGBRegressor = XGBRegressor()
        self.feature_names: list[str] | None = None

    def load_models(
        self,
        path_frequence: str,
        path_gravite: str,
    ) -> None:
        """
        Charge les modeles XGBoost depuis des fichiers JSON.

        Args:
            path_frequence: Chemin vers le fichier JSON du modele frequence.
            path_gravite: Chemin vers le fichier JSON du modele gravite.
        """
        self.model_frequence.load_model(path_frequence)
        self.model_gravite.load_model(path_gravite)
        try:
            booster = self.model_frequence.get_booster()
            feature_names_seq = booster.feature_names
            if feature_names_seq is not None:
                self.feature_names = list(feature_names_seq)
            else:
                self.feature_names = None
        except (AttributeError, ValueError, RuntimeError):
            self.feature_names = None

    def predict_frequence(self, df: pd.DataFrame) -> float:
        """Predict claim frequency and return a plain float."""
        result = self.model_frequence.predict(df)
        return float(result.flatten()[0])

    def predict_gravite(self, df: pd.DataFrame) -> float:
        """Predict average claim cost and return a plain float."""
        result = self.model_gravite.predict(df)
        return float(result.flatten()[0])

    def predict_prime(
        self,
        df: pd.DataFrame,
    ) -> dict:
        """
        Calcule la prime pure complete.
        Formule : frequence x cout moyen.

        Args:
            df: DataFrame d'une ligne pret pour la prediction.

        Returns:
            Dictionnaire avec frequence, gravite et prime pure.
        """
        frequence = self.predict_frequence(df)
        gravite = self.predict_gravite(df)
        prime = frequence * gravite
        return {
            "frequence_predite": frequence,
            "cout_moyen_predit": gravite,
            "prime_pure": prime,
        }

    def get_feature_names(self) -> list[str] | None:
        """Return the list of features expected by the model (or None)."""
        return self.feature_names
    