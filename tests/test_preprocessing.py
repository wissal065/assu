"""
Tests unitaires pour le module DataPreprocessor.
"""

import pytest
import pandas as pd
import numpy as np
from auto_insurance.src.preprocessing import DataPreprocessor


# ------------------------------------------------------------------ #
#  Données de test                                                     #
# ------------------------------------------------------------------ #

@pytest.fixture
def sample_df() -> pd.DataFrame:
    """DataFrame minimal qui ressemble aux données réelles."""
    return pd.DataFrame({
        "age_conducteur1": [35, 22, 45],
        "anciennete_permis1": [10, 1, 20],
        "poids_vehicule": [1200, 900, 1500],
        "din_vehicule": [100, 80, 120],
        "prix_vehicule": [15000, 8000, 25000],
        "debut_vente_vehicule": [2010, 2015, 2008],
        "fin_vente_vehicule": [2020, 2022, 2018],
        "conducteur2": ["Yes", "No", "Yes"],
        "paiement": ["Yes", "No", "No"],
        "sex_conducteur1": ["M", "F", "M"],
        "marque_vehicule": ["Renault", "Peugeot", "Renault"],
        "nombre_sinistres": [1, 0, 2],
        "montant_sinistre": [500, 0, 1200],
    })


@pytest.fixture
def sample_observation() -> dict:
    """Une seule observation au format JSON/dict."""
    return {
        "age_conducteur1": 30,
        "anciennete_permis1": 5,
        "poids_vehicule": 1100,
        "din_vehicule": 90,
        "prix_vehicule": 12000,
        "debut_vente_vehicule": 2012,
        "fin_vente_vehicule": 2020,
        "conducteur2": "No",
        "paiement": "Yes",
        "sex_conducteur1": "F",
        "marque_vehicule": "Peugeot",
    }


# ------------------------------------------------------------------ #
#  Tests                                                               #
# ------------------------------------------------------------------ #

class TestDataPreprocessor:
    """Tests unitaires pour DataPreprocessor."""

    def test_fit_returns_self(self, sample_df: pd.DataFrame) -> None:
        """fit() doit retourner l'objet lui-même (pour le chaînage)."""
        preprocessor = DataPreprocessor()
        result = preprocessor.fit(sample_df)
        assert result is preprocessor

    def test_fit_sets_encoder(self, sample_df: pd.DataFrame) -> None:
        """fit() doit initialiser l'encoder."""
        preprocessor = DataPreprocessor()
        preprocessor.fit(sample_df)
        assert preprocessor.encoder is not None

    def test_transform_returns_dataframe(
        self,
        sample_df: pd.DataFrame,
        sample_observation: dict
    ) -> None:
        """transform() doit retourner un DataFrame."""
        preprocessor = DataPreprocessor()
        preprocessor.fit(sample_df)
        result = preprocessor.transform(sample_observation)
        assert isinstance(result, pd.DataFrame)

    def test_transform_returns_one_row(
        self,
        sample_df: pd.DataFrame,
        sample_observation: dict
    ) -> None:
        """transform() doit retourner exactement une ligne."""
        preprocessor = DataPreprocessor()
        preprocessor.fit(sample_df)
        result = preprocessor.transform(sample_observation)
        assert len(result) == 1

    def test_id_cols_removed(
        self,
        sample_df: pd.DataFrame,
        sample_observation: dict
    ) -> None:
        """Les colonnes identifiants doivent être supprimées."""
        preprocessor = DataPreprocessor()
        preprocessor.fit(sample_df)
        result = preprocessor.transform(sample_observation)
        for col in DataPreprocessor.ID_COLS:
            assert col not in result.columns

    def test_binary_encoding_yes_no(
        self,
        sample_df: pd.DataFrame,
        sample_observation: dict
    ) -> None:
        """Les colonnes Yes/No doivent être encodées en 0/1."""
        preprocessor = DataPreprocessor()
        preprocessor.fit(sample_df)
        result = preprocessor.transform(sample_observation)
        if "conducteur2" in result.columns:
            assert result["conducteur2"].iloc[0] in [0.0, 1.0]

    def test_no_missing_values(
        self,
        sample_df: pd.DataFrame,
        sample_observation: dict
    ) -> None:
        """Il ne doit pas rester de NaN après transform."""
        preprocessor = DataPreprocessor()
        preprocessor.fit(sample_df)
        result = preprocessor.transform(sample_observation)
        assert result.isna().sum().sum() == 0

    def test_all_numeric_after_transform(
        self,
        sample_df: pd.DataFrame,
        sample_observation: dict
    ) -> None:
        """Toutes les colonnes doivent être numériques après transform."""
        preprocessor = DataPreprocessor()
        preprocessor.fit(sample_df)
        result = preprocessor.transform(sample_observation)
        non_numeric = result.select_dtypes(exclude=[np.number]).columns.tolist()
        assert len(non_numeric) == 0