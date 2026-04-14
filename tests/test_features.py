"""
Tests unitaires pour le module FeatureEngineer.
"""

import pytest
import pandas as pd
import numpy as np
from auto_insurance.src.features import FeatureEngineer


# ------------------------------------------------------------------ #
#  Données de test                                                     #
# ------------------------------------------------------------------ #

@pytest.fixture
def sample_df() -> pd.DataFrame:
    """DataFrame minimal pour tester le feature engineering."""
    return pd.DataFrame({
        "age_conducteur1": [35, 22, 45],
        "anciennete_permis1": [10, 1, 20],
        "poids_vehicule": [1200, 900, 1500],
        "din_vehicule": [100, 80, 120],
        "prix_vehicule": [15000, 8000, 25000],
        "debut_vente_vehicule": [2010, 2015, 2008],
        "fin_vente_vehicule": [2020, 2022, 2018],
    })


# ------------------------------------------------------------------ #
#  Tests                                                               #
# ------------------------------------------------------------------ #

class TestFeatureEngineer:
    """Tests unitaires pour FeatureEngineer."""

    def test_transform_returns_dataframe(
        self, sample_df: pd.DataFrame
    ) -> None:
        """transform() doit retourner un DataFrame."""
        engineer = FeatureEngineer()
        result = engineer.transform(sample_df)
        assert isinstance(result, pd.DataFrame)

    def test_ratio_poids_puissance_created(
        self, sample_df: pd.DataFrame
    ) -> None:
        """La feature ratio_poids_puissance doit être créée."""
        engineer = FeatureEngineer()
        result = engineer.transform(sample_df)
        assert "ratio_poids_puissance" in result.columns

    def test_ratio_poids_puissance_value(
        self, sample_df: pd.DataFrame
    ) -> None:
        """ratio_poids_puissance doit être correct."""
        engineer = FeatureEngineer()
        result = engineer.transform(sample_df)
        expected = 1200 / (100 + 1e-5)
        assert abs(result["ratio_poids_puissance"].iloc[0] - expected) < 0.01

    def test_age_obtention_permis_created(
        self, sample_df: pd.DataFrame
    ) -> None:
        """La feature age_obtention_permis doit être créée."""
        engineer = FeatureEngineer()
        result = engineer.transform(sample_df)
        assert "age_obtention_permis" in result.columns

    def test_age_obtention_permis_value(
        self, sample_df: pd.DataFrame
    ) -> None:
        """age_obtention_permis = age - anciennete_permis."""
        engineer = FeatureEngineer()
        result = engineer.transform(sample_df)
        assert result["age_obtention_permis"].iloc[0] == 25

    def test_jeune_conducteur_created(
        self, sample_df: pd.DataFrame
    ) -> None:
        """La feature jeune_conducteur doit être créée."""
        engineer = FeatureEngineer()
        result = engineer.transform(sample_df)
        assert "jeune_conducteur" in result.columns

    def test_jeune_conducteur_value(
        self, sample_df: pd.DataFrame
    ) -> None:
        """jeune_conducteur = 1 si anciennete_permis < 3 ans."""
        engineer = FeatureEngineer()
        result = engineer.transform(sample_df)
        # anciennete_permis1 = [10, 1, 20] → [0, 1, 0]
        assert result["jeune_conducteur"].iloc[0] == 0
        assert result["jeune_conducteur"].iloc[1] == 1
        assert result["jeune_conducteur"].iloc[2] == 0

    def test_duree_vie_modele_created(
        self, sample_df: pd.DataFrame
    ) -> None:
        """La feature duree_vie_modele doit être créée."""
        engineer = FeatureEngineer()
        result = engineer.transform(sample_df)
        assert "duree_vie_modele" in result.columns

    def test_duree_vie_modele_value(
        self, sample_df: pd.DataFrame
    ) -> None:
        """duree_vie_modele = fin_vente - debut_vente."""
        engineer = FeatureEngineer()
        result = engineer.transform(sample_df)
        assert result["duree_vie_modele"].iloc[0] == 10

    def test_log_prix_vehicule_created(
        self, sample_df: pd.DataFrame
    ) -> None:
        """La feature log_prix_vehicule doit être créée."""
        engineer = FeatureEngineer()
        result = engineer.transform(sample_df)
        assert "log_prix_vehicule" in result.columns

    def test_log_prix_vehicule_value(
        self, sample_df: pd.DataFrame
    ) -> None:
        """log_prix_vehicule = log1p(prix_vehicule)."""
        engineer = FeatureEngineer()
        result = engineer.transform(sample_df)
        expected = np.log1p(15000)
        assert abs(result["log_prix_vehicule"].iloc[0] - expected) < 0.01

    def test_original_df_not_modified(
        self, sample_df: pd.DataFrame
    ) -> None:
        """transform() ne doit pas modifier le DataFrame original."""
        engineer = FeatureEngineer()
        original_cols = sample_df.columns.tolist()
        engineer.transform(sample_df)
        assert sample_df.columns.tolist() == original_cols