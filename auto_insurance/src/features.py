"""
Module de feature engineering pour l'assurance auto.
Crée de nouvelles variables métier à partir des données nettoyées.
"""

import numpy as np
import pandas as pd


class FeatureEngineer:
    """
    Création des features métier pour l'assurance auto.
    Sans état (stateless) — pas besoin de fit/transform,
    les calculs sont les mêmes pour le train et une observation.
    """

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Ajoute les nouvelles features au DataFrame.

        Args:
            df: DataFrame nettoyé (issu de DataPreprocessor).

        Returns:
            DataFrame enrichi avec les nouvelles features.
        """
        df = df.copy()
        df = self._ratio_poids_puissance(df)
        df = self._age_obtention_permis(df)
        df = self._jeune_conducteur(df)
        df = self._duree_vie_modele(df)
        df = self._log_prix_vehicule(df)
        return df

    def _ratio_poids_puissance(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ratio poids / puissance du véhicule."""
        if 'poids_vehicule' in df.columns and 'din_vehicule' in df.columns:
            df['ratio_poids_puissance'] = (
                df['poids_vehicule'] / (df['din_vehicule'] + 1e-5)
            )
        return df

    def _age_obtention_permis(self, df: pd.DataFrame) -> pd.DataFrame:
        """Âge du conducteur lors de l'obtention du permis."""
        if 'age_conducteur1' in df.columns and 'anciennete_permis1' in df.columns:
            df['age_obtention_permis'] = (
                df['age_conducteur1'] - df['anciennete_permis1']
            )
        return df

    def _jeune_conducteur(self, df: pd.DataFrame) -> pd.DataFrame:
        """1 si permis de moins de 3 ans, 0 sinon."""
        if 'anciennete_permis1' in df.columns:
            df['jeune_conducteur'] = (
                df['anciennete_permis1'] < 3
            ).astype(int)
        return df

    def _duree_vie_modele(self, df: pd.DataFrame) -> pd.DataFrame:
        """Durée de commercialisation du modèle de véhicule."""
        if 'fin_vente_vehicule' in df.columns and 'debut_vente_vehicule' in df.columns:
            df['duree_vie_modele'] = (
                df['fin_vente_vehicule'] - df['debut_vente_vehicule']
            )
        return df

    def _log_prix_vehicule(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transformation logarithmique du prix véhicule."""
        if 'prix_vehicule' in df.columns:
            df['log_prix_vehicule'] = np.log1p(df['prix_vehicule'])
        return df

    def get_feature_names(self) -> list[str]:
        """Retourne la liste des features créées."""
        return [
        "ratio_poids_puissance",
        "age_obtention_permis",
        "jeune_conducteur",
        "duree_vie_modele",
        "log_prix_vehicule",
        ]
