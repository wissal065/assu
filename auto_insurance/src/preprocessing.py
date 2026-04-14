"""
Module de nettoyage des données pour l'assurance auto.
Gère la suppression des colonnes inutiles, le remplissage des valeurs
manquantes, l'encodage binaire et le CountEncoder.
"""
import pickle
import numpy as np
import pandas as pd
from category_encoders import CountEncoder


class DataPreprocessor:
    """
    Nettoyage des données brutes d'assurance auto.

    Attributes:
        encoder: CountEncoder fitté sur les données d'entraînement.
        cat_cols: Colonnes catégorielles détectées au fit.
        cols_to_drop: Colonnes supprimées lors du fit.
    """

    ID_COLS: list[str] = [
        'index', 'id_client', 'id_vehicule',
        'id_contrat', 'nombre_sinistres', 'montant_sinistre'
    ]
    YES_NO_COLS: list[str] = ['conducteur2', 'paiement']
    SEX_COLS: list[str] = ['sex_conducteur1']

    def __init__(self) -> None:
        self.encoder: CountEncoder | None = None
        self.cat_cols: list[str] = []
        self.cols_to_drop: list[str] = []

    def fit(self, df: pd.DataFrame) -> "DataPreprocessor":
        """
        Apprend les paramètres de nettoyage sur le DataFrame d'entraînement.

        Args:
            df: DataFrame d'entraînement complet.

        Returns:
            self
        """
        df = df.copy()
        df = self._drop_id_cols(df)
        df = self._fill_missing(df)
        df = self._encode_binary(df)
        df = self._drop_high_missing(df)
        df = self._drop_low_variance(df)
        df = self._drop_high_correlation(df)

        self.cat_cols = df.select_dtypes(exclude=['number']).columns.tolist()
        self.encoder = CountEncoder(cols=self.cat_cols, handle_unknown='value')
        self.encoder.fit(df[self.cat_cols])

        return self

    def transform(self, observation: dict) -> pd.DataFrame:
        """
        Nettoie une observation brute (dict/JSON).

        Args:
            observation: Dictionnaire représentant un contrat.

        Returns:
            DataFrame d'une ligne nettoyé.
        """
        df = pd.DataFrame([observation])
        df = self._drop_id_cols(df)
        df = self._fill_missing(df)
        df = self._encode_binary(df)
        df = df.drop(columns=[c for c in self.cols_to_drop if c in df.columns])

        if self.encoder is not None:
            cols_present = [c for c in self.cat_cols if c in df.columns]
            # Seules transformer les colonnes présentes — éviter d'appeler
            # l'encoder sur un DataFrame vide (erreur de dimension).
            if len(cols_present) > 0:
                transformed = self.encoder.transform(df[cols_present])
                # Some encoders / pandas versions return pandas 'string' dtypes
                # which can cause downstream libraries (XGBoost) to reject inputs.
                # Convert only 'string' dtypes to plain Python objects, keep numeric types.
                for col in transformed.columns:
                    if pd.api.types.is_string_dtype(transformed[col].dtype):
                        transformed[col] = transformed[col].astype(object)
                df[cols_present] = transformed

        # Ensure no column uses numpy string dtypes (e.g., '<U' or 'S'),
        # which can cause errors in downstream libraries. Convert such
        # columns to Python `str` objects (object dtype).
        for col in df.columns:
            try:
                dtype_kind = df[col].dtype.kind
            except (AttributeError, TypeError):
                # Some exotic dtypes may not expose `dtype.kind` or may
                # raise TypeError when queried; treat these as unknown.
                dtype_kind = None
            if dtype_kind in ("U", "S") or pd.api.types.is_string_dtype(df[col].dtype):
                # astype(str) converts numpy string types to Python str values;
                # then ensure object dtype.
                df[col] = df[col].astype(str).astype(object)

        return df

    def save_encoder(self, path: str) -> None:
        """Sauvegarde le CountEncoder en pickle."""
        with open(path, 'wb') as f:
            pickle.dump(self.encoder, f)

    def load_encoder(self, path: str) -> None:
        """Charge le CountEncoder depuis un pickle."""
        with open(path, 'rb') as f:
            self.encoder = pickle.load(f)

    def _drop_id_cols(self, df: pd.DataFrame) -> pd.DataFrame:
        """Supprime les colonnes identifiants."""
        cols = [c for c in self.ID_COLS if c in df.columns]
        return df.drop(columns=cols)

    def _fill_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remplace les NaN : 0 pour numérique, -999 pour catégoriel."""
        num = df.select_dtypes(include=['number']).columns.tolist()
        cat = df.select_dtypes(exclude=['number']).columns.tolist()
        df[num] = df[num].fillna(0)
        df[cat] = df[cat].fillna('-999')
        return df

    def _encode_binary(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode Yes/No → 1/0 et M/F → 1/0."""
        for col in self.YES_NO_COLS:
            if col in df.columns:
                df[col] = df[col].map(
                    {'Yes': 1.0, 'No': 0.0, '-999': 0.0}
                ).fillna(0.0)
        for col in self.SEX_COLS:
            if col in df.columns:
                df[col] = df[col].map(
                    {'M': 1.0, 'F': 0.0, '-999': 0.0}
                ).fillna(0.0)
        return df

    def _drop_high_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        """Supprime les colonnes avec plus de 40% de valeurs manquantes."""
        num = df.select_dtypes(include=['number']).columns.tolist()
        cat = df.select_dtypes(exclude=['number']).columns.tolist()
        to_drop = []
        for col in num:
            if (df[col] == 0).sum() / len(df) > 0.4:
                to_drop.append(col)
        for col in cat:
            if (df[col] == '-999').sum() / len(df) > 0.4:
                to_drop.append(col)
        self.cols_to_drop.extend(to_drop)
        return df.drop(columns=to_drop)

    def _drop_low_variance(self, df: pd.DataFrame) -> pd.DataFrame:
        """Supprime les colonnes numériques quasi-constantes."""
        num = df.select_dtypes(include=['number']).columns.tolist()
        low_var = df[num].var()[lambda x: x < 0.01].index.tolist()
        self.cols_to_drop.extend(low_var)
        return df.drop(columns=low_var)

    def _drop_high_correlation(self, df: pd.DataFrame) -> pd.DataFrame:
        """Supprime les colonnes numériques trop corrélées (> 0.95)."""
        num = df.select_dtypes(include=['number']).columns.tolist()
        corr = df[num].corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        high_corr = [c for c in upper.columns if any(upper[c] > 0.95)]
        self.cols_to_drop.extend(high_corr)
        return df.drop(columns=high_corr)
