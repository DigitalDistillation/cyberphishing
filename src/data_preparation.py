"""
Data Preparation Module
Cybersecurity - Detecting Phishing Emails

This module defines the DataPreparation class responsible for loading,
cleaning, and defining feature preprocessing pipelines (StandardScaler and OneHotEncoder).
"""

import logging
from typing import Dict, Any
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder


class DataPreparation:
    """
    Handles data cleaning, validation, and creation of Scikit-Learn
    ColumnTransformers for numerical scaling and categorical encoding.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        self.numerical_features = self.config.get("numerical_features", [])
        self.nominal_features = self.config.get("nominal_features", [])
        self.preprocessor: ColumnTransformer = self._create_preprocessor()

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        logging.info("Starting data cleaning and validation.")
        df_cleaned = df.copy()

        initial_rows = len(df_cleaned)
        df_cleaned.drop_duplicates(inplace=True)
        deduped_rows = len(df_cleaned)
        if initial_rows - deduped_rows > 0:
            logging.info(f"Removed {initial_rows - deduped_rows} duplicate rows.")

        for col in df_cleaned.select_dtypes(include=["object", "str"]).columns:
            df_cleaned[col] = df_cleaned[col].astype(str).str.strip()

        null_counts = df_cleaned.isnull().sum().sum()
        if null_counts > 0:
            logging.info(f"Handling {null_counts} missing values.")
            df_cleaned.dropna(inplace=True)

        logging.info(f"Data cleaning complete. Output shape: {df_cleaned.shape}")
        return df_cleaned

    def _create_preprocessor(self) -> ColumnTransformer:
        logging.info("Creating Scikit-Learn ColumnTransformer preprocessor.")

        numerical_transformer = StandardScaler()
        categorical_transformer = OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False
        )

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numerical_transformer, self.numerical_features),
                ("cat", categorical_transformer, self.nominal_features),
            ],
            remainder="passthrough",
        )

        return preprocessor
