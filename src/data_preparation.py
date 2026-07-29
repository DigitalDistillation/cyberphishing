"""
Data Preparation Module
Cybersecurity - Detecting Phishing Emails

Combines NLP TF-IDF text features with numerical scaling and categorical encoding
into a single ColumnTransformer preprocessor.
"""

import logging
from typing import Dict, Any
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, OneHotEncoder


class DataPreparation:
    """
    Handles data cleaning, validation, and creation of a hybrid NLP + Tabular
    Scikit-Learn ColumnTransformer preprocessor.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        self.text_feature = self.config.get("text_feature", "email_text")
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

        if self.text_feature in df_cleaned.columns:
            df_cleaned[self.text_feature] = df_cleaned[self.text_feature].fillna("")

        for col in df_cleaned.select_dtypes(include=["object", "str"]).columns:
            if col != self.text_feature:
                df_cleaned[col] = df_cleaned[col].astype(str).str.strip()

        null_counts = df_cleaned.isnull().sum().sum()
        if null_counts > 0:
            logging.info(f"Handling {null_counts} missing values.")
            df_cleaned.dropna(inplace=True)

        logging.info(f"Data cleaning complete. Output shape: {df_cleaned.shape}")
        return df_cleaned

    def _create_preprocessor(self) -> ColumnTransformer:
        logging.info("Creating Hybrid NLP + Tabular ColumnTransformer preprocessor.")

        text_transformer = TfidfVectorizer(
            max_features=2000,
            stop_words="english",
            ngram_range=(1, 2),
            sublinear_tf=True
        )
        numerical_transformer = StandardScaler()
        categorical_transformer = OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False
        )

        preprocessor = ColumnTransformer(
            transformers=[
                ("text", text_transformer, self.text_feature),
                ("num", numerical_transformer, self.numerical_features),
                ("cat", categorical_transformer, self.nominal_features),
            ],
            remainder="passthrough",
        )

        return preprocessor
