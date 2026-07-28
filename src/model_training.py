"""
Model Training Module
Cybersecurity - Detecting Phishing Emails

This module defines the ModelTraining class responsible for stratified data splitting,
baseline model training, hyperparameter tuning with GridSearchCV, and metrics evaluation.
"""

import logging
from typing import Dict, Any, Tuple
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)


class ModelTraining:
    """
    Handles data splitting, baseline model execution, hyperparameter tuning,
    and evaluation metrics for classification problems.
    """

    def __init__(self, config: Dict[str, Any], preprocessor: ColumnTransformer) -> None:
        self.config = config
        self.preprocessor = preprocessor
        self.random_state = self.config.get("random_state", 42)

    def split_data(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
        logging.info("Encoding target column and performing stratified train-val-test split.")

        target_col = self.config["target_column"]
        target_mapping = self.config.get("target_mapping", {"Legitimate": 0, "Phishing": 1})

        df_split = df.copy()
        if df_split[target_col].dtype == object or isinstance(df_split[target_col].iloc[0], str):
            df_split[target_col] = df_split[target_col].map(target_mapping)

        X = df_split.drop(columns=[target_col])
        y = df_split[target_col]

        val_test_size = self.config.get("val_test_size", 0.3)
        val_size = self.config.get("val_size", 0.5)

        X_train, X_temp, y_train, y_temp = train_test_split(
            X,
            y,
            test_size=val_test_size,
            random_state=self.random_state,
            stratify=y,
        )

        X_val, X_test, y_val, y_test = train_test_split(
            X_temp,
            y_temp,
            test_size=val_size,
            random_state=self.random_state,
            stratify=y_temp,
        )

        logging.info(
            f"Data split sizes -> Train: {X_train.shape[0]}, "
            f"Val: {X_val.shape[0]}, Test: {X_test.shape[0]}"
        )
        return X_train, X_val, X_test, y_train, y_val, y_test

    def _compute_metrics(
        self, model: Pipeline, X: pd.DataFrame, y: pd.Series
    ) -> Dict[str, Any]:
        y_pred = model.predict(X)

        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X)[:, 1]
            roc_auc = float(roc_auc_score(y, y_proba))
        else:
            roc_auc = 0.0

        cm = confusion_matrix(y, y_pred).tolist()

        return {
            "Accuracy": float(accuracy_score(y, y_pred)),
            "Precision": float(precision_score(y, y_pred, zero_division=0)),
            "Recall": float(recall_score(y, y_pred, zero_division=0)),
            "F1": float(f1_score(y, y_pred, zero_division=0)),
            "ROC-AUC": roc_auc,
            "Confusion Matrix": cm,
        }

    def train_and_evaluate_baseline_models(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
    ) -> Tuple[Dict[str, Pipeline], Dict[str, Dict[str, Any]]]:
        logging.info("Training and evaluating baseline models...")

        baseline_classifiers = {
            "Logistic Regression (Baseline)": LogisticRegression(
                random_state=self.random_state, class_weight="balanced"
            ),
            "Decision Tree (Baseline)": DecisionTreeClassifier(
                random_state=self.random_state, class_weight="balanced"
            ),
            "Random Forest (Baseline)": RandomForestClassifier(
                random_state=self.random_state, class_weight="balanced"
            ),
            "KNN (Baseline)": KNeighborsClassifier(n_neighbors=5),
        }

        models = {}
        metrics = {}

        for name, clf in baseline_classifiers.items():
            pipeline = Pipeline(
                steps=[
                    ("preprocessor", self.preprocessor),
                    ("classifier", clf),
                ]
            )
            pipeline.fit(X_train, y_train)
            val_metrics = self._compute_metrics(pipeline, X_val, y_val)

            models[name] = pipeline
            metrics[name] = val_metrics

            logging.info(
                f"{name} Val Metrics -> Accuracy: {val_metrics['Accuracy']:.4f}, "
                f"Precision: {val_metrics['Precision']:.4f}, Recall: {val_metrics['Recall']:.4f}, "
                f"F1: {val_metrics['F1']:.4f}, ROC-AUC: {val_metrics['ROC-AUC']:.4f}"
            )

        return models, metrics

    def train_and_evaluate_tuned_models(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
    ) -> Tuple[Dict[str, Pipeline], Dict[str, Dict[str, Any]]]:
        logging.info("Training and tuning models with GridSearchCV...")

        param_grids = self.config.get("param_grids", {})
        cv = self.config.get("cv", 5)
        scoring = self.config.get("scoring", "f1")

        algorithm_map = {
            "LogisticRegression": LogisticRegression(random_state=self.random_state),
            "DecisionTree": DecisionTreeClassifier(random_state=self.random_state),
            "RandomForest": RandomForestClassifier(random_state=self.random_state),
            "KNN": KNeighborsClassifier(),
        }

        tuned_models = {}
        tuned_metrics = {}

        for algo_name, param_grid in param_grids.items():
            if algo_name not in algorithm_map:
                continue

            base_clf = algorithm_map[algo_name]
            pipeline = Pipeline(
                steps=[
                    ("preprocessor", self.preprocessor),
                    ("classifier", base_clf),
                ]
            )

            grid_search = GridSearchCV(
                estimator=pipeline,
                param_grid=param_grid,
                cv=cv,
                scoring=scoring,
                n_jobs=-1,
            )

            grid_search.fit(X_train, y_train)
            best_pipeline = grid_search.best_estimator_
            val_metrics = self._compute_metrics(best_pipeline, X_val, y_val)

            model_name = f"{algo_name} (Tuned)"
            tuned_models[model_name] = best_pipeline
            tuned_metrics[model_name] = val_metrics

            logging.info(
                f"{model_name} Best Params: {grid_search.best_params_} | "
                f"Val F1: {val_metrics['F1']:.4f}, ROC-AUC: {val_metrics['ROC-AUC']:.4f}"
            )

        return tuned_models, tuned_metrics

    def evaluate_final_model(
        self, model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series, model_name: str
    ) -> Dict[str, Any]:
        logging.info(f"Evaluating best model '{model_name}' on test set.")
        test_metrics = self._compute_metrics(model, X_test, y_test)

        logging.info(f"=== TEST SET METRICS FOR {model_name} ===")
        for metric, val in test_metrics.items():
            if isinstance(val, float):
                logging.info(f"  {metric}: {val:.4f}")
            else:
                logging.info(f"  {metric}: {val}")

        return test_metrics
