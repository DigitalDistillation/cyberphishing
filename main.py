"""
Main Pipeline Script
Cybersecurity - Detecting Phishing Emails

Entry point script that loads configuration, executes data preparation,
trains baseline and tuned classification models, selects the best performer,
and saves the trained model artifact for inference.
"""

import os
import logging
import warnings
import pandas as pd
import yaml
import joblib
from sklearn.utils._testing import ignore_warnings

from src.data_preparation import DataPreparation
from src.model_training import ModelTraining

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


@ignore_warnings(category=Warning)
def main() -> None:
    logging.info("Starting Precision-Tuned Phishing Email Classification Pipeline...")

    config_path = "./src/config.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    data_path = config["file_path"]
    df = pd.read_csv(data_path)

    data_prep = DataPreparation(config)
    cleaned_df = data_prep.clean_data(df)

    model_training = ModelTraining(config, data_prep.preprocessor)
    X_train, X_val, X_test, y_train, y_val, y_test = model_training.split_data(cleaned_df)

    baseline_models, baseline_metrics = model_training.train_and_evaluate_baseline_models(
        X_train, y_train, X_val, y_val
    )

    tuned_models, tuned_metrics = model_training.train_and_evaluate_tuned_models(
        X_train, y_train, X_val, y_val
    )

    all_models = {**baseline_models, **tuned_models}
    all_metrics = {**baseline_metrics, **tuned_metrics}

    # Select best model based on Precision
    best_model_name = max(all_metrics, key=lambda k: all_metrics[k]["Precision"])
    best_model = all_models[best_model_name]

    logging.info(f"=== Best Precision Model Selected: '{best_model_name}' ===")

    final_metrics = model_training.evaluate_final_model(
        best_model, X_test, y_test, best_model_name
    )

    os.makedirs("models", exist_ok=True)
    model_path = "models/phishing_detector.pkl"
    joblib.dump(best_model, model_path)
    logging.info(f"Saved precision-tuned model to '{model_path}'")


if __name__ == "__main__":
    main()
