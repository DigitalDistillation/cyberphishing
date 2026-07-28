"""
Main Pipeline Script
AIAP Cybersecurity - Detecting Phishing Emails

Entry point script that loads project configuration, executes data preparation,
trains baseline and hyperparameter tuned classification models, selects the best performer
based on F1-score, evaluates performance on unseen test data, and saves the trained
model artifact for scanning emails in inference.
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

# Configure root logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


@ignore_warnings(category=Warning)
def main() -> None:
    """
    Executes the end-to-end Machine Learning pipeline and saves the best model.
    """
    logging.info("Starting AIAP Phishing Email Classification Pipeline...")

    # Load configuration
    config_path = "./src/config.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    logging.info(f"Loaded configuration from {config_path}")

    # Load dataset
    data_path = config["file_path"]
    df = pd.read_csv(data_path)
    logging.info(f"Loaded raw dataset from {data_path} with shape {df.shape}")

    # Step 1: Data Preparation & Preprocessor creation
    data_prep = DataPreparation(config)
    cleaned_df = data_prep.clean_data(df)

    # Step 2: Model Training initialization & Data Splitting
    model_training = ModelTraining(config, data_prep.preprocessor)
    X_train, X_val, X_test, y_train, y_val, y_test = model_training.split_data(cleaned_df)

    # Step 3: Baseline Models Training & Evaluation
    baseline_models, baseline_metrics = model_training.train_and_evaluate_baseline_models(
        X_train, y_train, X_val, y_val
    )

    # Step 4: Hyperparameter Tuning via GridSearchCV
    tuned_models, tuned_metrics = model_training.train_and_evaluate_tuned_models(
        X_train, y_train, X_val, y_val
    )

    # Combine all models and validation metrics
    all_models = {**baseline_models, **tuned_models}
    all_metrics = {**baseline_metrics, **tuned_metrics}

    # Step 5: Select best model based on validation F1 Score
    best_model_name = max(all_metrics, key=lambda k: all_metrics[k]["F1"])
    best_model = all_models[best_model_name]
    best_val_f1 = all_metrics[best_model_name]["F1"]

    logging.info(f"=== Best Model Selected: '{best_model_name}' (Val F1: {best_val_f1:.4f}) ===")

    # Step 6: Evaluate final best model on holdout Test Set
    final_metrics = model_training.evaluate_final_model(
        best_model, X_test, y_test, best_model_name
    )

    # Step 7: Save trained model artifact for inference / email scanning
    os.makedirs("models", exist_ok=True)
    model_path = "models/phishing_detector.pkl"
    joblib.dump(best_model, model_path)
    logging.info(f"Saved trained model pipeline to '{model_path}'")

    logging.info("Pipeline execution completed successfully.")


if __name__ == "__main__":
    main()
