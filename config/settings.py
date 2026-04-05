"""
Configuration settings for IPL Score Prediction
"""

import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
LOGS_DIR = os.path.join(BASE_DIR, "logs")
STATIC_DIR = os.path.join(BASE_DIR, "static")
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")

MODEL_PATHS = {
    "trained_models": os.path.join(MODELS_DIR, "trained_models.pkl"),
    "preprocessing_params": os.path.join(MODELS_DIR, "preprocessing_params.pkl"),
    "evaluation_results": os.path.join(MODELS_DIR, "evaluation_results.json"),
}

DATASET_CONFIG = {"csv_file": "IPL.csv", "test_size": 0.2, "random_state": 42}

MODEL_CONFIG = {
    "models": [
        "Linear Regression",
        "Ridge Regression",
        "Lasso Regression",
        "Decision Tree",
    ],
    "default_model": "Linear Regression",
}

API_CONFIG = {"host": "0.0.0.0", "port": 5000, "debug": True}
