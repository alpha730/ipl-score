"""
IPL Score Prediction Package

This package contains the core modules for IPL match score prediction:
- data_preprocessing: Data loading and feature engineering
- models: ML models (Linear, Ridge, Lasso, Decision Tree)
- evaluation: Performance metrics and model comparison
"""

from src.data_preprocessing import (
    load_data,
    clean_data,
    create_innings_summary,
    prepare_training_data,
    normalize_features,
    train_test_split,
)

from src.models import (
    LinearRegression,
    RidgeRegression,
    LassoRegression,
    DecisionTreeRegressor,
    ModelComparator,
)

from src.evaluation import (
    PerformanceEvaluator,
    mean_squared_error,
    root_mean_squared_error,
    r_squared,
)

__version__ = "1.0.0"
__author__ = "IPL Prediction Team"
