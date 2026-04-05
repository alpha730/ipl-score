"""
IPL Score Prediction - Main Training Script
Uses core parameters: runs, overs, wickets
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import json
import pickle
from src.data_preprocessing import (
    load_data,
    clean_data,
    create_innings_summary,
    prepare_training_data,
    normalize_features,
    train_test_split,
    get_feature_names,
    get_target_name,
)
from src.models import (
    LinearRegression,
    RidgeRegression,
    LassoRegression,
    DecisionTreeRegressor,
    ModelComparator,
)
from src.evaluation import PerformanceEvaluator


def main():
    print("=" * 70)
    print("IPL SCORE PREDICTION MODEL TRAINING")
    print("=" * 70)

    print("\nUsing core parameters: RUNS | OVERS | WICKETS")

    # Get base directory
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, "data", "IPL.csv")
    models_dir = os.path.join(base_dir, "models")

    # Ensure models directory exists
    os.makedirs(models_dir, exist_ok=True)

    # Step 1: Load and preprocess data
    print("\n[1/5] Loading IPL data...")
    data = load_data(data_path)
    print(f"Loaded {len(data)} rows")

    print("\n[2/5] Cleaning data...")
    cleaned = clean_data(data)
    print(f"Cleaned data: {len(cleaned)} rows")

    print("\n[3/5] Creating innings summary...")
    innings = create_innings_summary(cleaned)
    print(f"Created {len(innings)} innings summaries")

    # Step 4: Prepare training data
    print("\n[4/5] Preparing training data...")
    X, y, encoders = prepare_training_data(innings)

    if len(X) == 0:
        print("ERROR: No training data generated.")
        return

    print(f"Features shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    print(f"Feature names: {get_feature_names()}")

    # Normalize features
    X_normalized, mean, std = normalize_features(X, fit=True)
    print(f"Features normalized")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_normalized, y, test_size=0.2, random_state=42
    )
    print(f"Training: {len(X_train)} | Test: {len(X_test)}")

    # Save preprocessing params
    preprocessing_params = {
        "encoders": encoders,
        "mean": mean.tolist(),
        "std": std.tolist(),
        "feature_names": get_feature_names(),
        "target_name": get_target_name(),
    }

    params_path = os.path.join(models_dir, "preprocessing_params.pkl")
    with open(params_path, "wb") as f:
        pickle.dump(preprocessing_params, f)
    print("Saved preprocessing parameters")

    # Step 5: Train models
    print("\n[5/5] Training models...")

    models = {
        "Linear Regression": LinearRegression(learning_rate=0.01, n_iterations=2000),
        "Ridge Regression": RidgeRegression(
            alpha=1.0, learning_rate=0.01, n_iterations=2000
        ),
        "Lasso Regression": LassoRegression(
            alpha=0.1, learning_rate=0.01, n_iterations=2000
        ),
        "Decision Tree": DecisionTreeRegressor(max_depth=15, min_samples_split=5),
    }

    comparator = ModelComparator()
    for name, model in models.items():
        comparator.add_model(name, model)

    comparator.train_all(X_train, y_train)
    results = comparator.evaluate_all(X_test, y_test)
    comparator.comparison_table()

    # Evaluation
    print("\nGenerating evaluation reports...")
    evaluator = PerformanceEvaluator()
    evaluator.set_test_data(y_test)

    for name, result in results.items():
        evaluator.add_model_result(name, result["predictions"])

    evaluator.evaluate()
    evaluator.print_comparison()

    # Save models
    print("\nSaving trained models...")
    trained_models = {}
    for name, model in models.items():
        trained_models[name] = {
            "model": model,
            "weights": model.weights.tolist() if hasattr(model, "weights") else None,
            "bias": float(model.bias)
            if hasattr(model, "bias") and model.bias is not None
            else None,
            "tree": model.tree if hasattr(model, "tree") else None,
        }

    models_path = os.path.join(models_dir, "trained_models.pkl")
    with open(models_path, "wb") as f:
        pickle.dump(trained_models, f)

    # Save evaluation results
    evaluation_results = {"model_comparison": {}, "best_model": None, "best_r2": 0}

    for name, result in results.items():
        evaluation_results["model_comparison"][name] = {
            "mse": float(result["mse"]),
            "rmse": float(result["rmse"]),
            "r2": float(result["r2"]),
        }
        if result["r2"] > evaluation_results["best_r2"]:
            evaluation_results["best_model"] = name
            evaluation_results["best_r2"] = float(result["r2"])

    results_path = os.path.join(models_dir, "evaluation_results.json")
    with open(results_path, "w") as f:
        json.dump(evaluation_results, f, indent=2)

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"\nBest Model: {evaluation_results['best_model']}")
    print(f"Best R² Score: {evaluation_results['best_r2']:.4f}")
    print(f"\nModels saved to: {models_dir}/")

    return trained_models, preprocessing_params, evaluation_results


def predict_score(
    batting_team,
    bowling_team,
    city,
    runs,
    overs,
    wickets,
    model_name="Linear Regression",
):
    """
    Predict final score using core parameters

    Args:
        batting_team: Name of batting team
        bowling_team: Name of bowling team
        city: Match city
        runs: Current runs scored
        overs: Overs bowled (0-20)
        wickets: Wickets lost (0-10)
        model_name: Model to use

    Returns:
        Predicted final score
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    models_dir = os.path.join(base_dir, "models")

    with open(os.path.join(models_dir, "preprocessing_params.pkl"), "rb") as f:
        params = pickle.load(f)

    with open(os.path.join(models_dir, "trained_models.pkl"), "rb") as f:
        trained_models = pickle.load(f)

    encoders = params["encoders"]
    mean = np.array(params["mean"])
    std = np.array(params["std"])

    bat_encoded = encoders.get("batting_team", {}).get(batting_team, 0)
    bowl_encoded = encoders.get("bowling_team", {}).get(bowling_team, 0)
    city_encoded = encoders.get("city", {}).get(city, 0)

    features = np.array(
        [
            bat_encoded,
            bowl_encoded,
            city_encoded,
            float(runs),
            float(overs),
            float(wickets),
        ]
    ).reshape(1, -1)

    features_normalized = (features - mean) / std

    if model_name not in trained_models:
        model_name = "Linear Regression"

    model = trained_models[model_name]["model"]
    prediction = model.predict(features_normalized)

    return max(0, prediction[0])


if __name__ == "__main__":
    main()

    print("\n" + "=" * 70)
    print("SAMPLE PREDICTION")
    print("=" * 70)

    try:
        predicted = predict_score(
            batting_team="Mumbai Indians",
            bowling_team="Chennai Super Kings",
            city="Mumbai",
            runs=85,
            overs=10.0,
            wickets=3,
            model_name="Linear Regression",
        )
        print(f"\nInput: {85} runs, {10.0} overs, {3} wickets")
        print(f"Predicted Final Score: {predicted:.0f}")
    except FileNotFoundError:
        print("\nModels not trained yet. Run training first.")
