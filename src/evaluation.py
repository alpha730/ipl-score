"""
Performance Evaluation Module
MSE, RMSE, R² metrics and detailed model analysis
"""

import numpy as np
import json


def mean_squared_error(y_true, y_pred):
    """Calculate Mean Squared Error"""
    return np.mean((y_true - y_pred) ** 2)


def root_mean_squared_error(y_true, y_pred):
    """Calculate Root Mean Squared Error"""
    return np.sqrt(mean_squared_error(y_true, y_pred))


def r_squared(y_true, y_pred):
    """Calculate R² (Coefficient of Determination)"""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)

    if ss_tot == 0:
        return 0.0

    return 1 - (ss_res / ss_tot)


def mean_absolute_error(y_true, y_pred):
    """Calculate Mean Absolute Error"""
    return np.mean(np.abs(y_true - y_pred))


def mean_absolute_percentage_error(y_true, y_pred):
    """Calculate Mean Absolute Percentage Error"""
    # Avoid division by zero
    mask = y_true != 0
    if not np.any(mask):
        return 0.0

    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def adjusted_r_squared(y_true, y_pred, n_features):
    """Calculate Adjusted R²"""
    n = len(y_true)
    r2 = r_squared(y_true, y_pred)

    if n <= n_features + 1:
        return r2

    return 1 - (1 - r2) * (n - 1) / (n - n_features - 1)


def residual_analysis(y_true, y_pred):
    """Analyze residuals for model diagnostics"""
    residuals = y_true - y_pred

    analysis = {
        'mean_residual': float(np.mean(residuals)),
        'std_residual': float(np.std(residuals)),
        'min_residual': float(np.min(residuals)),
        'max_residual': float(np.max(residuals)),
        'median_residual': float(np.median(residuals)),
        'residual_skewness': float(np.mean(((residuals - np.mean(residuals)) / np.std(residuals)) ** 3)) if np.std(residuals) > 0 else 0,
        'residual_kurtosis': float(np.mean(((residuals - np.mean(residuals)) / np.std(residuals)) ** 4) - 3) if np.std(residuals) > 0 else 0
    }

    return analysis


def prediction_accuracy_ranges(y_true, y_pred, ranges=None):
    """Calculate accuracy within different prediction ranges"""
    if ranges is None:
        ranges = [
            (0, 50, "Low (0-50)"),
            (50, 100, "Medium (50-100)"),
            (100, 150, "High (100-150)"),
            (150, 200, "Very High (150-200)"),
            (200, float('inf'), "Extreme (200+)")
        ]

    accuracy_by_range = {}

    for low, high, label in ranges:
        mask = (y_true >= low) & (y_true < high)
        if np.any(mask):
            subset_true = y_true[mask]
            subset_pred = y_pred[mask]

            accuracy_by_range[label] = {
                'count': int(np.sum(mask)),
                'mae': float(mean_absolute_error(subset_true, subset_pred)),
                'rmse': float(root_mean_squared_error(subset_true, subset_pred)),
                'r2': float(r_squared(subset_true, subset_pred))
            }

    return accuracy_by_range


def cross_validation_score(model_class, X, y, k=5, **model_kwargs):
    """Perform k-fold cross-validation"""
    n_samples = len(X)
    fold_size = n_samples // k

    scores = {
        'mse': [],
        'rmse': [],
        'r2': [],
        'mae': []
    }

    indices = np.arange(n_samples)

    for fold in range(k):
        # Create fold indices
        val_start = fold * fold_size
        val_end = val_start + fold_size if fold < k - 1 else n_samples

        val_indices = indices[val_start:val_end]
        train_indices = np.concatenate([indices[:val_start], indices[val_end:]])

        # Split data
        X_train, X_val = X[train_indices], X[val_indices]
        y_train, y_val = y[train_indices], y[val_indices]

        # Train model
        model = model_class(**model_kwargs)
        model.fit(X_train, y_train)

        # Evaluate
        y_pred = model.predict(X_val)

        scores['mse'].append(mean_squared_error(y_val, y_pred))
        scores['rmse'].append(root_mean_squared_error(y_val, y_pred))
        scores['r2'].append(r_squared(y_val, y_pred))
        scores['mae'].append(mean_absolute_error(y_val, y_pred))

    # Aggregate results
    cv_results = {
        metric: {
            'mean': float(np.mean(values)),
            'std': float(np.std(values)),
            'min': float(np.min(values)),
            'max': float(np.max(values)),
            'fold_scores': [float(s) for s in values]
        }
        for metric, values in scores.items()
    }

    return cv_results


def model_comparison_analysis(results_dict, y_test):
    """Comprehensive comparison of multiple models"""
    comparison = {
        'models': {},
        'ranking': {},
        'summary': {}
    }

    for model_name, result in results_dict.items():
        if 'predictions' not in result:
            continue

        y_pred = result['predictions']

        comparison['models'][model_name] = {
            'mse': float(mean_squared_error(y_test, y_pred)),
            'rmse': float(root_mean_squared_error(y_test, y_pred)),
            'r2': float(r_squared(y_test, y_pred)),
            'mae': float(mean_absolute_error(y_test, y_pred)),
            'mape': float(mean_absolute_percentage_error(y_test, y_pred)),
            'residual_analysis': residual_analysis(y_test, y_pred)
        }

    # Rank models by R²
    r2_scores = [(name, data['r2']) for name, data in comparison['models'].items()]
    r2_scores.sort(key=lambda x: x[1], reverse=True)

    comparison['ranking']['by_r2'] = [{'model': name, 'r2': score} for name, score in r2_scores]

    # Rank models by RMSE
    rmse_scores = [(name, data['rmse']) for name, data in comparison['models'].items()]
    rmse_scores.sort(key=lambda x: x[1])

    comparison['ranking']['by_rmse'] = [{'model': name, 'rmse': score} for name, score in rmse_scores]

    # Summary statistics
    all_r2 = [data['r2'] for data in comparison['models'].values()]
    all_rmse = [data['rmse'] for data in comparison['models'].values()]

    comparison['summary'] = {
        'best_r2': {'model': r2_scores[0][0], 'value': r2_scores[0][1]} if r2_scores else None,
        'best_rmse': {'model': rmse_scores[0][0], 'value': rmse_scores[0][1]} if rmse_scores else None,
        'avg_r2': float(np.mean(all_r2)) if all_r2 else 0,
        'avg_rmse': float(np.mean(all_rmse)) if all_rmse else 0,
        'r2_spread': float(np.max(all_r2) - np.min(all_r2)) if all_r2 else 0
    }

    return comparison


def generate_evaluation_report(model_name, y_true, y_pred, X_test=None, include_residuals=True):
    """Generate a comprehensive evaluation report"""
    report = {
        'model_name': model_name,
        'dataset_info': {
            'n_samples': len(y_true),
            'n_features': X_test.shape[1] if X_test is not None else None
        },
        'target_stats': {
            'mean': float(np.mean(y_true)),
            'std': float(np.std(y_true)),
            'min': float(np.min(y_true)),
            'max': float(np.max(y_true)),
            'median': float(np.median(y_true))
        },
        'prediction_stats': {
            'mean': float(np.mean(y_pred)),
            'std': float(np.std(y_pred)),
            'min': float(np.min(y_pred)),
            'max': float(np.max(y_pred)),
            'median': float(np.median(y_pred))
        },
        'metrics': {
            'mse': float(mean_squared_error(y_true, y_pred)),
            'rmse': float(root_mean_squared_error(y_true, y_pred)),
            'r2': float(r_squared(y_true, y_pred)),
            'mae': float(mean_absolute_error(y_true, y_pred)),
            'mape': float(mean_absolute_percentage_error(y_true, y_pred))
        }
    }

    if X_test is not None:
        report['metrics']['adjusted_r2'] = float(adjusted_r_squared(y_true, y_pred, X_test.shape[1]))

    if include_residuals:
        report['residual_analysis'] = residual_analysis(y_true, y_pred)
        report['accuracy_by_range'] = prediction_accuracy_ranges(y_true, y_pred)

    return report


def print_evaluation_report(report):
    """Print a formatted evaluation report"""
    print("\n" + "=" * 70)
    print(f"EVALUATION REPORT: {report['model_name']}")
    print("=" * 70)

    print(f"\nDataset: {report['dataset_info']['n_samples']} samples", end="")
    if report['dataset_info']['n_features']:
        print(f", {report['dataset_info']['n_features']} features")
    else:
        print()

    print("\n--- Target Variable Statistics ---")
    for key, value in report['target_stats'].items():
        if value is not None:
            print(f"  {key}: {value:.4f}")

    print("\n--- Prediction Statistics ---")
    for key, value in report['prediction_stats'].items():
        if value is not None:
            print(f"  {key}: {value:.4f}")

    print("\n--- Performance Metrics ---")
    for key, value in report['metrics'].items():
        if value is not None:
            print(f"  {key.upper()}: {value:.4f}")

    if 'residual_analysis' in report:
        print("\n--- Residual Analysis ---")
        for key, value in report['residual_analysis'].items():
            print(f"  {key}: {value:.4f}")

    if 'accuracy_by_range' in report:
        print("\n--- Accuracy by Score Range ---")
        for range_name, metrics in report['accuracy_by_range'].items():
            print(f"\n  {range_name} ({metrics['count']} samples):")
            print(f"    MAE: {metrics['mae']:.4f}, RMSE: {metrics['rmse']:.4f}, R²: {metrics['r2']:.4f}")

    print("\n" + "=" * 70)


class PerformanceEvaluator:
    """Main class for model evaluation and comparison"""

    def __init__(self):
        self.models = {}
        self.results = {}
        self.y_test = None

    def add_model_result(self, name, predictions):
        """Add model predictions for evaluation"""
        self.models[name] = predictions

    def set_test_data(self, y_test):
        """Set the ground truth test data"""
        self.y_test = y_test

    def evaluate(self, model_name=None):
        """Evaluate one or all models"""
        if self.y_test is None:
            raise ValueError("Test data not set. Call set_test_data() first.")

        if model_name:
            models_to_eval = {model_name: self.models[model_name]}
        else:
            models_to_eval = self.models

        for name, predictions in models_to_eval.items():
            self.results[name] = generate_evaluation_report(
                name, self.y_test, predictions
            )

        return self.results

    def compare_models(self):
        """Compare all evaluated models"""
        if not self.results:
            raise ValueError("No models evaluated yet. Call evaluate() first.")

        # Prepare predictions dict for comparison
        results_dict = {
            name: {'predictions': self.models[name]}
            for name in self.results.keys()
        }

        comparison = model_comparison_analysis(results_dict, self.y_test)
        return comparison

    def print_comparison(self):
        """Print model comparison table"""
        comparison = self.compare_models()

        print("\n" + "=" * 90)
        print(f"{'Model':<25} {'MSE':<12} {'RMSE':<12} {'R²':<12} {'MAE':<12} {'MAPE(%)':<12}")
        print("=" * 90)

        # Sort by R²
        sorted_models = sorted(
            comparison['models'].items(),
            key=lambda x: x[1]['r2'],
            reverse=True
        )

        for name, metrics in sorted_models:
            print(f"{name:<25} {metrics['mse']:<12.4f} {metrics['rmse']:<12.4f} "
                  f"{metrics['r2']:<12.4f} {metrics['mae']:<12.4f} {metrics['mape']:<12.4f}")

        print("=" * 90)
        print(f"\nBest Model (R²): {comparison['summary']['best_r2']['model']} "
              f"with R² = {comparison['summary']['best_r2']['value']:.4f}")
        print(f"Best Model (RMSE): {comparison['summary']['best_rmse']['model']} "
              f"with RMSE = {comparison['summary']['best_rmse']['value']:.4f}")

        return comparison


if __name__ == "__main__":
    # Test the evaluation module
    print("Testing Performance Evaluation Module\n")

    np.random.seed(42)
    n_samples = 500

    # Generate sample predictions
    y_true = np.random.randn(n_samples) * 30 + 160
    y_pred_linear = y_true + np.random.randn(n_samples) * 15
    y_pred_ridge = y_true + np.random.randn(n_samples) * 16
    y_pred_lasso = y_true + np.random.randn(n_samples) * 17
    y_pred_tree = y_true + np.random.randn(n_samples) * 18

    # Create evaluator
    evaluator = PerformanceEvaluator()
    evaluator.set_test_data(y_true)

    evaluator.add_model_result("Linear Regression", y_pred_linear)
    evaluator.add_model_result("Ridge Regression", y_pred_ridge)
    evaluator.add_model_result("Lasso Regression", y_pred_lasso)
    evaluator.add_model_result("Decision Tree", y_pred_tree)

    # Evaluate all models
    evaluator.evaluate()

    # Print comparison
    evaluator.print_comparison()

    # Generate detailed report for best model
    print("\n\nDetailed Report for Linear Regression:")
    report = evaluator.results["Linear Regression"]
    print_evaluation_report(report)
