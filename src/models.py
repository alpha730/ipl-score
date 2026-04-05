"""
IPL Prediction Models
Implementing Linear Regression, Ridge, Lasso, and Decision Tree Regressor using NumPy only
"""

import numpy as np


class LinearRegression:
    """Linear Regression using Gradient Descent"""

    def __init__(self, learning_rate=0.01, n_iterations=1000, fit_intercept=True):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.fit_intercept = fit_intercept
        self.weights = None
        self.bias = None
        self.cost_history = []

    def _add_intercept(self, X):
        """Add intercept column to X"""
        if self.fit_intercept:
            return np.c_[np.ones((X.shape[0], 1)), X]
        return X

    def fit(self, X, y):
        """Fit the model using gradient descent"""
        X = self._add_intercept(X)
        n_samples, n_features = X.shape

        # Initialize weights
        self.weights = np.zeros(n_features)

        # Gradient descent
        for i in range(self.n_iterations):
            # Forward pass
            y_pred = X @ self.weights

            # Compute gradients
            error = y_pred - y
            gradient = (2 / n_samples) * (X.T @ error)

            # Update weights
            self.weights -= self.learning_rate * gradient

            # Store cost
            cost = np.mean(error ** 2)
            self.cost_history.append(cost)

        if self.fit_intercept:
            self.bias = self.weights[0]
            self.weights = self.weights[1:]

        return self

    def predict(self, X):
        """Predict target values"""
        X = self._add_intercept(X)
        return X @ self.weights if self.bias is None else X @ np.r_[self.bias, self.weights]

    def score(self, X, y):
        """Calculate R² score"""
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - (ss_res / ss_tot)


class RidgeRegression:
    """Ridge Regression (L2 Regularization)"""

    def __init__(self, alpha=1.0, learning_rate=0.01, n_iterations=1000, fit_intercept=True):
        self.alpha = alpha
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.fit_intercept = fit_intercept
        self.weights = None
        self.bias = None
        self.cost_history = []

    def _add_intercept(self, X):
        """Add intercept column to X"""
        if self.fit_intercept:
            return np.c_[np.ones((X.shape[0], 1)), X]
        return X

    def fit(self, X, y):
        """Fit the model using gradient descent with L2 regularization"""
        X = self._add_intercept(X)
        n_samples, n_features = X.shape

        # Initialize weights
        self.weights = np.zeros(n_features)

        # Gradient descent with L2 regularization
        for i in range(self.n_iterations):
            # Forward pass
            y_pred = X @ self.weights

            # Compute gradients with L2 penalty
            error = y_pred - y
            gradient = (2 / n_samples) * (X.T @ error) + (2 * self.alpha / n_samples) * self.weights

            # Update weights
            self.weights -= self.learning_rate * gradient

            # Store cost (MSE + regularization term)
            mse = np.mean(error ** 2)
            reg_term = self.alpha * np.sum(self.weights ** 2) / n_samples
            cost = mse + reg_term
            self.cost_history.append(cost)

        if self.fit_intercept:
            self.bias = self.weights[0]
            self.weights = self.weights[1:]

        return self

    def predict(self, X):
        """Predict target values"""
        X = self._add_intercept(X)
        return X @ self.weights if self.bias is None else X @ np.r_[self.bias, self.weights]

    def score(self, X, y):
        """Calculate R² score"""
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - (ss_res / ss_tot)


class LassoRegression:
    """Lasso Regression (L1 Regularization)"""

    def __init__(self, alpha=0.1, learning_rate=0.01, n_iterations=1000, fit_intercept=True):
        self.alpha = alpha
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.fit_intercept = fit_intercept
        self.weights = None
        self.bias = None
        self.cost_history = []

    def _add_intercept(self, X):
        """Add intercept column to X"""
        if self.fit_intercept:
            return np.c_[np.ones((X.shape[0], 1)), X]
        return X

    def _subgradient_l1(self, w):
        """Subgradient for L1 regularization"""
        return np.sign(w)

    def fit(self, X, y):
        """Fit the model using proximal gradient descent"""
        X = self._add_intercept(X)
        n_samples, n_features = X.shape

        # Initialize weights
        self.weights = np.zeros(n_features)

        # Proximal gradient descent
        for i in range(self.n_iterations):
            # Forward pass
            y_pred = X @ self.weights

            # Compute gradients
            error = y_pred - y
            gradient = (2 / n_samples) * (X.T @ error)

            # Gradient step
            self.weights -= self.learning_rate * gradient

            # Proximal step (soft thresholding) for L1
            threshold = self.learning_rate * self.alpha
            self.weights = np.sign(self.weights) * np.maximum(np.abs(self.weights) - threshold, 0)

            # Store cost
            mse = np.mean(error ** 2)
            l1_term = self.alpha * np.sum(np.abs(self.weights)) / n_samples
            cost = mse + l1_term
            self.cost_history.append(cost)

        if self.fit_intercept:
            self.bias = self.weights[0]
            self.weights = self.weights[1:]

        return self

    def predict(self, X):
        """Predict target values"""
        X = self._add_intercept(X)
        return X @ self.weights if self.bias is None else X @ np.r_[self.bias, self.weights]

    def score(self, X, y):
        """Calculate R² score"""
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - (ss_res / ss_tot)


class DecisionTreeRegressor:
    """Decision Tree Regressor implemented from scratch"""

    def __init__(self, max_depth=10, min_samples_split=2, min_samples_leaf=1):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.tree = None

    def _variance(self, y):
        """Calculate variance of target values"""
        if len(y) == 0:
            return 0
        return np.var(y)

    def _variance_reduction(self, y, left_indices, right_indices):
        """Calculate variance reduction for a split"""
        n = len(y)
        if n == 0:
            return 0

        left_y = y[left_indices]
        right_y = y[right_indices]

        n_left = len(left_y)
        n_right = len(right_y)

        if n_left == 0 or n_right == 0:
            return 0

        parent_var = self._variance(y)
        left_var = self._variance(left_y)
        right_var = self._variance(right_y)

        weighted_child_var = (n_left / n) * left_var + (n_right / n) * right_var

        return parent_var - weighted_child_var

    def _find_best_split(self, X, y):
        """Find the best feature and threshold to split on"""
        n_samples, n_features = X.shape

        if n_samples < self.min_samples_split:
            return None, None

        best_gain = -np.inf
        best_feature = None
        best_threshold = None

        for feature in range(n_features):
            thresholds = np.unique(X[:, feature])

            for threshold in thresholds:
                left_mask = X[:, feature] <= threshold
                right_mask = ~left_mask

                left_indices = np.where(left_mask)[0]
                right_indices = np.where(right_mask)[0]

                if len(left_indices) < self.min_samples_leaf or len(right_indices) < self.min_samples_leaf:
                    continue

                gain = self._variance_reduction(y, left_indices, right_indices)

                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold

        return best_feature, best_threshold

    def _build_tree(self, X, y, depth=0):
        """Recursively build the decision tree"""
        n_samples = len(y)

        # Stopping conditions
        if (depth >= self.max_depth or
            n_samples < self.min_samples_split or
            n_samples < 2 * self.min_samples_leaf):
            return {'leaf': True, 'value': np.mean(y)}

        # Find best split
        feature, threshold = self._find_best_split(X, y)

        if feature is None:
            return {'leaf': True, 'value': np.mean(y)}

        # Split data
        left_mask = X[:, feature] <= threshold
        right_mask = ~left_mask

        left_indices = np.where(left_mask)[0]
        right_indices = np.where(right_mask)[0]

        if len(left_indices) < self.min_samples_leaf or len(right_indices) < self.min_samples_leaf:
            return {'leaf': True, 'value': np.mean(y)}

        # Build subtrees
        left_subtree = self._build_tree(X[left_indices], y[left_indices], depth + 1)
        right_subtree = self._build_tree(X[right_indices], y[right_indices], depth + 1)

        return {
            'leaf': False,
            'feature': feature,
            'threshold': threshold,
            'left': left_subtree,
            'right': right_subtree
        }

    def fit(self, X, y):
        """Fit the decision tree"""
        self.tree = self._build_tree(X, y)
        return self

    def _predict_sample(self, x, node):
        """Predict for a single sample"""
        if node['leaf']:
            return node['value']

        if x[node['feature']] <= node['threshold']:
            return self._predict_sample(x, node['left'])
        else:
            return self._predict_sample(x, node['right'])

    def predict(self, X):
        """Predict target values"""
        return np.array([self._predict_sample(x, self.tree) for x in X])

    def score(self, X, y):
        """Calculate R² score"""
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - (ss_res / ss_tot)


class ModelComparator:
    """Compare and analyze multiple models"""

    def __init__(self):
        self.models = {}
        self.results = {}

    def add_model(self, name, model):
        """Add a model to the comparator"""
        self.models[name] = model

    def train_all(self, X_train, y_train):
        """Train all models"""
        for name, model in self.models.items():
            print(f"Training {name}...")
            model.fit(X_train, y_train)
            self.results[name] = {'trained': True}

    def evaluate_all(self, X_test, y_test):
        """Evaluate all models"""
        for name, model in self.models.items():
            if not self.results.get(name, {}).get('trained', False):
                continue

            y_pred = model.predict(X_test)

            # Calculate metrics
            mse = np.mean((y_test - y_pred) ** 2)
            rmse = np.sqrt(mse)
            ss_res = np.sum((y_test - y_pred) ** 2)
            ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
            r2 = 1 - (ss_res / ss_tot)

            self.results[name].update({
                'mse': mse,
                'rmse': rmse,
                'r2': r2,
                'predictions': y_pred
            })

        return self.results

    def get_best_model(self, metric='r2'):
        """Get the best model based on a metric"""
        best_model = None
        best_score = -np.inf if metric in ['r2'] else np.inf

        for name, result in self.results.items():
            if metric not in result:
                continue

            score = result[metric]
            if metric == 'r2' and score > best_score:
                best_score = score
                best_model = name
            elif metric in ['mse', 'rmse'] and score < best_score:
                best_score = score
                best_model = name

        return best_model, best_score

    def comparison_table(self):
        """Generate a comparison table"""
        print("\n" + "=" * 70)
        print(f"{'Model':<25} {'MSE':<15} {'RMSE':<15} {'R²':<15}")
        print("=" * 70)

        for name, result in sorted(self.results.items(),
                                    key=lambda x: x[1].get('r2', 0),
                                    reverse=True):
            if 'mse' in result:
                print(f"{name:<25} {result['mse']:<15.4f} {result['rmse']:<15.4f} {result['r2']:<15.4f}")

        print("=" * 70)

        best_model, best_r2 = self.get_best_model('r2')
        print(f"\nBest Model: {best_model} (R² = {best_r2:.4f})")

        return self.results


if __name__ == "__main__":
    # Test the models
    print("Testing IPL Prediction Models\n")

    # Generate sample data
    np.random.seed(42)
    n_samples = 1000

    X = np.random.randn(n_samples, 5)
    true_weights = np.array([10, 5, 2, 1, 0.5])
    y = X @ true_weights + np.random.randn(n_samples) * 10 + 150

    # Split data
    from data_preprocessing import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}\n")

    # Create comparator
    comparator = ModelComparator()

    # Add models
    comparator.add_model("Linear Regression", LinearRegression(learning_rate=0.01, n_iterations=1000))
    comparator.add_model("Ridge Regression", RidgeRegression(alpha=1.0, learning_rate=0.01, n_iterations=1000))
    comparator.add_model("Lasso Regression", LassoRegression(alpha=0.1, learning_rate=0.01, n_iterations=1000))
    comparator.add_model("Decision Tree", DecisionTreeRegressor(max_depth=10))

    # Train and evaluate
    comparator.train_all(X_train, y_train)
    comparator.evaluate_all(X_test, y_test)
    comparator.comparison_table()
