# Module 06: Classical Machine Learning Algorithms

## 🎯 Learning Objectives
By the end of this module, you will:
- Master fundamental ML algorithms and their mathematical foundations
- Understand when to use each algorithm and their trade-offs
- Implement algorithms from scratch for deep understanding
- Apply proper model selection and evaluation techniques

## 🧠 The Classical ML Landscape

*Source: "The Elements of Statistical Learning" - Hastie, Tibshirani, Friedman, Chapter 1*
> "Statistical learning refers to a vast set of tools for understanding data. These tools can be classified as supervised or unsupervised."

### Algorithm Categories
- **Linear Models**: Simple, interpretable, strong baselines
- **Tree-Based**: Non-linear, interpretable, robust to outliers
- **Instance-Based**: Local learning, non-parametric
- **Probabilistic**: Uncertainty quantification, principled approach
- **Ensemble Methods**: Combining weak learners for strong performance

## 📊 Linear Models

### 6.1 Linear Regression Deep Dive

*Source: "Introduction to Statistical Learning" - James et al., Chapter 3*

```python
"""
Linear Regression: Theory and Implementation
==========================================

Linear regression is the foundation of statistical learning.
Understanding it deeply provides insights into all of ML.

Mathematical Foundation:
- Model: y = β₀ + β₁x₁ + ... + βₚxₚ + ε
- Cost: J(β) = (1/2m) Σ(yᵢ - ŷᵢ)²
- Solution: β̂ = (XᵀX)⁻¹Xᵀy (Normal Equation)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_regression, load_boston
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

print("📊 Linear Models Deep Dive")
print("=" * 50)

class LinearRegressionFromScratch:
    """
    Complete implementation of linear regression with multiple solvers

    Source: "Pattern Recognition and ML" - Bishop, Chapter 3
    """

    def __init__(self, method='normal_equation', learning_rate=0.01, max_iterations=1000):
        self.method = method
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.weights = None
        self.intercept = None
        self.cost_history = []

    def fit(self, X, y):
        """Train the model using specified method"""
        # Add intercept term
        X_with_intercept = np.column_stack([np.ones(X.shape[0]), X])

        if self.method == 'normal_equation':
            self._fit_normal_equation(X_with_intercept, y)
        elif self.method == 'gradient_descent':
            self._fit_gradient_descent(X_with_intercept, y)
        elif self.method == 'sgd':
            self._fit_sgd(X_with_intercept, y)
        else:
            raise ValueError("Method must be 'normal_equation', 'gradient_descent', or 'sgd'")

    def _fit_normal_equation(self, X, y):
        """Solve using normal equation: θ = (XᵀX)⁻¹Xᵀy"""
        try:
            # Use pseudo-inverse for numerical stability
            theta = np.linalg.pinv(X.T @ X) @ X.T @ y
            self.intercept = theta[0]
            self.weights = theta[1:]

            # Calculate final cost
            predictions = X @ theta
            cost = np.mean((predictions - y) ** 2) / 2
            self.cost_history = [cost]

        except np.linalg.LinAlgError:
            raise ValueError("Singular matrix encountered. Try regularization.")

    def _fit_gradient_descent(self, X, y):
        """Batch gradient descent implementation"""
        # Initialize parameters
        theta = np.random.normal(0, 0.01, X.shape[1])
        m = X.shape[0]

        for i in range(self.max_iterations):
            # Forward pass
            predictions = X @ theta

            # Compute cost
            cost = np.mean((predictions - y) ** 2) / 2
            self.cost_history.append(cost)

            # Compute gradients
            gradients = (1/m) * X.T @ (predictions - y)

            # Update parameters
            theta = theta - self.learning_rate * gradients

            # Check for convergence
            if i > 0 and abs(self.cost_history[-2] - self.cost_history[-1]) < 1e-8:
                break

        self.intercept = theta[0]
        self.weights = theta[1:]

    def _fit_sgd(self, X, y):
        """Stochastic gradient descent implementation"""
        # Initialize parameters
        theta = np.random.normal(0, 0.01, X.shape[1])
        m = X.shape[0]

        for epoch in range(self.max_iterations):
            # Shuffle data
            indices = np.random.permutation(m)
            epoch_cost = 0

            for i in indices:
                # Single sample
                xi = X[i:i+1]
                yi = y[i:i+1]

                # Forward pass
                prediction = xi @ theta

                # Compute cost (for this sample)
                cost = (prediction - yi) ** 2 / 2
                epoch_cost += cost[0]

                # Compute gradient (for this sample)
                gradient = xi.T @ (prediction - yi)

                # Update parameters
                theta = theta - self.learning_rate * gradient.flatten()

            # Average cost for epoch
            self.cost_history.append(epoch_cost / m)

        self.intercept = theta[0]
        self.weights = theta[1:]

    def predict(self, X):
        """Make predictions"""
        if self.weights is None:
            raise ValueError("Model must be fitted before prediction")
        return X @ self.weights + self.intercept

    def score(self, X, y):
        """Calculate R² score"""
        predictions = self.predict(X)
        ss_res = np.sum((y - predictions) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - (ss_res / ss_tot)

def demonstrate_linear_regression_methods():
    """Compare different solution methods for linear regression"""

    print("\\n🔧 Comparing Linear Regression Solution Methods")
    print("-" * 50)

    # Generate regression data
    np.random.seed(42)
    X, y = make_regression(n_samples=1000, n_features=3, noise=10, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Standardize features for gradient descent
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    methods = {
        'Normal Equation': ('normal_equation', X_train, y_train, X_test, y_test),
        'Gradient Descent': ('gradient_descent', X_train_scaled, y_train, X_test_scaled, y_test),
        'SGD': ('sgd', X_train_scaled, y_train, X_test_scaled, y_test)
    }

    results = {}

    for name, (method, X_tr, y_tr, X_te, y_te) in methods.items():
        print(f"\\nTraining with {name}...")

        # Train model
        model = LinearRegressionFromScratch(method=method, learning_rate=0.01, max_iterations=1000)

        import time
        start_time = time.time()
        model.fit(X_tr, y_tr)
        training_time = time.time() - start_time

        # Evaluate
        train_score = model.score(X_tr, y_tr)
        test_score = model.score(X_te, y_te)

        results[name] = {
            'model': model,
            'train_score': train_score,
            'test_score': test_score,
            'training_time': training_time,
            'iterations': len(model.cost_history)
        }

        print(f"  Training time: {training_time:.4f}s")
        print(f"  Iterations: {len(model.cost_history)}")
        print(f"  Train R²: {train_score:.4f}")
        print(f"  Test R²: {test_score:.4f}")

    # Visualizations
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('🔧 Linear Regression Methods Comparison', fontsize=16, fontweight='bold')

    # Cost function convergence
    for name, result in results.items():
        if len(result['model'].cost_history) > 1:  # Skip normal equation
            axes[0, 0].plot(result['model'].cost_history, label=name, linewidth=2)

    axes[0, 0].set_title('Cost Function Convergence')
    axes[0, 0].set_xlabel('Iteration')
    axes[0, 0].set_ylabel('Cost')
    axes[0, 0].set_yscale('log')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Performance comparison
    methods_names = list(results.keys())
    train_scores = [results[name]['train_score'] for name in methods_names]
    test_scores = [results[name]['test_score'] for name in methods_names]

    x = np.arange(len(methods_names))
    width = 0.35

    axes[0, 1].bar(x - width/2, train_scores, width, label='Train R²', alpha=0.7)
    axes[0, 1].bar(x + width/2, test_scores, width, label='Test R²', alpha=0.7)
    axes[0, 1].set_title('Performance Comparison')
    axes[0, 1].set_ylabel('R² Score')
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(methods_names)
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Training time comparison
    training_times = [results[name]['training_time'] for name in methods_names]
    colors = ['skyblue', 'lightcoral', 'lightgreen']

    bars = axes[1, 0].bar(methods_names, training_times, color=colors, alpha=0.7)
    axes[1, 0].set_title('Training Time Comparison')
    axes[1, 0].set_ylabel('Time (seconds)')
    axes[1, 0].set_yscale('log')

    # Add value labels
    for bar, time_val in zip(bars, training_times):
        axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                       f'{time_val:.4f}s', ha='center', va='bottom')

    # Predictions vs actual (using normal equation model)
    normal_eq_model = results['Normal Equation']['model']
    y_pred = normal_eq_model.predict(X_test)

    axes[1, 1].scatter(y_test, y_pred, alpha=0.6)
    axes[1, 1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
                   'r--', linewidth=2, label='Perfect Prediction')
    axes[1, 1].set_title('Predictions vs Actual (Normal Equation)')
    axes[1, 1].set_xlabel('Actual Values')
    axes[1, 1].set_ylabel('Predicted Values')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    return results

# Run the demonstration
linear_results = demonstrate_linear_regression_methods()
```

### 6.2 Regularized Linear Models

*Source: "The Elements of Statistical Learning" - Hastie et al., Chapter 3*

```python
def demonstrate_regularization():
    """
    Demonstrate Ridge, Lasso, and Elastic Net regularization

    Mathematical Background:
    - Ridge: J(β) = MSE + α Σβᵢ² (L2 penalty)
    - Lasso: J(β) = MSE + α Σ|βᵢ| (L1 penalty)
    - Elastic Net: J(β) = MSE + α₁ Σ|βᵢ| + α₂ Σβᵢ²

    Source: "Introduction to Statistical Learning" - James et al., Chapter 6
    """

    print("\\n🎯 Regularization Techniques")
    print("-" * 40)

    # Create high-dimensional data with multicollinearity
    np.random.seed(42)
    n_samples, n_features = 100, 50
    X = np.random.randn(n_samples, n_features)

    # Create correlated features
    for i in range(5, 10):
        X[:, i] = X[:, i-5] + 0.1 * np.random.randn(n_samples)

    # True coefficients (sparse)
    true_coef = np.zeros(n_features)
    true_coef[0:5] = [1.5, -2.0, 0.5, 3.0, -1.5]

    y = X @ true_coef + 0.1 * np.random.randn(n_samples)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Compare different regularization methods
    alphas = np.logspace(-3, 2, 20)  # Range of regularization strengths

    models = {
        'Ridge': Ridge,
        'Lasso': Lasso,
        'Elastic Net': lambda alpha: ElasticNet(alpha=alpha, l1_ratio=0.5)
    }

    results = {}

    for name, model_class in models.items():
        train_scores = []
        test_scores = []
        coefficients = []

        for alpha in alphas:
            model = model_class(alpha)
            model.fit(X_train_scaled, y_train)

            train_score = model.score(X_train_scaled, y_train)
            test_score = model.score(X_test_scaled, y_test)

            train_scores.append(train_score)
            test_scores.append(test_score)
            coefficients.append(model.coef_.copy())

        results[name] = {
            'train_scores': train_scores,
            'test_scores': test_scores,
            'coefficients': np.array(coefficients)
        }

    # Visualizations
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('🎯 Regularization Comparison', fontsize=16, fontweight='bold')

    # 1. Regularization paths
    colors = ['blue', 'red', 'green']

    for i, (name, result) in enumerate(results.items()):
        # Plot coefficient paths for first 10 features
        for j in range(min(10, n_features)):
            axes[0, 0].plot(alphas, result['coefficients'][:, j],
                           color=colors[i], alpha=0.7, linewidth=1)

    axes[0, 0].set_xscale('log')
    axes[0, 0].set_xlabel('Regularization Strength (α)')
    axes[0, 0].set_ylabel('Coefficient Value')
    axes[0, 0].set_title('Regularization Paths (First 10 Features)')
    axes[0, 0].grid(True, alpha=0.3)

    # 2. Bias-variance tradeoff
    for i, (name, result) in enumerate(results.items()):
        axes[0, 1].plot(alphas, result['train_scores'], 'o-',
                       color=colors[i], label=f'{name} Train', alpha=0.7)
        axes[0, 1].plot(alphas, result['test_scores'], 's--',
                       color=colors[i], label=f'{name} Test', alpha=0.7)

    axes[0, 1].set_xscale('log')
    axes[0, 1].set_xlabel('Regularization Strength (α)')
    axes[0, 1].set_ylabel('R² Score')
    axes[0, 1].set_title('Bias-Variance Tradeoff')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # 3. Feature selection (Lasso)
    lasso_coefs = results['Lasso']['coefficients']
    feature_selection = np.sum(np.abs(lasso_coefs) > 0.01, axis=1)

    axes[1, 0].plot(alphas, feature_selection, 'o-', color='red', linewidth=2)
    axes[1, 0].set_xscale('log')
    axes[1, 0].set_xlabel('Regularization Strength (α)')
    axes[1, 0].set_ylabel('Number of Selected Features')
    axes[1, 0].set_title('Feature Selection with Lasso')
    axes[1, 0].grid(True, alpha=0.3)

    # 4. Coefficient comparison at optimal alpha
    # Find optimal alpha for each method (highest test score)
    optimal_results = {}
    for name, result in results.items():
        optimal_idx = np.argmax(result['test_scores'])
        optimal_alpha = alphas[optimal_idx]
        optimal_coefs = result['coefficients'][optimal_idx]
        optimal_results[name] = {
            'alpha': optimal_alpha,
            'coefficients': optimal_coefs,
            'test_score': result['test_scores'][optimal_idx]
        }

    x_pos = np.arange(min(15, n_features))  # Show first 15 features
    width = 0.25

    for i, (name, result) in enumerate(optimal_results.items()):
        coefs = result['coefficients'][:15]  # First 15 features
        axes[1, 1].bar(x_pos + i*width, coefs, width, label=name, alpha=0.7)

    # Add true coefficients
    axes[1, 1].bar(x_pos + 3*width, true_coef[:15], width,
                  label='True', alpha=0.7, color='black')

    axes[1, 1].set_xlabel('Feature Index')
    axes[1, 1].set_ylabel('Coefficient Value')
    axes[1, 1].set_title('Coefficient Comparison (Optimal α)')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # Print results
    print("\\n📊 Optimal Regularization Results:")
    for name, result in optimal_results.items():
        print(f"{name:12}: α = {result['alpha']:.4f}, Test R² = {result['test_score']:.4f}")

    print("\\n💡 Key Insights:")
    print("• Ridge: Shrinks coefficients smoothly, handles multicollinearity")
    print("• Lasso: Performs feature selection, creates sparse models")
    print("• Elastic Net: Combines benefits of Ridge and Lasso")
    print("• Optimal regularization balances bias and variance")

    return results, optimal_results

# Run regularization demonstration
regularization_results = demonstrate_regularization()
```

## 🌳 Tree-Based Methods

### 6.3 Decision Trees and Ensemble Methods

*Source: "Introduction to Statistical Learning" - James et al., Chapter 8*

```python
"""
Tree-Based Methods: From Single Trees to Powerful Ensembles
==========================================================

Decision trees are intuitive, interpretable, but prone to overfitting.
Ensemble methods combine multiple trees for robust performance.

Key Concepts:
- Recursive binary splitting
- Impurity measures (Gini, Entropy)
- Pruning for regularization
- Bootstrap aggregating (Bagging)
- Random feature selection
- Boosting algorithms
"""

from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, plot_tree
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.datasets import make_classification, load_wine
from sklearn.model_selection import validation_curve, learning_curve
import matplotlib.pyplot as plt

def demonstrate_decision_trees():
    """Comprehensive decision tree analysis"""

    print("\\n🌳 Decision Tree Analysis")
    print("-" * 40)

    # Generate classification data
    X, y = make_classification(n_samples=1000, n_features=2, n_redundant=0,
                              n_informative=2, n_clusters_per_class=1, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Different tree complexities
    max_depths = [1, 3, 5, 10, None]
    min_samples_splits = [2, 10, 50]

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('🌳 Decision Tree Complexity Analysis', fontsize=16, fontweight='bold')

    # 1. Effect of max_depth
    train_scores_depth = []
    test_scores_depth = []

    for depth in max_depths:
        if depth is None:
            depth_val = 20  # For plotting
        else:
            depth_val = depth

        tree = DecisionTreeClassifier(max_depth=depth, random_state=42)
        tree.fit(X_train, y_train)

        train_score = tree.score(X_train, y_train)
        test_score = tree.score(X_test, y_test)

        train_scores_depth.append(train_score)
        test_scores_depth.append(test_score)

    depth_values = [1, 3, 5, 10, 20]
    axes[0, 0].plot(depth_values, train_scores_depth, 'o-', label='Training', linewidth=2)
    axes[0, 0].plot(depth_values, test_scores_depth, 's-', label='Testing', linewidth=2)
    axes[0, 0].set_xlabel('Max Depth')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].set_title('Effect of Tree Depth')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # 2. Decision boundaries at different depths
    depths_to_show = [1, 3, 10]
    for i, depth in enumerate(depths_to_show):
        ax = axes[0, 1] if i == 0 else axes[0, 2] if i == 1 else axes[1, 0]

        tree = DecisionTreeClassifier(max_depth=depth, random_state=42)
        tree.fit(X_train, y_train)

        # Plot decision boundary
        h = 0.02
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                           np.arange(y_min, y_max, h))

        Z = tree.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        ax.contourf(xx, yy, Z, alpha=0.4, cmap=plt.cm.RdYlBu)
        scatter = ax.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdYlBu)
        ax.set_title(f'Decision Boundary (depth={depth})')
        ax.set_xlabel('Feature 1')
        ax.set_ylabel('Feature 2')

    # 3. Feature importance
    tree_full = DecisionTreeClassifier(max_depth=5, random_state=42)
    tree_full.fit(X_train, y_train)

    feature_names = ['Feature 1', 'Feature 2']
    importances = tree_full.feature_importances_

    bars = axes[1, 1].bar(feature_names, importances, color=['skyblue', 'lightcoral'])
    axes[1, 1].set_title('Feature Importance')
    axes[1, 1].set_ylabel('Importance Score')

    # Add value labels
    for bar, importance in zip(bars, importances):
        axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                       f'{importance:.3f}', ha='center', va='bottom')

    # 4. Tree visualization (simplified)
    tree_simple = DecisionTreeClassifier(max_depth=3, random_state=42)
    tree_simple.fit(X_train, y_train)

    plot_tree(tree_simple, ax=axes[1, 2], feature_names=feature_names,
             class_names=['Class 0', 'Class 1'], filled=True, fontsize=8)
    axes[1, 2].set_title('Tree Structure (depth=3)')

    plt.tight_layout()
    plt.show()

    return tree_full

def demonstrate_ensemble_methods():
    """Compare different ensemble methods"""

    print("\\n🌲 Ensemble Methods Comparison")
    print("-" * 40)

    # Load wine dataset for multi-class classification
    wine = load_wine()
    X, y = wine.data, wine.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Standardize features for some algorithms
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Define ensemble methods
    ensembles = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
        'AdaBoost': AdaBoostClassifier(n_estimators=100, random_state=42),
        'Single Tree': DecisionTreeClassifier(max_depth=5, random_state=42)
    }

    results = {}

    # Train and evaluate each method
    for name, model in ensembles.items():
        model.fit(X_train, y_train)

        train_score = model.score(X_train, y_train)
        test_score = model.score(X_test, y_test)

        results[name] = {
            'model': model,
            'train_score': train_score,
            'test_score': test_score
        }

        print(f"{name:16}: Train = {train_score:.4f}, Test = {test_score:.4f}")

    # Visualizations
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('🌲 Ensemble Methods Analysis', fontsize=16, fontweight='bold')

    # 1. Performance comparison
    method_names = list(results.keys())
    train_scores = [results[name]['train_score'] for name in method_names]
    test_scores = [results[name]['test_score'] for name in method_names]

    x = np.arange(len(method_names))
    width = 0.35

    axes[0, 0].bar(x - width/2, train_scores, width, label='Train', alpha=0.7)
    axes[0, 0].bar(x + width/2, test_scores, width, label='Test', alpha=0.7)
    axes[0, 0].set_title('Performance Comparison')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(method_names, rotation=45)
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # 2. Feature importance comparison (Random Forest vs Gradient Boosting)
    rf_importance = results['Random Forest']['model'].feature_importances_
    gb_importance = results['Gradient Boosting']['model'].feature_importances_

    # Show top 10 features
    top_features = np.argsort(rf_importance)[-10:]

    x_pos = np.arange(len(top_features))
    width = 0.35

    axes[0, 1].barh(x_pos - width/2, rf_importance[top_features], width,
                   label='Random Forest', alpha=0.7)
    axes[0, 1].barh(x_pos + width/2, gb_importance[top_features], width,
                   label='Gradient Boosting', alpha=0.7)
    axes[0, 1].set_title('Feature Importance (Top 10)')
    axes[0, 1].set_xlabel('Importance Score')
    axes[0, 1].set_yticks(x_pos)
    axes[0, 1].set_yticklabels([wine.feature_names[i] for i in top_features])
    axes[0, 1].legend()

    # 3. Learning curves for Random Forest
    train_sizes, train_scores_lc, val_scores_lc = learning_curve(
        RandomForestClassifier(n_estimators=50, random_state=42),
        X_train, y_train, cv=5, train_sizes=np.linspace(0.1, 1.0, 10),
        random_state=42
    )

    train_mean = np.mean(train_scores_lc, axis=1)
    train_std = np.std(train_scores_lc, axis=1)
    val_mean = np.mean(val_scores_lc, axis=1)
    val_std = np.std(val_scores_lc, axis=1)

    axes[1, 0].plot(train_sizes, train_mean, 'o-', color='blue', label='Training')
    axes[1, 0].fill_between(train_sizes, train_mean - train_std, train_mean + train_std,
                           alpha=0.2, color='blue')

    axes[1, 0].plot(train_sizes, val_mean, 'o-', color='red', label='Validation')
    axes[1, 0].fill_between(train_sizes, val_mean - val_std, val_mean + val_std,
                           alpha=0.2, color='red')

    axes[1, 0].set_title('Random Forest Learning Curves')
    axes[1, 0].set_xlabel('Training Set Size')
    axes[1, 0].set_ylabel('Accuracy Score')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # 4. Number of estimators effect
    n_estimators_range = [10, 25, 50, 100, 200]
    rf_scores = []
    gb_scores = []

    for n_est in n_estimators_range:
        rf = RandomForestClassifier(n_estimators=n_est, random_state=42)
        gb = GradientBoostingClassifier(n_estimators=n_est, random_state=42)

        rf.fit(X_train, y_train)
        gb.fit(X_train, y_train)

        rf_scores.append(rf.score(X_test, y_test))
        gb_scores.append(gb.score(X_test, y_test))

    axes[1, 1].plot(n_estimators_range, rf_scores, 'o-', label='Random Forest', linewidth=2)
    axes[1, 1].plot(n_estimators_range, gb_scores, 's-', label='Gradient Boosting', linewidth=2)
    axes[1, 1].set_title('Effect of Number of Estimators')
    axes[1, 1].set_xlabel('Number of Estimators')
    axes[1, 1].set_ylabel('Test Accuracy')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    return results

# Run demonstrations
tree_results = demonstrate_decision_trees()
ensemble_results = demonstrate_ensemble_methods()
```

## 🎯 Support Vector Machines

### 6.4 SVM Theory and Implementation

*Source: "Pattern Recognition and Machine Learning" - Bishop, Chapter 7*

```python
"""
Support Vector Machines: Maximum Margin Classification
====================================================

SVMs find the optimal hyperplane that maximizes the margin between classes.
They can handle non-linear data through the kernel trick.

Mathematical Foundation:
- Optimization: max margin subject to correct classification
- Dual formulation: α optimization with Lagrange multipliers
- Kernel trick: K(x, x') = φ(x)ᵀφ(x') for non-linear mapping
"""

from sklearn.svm import SVC, SVR
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import make_circles, make_moons
import numpy as np

def demonstrate_svm_concepts():
    """Demonstrate core SVM concepts with visualizations"""

    print("\\n🎯 Support Vector Machine Concepts")
    print("-" * 40)

    # Generate linearly separable data
    np.random.seed(42)
    X_linear = np.random.randn(100, 2)
    y_linear = (X_linear[:, 0] + X_linear[:, 1] > 0).astype(int)

    # Generate non-linearly separable data
    X_circles, y_circles = make_circles(n_samples=200, noise=0.1, factor=0.5, random_state=42)
    X_moons, y_moons = make_moons(n_samples=200, noise=0.1, random_state=42)

    datasets = [
        ('Linear', X_linear, y_linear),
        ('Circles', X_circles, y_circles),
        ('Moons', X_moons, y_moons)
    ]

    # Different kernels to try
    kernels = ['linear', 'rbf', 'poly']

    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    fig.suptitle('🎯 SVM with Different Kernels', fontsize=16, fontweight='bold')

    for i, (dataset_name, X, y) in enumerate(datasets):
        for j, kernel in enumerate(kernels):
            ax = axes[i, j]

            # Train SVM
            if kernel == 'poly':
                svm = SVC(kernel=kernel, degree=3, C=1.0, random_state=42)
            else:
                svm = SVC(kernel=kernel, C=1.0, random_state=42)

            svm.fit(X, y)

            # Create decision boundary
            h = 0.02
            x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
            y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
            xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                               np.arange(y_min, y_max, h))

            Z = svm.predict(np.c_[xx.ravel(), yy.ravel()])
            Z = Z.reshape(xx.shape)

            # Plot
            ax.contourf(xx, yy, Z, alpha=0.4, cmap=plt.cm.RdYlBu)
            scatter = ax.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdYlBu)

            # Highlight support vectors
            ax.scatter(X[svm.support_, 0], X[svm.support_, 1],
                      s=100, linewidth=1, facecolors='none', edgecolors='black')

            ax.set_title(f'{dataset_name} - {kernel.upper()} kernel')
            ax.set_xlabel('Feature 1')
            ax.set_ylabel('Feature 2')

            # Add accuracy in corner
            accuracy = svm.score(X, y)
            ax.text(0.02, 0.98, f'Acc: {accuracy:.3f}', transform=ax.transAxes,
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                   verticalalignment='top')

    plt.tight_layout()
    plt.show()

    return svm

def demonstrate_svm_hyperparameters():
    """Demonstrate effect of SVM hyperparameters"""

    print("\\n⚙️ SVM Hyperparameter Tuning")
    print("-" * 40)

    # Generate complex dataset
    X, y = make_circles(n_samples=300, noise=0.1, factor=0.5, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Parameter grids
    C_range = np.logspace(-2, 2, 5)
    gamma_range = np.logspace(-3, 1, 5)

    # Grid search for RBF kernel
    param_grid = {'C': C_range, 'gamma': gamma_range}
    grid = GridSearchCV(SVC(kernel='rbf'), param_grid, cv=5, scoring='accuracy')
    grid.fit(X_train, y_train)

    print(f"Best parameters: {grid.best_params_}")
    print(f"Best cross-validation score: {grid.best_score_:.4f}")
    print(f"Test score: {grid.score(X_test, y_test):.4f}")

    # Visualize parameter space
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('⚙️ SVM Hyperparameter Analysis', fontsize=16, fontweight='bold')

    # 1. Heatmap of parameter combinations
    scores = grid.cv_results_['mean_test_score'].reshape(len(C_range), len(gamma_range))

    im = axes[0].imshow(scores, interpolation='nearest', cmap='viridis')
    axes[0].set_xlabel('Gamma Index')
    axes[0].set_ylabel('C Index')
    axes[0].set_title('Cross-Validation Scores')

    # Add colorbar
    plt.colorbar(im, ax=axes[0])

    # 2. Effect of C parameter
    C_scores = []
    for C in C_range:
        svm = SVC(kernel='rbf', C=C, gamma='scale', random_state=42)
        scores = cross_val_score(svm, X_train, y_train, cv=5)
        C_scores.append(scores.mean())

    axes[1].semilogx(C_range, C_scores, 'o-', linewidth=2)
    axes[1].set_xlabel('C (Regularization Parameter)')
    axes[1].set_ylabel('Cross-Validation Score')
    axes[1].set_title('Effect of C Parameter')
    axes[1].grid(True, alpha=0.3)

    # 3. Effect of gamma parameter
    gamma_scores = []
    for gamma in gamma_range:
        svm = SVC(kernel='rbf', C=1.0, gamma=gamma, random_state=42)
        scores = cross_val_score(svm, X_train, y_train, cv=5)
        gamma_scores.append(scores.mean())

    axes[2].semilogx(gamma_range, gamma_scores, 's-', linewidth=2, color='red')
    axes[2].set_xlabel('Gamma (Kernel Coefficient)')
    axes[2].set_ylabel('Cross-Validation Score')
    axes[2].set_title('Effect of Gamma Parameter')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    return grid

# Run SVM demonstrations
svm_model = demonstrate_svm_concepts()
svm_grid = demonstrate_svm_hyperparameters()

print("\\n💡 Classical ML Algorithm Summary:")
print("=" * 50)
print("🔹 **Linear Models**:")
print("  • Simple, interpretable, good baselines")
print("  • Regularization prevents overfitting")
print("  • Work well with high-dimensional data")

print("\\n🔹 **Tree-Based Methods**:")
print("  • Handle non-linear relationships naturally")
print("  • Robust to outliers and missing values")
print("  • Ensembles reduce overfitting")

print("\\n🔹 **Support Vector Machines**:")
print("  • Maximum margin principle")
print("  • Kernel trick for non-linear data")
print("  • Effective in high dimensions")

print("\\n🔹 **Algorithm Selection Guidelines**:")
print("  • Start with simple baselines (linear models)")
print("  • Use tree-based methods for tabular data")
print("  • Try SVMs for high-dimensional problems")
print("  • Always use cross-validation for evaluation")
print("  • Consider ensemble methods for best performance")
```

## 📚 Sources & Further Reading

### 📖 **Primary Textbooks** (with specific chapter references):

1. **"Introduction to Statistical Learning"** - James, Witten, Hastie, Tibshirani
   - **🆓 Free PDF**: [https://www.statlearning.com/](https://www.statlearning.com/)
   - **📑 Key Chapters**:
     - Chapter 3: Linear Regression *(simple and multiple regression)*
     - Chapter 4: Classification *(logistic regression, LDA)*
     - Chapter 6: Linear Model Selection *(Ridge, Lasso, PCR)*
     - Chapter 8: Tree-Based Methods *(decision trees, random forests)*

2. **"The Elements of Statistical Learning"** - Hastie, Tibshirani, Friedman
   - **🆓 Free PDF**: [https://web.stanford.edu/~hastie/ElemStatLearn/](https://web.stanford.edu/~hastie/ElemStatLearn/)
   - **📑 Key Chapters**:
     - Chapter 3: Linear Methods for Regression *(regularization theory)*
     - Chapter 4: Linear Methods for Classification *(discriminant analysis)*
     - Chapter 9: Additive Models *(trees, MARS, boosting)*
     - Chapter 12: Support Vector Machines *(kernel methods)*

3. **"Pattern Recognition and Machine Learning"** - Christopher Bishop
   - **🆓 Free PDF**: [Microsoft Research](https://www.microsoft.com/en-us/research/uploads/prod/2006/01/Bishop-Pattern-Recognition-and-Machine-Learning-2006.pdf)
   - **📑 Key Chapters**:
     - Chapter 3: Linear Models for Regression *(Bayesian perspective)*
     - Chapter 4: Linear Models for Classification *(discriminative models)*
     - Chapter 7: Sparse Kernel Machines *(SVM theory)*

### 📑 **Seminal Papers**:

1. **"Regularization and Variable Selection via the Elastic Net"** - Zou & Hastie (2005)
   - **Source**: Journal of the Royal Statistical Society
   - **💡 Impact**: Introduced Elastic Net regularization

2. **"Random Forests"** - Leo Breiman (2001)
   - **Source**: Machine Learning Journal
   - **💡 Impact**: Established random forest algorithm

3. **"Support Vector Networks"** - Cortes & Vapnik (1995)
   - **Source**: Machine Learning Journal
   - **💡 Impact**: Foundational SVM paper

## ✅ Practical Exercises

### Exercise 1: Regularization Path Analysis
Implement your own regularization path algorithm for Lasso regression and compare with scikit-learn's implementation.

### Exercise 2: Ensemble Method Comparison
Build a voting classifier that combines different base algorithms and analyze when it outperforms individual models.

### Exercise 3: SVM from Scratch
Implement a simplified SVM using quadratic programming and compare with scikit-learn's SVC.

## 🚀 Next Steps

Ready for modern ML? Continue to [Module 07: Deep Learning](../07_Deep_Learning/README.md) where we'll explore neural networks and deep architectures!

---
*Estimated completion time: 8-10 hours*
*Prerequisites: Modules 02-04 recommended*