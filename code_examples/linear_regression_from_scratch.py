"""
Linear Regression Implementation from Scratch
============================================

This module demonstrates how to build linear regression from the ground up,
helping you understand the mathematical foundations of machine learning.

Source: "The Elements of Statistical Learning" - Hastie, Tibshirani, Friedman
Chapter 3: Linear Methods for Regression

Author: ML Learning Journey
Date: 2024
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class LinearRegressionFromScratch:
    """
    Linear Regression implementation using gradient descent and normal equation.

    This implementation shows two approaches:
    1. Gradient Descent (iterative optimization)
    2. Normal Equation (closed-form solution)

    Mathematical Foundation:
    ----------------------
    Linear model: y = Xw + b
    Cost function: J(w) = (1/2m) * sum((y_pred - y_true)²)
    Gradient: ∇J(w) = (1/m) * X^T * (y_pred - y_true)

    Source: "Pattern Recognition and Machine Learning" - Bishop, Section 3.1
    """

    def __init__(self, learning_rate: float = 0.01, max_iterations: int = 1000,
                 tolerance: float = 1e-6):
        """
        Initialize Linear Regression model.

        Parameters:
        -----------
        learning_rate : float
            Step size for gradient descent
        max_iterations : int
            Maximum number of training iterations
        tolerance : float
            Convergence tolerance for cost function
        """
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.weights = None
        self.bias = None
        self.cost_history = []

    def _add_bias_term(self, X: np.ndarray) -> np.ndarray:
        """Add bias term (column of ones) to feature matrix."""
        return np.column_stack([np.ones(X.shape[0]), X])

    def _compute_cost(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Compute Mean Squared Error cost function.

        Formula: J = (1/2m) * sum((y_pred - y_true)²)

        Source: "Introduction to Statistical Learning" - James et al., Section 3.1
        """
        m = len(y_true)
        cost = (1 / (2 * m)) * np.sum((y_pred - y_true) ** 2)
        return cost

    def fit_gradient_descent(self, X: np.ndarray, y: np.ndarray, verbose: bool = True) -> None:
        """
        Train the model using gradient descent optimization.

        Algorithm:
        ----------
        1. Initialize weights randomly
        2. For each iteration:
           a. Compute predictions: y_pred = X * w
           b. Compute cost: J = (1/2m) * sum((y_pred - y)²)
           c. Compute gradients: ∇J = (1/m) * X^T * (y_pred - y)
           d. Update weights: w = w - α * ∇J
        3. Repeat until convergence

        Source: "Deep Learning" - Goodfellow, Bengio, Courville, Chapter 4
        """
        # Add bias term
        X_with_bias = self._add_bias_term(X)
        m, n = X_with_bias.shape

        # Initialize weights randomly (small values)
        np.random.seed(42)
        self.weights = np.random.normal(0, 0.01, n)

        self.cost_history = []

        if verbose:
            print(f"🎓 Training Linear Regression with Gradient Descent")
            print(f"📊 Dataset: {m} samples, {n-1} features")
            print(f"⚙️  Learning rate: {self.learning_rate}")
            print(f"🔄 Max iterations: {self.max_iterations}")
            print("-" * 50)

        for iteration in range(self.max_iterations):
            # Forward pass: compute predictions
            y_pred = X_with_bias @ self.weights

            # Compute cost
            cost = self._compute_cost(y, y_pred)
            self.cost_history.append(cost)

            # Compute gradients
            gradients = (1 / m) * X_with_bias.T @ (y_pred - y)

            # Update weights
            self.weights -= self.learning_rate * gradients

            # Check for convergence
            if len(self.cost_history) > 1:
                cost_change = abs(self.cost_history[-2] - self.cost_history[-1])
                if cost_change < self.tolerance:
                    if verbose:
                        print(f"✅ Converged at iteration {iteration + 1}")
                    break

            # Print progress
            if verbose and (iteration + 1) % 100 == 0:
                print(f"Iteration {iteration + 1}: Cost = {cost:.6f}")

        # Extract bias and weights
        self.bias = self.weights[0]
        self.weights = self.weights[1:]

        if verbose:
            print(f"🎯 Final cost: {self.cost_history[-1]:.6f}")
            print(f"📈 Bias: {self.bias:.4f}")
            print(f"⚖️  Weights: {self.weights}")

    def fit_normal_equation(self, X: np.ndarray, y: np.ndarray, verbose: bool = True) -> None:
        """
        Train the model using the normal equation (closed-form solution).

        Mathematical Formula:
        --------------------
        w = (X^T * X)^(-1) * X^T * y

        This gives the optimal weights in one step, without iteration.

        Advantages:
        - No need to choose learning rate
        - Guaranteed to find global minimum
        - No iterations needed

        Disadvantages:
        - Computationally expensive for large datasets O(n³)
        - Requires matrix inversion

        Source: "The Elements of Statistical Learning" - Hastie et al., Section 3.2
        """
        # Add bias term
        X_with_bias = self._add_bias_term(X)

        if verbose:
            print(f"🎓 Training Linear Regression with Normal Equation")
            print(f"📊 Dataset: {X.shape[0]} samples, {X.shape[1]} features")
            print("-" * 50)

        try:
            # Normal equation: w = (X^T * X)^(-1) * X^T * y
            XTX = X_with_bias.T @ X_with_bias
            XTy = X_with_bias.T @ y
            weights = np.linalg.solve(XTX, XTy)  # More stable than inv(XTX) @ XTy

            # Extract bias and weights
            self.bias = weights[0]
            self.weights = weights[1:]

            # Compute final cost
            y_pred = self.predict(X)
            final_cost = self._compute_cost(y, y_pred)
            self.cost_history = [final_cost]

            if verbose:
                print(f"✅ Normal equation solved successfully")
                print(f"🎯 Final cost: {final_cost:.6f}")
                print(f"📈 Bias: {self.bias:.4f}")
                print(f"⚖️  Weights: {self.weights}")

        except np.linalg.LinAlgError:
            raise ValueError("Matrix is singular. Try adding regularization or removing correlated features.")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using the trained model.

        Formula: y_pred = X * w + b
        """
        if self.weights is None:
            raise ValueError("Model must be trained before making predictions.")

        return X @ self.weights + self.bias

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Calculate R² score (coefficient of determination).

        R² = 1 - SS_res / SS_tot
        where:
        - SS_res = sum of squares of residuals
        - SS_tot = total sum of squares

        R² ranges from 0 to 1, where 1 indicates perfect fit.

        Source: "Introduction to Statistical Learning" - James et al., Section 3.1.3
        """
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = 1 - (ss_res / ss_tot)
        return r2


def generate_sample_data(n_samples: int = 100, noise_level: float = 0.1,
                        random_seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic linear regression data for demonstration.

    Parameters:
    -----------
    n_samples : int
        Number of data points
    noise_level : float
        Amount of noise to add (standard deviation)
    random_seed : int
        Random seed for reproducibility

    Returns:
    --------
    X : np.ndarray
        Feature matrix
    y : np.ndarray
        Target values
    """
    np.random.seed(random_seed)

    # Generate features
    X = np.random.randn(n_samples, 2)  # 2 features

    # True relationship: y = 3 + 2*x1 - 1.5*x2 + noise
    true_bias = 3.0
    true_weights = np.array([2.0, -1.5])

    y = true_bias + X @ true_weights + noise_level * np.random.randn(n_samples)

    return X, y, true_bias, true_weights


def visualize_training_process(model: LinearRegressionFromScratch,
                             X: np.ndarray, y: np.ndarray,
                             true_bias: float, true_weights: np.ndarray) -> None:
    """Create comprehensive visualizations of the training process."""

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('📊 Linear Regression Analysis', fontsize=16, fontweight='bold')

    # 1. Cost function over iterations (if gradient descent was used)
    if len(model.cost_history) > 1:
        axes[0, 0].plot(model.cost_history, 'b-', linewidth=2)
        axes[0, 0].set_title('📉 Cost Function Over Iterations')
        axes[0, 0].set_xlabel('Iteration')
        axes[0, 0].set_ylabel('Cost (MSE)')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].set_yscale('log')  # Log scale for better visualization
    else:
        axes[0, 0].text(0.5, 0.5, 'Normal Equation\n(Single Step Solution)',
                       ha='center', va='center', fontsize=14,
                       transform=axes[0, 0].transAxes)
        axes[0, 0].set_title('📉 Cost Function')

    # 2. Predictions vs Actual values
    y_pred = model.predict(X)
    axes[0, 1].scatter(y, y_pred, alpha=0.6, color='blue')

    # Perfect prediction line
    min_val, max_val = min(y.min(), y_pred.min()), max(y.max(), y_pred.max())
    axes[0, 1].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')

    axes[0, 1].set_title('🎯 Predictions vs Actual Values')
    axes[0, 1].set_xlabel('Actual Values')
    axes[0, 1].set_ylabel('Predicted Values')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Add R² score
    r2 = model.score(X, y)
    axes[0, 1].text(0.05, 0.95, f'R² = {r2:.3f}', transform=axes[0, 1].transAxes,
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # 3. Residuals plot
    residuals = y - y_pred
    axes[1, 0].scatter(y_pred, residuals, alpha=0.6, color='green')
    axes[1, 0].axhline(y=0, color='red', linestyle='--', linewidth=2)
    axes[1, 0].set_title('📊 Residuals Plot')
    axes[1, 0].set_xlabel('Predicted Values')
    axes[1, 0].set_ylabel('Residuals (Actual - Predicted)')
    axes[1, 0].grid(True, alpha=0.3)

    # 4. Model parameters comparison
    param_names = ['Bias', 'Weight 1', 'Weight 2']
    learned_params = [model.bias, model.weights[0], model.weights[1]]
    true_params = [true_bias, true_weights[0], true_weights[1]]

    x_pos = np.arange(len(param_names))
    width = 0.35

    axes[1, 1].bar(x_pos - width/2, true_params, width, label='True Parameters',
                  alpha=0.7, color='orange')
    axes[1, 1].bar(x_pos + width/2, learned_params, width, label='Learned Parameters',
                  alpha=0.7, color='blue')

    axes[1, 1].set_title('⚖️ Parameter Comparison')
    axes[1, 1].set_xlabel('Parameters')
    axes[1, 1].set_ylabel('Value')
    axes[1, 1].set_xticks(x_pos)
    axes[1, 1].set_xticklabels(param_names)
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    # Add parameter values as text
    for i, (true_val, learned_val) in enumerate(zip(true_params, learned_params)):
        axes[1, 1].text(i - width/2, true_val + 0.1, f'{true_val:.2f}',
                       ha='center', va='bottom', fontweight='bold')
        axes[1, 1].text(i + width/2, learned_val + 0.1, f'{learned_val:.2f}',
                       ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    plt.show()


def demo_linear_regression():
    """
    Complete demonstration of linear regression implementation.

    This function showcases:
    1. Data generation
    2. Model training with both methods
    3. Performance evaluation
    4. Visualizations
    """
    print("🚀 Linear Regression from Scratch Demo")
    print("=" * 60)

    # Generate sample data
    print("🎲 Generating synthetic data...")
    X, y, true_bias, true_weights = generate_sample_data(n_samples=200, noise_level=0.5)
    print(f"📊 Generated {X.shape[0]} samples with {X.shape[1]} features")
    print(f"🎯 True relationship: y = {true_bias} + {true_weights[0]}*x1 + {true_weights[1]}*x2 + noise")
    print()

    # Method 1: Gradient Descent
    print("=" * 60)
    print("METHOD 1: GRADIENT DESCENT")
    print("=" * 60)

    model_gd = LinearRegressionFromScratch(learning_rate=0.01, max_iterations=1000)
    model_gd.fit_gradient_descent(X, y, verbose=True)

    print()
    print("📊 Gradient Descent Results:")
    print(f"🎯 R² Score: {model_gd.score(X, y):.4f}")
    print(f"📈 Final Cost: {model_gd.cost_history[-1]:.6f}")
    print(f"🔄 Iterations: {len(model_gd.cost_history)}")

    # Method 2: Normal Equation
    print("\\n" + "=" * 60)
    print("METHOD 2: NORMAL EQUATION")
    print("=" * 60)

    model_ne = LinearRegressionFromScratch()
    model_ne.fit_normal_equation(X, y, verbose=True)

    print()
    print("📊 Normal Equation Results:")
    print(f"🎯 R² Score: {model_ne.score(X, y):.4f}")
    print(f"📈 Final Cost: {model_ne.cost_history[-1]:.6f}")

    # Compare methods
    print("\\n" + "=" * 60)
    print("COMPARISON OF METHODS")
    print("=" * 60)

    print(f"Parameter Differences (Gradient Descent vs Normal Equation):")
    print(f"Bias difference: {abs(model_gd.bias - model_ne.bias):.8f}")
    print(f"Weight differences: {np.abs(model_gd.weights - model_ne.weights)}")

    # Visualizations
    print("\\n🎨 Creating visualizations...")
    visualize_training_process(model_gd, X, y, true_bias, true_weights)

    return model_gd, model_ne, X, y


if __name__ == "__main__":
    # Run the complete demonstration
    model_gd, model_ne, X, y = demo_linear_regression()

    print("\\n🎓 Learning Summary:")
    print("=" * 60)
    print("✅ Implemented linear regression from scratch")
    print("✅ Compared gradient descent vs normal equation")
    print("✅ Visualized training process and results")
    print("✅ Evaluated model performance with R² score")
    print("\\n💡 Key Insights:")
    print("- Gradient descent is iterative but scales well")
    print("- Normal equation is direct but computationally expensive")
    print("- Both methods converge to the same solution")
    print("- Visualization helps understand model behavior")

    print("\\n📚 Sources:")
    print("- Mathematical foundations: 'The Elements of Statistical Learning'")
    print("- Implementation details: 'Pattern Recognition and Machine Learning'")
    print("- Optimization theory: 'Deep Learning' by Goodfellow et al.")