"""
Beginner Machine Learning Exercises
===================================

This module contains practical exercises for beginners to reinforce ML concepts.
Each exercise includes the problem, hints, solution, and educational explanations.

Sources:
- "Introduction to Statistical Learning" - James, Witten, Hastie, Tibshirani
- "Pattern Recognition and Machine Learning" - Christopher Bishop
- "Hands-On Machine Learning" - Aurélien Géron

Learning Objectives:
- Understand basic ML concepts through hands-on practice
- Implement simple algorithms from scratch
- Visualize and interpret results
- Build intuition for model selection and evaluation

Author: ML Learning Journey
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_regression, make_classification, load_iris, load_boston
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Set style for consistent plotting
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class Exercise:
    """Base class for exercises with common functionality"""

    def __init__(self, title, difficulty, concepts):
        self.title = title
        self.difficulty = difficulty
        self.concepts = concepts

    def print_header(self):
        """Print exercise header"""
        print("=" * 60)
        print(f"📚 {self.title}")
        print(f"🎯 Difficulty: {self.difficulty}")
        print(f"🧠 Concepts: {', '.join(self.concepts)}")
        print("=" * 60)

    def print_solution_header(self):
        """Print solution header"""
        print("\\n" + "💡 SOLUTION" + "\\n" + "-" * 60)


class Exercise1_LinearRegression(Exercise):
    """
    Exercise 1: Linear Regression from Scratch

    Source: "Introduction to Statistical Learning" - Chapter 3
    """

    def __init__(self):
        super().__init__(
            "Linear Regression Implementation",
            "Beginner",
            ["Linear Regression", "Gradient Descent", "Cost Function"]
        )

    def problem_statement(self):
        """Present the problem to solve"""
        self.print_header()

        print("""
🎯 YOUR TASK:
Implement linear regression from scratch using gradient descent.

📊 DATASET:
You'll work with synthetic data: y = 3 + 2*x + noise

🔧 REQUIREMENTS:
1. Implement the cost function (Mean Squared Error)
2. Implement gradient computation
3. Implement gradient descent algorithm
4. Train the model and visualize results
5. Compare with sklearn's LinearRegression

💡 MATHEMATICAL BACKGROUND:
- Linear model: y = β₀ + β₁x
- Cost function: J(β) = (1/2m) × Σ(y_pred - y_true)²
- Gradients: ∂J/∂β₀ = (1/m) × Σ(y_pred - y_true)
            ∂J/∂β₁ = (1/m) × Σ(y_pred - y_true) × x

🚀 STARTER CODE:
        """)

        # Generate sample data
        np.random.seed(42)
        n_samples = 100
        x = np.random.randn(n_samples)
        true_slope = 2.0
        true_intercept = 3.0
        noise = 0.5
        y = true_intercept + true_slope * x + noise * np.random.randn(n_samples)

        print(f"""
class LinearRegressionScratch:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.cost_history = []

    def fit(self, X, y):
        # TODO: Implement this method
        pass

    def predict(self, X):
        # TODO: Implement this method
        pass

    def compute_cost(self, X, y):
        # TODO: Implement this method
        pass

# Test your implementation:
model = LinearRegressionScratch()
model.fit(x.reshape(-1, 1), y)
predictions = model.predict(x.reshape(-1, 1))
        """)

        return x, y, true_slope, true_intercept

    def hints(self):
        """Provide hints for solving the problem"""
        print("""
💡 HINTS:
1. Initialize weights (slope and intercept) to small random values
2. In each iteration:
   - Compute predictions: y_pred = intercept + slope * x
   - Compute cost: MSE = mean((y_pred - y_true)²)
   - Compute gradients for both parameters
   - Update parameters: param = param - learning_rate * gradient
3. Store cost history to monitor convergence
4. Use vectorized operations for efficiency
        """)

    def solution(self):
        """Provide complete solution with explanation"""
        self.print_solution_header()

        # Generate the same data as in problem
        np.random.seed(42)
        n_samples = 100
        x = np.random.randn(n_samples)
        true_slope = 2.0
        true_intercept = 3.0
        noise = 0.5
        y = true_intercept + true_slope * x + noise * np.random.randn(n_samples)

        class LinearRegressionScratch:
            def __init__(self, learning_rate=0.01, n_iterations=1000):
                self.learning_rate = learning_rate
                self.n_iterations = n_iterations
                self.cost_history = []
                self.slope = None
                self.intercept = None

            def fit(self, X, y):
                """
                Train the linear regression model using gradient descent

                Source: "Pattern Recognition and ML" - Bishop, Section 3.1
                """
                # Initialize parameters
                self.slope = np.random.normal(0, 0.01)
                self.intercept = np.random.normal(0, 0.01)

                # Gradient descent
                for i in range(self.n_iterations):
                    # Forward pass: compute predictions
                    y_pred = self.intercept + self.slope * X.flatten()

                    # Compute cost
                    cost = self.compute_cost(X, y)
                    self.cost_history.append(cost)

                    # Compute gradients
                    m = len(y)
                    dJ_dslope = (1/m) * np.sum((y_pred - y) * X.flatten())
                    dJ_dintercept = (1/m) * np.sum(y_pred - y)

                    # Update parameters
                    self.slope -= self.learning_rate * dJ_dslope
                    self.intercept -= self.learning_rate * dJ_dintercept

            def predict(self, X):
                """Make predictions using learned parameters"""
                return self.intercept + self.slope * X.flatten()

            def compute_cost(self, X, y):
                """Compute Mean Squared Error cost function"""
                y_pred = self.predict(X)
                return (1/(2*len(y))) * np.sum((y_pred - y)**2)

        # Train the model
        print("🎓 Training custom linear regression model...")
        model = LinearRegressionScratch(learning_rate=0.1, n_iterations=1000)
        model.fit(x.reshape(-1, 1), y)

        # Compare with sklearn
        sklearn_model = LinearRegression()
        sklearn_model.fit(x.reshape(-1, 1), y)

        # Results
        print(f"\\n📊 RESULTS:")
        print(f"True parameters: slope = {true_slope:.3f}, intercept = {true_intercept:.3f}")
        print(f"Our model:      slope = {model.slope:.3f}, intercept = {model.intercept:.3f}")
        print(f"Sklearn model:  slope = {sklearn_model.coef_[0]:.3f}, intercept = {sklearn_model.intercept_:.3f}")

        # Visualizations
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('📈 Linear Regression Results', fontsize=16, fontweight='bold')

        # 1. Data and fitted line
        axes[0, 0].scatter(x, y, alpha=0.6, label='Data')
        axes[0, 0].plot(x, model.predict(x.reshape(-1, 1)), 'r-', label='Our Model', linewidth=2)
        axes[0, 0].plot(x, sklearn_model.predict(x.reshape(-1, 1)), 'g--', label='Sklearn', linewidth=2)
        axes[0, 0].set_title('🎯 Model Fit')
        axes[0, 0].set_xlabel('x')
        axes[0, 0].set_ylabel('y')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # 2. Cost function over iterations
        axes[0, 1].plot(model.cost_history, 'b-', linewidth=2)
        axes[0, 1].set_title('📉 Cost Function Convergence')
        axes[0, 1].set_xlabel('Iteration')
        axes[0, 1].set_ylabel('Cost (MSE)')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].set_yscale('log')

        # 3. Residuals
        our_predictions = model.predict(x.reshape(-1, 1))
        residuals = y - our_predictions
        axes[1, 0].scatter(our_predictions, residuals, alpha=0.6)
        axes[1, 0].axhline(y=0, color='red', linestyle='--')
        axes[1, 0].set_title('📊 Residuals Plot')
        axes[1, 0].set_xlabel('Predicted Values')
        axes[1, 0].set_ylabel('Residuals')
        axes[1, 0].grid(True, alpha=0.3)

        # 4. Parameter convergence
        # Show how parameters converged (simplified visualization)
        axes[1, 1].axhline(y=true_slope, color='red', linestyle='--', label=f'True slope = {true_slope}')
        axes[1, 1].axhline(y=true_intercept, color='blue', linestyle='--', label=f'True intercept = {true_intercept}')
        axes[1, 1].axhline(y=model.slope, color='red', alpha=0.7, label=f'Learned slope = {model.slope:.3f}')
        axes[1, 1].axhline(y=model.intercept, color='blue', alpha=0.7, label=f'Learned intercept = {model.intercept:.3f}')
        axes[1, 1].set_title('⚖️ Parameter Values')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

        # Calculate R² score
        r2_our = r2_score(y, our_predictions)
        r2_sklearn = r2_score(y, sklearn_model.predict(x.reshape(-1, 1)))

        print(f"\\n🎯 MODEL PERFORMANCE:")
        print(f"Our model R² score:     {r2_our:.4f}")
        print(f"Sklearn model R² score: {r2_sklearn:.4f}")
        print(f"Final cost:             {model.cost_history[-1]:.6f}")

        print(f"\\n💡 KEY LEARNINGS:")
        print("✅ Implemented gradient descent from scratch")
        print("✅ Understanding of cost function optimization")
        print("✅ Compared custom implementation with sklearn")
        print("✅ Visualized training process and results")


class Exercise2_LogisticRegression(Exercise):
    """
    Exercise 2: Logistic Regression and Classification Metrics

    Source: "Pattern Recognition and ML" - Chapter 4
    """

    def __init__(self):
        super().__init__(
            "Logistic Regression & Evaluation Metrics",
            "Beginner",
            ["Logistic Regression", "Classification", "Evaluation Metrics"]
        )

    def problem_statement(self):
        """Present the classification problem"""
        self.print_header()

        print("""
🎯 YOUR TASK:
Build a binary classifier using logistic regression and understand evaluation metrics.

📊 DATASET:
You'll work with the famous Iris dataset, but simplified to binary classification:
- Class 0: Setosa flowers
- Class 1: Non-Setosa flowers (Versicolor + Virginica)

🔧 REQUIREMENTS:
1. Load and prepare the Iris dataset
2. Split into train/test sets
3. Train a logistic regression model
4. Calculate and interpret these metrics:
   - Accuracy
   - Precision
   - Recall
   - F1-score
   - Confusion Matrix
5. Visualize decision boundary
6. Discuss when each metric is most important

💡 MATHEMATICAL BACKGROUND:
- Sigmoid function: σ(z) = 1/(1 + e^(-z))
- Log-likelihood: L = Σ[y*log(p) + (1-y)*log(1-p)]
- Decision boundary: where σ(z) = 0.5, i.e., z = 0

🚀 YOUR TASK:
Implement the analysis below and answer the interpretation questions.
        """)

    def solution(self):
        """Complete solution with detailed explanations"""
        self.print_solution_header()

        # Load and prepare data
        print("📊 Loading and preparing Iris dataset...")
        iris = load_iris()

        # Create binary classification problem
        X = iris.data[:, :2]  # Use only first 2 features for visualization
        y = (iris.target != 0).astype(int)  # 0 for Setosa, 1 for others

        feature_names = iris.feature_names[:2]
        class_names = ['Setosa', 'Non-Setosa']

        print(f"Dataset shape: {X.shape}")
        print(f"Features: {feature_names}")
        print(f"Classes: {class_names}")
        print(f"Class distribution: {np.bincount(y)}")

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )

        # Train model
        print("\\n🎓 Training logistic regression model...")
        model = LogisticRegression(random_state=42)
        model.fit(X_train, y_train)

        # Make predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]

        # Calculate metrics
        from sklearn.metrics import confusion_matrix, classification_report, precision_recall_fscore_support

        accuracy = accuracy_score(y_test, y_pred)
        precision, recall, f1, support = precision_recall_fscore_support(y_test, y_pred, average='binary')
        cm = confusion_matrix(y_test, y_pred)

        print(f"\\n📊 MODEL PERFORMANCE:")
        print(f"Accuracy:  {accuracy:.3f}")
        print(f"Precision: {precision:.3f}")
        print(f"Recall:    {recall:.3f}")
        print(f"F1-score:  {f1:.3f}")

        # Detailed classification report
        print(f"\\n📋 DETAILED CLASSIFICATION REPORT:")
        print(classification_report(y_test, y_pred, target_names=class_names))

        # Visualizations
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('🎯 Logistic Regression Analysis', fontsize=16, fontweight='bold')

        # 1. Decision boundary
        def plot_decision_boundary(X, y, model, ax, title):
            h = 0.02
            x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
            y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
            xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                               np.arange(y_min, y_max, h))

            Z = model.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
            Z = Z.reshape(xx.shape)

            ax.contourf(xx, yy, Z, levels=50, alpha=0.6, cmap='RdYlBu')
            ax.contour(xx, yy, Z, levels=[0.5], colors='black', linestyles='--', linewidths=2)

            scatter = ax.scatter(X[:, 0], X[:, 1], c=y, cmap='RdYlBu', edgecolors='black')
            ax.set_title(title)
            ax.set_xlabel(feature_names[0])
            ax.set_ylabel(feature_names[1])
            return scatter

        plot_decision_boundary(X_test, y_test, model, axes[0, 0], '🎯 Decision Boundary')

        # 2. Confusion Matrix
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names, ax=axes[0, 1])
        axes[0, 1].set_title('📊 Confusion Matrix')
        axes[0, 1].set_xlabel('Predicted')
        axes[0, 1].set_ylabel('Actual')

        # 3. Prediction probabilities
        axes[1, 0].hist(y_pred_proba[y_test == 0], alpha=0.7, label='Setosa', bins=15, color='blue')
        axes[1, 0].hist(y_pred_proba[y_test == 1], alpha=0.7, label='Non-Setosa', bins=15, color='red')
        axes[1, 0].axvline(x=0.5, color='black', linestyle='--', label='Decision Threshold')
        axes[1, 0].set_title('🎲 Prediction Probabilities')
        axes[1, 0].set_xlabel('Predicted Probability')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].legend()

        # 4. Metrics comparison
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        values = [accuracy, precision, recall, f1]

        bars = axes[1, 1].bar(metrics, values, color=['skyblue', 'lightcoral', 'lightgreen', 'gold'])
        axes[1, 1].set_title('📈 Performance Metrics')
        axes[1, 1].set_ylabel('Score')
        axes[1, 1].set_ylim(0, 1)

        # Add value labels on bars
        for bar, value in zip(bars, values):
            axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                           f'{value:.3f}', ha='center', va='bottom')

        plt.tight_layout()
        plt.show()

        # Educational explanations
        print(f"\\n🧠 UNDERSTANDING THE METRICS:")
        print("=" * 60)

        tn, fp, fn, tp = cm.ravel()
        print(f"True Positives (TP):  {tp} - Correctly identified Non-Setosa")
        print(f"True Negatives (TN):  {tn} - Correctly identified Setosa")
        print(f"False Positives (FP): {fp} - Incorrectly identified as Non-Setosa")
        print(f"False Negatives (FN): {fn} - Incorrectly identified as Setosa")

        print(f"\\n📊 METRIC INTERPRETATIONS:")
        print(f"• Accuracy = (TP + TN) / Total = {(tp + tn)}/{len(y_test)} = {accuracy:.3f}")
        print(f"  → {accuracy:.1%} of all predictions were correct")
        print(f"\\n• Precision = TP / (TP + FP) = {tp}/{tp + fp} = {precision:.3f}")
        print(f"  → Of all Non-Setosa predictions, {precision:.1%} were actually Non-Setosa")
        print(f"\\n• Recall = TP / (TP + FN) = {tp}/{tp + fn} = {recall:.3f}")
        print(f"  → Of all actual Non-Setosa flowers, we correctly identified {recall:.1%}")
        print(f"\\n• F1-Score = 2 × (Precision × Recall) / (Precision + Recall) = {f1:.3f}")
        print(f"  → Harmonic mean of precision and recall")

        print(f"\\n💡 WHEN TO USE EACH METRIC:")
        print("🎯 Accuracy: Good when classes are balanced and all errors are equally costly")
        print("🎯 Precision: Important when false positives are costly (e.g., spam detection)")
        print("🎯 Recall: Important when false negatives are costly (e.g., disease detection)")
        print("🎯 F1-Score: Good balance when you need both precision and recall")

        return model, X_test, y_test, y_pred


class Exercise3_TreesAndEnsembles(Exercise):
    """
    Exercise 3: Decision Trees and Random Forests

    Source: "Introduction to Statistical Learning" - Chapter 8
    """

    def __init__(self):
        super().__init__(
            "Decision Trees vs Random Forests",
            "Intermediate",
            ["Decision Trees", "Random Forests", "Overfitting", "Feature Importance"]
        )

    def problem_statement(self):
        """Present the ensemble learning problem"""
        self.print_header()

        print("""
🎯 YOUR TASK:
Compare Decision Trees with Random Forests to understand ensemble learning benefits.

📊 DATASET:
You'll work with a classification dataset with multiple features.

🔧 REQUIREMENTS:
1. Train a single Decision Tree with different max_depth values
2. Train a Random Forest with the same depths
3. Compare performance on train vs test sets
4. Analyze feature importance
5. Visualize overfitting behavior
6. Explain why Random Forests often perform better

💡 KEY CONCEPTS:
- Single trees can overfit easily
- Random Forests reduce overfitting through:
  * Bootstrap aggregating (bagging)
  * Random feature selection
  * Voting/averaging predictions
- Feature importance through impurity reduction

🚀 YOUR TASK:
Implement the comparison and interpret the results.
        """)

    def solution(self):
        """Complete solution comparing trees and forests"""
        self.print_solution_header()

        # Create a more complex dataset
        print("📊 Creating classification dataset...")
        X, y = make_classification(
            n_samples=1000, n_features=20, n_informative=15, n_redundant=5,
            n_clusters_per_class=1, random_state=42
        )

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )

        print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")
        print(f"Training set: {X_train.shape[0]} samples")
        print(f"Test set: {X_test.shape[0]} samples")

        # Compare different max_depth values
        depths = range(1, 21)
        tree_train_scores = []
        tree_test_scores = []
        forest_train_scores = []
        forest_test_scores = []

        print("\\n🌳 Training models with different complexities...")

        for depth in depths:
            # Single Decision Tree
            tree = DecisionTreeClassifier(max_depth=depth, random_state=42)
            tree.fit(X_train, y_train)
            tree_train_scores.append(tree.score(X_train, y_train))
            tree_test_scores.append(tree.score(X_test, y_test))

            # Random Forest
            forest = RandomForestClassifier(n_estimators=100, max_depth=depth, random_state=42)
            forest.fit(X_train, y_train)
            forest_train_scores.append(forest.score(X_train, y_train))
            forest_test_scores.append(forest.score(X_test, y_test))

        # Train final models for detailed analysis
        final_tree = DecisionTreeClassifier(max_depth=10, random_state=42)
        final_forest = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)

        final_tree.fit(X_train, y_train)
        final_forest.fit(X_train, y_train)

        # Visualizations
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('🌳 Decision Trees vs Random Forests', fontsize=16, fontweight='bold')

        # 1. Learning curves (overfitting analysis)
        axes[0, 0].plot(depths, tree_train_scores, 'b-o', label='Tree Train', linewidth=2)
        axes[0, 0].plot(depths, tree_test_scores, 'b--s', label='Tree Test', linewidth=2)
        axes[0, 0].plot(depths, forest_train_scores, 'r-o', label='Forest Train', linewidth=2)
        axes[0, 0].plot(depths, forest_test_scores, 'r--s', label='Forest Test', linewidth=2)

        axes[0, 0].set_title('📈 Performance vs Model Complexity')
        axes[0, 0].set_xlabel('Max Depth')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # 2. Overfitting gap
        tree_gaps = [train - test for train, test in zip(tree_train_scores, tree_test_scores)]
        forest_gaps = [train - test for train, test in zip(forest_train_scores, forest_test_scores)]

        axes[0, 1].plot(depths, tree_gaps, 'b-o', label='Decision Tree', linewidth=2)
        axes[0, 1].plot(depths, forest_gaps, 'r-o', label='Random Forest', linewidth=2)

        axes[0, 1].set_title('📊 Overfitting Gap (Train - Test)')
        axes[0, 1].set_xlabel('Max Depth')
        axes[0, 1].set_ylabel('Accuracy Gap')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # 3. Feature importance comparison
        tree_importance = final_tree.feature_importances_
        forest_importance = final_forest.feature_importances_

        feature_indices = np.arange(len(tree_importance))
        width = 0.35

        axes[1, 0].bar(feature_indices - width/2, tree_importance, width,
                      label='Decision Tree', alpha=0.7)
        axes[1, 0].bar(feature_indices + width/2, forest_importance, width,
                      label='Random Forest', alpha=0.7)

        axes[1, 0].set_title('🔍 Feature Importance')
        axes[1, 0].set_xlabel('Feature Index')
        axes[1, 0].set_ylabel('Importance')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # 4. Performance summary
        models = ['Decision Tree', 'Random Forest']
        train_accs = [final_tree.score(X_train, y_train), final_forest.score(X_train, y_train)]
        test_accs = [final_tree.score(X_test, y_test), final_forest.score(X_test, y_test)]

        x = np.arange(len(models))
        width = 0.35

        axes[1, 1].bar(x - width/2, train_accs, width, label='Train Accuracy', alpha=0.7)
        axes[1, 1].bar(x + width/2, test_accs, width, label='Test Accuracy', alpha=0.7)

        axes[1, 1].set_title('🎯 Final Model Comparison')
        axes[1, 1].set_ylabel('Accuracy')
        axes[1, 1].set_xticks(x)
        axes[1, 1].set_xticklabels(models)
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        # Add value labels
        for i, (train_acc, test_acc) in enumerate(zip(train_accs, test_accs)):
            axes[1, 1].text(i - width/2, train_acc + 0.01, f'{train_acc:.3f}',
                           ha='center', va='bottom')
            axes[1, 1].text(i + width/2, test_acc + 0.01, f'{test_acc:.3f}',
                           ha='center', va='bottom')

        plt.tight_layout()
        plt.show()

        # Analysis and insights
        print(f"\\n📊 PERFORMANCE COMPARISON:")
        print("=" * 60)
        print(f"Decision Tree (depth=10):")
        print(f"  Train Accuracy: {final_tree.score(X_train, y_train):.3f}")
        print(f"  Test Accuracy:  {final_tree.score(X_test, y_test):.3f}")
        print(f"  Overfitting Gap: {final_tree.score(X_train, y_train) - final_tree.score(X_test, y_test):.3f}")

        print(f"\\nRandom Forest (depth=10, n_estimators=100):")
        print(f"  Train Accuracy: {final_forest.score(X_train, y_train):.3f}")
        print(f"  Test Accuracy:  {final_forest.score(X_test, y_test):.3f}")
        print(f"  Overfitting Gap: {final_forest.score(X_train, y_train) - final_forest.score(X_test, y_test):.3f}")

        # Find optimal depths
        optimal_tree_depth = depths[np.argmax(tree_test_scores)]
        optimal_forest_depth = depths[np.argmax(forest_test_scores)]

        print(f"\\n🎯 OPTIMAL COMPLEXITY:")
        print(f"Best tree depth: {optimal_tree_depth} (test accuracy: {max(tree_test_scores):.3f})")
        print(f"Best forest depth: {optimal_forest_depth} (test accuracy: {max(forest_test_scores):.3f})")

        print(f"\\n💡 KEY INSIGHTS:")
        print("=" * 60)
        print("🌳 Single Decision Trees:")
        print("  ✅ Easy to interpret and visualize")
        print("  ✅ Fast to train and predict")
        print("  ❌ Prone to overfitting")
        print("  ❌ High variance (sensitive to data changes)")
        print("  ❌ Can create overly complex boundaries")

        print("\\n🌲 Random Forests:")
        print("  ✅ Reduced overfitting through averaging")
        print("  ✅ More stable predictions")
        print("  ✅ Built-in feature importance")
        print("  ✅ Handle missing values well")
        print("  ❌ Less interpretable than single trees")
        print("  ❌ Slower to train and predict")

        print(f"\\n🧠 WHY RANDOM FORESTS WORK BETTER:")
        print("1. **Bootstrap Aggregating**: Each tree sees different data")
        print("2. **Random Feature Selection**: Reduces correlation between trees")
        print("3. **Voting**: Multiple weak learners → strong learner")
        print("4. **Variance Reduction**: Averaging reduces prediction variance")

        return final_tree, final_forest, X_test, y_test


def run_all_exercises():
    """Run all exercises in sequence"""
    print("🚀 Welcome to ML Beginner Exercises!")
    print("=" * 60)
    print("This module contains 3 hands-on exercises to build your ML intuition.")
    print("Each exercise focuses on different fundamental concepts.")
    print("\\n")

    # Exercise 1: Linear Regression
    ex1 = Exercise1_LinearRegression()
    print("Starting Exercise 1...")
    x, y, true_slope, true_intercept = ex1.problem_statement()

    input("\\nPress Enter to see hints...")
    ex1.hints()

    input("\\nPress Enter to see the solution...")
    ex1.solution()

    input("\\nPress Enter to continue to Exercise 2...")

    # Exercise 2: Logistic Regression
    ex2 = Exercise2_LogisticRegression()
    print("\\nStarting Exercise 2...")
    ex2.problem_statement()

    input("\\nPress Enter to see the solution...")
    model, X_test, y_test, y_pred = ex2.solution()

    input("\\nPress Enter to continue to Exercise 3...")

    # Exercise 3: Trees and Ensembles
    ex3 = Exercise3_TreesAndEnsembles()
    print("\\nStarting Exercise 3...")
    ex3.problem_statement()

    input("\\nPress Enter to see the solution...")
    tree_model, forest_model, X_test_3, y_test_3 = ex3.solution()

    print("\\n" + "🎉 CONGRATULATIONS!" + "\\n" + "=" * 60)
    print("You've completed all beginner exercises!")
    print("\\n💪 Skills you've developed:")
    print("✅ Implemented linear regression from scratch")
    print("✅ Understanding of gradient descent optimization")
    print("✅ Classification metrics and their interpretations")
    print("✅ Comparison of single models vs ensembles")
    print("✅ Overfitting analysis and model complexity")
    print("✅ Data visualization and result interpretation")

    print("\\n🚀 Next Steps:")
    print("1. Try modifying the hyperparameters in each exercise")
    print("2. Experiment with different datasets")
    print("3. Move on to intermediate exercises")
    print("4. Explore more advanced algorithms")

    return ex1, ex2, ex3


if __name__ == "__main__":
    # Allow running individual exercises or all together
    print("🎓 ML Beginner Exercises")
    print("=" * 40)
    print("Choose an option:")
    print("1. Run all exercises")
    print("2. Exercise 1: Linear Regression")
    print("3. Exercise 2: Logistic Regression")
    print("4. Exercise 3: Trees vs Forests")

    choice = input("Enter your choice (1-4): ").strip()

    if choice == "1":
        run_all_exercises()
    elif choice == "2":
        ex1 = Exercise1_LinearRegression()
        ex1.problem_statement()
        ex1.hints()
        ex1.solution()
    elif choice == "3":
        ex2 = Exercise2_LogisticRegression()
        ex2.problem_statement()
        ex2.solution()
    elif choice == "4":
        ex3 = Exercise3_TreesAndEnsembles()
        ex3.problem_statement()
        ex3.solution()
    else:
        print("Invalid choice. Running all exercises...")
        run_all_exercises()