"""
Interactive Machine Learning Concept Visualizations
==================================================

This module provides interactive visualizations to help understand key ML concepts.
All visualizations are designed to be educational and interactive.

Sources:
- "Pattern Recognition and Machine Learning" - Christopher Bishop
- "The Elements of Statistical Learning" - Hastie, Tibshirani, Friedman
- "Introduction to Statistical Learning" - James, Witten, Hastie, Tibshirani

Author: ML Learning Journey
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.widgets import Slider, Button
from sklearn.datasets import make_classification, make_regression, make_blobs
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class BiasVarianceDemo:
    """
    Interactive demonstration of bias-variance tradeoff.

    Source: "The Elements of Statistical Learning" - Section 7.3
    """

    def __init__(self):
        self.fig, self.axes = plt.subplots(2, 2, figsize=(15, 10))
        self.fig.suptitle('🎯 Bias-Variance Tradeoff Visualization', fontsize=16, fontweight='bold')

    def generate_true_function(self, x):
        """True underlying function: quadratic with noise"""
        return 0.5 * x**2 + 0.3 * x + 0.1

    def visualize(self, model_complexity=2, noise_level=0.3, n_datasets=50):
        """
        Visualize bias-variance tradeoff for different model complexities.

        Parameters:
        -----------
        model_complexity : int
            Polynomial degree (1=linear, high=complex)
        noise_level : float
            Amount of noise in data
        n_datasets : int
            Number of bootstrap datasets to generate
        """
        # Clear previous plots
        for ax in self.axes.flat:
            ax.clear()

        # Generate test points
        x_test = np.linspace(-2, 2, 100)
        y_true = self.generate_true_function(x_test)

        # Store predictions from different models
        predictions = []

        # Generate multiple datasets and train models
        np.random.seed(42)
        for i in range(n_datasets):
            # Generate training data with noise
            x_train = np.random.uniform(-2, 2, 20)
            y_train = self.generate_true_function(x_train) + noise_level * np.random.randn(20)

            # Fit polynomial model
            coeffs = np.polyfit(x_train, y_train, model_complexity)
            y_pred = np.polyval(coeffs, x_test)
            predictions.append(y_pred)

            # Plot some individual models (first 10)
            if i < 10:
                self.axes[0, 0].plot(x_test, y_pred, 'b-', alpha=0.3, linewidth=1)

        predictions = np.array(predictions)

        # Calculate bias and variance
        mean_prediction = np.mean(predictions, axis=0)
        bias_squared = (mean_prediction - y_true) ** 2
        variance = np.var(predictions, axis=0)

        # Plot 1: Individual models and average
        self.axes[0, 0].plot(x_test, y_true, 'r-', linewidth=3, label='True Function')
        self.axes[0, 0].plot(x_test, mean_prediction, 'g-', linewidth=3, label='Average Prediction')
        self.axes[0, 0].set_title(f'Individual Models (Degree {model_complexity})')
        self.axes[0, 0].legend()
        self.axes[0, 0].grid(True, alpha=0.3)

        # Plot 2: Bias
        self.axes[0, 1].plot(x_test, bias_squared, 'orange', linewidth=3)
        self.axes[0, 1].fill_between(x_test, bias_squared, alpha=0.3, color='orange')
        self.axes[0, 1].set_title('Bias² (Systematic Error)')
        self.axes[0, 1].grid(True, alpha=0.3)
        self.axes[0, 1].set_ylabel('Bias²')

        # Plot 3: Variance
        self.axes[1, 0].plot(x_test, variance, 'purple', linewidth=3)
        self.axes[1, 0].fill_between(x_test, variance, alpha=0.3, color='purple')
        self.axes[1, 0].set_title('Variance (Sensitivity to Data)')
        self.axes[1, 0].grid(True, alpha=0.3)
        self.axes[1, 0].set_ylabel('Variance')
        self.axes[1, 0].set_xlabel('x')

        # Plot 4: Total Error Decomposition
        noise_term = noise_level**2  # Irreducible error
        total_error = bias_squared + variance + noise_term

        self.axes[1, 1].plot(x_test, bias_squared, label='Bias²', linewidth=2)
        self.axes[1, 1].plot(x_test, variance, label='Variance', linewidth=2)
        self.axes[1, 1].axhline(y=noise_term, color='red', linestyle='--',
                              label=f'Irreducible Error = {noise_term:.3f}', linewidth=2)
        self.axes[1, 1].plot(x_test, total_error, 'black', linewidth=3, label='Total Error')
        self.axes[1, 1].set_title('Bias-Variance Decomposition')
        self.axes[1, 1].legend()
        self.axes[1, 1].grid(True, alpha=0.3)
        self.axes[1, 1].set_xlabel('x')
        self.axes[1, 1].set_ylabel('Error')

        # Add summary statistics
        avg_bias = np.mean(bias_squared)
        avg_variance = np.mean(variance)
        avg_total = np.mean(total_error)

        summary_text = f"""
        📊 Summary (Averaged over x):
        Bias² = {avg_bias:.4f}
        Variance = {avg_variance:.4f}
        Noise = {noise_term:.4f}
        Total Error = {avg_total:.4f}

        💡 Interpretation:
        - Low complexity: High bias, Low variance
        - High complexity: Low bias, High variance
        - Optimal: Balance between bias and variance
        """

        self.fig.text(0.02, 0.02, summary_text, fontsize=10,
                     bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

        plt.tight_layout()
        return avg_bias, avg_variance, avg_total


class DecisionBoundaryVisualizer:
    """
    Visualize decision boundaries for different ML algorithms.

    Source: "Pattern Recognition and Machine Learning" - Bishop, Chapter 4
    """

    def __init__(self):
        self.fig, self.axes = plt.subplots(2, 3, figsize=(18, 12))
        self.fig.suptitle('🎯 Decision Boundaries: Different ML Algorithms', fontsize=16, fontweight='bold')

    def create_dataset(self, dataset_type='circles', n_samples=300, noise=0.1):
        """Create different types of datasets for visualization"""
        np.random.seed(42)

        if dataset_type == 'circles':
            # Non-linearly separable circular data
            theta = np.random.uniform(0, 2*np.pi, n_samples)
            r = np.random.uniform(0.5, 1.5, n_samples)
            X = np.column_stack([r * np.cos(theta), r * np.sin(theta)])
            y = (r > 1.0).astype(int)
            X += noise * np.random.randn(n_samples, 2)

        elif dataset_type == 'moons':
            # Half-moon shaped data
            from sklearn.datasets import make_moons
            X, y = make_moons(n_samples=n_samples, noise=noise, random_state=42)

        elif dataset_type == 'linear':
            # Linearly separable data
            X, y = make_classification(n_samples=n_samples, n_features=2,
                                     n_redundant=0, n_informative=2,
                                     n_clusters_per_class=1, random_state=42)
        else:
            # Blob clusters
            X, y = make_blobs(n_samples=n_samples, centers=3, n_features=2,
                            cluster_std=1.0, random_state=42)
            y = (y > 0).astype(int)  # Make binary

        return X, y

    def plot_decision_boundary(self, X, y, model, ax, title, h=0.02):
        """Plot decision boundary for a given model"""
        # Create a mesh
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                           np.arange(y_min, y_max, h))

        # Make predictions on the mesh
        Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        # Plot decision boundary
        ax.contourf(xx, yy, Z, alpha=0.4, cmap=plt.cm.RdYlBu)

        # Plot data points
        scatter = ax.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdYlBu, edgecolors='black')
        ax.set_title(title)
        ax.set_xlabel('Feature 1')
        ax.set_ylabel('Feature 2')

        return scatter

    def visualize_all_algorithms(self, dataset_type='circles'):
        """Visualize decision boundaries for all algorithms"""
        # Clear previous plots
        for ax in self.axes.flat:
            ax.clear()

        # Create dataset
        X, y = self.create_dataset(dataset_type)

        # Define algorithms
        algorithms = [
            ('Logistic Regression', LogisticRegression(random_state=42)),
            ('Decision Tree', DecisionTreeClassifier(max_depth=5, random_state=42)),
            ('Random Forest', RandomForestClassifier(n_estimators=100, random_state=42)),
            ('SVM (Linear)', SVC(kernel='linear', random_state=42)),
            ('SVM (RBF)', SVC(kernel='rbf', random_state=42)),
            ('K-Nearest Neighbors', KNeighborsClassifier(n_neighbors=5))
        ]

        # Plot each algorithm
        for i, (name, model) in enumerate(algorithms):
            row = i // 3
            col = i % 3

            # Train model
            model.fit(X, y)

            # Plot decision boundary
            self.plot_decision_boundary(X, y, model, self.axes[row, col], name)

            # Calculate and display accuracy
            accuracy = model.score(X, y)
            self.axes[row, col].text(0.02, 0.98, f'Accuracy: {accuracy:.3f}',
                                   transform=self.axes[row, col].transAxes,
                                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                                   verticalalignment='top')

        plt.tight_layout()

        # Add interpretation text
        interpretation = f"""
        🧠 Understanding Decision Boundaries:

        📊 Dataset: {dataset_type.title()}

        💡 Key Insights:
        • Linear models create straight boundaries
        • Tree-based models create rectangular regions
        • SVM with RBF kernel creates curved boundaries
        • KNN creates irregular, local boundaries

        🎯 Choice depends on:
        • Data complexity
        • Interpretability needs
        • Training time constraints
        • Overfitting concerns
        """

        self.fig.text(0.02, 0.02, interpretation, fontsize=10,
                     bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))


class LearningCurveVisualizer:
    """
    Visualize learning curves to understand overfitting/underfitting.

    Source: "Introduction to Statistical Learning" - James et al., Section 5.1
    """

    def __init__(self):
        self.fig, self.axes = plt.subplots(2, 2, figsize=(15, 10))
        self.fig.suptitle('📈 Learning Curves: Diagnosing Model Performance', fontsize=16, fontweight='bold')

    def generate_learning_curves(self, model_complexity='optimal'):
        """Generate learning curves for different model complexities"""

        # Generate dataset
        np.random.seed(42)
        n_total = 1000
        X, y = make_regression(n_samples=n_total, n_features=1, noise=15, random_state=42)

        # Split into train and validation
        split_idx = int(0.8 * n_total)
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]

        # Define different complexities
        if model_complexity == 'underfit':
            max_degree = 1
            title_suffix = '(Underfitting)'
            color = 'red'
        elif model_complexity == 'optimal':
            max_degree = 3
            title_suffix = '(Good Fit)'
            color = 'green'
        else:  # overfit
            max_degree = 15
            title_suffix = '(Overfitting)'
            color = 'orange'

        # Calculate learning curves
        train_sizes = np.linspace(50, len(X_train), 20, dtype=int)
        train_errors = []
        val_errors = []

        for train_size in train_sizes:
            # Use subset of training data
            X_subset = X_train[:train_size]
            y_subset = y_train[:train_size]

            # Fit polynomial model
            coeffs = np.polyfit(X_subset.ravel(), y_subset, max_degree)

            # Calculate training error
            y_train_pred = np.polyval(coeffs, X_subset.ravel())
            train_mse = np.mean((y_subset - y_train_pred) ** 2)
            train_errors.append(train_mse)

            # Calculate validation error
            y_val_pred = np.polyval(coeffs, X_val.ravel())
            val_mse = np.mean((y_val - y_val_pred) ** 2)
            val_errors.append(val_mse)

        return train_sizes, train_errors, val_errors, title_suffix, color

    def plot_all_scenarios(self):
        """Plot learning curves for all scenarios"""
        scenarios = ['underfit', 'optimal', 'overfit']
        positions = [(0, 0), (0, 1), (1, 0)]

        for scenario, (row, col) in zip(scenarios, positions):
            train_sizes, train_errors, val_errors, title_suffix, color = self.generate_learning_curves(scenario)

            ax = self.axes[row, col]
            ax.clear()

            # Plot learning curves
            ax.plot(train_sizes, train_errors, 'o-', color=color, label='Training Error', linewidth=2, markersize=6)
            ax.plot(train_sizes, val_errors, 's-', color='blue', label='Validation Error', linewidth=2, markersize=6)

            ax.set_title(f'Learning Curve {title_suffix}')
            ax.set_xlabel('Training Set Size')
            ax.set_ylabel('Mean Squared Error')
            ax.legend()
            ax.grid(True, alpha=0.3)

            # Add interpretation
            final_train_error = train_errors[-1]
            final_val_error = val_errors[-1]
            gap = final_val_error - final_train_error

            interpretation = f"""
            Final Training Error: {final_train_error:.0f}
            Final Validation Error: {final_val_error:.0f}
            Gap: {gap:.0f}
            """

            ax.text(0.05, 0.95, interpretation, transform=ax.transAxes,
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                   verticalalignment='top', fontsize=9)

        # Summary plot
        ax_summary = self.axes[1, 1]
        ax_summary.clear()

        # Create summary comparison
        scenarios_data = []
        for scenario in scenarios:
            train_sizes, train_errors, val_errors, title_suffix, color = self.generate_learning_curves(scenario)
            scenarios_data.append((scenario, train_errors[-1], val_errors[-1], color))

        scenario_names = [s[0].title() for s in scenarios_data]
        train_final = [s[1] for s in scenarios_data]
        val_final = [s[2] for s in scenarios_data]
        colors = [s[3] for s in scenarios_data]

        x = np.arange(len(scenario_names))
        width = 0.35

        ax_summary.bar(x - width/2, train_final, width, label='Training Error', alpha=0.7)
        ax_summary.bar(x + width/2, val_final, width, label='Validation Error', alpha=0.7)

        ax_summary.set_title('📊 Final Error Comparison')
        ax_summary.set_xlabel('Model Complexity')
        ax_summary.set_ylabel('Final Error')
        ax_summary.set_xticks(x)
        ax_summary.set_xticklabels(scenario_names)
        ax_summary.legend()
        ax_summary.grid(True, alpha=0.3)

        plt.tight_layout()

        # Add learning insights
        insights = """
        📚 Learning Curve Insights:

        🔴 Underfitting:
        • High training & validation error
        • Small gap between them
        • Solution: Increase model complexity

        🟢 Good Fit:
        • Low training & validation error
        • Reasonable gap
        • Validation error stabilizes

        🟠 Overfitting:
        • Low training error
        • High validation error
        • Large gap between them
        • Solution: Reduce complexity or get more data
        """

        self.fig.text(0.02, 0.02, insights, fontsize=10,
                     bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))


def create_interactive_bias_variance():
    """Create interactive bias-variance demonstration"""
    demo = BiasVarianceDemo()

    # Create sliders
    ax_complexity = plt.axes([0.2, 0.92, 0.3, 0.03])
    ax_noise = plt.axes([0.6, 0.92, 0.3, 0.03])

    slider_complexity = Slider(ax_complexity, 'Model Complexity', 1, 15, valinit=3, valfmt='%d')
    slider_noise = Slider(ax_noise, 'Noise Level', 0.1, 1.0, valinit=0.3, valfmt='%.1f')

    def update(val):
        complexity = int(slider_complexity.val)
        noise = slider_noise.val
        demo.visualize(model_complexity=complexity, noise_level=noise)
        plt.draw()

    slider_complexity.on_changed(update)
    slider_noise.on_changed(update)

    # Initial plot
    demo.visualize()
    plt.show()


def create_decision_boundary_demo():
    """Create interactive decision boundary demonstration"""
    visualizer = DecisionBoundaryVisualizer()

    # Create dataset selection buttons
    ax_linear = plt.axes([0.1, 0.92, 0.1, 0.04])
    ax_circles = plt.axes([0.25, 0.92, 0.1, 0.04])
    ax_moons = plt.axes([0.4, 0.92, 0.1, 0.04])
    ax_blobs = plt.axes([0.55, 0.92, 0.1, 0.04])

    button_linear = Button(ax_linear, 'Linear')
    button_circles = Button(ax_circles, 'Circles')
    button_moons = Button(ax_moons, 'Moons')
    button_blobs = Button(ax_blobs, 'Blobs')

    def update_dataset(dataset_type):
        def inner(event):
            visualizer.visualize_all_algorithms(dataset_type)
            plt.draw()
        return inner

    button_linear.on_clicked(update_dataset('linear'))
    button_circles.on_clicked(update_dataset('circles'))
    button_moons.on_clicked(update_dataset('moons'))
    button_blobs.on_clicked(update_dataset('blobs'))

    # Initial plot
    visualizer.visualize_all_algorithms()
    plt.show()


def create_learning_curves_demo():
    """Create learning curves demonstration"""
    visualizer = LearningCurveVisualizer()
    visualizer.plot_all_scenarios()
    plt.show()


if __name__ == "__main__":
    print("🎨 Interactive ML Visualizations")
    print("=" * 50)
    print("Choose a demonstration:")
    print("1. Bias-Variance Tradeoff (Interactive)")
    print("2. Decision Boundaries")
    print("3. Learning Curves")
    print("4. All Demonstrations")

    choice = input("Enter your choice (1-4): ").strip()

    if choice == "1":
        print("🎯 Starting Bias-Variance Interactive Demo...")
        create_interactive_bias_variance()
    elif choice == "2":
        print("🎯 Starting Decision Boundaries Demo...")
        create_decision_boundary_demo()
    elif choice == "3":
        print("📈 Starting Learning Curves Demo...")
        create_learning_curves_demo()
    elif choice == "4":
        print("🎨 Starting All Demonstrations...")
        print("\\n1. Bias-Variance Tradeoff:")
        create_interactive_bias_variance()
        print("\\n2. Decision Boundaries:")
        create_decision_boundary_demo()
        print("\\n3. Learning Curves:")
        create_learning_curves_demo()
    else:
        print("Invalid choice. Running all demonstrations...")
        create_interactive_bias_variance()
        create_decision_boundary_demo()
        create_learning_curves_demo()

    print("\\n✅ Visualization demos completed!")
    print("💡 Key takeaways:")
    print("- Bias-variance tradeoff is fundamental to ML")
    print("- Different algorithms create different decision boundaries")
    print("- Learning curves help diagnose model performance")
    print("- Interactive exploration enhances understanding")