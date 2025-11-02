# Bias-Variance Tradeoff and Overfitting Guide

A comprehensive guide to understanding and managing the bias-variance tradeoff in machine learning.

## Table of Contents
- [Understanding Bias and Variance](#understanding-bias-and-variance)
- [The Bias-Variance Tradeoff](#the-bias-variance-tradeoff)
- [Overfitting vs Underfitting](#overfitting-vs-underfitting)
- [Diagnosing Model Problems](#diagnosing-model-problems)
- [Solutions to Overfitting](#solutions-to-overfitting)
- [Solutions to Underfitting](#solutions-to-underfitting)
- [Model Complexity and Performance](#model-complexity-and-performance)
- [Practical Examples](#practical-examples)
- [Best Practices](#best-practices)

---

## Understanding Bias and Variance

### What is Bias?

**Bias** is the error from incorrect assumptions in the learning algorithm. High bias causes the model to miss relevant relations between features and target (underfitting).

**Intuition:** Bias measures how far off predictions are from correct values *on average*.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# Generate nonlinear data
np.random.seed(42)
X = np.linspace(0, 10, 100).reshape(-1, 1)
y = 2 + 3*X.ravel() + 5*np.sin(X.ravel()) + np.random.randn(100) * 2

# High bias model (too simple - linear for nonlinear data)
model_high_bias = LinearRegression()
model_high_bias.fit(X, y)
y_pred_bias = model_high_bias.predict(X)

# Visualize
plt.figure(figsize=(10, 6))
plt.scatter(X, y, alpha=0.5, label='Data')
plt.plot(X, y_pred_bias, 'r-', linewidth=2, label='High Bias Model (Linear)')
plt.plot(X, 2 + 3*X.ravel() + 5*np.sin(X.ravel()), 'g--',
         linewidth=2, label='True Function')
plt.legend()
plt.title('High Bias Model - Underfitting')
plt.xlabel('X')
plt.ylabel('y')
plt.show()

print(f"Training Error (High Bias): {np.mean((y - y_pred_bias)**2):.2f}")
```

**Characteristics of High Bias:**
- Model is too simple
- Poor performance on both training and test sets
- Systematic errors (predictions consistently off in same direction)
- Underfitting

### What is Variance?

**Variance** is the error from sensitivity to small fluctuations in the training set. High variance causes the model to fit random noise (overfitting).

**Intuition:** Variance measures how much predictions vary for different training sets.

```python
# High variance model (too complex - degree 15 polynomial)
poly = PolynomialFeatures(degree=15)
X_poly = poly.fit_transform(X)

model_high_variance = LinearRegression()
model_high_variance.fit(X_poly, y)
y_pred_variance = model_high_variance.predict(X_poly)

# Visualize
plt.figure(figsize=(10, 6))
plt.scatter(X, y, alpha=0.5, label='Data')
plt.plot(X, y_pred_variance, 'r-', linewidth=2, label='High Variance Model (Degree 15)')
plt.plot(X, 2 + 3*X.ravel() + 5*np.sin(X.ravel()), 'g--',
         linewidth=2, label='True Function')
plt.legend()
plt.title('High Variance Model - Overfitting')
plt.xlabel('X')
plt.ylabel('y')
plt.show()

print(f"Training Error (High Variance): {np.mean((y - y_pred_variance)**2):.2f}")
```

**Characteristics of High Variance:**
- Model is too complex
- Excellent performance on training set
- Poor performance on test set
- Fits random noise
- Overfitting

### The Mathematical Decomposition

Expected error can be decomposed into three parts:

```
Expected Test Error = Bias² + Variance + Irreducible Error
```

Where:
- **Bias²**: Error from wrong assumptions
- **Variance**: Error from sensitivity to training set
- **Irreducible Error**: Noise in data that cannot be reduced

```python
def compute_bias_variance(model_class, X_train, y_train, X_test, y_test,
                         n_iterations=100, **model_params):
    """
    Compute bias and variance of a model through bootstrap sampling.
    """
    predictions = []

    for i in range(n_iterations):
        # Bootstrap sample
        indices = np.random.choice(len(X_train), len(X_train), replace=True)
        X_boot = X_train[indices]
        y_boot = y_train[indices]

        # Train model
        model = model_class(**model_params)
        model.fit(X_boot, y_boot)

        # Predict on test set
        y_pred = model.predict(X_test)
        predictions.append(y_pred)

    predictions = np.array(predictions)

    # Calculate bias and variance
    mean_predictions = predictions.mean(axis=0)
    bias_squared = np.mean((mean_predictions - y_test) ** 2)
    variance = np.mean(predictions.var(axis=0))

    return bias_squared, variance

# Example usage
from sklearn.tree import DecisionTreeRegressor

X_train, X_test = X[:80], X[80:]
y_train, y_test = y[:80], y[80:]

# Simple model (high bias, low variance)
bias_sq_simple, var_simple = compute_bias_variance(
    DecisionTreeRegressor, X_train, y_train, X_test, y_test,
    max_depth=2, random_state=42
)

# Complex model (low bias, high variance)
bias_sq_complex, var_complex = compute_bias_variance(
    DecisionTreeRegressor, X_train, y_train, X_test, y_test,
    max_depth=20, random_state=42
)

print("Simple Model (max_depth=2):")
print(f"  Bias²: {bias_sq_simple:.2f}")
print(f"  Variance: {var_simple:.2f}")
print(f"  Total: {bias_sq_simple + var_simple:.2f}")

print("\nComplex Model (max_depth=20):")
print(f"  Bias²: {bias_sq_complex:.2f}")
print(f"  Variance: {var_complex:.2f}")
print(f"  Total: {bias_sq_complex + var_complex:.2f}")
```

---

## The Bias-Variance Tradeoff

As model complexity increases:
- **Bias decreases** (model can fit data better)
- **Variance increases** (model becomes more sensitive to training data)

The goal is to find the sweet spot that minimizes total error.

### Visualizing the Tradeoff

```python
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error

# Generate data
np.random.seed(42)
X = np.sort(np.random.rand(200, 1) * 10, axis=0)
y = 2 + 3*X.ravel() + 5*np.sin(X.ravel()) + np.random.randn(200) * 2

# Split data
X_train, X_test = X[:150], X[150:]
y_train, y_test = y[:150], y[150:]

# Test different model complexities
depths = range(1, 21)
train_errors = []
test_errors = []
bias_squared_values = []
variance_values = []

for depth in depths:
    # Train model
    model = DecisionTreeRegressor(max_depth=depth, random_state=42)
    model.fit(X_train, y_train)

    # Calculate errors
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    train_errors.append(mean_squared_error(y_train, train_pred))
    test_errors.append(mean_squared_error(y_test, test_pred))

    # Estimate bias and variance
    bias_sq, var = compute_bias_variance(
        DecisionTreeRegressor, X_train, y_train, X_test, y_test,
        max_depth=depth, random_state=42, n_iterations=50
    )
    bias_squared_values.append(bias_sq)
    variance_values.append(var)

# Plot bias-variance tradeoff
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

# Plot 1: Train vs Test Error
ax1.plot(depths, train_errors, 'o-', label='Training Error', linewidth=2)
ax1.plot(depths, test_errors, 's-', label='Test Error', linewidth=2)
ax1.axvline(x=depths[np.argmin(test_errors)], color='r', linestyle='--',
            label=f'Optimal Depth = {depths[np.argmin(test_errors)]}')
ax1.set_xlabel('Model Complexity (Tree Depth)')
ax1.set_ylabel('Mean Squared Error')
ax1.set_title('Training vs Test Error')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Bias-Variance Decomposition
ax2.plot(depths, bias_squared_values, 'o-', label='Bias²', linewidth=2)
ax2.plot(depths, variance_values, 's-', label='Variance', linewidth=2)
ax2.plot(depths, np.array(bias_squared_values) + np.array(variance_values),
         '^-', label='Total Error', linewidth=2)
ax2.axvline(x=depths[np.argmin(np.array(bias_squared_values) + np.array(variance_values))],
            color='r', linestyle='--', label='Optimal Complexity')
ax2.set_xlabel('Model Complexity (Tree Depth)')
ax2.set_ylabel('Error')
ax2.set_title('Bias-Variance Tradeoff')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

**Key Observations:**
1. As complexity increases, training error decreases monotonically
2. Test error first decreases, then increases (U-shaped curve)
3. Bias decreases with complexity
4. Variance increases with complexity
5. Optimal complexity minimizes bias² + variance

---

## Overfitting vs Underfitting

### Underfitting (High Bias)

**Definition:** Model is too simple to capture underlying patterns in data.

**Signs:**
- High training error
- High test error
- Training error ≈ Test error
- Model predictions look "too smooth"

```python
# Example: Linear model for nonlinear data
from sklearn.datasets import make_circles
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Generate nonlinear data
X, y = make_circles(n_samples=500, noise=0.1, factor=0.5, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Underfitting model
model_underfit = LogisticRegression()
model_underfit.fit(X_train, y_train)

print("Underfitting Model:")
print(f"  Train Accuracy: {model_underfit.score(X_train, y_train):.3f}")
print(f"  Test Accuracy: {model_underfit.score(X_test, y_test):.3f}")
print(f"  Gap: {abs(model_underfit.score(X_train, y_train) - model_underfit.score(X_test, y_test)):.3f}")

# Visualize decision boundary
def plot_decision_boundary(model, X, y, title):
    h = 0.02
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap='RdYlBu')
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='RdYlBu', edgecolor='black')
    plt.title(title)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.show()

plot_decision_boundary(model_underfit, X_train, y_train,
                      'Underfitting: Linear Boundary for Circular Data')
```

### Overfitting (High Variance)

**Definition:** Model is too complex and fits random noise in training data.

**Signs:**
- Very low training error
- High test error
- Large gap between training and test error
- Model predictions look "too wiggly"

```python
from sklearn.tree import DecisionTreeClassifier

# Overfitting model
model_overfit = DecisionTreeClassifier(max_depth=20, random_state=42)
model_overfit.fit(X_train, y_train)

print("\nOverfitting Model:")
print(f"  Train Accuracy: {model_overfit.score(X_train, y_train):.3f}")
print(f"  Test Accuracy: {model_overfit.score(X_test, y_test):.3f}")
print(f"  Gap: {abs(model_overfit.score(X_train, y_train) - model_overfit.score(X_test, y_test)):.3f}")

plot_decision_boundary(model_overfit, X_train, y_train,
                      'Overfitting: Too Complex Boundary')
```

### Just Right (Good Generalization)

**Definition:** Model captures true patterns without fitting noise.

**Signs:**
- Low training error
- Low test error
- Small gap between training and test error

```python
from sklearn.ensemble import RandomForestClassifier

# Well-fitted model
model_good = RandomForestClassifier(max_depth=5, n_estimators=100, random_state=42)
model_good.fit(X_train, y_train)

print("\nWell-Fitted Model:")
print(f"  Train Accuracy: {model_good.score(X_train, y_train):.3f}")
print(f"  Test Accuracy: {model_good.score(X_test, y_test):.3f}")
print(f"  Gap: {abs(model_good.score(X_train, y_train) - model_good.score(X_test, y_test)):.3f}")

plot_decision_boundary(model_good, X_train, y_train,
                      'Good Fit: Captures Pattern Without Noise')
```

### Quick Reference Table

| Characteristic | Underfitting | Good Fit | Overfitting |
|---------------|--------------|----------|-------------|
| Train Error | High | Low | Very Low |
| Test Error | High | Low | High |
| Gap | Small | Small | Large |
| Bias | High | Balanced | Low |
| Variance | Low | Balanced | High |
| Complexity | Too Simple | Just Right | Too Complex |

---

## Diagnosing Model Problems

### 1. Learning Curves

**Learning curves** show how training and validation errors change with training set size.

```python
from sklearn.model_selection import learning_curve

def plot_learning_curve(model, X, y, title):
    """
    Plot learning curves to diagnose bias/variance.
    """
    train_sizes, train_scores, val_scores = learning_curve(
        model, X, y, cv=5,
        train_sizes=np.linspace(0.1, 1.0, 10),
        scoring='neg_mean_squared_error',
        n_jobs=-1
    )

    train_mean = -train_scores.mean(axis=1)
    train_std = train_scores.std(axis=1)
    val_mean = -val_scores.mean(axis=1)
    val_std = val_scores.std(axis=1)

    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_mean, 'o-', label='Training Error', linewidth=2)
    plt.fill_between(train_sizes, train_mean - train_std,
                     train_mean + train_std, alpha=0.1)
    plt.plot(train_sizes, val_mean, 's-', label='Validation Error', linewidth=2)
    plt.fill_between(train_sizes, val_mean - val_std,
                     val_mean + val_std, alpha=0.1)
    plt.xlabel('Training Set Size')
    plt.ylabel('Mean Squared Error')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

# Generate regression data
X_reg = np.sort(np.random.rand(500, 1) * 10, axis=0)
y_reg = 2 + 3*X_reg.ravel() + 5*np.sin(X_reg.ravel()) + np.random.randn(500) * 2

# High bias model
from sklearn.linear_model import LinearRegression
model_bias = LinearRegression()
plot_learning_curve(model_bias, X_reg, y_reg,
                   'Learning Curve: High Bias (Underfitting)')

# High variance model
from sklearn.tree import DecisionTreeRegressor
model_variance = DecisionTreeRegressor(max_depth=20, random_state=42)
plot_learning_curve(model_variance, X_reg, y_reg,
                   'Learning Curve: High Variance (Overfitting)')

# Good model
from sklearn.ensemble import RandomForestRegressor
model_good = RandomForestRegressor(max_depth=5, n_estimators=100, random_state=42)
plot_learning_curve(model_good, X_reg, y_reg,
                   'Learning Curve: Good Fit')
```

**How to Read Learning Curves:**

**High Bias (Underfitting):**
- Training and validation errors converge
- Both errors are high
- Adding more data doesn't help much
- Curves plateau early

**High Variance (Overfitting):**
- Large gap between training and validation errors
- Training error is low
- Validation error is high
- Adding more data helps (gap decreases)

**Good Fit:**
- Small gap between training and validation errors
- Both errors are reasonably low
- Curves converge to a low error value

### 2. Validation Curves

**Validation curves** show how errors change with a hyperparameter.

```python
from sklearn.model_selection import validation_curve

def plot_validation_curve(model_class, X, y, param_name, param_range, title):
    """
    Plot validation curves for hyperparameter tuning.
    """
    train_scores, val_scores = validation_curve(
        model_class(), X, y,
        param_name=param_name,
        param_range=param_range,
        cv=5,
        scoring='neg_mean_squared_error',
        n_jobs=-1
    )

    train_mean = -train_scores.mean(axis=1)
    train_std = train_scores.std(axis=1)
    val_mean = -val_scores.mean(axis=1)
    val_std = val_scores.std(axis=1)

    plt.figure(figsize=(10, 6))
    plt.plot(param_range, train_mean, 'o-', label='Training Error', linewidth=2)
    plt.fill_between(param_range, train_mean - train_std,
                     train_mean + train_std, alpha=0.1)
    plt.plot(param_range, val_mean, 's-', label='Validation Error', linewidth=2)
    plt.fill_between(param_range, val_mean - val_std,
                     val_mean + val_std, alpha=0.1)
    plt.xlabel(param_name)
    plt.ylabel('Mean Squared Error')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

# Example: max_depth for Decision Tree
param_range = range(1, 21)
plot_validation_curve(
    DecisionTreeRegressor,
    X_reg, y_reg,
    param_name='max_depth',
    param_range=param_range,
    title='Validation Curve: Decision Tree max_depth'
)
```

### 3. Train-Test Error Analysis

**Simple diagnostic:**

```python
def diagnose_model(model, X_train, y_train, X_test, y_test, threshold=0.05):
    """
    Diagnose if model is underfitting, overfitting, or good.
    """
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    gap = train_score - test_score

    print(f"Train Score: {train_score:.4f}")
    print(f"Test Score: {test_score:.4f}")
    print(f"Gap: {gap:.4f}")

    if train_score < 0.7:
        print("\n⚠️ UNDERFITTING (High Bias)")
        print("  - Train score is low")
        print("  - Model is too simple")
        print("  - Suggestions: Use more complex model, add features, reduce regularization")
    elif gap > threshold:
        print("\n⚠️ OVERFITTING (High Variance)")
        print("  - Large gap between train and test scores")
        print("  - Model is too complex")
        print("  - Suggestions: Use regularization, get more data, reduce model complexity")
    else:
        print("\n✓ GOOD FIT")
        print("  - Low training error")
        print("  - Low test error")
        print("  - Small gap")

# Example usage
from sklearn.ensemble import RandomForestRegressor

X_train, X_test = X_reg[:400], X_reg[400:]
y_train, y_test = y_reg[:400], y_reg[400:]

# Test underfitting
model_under = LinearRegression()
model_under.fit(X_train, y_train)
print("=" * 50)
print("Diagnosing Underfitting Model:")
print("=" * 50)
diagnose_model(model_under, X_train, y_train, X_test, y_test)

# Test overfitting
model_over = DecisionTreeRegressor(max_depth=20, random_state=42)
model_over.fit(X_train, y_train)
print("\n" + "=" * 50)
print("Diagnosing Overfitting Model:")
print("=" * 50)
diagnose_model(model_over, X_train, y_train, X_test, y_test)

# Test good fit
model_good = RandomForestRegressor(max_depth=5, n_estimators=100, random_state=42)
model_good.fit(X_train, y_train)
print("\n" + "=" * 50)
print("Diagnosing Good Model:")
print("=" * 50)
diagnose_model(model_good, X_train, y_train, X_test, y_test)
```

---

## Solutions to Overfitting

### 1. Get More Training Data

More data reduces variance by averaging out noise.

```python
# Demonstrate effect of training data size
train_sizes = [50, 100, 200, 500, 1000, 2000]
test_errors = []

for size in train_sizes:
    # Generate data
    X_large = np.sort(np.random.rand(size + 200, 1) * 10, axis=0)
    y_large = 2 + 3*X_large.ravel() + 5*np.sin(X_large.ravel()) + np.random.randn(size + 200) * 2

    X_tr, X_te = X_large[:size], X_large[size:size+200]
    y_tr, y_te = y_large[:size], y_large[size:size+200]

    # Train complex model
    model = DecisionTreeRegressor(max_depth=10, random_state=42)
    model.fit(X_tr, y_tr)

    test_errors.append(mean_squared_error(y_te, model.predict(X_te)))

plt.figure(figsize=(10, 6))
plt.plot(train_sizes, test_errors, 'o-', linewidth=2)
plt.xlabel('Training Set Size')
plt.ylabel('Test Error')
plt.title('Effect of Training Data Size on Overfitting')
plt.grid(True, alpha=0.3)
plt.show()
```

### 2. Use Regularization

Regularization adds penalty for model complexity.

```python
from sklearn.linear_model import Ridge, Lasso

# Without regularization
poly = PolynomialFeatures(degree=10)
X_poly_train = poly.fit_transform(X_train)
X_poly_test = poly.transform(X_test)

model_no_reg = LinearRegression()
model_no_reg.fit(X_poly_train, y_train)

# With L2 regularization (Ridge)
model_ridge = Ridge(alpha=10.0)
model_ridge.fit(X_poly_train, y_train)

# With L1 regularization (Lasso)
model_lasso = Lasso(alpha=0.1)
model_lasso.fit(X_poly_train, y_train)

print("No Regularization:")
print(f"  Train Score: {model_no_reg.score(X_poly_train, y_train):.4f}")
print(f"  Test Score: {model_no_reg.score(X_poly_test, y_test):.4f}")

print("\nRidge (L2) Regularization:")
print(f"  Train Score: {model_ridge.score(X_poly_train, y_train):.4f}")
print(f"  Test Score: {model_ridge.score(X_poly_test, y_test):.4f}")

print("\nLasso (L1) Regularization:")
print(f"  Train Score: {model_lasso.score(X_poly_train, y_train):.4f}")
print(f"  Test Score: {model_lasso.score(X_poly_test, y_test):.4f}")

# Compare coefficient magnitudes
print(f"\nCoefficient magnitudes:")
print(f"  No regularization: {np.abs(model_no_reg.coef_).mean():.2f}")
print(f"  Ridge: {np.abs(model_ridge.coef_).mean():.2f}")
print(f"  Lasso: {np.abs(model_lasso.coef_).mean():.2f}")
```

### 3. Reduce Model Complexity

Use simpler models or constrain complexity.

```python
# Compare different tree depths
depths = [3, 5, 10, 20]

for depth in depths:
    model = DecisionTreeRegressor(max_depth=depth, random_state=42)
    model.fit(X_train, y_train)

    print(f"max_depth={depth}:")
    print(f"  Train Score: {model.score(X_train, y_train):.4f}")
    print(f"  Test Score: {model.score(X_test, y_test):.4f}")
    print(f"  Gap: {model.score(X_train, y_train) - model.score(X_test, y_test):.4f}")
    print()
```

### 4. Feature Selection

Remove irrelevant or redundant features.

```python
from sklearn.feature_selection import SelectKBest, f_regression

# Add noise features
X_with_noise = np.hstack([X_reg, np.random.randn(len(X_reg), 20)])
X_train_noise, X_test_noise = X_with_noise[:400], X_with_noise[400:]

# Without feature selection
model_all = RandomForestRegressor(random_state=42)
model_all.fit(X_train_noise, y_train)

# With feature selection
selector = SelectKBest(f_regression, k=5)
X_train_selected = selector.fit_transform(X_train_noise, y_train)
X_test_selected = selector.transform(X_test_noise)

model_selected = RandomForestRegressor(random_state=42)
model_selected.fit(X_train_selected, y_train)

print("Without Feature Selection (21 features):")
print(f"  Train Score: {model_all.score(X_train_noise, y_train):.4f}")
print(f"  Test Score: {model_all.score(X_test_noise, y_test):.4f}")

print("\nWith Feature Selection (5 features):")
print(f"  Train Score: {model_selected.score(X_train_selected, y_train):.4f}")
print(f"  Test Score: {model_selected.score(X_test_selected, y_test):.4f}")
```

### 5. Ensemble Methods

Combine multiple models to reduce variance.

```python
from sklearn.ensemble import BaggingRegressor, RandomForestRegressor

# Single decision tree (high variance)
single_tree = DecisionTreeRegressor(max_depth=10, random_state=42)
single_tree.fit(X_train, y_train)

# Bagging (reduces variance)
bagging = BaggingRegressor(
    DecisionTreeRegressor(max_depth=10),
    n_estimators=100, random_state=42
)
bagging.fit(X_train, y_train)

# Random Forest (bagging + feature randomness)
rf = RandomForestRegressor(max_depth=10, n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

print("Single Decision Tree:")
print(f"  Train Score: {single_tree.score(X_train, y_train):.4f}")
print(f"  Test Score: {single_tree.score(X_test, y_test):.4f}")

print("\nBagging:")
print(f"  Train Score: {bagging.score(X_train, y_train):.4f}")
print(f"  Test Score: {bagging.score(X_test, y_test):.4f}")

print("\nRandom Forest:")
print(f"  Train Score: {rf.score(X_train, y_train):.4f}")
print(f"  Test Score: {rf.score(X_test, y_test):.4f}")
```

### 6. Dropout (for Neural Networks)

```python
import tensorflow as tf
from tensorflow import keras

# Without dropout
model_no_dropout = keras.Sequential([
    keras.layers.Dense(128, activation='relu', input_shape=(10,)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(1)
])

# With dropout
model_dropout = keras.Sequential([
    keras.layers.Dense(128, activation='relu', input_shape=(10,)),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(1)
])

print("Dropout randomly deactivates 50% of neurons during training")
print("This prevents co-adaptation and reduces overfitting")
```

### 7. Early Stopping

```python
from sklearn.neural_network import MLPRegressor

# Train with early stopping
model_early_stop = MLPRegressor(
    hidden_layer_sizes=(100, 100),
    max_iter=1000,
    early_stopping=True,
    validation_fraction=0.2,
    n_iter_no_change=10,
    random_state=42
)

model_early_stop.fit(X_train, y_train)

print(f"Stopped at iteration: {model_early_stop.n_iter_}")
print(f"Best validation score: {model_early_stop.best_validation_score_:.4f}")
```

### 8. Data Augmentation

```python
# Example: Add noise to create more training samples
def augment_data(X, y, n_augmentations=5, noise_level=0.1):
    """
    Create augmented samples by adding noise.
    """
    X_aug = [X]
    y_aug = [y]

    for _ in range(n_augmentations):
        X_noisy = X + np.random.randn(*X.shape) * noise_level
        X_aug.append(X_noisy)
        y_aug.append(y)

    return np.vstack(X_aug), np.concatenate(y_aug)

X_train_aug, y_train_aug = augment_data(X_train, y_train)

print(f"Original training size: {len(X_train)}")
print(f"Augmented training size: {len(X_train_aug)}")

model = DecisionTreeRegressor(max_depth=10, random_state=42)

# Train on original data
model.fit(X_train, y_train)
score_original = model.score(X_test, y_test)

# Train on augmented data
model.fit(X_train_aug, y_train_aug)
score_augmented = model.score(X_test, y_test)

print(f"\nTest score (original): {score_original:.4f}")
print(f"Test score (augmented): {score_augmented:.4f}")
```

---

## Solutions to Underfitting

### 1. Increase Model Complexity

```python
# Simple model (underfitting)
model_simple = DecisionTreeRegressor(max_depth=2, random_state=42)
model_simple.fit(X_train, y_train)

# More complex model
model_complex = DecisionTreeRegressor(max_depth=10, random_state=42)
model_complex.fit(X_train, y_train)

print("Simple Model (max_depth=2):")
print(f"  Train Score: {model_simple.score(X_train, y_train):.4f}")
print(f"  Test Score: {model_simple.score(X_test, y_test):.4f}")

print("\nComplex Model (max_depth=10):")
print(f"  Train Score: {model_complex.score(X_train, y_train):.4f}")
print(f"  Test Score: {model_complex.score(X_test, y_test):.4f}")
```

### 2. Add More Features

```python
# Original features
X_original = X_reg

# Engineer new features
X_engineered = np.hstack([
    X_original,
    np.sin(X_original),
    np.cos(X_original),
    X_original ** 2,
    np.sqrt(np.abs(X_original))
])

X_train_orig, X_test_orig = X_original[:400], X_original[400:]
X_train_eng, X_test_eng = X_engineered[:400], X_engineered[400:]

# Train with original features
model_orig = LinearRegression()
model_orig.fit(X_train_orig, y_train)

# Train with engineered features
model_eng = LinearRegression()
model_eng.fit(X_train_eng, y_train)

print("Original Features (1 feature):")
print(f"  Train Score: {model_orig.score(X_train_orig, y_train):.4f}")
print(f"  Test Score: {model_orig.score(X_test_orig, y_test):.4f}")

print("\nEngineered Features (5 features):")
print(f"  Train Score: {model_eng.score(X_train_eng, y_train):.4f}")
print(f"  Test Score: {model_eng.score(X_test_eng, y_test):.4f}")
```

### 3. Reduce Regularization

```python
# High regularization (underfitting)
model_high_reg = Ridge(alpha=1000.0)
model_high_reg.fit(X_poly_train, y_train)

# Lower regularization
model_low_reg = Ridge(alpha=1.0)
model_low_reg.fit(X_poly_train, y_train)

print("High Regularization (alpha=1000):")
print(f"  Train Score: {model_high_reg.score(X_poly_train, y_train):.4f}")
print(f"  Test Score: {model_high_reg.score(X_poly_test, y_test):.4f}")

print("\nLower Regularization (alpha=1):")
print(f"  Train Score: {model_low_reg.score(X_poly_train, y_train):.4f}")
print(f"  Test Score: {model_low_reg.score(X_poly_test, y_test):.4f}")
```

### 4. Train Longer

```python
from sklearn.neural_network import MLPRegressor

# Train for few iterations
model_short = MLPRegressor(hidden_layer_sizes=(100,), max_iter=10, random_state=42)
model_short.fit(X_train, y_train)

# Train for more iterations
model_long = MLPRegressor(hidden_layer_sizes=(100,), max_iter=500, random_state=42)
model_long.fit(X_train, y_train)

print("Short Training (10 iterations):")
print(f"  Train Score: {model_short.score(X_train, y_train):.4f}")
print(f"  Test Score: {model_short.score(X_test, y_test):.4f}")

print("\nLong Training (500 iterations):")
print(f"  Train Score: {model_long.score(X_train, y_train):.4f}")
print(f"  Test Score: {model_long.score(X_test, y_test):.4f}")
```

### 5. Use Different Algorithm

```python
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor

algorithms = {
    'Linear Regression': LinearRegression(),
    'Decision Tree': DecisionTreeRegressor(max_depth=5, random_state=42),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'SVM': SVR(kernel='rbf'),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
}

results = []
for name, model in algorithms.items():
    model.fit(X_train, y_train)
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    results.append({
        'Algorithm': name,
        'Train Score': train_score,
        'Test Score': test_score,
        'Gap': train_score - test_score
    })

import pandas as pd
df_results = pd.DataFrame(results)
print(df_results.to_string(index=False))
```

---

## Model Complexity and Performance

### Complexity vs Performance Chart

```python
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR

# Define models with increasing complexity
models = [
    ('Linear (simple)', LinearRegression()),
    ('Ridge (α=10)', Ridge(alpha=10)),
    ('Ridge (α=1)', Ridge(alpha=1)),
    ('Tree (depth=3)', DecisionTreeRegressor(max_depth=3, random_state=42)),
    ('Tree (depth=10)', DecisionTreeRegressor(max_depth=10, random_state=42)),
    ('Random Forest (n=10)', RandomForestRegressor(n_estimators=10, random_state=42)),
    ('Random Forest (n=100)', RandomForestRegressor(n_estimators=100, random_state=42)),
    ('GBM', GradientBoostingRegressor(n_estimators=100, random_state=42)),
]

train_scores = []
test_scores = []

for name, model in models:
    model.fit(X_train, y_train)
    train_scores.append(model.score(X_train, y_train))
    test_scores.append(model.score(X_test, y_test))

# Plot
x_pos = np.arange(len(models))
plt.figure(figsize=(14, 6))
plt.plot(x_pos, train_scores, 'o-', label='Train Score', linewidth=2, markersize=8)
plt.plot(x_pos, test_scores, 's-', label='Test Score', linewidth=2, markersize=8)
plt.xticks(x_pos, [name for name, _ in models], rotation=45, ha='right')
plt.ylabel('R² Score')
plt.title('Model Complexity vs Performance')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Identify best model
best_idx = np.argmax(test_scores)
print(f"\nBest Model: {models[best_idx][0]}")
print(f"Test Score: {test_scores[best_idx]:.4f}")
```

---

## Practical Examples

### Example 1: Complete Workflow

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score

# Step 1: Generate data
np.random.seed(42)
X = np.sort(np.random.rand(300, 1) * 10, axis=0)
y = 2 + 3*X.ravel() + 5*np.sin(X.ravel()) + np.random.randn(300) * 2

# Step 2: Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Step 3: Try different polynomial degrees
degrees = [1, 3, 5, 10, 15]
results = []

for degree in degrees:
    # Transform features
    poly = PolynomialFeatures(degree=degree)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)

    # Train model with regularization
    model = Ridge(alpha=1.0)
    model.fit(X_train_poly, y_train)

    # Evaluate
    train_pred = model.predict(X_train_poly)
    test_pred = model.predict(X_test_poly)

    train_mse = mean_squared_error(y_train, train_pred)
    test_mse = mean_squared_error(y_test, test_pred)
    train_r2 = r2_score(y_train, train_pred)
    test_r2 = r2_score(y_test, test_pred)

    results.append({
        'Degree': degree,
        'Train MSE': train_mse,
        'Test MSE': test_mse,
        'Train R²': train_r2,
        'Test R²': test_r2,
        'Gap': train_r2 - test_r2
    })

    print(f"Degree {degree}:")
    print(f"  Train R²: {train_r2:.4f}, Test R²: {test_r2:.4f}, Gap: {train_r2 - test_r2:.4f}")

# Step 4: Find best degree
df_results = pd.DataFrame(results)
best_degree = df_results.loc[df_results['Test R²'].idxmax(), 'Degree']
print(f"\nBest degree: {best_degree}")

# Step 5: Train final model
poly_best = PolynomialFeatures(degree=int(best_degree))
X_train_best = poly_best.fit_transform(X_train)
X_test_best = poly_best.transform(X_test)

final_model = Ridge(alpha=1.0)
final_model.fit(X_train_best, y_train)

# Step 6: Visualize
X_plot = np.linspace(0, 10, 300).reshape(-1, 1)
X_plot_poly = poly_best.transform(X_plot)
y_plot = final_model.predict(X_plot_poly)

plt.figure(figsize=(12, 6))
plt.scatter(X_train, y_train, alpha=0.5, label='Training Data')
plt.scatter(X_test, y_test, alpha=0.5, label='Test Data')
plt.plot(X_plot, y_plot, 'r-', linewidth=2, label=f'Model (degree={int(best_degree)})')
plt.plot(X_plot, 2 + 3*X_plot.ravel() + 5*np.sin(X_plot.ravel()),
         'g--', linewidth=2, label='True Function')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Final Model')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

### Example 2: Real-World Dataset

```python
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score

# Load data
data = fetch_california_housing()
X, y = data.data, data.target

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Test different max_depths
depths = [3, 5, 10, 15, 20, None]

for depth in depths:
    model = RandomForestRegressor(
        max_depth=depth,
        n_estimators=100,
        random_state=42
    )

    # Cross-validation scores
    cv_scores = cross_val_score(
        model, X_train_scaled, y_train,
        cv=5, scoring='r2'
    )

    # Test score
    model.fit(X_train_scaled, y_train)
    test_score = model.score(X_test_scaled, y_test)

    print(f"max_depth={depth}:")
    print(f"  CV Score: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    print(f"  Test Score: {test_score:.4f}")
    print()
```

---

## Best Practices

### 1. Always Use Separate Train/Validation/Test Sets

```python
# Three-way split
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.25, random_state=42
)  # 0.25 * 0.8 = 0.2 of total

print(f"Train: {len(X_train)} samples (60%)")
print(f"Validation: {len(X_val)} samples (20%)")
print(f"Test: {len(X_test)} samples (20%)")
```

### 2. Start Simple, Then Increase Complexity

```python
# Start with simplest baseline
baseline = LinearRegression()
baseline.fit(X_train, y_train)
baseline_score = baseline.score(X_test, y_test)

print(f"Baseline (Linear): {baseline_score:.4f}")

# If baseline is poor, try more complex models
if baseline_score < 0.7:
    print("Baseline is poor, trying more complex models...")
    # Try polynomial features, tree-based models, etc.
```

### 3. Use Cross-Validation

```python
from sklearn.model_selection import cross_val_score

# Single train-test split (can be misleading)
model.fit(X_train, y_train)
single_score = model.score(X_test, y_test)

# Cross-validation (more robust)
cv_scores = cross_val_score(model, X, y, cv=5)
cv_mean = cv_scores.mean()
cv_std = cv_scores.std()

print(f"Single split: {single_score:.4f}")
print(f"Cross-validation: {cv_mean:.4f} ± {cv_std:.4f}")
```

### 4. Monitor Training Progress

```python
# For models that support partial_fit or have training history
from sklearn.neural_network import MLPRegressor

model = MLPRegressor(hidden_layer_sizes=(100,), max_iter=500, random_state=42)
model.fit(X_train, y_train)

# Plot loss curve
plt.figure(figsize=(10, 6))
plt.plot(model.loss_curve_)
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Training Loss Over Time')
plt.grid(True, alpha=0.3)
plt.show()
```

### 5. Document Your Experiments

```python
import json
from datetime import datetime

experiment_log = {
    'timestamp': str(datetime.now()),
    'model': 'RandomForestRegressor',
    'hyperparameters': {
        'n_estimators': 100,
        'max_depth': 10,
        'random_state': 42
    },
    'train_score': 0.8532,
    'val_score': 0.7845,
    'test_score': 0.7912,
    'notes': 'Good generalization, slight overfitting'
}

# Save to file
with open('experiment_log.json', 'a') as f:
    json.dump(experiment_log, f)
    f.write('\n')
```

### Quick Decision Guide

```
Is train error high?
├─ YES → Underfitting
│   ├─ Increase model complexity
│   ├─ Add more features
│   ├─ Reduce regularization
│   └─ Train longer
│
└─ NO → Is test error high?
    ├─ YES → Overfitting
    │   ├─ Get more training data
    │   ├─ Use regularization
    │   ├─ Reduce model complexity
    │   ├─ Feature selection
    │   ├─ Use ensemble methods
    │   └─ Use dropout/early stopping
    │
    └─ NO → Good fit!
        └─ Consider slight tuning for marginal improvements
```

---

## Summary

### Key Takeaways

1. **Bias-Variance Tradeoff is fundamental** - You can't minimize both simultaneously
2. **Use learning curves** - Best diagnostic tool for bias/variance problems
3. **Start simple** - Baseline model first, then increase complexity if needed
4. **Monitor the gap** - Large train-test gap indicates overfitting
5. **Cross-validation is essential** - Single split can be misleading
6. **Regularization is powerful** - Often better than reducing complexity
7. **More data helps variance** - But not bias
8. **Ensemble methods reduce variance** - Without increasing bias

### Checklist

- [ ] Split data into train/validation/test sets
- [ ] Start with simple baseline model
- [ ] Plot learning curves
- [ ] Check train vs test error
- [ ] If underfitting: increase complexity, add features
- [ ] If overfitting: regularize, get more data, reduce complexity
- [ ] Use cross-validation for final evaluation
- [ ] Document experiments and results

Understanding and managing the bias-variance tradeoff is crucial for building models that generalize well to new data!
