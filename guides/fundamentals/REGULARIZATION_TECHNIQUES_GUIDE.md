# Regularization Techniques Guide

A comprehensive guide to regularization methods for preventing overfitting in machine learning models.

## Table of Contents
- [What is Regularization?](#what-is-regularization)
- [L2 Regularization (Ridge)](#l2-regularization-ridge)
- [L1 Regularization (Lasso)](#l1-regularization-lasso)
- [Elastic Net](#elastic-net)
- [Dropout](#dropout)
- [Early Stopping](#early-stopping)
- [Data Augmentation](#data-augmentation)
- [Batch Normalization](#batch-normalization)
- [Weight Decay](#weight-decay)
- [When to Use Which Technique](#when-to-use-which-technique)
- [Practical Examples](#practical-examples)
- [Best Practices](#best-practices)

---

## What is Regularization?

**Regularization** is any technique that helps prevent overfitting by constraining model complexity.

### The Problem: Overfitting

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Generate data
np.random.seed(42)
X = np.linspace(0, 10, 100).reshape(-1, 1)
y = 2 + 3*X.ravel() + 5*np.sin(X.ravel()) + np.random.randn(100) * 2

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# High-degree polynomial without regularization
poly = PolynomialFeatures(degree=15)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

model_overfit = LinearRegression()
model_overfit.fit(X_train_poly, y_train)

print("Without Regularization:")
print(f"  Train R²: {model_overfit.score(X_train_poly, y_train):.4f}")
print(f"  Test R²: {model_overfit.score(X_test_poly, y_test):.4f}")
print(f"  Coefficient magnitudes: {np.abs(model_overfit.coef_).max():.2e}")

# Visualize
X_plot = np.linspace(0, 10, 300).reshape(-1, 1)
X_plot_poly = poly.transform(X_plot)
y_plot = model_overfit.predict(X_plot_poly)

plt.figure(figsize=(12, 6))
plt.scatter(X_train, y_train, alpha=0.5, label='Training Data')
plt.scatter(X_test, y_test, alpha=0.5, label='Test Data')
plt.plot(X_plot, y_plot, 'r-', linewidth=2, label='Overfitted Model')
plt.plot(X_plot, 2 + 3*X_plot.ravel() + 5*np.sin(X_plot.ravel()),
         'g--', linewidth=2, label='True Function')
plt.ylim(-20, 40)
plt.legend()
plt.title('Overfitting Without Regularization')
plt.show()
```

**Observation:** Coefficients become extremely large, causing wild oscillations.

### How Regularization Helps

Regularization adds a penalty term to the loss function:

```
Loss = Data Loss + λ × Regularization Term
```

Where:
- **Data Loss**: How well model fits training data (e.g., MSE)
- **λ (lambda)**: Regularization strength (hyperparameter)
- **Regularization Term**: Penalty for model complexity

**Effect:**
- Forces model to balance fitting data and staying simple
- Prevents coefficients from becoming too large
- Improves generalization

---

## L2 Regularization (Ridge)

**Also called:** Ridge Regression, Weight Decay, Tikhonov Regularization

### Formula

```
Loss = MSE + λ × Σ(wi²)
```

Penalizes sum of squared weights.

### Implementation

```python
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

# Ridge regression with different alpha values
alphas = [0, 0.01, 0.1, 1, 10, 100]
results = []

for alpha in alphas:
    model = Ridge(alpha=alpha)
    model.fit(X_train_poly, y_train)

    train_score = model.score(X_train_poly, y_train)
    test_score = model.score(X_test_poly, y_test)
    max_coef = np.abs(model.coef_).max()

    results.append({
        'Alpha': alpha,
        'Train R²': train_score,
        'Test R²': test_score,
        'Gap': train_score - test_score,
        'Max |Coef|': max_coef
    })

    print(f"Alpha={alpha}:")
    print(f"  Train R²: {train_score:.4f}, Test R²: {test_score:.4f}")
    print(f"  Max coefficient: {max_coef:.2e}")
```

### Visualizing Ridge Effect

```python
import pandas as pd

df_results = pd.DataFrame(results)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Plot 1: Train vs Test scores
axes[0].plot(df_results['Alpha'], df_results['Train R²'], 'o-',
            label='Train R²', linewidth=2)
axes[0].plot(df_results['Alpha'], df_results['Test R²'], 's-',
            label='Test R²', linewidth=2)
axes[0].set_xscale('log')
axes[0].set_xlabel('Alpha (λ)')
axes[0].set_ylabel('R² Score')
axes[0].set_title('Ridge: Performance vs Regularization')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Plot 2: Gap
axes[1].plot(df_results['Alpha'], df_results['Gap'], 'o-', linewidth=2)
axes[1].set_xscale('log')
axes[1].set_xlabel('Alpha (λ)')
axes[1].set_ylabel('Train - Test Gap')
axes[1].set_title('Overfitting Gap')
axes[1].grid(True, alpha=0.3)

# Plot 3: Coefficient magnitudes
axes[2].plot(df_results['Alpha'], df_results['Max |Coef|'], 'o-', linewidth=2)
axes[2].set_xscale('log')
axes[2].set_yscale('log')
axes[2].set_xlabel('Alpha (λ)')
axes[2].set_ylabel('Max |Coefficient|')
axes[2].set_title('Coefficient Shrinkage')
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

### Coefficient Paths

```python
# Show how coefficients shrink with regularization
alphas_range = np.logspace(-2, 4, 100)
coef_paths = []

for alpha in alphas_range:
    model = Ridge(alpha=alpha)
    model.fit(X_train_poly, y_train)
    coef_paths.append(model.coef_)

coef_paths = np.array(coef_paths)

plt.figure(figsize=(12, 6))
for i in range(coef_paths.shape[1]):
    plt.plot(alphas_range, coef_paths[:, i], linewidth=1)

plt.xscale('log')
plt.xlabel('Alpha (λ)')
plt.ylabel('Coefficient Value')
plt.title('Ridge: Coefficient Paths')
plt.grid(True, alpha=0.3)
plt.axhline(y=0, color='k', linestyle='--', linewidth=0.5)
plt.show()
```

### Properties of Ridge

**Advantages:**
- ✅ Smooth, continuous shrinkage
- ✅ Computationally efficient
- ✅ Handles correlated features well
- ✅ Always has a solution

**Disadvantages:**
- ❌ Doesn't perform feature selection (keeps all features)
- ❌ Coefficients never exactly zero

**When to use:**
- Many features with small-to-medium effects
- Correlated features
- When you want to keep all features

---

## L1 Regularization (Lasso)

**Also called:** Lasso Regression (Least Absolute Shrinkage and Selection Operator)

### Formula

```
Loss = MSE + λ × Σ|wi|
```

Penalizes sum of absolute values of weights.

### Implementation

```python
from sklearn.linear_model import Lasso

# Lasso regression with different alpha values
alphas = [0.001, 0.01, 0.1, 1, 10]
results_lasso = []

for alpha in alphas:
    model = Lasso(alpha=alpha, max_iter=10000)
    model.fit(X_train_poly, y_train)

    train_score = model.score(X_train_poly, y_train)
    test_score = model.score(X_test_poly, y_test)
    n_nonzero = np.sum(model.coef_ != 0)

    results_lasso.append({
        'Alpha': alpha,
        'Train R²': train_score,
        'Test R²': test_score,
        'Non-zero Coefs': n_nonzero,
        'Total Coefs': len(model.coef_)
    })

    print(f"Alpha={alpha}:")
    print(f"  Train R²: {train_score:.4f}, Test R²: {test_score:.4f}")
    print(f"  Non-zero coefficients: {n_nonzero}/{len(model.coef_)}")
```

### Feature Selection with Lasso

```python
# Visualize which features Lasso selects
model_lasso = Lasso(alpha=0.1, max_iter=10000)
model_lasso.fit(X_train_poly, y_train)

# Plot coefficients
plt.figure(figsize=(12, 6))
plt.bar(range(len(model_lasso.coef_)), model_lasso.coef_)
plt.xlabel('Feature Index')
plt.ylabel('Coefficient Value')
plt.title(f'Lasso Coefficients (α=0.1) - {np.sum(model_lasso.coef_ != 0)} non-zero')
plt.axhline(y=0, color='k', linestyle='--', linewidth=0.5)
plt.grid(True, alpha=0.3, axis='y')
plt.show()

print(f"Features set to zero: {np.sum(model_lasso.coef_ == 0)}/{len(model_lasso.coef_)}")
```

### Coefficient Paths for Lasso

```python
# Show how coefficients go to exactly zero
alphas_range = np.logspace(-3, 1, 100)
coef_paths_lasso = []

for alpha in alphas_range:
    model = Lasso(alpha=alpha, max_iter=10000)
    model.fit(X_train_poly, y_train)
    coef_paths_lasso.append(model.coef_)

coef_paths_lasso = np.array(coef_paths_lasso)

plt.figure(figsize=(12, 6))
for i in range(coef_paths_lasso.shape[1]):
    plt.plot(alphas_range, coef_paths_lasso[:, i], linewidth=1)

plt.xscale('log')
plt.xlabel('Alpha (λ)')
plt.ylabel('Coefficient Value')
plt.title('Lasso: Coefficient Paths (Note: Some paths hit zero)')
plt.grid(True, alpha=0.3)
plt.axhline(y=0, color='k', linestyle='--', linewidth=0.5)
plt.show()
```

### Properties of Lasso

**Advantages:**
- ✅ Performs automatic feature selection
- ✅ Produces sparse models (many coefficients = 0)
- ✅ Interpretable (fewer features to explain)
- ✅ Good for high-dimensional data

**Disadvantages:**
- ❌ Can be unstable with correlated features (small data changes → large coefficient changes)
- ❌ Arbitrarily selects one feature from correlated group (non-deterministic selection)
- ❌ Slower to compute than Ridge (no closed-form solution, requires iterative optimization)
- ❌ May struggle when p > n (more features than samples) without additional constraints

**Why instability with correlated features?**
When features are highly correlated, Lasso tends to arbitrarily pick one and ignore others. For example, if features X₁ and X₂ are identical, Lasso might give all weight to X₁ in one run and all to X₂ in another run with slightly different data. Ridge, by contrast, distributes weight equally among correlated features.

**When to use:**
- High-dimensional data (many features)
- When you believe many features are irrelevant
- When interpretability is important
- When you want automatic feature selection

---

## Elastic Net

**Combines L1 and L2 regularization**

### Formula

```
Loss = MSE + λ₁ × Σ|wi| + λ₂ × Σ(wi²)
```

Or equivalently:
```
Loss = MSE + λ × (α × Σ|wi| + (1-α) × Σ(wi²))
```

Where α is the mixing parameter (0 ≤ α ≤ 1).

### Implementation

```python
from sklearn.linear_model import ElasticNet

# Test different l1_ratio values
l1_ratios = [0, 0.25, 0.5, 0.75, 1.0]  # 0=Ridge, 1=Lasso

results_elastic = []

for l1_ratio in l1_ratios:
    model = ElasticNet(alpha=0.1, l1_ratio=l1_ratio, max_iter=10000)
    model.fit(X_train_poly, y_train)

    train_score = model.score(X_train_poly, y_train)
    test_score = model.score(X_test_poly, y_test)
    n_nonzero = np.sum(model.coef_ != 0)

    results_elastic.append({
        'L1 Ratio': l1_ratio,
        'Train R²': train_score,
        'Test R²': test_score,
        'Non-zero Coefs': n_nonzero
    })

    print(f"L1 Ratio={l1_ratio} ({'Lasso' if l1_ratio==1 else 'Ridge' if l1_ratio==0 else 'Elastic Net'}):")
    print(f"  Train R²: {train_score:.4f}, Test R²: {test_score:.4f}")
    print(f"  Non-zero coefficients: {n_nonzero}/{len(model.coef_)}")
```

### Visualizing Elastic Net Effect

```python
# Compare Ridge, Lasso, and Elastic Net
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

models = [
    ('Ridge (L2)', Ridge(alpha=0.1)),
    ('Elastic Net', ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=10000)),
    ('Lasso (L1)', Lasso(alpha=0.1, max_iter=10000))
]

for ax, (name, model) in zip(axes, models):
    model.fit(X_train_poly, y_train)

    ax.bar(range(len(model.coef_)), model.coef_)
    ax.set_xlabel('Feature Index')
    ax.set_ylabel('Coefficient Value')
    ax.set_title(f'{name}\nNon-zero: {np.sum(model.coef_ != 0)}/{len(model.coef_)}')
    ax.axhline(y=0, color='k', linestyle='--', linewidth=0.5)
    ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.show()
```

### Properties of Elastic Net

**Advantages:**
- ✅ Best of both worlds: feature selection + stable with correlated features
- ✅ Works well when features are correlated
- ✅ Can select groups of correlated features

**Disadvantages:**
- ❌ Two hyperparameters to tune (α and λ)
- ❌ More computationally expensive

**When to use:**
- Correlated features + want feature selection
- Not sure whether to use Ridge or Lasso
- High-dimensional data with feature groups

---

## Dropout

**Randomly deactivates neurons during training** (for neural networks)

### How It Works

During training:
1. Randomly set some neurons to 0 (with probability p)
2. Scale remaining neurons by 1/(1-p)
3. Forces network to not rely on any single neuron

During testing:
- Use all neurons (no dropout)

```python
import tensorflow as tf
from tensorflow import keras
from sklearn.datasets import make_regression
from sklearn.preprocessing import StandardScaler

# Generate data
X_nn, y_nn = make_regression(n_samples=1000, n_features=20, noise=10, random_state=42)
X_train_nn, X_test_nn, y_train_nn, y_test_nn = train_test_split(
    X_nn, y_nn, test_size=0.2, random_state=42
)

# Scale data
scaler = StandardScaler()
X_train_nn = scaler.fit_transform(X_train_nn)
X_test_nn = scaler.transform(X_test_nn)

# Model WITHOUT dropout
model_no_dropout = keras.Sequential([
    keras.layers.Dense(128, activation='relu', input_shape=(20,)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(1)
])

model_no_dropout.compile(optimizer='adam', loss='mse', metrics=['mae'])

history_no_dropout = model_no_dropout.fit(
    X_train_nn, y_train_nn,
    validation_split=0.2,
    epochs=100,
    batch_size=32,
    verbose=0
)

# Model WITH dropout
model_dropout = keras.Sequential([
    keras.layers.Dense(128, activation='relu', input_shape=(20,)),
    keras.layers.Dropout(0.5),  # 50% dropout
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(1)
])

model_dropout.compile(optimizer='adam', loss='mse', metrics=['mae'])

history_dropout = model_dropout.fit(
    X_train_nn, y_train_nn,
    validation_split=0.2,
    epochs=100,
    batch_size=32,
    verbose=0
)

# Compare training curves
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Without dropout
axes[0].plot(history_no_dropout.history['loss'], label='Train Loss')
axes[0].plot(history_no_dropout.history['val_loss'], label='Val Loss')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].set_title('Without Dropout')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# With dropout
axes[1].plot(history_dropout.history['loss'], label='Train Loss')
axes[1].plot(history_dropout.history['val_loss'], label='Val Loss')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Loss')
axes[1].set_title('With Dropout (p=0.5)')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Final test performance
test_loss_no_dropout = model_no_dropout.evaluate(X_test_nn, y_test_nn, verbose=0)[0]
test_loss_dropout = model_dropout.evaluate(X_test_nn, y_test_nn, verbose=0)[0]

print(f"Test Loss without Dropout: {test_loss_no_dropout:.2f}")
print(f"Test Loss with Dropout: {test_loss_dropout:.2f}")
```

### Different Dropout Rates

```python
# Test different dropout rates
dropout_rates = [0, 0.1, 0.3, 0.5, 0.7]
results_dropout = []

for rate in dropout_rates:
    model = keras.Sequential([
        keras.layers.Dense(128, activation='relu', input_shape=(20,)),
        keras.layers.Dropout(rate),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dropout(rate),
        keras.layers.Dense(1)
    ])

    model.compile(optimizer='adam', loss='mse')

    history = model.fit(
        X_train_nn, y_train_nn,
        validation_split=0.2,
        epochs=50,
        batch_size=32,
        verbose=0
    )

    final_train_loss = history.history['loss'][-1]
    final_val_loss = history.history['val_loss'][-1]

    results_dropout.append({
        'Rate': rate,
        'Train Loss': final_train_loss,
        'Val Loss': final_val_loss,
        'Gap': final_train_loss - final_val_loss
    })

df_dropout = pd.DataFrame(results_dropout)
print(df_dropout.to_string(index=False))
```

### Properties of Dropout

**Advantages:**
- ✅ Very effective for deep networks
- ✅ Acts as ensemble of many sub-networks
- ✅ Simple to implement
- ✅ No additional parameters to learn

**Disadvantages:**
- ❌ Increases training time (need more epochs)
- ❌ Only applicable to neural networks

**When to use:**
- Deep neural networks
- Network is overfitting
- Typical dropout rates: 0.2-0.5 for hidden layers, 0.5 for input layer

---

## Early Stopping

**Stop training when validation performance stops improving**

### Implementation

```python
from sklearn.neural_network import MLPRegressor

# Without early stopping
model_no_early = MLPRegressor(
    hidden_layer_sizes=(100, 100),
    max_iter=500,
    early_stopping=False,
    random_state=42
)

model_no_early.fit(X_train_nn, y_train_nn)

# With early stopping
model_early = MLPRegressor(
    hidden_layer_sizes=(100, 100),
    max_iter=500,
    early_stopping=True,
    validation_fraction=0.2,
    n_iter_no_change=10,  # Stop if no improvement for 10 iterations
    random_state=42
)

model_early.fit(X_train_nn, y_train_nn)

print("Without Early Stopping:")
print(f"  Iterations: {model_no_early.n_iter_}")
print(f"  Train Loss: {model_no_early.loss_:.4f}")

print("\nWith Early Stopping:")
print(f"  Iterations: {model_early.n_iter_} (stopped early)")
print(f"  Train Loss: {model_early.loss_:.4f}")
print(f"  Best Validation Score: {model_early.best_validation_score_:.4f}")

# Plot loss curves
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

axes[0].plot(model_no_early.loss_curve_)
axes[0].set_xlabel('Iteration')
axes[0].set_ylabel('Loss')
axes[0].set_title('Without Early Stopping')
axes[0].grid(True, alpha=0.3)

axes[1].plot(model_early.loss_curve_, label='Train')
axes[1].plot(model_early.validation_scores_, label='Validation')
axes[1].axvline(x=len(model_early.loss_curve_), color='r',
               linestyle='--', label='Stopped Here')
axes[1].set_xlabel('Iteration')
axes[1].set_ylabel('Loss/Score')
axes[1].set_title('With Early Stopping')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

### Keras Early Stopping

```python
# More sophisticated early stopping with Keras
early_stopping = keras.callbacks.EarlyStopping(
    monitor='val_loss',      # Metric to monitor
    patience=10,             # Number of epochs to wait
    restore_best_weights=True,  # Restore weights from best epoch
    verbose=1
)

model = keras.Sequential([
    keras.layers.Dense(128, activation='relu', input_shape=(20,)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(1)
])

model.compile(optimizer='adam', loss='mse')

history = model.fit(
    X_train_nn, y_train_nn,
    validation_split=0.2,
    epochs=200,
    batch_size=32,
    callbacks=[early_stopping],
    verbose=0
)

print(f"Training stopped at epoch {len(history.history['loss'])}")

plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.axvline(x=len(history.history['loss']) - early_stopping.patience,
           color='r', linestyle='--', label='Best Model')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Early Stopping')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

### Properties of Early Stopping

**Advantages:**
- ✅ Simple and effective
- ✅ No hyperparameters to tune (besides patience)
- ✅ Saves training time
- ✅ Often as effective as other regularization

**Disadvantages:**
- ❌ Requires validation set
- ❌ Can stop too early with noisy validation loss

**When to use:**
- Always! (for iterative training algorithms)
- Neural networks
- Gradient boosting
- Any model trained iteratively

---

## Data Augmentation

**Create additional training samples by applying transformations**

### Image Data Augmentation

```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Example augmentation pipeline
datagen = ImageDataGenerator(
    rotation_range=20,          # Rotate ±20 degrees
    width_shift_range=0.2,      # Shift horizontally ±20%
    height_shift_range=0.2,     # Shift vertically ±20%
    horizontal_flip=True,       # Random horizontal flip
    zoom_range=0.2,             # Random zoom ±20%
    fill_mode='nearest'         # Fill pixels with nearest value
)

print("Data augmentation techniques:")
print("- Rotation, flips, shifts, zooms")
print("- Increases effective training set size")
print("- Improves model robustness")
```

### Tabular Data Augmentation

```python
def augment_tabular_data(X, y, noise_level=0.05, n_augmentations=3):
    """
    Augment tabular data by adding noise.
    """
    X_aug = [X]
    y_aug = [y]

    for _ in range(n_augmentations):
        # Add Gaussian noise
        X_noisy = X + np.random.randn(*X.shape) * noise_level * X.std(axis=0)
        X_aug.append(X_noisy)
        y_aug.append(y)

    return np.vstack(X_aug), np.concatenate(y_aug)

# Example usage
X_orig = X_train[:100]
y_orig = y_train[:100]

X_aug, y_aug = augment_tabular_data(X_orig, y_orig, noise_level=0.1, n_augmentations=5)

print(f"Original size: {len(X_orig)}")
print(f"Augmented size: {len(X_aug)}")

# Train on augmented data
model_aug = Ridge(alpha=1.0)
model_aug.fit(X_aug, y_aug)

model_orig = Ridge(alpha=1.0)
model_orig.fit(X_orig, y_orig)

print(f"\nTest score (original): {model_orig.score(X_test, y_test):.4f}")
print(f"Test score (augmented): {model_aug.score(X_test, y_test):.4f}")
```

### Properties of Data Augmentation

**Advantages:**
- ✅ Increases effective training set size
- ✅ Improves robustness
- ✅ Domain-specific (can encode prior knowledge)

**Disadvantages:**
- ❌ Requires domain knowledge
- ❌ Can introduce unrealistic samples
- ❌ Increases training time

**When to use:**
- Small training sets
- Image/audio data (very effective)
- When transformations preserve label

---

## Batch Normalization

**Normalizes layer inputs during training**

### Implementation

```python
# Without Batch Normalization
model_no_bn = keras.Sequential([
    keras.layers.Dense(128, input_shape=(20,)),
    keras.layers.Activation('relu'),
    keras.layers.Dense(128),
    keras.layers.Activation('relu'),
    keras.layers.Dense(1)
])

# With Batch Normalization
model_bn = keras.Sequential([
    keras.layers.Dense(128, input_shape=(20,)),
    keras.layers.BatchNormalization(),
    keras.layers.Activation('relu'),
    keras.layers.Dense(128),
    keras.layers.BatchNormalization(),
    keras.layers.Activation('relu'),
    keras.layers.Dense(1)
])

# Train both models
for name, model in [('Without BN', model_no_bn), ('With BN', model_bn)]:
    model.compile(optimizer='adam', loss='mse')
    history = model.fit(
        X_train_nn, y_train_nn,
        validation_split=0.2,
        epochs=100,
        batch_size=32,
        verbose=0
    )

    print(f"{name}:")
    print(f"  Final train loss: {history.history['loss'][-1]:.4f}")
    print(f"  Final val loss: {history.history['val_loss'][-1]:.4f}")
```

### Properties of Batch Normalization

**Advantages:**
- ✅ Accelerates training
- ✅ Reduces sensitivity to initialization
- ✅ Acts as regularization (slight regularizing effect)
- ✅ Allows higher learning rates

**Disadvantages:**
- ❌ Adds complexity
- ❌ Small batch sizes can be problematic
- ❌ Behavior differs between training and inference

**When to use:**
- Deep neural networks
- Struggling with convergence
- Want faster training

---

## Weight Decay

**Gradually decrease weights towards zero** (equivalent to L2 for SGD)

```python
# Weight decay in optimizer
model = keras.Sequential([
    keras.layers.Dense(128, activation='relu', input_shape=(20,)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(1)
])

# Adam with weight decay
optimizer = keras.optimizers.Adam(learning_rate=0.001, weight_decay=0.01)

model.compile(optimizer=optimizer, loss='mse')
```

---

## When to Use Which Technique

### Decision Tree

```
Start here
│
├─ Linear Model?
│  ├─ YES → Want feature selection?
│  │  ├─ YES → Use Lasso (L1)
│  │  └─ NO → Use Ridge (L2)
│  │
│  └─ NO → Neural Network?
│     ├─ YES → Deep?
│     │  ├─ YES → Use Dropout + Batch Norm + Early Stopping
│     │  └─ NO → Use L2 + Early Stopping
│     │
│     └─ NO → Tree-based?
│        ├─ YES → Limit depth, min_samples_split
│        └─ NO → Use appropriate regularization for algorithm
│
└─ Still overfitting?
   ├─ Get more data
   ├─ Use data augmentation
   └─ Use ensemble methods
```

### Quick Reference Table

| Model Type | Primary Regularization | Secondary | Hyperparameter |
|-----------|----------------------|-----------|----------------|
| Linear Regression | Ridge (L2) | Lasso (L1), Elastic Net | α (alpha) |
| Logistic Regression | L2 | L1 | C (inverse of α) |
| Neural Networks | Dropout | Early Stopping, L2 | dropout_rate, α |
| Decision Trees | max_depth | min_samples_split | max_depth |
| Random Forest | max_depth | max_features | max_depth |
| SVM | C parameter | Kernel choice | C |
| Gradient Boosting | learning_rate | max_depth, subsample | learning_rate |

### Choosing Between L1 and L2

**Use Ridge (L2) when:**
- All features potentially relevant
- Features are correlated
- Want smooth coefficient shrinkage

**Use Lasso (L1) when:**
- Many irrelevant features
- Want automatic feature selection
- Need interpretable model (few features)

**Use Elastic Net when:**
- Unsure about L1 vs L2
- Correlated features + want selection
- High-dimensional data

### Combining Multiple Techniques

```python
# Example: Combining multiple regularization techniques

# 1. L2 regularization through optimizer
optimizer = keras.optimizers.Adam(learning_rate=0.001, weight_decay=0.01)

# 2. Dropout
# 3. Batch Normalization
# 4. Early Stopping
early_stopping = keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)

model = keras.Sequential([
    # Input layer
    keras.layers.Dense(256, input_shape=(20,)),
    keras.layers.BatchNormalization(),  # Batch norm
    keras.layers.Activation('relu'),
    keras.layers.Dropout(0.3),  # Dropout

    # Hidden layer
    keras.layers.Dense(128),
    keras.layers.BatchNormalization(),
    keras.layers.Activation('relu'),
    keras.layers.Dropout(0.3),

    # Output layer
    keras.layers.Dense(1)
])

model.compile(optimizer=optimizer, loss='mse')

history = model.fit(
    X_train_nn, y_train_nn,
    validation_split=0.2,
    epochs=200,
    batch_size=32,
    callbacks=[early_stopping],  # Early stopping
    verbose=0
)

print("Model uses:")
print("- L2 regularization (weight decay)")
print("- Dropout (30%)")
print("- Batch Normalization")
print("- Early Stopping")
```

---

## Practical Examples

### Example 1: Finding Optimal Regularization Strength

```python
from sklearn.model_selection import cross_val_score

# Test range of alpha values
alphas = np.logspace(-4, 2, 50)
train_scores = []
cv_scores = []

for alpha in alphas:
    model = Ridge(alpha=alpha)

    # Training score
    model.fit(X_train_poly, y_train)
    train_scores.append(model.score(X_train_poly, y_train))

    # Cross-validation score
    cv_score = cross_val_score(model, X_train_poly, y_train, cv=5, scoring='r2')
    cv_scores.append(cv_score.mean())

# Find optimal alpha
optimal_idx = np.argmax(cv_scores)
optimal_alpha = alphas[optimal_idx]

# Plot
plt.figure(figsize=(12, 6))
plt.plot(alphas, train_scores, 'o-', label='Train Score', linewidth=2)
plt.plot(alphas, cv_scores, 's-', label='CV Score', linewidth=2)
plt.axvline(x=optimal_alpha, color='r', linestyle='--',
           label=f'Optimal α = {optimal_alpha:.4f}')
plt.xscale('log')
plt.xlabel('Alpha (λ)')
plt.ylabel('R² Score')
plt.title('Finding Optimal Regularization Strength')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

print(f"Optimal alpha: {optimal_alpha:.4f}")
print(f"CV Score: {cv_scores[optimal_idx]:.4f}")
```

### Example 2: Comparing All Regularization Methods

```python
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor

# Define models
models = {
    'No Regularization': LinearRegression(),
    'Ridge': Ridge(alpha=1.0),
    'Lasso': Lasso(alpha=0.1, max_iter=10000),
    'Elastic Net': ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=10000),
}

results_comparison = []

for name, model in models.items():
    model.fit(X_train_poly, y_train)

    train_score = model.score(X_train_poly, y_train)
    test_score = model.score(X_test_poly, y_test)

    if hasattr(model, 'coef_'):
        n_nonzero = np.sum(np.abs(model.coef_) > 1e-10)
        max_coef = np.abs(model.coef_).max()
    else:
        n_nonzero = n_features
        max_coef = np.nan

    results_comparison.append({
        'Model': name,
        'Train R²': train_score,
        'Test R²': test_score,
        'Gap': train_score - test_score,
        'Non-zero Coefs': n_nonzero,
        'Max |Coef|': max_coef
    })

df_comparison = pd.DataFrame(results_comparison)
print(df_comparison.to_string(index=False))
```

---

## Best Practices

### 1. Always Start with Validation Curve

```python
from sklearn.model_selection import validation_curve

# Find good range for alpha
param_range = np.logspace(-4, 4, 20)

train_scores, val_scores = validation_curve(
    Ridge(), X_train_poly, y_train,
    param_name='alpha',
    param_range=param_range,
    cv=5,
    scoring='r2'
)

# Plot
plt.figure(figsize=(10, 6))
plt.plot(param_range, train_scores.mean(axis=1), 'o-', label='Train')
plt.plot(param_range, val_scores.mean(axis=1), 's-', label='Validation')
plt.xscale('log')
plt.xlabel('Alpha')
plt.ylabel('R² Score')
plt.title('Validation Curve for Ridge')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

### 2. Use Cross-Validation for Hyperparameter Tuning

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'alpha': np.logspace(-4, 2, 20)
}

grid_search = GridSearchCV(
    Ridge(), param_grid, cv=5, scoring='r2', n_jobs=-1
)

grid_search.fit(X_train_poly, y_train)

print(f"Best alpha: {grid_search.best_params_['alpha']:.4f}")
print(f"Best CV score: {grid_search.best_score_:.4f}")
print(f"Test score: {grid_search.score(X_test_poly, y_test):.4f}")
```

### 3. Monitor Training vs Validation Loss

```python
# Always plot train vs validation curves
def plot_train_val_curves(history):
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Progress')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

    final_gap = history.history['loss'][-1] - history.history['val_loss'][-1]
    if final_gap > 0.1 * history.history['val_loss'][-1]:
        print("⚠️ Possible overfitting detected!")
    else:
        print("✓ Good generalization")
```

### 4. Combine Multiple Techniques

```python
# Best practice: Use multiple regularization techniques together
# Example for neural networks:
# 1. L2 weight decay in optimizer
# 2. Dropout
# 3. Batch normalization
# 4. Early stopping
# 5. Data augmentation (if applicable)
```

### 5. Check Coefficient Magnitudes

```python
def check_coefficient_health(model):
    """
    Check if coefficients are too large (potential overfitting).
    """
    if hasattr(model, 'coef_'):
        max_coef = np.abs(model.coef_).max()
        mean_coef = np.abs(model.coef_).mean()

        print(f"Max |coefficient|: {max_coef:.2e}")
        print(f"Mean |coefficient|: {mean_coef:.2e}")

        if max_coef > 1000:
            print("⚠️ Very large coefficients - consider increasing regularization")
        else:
            print("✓ Coefficient magnitudes are reasonable")
```

---

## Summary

### Key Takeaways

1. **Regularization prevents overfitting** by constraining model complexity
2. **Ridge (L2)** shrinks all coefficients smoothly
3. **Lasso (L1)** performs feature selection by setting coefficients to zero
4. **Elastic Net** combines benefits of L1 and L2
5. **Dropout** is essential for deep neural networks
6. **Early stopping** is simple but very effective
7. **Always use validation curves** to find optimal regularization strength
8. **Combine multiple techniques** for best results

### Quick Reference

| Problem | Solution |
|---------|----------|
| Overfitting in linear models | Ridge, Lasso, or Elastic Net |
| Too many features | Lasso or Elastic Net |
| Correlated features | Ridge or Elastic Net |
| Neural network overfitting | Dropout + Early Stopping + L2 |
| Deep network training | Batch Normalization |
| Small dataset | Data Augmentation |

### Checklist

- [ ] Split data into train/validation/test
- [ ] Try model without regularization (baseline)
- [ ] Plot validation curves for different regularization strengths
- [ ] Use cross-validation to select hyperparameters
- [ ] Check coefficient magnitudes
- [ ] Monitor train vs validation loss
- [ ] Use early stopping for iterative algorithms
- [ ] Combine multiple regularization techniques
- [ ] Evaluate on held-out test set

Proper regularization is essential for building models that generalize well!
