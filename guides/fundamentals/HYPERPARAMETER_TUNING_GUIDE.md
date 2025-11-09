# ⚙️ Hyperparameter Tuning Guide: Optimize Your Models

**The difference between good and great model performance often comes down to hyperparameter tuning.** This guide shows you how to systematically optimize your models.

---

## 📋 Table of Contents

- [Understanding Hyperparameters](#understanding-hyperparameters)
- [Tuning Strategies](#tuning-strategies)
- [Grid Search](#grid-search)
- [Random Search](#random-search)
- [Bayesian Optimization](#bayesian-optimization)
- [Hyperband and ASHA](#hyperband-and-asha)
- [Algorithm-Specific Tuning](#algorithm-specific-tuning)
- [Best Practices](#best-practices)
- [Common Pitfalls](#common-pitfalls)

---

## Understanding Hyperparameters

### Parameters vs Hyperparameters

**Parameters:**
- Learned during training (weights, biases)
- Not set manually
- Example: Neural network weights, linear regression coefficients

**Hyperparameters:**
- Set before training
- Control learning process
- Example: learning rate, number of trees, max depth

### Why Tune Hyperparameters?

**Impact on performance:**
```python
# Default hyperparameters
model = RandomForestClassifier()
model.fit(X_train, y_train)
# Accuracy: 82%

# Tuned hyperparameters
model = RandomForestClassifier(
    n_estimators=500,
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=2
)
model.fit(X_train, y_train)
# Accuracy: 89%  # 8.5% improvement!
```

### Types of Hyperparameters

**Model complexity:**
- Tree depth
- Number of layers
- Number of neurons

**Regularization:**
- L1/L2 penalty (alpha, lambda)
- Dropout rate
- Early stopping patience

**Learning control:**
- Learning rate
- Batch size
- Number of iterations

**Algorithm behavior:**
- Number of neighbors (KNN)
- Kernel type (SVM)
- Distance metric

---

## Tuning Strategies

### 1. Manual Tuning

**When to use:** Initial exploration, small search space

```python
# Try values manually
for max_depth in [3, 5, 10, 15, 20]:
    model = DecisionTreeClassifier(max_depth=max_depth)
    scores = cross_val_score(model, X, y, cv=5)
    print(f"max_depth={max_depth}: {scores.mean():.3f}")

# max_depth=3: 0.812
# max_depth=5: 0.845
# max_depth=10: 0.867
# max_depth=15: 0.871  ← Best
# max_depth=20: 0.865  (overfitting)
```

**Pros:**
- Good for understanding hyperparameter effects
- Fast for small search spaces

**Cons:**
- Time-consuming for many hyperparameters
- Easy to miss optimal combinations

### 2. Default First, Then Tune

**Smart approach:**
```python
# 1. Start with defaults
model = RandomForestClassifier(random_state=42)
baseline_score = cross_val_score(model, X_train, y_train, cv=5).mean()
print(f"Baseline: {baseline_score:.3f}")

# 2. Tune one hyperparameter at a time
# 3. Combine best values
# 4. Fine-tune together
```

---

## Grid Search

### Basic Grid Search

**Exhaustive search over parameter grid**

```python
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

# Define parameter grid
param_grid = {
    'n_estimators': [100, 200, 500],
    'max_depth': [10, 15, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Create grid search
grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,  # Use all CPU cores
    verbose=2
)

# Fit
grid_search.fit(X_train, y_train)

# Best parameters
print(f"Best params: {grid_search.best_params_}")
print(f"Best score: {grid_search.best_score_:.3f}")

# Best model
best_model = grid_search.best_estimator_
```

### Analyze Grid Search Results

```python
import pandas as pd
import matplotlib.pyplot as plt

# Convert results to DataFrame
results = pd.DataFrame(grid_search.cv_results_)

# View top results
print(results[['params', 'mean_test_score', 'std_test_score']]
      .sort_values('mean_test_score', ascending=False)
      .head(10))

# Visualize parameter impact
for param in param_grid.keys():
    if param in results.columns:
        plt.figure(figsize=(10, 4))
        results.boxplot(column='mean_test_score', by=param)
        plt.title(f'Impact of {param}')
        plt.suptitle('')
        plt.show()
```

### Computational Cost

```python
# Grid search tries ALL combinations
param_grid = {
    'n_estimators': [100, 200, 500],      # 3 values
    'max_depth': [10, 15, 20, None],      # 4 values
    'min_samples_split': [2, 5, 10],      # 3 values
    'min_samples_leaf': [1, 2, 4]         # 3 values
}

# Total: 3 × 4 × 3 × 3 = 108 combinations
# With 5-fold CV: 108 × 5 = 540 model trainings!
```

### Reduce Search Space

```python
# Strategy 1: Coarse to fine
# First: Broad search
param_grid_coarse = {
    'n_estimators': [50, 100, 200, 500],
    'max_depth': [5, 10, 15, 20, None]
}

grid_coarse = GridSearchCV(model, param_grid_coarse, cv=3)
grid_coarse.fit(X_train, y_train)
print(f"Best n_estimators: {grid_coarse.best_params_['n_estimators']}")
print(f"Best max_depth: {grid_coarse.best_params_['max_depth']}")

# Then: Fine search around best values
param_grid_fine = {
    'n_estimators': [180, 200, 220],  # Around best value
    'max_depth': [13, 15, 17],        # Around best value
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

grid_fine = GridSearchCV(model, param_grid_fine, cv=5)
grid_fine.fit(X_train, y_train)
```

---

## Random Search

### Why Random Search?

**More efficient than grid search for high dimensions**

**Theory (Bergstra & Bengio, 2012):** Random search explores parameter space more efficiently when hyperparameters have varying importance.

**Mathematical insight:** For n trials over d hyperparameters:
- Grid search: Tests n^(1/d) unique values per parameter
- Random search: Tests n unique values per parameter (expected)
- When only k << d parameters are important, random search provides O(n^(k/d)) vs O(n) coverage of important subspace

**Concrete example:** With 9 trials testing 2 hyperparameters:
- Grid search: 3×3 grid = 3 unique values per parameter
- Random search: 9 unique values per parameter (expected)
- If only 1 parameter is critical, random search samples 9 values vs 3 values from the critical range
- Expected improvement: ~3× better exploration of the important dimension

```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform

# Define distributions (not fixed values!)
param_distributions = {
    'n_estimators': randint(100, 1000),           # Integer uniform
    'max_depth': randint(5, 30),
    'min_samples_split': randint(2, 20),
    'min_samples_leaf': randint(1, 10),
    'max_features': uniform(0.1, 0.9),            # Float uniform
    'bootstrap': [True, False]                     # Categorical
}

# Random search
random_search = RandomizedSearchCV(
    RandomForestClassifier(random_state=42),
    param_distributions,
    n_iter=100,      # Number of random combinations to try
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    random_state=42,
    verbose=2
)

random_search.fit(X_train, y_train)

print(f"Best params: {random_search.best_params_}")
print(f"Best score: {random_search.best_score_:.3f}")
```

### Custom Distributions

```python
from scipy.stats import loguniform

# Log-uniform for learning rate (common in deep learning)
param_distributions = {
    'learning_rate': loguniform(1e-5, 1e-1),  # Search log-scale
    'n_estimators': randint(50, 500),
    'max_depth': randint(3, 20)
}

# Categorical sampling
from sklearn.utils.fixes import loguniform
import numpy as np

param_distributions = {
    'kernel': ['linear', 'rbf', 'poly'],
    'C': loguniform(1e-3, 1e3),
    'gamma': loguniform(1e-4, 1e-1)
}
```

### Grid Search vs Random Search

**When to use Grid Search:**
- Small parameter space (<50 combinations)
- Want to test all combinations
- Parameters are discrete with few values

**When to use Random Search:**
- Large parameter space (>100 combinations)
- Continuous hyperparameters
- Limited time/compute budget
- Want to explore broadly

**Rule of thumb:** Random search with n_iter=100 often beats grid search with same number of trials, especially when:
- Some hyperparameters are much more important than others
- You're tuning 3+ hyperparameters simultaneously
- You're not sure which hyperparameters matter most

**When grid search is better:**
- You have domain knowledge about which discrete values to test
- Very small search space (<50 combinations)
- You want complete coverage of a small region

---

## Bayesian Optimization

### Concept

**Smarter than random:** Uses previous results to guide search

**How it works:**
1. Build probabilistic surrogate model of objective function f(θ)
   - Gaussian Process (GP): Models f(θ) ~ GP(μ, k) with mean μ and kernel k
   - Tree-structured Parzen Estimator (TPE): Models p(θ|y) using p(y|θ)p(θ)
2. Use acquisition function to select next point θ_next
   - Expected Improvement (EI): E[max(f(θ) - f(θ*), 0)]
   - Upper Confidence Bound (UCB): μ(θ) + κσ(θ)
   - Balances exploitation (high mean) vs exploration (high uncertainty)
3. Evaluate f(θ_next) and observe result y_next
4. Update surrogate model with (θ_next, y_next) and repeat

**When Bayesian optimization is worth the overhead:**
- Each trial is expensive (>1 minute to train)
- Limited budget (20-300 trials optimal range)
- Hyperparameters have complex interactions
- Objective function is smooth (Lipschitz continuous or nearly so)
- Signal-to-noise ratio is reasonable (SNR > 0.1)

**When to stick with random search:**
- Fast models (<10 seconds per trial) - surrogate model overhead dominates
- Very noisy objective function (validation score variance >> improvement potential)
- Large budget (>500 trials) - random search achieves similar coverage
- Simple hyperparameter relationships (additive effects, no interactions)
- High-dimensional search (>20 dimensions) - curse of dimensionality affects GP

**Libraries:**
- Optuna (recommended, easy to use)
- Hyperopt
- Scikit-Optimize

### Using Optuna

```python
import optuna
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier

def objective(trial):
    """Objective function to minimize"""

    # Suggest hyperparameters
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 500),
        'max_depth': trial.suggest_int('max_depth', 5, 30),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
        'max_features': trial.suggest_float('max_features', 0.1, 1.0),
        'random_state': 42
    }

    # Train and evaluate
    model = RandomForestClassifier(**params)
    score = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy').mean()

    return score

# Create study
study = optuna.create_study(
    direction='maximize',  # or 'minimize' for loss
    sampler=optuna.samplers.TPESampler(seed=42)
)

# Optimize
study.optimize(objective, n_trials=100, show_progress_bar=True)

# Best parameters
print(f"Best params: {study.best_params}")
print(f"Best score: {study.best_value:.3f}")

# Visualize optimization
optuna.visualization.plot_optimization_history(study)
optuna.visualization.plot_param_importances(study)
```

### Advanced Optuna Features

```python
# Pruning: Stop unpromising trials early
def objective_with_pruning(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 500),
        'max_depth': trial.suggest_int('max_depth', 5, 30),
    }

    model = RandomForestClassifier(**params)

    # Train with intermediate pruning
    for fold, (train_idx, val_idx) in enumerate(kfold.split(X_train)):
        X_tr, X_val = X_train[train_idx], X_train[val_idx]
        y_tr, y_val = y_train[train_idx], y_train[val_idx]

        model.fit(X_tr, y_tr)
        score = model.score(X_val, y_val)

        # Report intermediate value
        trial.report(score, fold)

        # Prune if doing poorly
        if trial.should_prune():
            raise optuna.TrialPruned()

    return score

# Multi-objective optimization
def multi_objective(trial):
    params = {...}

    model = RandomForestClassifier(**params)
    accuracy = cross_val_score(model, X_train, y_train, cv=5).mean()

    # Also optimize for model size
    model.fit(X_train, y_train)
    model_size = model.n_features_in_ * sum(tree.tree_.node_count for tree in model.estimators_)

    return accuracy, -model_size  # Maximize accuracy, minimize size

study = optuna.create_study(directions=['maximize', 'minimize'])
study.optimize(multi_objective, n_trials=100)
```

---

## Hyperband and ASHA

### Successive Halving

**Idea:** Allocate more resources to promising configurations

```python
from sklearn.model_selection import HalvingRandomSearchCV

param_distributions = {
    'n_estimators': randint(50, 500),
    'max_depth': randint(5, 30),
    'min_samples_split': randint(2, 20)
}

halving_search = HalvingRandomSearchCV(
    RandomForestClassifier(random_state=42),
    param_distributions,
    factor=3,           # Eliminate 2/3 of candidates each round
    resource='n_samples',  # or 'n_estimators' for iterative models
    max_resources='auto',
    random_state=42,
    n_jobs=-1
)

halving_search.fit(X_train, y_train)

print(f"Best params: {halving_search.best_params_}")
print(f"Best score: {halving_search.best_score_:.3f}")
```

### How It Works

```
Round 1: n configs × r^0 resource → Keep top n/η
Round 2: n/η configs × r^1 resource → Keep top n/η²
Round 3: n/η² configs × r^2 resource → Keep top 1

Example with n=27, η=3, r=n_samples:
Round 1: 27 configs × 33% data → Keep top 9
Round 2:  9 configs × 66% data → Keep top 3
Round 3:  3 configs × 100% data → Keep top 1

Total cost: 27×(1/3) + 9×(2/3) + 3×1 = 9 + 6 + 3 = 18 "full trainings"
vs full grid: 27×1 = 27 "full trainings" (33% reduction)
```

**Critical assumption:** Early performance strongly correlates with final performance (rank correlation ρ > 0.7). This works well for:
- Increasing n_estimators (more trees generally helps monotonically)
- Neural network training (early learning curves are predictive)
- Increasing training data (more data usually helps)

**May not work well when:**
- Early performance is noisy and uncorrelated with final performance
- Some configs need more data/iterations to show their potential (e.g., high-capacity models that need more data to shine)
- Small performance differences that matter are washed out by noise at small scales

**Recommendation:** Use successive halving when you're tuning iteration-based hyperparameters or have strong correlation between early and final performance. For other cases, random/Bayesian search may be safer.

---

## Algorithm-Specific Tuning

### Random Forest

```python
# Important hyperparameters (priority order)
param_grid = {
    # 1. Number of trees (more is better, up to a point)
    'n_estimators': [100, 200, 500, 1000],

    # 2. Tree depth (controls overfitting)
    'max_depth': [10, 15, 20, 25, None],

    # 3. Minimum samples per split
    'min_samples_split': [2, 5, 10, 20],

    # 4. Minimum samples per leaf
    'min_samples_leaf': [1, 2, 4, 8],

    # 5. Features per split
    'max_features': ['sqrt', 'log2', 0.3, 0.5],

    # 6. Bootstrap samples
    'bootstrap': [True, False]
}

# Start with defaults, tune n_estimators and max_depth first
```

### XGBoost

```python
# Important hyperparameters
param_grid = {
    # Learning rate (most important!)
    'learning_rate': [0.01, 0.05, 0.1, 0.3],

    # Number of boosting rounds
    'n_estimators': [100, 200, 500, 1000],

    # Tree depth
    'max_depth': [3, 5, 7, 9],

    # L2 regularization
    'reg_lambda': [0, 0.1, 1, 10],

    # L1 regularization
    'reg_alpha': [0, 0.1, 1, 10],

    # Subsample ratio
    'subsample': [0.6, 0.8, 1.0],

    # Feature subsample ratio
    'colsample_bytree': [0.6, 0.8, 1.0],

    # Minimum loss reduction
    'gamma': [0, 0.1, 0.5, 1]
}

# Common starting point
best_params = {
    'learning_rate': 0.1,
    'n_estimators': 100,  # Tune with early stopping
    'max_depth': 5,
    'subsample': 0.8,
    'colsample_bytree': 0.8
}
```

**Early stopping for XGBoost:**
```python
import xgboost as xgb

model = xgb.XGBClassifier(
    learning_rate=0.1,
    n_estimators=1000,  # Set high
    eval_metric='logloss',  # Must match your scoring objective!
    random_state=42
)

# Use early stopping to find optimal n_estimators
model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    early_stopping_rounds=50,  # Stop if no improvement for 50 rounds
    verbose=False
)

print(f"Best iteration: {model.best_iteration}")

# Important: eval_metric should align with your goal
# Classification: 'logloss', 'error', 'auc'
# Regression: 'rmse', 'mae', 'mape'
# Custom: Define your own metric function
```

### Neural Networks

```python
# Important hyperparameters
param_grid = {
    # Architecture
    'hidden_layer_sizes': [(50,), (100,), (100, 50), (200, 100, 50)],

    # Activation function
    'activation': ['relu', 'tanh'],

    # Learning rate (most important!)
    'learning_rate_init': [0.0001, 0.001, 0.01],

    # Regularization
    'alpha': [0.0001, 0.001, 0.01],

    # Batch size
    'batch_size': [32, 64, 128, 256],

    # Optimizer
    'solver': ['adam', 'sgd'],

    # Max iterations
    'max_iter': [200, 500, 1000]
}

# Use early stopping
model = MLPClassifier(
    hidden_layer_sizes=(100, 50),
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=10
)
```

### SVM

```python
# Important hyperparameters
param_grid = {
    # Regularization parameter
    'C': [0.001, 0.01, 0.1, 1, 10, 100],

    # Kernel type
    'kernel': ['linear', 'rbf', 'poly'],

    # Kernel coefficient (for rbf, poly)
    'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1]
}

# For RBF kernel specifically
param_grid_rbf = {
    'C': np.logspace(-3, 3, 7),      # [0.001, 0.01, ..., 1000]
    'gamma': np.logspace(-4, 1, 6)   # [0.0001, 0.001, ..., 10]
}
```

---

## Best Practices

### 1. Use Cross-Validation

```python
# ✅ Good: Use CV to avoid overfitting to validation set
grid_search = GridSearchCV(
    model,
    param_grid,
    cv=5,  # 5-fold cross-validation
    scoring='accuracy'
)

# ❌ Bad: Using single train/val split
# Easy to overfit to validation set during tuning
```

**Important:** Even with CV, extensive hyperparameter tuning introduces optimistic bias. The bias grows with the number of hyperparameter configurations tested.

**Expected bias:** For m configurations tested with CV validation score s and true variance σ²:
- Expected optimistic bias ≈ σ × √(2 log m)
- Example: Testing m=100 configs with σ=0.02 → bias ≈ 0.02 × √(2 log 100) ≈ 0.06
- This means your CV score may be ~6% higher than true generalization performance

**Nested CV eliminates this bias:**

```python
from sklearn.model_selection import cross_val_score

# Nested CV: Outer loop for unbiased evaluation, inner loop for tuning
outer_cv = 5
inner_cv = 3

grid_search = GridSearchCV(model, param_grid, cv=inner_cv, scoring='accuracy')

# Get unbiased estimate of performance
nested_scores = cross_val_score(grid_search, X_train, y_train, cv=outer_cv)
print(f"Nested CV score: {nested_scores.mean():.3f} ± {nested_scores.std():.3f}")

# After getting unbiased estimate, retrain on all data for deployment
grid_search.fit(X_train, y_train)
final_model = grid_search.best_estimator_
```

**When to use nested CV:**
- Comparing multiple tuning strategies (requires fair comparison)
- Reporting performance in publication/research (unbiased estimates required)
- Extensive hyperparameter search (>50 combinations)
- Want to quantify true expected performance, not selection-biased estimate

### 2. Tune on Training Set, Evaluate on Test Set

```python
# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Tune hyperparameters (uses CV internally)
grid_search = GridSearchCV(model, param_grid, cv=5)
grid_search.fit(X_train, y_train)  # Only use X_train!

# Evaluate on test set (ONCE at the end)
test_score = grid_search.score(X_test, y_test)
```

### 3. Start Simple

```python
# 1. Baseline with defaults
baseline = RandomForestClassifier(random_state=42)
baseline_score = cross_val_score(baseline, X_train, y_train, cv=5).mean()

# 2. Tune most important hyperparameter first
param_grid = {'n_estimators': [50, 100, 200, 500]}
grid_search = GridSearchCV(baseline, param_grid, cv=5)
grid_search.fit(X_train, y_train)

# 3. Add next most important hyperparameter
# Continue iteratively
```

### 4. Log Scale for Learning Rate

```python
# Learning rates often work on log scale
# Try: 0.001, 0.01, 0.1, 1.0 (not 0.1, 0.2, 0.3, 0.4)

from scipy.stats import loguniform

param_distributions = {
    'learning_rate': loguniform(1e-4, 1e0)  # 0.0001 to 1.0
}
```

### 5. Use Warm Start for Iterative Models

```python
# For models that support warm_start
model = RandomForestClassifier(warm_start=True, random_state=42)

# Incrementally add trees
for n_estimators in [50, 100, 200, 500]:
    model.set_params(n_estimators=n_estimators)
    model.fit(X_train, y_train)
    score = model.score(X_val, y_val)
    print(f"n_estimators={n_estimators}: {score:.3f}")
```

### 6. Monitor Validation Curves

```python
from sklearn.model_selection import validation_curve

# Study effect of single hyperparameter
param_range = [10, 20, 50, 100, 200, 500]
train_scores, val_scores = validation_curve(
    RandomForestClassifier(),
    X_train, y_train,
    param_name='n_estimators',
    param_range=param_range,
    cv=5
)

# Plot
plt.figure(figsize=(10, 6))
plt.plot(param_range, train_scores.mean(axis=1), label='Training')
plt.plot(param_range, val_scores.mean(axis=1), label='Validation')
plt.xlabel('n_estimators')
plt.ylabel('Score')
plt.legend()
plt.show()
```

### 7. Save Best Model

```python
import joblib

# After tuning
grid_search.fit(X_train, y_train)

# Save best model
best_model = grid_search.best_estimator_
joblib.dump(best_model, 'best_model.pkl')

# Save search results
results = pd.DataFrame(grid_search.cv_results_)
results.to_csv('tuning_results.csv', index=False)
```

---

## Common Pitfalls

### 1. Overfitting to Validation Set

```python
# ❌ Bad: Trying hundreds of combinations without CV
for params in all_possible_combinations:  # 1000+ combinations
    model.set_params(**params)
    model.fit(X_train, y_train)
    score = model.score(X_val, y_val)
    # After 1000 tries, you'll find one that works well by chance!

# ✅ Good: Use cross-validation
grid_search = GridSearchCV(model, param_grid, cv=5)
# CV protects against overfitting to single validation set
```

**Why this is a problem (multiple comparisons / selection bias):**
- Each hyperparameter trial is an independent draw from a noisy estimator
- Maximum of m draws from N(μ, σ²) has expected value μ + σ × √(2 log m)
- This is the ML equivalent of p-hacking / multiple hypothesis testing in statistics
- The best validation score is biased upward by O(σ√log m)

**Quantitative example:**
- True accuracy μ = 0.85, validation noise σ = 0.02 (typical for n_val ≈ 1000)
- Test 100 configurations: expected max ≈ 0.85 + 0.02 × √(2 log 100) ≈ 0.85 + 0.06 = 0.91
- Test 1000 configurations: expected max ≈ 0.85 + 0.02 × √(2 log 1000) ≈ 0.85 + 0.075 = 0.925
- This 4-7.5% optimistic bias is pure selection bias, not generalization improvement

**Solutions:**
- Use cross-validation (helps but doesn't eliminate the problem)
- Use nested CV for unbiased estimates
- Hold out a separate test set that you ONLY evaluate once
- Limit the number of hyperparameter trials if using a single validation set

### 2. Tuning on Test Set

```python
# ❌ WRONG: Using test set for tuning
grid_search.fit(X_test, y_test)  # NO!

# ✅ CORRECT: Tune on train, evaluate once on test
grid_search.fit(X_train, y_train)
final_score = grid_search.score(X_test, y_test)
```

### 3. Not Accounting for Class Imbalance

```python
# For imbalanced data, use appropriate scoring metric

# ❌ Bad: Using accuracy
grid_search = GridSearchCV(model, param_grid, scoring='accuracy')
# May optimize for majority class!

# ✅ Good: Use F1, precision, recall, or ROC-AUC
grid_search = GridSearchCV(model, param_grid, scoring='f1')
# or
grid_search = GridSearchCV(model, param_grid, scoring='roc_auc')
```

### 4. Too Fine Grid Initially

```python
# ❌ Bad: Very fine grid from start
param_grid = {
    'C': [0.01, 0.02, 0.03, ..., 0.99, 1.0],  # 100 values!
    'gamma': [0.001, 0.002, 0.003, ..., 0.1]   # 100 values!
}
# 100 × 100 = 10,000 combinations (too many!)

# ✅ Good: Coarse to fine
# First: Broad search
param_grid_coarse = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100],
    'gamma': [0.001, 0.01, 0.1, 1]
}
# Then: Fine search around best values
```

### 5. Ignoring Computation Time

```python
# Balance performance vs time

# For quick experiments
grid_search = GridSearchCV(model, param_grid, cv=3, n_jobs=-1)

# For final tuning
grid_search = GridSearchCV(model, param_grid, cv=10, n_jobs=-1)

# Monitor time
import time
start = time.time()
grid_search.fit(X_train, y_train)
print(f"Tuning took {time.time() - start:.1f} seconds")
```

---

## Complete Example

```python
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import optuna

# Generate data
X, y = make_classification(n_samples=1000, n_features=20, random_state=42)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 1. Baseline
baseline = RandomForestClassifier(random_state=42)
baseline.fit(X_train, y_train)
baseline_score = baseline.score(X_test, y_test)
print(f"Baseline: {baseline_score:.3f}")

# 2. Grid Search (comprehensive)
param_grid = {
    'n_estimators': [100, 200, 500],
    'max_depth': [10, 15, 20, None],
    'min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=5,
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train, y_train)
grid_score = grid_search.score(X_test, y_test)
print(f"Grid Search: {grid_score:.3f}")
print(f"Best params: {grid_search.best_params_}")

# 3. Random Search (faster exploration)
from scipy.stats import randint

param_distributions = {
    'n_estimators': randint(50, 500),
    'max_depth': randint(5, 30),
    'min_samples_split': randint(2, 20),
    'min_samples_leaf': randint(1, 10)
}

random_search = RandomizedSearchCV(
    RandomForestClassifier(random_state=42),
    param_distributions,
    n_iter=50,
    cv=5,
    random_state=42,
    n_jobs=-1
)

random_search.fit(X_train, y_train)
random_score = random_search.score(X_test, y_test)
print(f"Random Search: {random_score:.3f}")
print(f"Best params: {random_search.best_params_}")

# 4. Bayesian Optimization (smartest)
def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 500),
        'max_depth': trial.suggest_int('max_depth', 5, 30),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
        'random_state': 42
    }

    # Use cross-validation, NOT test set!
    model = RandomForestClassifier(**params)
    score = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy').mean()
    return score

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50, show_progress_bar=True)

# Train best model and evaluate on test set
best_params = study.best_params
best_params['random_state'] = 42
best_model = RandomForestClassifier(**best_params)
best_model.fit(X_train, y_train)
optuna_score = best_model.score(X_test, y_test)

print(f"Optuna CV score: {study.best_value:.3f}")
print(f"Optuna test score: {optuna_score:.3f}")
print(f"Best params: {study.best_params}")

# Summary
print("\n=== Summary ===")
print(f"Baseline:      {baseline_score:.3f}")
print(f"Grid Search:   {grid_score:.3f}")
print(f"Random Search: {random_score:.3f}")
print(f"Optuna:        {optuna_score:.3f}")
print("\nNote: All methods used CV for tuning, then evaluated once on held-out test set")
```

---

## Key Takeaways

**Tuning strategy:**
1. Start with defaults (baseline)
2. Tune most important hyperparameters first
3. Use coarse-to-fine approach
4. Validate with cross-validation
5. Evaluate once on test set

**Method selection:**
- **Small search space** → Grid Search
- **Large search space** → Random Search
- **Limited budget** → Bayesian Optimization (Optuna)
- **Very large models** → Successive Halving

**Remember:**
- More data > better hyperparameters
- Feature engineering > hyperparameter tuning
- Good defaults work surprisingly well
- Don't overfit to validation set
- Document your search process

---

**Related guides:**
- [Model Selection Guide](./MODEL_SELECTION_GUIDE.md) - Choose the right algorithm
- [Cross-Validation Guide](./CROSS_VALIDATION_GUIDE.md) - Proper model evaluation
- [Common ML Mistakes](../../resources/COMMON_ML_MISTAKES.md) - Avoid tuning pitfalls

---

*Last Updated: October 30, 2025*
