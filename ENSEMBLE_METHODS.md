# Ensemble Methods Guide

Learn how to combine multiple models to achieve superior performance.

## Table of Contents
1. [Introduction to Ensembles](#introduction-to-ensembles)
2. [Bagging Methods](#bagging-methods)
3. [Boosting Methods](#boosting-methods)
4. [Stacking and Blending](#stacking-and-blending)
5. [Voting Ensembles](#voting-ensembles)
6. [Neural Network Ensembles](#neural-network-ensembles)
7. [Advanced Techniques](#advanced-techniques)
8. [Best Practices](#best-practices)

---

## Introduction to Ensembles

### Why Ensemble?

**Core Principle:** Combine multiple weak learners to create a strong learner.

**Theoretical Foundation:**

**Bias-Variance Decomposition:**
For regression, expected prediction error can be decomposed:
```
E[(y - f̂(x))²] = Bias²[f̂(x)] + Var[f̂(x)] + σ²

Where:
- Bias²[f̂(x)] = (E[f̂(x)] - f(x))²: How far off is the average prediction?
- Var[f̂(x)] = E[(f̂(x) - E[f̂(x)])²]: How much do predictions vary?
- σ²: Irreducible error (noise in data)
```

**Ensemble Effect on Bias-Variance:**
- **Bagging**: Reduces variance, keeps bias constant
  - Var[avg of M models] ≈ Var[single model] / M (if uncorrelated)
- **Boosting**: Reduces both bias and variance
  - Sequential focus on errors reduces bias
  - Regularization controls variance

**Statistical Learning Theory:**
For an ensemble of M classifiers with error rate ε < 0.5 (better than random):
```
P(ensemble error) = Σ(k=⌈M/2⌉ to M) C(M,k) × ε^k × (1-ε)^(M-k)

This decreases exponentially with M if ε < 0.5!
```

**Diversity-Accuracy Tradeoff:**
Ensemble error bounded by:
```
E_ensemble ≤ Ē - λ·D̄

Where:
- Ē: Average individual model error
- D̄: Average pairwise diversity
- λ: Weight depending on problem
```

```python
# Single model performance
single_model_accuracy = 0.85

# Ensemble of 5 models
ensemble_accuracy = 0.92  # +7% improvement!

# Theoretical bound (assuming independent errors)
import math
def ensemble_bound(individual_accuracy, n_models):
    """Upper bound on ensemble error"""
    p_error = 1 - individual_accuracy
    # Probability that majority is wrong
    bound = sum(math.comb(n_models, k) * p_error**k * (1-p_error)**(n_models-k)
                for k in range(n_models//2 + 1, n_models + 1))
    return 1 - bound

print(f"Theoretical bound: {ensemble_bound(0.85, 5):.3f}")
```

**When Ensembles Work Best:**
- Models make different types of errors (diversity)
- Each model performs better than random (ε < 0.5)
- Models are not perfectly correlated
- Sufficient training data for multiple models

---

## Bagging Methods

### Bootstrap Aggregating (Bagging)

**Strategy:** Train multiple models on different random subsets of data (with replacement).

```python
import numpy as np
from sklearn.tree import DecisionTreeClassifier

class BaggingEnsemble:
    """Custom bagging implementation"""

    def __init__(self, base_model, n_estimators=10, max_samples=1.0):
        self.base_model = base_model
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.models = []

    def fit(self, X, y):
        n_samples = int(len(X) * self.max_samples)

        for i in range(self.n_estimators):
            # Bootstrap sample
            indices = np.random.choice(len(X), size=n_samples, replace=True)
            X_bootstrap = X[indices]
            y_bootstrap = y[indices]

            # Train model
            model = self._clone_model(self.base_model)
            model.fit(X_bootstrap, y_bootstrap)
            self.models.append(model)

    def predict(self, X):
        # Aggregate predictions (majority vote)
        predictions = np.array([model.predict(X) for model in self.models])
        # Take mode along axis 0
        return np.apply_along_axis(
            lambda x: np.bincount(x).argmax(),
            axis=0,
            arr=predictions
        )

    def predict_proba(self, X):
        # Average probabilities
        probas = np.array([model.predict_proba(X) for model in self.models])
        return probas.mean(axis=0)

    def _clone_model(self, model):
        return type(model)(**model.get_params())

# Usage
base_model = DecisionTreeClassifier(max_depth=10)
bagging = BaggingEnsemble(base_model, n_estimators=50)
bagging.fit(X_train, y_train)
predictions = bagging.predict(X_test)
```

**Using scikit-learn:**
```python
from sklearn.ensemble import BaggingClassifier

bagging = BaggingClassifier(
    base_estimator=DecisionTreeClassifier(max_depth=10),
    n_estimators=50,
    max_samples=0.8,  # Use 80% of data for each bag
    max_features=0.8,  # Use 80% of features
    bootstrap=True,
    n_jobs=-1
)

bagging.fit(X_train, y_train)
accuracy = bagging.score(X_test, y_test)
```

---

### Random Forests

**Key Innovation:** Bagging + random feature selection at each split.

**Algorithm:**
```
Random Forest Training:
1. For b = 1 to B (number of trees):
   a) Draw bootstrap sample Z*_b of size n from training data
   b) Grow tree T_b on Z*_b with modification:
      - At each node, randomly select m features from p total features
      - Choose best split among these m features only
      - Split node using best feature/threshold
      - Repeat until min_samples_leaf reached
2. Output ensemble: {T_b}^B_b=1

Prediction:
- Classification: Majority vote
  ĈRF(x) = majority vote of {T_b(x)}^B_b=1

- Regression: Average
  f̂_RF(x) = (1/B) Σ^B_b=1 T_b(x)

Key Parameter:
- m = √p for classification (default)
- m = p/3 for regression (default)
- Smaller m → more diversity, less correlation
```

**Theoretical Properties:**

**Out-of-Bag (OOB) Error Estimate:**
```
For each observation (x_i, y_i):
- ~63.2% of bootstrap samples include it (1 - (1-1/n)^n ≈ 1 - e^(-1))
- ~36.8% don't include it (OOB samples)

OOB Prediction for x_i:
ŷ_i^OOB = aggregate predictions from trees not trained on (x_i, y_i)

OOB Error:
OOB_err = (1/n) Σ^n_i=1 L(y_i, ŷ_i^OOB)

This provides unbiased estimate of test error without separate validation set!
```

**Computational Complexity:**
```
Training: O(B · n · p · log n)
- B: Number of trees
- n: Number of samples
- p: Number of features
- log n: Tree depth (balanced tree)

Prediction: O(B · log n)
- Evaluate B trees, each O(log n) depth

Space: O(B · n_nodes · p)
- Store B trees with average n_nodes nodes per tree
```

```python
from sklearn.ensemble import RandomForestClassifier

# Random Forest
rf = RandomForestClassifier(
    n_estimators=100,  # B = 100 trees
    max_depth=20,  # Maximum tree depth
    min_samples_split=5,  # Minimum samples to split node
    min_samples_leaf=2,  # Minimum samples in leaf
    max_features='sqrt',  # m = sqrt(p) for each split
    bootstrap=True,  # Use bootstrap sampling
    oob_score=True,  # Compute OOB error estimate
    n_jobs=-1,  # Parallel training
    random_state=42
)

rf.fit(X_train, y_train)

# Feature importance
importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]

print("Top 10 features:")
for i in range(10):
    print(f"{i+1}. {feature_names[indices[i]]}: {importances[indices[i]]:.4f}")

# Predictions
y_pred = rf.predict(X_test)
y_proba = rf.predict_proba(X_test)
```

**Tuning Random Forest:**
```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [100, 200, 500],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2', None],
}

grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)

grid_search.fit(X_train, y_train)
print(f"Best params: {grid_search.best_params_}")
```

---

### Extra Trees

**Key Innovation:** Fully random splits (faster than Random Forest).

```python
from sklearn.ensemble import ExtraTreesClassifier

extra_trees = ExtraTreesClassifier(
    n_estimators=100,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    n_jobs=-1,
    random_state=42
)

extra_trees.fit(X_train, y_train)
predictions = extra_trees.predict(X_test)
```

**Random Forest vs Extra Trees:**

| Feature | Random Forest | Extra Trees |
|---------|--------------|-------------|
| **Split Selection** | Best threshold | Random threshold |
| **Speed** | Slower | Faster |
| **Variance** | Lower | Higher |
| **Overfitting** | Less prone | More prone |

---

## Boosting Methods

### AdaBoost

**Strategy:** Sequentially train models, giving more weight to misclassified samples.

```python
from sklearn.ensemble import AdaBoostClassifier

adaboost = AdaBoostClassifier(
    base_estimator=DecisionTreeClassifier(max_depth=1),  # Weak learner (stump)
    n_estimators=50,
    learning_rate=1.0,
    algorithm='SAMME.R',
    random_state=42
)

adaboost.fit(X_train, y_train)
predictions = adaboost.predict(X_test)

# Feature importance
importances = adaboost.feature_importances_
```

**Custom AdaBoost Implementation:**
```python
class SimpleAdaBoost:
    """AdaBoost for binary classification"""

    def __init__(self, n_estimators=50):
        self.n_estimators = n_estimators
        self.models = []
        self.alphas = []

    def fit(self, X, y):
        n_samples = len(X)
        # Initialize weights uniformly
        weights = np.ones(n_samples) / n_samples

        for _ in range(self.n_estimators):
            # Train weak learner
            model = DecisionTreeClassifier(max_depth=1)
            model.fit(X, y, sample_weight=weights)

            # Predictions
            predictions = model.predict(X)

            # Calculate error
            error = np.sum(weights * (predictions != y)) / np.sum(weights)

            # Calculate alpha (model weight)
            alpha = 0.5 * np.log((1 - error) / (error + 1e-10))

            # Update sample weights
            weights *= np.exp(-alpha * y * predictions)
            weights /= np.sum(weights)  # Normalize

            self.models.append(model)
            self.alphas.append(alpha)

    def predict(self, X):
        # Weighted majority vote
        predictions = np.array([alpha * model.predict(X)
                               for alpha, model in zip(self.alphas, self.models)])
        return np.sign(predictions.sum(axis=0))
```

---

### Gradient Boosting

**Strategy:** Sequentially fit models to residuals (errors) of previous models.

**Mathematical Framework:**
```
Goal: Minimize loss L(y, F(x))

Gradient Boosting Algorithm:
1. Initialize: F_0(x) = argmin_γ Σ L(y_i, γ)
2. For m = 1 to M:
   a) Compute pseudo-residuals: r_im = -[∂L(y_i, F(x_i))/∂F(x_i)]_{F=F_{m-1}}
   b) Fit base learner h_m(x) to pseudo-residuals r_im
   c) Find step size: γ_m = argmin_γ Σ L(y_i, F_{m-1}(x_i) + γ·h_m(x_i))
   d) Update: F_m(x) = F_{m-1}(x) + γ_m·h_m(x)

Final model: F_M(x) = F_0(x) + Σ γ_m·h_m(x)

Key Insight: Gradient boosting performs gradient descent in function space!
```

**Common Loss Functions:**
- Squared loss (regression): L(y,F) = (y - F)²/2
  - Pseudo-residual: r = y - F(x) (simple residual)
- Logistic loss (classification): L(y,F) = log(1 + exp(-y·F))
  - Pseudo-residual: r = y/(1 + exp(y·F))
- Exponential loss (AdaBoost): L(y,F) = exp(-y·F)

```python
from sklearn.ensemble import GradientBoostingClassifier

gb = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    min_samples_split=5,
    min_samples_leaf=2,
    subsample=0.8,
    max_features='sqrt',
    random_state=42
)

gb.fit(X_train, y_train)
predictions = gb.predict(X_test)

# Learning curve
train_scores = []
val_scores = []

for i, (train_pred, val_pred) in enumerate(zip(
    gb.staged_predict(X_train),
    gb.staged_predict(X_test)
)):
    train_scores.append(accuracy_score(y_train, train_pred))
    val_scores.append(accuracy_score(y_test, val_pred))

# Plot learning curve to find optimal n_estimators
```

---

### XGBoost

**Best for:** Tabular data, competitions, production systems.

```python
import xgboost as xgb

# XGBoost Classifier
xgb_model = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,  # L1 regularization
    reg_lambda=1.0,  # L2 regularization
    gamma=0,  # Min loss reduction for split
    min_child_weight=1,
    objective='binary:logistic',
    eval_metric='logloss',
    use_label_encoder=False,
    random_state=42
)

# Training with early stopping
eval_set = [(X_train, y_train), (X_test, y_test)]

xgb_model.fit(
    X_train, y_train,
    eval_set=eval_set,
    eval_metric='logloss',
    early_stopping_rounds=10,
    verbose=True
)

# Predictions
y_pred = xgb_model.predict(X_test)
y_proba = xgb_model.predict_proba(X_test)

# Feature importance
xgb.plot_importance(xgb_model, importance_type='gain', max_num_features=20)
```

**XGBoost with Custom Objective:**
```python
def custom_logloss(preds, dtrain):
    """Custom log loss with weights"""
    labels = dtrain.get_label()
    weights = dtrain.get_weight()

    # Gradient
    grad = weights * (preds - labels)

    # Hessian
    hess = weights * preds * (1 - preds)

    return grad, hess

# DMatrix format
dtrain = xgb.DMatrix(X_train, label=y_train, weight=sample_weights)
dtest = xgb.DMatrix(X_test, label=y_test)

params = {
    'max_depth': 6,
    'eta': 0.1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
}

# Train with custom objective
bst = xgb.train(
    params,
    dtrain,
    num_boost_round=100,
    obj=custom_logloss,
    evals=[(dtrain, 'train'), (dtest, 'test')],
    early_stopping_rounds=10
)
```

---

### LightGBM

**Best for:** Large datasets, speed, memory efficiency.

```python
import lightgbm as lgb

# LightGBM Classifier
lgb_model = lgb.LGBMClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    num_leaves=31,
    min_child_samples=20,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=42,
    n_jobs=-1
)

# Training
lgb_model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    eval_metric='logloss',
    early_stopping_rounds=10,
    verbose=100
)

# Predictions
y_pred = lgb_model.predict(X_test)

# Feature importance
lgb.plot_importance(lgb_model, importance_type='gain', max_num_features=20)
```

**LightGBM with Categorical Features:**
```python
# Mark categorical features
categorical_features = ['category', 'brand', 'store_id']

lgb_model = lgb.LGBMClassifier(
    n_estimators=100,
    max_depth=-1,  # No limit
    learning_rate=0.05,
    num_leaves=31,
    random_state=42
)

lgb_model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    categorical_feature=categorical_features,
    early_stopping_rounds=10
)
```

---

### CatBoost

**Best for:** Categorical features, reducing overfitting.

```python
from catboost import CatBoostClassifier

# CatBoost
cat_model = CatBoostClassifier(
    iterations=100,
    learning_rate=0.1,
    depth=6,
    l2_leaf_reg=3,
    random_seed=42,
    verbose=100
)

# Automatic handling of categorical features
cat_features = ['category', 'brand', 'store_id']

cat_model.fit(
    X_train, y_train,
    cat_features=cat_features,
    eval_set=(X_test, y_test),
    early_stopping_rounds=10,
    plot=True  # Interactive plot
)

# Predictions
y_pred = cat_model.predict(X_test)
y_proba = cat_model.predict_proba(X_test)

# Feature importance
cat_model.get_feature_importance(prettified=True)
```

**CatBoost for Text Features:**
```python
from catboost import CatBoostClassifier, Pool

# Pool with text features
train_pool = Pool(
    data=X_train,
    label=y_train,
    cat_features=cat_features,
    text_features=['description', 'title']  # Automatic text processing
)

test_pool = Pool(
    data=X_test,
    label=y_test,
    cat_features=cat_features,
    text_features=['description', 'title']
)

cat_model = CatBoostClassifier(iterations=100, random_seed=42)
cat_model.fit(train_pool, eval_set=test_pool)
```

---

## Stacking and Blending

### Stacking (Stacked Generalization)

**Strategy:** Use predictions from multiple models as features for a meta-learner.

```python
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC

# Base models
base_models = [
    ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
    ('gb', GradientBoostingClassifier(n_estimators=100, random_state=42)),
    ('svc', SVC(probability=True, random_state=42))
]

# Meta-learner
meta_learner = LogisticRegression()

# Stacking ensemble
stacking = StackingClassifier(
    estimators=base_models,
    final_estimator=meta_learner,
    cv=5,  # Cross-validation for meta-features
    stack_method='predict_proba',  # Use probabilities
    n_jobs=-1
)

stacking.fit(X_train, y_train)
predictions = stacking.predict(X_test)
probabilities = stacking.predict_proba(X_test)
```

**Custom Stacking Implementation:**
```python
class StackingEnsemble:
    """Custom stacking with out-of-fold predictions"""

    def __init__(self, base_models, meta_model, n_folds=5):
        self.base_models = base_models
        self.meta_model = meta_model
        self.n_folds = n_folds

    def fit(self, X, y):
        # Step 1: Generate out-of-fold predictions
        n_samples = X.shape[0]
        n_models = len(self.base_models)
        meta_features = np.zeros((n_samples, n_models))

        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=42)

        for i, model in enumerate(self.base_models):
            for train_idx, val_idx in kf.split(X):
                X_train_fold, X_val_fold = X[train_idx], X[val_idx]
                y_train_fold = y[train_idx]

                # Train on fold
                model_clone = clone(model)
                model_clone.fit(X_train_fold, y_train_fold)

                # Predict on validation fold
                meta_features[val_idx, i] = model_clone.predict_proba(X_val_fold)[:, 1]

        # Step 2: Train base models on full training set
        self.trained_base_models = []
        for model in self.base_models:
            model_clone = clone(model)
            model_clone.fit(X, y)
            self.trained_base_models.append(model_clone)

        # Step 3: Train meta-model on out-of-fold predictions
        self.meta_model.fit(meta_features, y)

    def predict(self, X):
        # Get predictions from base models
        meta_features = np.column_stack([
            model.predict_proba(X)[:, 1]
            for model in self.trained_base_models
        ])

        # Meta-model prediction
        return self.meta_model.predict(meta_features)

    def predict_proba(self, X):
        meta_features = np.column_stack([
            model.predict_proba(X)[:, 1]
            for model in self.trained_base_models
        ])
        return self.meta_model.predict_proba(meta_features)

# Usage
base_models = [
    RandomForestClassifier(n_estimators=100),
    GradientBoostingClassifier(n_estimators=100),
    xgb.XGBClassifier(n_estimators=100)
]

meta_model = LogisticRegression()

stacking = StackingEnsemble(base_models, meta_model, n_folds=5)
stacking.fit(X_train, y_train)
predictions = stacking.predict(X_test)
```

---

### Blending

**Strategy:** Like stacking, but uses a holdout set instead of cross-validation.

```python
from sklearn.model_selection import train_test_split

class BlendingEnsemble:
    """Blending ensemble"""

    def __init__(self, base_models, meta_model, blend_size=0.2):
        self.base_models = base_models
        self.meta_model = meta_model
        self.blend_size = blend_size

    def fit(self, X, y):
        # Split into training and blending sets
        X_train, X_blend, y_train, y_blend = train_test_split(
            X, y, test_size=self.blend_size, random_state=42
        )

        # Train base models on training set
        self.trained_base_models = []
        blend_features = []

        for model in self.base_models:
            # Train
            model.fit(X_train, y_train)
            self.trained_base_models.append(model)

            # Predict on blend set
            blend_preds = model.predict_proba(X_blend)[:, 1]
            blend_features.append(blend_preds)

        # Stack blend predictions
        blend_features = np.column_stack(blend_features)

        # Train meta-model on blend set
        self.meta_model.fit(blend_features, y_blend)

    def predict(self, X):
        # Get predictions from base models
        meta_features = np.column_stack([
            model.predict_proba(X)[:, 1]
            for model in self.trained_base_models
        ])

        # Meta-model prediction
        return self.meta_model.predict(meta_features)
```

---

## Voting Ensembles

### Hard Voting

**Strategy:** Majority vote from multiple classifiers.

```python
from sklearn.ensemble import VotingClassifier

# Define models
model1 = RandomForestClassifier(n_estimators=100, random_state=42)
model2 = GradientBoostingClassifier(n_estimators=100, random_state=42)
model3 = SVC(kernel='rbf', random_state=42)

# Voting ensemble (hard voting)
voting_hard = VotingClassifier(
    estimators=[('rf', model1), ('gb', model2), ('svc', model3)],
    voting='hard'  # Majority vote
)

voting_hard.fit(X_train, y_train)
predictions = voting_hard.predict(X_test)
```

### Soft Voting

**Strategy:** Average predicted probabilities.

```python
# Voting ensemble (soft voting)
voting_soft = VotingClassifier(
    estimators=[('rf', model1), ('gb', model2), ('svc', model3)],
    voting='soft',  # Average probabilities
    weights=[2, 1, 1]  # Model weights (optional)
)

voting_soft.fit(X_train, y_train)
predictions = voting_soft.predict(X_test)
probabilities = voting_soft.predict_proba(X_test)
```

**Custom Weighted Voting:**
```python
class WeightedVoting:
    """Weighted voting based on validation performance"""

    def __init__(self, models):
        self.models = models
        self.weights = None

    def fit(self, X_train, y_train, X_val, y_val):
        # Train all models
        self.trained_models = []
        accuracies = []

        for model in self.models:
            model.fit(X_train, y_train)
            self.trained_models.append(model)

            # Validation accuracy
            acc = model.score(X_val, y_val)
            accuracies.append(acc)

        # Compute weights based on validation accuracy
        accuracies = np.array(accuracies)
        self.weights = accuracies / accuracies.sum()

        print(f"Model weights: {self.weights}")

    def predict_proba(self, X):
        # Weighted average of probabilities
        probas = np.array([model.predict_proba(X) for model in self.trained_models])
        weighted_probas = np.average(probas, axis=0, weights=self.weights)
        return weighted_probas

    def predict(self, X):
        probas = self.predict_proba(X)
        return probas.argmax(axis=1)
```

---

## Neural Network Ensembles

### Snapshot Ensembles

**Strategy:** Save model checkpoints during cyclic learning rate schedule.

```python
import torch
import torch.nn as nn
import torch.optim as optim
import math

class SnapshotEnsemble:
    """Snapshot ensemble with cyclic learning rate"""

    def __init__(self, model_fn, n_snapshots=5, n_epochs=50):
        self.model_fn = model_fn
        self.n_snapshots = n_snapshots
        self.n_epochs = n_epochs
        self.snapshots = []

    def cosine_annealing(self, epoch, n_epochs, lr_min=1e-5, lr_max=1e-2):
        """Cosine annealing learning rate"""
        return lr_min + 0.5 * (lr_max - lr_min) * (
            1 + math.cos(math.pi * epoch / n_epochs)
        )

    def train(self, train_loader, val_loader):
        model = self.model_fn()
        optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
        criterion = nn.CrossEntropyLoss()

        epochs_per_cycle = self.n_epochs // self.n_snapshots

        for epoch in range(self.n_epochs):
            # Update learning rate
            lr = self.cosine_annealing(
                epoch % epochs_per_cycle,
                epochs_per_cycle
            )
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

            # Train epoch
            model.train()
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()

            # Save snapshot at end of cycle (high LR → low LR)
            if (epoch + 1) % epochs_per_cycle == 0:
                snapshot = self.model_fn()
                snapshot.load_state_dict(model.state_dict())
                self.snapshots.append(snapshot)
                print(f"Saved snapshot {len(self.snapshots)} at epoch {epoch+1}")

    def predict(self, X):
        """Ensemble prediction"""
        predictions = []

        for model in self.snapshots:
            model.eval()
            with torch.no_grad():
                pred = model(X)
                predictions.append(pred)

        # Average predictions
        ensemble_pred = torch.stack(predictions).mean(dim=0)
        return ensemble_pred.argmax(dim=1)

# Usage
def create_model():
    return nn.Sequential(
        nn.Linear(784, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, 10)
    )

ensemble = SnapshotEnsemble(create_model, n_snapshots=5, n_epochs=50)
ensemble.train(train_loader, val_loader)
predictions = ensemble.predict(X_test)
```

---

### Deep Ensembles

**Strategy:** Train multiple neural networks with different initializations.

```python
class DeepEnsemble:
    """Ensemble of independently trained neural networks"""

    def __init__(self, model_fn, n_models=5):
        self.model_fn = model_fn
        self.n_models = n_models
        self.models = []

    def train(self, train_loader, val_loader, epochs=50):
        for i in range(self.n_models):
            print(f"\nTraining model {i+1}/{self.n_models}")

            # Create model with random initialization
            model = self.model_fn()
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            criterion = nn.CrossEntropyLoss()

            best_val_acc = 0

            for epoch in range(epochs):
                # Train
                model.train()
                for X_batch, y_batch in train_loader:
                    optimizer.zero_grad()
                    outputs = model(X_batch)
                    loss = criterion(outputs, y_batch)
                    loss.backward()
                    optimizer.step()

                # Validate
                model.eval()
                val_acc = self.evaluate(model, val_loader)

                if val_acc > best_val_acc:
                    best_val_acc = val_acc

            self.models.append(model)
            print(f"Model {i+1} best val acc: {best_val_acc:.4f}")

    def evaluate(self, model, loader):
        correct = 0
        total = 0
        with torch.no_grad():
            for X_batch, y_batch in loader:
                outputs = model(X_batch)
                _, predicted = outputs.max(1)
                total += y_batch.size(0)
                correct += predicted.eq(y_batch).sum().item()
        return correct / total

    def predict(self, X):
        """Ensemble prediction with uncertainty"""
        predictions = []

        for model in self.models:
            model.eval()
            with torch.no_grad():
                pred = model(X)
                predictions.append(torch.softmax(pred, dim=1))

        # Stack predictions
        predictions = torch.stack(predictions)

        # Mean prediction
        mean_pred = predictions.mean(dim=0)

        # Uncertainty (variance)
        uncertainty = predictions.var(dim=0)

        return mean_pred.argmax(dim=1), uncertainty

# Usage
ensemble = DeepEnsemble(create_model, n_models=5)
ensemble.train(train_loader, val_loader, epochs=50)
predictions, uncertainty = ensemble.predict(X_test)
```

---

### Dropout as Ensemble (MC Dropout)

**Strategy:** Use dropout at inference time for ensemble effect.

```python
class MCDropoutEnsemble:
    """Monte Carlo Dropout for uncertainty estimation"""

    def __init__(self, model, n_samples=50):
        self.model = model
        self.n_samples = n_samples

    def enable_dropout(self):
        """Enable dropout during inference"""
        for module in self.model.modules():
            if isinstance(module, nn.Dropout):
                module.train()

    def predict(self, X):
        """Ensemble prediction using dropout"""
        self.enable_dropout()

        predictions = []
        for _ in range(self.n_samples):
            with torch.no_grad():
                pred = self.model(X)
                predictions.append(torch.softmax(pred, dim=1))

        # Stack predictions
        predictions = torch.stack(predictions)

        # Mean and variance
        mean_pred = predictions.mean(dim=0)
        uncertainty = predictions.var(dim=0)

        return mean_pred.argmax(dim=1), uncertainty

# Usage
model = create_model_with_dropout()
mc_ensemble = MCDropoutEnsemble(model, n_samples=50)
predictions, uncertainty = mc_ensemble.predict(X_test)

# High uncertainty samples
high_uncertainty_idx = uncertainty.max(dim=1)[0] > 0.1
print(f"Samples with high uncertainty: {high_uncertainty_idx.sum().item()}")
```

---

## Advanced Techniques

### Feature-Weighted Linear Stacking (FWLS)

```python
class FeatureWeightedStacking:
    """Stacking with feature-level weights"""

    def __init__(self, base_models):
        self.base_models = base_models
        self.feature_weights = None

    def fit(self, X, y):
        # Train base models
        self.trained_models = []
        all_predictions = []

        for model in self.base_models:
            model.fit(X, y)
            self.trained_models.append(model)

            # Get feature importances
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
            else:
                importances = np.ones(X.shape[1])

            all_predictions.append(model.predict(X))

        # Calculate feature weights based on agreement
        predictions_array = np.array(all_predictions)
        agreement = np.std(predictions_array, axis=0)

        # Inverse agreement = weight (more agreement = more weight)
        self.weights = 1 / (agreement + 1e-10)
        self.weights /= self.weights.sum()

    def predict(self, X):
        predictions = np.array([model.predict(X) for model in self.trained_models])

        # Weighted combination
        weighted_preds = predictions * self.weights[:, np.newaxis]
        return weighted_preds.sum(axis=0).astype(int)
```

---

### Dynamic Ensemble Selection

```python
class DynamicEnsembleSelection:
    """Select best models for each test instance"""

    def __init__(self, base_models, k=5):
        self.base_models = base_models
        self.k = k  # Number of models to select

    def fit(self, X_train, y_train, X_val, y_val):
        # Train models
        self.trained_models = []
        self.val_predictions = []

        for model in self.base_models:
            model.fit(X_train, y_train)
            self.trained_models.append(model)

            # Validation predictions
            val_pred = model.predict(X_val)
            self.val_predictions.append(val_pred)

        self.X_val = X_val
        self.y_val = y_val
        self.val_predictions = np.array(self.val_predictions)

    def predict(self, X):
        predictions = []

        for x in X:
            # Find k nearest neighbors in validation set
            distances = np.linalg.norm(self.X_val - x, axis=1)
            nearest_idx = np.argsort(distances)[:self.k]

            # Evaluate model performance on nearest neighbors
            model_scores = []
            for i, model in enumerate(self.trained_models):
                accuracy = (self.val_predictions[i][nearest_idx] ==
                           self.y_val[nearest_idx]).mean()
                model_scores.append(accuracy)

            # Select best model
            best_model_idx = np.argmax(model_scores)
            pred = self.trained_models[best_model_idx].predict([x])[0]
            predictions.append(pred)

        return np.array(predictions)
```

---

## Best Practices

### 1. Ensure Model Diversity

```python
def calculate_diversity(models, X, y):
    """Calculate pairwise diversity between models"""
    n_models = len(models)
    predictions = np.array([model.predict(X) for model in models])

    # Q-statistic for diversity
    q_matrix = np.zeros((n_models, n_models))

    for i in range(n_models):
        for j in range(i+1, n_models):
            # Agreement and disagreement
            both_correct = np.sum((predictions[i] == y) & (predictions[j] == y))
            both_wrong = np.sum((predictions[i] != y) & (predictions[j] != y))
            one_correct = np.sum((predictions[i] == y) & (predictions[j] != y))
            other_correct = np.sum((predictions[i] != y) & (predictions[j] == y))

            # Q-statistic
            q = (both_correct * both_wrong - one_correct * other_correct) / \
                (both_correct * both_wrong + one_correct * other_correct + 1e-10)

            q_matrix[i, j] = q
            q_matrix[j, i] = q

    avg_diversity = np.mean(q_matrix[np.triu_indices_from(q_matrix, k=1)])
    print(f"Average pairwise diversity (Q-statistic): {avg_diversity:.4f}")
    # Lower Q = more diverse
    return q_matrix
```

### 2. Optimal Ensemble Size

```python
def find_optimal_ensemble_size(models, X_val, y_val):
    """Find optimal number of models to include"""
    n_models = len(models)
    accuracies = []

    for n in range(1, n_models + 1):
        # Ensemble of top n models
        ensemble_preds = []
        for model in models[:n]:
            ensemble_preds.append(model.predict(X_val))

        # Majority vote
        ensemble_preds = np.array(ensemble_preds)
        final_preds = np.apply_along_axis(
            lambda x: np.bincount(x).argmax(),
            axis=0,
            arr=ensemble_preds
        )

        acc = (final_preds == y_val).mean()
        accuracies.append(acc)

    optimal_n = np.argmax(accuracies) + 1
    print(f"Optimal ensemble size: {optimal_n}")
    print(f"Best accuracy: {accuracies[optimal_n-1]:.4f}")

    return optimal_n
```

### 3. Performance Comparison

```python
def compare_ensemble_methods(X_train, y_train, X_test, y_test):
    """Compare different ensemble methods"""

    results = {}

    # Single model baseline
    rf_single = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_single.fit(X_train, y_train)
    results['Single RF'] = rf_single.score(X_test, y_test)

    # Bagging
    bagging = BaggingClassifier(
        DecisionTreeClassifier(max_depth=10),
        n_estimators=50,
        random_state=42
    )
    bagging.fit(X_train, y_train)
    results['Bagging'] = bagging.score(X_test, y_test)

    # Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    results['Random Forest'] = rf.score(X_test, y_test)

    # Gradient Boosting
    gb = GradientBoostingClassifier(n_estimators=100, random_state=42)
    gb.fit(X_train, y_train)
    results['Gradient Boosting'] = gb.score(X_test, y_test)

    # XGBoost
    xgb_model = xgb.XGBClassifier(n_estimators=100, random_state=42)
    xgb_model.fit(X_train, y_train)
    results['XGBoost'] = xgb_model.score(X_test, y_test)

    # Voting
    voting = VotingClassifier(
        estimators=[('rf', rf), ('gb', gb), ('xgb', xgb_model)],
        voting='soft'
    )
    voting.fit(X_train, y_train)
    results['Voting'] = voting.score(X_test, y_test)

    # Stacking
    stacking = StackingClassifier(
        estimators=[('rf', rf), ('gb', gb), ('xgb', xgb_model)],
        final_estimator=LogisticRegression(),
        cv=5
    )
    stacking.fit(X_train, y_train)
    results['Stacking'] = stacking.score(X_test, y_test)

    # Print results
    print("\nEnsemble Method Comparison:")
    print("-" * 40)
    for method, acc in sorted(results.items(), key=lambda x: x[1], reverse=True):
        print(f"{method:20s}: {acc:.4f}")

    return results
```

---

## Summary: Choosing the Right Ensemble

| Method | Best For | Pros | Cons |
|--------|----------|------|------|
| **Bagging** | Reducing variance, parallel training | Fast, reduces overfitting | Limited improvement |
| **Random Forest** | Tabular data, feature importance | Robust, interpretable | Can overfit with deep trees |
| **AdaBoost** | Binary classification, weak learners | Simple, effective | Sensitive to noise |
| **Gradient Boosting** | Competitions, best accuracy | High accuracy | Slow, prone to overfitting |
| **XGBoost** | Tabular data, production | Fast, regularized, scalable | Memory intensive |
| **LightGBM** | Large datasets, speed | Very fast, efficient | Requires tuning |
| **CatBoost** | Categorical features | Handles cats well, less overfitting | Slower than LightGBM |
| **Stacking** | Squeezing last % accuracy | Highest accuracy | Complex, slow |
| **Voting** | Quick ensemble, diversity | Simple, effective | Limited flexibility |

---

## Key Takeaways

1. **Diversity is critical** - models should make different errors
2. **More models ≠ better** - find optimal ensemble size
3. **Boosting for bias**, **bagging for variance**
4. **XGBoost/LightGBM** dominate tabular data competitions
5. **Stacking** for maximum accuracy, **voting** for simplicity
6. **Neural network ensembles** provide uncertainty estimates
7. **Always validate** ensemble improvements on held-out data

**Next Steps:**
- Implement ensemble for your specific problem
- Experiment with different base models for diversity
- Try stacking with neural network meta-learner
- Measure uncertainty in predictions

---

## 📚 References

**Key Papers:**

1. **Breiman, L.** (1996). "Bagging predictors." *Machine Learning*, 24(2), 123-140.
   - Original bagging algorithm

2. **Breiman, L.** (2001). "Random forests." *Machine Learning*, 45(1), 5-32.
   - Random Forest algorithm

3. **Freund, Y., & Schapire, R. E.** (1997). "A decision-theoretic generalization of on-line learning and an application to boosting." *Journal of Computer and System Sciences*, 55(1), 119-139.
   - AdaBoost algorithm

4. **Friedman, J. H.** (2001). "Greedy function approximation: A gradient boosting machine." *Annals of Statistics*, 29(5), 1189-1232.
   - Gradient boosting framework

5. **Chen, T., & Guestrin, C.** (2016). "XGBoost: A scalable tree boosting system." *Proceedings of the 22nd ACM SIGKDD*, 785-794.
   - XGBoost algorithm

6. **Ke, G., Meng, Q., Finley, T., et al.** (2017). "LightGBM: A highly efficient gradient boosting decision tree." *NIPS 2017*, 3146-3154.
   - LightGBM algorithm

7. **Prokhorenkova, L., Gusev, G., Vorobev, A., et al.** (2018). "CatBoost: Unbiased boosting with categorical features." *NeurIPS 2018*, 6638-6648.
   - CatBoost algorithm

8. **Wolpert, D. H.** (1992). "Stacked generalization." *Neural Networks*, 5(2), 241-259.
   - Stacking ensemble method

9. **Dietterich, T. G.** (2000). "Ensemble methods in machine learning." *Multiple Classifier Systems*, 1-15.
   - Comprehensive ensemble methods overview

10. **Kuncheva, L. I., & Whitaker, C. J.** (2003). "Measures of diversity in classifier ensembles and their relationship with the ensemble accuracy." *Machine Learning*, 51(2), 181-207.
    - Diversity measures in ensembles

**Books:**

1. **Zhou, Z. H.** (2012). *Ensemble Methods: Foundations and Algorithms*. CRC Press.
   - Comprehensive textbook on ensemble learning

2. **Hastie, T., Tibshirani, R., & Friedman, J.** (2009). *The Elements of Statistical Learning* (2nd ed.). Springer.
   - Chapters 8, 10, 15, 16 on ensemble methods

3. **Géron, A.** (2019). *Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow* (2nd ed.). O'Reilly.
   - Practical ensemble implementations

**Online Resources:**
- XGBoost documentation: https://xgboost.readthedocs.io/
- LightGBM documentation: https://lightgbm.readthedocs.io/
- CatBoost documentation: https://catboost.ai/docs/
- scikit-learn ensemble module: https://scikit-learn.org/stable/modules/ensemble.html
